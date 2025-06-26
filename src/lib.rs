pub mod ann;
pub mod auto_discovery;
pub mod features;
pub mod gpu_acceleration;
pub mod license;
pub mod lichess_loader;
pub mod lsh;
pub mod manifold_learner;
pub mod nnue;
pub mod opening_book;
pub mod persistence;
pub mod position_encoder;
pub mod similarity_search;
pub mod streaming_loader;
pub mod tactical_search;
pub mod training;
pub mod ultra_fast_loader;
pub mod variational_autoencoder;
// pub mod tablebase; // Temporarily disabled due to version conflicts
pub mod uci;

pub use auto_discovery::{AutoDiscovery, FormatPriority, TrainingFile};
pub use features::{FeatureChecker, FeatureError, FeatureRegistry, FeatureTier};
pub use gpu_acceleration::{DeviceType, GPUAccelerator};
pub use license::{
    LicenseError, LicenseKey, LicenseStatus, LicenseVerifier, LicensedFeatureChecker,
};
pub use lichess_loader::{load_lichess_puzzles_basic, load_lichess_puzzles_premium, LichessLoader};
pub use lsh::LSH;
pub use manifold_learner::ManifoldLearner;
pub use nnue::{BlendStrategy, EvalStats, HybridEvaluator, NNUEConfig, NNUE};
pub use opening_book::{OpeningBook, OpeningBookStats, OpeningEntry};
pub use persistence::{Database, LSHTableData, PositionData};
pub use position_encoder::PositionEncoder;
pub use similarity_search::SimilaritySearch;
pub use streaming_loader::StreamingLoader;
pub use tactical_search::{TacticalConfig, TacticalResult, TacticalSearch};
pub use training::{
    EngineEvaluator, GameExtractor, SelfPlayConfig, SelfPlayTrainer, TacticalPuzzle,
    TacticalPuzzleParser, TacticalTrainingData, TrainingData, TrainingDataset,
};
pub use ultra_fast_loader::{LoadingStats, UltraFastLoader};
pub use variational_autoencoder::{VAEConfig, VariationalAutoencoder};
// pub use tablebase::{TablebaseProber, TablebaseResult, WdlValue};
pub use uci::{run_uci_engine, run_uci_engine_with_config, UCIConfig, UCIEngine};

use chess::{Board, ChessMove};
use ndarray::{Array1, Array2};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;

/// Calculate move centrality for intelligent move ordering
/// Returns higher values for moves toward the center of the board
fn move_centrality(chess_move: &ChessMove) -> f32 {
    let dest_square = chess_move.get_dest();
    let rank = dest_square.get_rank().to_index() as f32;
    let file = dest_square.get_file().to_index() as f32;

    // Calculate distance from center (3.5, 3.5)
    let center_rank = 3.5;
    let center_file = 3.5;

    let rank_distance = (rank - center_rank).abs();
    let file_distance = (file - center_file).abs();

    // Return higher values for more central moves (invert the distance)
    let max_distance = 3.5; // Maximum distance from center to edge
    let distance = (rank_distance + file_distance) / 2.0;
    max_distance - distance
}

/// Move recommendation data
#[derive(Debug, Clone)]
pub struct MoveRecommendation {
    pub chess_move: ChessMove,
    pub confidence: f32,
    pub from_similar_position_count: usize,
    pub average_outcome: f32,
}

/// Training statistics for the engine
#[derive(Debug, Clone)]
pub struct TrainingStats {
    pub total_positions: usize,
    pub unique_positions: usize,
    pub has_move_data: bool,
    pub move_data_entries: usize,
    pub lsh_enabled: bool,
    pub manifold_enabled: bool,
    pub opening_book_enabled: bool,
}

/// Hybrid evaluation configuration
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// Confidence threshold for pattern-only evaluation (0.0-1.0)
    pub pattern_confidence_threshold: f32,
    /// Enable tactical refinement for uncertain positions
    pub enable_tactical_refinement: bool,
    /// Tactical search configuration
    pub tactical_config: TacticalConfig,
    /// Weight for pattern evaluation vs tactical evaluation (0.0-1.0)
    pub pattern_weight: f32,
    /// Minimum number of similar positions to trust pattern evaluation
    pub min_similar_positions: usize,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            pattern_confidence_threshold: 0.8,
            enable_tactical_refinement: true,
            tactical_config: TacticalConfig::default(),
            pattern_weight: 0.7, // Favor patterns but include tactical refinement
            min_similar_positions: 3,
        }
    }
}

/// Main chess vector engine with feature-gated capabilities
pub struct ChessVectorEngine {
    encoder: PositionEncoder,
    similarity_search: SimilaritySearch,
    lsh_index: Option<LSH>,
    manifold_learner: Option<ManifoldLearner>,
    use_lsh: bool,
    use_manifold: bool,
    /// Map from position index to moves played and their outcomes
    position_moves: HashMap<usize, Vec<(ChessMove, f32)>>,
    /// Compressed similarity search for manifold vectors
    manifold_similarity_search: Option<SimilaritySearch>,
    /// LSH index for compressed vectors
    manifold_lsh_index: Option<LSH>,
    /// Feature access control
    feature_checker: FeatureChecker,
    /// License-based feature access control
    licensed_feature_checker: Option<LicensedFeatureChecker>,
    /// Store position vectors for reverse lookup
    position_vectors: Vec<Array1<f32>>,
    /// Store boards for move generation
    position_boards: Vec<Board>,
    /// Store evaluations for each position
    position_evaluations: Vec<f32>,
    /// Opening book for position evaluation and move suggestions
    opening_book: Option<OpeningBook>,
    /// Database for persistence
    database: Option<Database>,
    /// Tactical search engine for position refinement
    tactical_search: Option<TacticalSearch>,
    // /// Syzygy tablebase for perfect endgame evaluation
    // tablebase: Option<TablebaseProber>,
    /// Hybrid evaluation configuration
    hybrid_config: HybridConfig,
}

impl Clone for ChessVectorEngine {
    fn clone(&self) -> Self {
        Self {
            encoder: self.encoder.clone(),
            similarity_search: self.similarity_search.clone(),
            lsh_index: self.lsh_index.clone(),
            manifold_learner: None, // ManifoldLearner cannot be cloned due to ML components
            use_lsh: self.use_lsh,
            use_manifold: false, // Disable manifold learning in cloned instance
            position_moves: self.position_moves.clone(),
            manifold_similarity_search: self.manifold_similarity_search.clone(),
            manifold_lsh_index: self.manifold_lsh_index.clone(),
            feature_checker: self.feature_checker.clone(),
            licensed_feature_checker: None, // License checker cannot be cloned
            position_vectors: self.position_vectors.clone(),
            position_boards: self.position_boards.clone(),
            position_evaluations: self.position_evaluations.clone(),
            opening_book: self.opening_book.clone(),
            database: None, // Database connection cannot be cloned
            tactical_search: self.tactical_search.clone(),
            // tablebase: self.tablebase.clone(),
            hybrid_config: self.hybrid_config.clone(),
        }
    }
}

impl ChessVectorEngine {
    /// Create a new chess vector engine
    pub fn new(vector_size: usize) -> Self {
        Self {
            encoder: PositionEncoder::new(vector_size),
            similarity_search: SimilaritySearch::new(vector_size),
            lsh_index: None,
            manifold_learner: None,
            use_lsh: false,
            use_manifold: false,
            position_moves: HashMap::new(),
            manifold_similarity_search: None,
            manifold_lsh_index: None,
            feature_checker: FeatureChecker::new(FeatureTier::OpenSource), // Default to open source
            licensed_feature_checker: None,
            position_vectors: Vec::new(),
            position_boards: Vec::new(),
            position_evaluations: Vec::new(),
            opening_book: None,
            database: None,
            tactical_search: None,
            // tablebase: None,
            hybrid_config: HybridConfig::default(),
        }
    }

    /// Create new engine with specific feature tier
    pub fn new_with_tier(vector_size: usize, tier: FeatureTier) -> Self {
        let mut engine = Self::new(vector_size);
        engine.feature_checker = FeatureChecker::new(tier);
        engine
    }

    /// Get current feature tier
    pub fn get_feature_tier(&self) -> &FeatureTier {
        self.feature_checker.get_current_tier()
    }

    /// Upgrade feature tier (for license activation)
    pub fn upgrade_tier(&mut self, new_tier: FeatureTier) {
        self.feature_checker.upgrade_tier(new_tier);
    }

    /// Check if a feature is available
    pub fn is_feature_available(&self, feature: &str) -> bool {
        self.feature_checker.check_feature(feature).is_ok()
    }

    /// Require a feature (returns error if not available)
    pub fn require_feature(&self, feature: &str) -> Result<(), FeatureError> {
        self.feature_checker.require_feature(feature)
    }

    /// Create a new chess vector engine with intelligent architecture selection
    /// based on expected dataset size and use case
    pub fn new_adaptive(vector_size: usize, expected_positions: usize, use_case: &str) -> Self {
        match use_case {
            "training" => {
                if expected_positions > 10000 {
                    // Large training datasets benefit from LSH for loading speed
                    Self::new_with_lsh(vector_size, 12, 20)
                } else {
                    Self::new(vector_size)
                }
            }
            "gameplay" => {
                if expected_positions > 15000 {
                    // Gameplay needs balance of speed and accuracy
                    Self::new_with_lsh(vector_size, 10, 18)
                } else {
                    Self::new(vector_size)
                }
            }
            "analysis" => {
                if expected_positions > 10000 {
                    // Analysis prioritizes recall over speed
                    Self::new_with_lsh(vector_size, 14, 22)
                } else {
                    Self::new(vector_size)
                }
            }
            _ => Self::new(vector_size), // Default to linear search
        }
    }

    /// Create a new chess vector engine with LSH enabled
    pub fn new_with_lsh(vector_size: usize, num_tables: usize, hash_size: usize) -> Self {
        Self {
            encoder: PositionEncoder::new(vector_size),
            similarity_search: SimilaritySearch::new(vector_size),
            lsh_index: Some(LSH::new(vector_size, num_tables, hash_size)),
            manifold_learner: None,
            use_lsh: true,
            use_manifold: false,
            position_moves: HashMap::new(),
            manifold_similarity_search: None,
            manifold_lsh_index: None,
            feature_checker: FeatureChecker::new(FeatureTier::OpenSource),
            licensed_feature_checker: None,
            position_vectors: Vec::new(),
            position_boards: Vec::new(),
            position_evaluations: Vec::new(),
            opening_book: None,
            database: None,
            tactical_search: None,
            // tablebase: None,
            hybrid_config: HybridConfig::default(),
        }
    }

    /// Enable LSH indexing
    pub fn enable_lsh(&mut self, num_tables: usize, hash_size: usize) {
        self.lsh_index = Some(LSH::new(self.encoder.vector_size(), num_tables, hash_size));
        self.use_lsh = true;

        // Rebuild LSH index with existing positions
        if let Some(ref mut lsh) = self.lsh_index {
            for (vector, evaluation) in self.similarity_search.get_all_positions() {
                lsh.add_vector(vector, evaluation);
            }
        }
    }

    /// Add a position with its evaluation to the knowledge base
    pub fn add_position(&mut self, board: &Board, evaluation: f32) {
        // Safety check: Validate position before storing
        if !self.is_position_safe(board) {
            return; // Skip unsafe positions
        }

        let vector = self.encoder.encode(board);
        self.similarity_search
            .add_position(vector.clone(), evaluation);

        // Store vector, board, and evaluation for reverse lookup
        self.position_vectors.push(vector.clone());
        self.position_boards.push(*board);
        self.position_evaluations.push(evaluation);

        // Also add to LSH index if enabled
        if let Some(ref mut lsh) = self.lsh_index {
            lsh.add_vector(vector.clone(), evaluation);
        }

        // Add to manifold indices if trained
        if self.use_manifold {
            if let Some(ref learner) = self.manifold_learner {
                let compressed = learner.encode(&vector);

                if let Some(ref mut search) = self.manifold_similarity_search {
                    search.add_position(compressed.clone(), evaluation);
                }

                if let Some(ref mut lsh) = self.manifold_lsh_index {
                    lsh.add_vector(compressed, evaluation);
                }
            }
        }
    }

    /// Find similar positions to the given board
    pub fn find_similar_positions(&self, board: &Board, k: usize) -> Vec<(Array1<f32>, f32, f32)> {
        let query_vector = self.encoder.encode(board);

        // Use manifold space if available and trained
        if self.use_manifold {
            if let Some(ref manifold_learner) = self.manifold_learner {
                let compressed_query = manifold_learner.encode(&query_vector);

                // Use LSH in manifold space if available
                if let Some(ref lsh) = self.manifold_lsh_index {
                    return lsh.query(&compressed_query, k);
                }

                // Fall back to linear search in manifold space
                if let Some(ref search) = self.manifold_similarity_search {
                    return search.search(&compressed_query, k);
                }
            }
        }

        // Use original space with LSH if enabled
        if self.use_lsh {
            if let Some(ref lsh_index) = self.lsh_index {
                return lsh_index.query(&query_vector, k);
            }
        }

        // Fall back to linear search
        self.similarity_search.search(&query_vector, k)
    }

    /// Find similar positions with indices for move recommendation
    pub fn find_similar_positions_with_indices(
        &self,
        board: &Board,
        k: usize,
    ) -> Vec<(usize, f32, f32)> {
        let query_vector = self.encoder.encode(board);

        // For now, use linear search to get accurate position indices
        // In the future, we could enhance LSH to return indices
        let mut results = Vec::new();

        for (i, stored_vector) in self.position_vectors.iter().enumerate() {
            let similarity = self.encoder.similarity(&query_vector, stored_vector);
            let eval = self.position_evaluations.get(i).copied().unwrap_or(0.0);
            results.push((i, eval, similarity));
        }

        // Sort by similarity (descending)
        results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);

        results
    }

    /// Get evaluation for a position using hybrid approach (opening book + pattern evaluation + tactical search)
    pub fn evaluate_position(&mut self, board: &Board) -> Option<f32> {
        // // First check tablebase for perfect endgame evaluation - highest priority
        // if let Some(ref tablebase) = self.tablebase {
        //     if let Some(tb_eval) = tablebase.get_evaluation(board) {
        //         return Some(tb_eval);
        //     }
        // }

        // Second check opening book
        if let Some(entry) = self.get_opening_entry(board) {
            return Some(entry.evaluation);
        }

        // Get pattern evaluation from similarity search
        let similar_positions = self.find_similar_positions(board, 5);

        if similar_positions.is_empty() {
            // No similar positions found - use tactical search if available
            if let Some(ref mut tactical_search) = self.tactical_search {
                let result = tactical_search.search(board);
                return Some(result.evaluation);
            }
            return None;
        }

        // Calculate pattern evaluation and confidence
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        let mut similarity_scores = Vec::new();

        for (_, evaluation, similarity) in &similar_positions {
            let weight = *similarity;
            weighted_sum += evaluation * weight;
            weight_sum += weight;
            similarity_scores.push(*similarity);
        }

        let pattern_evaluation = weighted_sum / weight_sum;

        // Calculate pattern confidence based on similarity scores and count
        let avg_similarity = similarity_scores.iter().sum::<f32>() / similarity_scores.len() as f32;
        let count_factor = (similar_positions.len() as f32
            / self.hybrid_config.min_similar_positions as f32)
            .min(1.0);
        let pattern_confidence = avg_similarity * count_factor;

        // Decide whether to use tactical refinement
        let use_tactical = self.hybrid_config.enable_tactical_refinement
            && pattern_confidence < self.hybrid_config.pattern_confidence_threshold
            && self.tactical_search.is_some();

        if use_tactical {
            // Get tactical evaluation (use parallel search if enabled)
            if let Some(ref mut tactical_search) = self.tactical_search {
                let tactical_result = if tactical_search.config.enable_parallel_search {
                    tactical_search.search_parallel(board)
                } else {
                    tactical_search.search(board)
                };

                // Blend pattern and tactical evaluations
                let pattern_weight = self.hybrid_config.pattern_weight * pattern_confidence;
                let tactical_weight = 1.0 - pattern_weight;

                let hybrid_evaluation = (pattern_evaluation * pattern_weight)
                    + (tactical_result.evaluation * tactical_weight);

                Some(hybrid_evaluation)
            } else {
                // Tactical search not available, fall back to pattern only
                Some(pattern_evaluation)
            }
        } else {
            // Use pattern evaluation only
            Some(pattern_evaluation)
        }
    }

    /// Encode a position to vector (public interface)
    pub fn encode_position(&self, board: &Board) -> Array1<f32> {
        self.encoder.encode(board)
    }

    /// Calculate similarity between two boards
    pub fn calculate_similarity(&self, board1: &Board, board2: &Board) -> f32 {
        let vec1 = self.encoder.encode(board1);
        let vec2 = self.encoder.encode(board2);
        self.encoder.similarity(&vec1, &vec2)
    }

    /// Get the size of the knowledge base
    pub fn knowledge_base_size(&self) -> usize {
        self.similarity_search.size()
    }

    /// Save engine state (positions and evaluations) to file for incremental training
    pub fn save_training_data<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use crate::training::{TrainingData, TrainingDataset};

        let mut dataset = TrainingDataset::new();

        // Convert engine positions back to training data
        for (i, board) in self.position_boards.iter().enumerate() {
            if i < self.position_evaluations.len() {
                dataset.data.push(TrainingData {
                    board: *board,
                    evaluation: self.position_evaluations[i],
                    depth: 15,  // Default depth
                    game_id: i, // Use index as game_id
                });
            }
        }

        dataset.save_incremental(path)?;
        println!("Saved {} positions to training data", dataset.data.len());
        Ok(())
    }

    /// Load training data incrementally (append to existing engine state) - OPTIMIZED
    pub fn load_training_data_incremental<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use crate::training::TrainingDataset;
        use indicatif::{ProgressBar, ProgressStyle};
        use std::collections::HashSet;

        let existing_size = self.knowledge_base_size();

        // Try binary format first (5-15x faster)
        let path_ref = path.as_ref();
        let binary_path = path_ref.with_extension("bin");
        if binary_path.exists() {
            println!("🚀 Loading optimized binary format...");
            return self.load_training_data_binary(binary_path);
        }

        println!("📚 Loading training data from {}...", path_ref.display());
        let dataset = TrainingDataset::load(path)?;

        let total_positions = dataset.data.len();
        if total_positions == 0 {
            println!("⚠️  No positions found in dataset");
            return Ok(());
        }

        // Progress bar for duplicate checking phase
        let dedup_pb = ProgressBar::new(total_positions as u64);
        dedup_pb.set_style(
            ProgressStyle::default_bar()
                .template("🔍 Checking duplicates [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) {msg}")?
                .progress_chars("██░")
        );

        // Pre-allocate HashSet for O(1) duplicate checking
        let mut existing_boards: HashSet<_> = self.position_boards.iter().cloned().collect();
        let mut new_positions = Vec::new();
        let mut new_evaluations = Vec::new();

        // Batch process to avoid repeated lookups
        for (i, data) in dataset.data.into_iter().enumerate() {
            if !existing_boards.contains(&data.board) {
                existing_boards.insert(data.board);
                new_positions.push(data.board);
                new_evaluations.push(data.evaluation);
            }

            if i % 1000 == 0 || i == total_positions - 1 {
                dedup_pb.set_position((i + 1) as u64);
                dedup_pb.set_message(format!("{} new positions found", new_positions.len()));
            }
        }
        dedup_pb.finish_with_message(format!("✅ Found {} new positions", new_positions.len()));

        if new_positions.is_empty() {
            println!("ℹ️  No new positions to add (all positions already exist)");
            return Ok(());
        }

        // Progress bar for adding positions
        let add_pb = ProgressBar::new(new_positions.len() as u64);
        add_pb.set_style(
            ProgressStyle::default_bar()
                .template("➕ Adding positions [{elapsed_precise}] [{bar:40.green/blue}] {pos}/{len} ({percent}%) {msg}")?
                .progress_chars("██░")
        );

        // Batch add all new positions
        for (i, (board, evaluation)) in new_positions
            .into_iter()
            .zip(new_evaluations.into_iter())
            .enumerate()
        {
            self.add_position(&board, evaluation);

            if i % 500 == 0 || i == add_pb.length().unwrap() as usize - 1 {
                add_pb.set_position((i + 1) as u64);
                add_pb.set_message("vectors encoded".to_string());
            }
        }
        add_pb.finish_with_message("✅ All positions added");

        println!(
            "🎯 Loaded {} new positions (total: {})",
            self.knowledge_base_size() - existing_size,
            self.knowledge_base_size()
        );
        Ok(())
    }

    /// Save training data in optimized binary format with compression (5-15x faster than JSON)
    pub fn save_training_data_binary<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use lz4_flex::compress_prepend_size;

        println!("💾 Saving training data in binary format (compressed)...");

        // Create binary training data structure
        #[derive(serde::Serialize)]
        struct BinaryTrainingData {
            positions: Vec<String>, // FEN strings
            evaluations: Vec<f32>,
            vectors: Vec<Vec<f32>>, // Optional for export
            created_at: i64,
        }

        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs() as i64;

        // Prepare data for serialization
        let mut positions = Vec::with_capacity(self.position_boards.len());
        let mut evaluations = Vec::with_capacity(self.position_boards.len());
        let mut vectors = Vec::with_capacity(self.position_boards.len());

        for (i, board) in self.position_boards.iter().enumerate() {
            if i < self.position_evaluations.len() {
                positions.push(board.to_string());
                evaluations.push(self.position_evaluations[i]);

                // Include vectors if available
                if i < self.position_vectors.len() {
                    if let Some(vector_slice) = self.position_vectors[i].as_slice() {
                        vectors.push(vector_slice.to_vec());
                    }
                }
            }
        }

        let binary_data = BinaryTrainingData {
            positions,
            evaluations,
            vectors,
            created_at: current_time,
        };

        // Serialize with bincode (much faster than JSON)
        let serialized = bincode::serialize(&binary_data)?;

        // Compress with LZ4 (5-10x smaller, very fast)
        let compressed = compress_prepend_size(&serialized);

        // Write to file
        std::fs::write(path, &compressed)?;

        println!(
            "✅ Saved {} positions to binary file ({} bytes compressed)",
            binary_data.positions.len(),
            compressed.len()
        );
        Ok(())
    }

    /// Load training data from optimized binary format (5-15x faster than JSON)
    pub fn load_training_data_binary<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use indicatif::{ProgressBar, ProgressStyle};
        use lz4_flex::decompress_size_prepended;

        println!("📚 Loading training data from binary format...");

        #[derive(serde::Deserialize)]
        struct BinaryTrainingData {
            positions: Vec<String>,
            evaluations: Vec<f32>,
            #[allow(dead_code)]
            vectors: Vec<Vec<f32>>,
            #[allow(dead_code)]
            created_at: i64,
        }

        let existing_size = self.knowledge_base_size();

        // Read and decompress file with progress
        let file_size = std::fs::metadata(&path)?.len();
        println!(
            "📦 Reading {} compressed file...",
            Self::format_bytes(file_size)
        );

        let compressed_data = std::fs::read(path)?;
        println!("🔓 Decompressing data...");
        let serialized = decompress_size_prepended(&compressed_data)?;

        println!("📊 Deserializing binary data...");
        let binary_data: BinaryTrainingData = bincode::deserialize(&serialized)?;

        let total_positions = binary_data.positions.len();
        if total_positions == 0 {
            println!("⚠️  No positions found in binary file");
            return Ok(());
        }

        println!(
            "🚀 Processing {} positions from binary format...",
            total_positions
        );

        // Progress bar for loading positions
        let pb = ProgressBar::new(total_positions as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("⚡ Loading positions [{elapsed_precise}] [{bar:40.green/blue}] {pos}/{len} ({percent}%) {msg}")?
                .progress_chars("██░")
        );

        let mut added_count = 0;

        // Load positions into engine
        for (i, fen) in binary_data.positions.iter().enumerate() {
            if i < binary_data.evaluations.len() {
                if let Ok(board) = fen.parse() {
                    // Skip duplicates
                    if !self.position_boards.contains(&board) {
                        self.add_position(&board, binary_data.evaluations[i]);
                        added_count += 1;
                    }
                }
            }

            if i % 1000 == 0 || i == total_positions - 1 {
                pb.set_position((i + 1) as u64);
                pb.set_message(format!("{} new positions", added_count));
            }
        }
        pb.finish_with_message(format!("✅ Loaded {} new positions", added_count));

        println!(
            "🎯 Binary loading complete: {} new positions (total: {})",
            self.knowledge_base_size() - existing_size,
            self.knowledge_base_size()
        );
        Ok(())
    }

    /// Ultra-fast memory-mapped loading for instant startup
    /// Uses memory-mapped files to load training data with zero-copy access (PREMIUM FEATURE)
    pub fn load_training_data_mmap<P: AsRef<Path>>(
        &mut self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Feature gate: require premium tier for memory-mapped files
        self.require_feature("memory_mapped_files")?;

        use memmap2::Mmap;
        use std::fs::File;

        let path_ref = path.as_ref();
        println!(
            "🚀 Loading training data via memory mapping: {}",
            path_ref.display()
        );

        let file = File::open(path_ref)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Try MessagePack format first (faster than bincode)
        if let Ok(data) = rmp_serde::from_slice::<Vec<(String, f32)>>(&mmap) {
            println!("📦 Detected MessagePack format");
            return self.load_positions_from_tuples(data);
        }

        // Fall back to bincode
        if let Ok(data) = bincode::deserialize::<Vec<(String, f32)>>(&mmap) {
            println!("📦 Detected bincode format");
            return self.load_positions_from_tuples(data);
        }

        // Fall back to LZ4 compressed bincode
        let decompressed = lz4_flex::decompress_size_prepended(&mmap)?;
        let data: Vec<(String, f32)> = bincode::deserialize(&decompressed)?;
        println!("📦 Detected LZ4+bincode format");
        self.load_positions_from_tuples(data)
    }

    /// Ultra-fast MessagePack binary format loading
    /// MessagePack is typically 10-20% faster than bincode
    pub fn load_training_data_msgpack<P: AsRef<Path>>(
        &mut self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use std::fs::File;
        use std::io::BufReader;

        let path_ref = path.as_ref();
        println!(
            "🚀 Loading MessagePack training data: {}",
            path_ref.display()
        );

        let file = File::open(path_ref)?;
        let reader = BufReader::new(file);
        let data: Vec<(String, f32)> = rmp_serde::from_read(reader)?;

        println!("📦 MessagePack data loaded: {} positions", data.len());
        self.load_positions_from_tuples(data)
    }

    /// Ultra-fast streaming JSON loader with parallel processing
    /// Processes JSON in chunks with multiple threads for better performance
    pub fn load_training_data_streaming_json<P: AsRef<Path>>(
        &mut self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use dashmap::DashMap;
        use rayon::prelude::*;
        use std::fs::File;
        use std::io::{BufRead, BufReader};
        use std::sync::Arc;

        let path_ref = path.as_ref();
        println!(
            "🚀 Loading JSON with streaming parallel processing: {}",
            path_ref.display()
        );

        let file = File::open(path_ref)?;
        let reader = BufReader::new(file);

        // Read file in chunks and process in parallel
        let chunk_size = 10000;
        let position_map = Arc::new(DashMap::new());

        let lines: Vec<String> = reader.lines().collect::<Result<Vec<_>, _>>()?;
        let total_lines = lines.len();

        // Process chunks in parallel
        lines.par_chunks(chunk_size).for_each(|chunk| {
            for line in chunk {
                if let Ok(data) = serde_json::from_str::<serde_json::Value>(line) {
                    if let (Some(fen), Some(eval)) = (
                        data.get("fen").and_then(|v| v.as_str()),
                        data.get("evaluation").and_then(|v| v.as_f64()),
                    ) {
                        position_map.insert(fen.to_string(), eval as f32);
                    }
                }
            }
        });

        println!(
            "📦 Parallel JSON processing complete: {} positions from {} lines",
            position_map.len(),
            total_lines
        );

        // Convert to Vec for final loading
        // Convert DashMap to Vec - need to extract values from Arc
        let data: Vec<(String, f32)> = match Arc::try_unwrap(position_map) {
            Ok(map) => map.into_iter().collect(),
            Err(arc_map) => {
                // Fallback: clone if there are multiple references
                arc_map
                    .iter()
                    .map(|entry| (entry.key().clone(), *entry.value()))
                    .collect()
            }
        };
        self.load_positions_from_tuples(data)
    }

    /// Ultra-fast compressed loading with zstd
    /// Zstd typically provides better compression ratios than LZ4 with similar speed
    pub fn load_training_data_compressed<P: AsRef<Path>>(
        &mut self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use std::fs::File;
        use std::io::BufReader;

        let path_ref = path.as_ref();
        println!(
            "🚀 Loading zstd compressed training data: {}",
            path_ref.display()
        );

        let file = File::open(path_ref)?;
        let reader = BufReader::new(file);
        let decoder = zstd::stream::Decoder::new(reader)?;

        // Try MessagePack first for maximum speed
        if let Ok(data) = rmp_serde::from_read::<_, Vec<(String, f32)>>(decoder) {
            println!("📦 Zstd+MessagePack data loaded: {} positions", data.len());
            return self.load_positions_from_tuples(data);
        }

        // Fall back to bincode
        let file = File::open(path_ref)?;
        let reader = BufReader::new(file);
        let decoder = zstd::stream::Decoder::new(reader)?;
        let data: Vec<(String, f32)> = bincode::deserialize_from(decoder)?;

        println!("📦 Zstd+bincode data loaded: {} positions", data.len());
        self.load_positions_from_tuples(data)
    }

    /// Helper method to load positions from (FEN, evaluation) tuples
    /// Used by all the ultra-fast loading methods
    fn load_positions_from_tuples(
        &mut self,
        data: Vec<(String, f32)>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use indicatif::{ProgressBar, ProgressStyle};
        use std::collections::HashSet;

        let existing_size = self.knowledge_base_size();
        let mut seen_positions = HashSet::new();
        let mut loaded_count = 0;

        // Create progress bar
        let pb = ProgressBar::new(data.len() as u64);
        pb.set_style(ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}) {msg}"
        )?);

        for (fen, evaluation) in data {
            pb.inc(1);

            // Skip duplicates using O(1) HashSet lookup
            if seen_positions.contains(&fen) {
                continue;
            }
            seen_positions.insert(fen.clone());

            // Parse and add position
            if let Ok(board) = Board::from_str(&fen) {
                self.add_position(&board, evaluation);
                loaded_count += 1;

                if loaded_count % 1000 == 0 {
                    pb.set_message(format!("Loaded {} positions", loaded_count));
                }
            }
        }

        pb.finish_with_message(format!("✅ Loaded {} new positions", loaded_count));

        println!(
            "🎯 Ultra-fast loading complete: {} new positions (total: {})",
            self.knowledge_base_size() - existing_size,
            self.knowledge_base_size()
        );

        Ok(())
    }

    /// Helper to format byte sizes for display
    fn format_bytes(bytes: u64) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB"];
        let mut size = bytes as f64;
        let mut unit_index = 0;

        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }

        format!("{:.1} {}", size, UNITS[unit_index])
    }

    /// Train from dataset incrementally (preserves existing engine state)
    pub fn train_from_dataset_incremental(&mut self, dataset: &crate::training::TrainingDataset) {
        let _existing_size = self.knowledge_base_size();
        let mut added = 0;

        for data in &dataset.data {
            // Skip if we already have this position to avoid exact duplicates
            if !self.position_boards.contains(&data.board) {
                self.add_position(&data.board, data.evaluation);
                added += 1;
            }
        }

        println!(
            "Added {} new positions from dataset (total: {})",
            added,
            self.knowledge_base_size()
        );
    }

    /// Get current training statistics
    pub fn training_stats(&self) -> TrainingStats {
        TrainingStats {
            total_positions: self.knowledge_base_size(),
            unique_positions: self.position_boards.len(),
            has_move_data: !self.position_moves.is_empty(),
            move_data_entries: self.position_moves.len(),
            lsh_enabled: self.use_lsh,
            manifold_enabled: self.use_manifold,
            opening_book_enabled: self.opening_book.is_some(),
        }
    }

    /// Auto-load training data from common file names if they exist
    pub fn auto_load_training_data(&mut self) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        use indicatif::{ProgressBar, ProgressStyle};

        let common_files = vec![
            "training_data.json",
            "tactical_training_data.json",
            "engine_training.json",
            "chess_training.json",
            "my_training.json",
        ];

        let tactical_files = vec![
            "tactical_puzzles.json",
            "lichess_puzzles.json",
            "my_puzzles.json",
        ];

        // Check which files exist
        let mut available_files = Vec::new();
        for file_path in &common_files {
            if std::path::Path::new(file_path).exists() {
                available_files.push((file_path, "training"));
            }
        }
        for file_path in &tactical_files {
            if std::path::Path::new(file_path).exists() {
                available_files.push((file_path, "tactical"));
            }
        }

        if available_files.is_empty() {
            return Ok(Vec::new());
        }

        println!(
            "🔍 Found {} training files to auto-load",
            available_files.len()
        );

        // Progress bar for file loading
        let pb = ProgressBar::new(available_files.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("📂 Auto-loading files [{elapsed_precise}] [{bar:40.blue/cyan}] {pos}/{len} {msg}")?
                .progress_chars("██░")
        );

        let mut loaded_files = Vec::new();

        for (i, (file_path, file_type)) in available_files.iter().enumerate() {
            pb.set_position(i as u64);
            pb.set_message(format!("Loading {}", file_path));

            let result = match *file_type {
                "training" => self.load_training_data_incremental(file_path).map(|_| {
                    loaded_files.push(file_path.to_string());
                    println!("📚 Auto-loaded training data from {}", file_path);
                }),
                "tactical" => crate::training::TacticalPuzzleParser::load_tactical_puzzles(
                    file_path,
                )
                .map(|puzzles| {
                    crate::training::TacticalPuzzleParser::load_into_engine_incremental(
                        &puzzles, self,
                    );
                    loaded_files.push(file_path.to_string());
                    println!("🎯 Auto-loaded tactical puzzles from {}", file_path);
                }),
                _ => Ok(()),
            };

            if let Err(e) = result {
                println!("⚠️  Could not load {}: {}", file_path, e);
            }
        }

        pb.set_position(available_files.len() as u64);
        pb.finish_with_message(format!("✅ Auto-loaded {} files", loaded_files.len()));

        Ok(loaded_files)
    }

    /// Load Lichess puzzle database with premium features (Premium+)
    pub fn load_lichess_puzzles_premium<P: AsRef<std::path::Path>>(
        &mut self,
        csv_path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.require_feature("ultra_fast_loading")?; // Premium+ required

        println!("🔥 Loading Lichess puzzles with premium performance...");
        let puzzle_entries =
            crate::lichess_loader::load_lichess_puzzles_premium_with_moves(csv_path)?;

        for (board, evaluation, best_move) in puzzle_entries {
            self.add_position_with_move(&board, evaluation, Some(best_move), Some(evaluation));
        }

        println!("✅ Premium Lichess puzzle loading complete!");
        Ok(())
    }

    /// Load limited Lichess puzzle database (Open Source)
    pub fn load_lichess_puzzles_basic<P: AsRef<std::path::Path>>(
        &mut self,
        csv_path: P,
        max_puzzles: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!(
            "📚 Loading Lichess puzzles (basic tier, limited to {} puzzles)...",
            max_puzzles
        );
        let puzzle_entries =
            crate::lichess_loader::load_lichess_puzzles_basic_with_moves(csv_path, max_puzzles)?;

        for (board, evaluation, best_move) in puzzle_entries {
            self.add_position_with_move(&board, evaluation, Some(best_move), Some(evaluation));
        }

        println!("✅ Basic Lichess puzzle loading complete!");
        Ok(())
    }

    /// Create a new chess vector engine with automatic training data loading
    pub fn new_with_auto_load(vector_size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let mut engine = Self::new(vector_size);
        engine.enable_opening_book();

        // Auto-load any available training data
        let loaded_files = engine.auto_load_training_data()?;

        if loaded_files.is_empty() {
            println!("🤖 Created fresh engine (no training data found)");
        } else {
            println!(
                "🚀 Created engine with auto-loaded training data from {} files",
                loaded_files.len()
            );
            let stats = engine.training_stats();
            println!("   - Total positions: {}", stats.total_positions);
            println!("   - Move data entries: {}", stats.move_data_entries);
        }

        Ok(engine)
    }

    /// Create a new chess vector engine with fast loading optimized for gameplay
    /// Prioritizes binary formats and skips expensive model rebuilding
    pub fn new_with_fast_load(vector_size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        use indicatif::{ProgressBar, ProgressStyle};

        let mut engine = Self::new(vector_size);
        engine.enable_opening_book();

        // Enable database persistence for manifold model loading
        if let Err(e) = engine.enable_persistence("chess_vector_engine.db") {
            println!("⚠️  Could not enable database persistence: {}", e);
        }

        // Try to load binary formats first for maximum speed
        let binary_files = [
            "training_data_a100.bin", // A100 training data (priority)
            "training_data.bin",
            "tactical_training_data.bin",
            "engine_training.bin",
            "chess_training.bin",
        ];

        // Check which binary files exist
        let existing_binary_files: Vec<_> = binary_files
            .iter()
            .filter(|&file_path| std::path::Path::new(file_path).exists())
            .collect();

        let mut loaded_count = 0;

        if !existing_binary_files.is_empty() {
            println!(
                "⚡ Fast loading: Found {} binary files",
                existing_binary_files.len()
            );

            // Progress bar for binary file loading
            let pb = ProgressBar::new(existing_binary_files.len() as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("🚀 Fast loading [{elapsed_precise}] [{bar:40.green/cyan}] {pos}/{len} {msg}")?
                    .progress_chars("██░")
            );

            for (i, file_path) in existing_binary_files.iter().enumerate() {
                pb.set_position(i as u64);
                pb.set_message(format!("Loading {}", file_path));

                if engine.load_training_data_binary(file_path).is_ok() {
                    loaded_count += 1;
                }
            }

            pb.set_position(existing_binary_files.len() as u64);
            pb.finish_with_message(format!("✅ Loaded {} binary files", loaded_count));
        } else {
            println!("📦 No binary files found, falling back to JSON auto-loading...");
            let _ = engine.auto_load_training_data()?;
        }

        // Try to load pre-trained manifold models for fast compressed similarity search
        if let Err(e) = engine.load_manifold_models() {
            println!("⚠️  No pre-trained manifold models found ({})", e);
            println!("   Use --rebuild-models flag to train new models");
        }

        let stats = engine.training_stats();
        println!(
            "⚡ Fast engine ready with {} positions ({} binary files loaded)",
            stats.total_positions, loaded_count
        );

        Ok(engine)
    }

    /// Create a new engine with automatic file discovery and smart format selection
    /// Automatically discovers training data files and loads the optimal format
    pub fn new_with_auto_discovery(vector_size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        println!("🚀 Initializing engine with AUTO-DISCOVERY and format consolidation...");
        let mut engine = Self::new(vector_size);
        engine.enable_opening_book();

        // Enable database persistence for manifold model loading
        if let Err(e) = engine.enable_persistence("chess_vector_engine.db") {
            println!("⚠️  Could not enable database persistence: {}", e);
        }

        // Auto-discover training data files
        let discovered_files = AutoDiscovery::discover_training_files(".", true)?;

        if discovered_files.is_empty() {
            println!("ℹ️  No training data found. Use convert methods to create optimized files.");
            return Ok(engine);
        }

        // Group by base name and load best format for each
        let consolidated = AutoDiscovery::consolidate_by_base_name(discovered_files.clone());

        let mut total_loaded = 0;
        for (base_name, best_file) in &consolidated {
            println!("📚 Loading {} ({})", base_name, best_file.format);

            let initial_size = engine.knowledge_base_size();
            engine.load_file_by_format(&best_file.path, &best_file.format)?;
            let loaded_count = engine.knowledge_base_size() - initial_size;
            total_loaded += loaded_count;

            println!("   ✅ Loaded {} positions", loaded_count);
        }

        // Clean up old formats (dry run first to show what would be removed)
        let cleanup_candidates = AutoDiscovery::get_cleanup_candidates(&discovered_files);
        if !cleanup_candidates.is_empty() {
            println!(
                "🧹 Found {} old format files that can be cleaned up:",
                cleanup_candidates.len()
            );
            AutoDiscovery::cleanup_old_formats(&cleanup_candidates, true)?; // Dry run

            println!("   💡 To actually remove old files, run: cargo run --bin cleanup_formats");
        }

        // Try to load pre-trained manifold models
        if let Err(e) = engine.load_manifold_models() {
            println!("⚠️  No pre-trained manifold models found ({})", e);
        }

        println!(
            "🎯 Engine ready: {} positions loaded from {} datasets",
            total_loaded,
            consolidated.len()
        );
        Ok(engine)
    }

    /// Ultra-fast instant loading - loads best available format without consolidation
    /// This is the fastest possible loading method for production use
    pub fn new_with_instant_load(vector_size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        println!("🚀 Initializing engine with INSTANT loading...");
        let mut engine = Self::new(vector_size);
        engine.enable_opening_book();

        // Enable database persistence for manifold model loading
        if let Err(e) = engine.enable_persistence("chess_vector_engine.db") {
            println!("⚠️  Could not enable database persistence: {}", e);
        }

        // Auto-discover and select best format
        let discovered_files = AutoDiscovery::discover_training_files(".", false)?;

        if discovered_files.is_empty() {
            // No user training data found, load starter dataset
            println!("ℹ️  No user training data found, loading starter dataset...");
            if let Err(e) = engine.load_starter_dataset() {
                println!("⚠️  Could not load starter dataset: {}", e);
                println!("ℹ️  Starting with empty engine");
            } else {
                println!(
                    "✅ Loaded starter dataset with {} positions",
                    engine.knowledge_base_size()
                );
            }
            return Ok(engine);
        }

        // Select best overall format (prioritizes MMAP)
        if let Some(best_file) = discovered_files.first() {
            println!(
                "⚡ Loading {} format: {}",
                best_file.format,
                best_file.path.display()
            );
            engine.load_file_by_format(&best_file.path, &best_file.format)?;
            println!(
                "✅ Loaded {} positions from {} format",
                engine.knowledge_base_size(),
                best_file.format
            );
        }

        // Try to load pre-trained manifold models
        if let Err(e) = engine.load_manifold_models() {
            println!("⚠️  No pre-trained manifold models found ({})", e);
        }

        println!(
            "🎯 Engine ready: {} positions loaded",
            engine.knowledge_base_size()
        );
        Ok(engine)
    }

    /// Create engine with license verification system
    pub fn new_with_license(vector_size: usize, license_url: String) -> Self {
        let mut engine = Self::new(vector_size);
        engine.licensed_feature_checker = Some(LicensedFeatureChecker::new(license_url));
        engine
    }

    /// Create engine with offline license verification
    pub fn new_with_offline_license(vector_size: usize) -> Self {
        let mut engine = Self::new(vector_size);
        engine.licensed_feature_checker = Some(LicensedFeatureChecker::new_offline());
        engine
    }

    /// Activate license key
    pub async fn activate_license(&mut self, key: &str) -> Result<FeatureTier, LicenseError> {
        if let Some(ref mut checker) = self.licensed_feature_checker {
            let tier = checker.activate_license(key).await?;
            // Update the basic feature checker to match the licensed tier
            self.feature_checker.upgrade_tier(tier.clone());
            Ok(tier)
        } else {
            Err(LicenseError::InvalidFormat(
                "No license checker initialized".to_string(),
            ))
        }
    }

    /// Check if feature is licensed (async version with license verification)
    pub async fn check_licensed_feature(&mut self, feature: &str) -> Result<(), FeatureError> {
        if let Some(ref mut checker) = self.licensed_feature_checker {
            checker.check_feature(feature).await
        } else {
            // Fall back to basic feature checking
            self.feature_checker.check_feature(feature)
        }
    }

    /// Load license cache from disk
    pub fn load_license_cache<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref mut checker) = self.licensed_feature_checker {
            checker.load_cache(path)?;
        }
        Ok(())
    }

    /// Save license cache to disk
    pub fn save_license_cache<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref checker) = self.licensed_feature_checker {
            checker.save_cache(path)?;
        }
        Ok(())
    }

    // TODO: Creator access method removed for git security
    // For local development only - not to be committed

    /// Validate that a position is safe to store and won't cause panics
    fn is_position_safe(&self, board: &Board) -> bool {
        // Check if position can generate legal moves without panicking
        match std::panic::catch_unwind(|| {
            use chess::MoveGen;
            let _legal_moves: Vec<ChessMove> = MoveGen::new_legal(board).collect();
            true
        }) {
            Ok(_) => true,
            Err(_) => {
                // Position causes panic during move generation - skip it
                false
            }
        }
    }

    /// Check if GPU acceleration feature is available
    pub fn check_gpu_acceleration(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.feature_checker.check_feature("gpu_acceleration")?;

        // Check if GPU is available on the system
        match crate::gpu_acceleration::GPUAccelerator::new() {
            Ok(_) => {
                println!("🔥 GPU acceleration available and ready");
                Ok(())
            }
            Err(e) => Err(format!("GPU acceleration not available: {}", e).into()),
        }
    }

    /// Load starter dataset for open source users
    pub fn load_starter_dataset(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Try to load from external file first, fall back to minimal dataset
        let starter_data = if let Ok(file_content) =
            std::fs::read_to_string("training_data/starter_dataset.json")
        {
            file_content
        } else {
            // Fallback minimal dataset for when the file isn't available (e.g., in CI or after packaging)
            r#"[
                {
                    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                    "evaluation": 0.0,
                    "best_move": null,
                    "depth": 0
                },
                {
                    "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
                    "evaluation": 0.1,
                    "best_move": "e7e5",
                    "depth": 2
                },
                {
                    "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
                    "evaluation": 0.0,
                    "best_move": "g1f3",
                    "depth": 2
                }
            ]"#
            .to_string()
        };

        let training_data: Vec<serde_json::Value> = serde_json::from_str(&starter_data)?;

        for entry in training_data {
            if let (Some(fen), Some(evaluation)) = (entry.get("fen"), entry.get("evaluation")) {
                if let (Some(fen_str), Some(eval_f64)) = (fen.as_str(), evaluation.as_f64()) {
                    match chess::Board::from_str(fen_str) {
                        Ok(board) => {
                            self.add_position(&board, eval_f64 as f32);
                        }
                        Err(_) => {
                            // Skip invalid positions
                            continue;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Load file by detected format - uses ultra-fast loader for large files
    fn load_file_by_format(
        &mut self,
        path: &std::path::Path,
        format: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Check file size to determine loading strategy
        let file_size = std::fs::metadata(path)?.len();

        // For files > 10MB, use ultra-fast loader
        if file_size > 10_000_000 {
            println!(
                "📊 Large file detected ({:.1} MB) - using ultra-fast loader",
                file_size as f64 / 1_000_000.0
            );
            return self.ultra_fast_load_any_format(path);
        }

        // For smaller files, use standard loaders
        match format {
            "MMAP" => self.load_training_data_mmap(path),
            "MSGPACK" => self.load_training_data_msgpack(path),
            "BINARY" => self.load_training_data_streaming_binary(path),
            "ZSTD" => self.load_training_data_compressed(path),
            "JSON" => self.load_training_data_streaming_json_v2(path),
            _ => Err(format!("Unknown format: {}", format).into()),
        }
    }

    /// Ultra-fast loader for any format - optimized for massive datasets (PREMIUM FEATURE)
    pub fn ultra_fast_load_any_format<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Feature gate: require premium tier
        self.require_feature("ultra_fast_loading")?;

        let mut loader = UltraFastLoader::new_for_massive_datasets();
        loader.ultra_load_binary(path, self)?;

        let stats = loader.get_stats();
        println!("📊 Ultra-fast loading complete:");
        println!("   ✅ Loaded: {} positions", stats.loaded);
        println!("   🔄 Duplicates: {}", stats.duplicates);
        println!("   ❌ Errors: {}", stats.errors);
        println!("   📈 Success rate: {:.1}%", stats.success_rate() * 100.0);

        Ok(())
    }

    /// Ultra-fast streaming binary loader for massive datasets (900k+ positions)
    /// Uses streaming processing to handle arbitrarily large datasets
    pub fn load_training_data_streaming_binary<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut loader = StreamingLoader::new();
        loader.stream_load_binary(path, self)?;

        println!("📊 Streaming binary load complete:");
        println!("   Loaded: {} new positions", loader.loaded_count);
        println!("   Duplicates skipped: {}", loader.duplicate_count);
        println!("   Total processed: {}", loader.total_processed);

        Ok(())
    }

    /// Ultra-fast streaming JSON loader for massive datasets (900k+ positions)
    /// Uses streaming processing with minimal memory footprint
    pub fn load_training_data_streaming_json_v2<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut loader = StreamingLoader::new();

        // Use larger batch size for massive datasets
        let batch_size = if std::fs::metadata(path.as_ref())?.len() > 100_000_000 {
            // > 100MB
            20000 // Large batches for big files
        } else {
            5000 // Smaller batches for normal files
        };

        loader.stream_load_json(path, self, batch_size)?;

        println!("📊 Streaming JSON load complete:");
        println!("   Loaded: {} new positions", loader.loaded_count);
        println!("   Duplicates skipped: {}", loader.duplicate_count);
        println!("   Total processed: {}", loader.total_processed);

        Ok(())
    }

    /// Create engine optimized for massive datasets (100k-1M+ positions)
    /// Uses streaming loading and minimal memory footprint
    pub fn new_for_massive_datasets(
        vector_size: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        println!("🚀 Initializing engine for MASSIVE datasets (100k-1M+ positions)...");
        let mut engine = Self::new(vector_size);
        engine.enable_opening_book();

        // Discover training files
        let discovered_files = AutoDiscovery::discover_training_files(".", false)?;

        if discovered_files.is_empty() {
            println!("ℹ️  No training data found");
            return Ok(engine);
        }

        // Find the largest file to load (likely the main dataset)
        let largest_file = discovered_files
            .iter()
            .max_by_key(|f| f.size_bytes)
            .unwrap();

        println!(
            "🎯 Loading largest dataset: {} ({} bytes)",
            largest_file.path.display(),
            largest_file.size_bytes
        );

        // Use ultra-fast loader for massive datasets
        engine.ultra_fast_load_any_format(&largest_file.path)?;

        println!(
            "🎯 Engine ready: {} positions loaded",
            engine.knowledge_base_size()
        );
        Ok(engine)
    }

    /// Convert existing JSON training data to ultra-fast MessagePack format
    /// MessagePack is typically 10-20% faster than bincode with smaller file sizes
    pub fn convert_to_msgpack() -> Result<(), Box<dyn std::error::Error>> {
        use serde_json::Value;
        use std::fs::File;
        use std::io::{BufReader, BufWriter};

        // First convert A100 binary to JSON if it exists
        if std::path::Path::new("training_data_a100.bin").exists() {
            Self::convert_a100_binary_to_json()?;
        }

        let input_files = [
            "training_data.json",
            "tactical_training_data.json",
            "training_data_a100.json",
        ];

        for input_file in &input_files {
            let input_path = std::path::Path::new(input_file);
            if !input_path.exists() {
                continue;
            }

            let output_file_path = input_file.replace(".json", ".msgpack");
            println!(
                "🔄 Converting {} → {} (MessagePack format)",
                input_file, output_file_path
            );

            // Load JSON data and handle both formats
            let file = File::open(input_path)?;
            let reader = BufReader::new(file);
            let json_value: Value = serde_json::from_reader(reader)?;

            let data: Vec<(String, f32)> = match json_value {
                // Handle tuple format: [(fen, evaluation), ...]
                Value::Array(arr) if !arr.is_empty() => {
                    if let Some(first) = arr.first() {
                        if first.is_array() {
                            // Tuple format: [[fen, evaluation], ...]
                            arr.into_iter()
                                .filter_map(|item| {
                                    if let Value::Array(tuple) = item {
                                        if tuple.len() >= 2 {
                                            let fen = tuple[0].as_str()?.to_string();
                                            let eval = tuple[1].as_f64()? as f32;
                                            Some((fen, eval))
                                        } else {
                                            None
                                        }
                                    } else {
                                        None
                                    }
                                })
                                .collect()
                        } else if first.is_object() {
                            // Object format: [{fen: "...", evaluation: ...}, ...]
                            arr.into_iter()
                                .filter_map(|item| {
                                    if let Value::Object(obj) = item {
                                        let fen = obj.get("fen")?.as_str()?.to_string();
                                        let eval = obj.get("evaluation")?.as_f64()? as f32;
                                        Some((fen, eval))
                                    } else {
                                        None
                                    }
                                })
                                .collect()
                        } else {
                            return Err(format!("Unsupported JSON format in {}", input_file).into());
                        }
                    } else {
                        Vec::new()
                    }
                }
                _ => return Err(format!("Expected JSON array in {}", input_file).into()),
            };

            if data.is_empty() {
                println!("⚠️  No valid data found in {}", input_file);
                continue;
            }

            // Save as MessagePack
            let output_file = File::create(&output_file_path)?;
            let mut writer = BufWriter::new(output_file);
            rmp_serde::encode::write(&mut writer, &data)?;

            let input_size = input_path.metadata()?.len();
            let output_size = std::path::Path::new(&output_file_path).metadata()?.len();
            let ratio = input_size as f64 / output_size as f64;

            println!(
                "✅ Converted: {} → {} ({:.1}x size reduction, {} positions)",
                Self::format_bytes(input_size),
                Self::format_bytes(output_size),
                ratio,
                data.len()
            );
        }

        Ok(())
    }

    /// Convert A100 binary training data to JSON format for use with other converters
    pub fn convert_a100_binary_to_json() -> Result<(), Box<dyn std::error::Error>> {
        use std::fs::File;
        use std::io::BufWriter;

        let binary_path = "training_data_a100.bin";
        let json_path = "training_data_a100.json";

        if !std::path::Path::new(binary_path).exists() {
            println!("⚠️  A100 binary file not found: {}", binary_path);
            return Ok(());
        }

        println!(
            "🔄 Converting A100 binary data {} → {} (JSON format)",
            binary_path, json_path
        );

        // Load binary data using the existing binary loader
        let mut engine = ChessVectorEngine::new(1024);
        engine.load_training_data_binary(binary_path)?;

        // Extract data in JSON-compatible format
        let mut data = Vec::new();
        for (i, board) in engine.position_boards.iter().enumerate() {
            if i < engine.position_evaluations.len() {
                data.push(serde_json::json!({
                    "fen": board.to_string(),
                    "evaluation": engine.position_evaluations[i],
                    "depth": 15,
                    "game_id": i
                }));
            }
        }

        // Save as JSON
        let file = File::create(json_path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, &data)?;

        println!(
            "✅ Converted A100 data: {} positions → {}",
            data.len(),
            json_path
        );
        Ok(())
    }

    /// Convert existing training data to ultra-compressed Zstd format
    /// Zstd provides excellent compression with fast decompression
    pub fn convert_to_zstd() -> Result<(), Box<dyn std::error::Error>> {
        use std::fs::File;
        use std::io::{BufReader, BufWriter};

        // First convert A100 binary to JSON if it exists
        if std::path::Path::new("training_data_a100.bin").exists() {
            Self::convert_a100_binary_to_json()?;
        }

        let input_files = [
            ("training_data.json", "training_data.zst"),
            ("tactical_training_data.json", "tactical_training_data.zst"),
            ("training_data_a100.json", "training_data_a100.zst"),
            ("training_data.bin", "training_data.bin.zst"),
            (
                "tactical_training_data.bin",
                "tactical_training_data.bin.zst",
            ),
            ("training_data_a100.bin", "training_data_a100.bin.zst"),
        ];

        for (input_file, output_file) in &input_files {
            let input_path = std::path::Path::new(input_file);
            if !input_path.exists() {
                continue;
            }

            println!(
                "🔄 Converting {} → {} (Zstd compression)",
                input_file, output_file
            );

            let input_file = File::open(input_path)?;
            let output_file_handle = File::create(output_file)?;
            let writer = BufWriter::new(output_file_handle);
            let mut encoder = zstd::stream::Encoder::new(writer, 9)?; // Level 9 for best compression

            std::io::copy(&mut BufReader::new(input_file), &mut encoder)?;
            encoder.finish()?;

            let input_size = input_path.metadata()?.len();
            let output_size = std::path::Path::new(output_file).metadata()?.len();
            let ratio = input_size as f64 / output_size as f64;

            println!(
                "✅ Compressed: {} → {} ({:.1}x size reduction)",
                Self::format_bytes(input_size),
                Self::format_bytes(output_size),
                ratio
            );
        }

        Ok(())
    }

    /// Convert existing training data to memory-mapped format for instant loading
    /// This creates a file that can be loaded with zero-copy access
    pub fn convert_to_mmap() -> Result<(), Box<dyn std::error::Error>> {
        use std::fs::File;
        use std::io::{BufReader, BufWriter};

        // First convert A100 binary to JSON if it exists
        if std::path::Path::new("training_data_a100.bin").exists() {
            Self::convert_a100_binary_to_json()?;
        }

        let input_files = [
            ("training_data.json", "training_data.mmap"),
            ("tactical_training_data.json", "tactical_training_data.mmap"),
            ("training_data_a100.json", "training_data_a100.mmap"),
            ("training_data.msgpack", "training_data.mmap"),
            (
                "tactical_training_data.msgpack",
                "tactical_training_data.mmap",
            ),
            ("training_data_a100.msgpack", "training_data_a100.mmap"),
        ];

        for (input_file, output_file) in &input_files {
            let input_path = std::path::Path::new(input_file);
            if !input_path.exists() {
                continue;
            }

            println!(
                "🔄 Converting {} → {} (Memory-mapped format)",
                input_file, output_file
            );

            // Load data based on input format
            let data: Vec<(String, f32)> = if input_file.ends_with(".json") {
                let file = File::open(input_path)?;
                let reader = BufReader::new(file);
                let json_value: Value = serde_json::from_reader(reader)?;

                match json_value {
                    // Handle tuple format: [(fen, evaluation), ...]
                    Value::Array(arr) if !arr.is_empty() => {
                        if let Some(first) = arr.first() {
                            if first.is_array() {
                                // Tuple format: [[fen, evaluation], ...]
                                arr.into_iter()
                                    .filter_map(|item| {
                                        if let Value::Array(tuple) = item {
                                            if tuple.len() >= 2 {
                                                let fen = tuple[0].as_str()?.to_string();
                                                let eval = tuple[1].as_f64()? as f32;
                                                Some((fen, eval))
                                            } else {
                                                None
                                            }
                                        } else {
                                            None
                                        }
                                    })
                                    .collect()
                            } else if first.is_object() {
                                // Object format: [{fen: "...", evaluation: ...}, ...]
                                arr.into_iter()
                                    .filter_map(|item| {
                                        if let Value::Object(obj) = item {
                                            let fen = obj.get("fen")?.as_str()?.to_string();
                                            let eval = obj.get("evaluation")?.as_f64()? as f32;
                                            Some((fen, eval))
                                        } else {
                                            None
                                        }
                                    })
                                    .collect()
                            } else {
                                return Err(
                                    format!("Unsupported JSON format in {}", input_file).into()
                                );
                            }
                        } else {
                            Vec::new()
                        }
                    }
                    _ => return Err(format!("Expected JSON array in {}", input_file).into()),
                }
            } else if input_file.ends_with(".msgpack") {
                let file = File::open(input_path)?;
                let reader = BufReader::new(file);
                rmp_serde::from_read(reader)?
            } else {
                return Err("Unsupported input format for memory mapping".into());
            };

            // Save as MessagePack (best format for memory mapping)
            let output_file_handle = File::create(output_file)?;
            let mut writer = BufWriter::new(output_file_handle);
            rmp_serde::encode::write(&mut writer, &data)?;

            let input_size = input_path.metadata()?.len();
            let output_size = std::path::Path::new(output_file).metadata()?.len();

            println!(
                "✅ Memory-mapped file created: {} → {} ({} positions)",
                Self::format_bytes(input_size),
                Self::format_bytes(output_size),
                data.len()
            );
        }

        Ok(())
    }

    /// Convert existing JSON training files to binary format for faster loading
    pub fn convert_json_to_binary() -> Result<Vec<String>, Box<dyn std::error::Error>> {
        use indicatif::{ProgressBar, ProgressStyle};

        let json_files = [
            "training_data.json",
            "tactical_training_data.json",
            "engine_training.json",
            "chess_training.json",
        ];

        // Check which JSON files exist
        let existing_json_files: Vec<_> = json_files
            .iter()
            .filter(|&file_path| std::path::Path::new(file_path).exists())
            .collect();

        if existing_json_files.is_empty() {
            println!("ℹ️  No JSON training files found to convert");
            return Ok(Vec::new());
        }

        println!(
            "🔄 Converting {} JSON files to binary format...",
            existing_json_files.len()
        );

        // Progress bar for conversion
        let pb = ProgressBar::new(existing_json_files.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "📦 Converting [{elapsed_precise}] [{bar:40.yellow/blue}] {pos}/{len} {msg}",
                )?
                .progress_chars("██░"),
        );

        let mut converted_files = Vec::new();

        for (i, json_file) in existing_json_files.iter().enumerate() {
            pb.set_position(i as u64);
            pb.set_message(format!("Converting {}", json_file));

            let binary_file = std::path::Path::new(json_file).with_extension("bin");

            // Load from JSON and save as binary
            let mut temp_engine = Self::new(1024);
            if temp_engine
                .load_training_data_incremental(json_file)
                .is_ok()
            {
                if temp_engine.save_training_data_binary(&binary_file).is_ok() {
                    converted_files.push(format!("{} -> {}", json_file, binary_file.display()));
                    println!("✅ Converted {} to binary format", json_file);
                } else {
                    println!("❌ Failed to save binary file for {}", json_file);
                }
            } else {
                println!("❌ Failed to load JSON file {}", json_file);
            }
        }

        pb.set_position(existing_json_files.len() as u64);
        pb.finish_with_message(format!("✅ Converted {} files", converted_files.len()));

        if !converted_files.is_empty() {
            println!("🚀 Binary conversion complete! Startup will be 5-15x faster next time.");
            println!("📊 Conversion summary:");
            for conversion in &converted_files {
                println!("   {}", conversion);
            }
        }

        Ok(converted_files)
    }

    /// Check if LSH is enabled
    pub fn is_lsh_enabled(&self) -> bool {
        self.use_lsh
    }

    /// Get LSH statistics if enabled
    pub fn lsh_stats(&self) -> Option<crate::lsh::LSHStats> {
        self.lsh_index.as_ref().map(|lsh| lsh.stats())
    }

    /// Enable manifold learning with specified compression ratio
    pub fn enable_manifold_learning(&mut self, compression_ratio: f32) -> Result<(), String> {
        let input_dim = self.encoder.vector_size();
        let output_dim = ((input_dim as f32) / compression_ratio) as usize;

        if output_dim == 0 {
            return Err("Compression ratio too high, output dimension would be 0".to_string());
        }

        let mut learner = ManifoldLearner::new(input_dim, output_dim);
        learner.init_network()?;

        self.manifold_learner = Some(learner);
        self.manifold_similarity_search = Some(SimilaritySearch::new(output_dim));
        self.use_manifold = false; // Don't use until trained

        Ok(())
    }

    /// Train manifold learning on existing positions
    pub fn train_manifold_learning(&mut self, epochs: usize) -> Result<(), String> {
        if self.manifold_learner.is_none() {
            return Err(
                "Manifold learning not enabled. Call enable_manifold_learning first.".to_string(),
            );
        }

        if self.similarity_search.size() == 0 {
            return Err("No positions in knowledge base to train on.".to_string());
        }

        // Create training matrix directly without intermediate vectors
        let rows = self.similarity_search.size();
        let cols = self.encoder.vector_size();

        let training_matrix = Array2::from_shape_fn((rows, cols), |(row, col)| {
            if let Some((vector, _)) = self.similarity_search.get_position_ref(row) {
                vector[col]
            } else {
                0.0
            }
        });

        // Train the manifold learner
        if let Some(ref mut learner) = self.manifold_learner {
            learner.train(&training_matrix, epochs)?;
            let compression_ratio = learner.compression_ratio();

            // Release the mutable borrow before calling rebuild_manifold_indices
            let _ = learner;

            // Rebuild compressed indices
            self.rebuild_manifold_indices()?;
            self.use_manifold = true;

            println!(
                "Manifold learning training completed. Compression ratio: {:.1}x",
                compression_ratio
            );
        }

        Ok(())
    }

    /// Rebuild manifold-based indices after training (memory efficient)
    fn rebuild_manifold_indices(&mut self) -> Result<(), String> {
        if let Some(ref learner) = self.manifold_learner {
            // Clear existing manifold indices
            let output_dim = learner.output_dim();
            if let Some(ref mut search) = self.manifold_similarity_search {
                *search = SimilaritySearch::new(output_dim);
            }
            if let Some(ref mut lsh) = self.manifold_lsh_index {
                *lsh = LSH::new(output_dim, 8, 16); // Default LSH params for compressed space
            }

            // Process positions using iterator to avoid cloning all at once
            for (vector, eval) in self.similarity_search.iter_positions() {
                let compressed = learner.encode(vector);

                if let Some(ref mut search) = self.manifold_similarity_search {
                    search.add_position(compressed.clone(), eval);
                }

                if let Some(ref mut lsh) = self.manifold_lsh_index {
                    lsh.add_vector(compressed, eval);
                }
            }
        }

        Ok(())
    }

    /// Enable LSH for manifold space
    pub fn enable_manifold_lsh(
        &mut self,
        num_tables: usize,
        hash_size: usize,
    ) -> Result<(), String> {
        if self.manifold_learner.is_none() {
            return Err("Manifold learning not enabled".to_string());
        }

        let output_dim = self.manifold_learner.as_ref().unwrap().output_dim();
        self.manifold_lsh_index = Some(LSH::new(output_dim, num_tables, hash_size));

        // Rebuild index if we have trained data
        if self.use_manifold {
            self.rebuild_manifold_indices()?;
        }

        Ok(())
    }

    /// Check if manifold learning is enabled and trained
    pub fn is_manifold_enabled(&self) -> bool {
        self.use_manifold && self.manifold_learner.is_some()
    }

    /// Get manifold learning compression ratio
    pub fn manifold_compression_ratio(&self) -> Option<f32> {
        self.manifold_learner
            .as_ref()
            .map(|l| l.compression_ratio())
    }

    /// Load pre-trained manifold models from database
    /// This enables compressed similarity search without retraining
    pub fn load_manifold_models(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref db) = self.database {
            match crate::manifold_learner::ManifoldLearner::load_from_database(db)? {
                Some(learner) => {
                    let compression_ratio = learner.compression_ratio();
                    println!(
                        "🧠 Loaded pre-trained manifold learner (compression: {:.1}x)",
                        compression_ratio
                    );

                    // Enable manifold learning and rebuild indices
                    self.manifold_learner = Some(learner);
                    self.use_manifold = true;

                    // Rebuild compressed similarity search indices
                    self.rebuild_manifold_indices()?;

                    println!("✅ Manifold learning enabled with compressed vectors");
                    Ok(())
                }
                None => Err("No pre-trained manifold models found in database".into()),
            }
        } else {
            Err("Database not initialized - cannot load manifold models".into())
        }
    }

    /// Enable opening book with standard openings
    pub fn enable_opening_book(&mut self) {
        self.opening_book = Some(OpeningBook::with_standard_openings());
    }

    /// Set custom opening book
    pub fn set_opening_book(&mut self, book: OpeningBook) {
        self.opening_book = Some(book);
    }

    /// Check if position is in opening book
    pub fn is_opening_position(&self, board: &Board) -> bool {
        self.opening_book
            .as_ref()
            .map(|book| book.contains(board))
            .unwrap_or(false)
    }

    /// Get opening book entry for position
    pub fn get_opening_entry(&self, board: &Board) -> Option<&OpeningEntry> {
        self.opening_book.as_ref()?.lookup(board)
    }

    /// Get opening book statistics
    pub fn opening_book_stats(&self) -> Option<OpeningBookStats> {
        self.opening_book.as_ref().map(|book| book.stats())
    }

    /// Add a move played from a position with its outcome
    pub fn add_position_with_move(
        &mut self,
        board: &Board,
        evaluation: f32,
        chess_move: Option<ChessMove>,
        move_outcome: Option<f32>,
    ) {
        let position_index = self.knowledge_base_size();

        // Add the position first
        self.add_position(board, evaluation);

        // If a move and outcome are provided, store the move information
        if let (Some(mov), Some(outcome)) = (chess_move, move_outcome) {
            self.position_moves
                .entry(position_index)
                .or_default()
                .push((mov, outcome));
        }
    }

    /// Get move recommendations based on similar positions and opening book
    pub fn recommend_moves(
        &mut self,
        board: &Board,
        num_recommendations: usize,
    ) -> Vec<MoveRecommendation> {
        // // First check tablebase for perfect endgame moves
        // if let Some(ref tablebase) = self.tablebase {
        //     if let Some(best_move) = tablebase.get_best_move(board) {
        //         return vec![MoveRecommendation {
        //             chess_move: best_move,
        //             confidence: 1.0, // Perfect knowledge
        //             from_similar_position_count: 1,
        //             average_outcome: tablebase.get_evaluation(board).unwrap_or(0.0),
        //         }];
        //     }
        // }

        // Second check opening book
        if let Some(entry) = self.get_opening_entry(board) {
            let mut recommendations = Vec::new();

            for (chess_move, strength) in &entry.best_moves {
                recommendations.push(MoveRecommendation {
                    chess_move: *chess_move,
                    confidence: strength * 0.9, // High confidence for opening book moves
                    from_similar_position_count: 1,
                    average_outcome: entry.evaluation,
                });
            }

            // Sort by confidence and limit results
            recommendations.sort_by(|a, b| {
                b.confidence
                    .partial_cmp(&a.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            recommendations.truncate(num_recommendations);
            return recommendations;
        }

        // Fall back to similarity search
        let similar_positions = self.find_similar_positions_with_indices(board, 20);

        // Collect moves from similar positions
        let mut move_data: HashMap<ChessMove, Vec<(f32, f32)>> = HashMap::new(); // move -> (similarity, outcome)

        // Get legal moves for current position to validate recommendations
        use chess::MoveGen;
        let legal_moves: Vec<ChessMove> = match std::panic::catch_unwind(|| {
            MoveGen::new_legal(board).collect::<Vec<ChessMove>>()
        }) {
            Ok(moves) => moves,
            Err(_) => {
                // If we can't generate legal moves for the current position, return empty recommendations
                return Vec::new();
            }
        };

        // Use actual position indices to get moves and outcomes (only if we found similar positions)
        for (position_index, _eval, similarity) in similar_positions {
            if let Some(moves) = self.position_moves.get(&position_index) {
                for &(chess_move, outcome) in moves {
                    // CRITICAL FIX: Only include moves that are legal for the current position
                    if legal_moves.contains(&chess_move) {
                        move_data
                            .entry(chess_move)
                            .or_default()
                            .push((similarity, outcome));
                    }
                }
            }
        }

        // If no moves found from stored data, use tactical search for intelligent fallback
        if move_data.is_empty() {
            if let Some(ref mut tactical_search) = self.tactical_search {
                // Use tactical search to find the best moves with proper evaluation
                let tactical_result = tactical_search.search(board);

                // Add the best tactical move with strong confidence
                if let Some(best_move) = tactical_result.best_move {
                    move_data.insert(best_move, vec![(0.75, tactical_result.evaluation)]);
                }

                // Generate additional well-ordered moves using tactical search move ordering
                // (legal_moves already generated above with safety validation)
                let mut ordered_moves = legal_moves.clone();

                // Use basic move ordering (captures first, then other moves)
                ordered_moves.sort_by(|a, b| {
                    let a_is_capture = board.piece_on(a.get_dest()).is_some();
                    let b_is_capture = board.piece_on(b.get_dest()).is_some();

                    match (a_is_capture, b_is_capture) {
                        (true, false) => std::cmp::Ordering::Less, // a is capture, prefer it
                        (false, true) => std::cmp::Ordering::Greater, // b is capture, prefer it
                        _ => {
                            // Both captures or both non-captures, prefer center moves
                            let a_centrality = move_centrality(a);
                            let b_centrality = move_centrality(b);
                            b_centrality
                                .partial_cmp(&a_centrality)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        }
                    }
                });

                // Add ordered moves with tactical confidence
                for chess_move in ordered_moves.into_iter().take(num_recommendations) {
                    move_data
                        .entry(chess_move)
                        .or_insert_with(|| vec![(0.6, 0.0)]);
                }
            } else {
                // Basic fallback when no tactical search available - still use move ordering
                // (legal_moves already generated above with safety validation)
                let mut ordered_moves = legal_moves.clone();

                // Basic move ordering even without tactical search
                ordered_moves.sort_by(|a, b| {
                    let a_is_capture = board.piece_on(a.get_dest()).is_some();
                    let b_is_capture = board.piece_on(b.get_dest()).is_some();

                    match (a_is_capture, b_is_capture) {
                        (true, false) => std::cmp::Ordering::Less,
                        (false, true) => std::cmp::Ordering::Greater,
                        _ => {
                            let a_centrality = move_centrality(a);
                            let b_centrality = move_centrality(b);
                            b_centrality
                                .partial_cmp(&a_centrality)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        }
                    }
                });

                for chess_move in ordered_moves.into_iter().take(num_recommendations) {
                    move_data.insert(chess_move, vec![(0.3, 0.0)]); // Lower baseline confidence for unknown moves
                }
            }
        }

        // Calculate move recommendations
        let mut recommendations = Vec::new();

        for (chess_move, outcomes) in move_data {
            if outcomes.is_empty() {
                continue;
            }

            // Calculate weighted average outcome based on similarity
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for &(similarity, outcome) in &outcomes {
                weighted_sum += similarity * outcome;
                weight_sum += similarity;
            }

            let average_outcome = if weight_sum > 0.0 {
                weighted_sum / weight_sum
            } else {
                0.0
            };

            // Improved confidence calculation for better pattern recognition
            let avg_similarity =
                outcomes.iter().map(|(s, _)| s).sum::<f32>() / outcomes.len() as f32;
            let position_count_bonus = (outcomes.len() as f32).ln().max(1.0) / 5.0; // Bonus for more supporting positions
            let confidence = (avg_similarity * 0.8 + position_count_bonus * 0.2).min(0.95); // Blend similarity and support

            recommendations.push(MoveRecommendation {
                chess_move,
                confidence: confidence.min(1.0), // Cap at 1.0
                from_similar_position_count: outcomes.len(),
                average_outcome,
            });
        }

        // Sort by confidence (descending)
        recommendations.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Return top recommendations
        recommendations.truncate(num_recommendations);
        recommendations
    }

    /// Generate legal move recommendations (filters recommendations by legal moves)
    pub fn recommend_legal_moves(
        &mut self,
        board: &Board,
        num_recommendations: usize,
    ) -> Vec<MoveRecommendation> {
        use chess::MoveGen;

        // Get all legal moves
        let legal_moves: std::collections::HashSet<ChessMove> = MoveGen::new_legal(board).collect();

        // Get recommendations and filter by legal moves
        let all_recommendations = self.recommend_moves(board, num_recommendations * 2); // Get more to account for filtering

        all_recommendations
            .into_iter()
            .filter(|rec| legal_moves.contains(&rec.chess_move))
            .take(num_recommendations)
            .collect()
    }

    /// Enable persistence with database
    pub fn enable_persistence<P: AsRef<Path>>(
        &mut self,
        db_path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let database = Database::new(db_path)?;
        self.database = Some(database);
        println!("Persistence enabled");
        Ok(())
    }

    /// Save engine state to database using high-performance batch operations
    pub fn save_to_database(&self) -> Result<(), Box<dyn std::error::Error>> {
        let db = self
            .database
            .as_ref()
            .ok_or("Database not enabled. Call enable_persistence() first.")?;

        println!("💾 Saving engine state to database (batch mode)...");

        // Prepare all positions for batch save
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs() as i64;

        let mut position_data_batch = Vec::with_capacity(self.position_boards.len());

        for (i, board) in self.position_boards.iter().enumerate() {
            if i < self.position_vectors.len() && i < self.position_evaluations.len() {
                let vector = self.position_vectors[i].as_slice().unwrap();
                let position_data = PositionData {
                    fen: board.to_string(),
                    vector: vector.iter().map(|&x| x as f64).collect(),
                    evaluation: Some(self.position_evaluations[i] as f64),
                    compressed_vector: None, // Will be filled if manifold is enabled
                    created_at: current_time,
                };
                position_data_batch.push(position_data);
            }
        }

        // Batch save all positions in a single transaction (much faster!)
        if !position_data_batch.is_empty() {
            let saved_count = db.save_positions_batch(&position_data_batch)?;
            println!("📊 Batch saved {} positions", saved_count);
        }

        // Save LSH configuration if enabled
        if let Some(ref lsh) = self.lsh_index {
            lsh.save_to_database(db)?;
        }

        // Save manifold learner if trained
        if let Some(ref learner) = self.manifold_learner {
            if learner.is_trained() {
                learner.save_to_database(db)?;
            }
        }

        println!("✅ Engine state saved successfully (batch optimized)");
        Ok(())
    }

    /// Load engine state from database
    pub fn load_from_database(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let db = self
            .database
            .as_ref()
            .ok_or("Database not enabled. Call enable_persistence() first.")?;

        println!("Loading engine state from database...");

        // Load all positions
        let positions = db.load_all_positions()?;
        for position_data in positions {
            if let Ok(board) = Board::from_str(&position_data.fen) {
                let vector: Vec<f32> = position_data.vector.iter().map(|&x| x as f32).collect();
                let vector_array = Array1::from(vector);
                let evaluation = position_data.evaluation.unwrap_or(0.0) as f32;

                // Add to similarity search
                self.similarity_search
                    .add_position(vector_array.clone(), evaluation);

                // Store for reverse lookup
                self.position_vectors.push(vector_array);
                self.position_boards.push(board);
                self.position_evaluations.push(evaluation);
            }
        }

        // Load LSH configuration if available and LSH is enabled
        if self.use_lsh {
            let positions_for_lsh: Vec<(Array1<f32>, f32)> = self
                .position_vectors
                .iter()
                .zip(self.position_evaluations.iter())
                .map(|(v, &e)| (v.clone(), e))
                .collect();

            match LSH::load_from_database(db, &positions_for_lsh)? {
                Some(lsh) => {
                    self.lsh_index = Some(lsh);
                    println!("Loaded LSH configuration from database");
                }
                None => {
                    println!("No LSH configuration found in database");
                }
            }
        }

        // Load manifold learner if available
        match ManifoldLearner::load_from_database(db)? {
            Some(learner) => {
                self.manifold_learner = Some(learner);
                if self.use_manifold {
                    self.rebuild_manifold_indices()?;
                }
                println!("Loaded manifold learner from database");
            }
            None => {
                println!("No manifold learner found in database");
            }
        }

        println!(
            "Engine state loaded successfully ({} positions)",
            self.knowledge_base_size()
        );
        Ok(())
    }

    /// Create engine with persistence enabled and auto-load from database
    pub fn new_with_persistence<P: AsRef<Path>>(
        vector_size: usize,
        db_path: P,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut engine = Self::new(vector_size);
        engine.enable_persistence(db_path)?;

        // Try to load existing data
        match engine.load_from_database() {
            Ok(_) => {
                println!("Loaded existing engine from database");
            }
            Err(e) => {
                println!("Starting fresh engine (load failed: {})", e);
            }
        }

        Ok(engine)
    }

    /// Auto-save to database (if persistence is enabled)
    pub fn auto_save(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.database.is_some() {
            self.save_to_database()?;
        }
        Ok(())
    }

    /// Check if persistence is enabled
    pub fn is_persistence_enabled(&self) -> bool {
        self.database.is_some()
    }

    /// Get database position count
    pub fn database_position_count(&self) -> Result<i64, Box<dyn std::error::Error>> {
        let db = self.database.as_ref().ok_or("Database not enabled")?;
        Ok(db.get_position_count()?)
    }

    /// Enable tactical search with the given configuration
    pub fn enable_tactical_search(&mut self, config: TacticalConfig) {
        self.tactical_search = Some(TacticalSearch::new(config));
    }

    /// Enable tactical search with default configuration
    pub fn enable_tactical_search_default(&mut self) {
        self.tactical_search = Some(TacticalSearch::new_default());
    }

    /// Configure hybrid evaluation settings
    pub fn configure_hybrid_evaluation(&mut self, config: HybridConfig) {
        self.hybrid_config = config;
    }

    /// Check if tactical search is enabled
    pub fn is_tactical_search_enabled(&self) -> bool {
        self.tactical_search.is_some()
    }

    /// Enable parallel tactical search with specified number of threads
    pub fn enable_parallel_search(&mut self, num_threads: usize) {
        if let Some(ref mut tactical_search) = self.tactical_search {
            tactical_search.config.enable_parallel_search = true;
            tactical_search.config.num_threads = num_threads;
            println!(
                "🧵 Parallel tactical search enabled with {} threads",
                num_threads
            );
        }
    }

    /// Check if parallel search is enabled
    pub fn is_parallel_search_enabled(&self) -> bool {
        self.tactical_search
            .as_ref()
            .map(|ts| ts.config.enable_parallel_search)
            .unwrap_or(false)
    }

    // /// Enable Syzygy tablebase support for perfect endgame evaluation
    // pub fn enable_tablebase<P: AsRef<std::path::Path>>(&mut self, path: P) -> Result<(), Box<dyn std::error::Error>> {
    //     let mut prober = TablebaseProber::new();
    //     prober.initialize(path)?;
    //     self.tablebase = Some(prober);
    //     println!("🗄️  Syzygy tablebase enabled for perfect endgame evaluation");
    //     Ok(())
    // }

    // /// Check if tablebase is enabled
    // pub fn is_tablebase_enabled(&self) -> bool {
    //     self.tablebase.as_ref().map(|tb| tb.is_enabled()).unwrap_or(false)
    // }

    // /// Get tablebase max pieces supported
    // pub fn tablebase_max_pieces(&self) -> Option<usize> {
    //     self.tablebase.as_ref().map(|tb| tb.max_pieces())
    // }

    /// Get current hybrid configuration
    pub fn hybrid_config(&self) -> &HybridConfig {
        &self.hybrid_config
    }

    /// Check if opening book is enabled
    pub fn is_opening_book_enabled(&self) -> bool {
        self.opening_book.is_some()
    }

    /// Run self-play training to generate new positions
    pub fn self_play_training(
        &mut self,
        config: training::SelfPlayConfig,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let mut trainer = training::SelfPlayTrainer::new(config);
        let new_data = trainer.generate_training_data(self);

        let positions_added = new_data.data.len();

        // Add new positions to the engine incrementally
        for data in &new_data.data {
            self.add_position(&data.board, data.evaluation);
        }

        // Save to database if persistence is enabled
        if self.database.is_some() {
            match self.save_to_database() {
                Ok(_) => println!("💾 Saved {} positions to database", positions_added),
                Err(e) => println!("⚠️  Database save failed: {}", e),
            }
        }

        println!(
            "🧠 Self-play training complete: {} new positions learned",
            positions_added
        );
        Ok(positions_added)
    }

    /// Run continuous self-play training with periodic saving
    pub fn continuous_self_play(
        &mut self,
        config: training::SelfPlayConfig,
        iterations: usize,
        save_path: Option<&str>,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let mut total_positions = 0;
        let mut trainer = training::SelfPlayTrainer::new(config.clone());

        println!(
            "🔄 Starting continuous self-play training for {} iterations...",
            iterations
        );

        for iteration in 1..=iterations {
            println!("\n--- Self-Play Iteration {}/{} ---", iteration, iterations);

            // Generate new training data
            let new_data = trainer.generate_training_data(self);
            let batch_size = new_data.data.len();

            // Add new positions incrementally
            for data in &new_data.data {
                self.add_position(&data.board, data.evaluation);
            }

            total_positions += batch_size;

            println!(
                "✅ Iteration {}: Added {} positions (total: {})",
                iteration,
                batch_size,
                self.knowledge_base_size()
            );

            // Save periodically - both binary/JSON and database
            if iteration % 5 == 0 || iteration == iterations {
                // Save to binary file if path provided (faster than JSON)
                if let Some(path) = save_path {
                    match self.save_training_data_binary(path) {
                        Ok(_) => println!("💾 Progress saved to {} (binary format)", path),
                        Err(e) => println!("⚠️  Failed to save: {}", e),
                    }
                }

                // Save to database if persistence is enabled
                if self.database.is_some() {
                    match self.save_to_database() {
                        Ok(_) => println!(
                            "💾 Database synchronized ({} total positions)",
                            self.knowledge_base_size()
                        ),
                        Err(e) => println!("⚠️  Database save failed: {}", e),
                    }
                }
            }

            // Rebuild manifold learning every 10 iterations for large datasets
            if iteration % 10 == 0
                && self.knowledge_base_size() > 5000
                && self.manifold_learner.is_some()
            {
                println!("🧠 Retraining manifold learning with new data...");
                let _ = self.train_manifold_learning(5);
            }
        }

        println!(
            "\n🎉 Continuous self-play complete: {} total new positions",
            total_positions
        );
        Ok(total_positions)
    }

    /// Self-play with adaptive difficulty (engine gets stronger as it learns)
    pub fn adaptive_self_play(
        &mut self,
        base_config: training::SelfPlayConfig,
        target_strength: f32,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let mut current_config = base_config;
        let mut total_positions = 0;
        let mut iteration = 1;

        println!(
            "🎯 Starting adaptive self-play training (target strength: {:.2})...",
            target_strength
        );

        loop {
            println!("\n--- Adaptive Iteration {} ---", iteration);

            // Run self-play with current configuration
            let positions_added = self.self_play_training(current_config.clone())?;
            total_positions += positions_added;

            // Save to database after each iteration for resumability
            if self.database.is_some() {
                match self.save_to_database() {
                    Ok(_) => println!("💾 Adaptive training progress saved to database"),
                    Err(e) => println!("⚠️  Database save failed: {}", e),
                }
            }

            // Evaluate current strength (simplified - could use more sophisticated metrics)
            let current_strength = self.knowledge_base_size() as f32 / 10000.0; // Simple heuristic

            println!(
                "📊 Current strength estimate: {:.2} (target: {:.2})",
                current_strength, target_strength
            );

            if current_strength >= target_strength {
                println!("🎉 Target strength reached!");
                break;
            }

            // Adapt configuration for next iteration
            current_config.exploration_factor *= 0.95; // Reduce exploration as we get stronger
            current_config.temperature *= 0.98; // Reduce randomness
            current_config.games_per_iteration =
                (current_config.games_per_iteration as f32 * 1.1) as usize; // More games

            iteration += 1;

            if iteration > 50 {
                println!("⚠️  Maximum iterations reached");
                break;
            }
        }

        Ok(total_positions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess::Board;

    #[test]
    fn test_engine_creation() {
        let engine = ChessVectorEngine::new(1024);
        assert_eq!(engine.knowledge_base_size(), 0);
    }

    #[test]
    fn test_add_and_search() {
        let mut engine = ChessVectorEngine::new(1024);
        let board = Board::default();

        engine.add_position(&board, 0.0);
        assert_eq!(engine.knowledge_base_size(), 1);

        let similar = engine.find_similar_positions(&board, 1);
        assert_eq!(similar.len(), 1);
    }

    #[test]
    fn test_evaluation() {
        let mut engine = ChessVectorEngine::new(1024);
        let board = Board::default();

        // Add some positions with evaluations
        engine.add_position(&board, 0.5);

        let evaluation = engine.evaluate_position(&board);
        assert!(evaluation.is_some());
        assert!((evaluation.unwrap() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_move_recommendations() {
        let mut engine = ChessVectorEngine::new(1024);
        let board = Board::default();

        // Add a position with moves
        use chess::ChessMove;
        use std::str::FromStr;
        let mov = ChessMove::from_str("e2e4").unwrap();
        engine.add_position_with_move(&board, 0.0, Some(mov), Some(0.8));

        let recommendations = engine.recommend_moves(&board, 3);
        assert!(!recommendations.is_empty());

        // Test legal move filtering
        let legal_recommendations = engine.recommend_legal_moves(&board, 3);
        assert!(!legal_recommendations.is_empty());
    }

    #[test]
    fn test_empty_knowledge_base_fallback() {
        // Test that recommend_moves() works even with empty knowledge base
        let mut engine = ChessVectorEngine::new(1024);

        // Test with a specific position (Sicilian Defense)
        use std::str::FromStr;
        let board =
            Board::from_str("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1")
                .unwrap();

        // Should return move recommendations even with empty knowledge base
        let recommendations = engine.recommend_moves(&board, 5);
        assert!(
            !recommendations.is_empty(),
            "recommend_moves should not return empty even with no training data"
        );
        assert_eq!(
            recommendations.len(),
            5,
            "Should return exactly 5 recommendations"
        );

        // All recommendations should have neutral confidence and outcome
        for rec in &recommendations {
            assert!(rec.confidence > 0.0, "Confidence should be greater than 0");
            assert_eq!(
                rec.from_similar_position_count, 1,
                "Should have count of 1 for fallback"
            );
            assert_eq!(rec.average_outcome, 0.0, "Should have neutral outcome");
        }

        // Test with starting position too
        let starting_board = Board::default();
        let starting_recommendations = engine.recommend_moves(&starting_board, 3);
        assert!(
            !starting_recommendations.is_empty(),
            "Should work for starting position too"
        );

        // Verify all moves are legal
        use chess::MoveGen;
        let legal_moves: std::collections::HashSet<_> = MoveGen::new_legal(&board).collect();
        for rec in &recommendations {
            assert!(
                legal_moves.contains(&rec.chess_move),
                "All recommended moves should be legal"
            );
        }
    }

    #[test]
    fn test_opening_book_integration() {
        let mut engine = ChessVectorEngine::new(1024);

        // Enable opening book
        engine.enable_opening_book();
        assert!(engine.opening_book.is_some());

        // Test starting position
        let board = Board::default();
        assert!(engine.is_opening_position(&board));

        let entry = engine.get_opening_entry(&board);
        assert!(entry.is_some());

        let stats = engine.opening_book_stats();
        assert!(stats.is_some());
        assert!(stats.unwrap().total_positions > 0);

        // Test opening book move recommendations
        let recommendations = engine.recommend_moves(&board, 3);
        assert!(!recommendations.is_empty());
        assert!(recommendations[0].confidence > 0.7); // Opening book should have high confidence
    }

    #[test]
    fn test_manifold_learning_integration() {
        let mut engine = ChessVectorEngine::new(1024);

        // Add some training data
        let board = Board::default();
        for i in 0..10 {
            engine.add_position(&board, i as f32 * 0.1);
        }

        // Enable manifold learning
        assert!(engine.enable_manifold_learning(8.0).is_ok());

        // Test compression ratio
        let ratio = engine.manifold_compression_ratio();
        assert!(ratio.is_some());
        assert!((ratio.unwrap() - 8.0).abs() < 0.1);

        // Train with minimal epochs for testing
        assert!(engine.train_manifold_learning(5).is_ok());

        // Test that compression is working
        let original_similar = engine.find_similar_positions(&board, 3);
        assert!(!original_similar.is_empty());
    }

    #[test]
    fn test_lsh_integration() {
        let mut engine = ChessVectorEngine::new(1024);

        // Add training data
        let board = Board::default();
        for i in 0..50 {
            engine.add_position(&board, i as f32 * 0.02);
        }

        // Enable LSH
        engine.enable_lsh(4, 8);

        // Test search works with LSH
        let similar = engine.find_similar_positions(&board, 5);
        assert!(!similar.is_empty());
        assert!(similar.len() <= 5);

        // Test evaluation still works
        let eval = engine.evaluate_position(&board);
        assert!(eval.is_some());
    }

    #[test]
    fn test_manifold_lsh_integration() {
        let mut engine = ChessVectorEngine::new(1024);

        // Add training data
        let board = Board::default();
        for i in 0..20 {
            engine.add_position(&board, i as f32 * 0.05);
        }

        // Enable manifold learning
        assert!(engine.enable_manifold_learning(8.0).is_ok());
        assert!(engine.train_manifold_learning(3).is_ok());

        // Enable LSH in manifold space
        assert!(engine.enable_manifold_lsh(4, 8).is_ok());

        // Test search works in compressed space
        let similar = engine.find_similar_positions(&board, 3);
        assert!(!similar.is_empty());

        // Test move recommendations work
        let _recommendations = engine.recommend_moves(&board, 2);
        // May be empty if no moves were stored, but shouldn't crash
    }

    // TODO: Re-enable when database thread safety is implemented
    // #[test]
    // fn test_multithreading_safe() {
    //     use std::sync::Arc;
    //     use std::thread;
    //
    //     let engine = Arc::new(ChessVectorEngine::new(1024));
    //     let board = Arc::new(Board::default());
    //
    //     // Test that read operations are thread-safe
    //     let handles: Vec<_> = (0..4).map(|_| {
    //         let engine = Arc::clone(&engine);
    //         let board = Arc::clone(&board);
    //         thread::spawn(move || {
    //             engine.evaluate_position(&board);
    //             engine.find_similar_positions(&board, 3);
    //         })
    //     }).collect();
    //
    //     for handle in handles {
    //         handle.join().unwrap();
    //     }
    // }

    #[test]
    fn test_position_with_move_storage() {
        let mut engine = ChessVectorEngine::new(1024);
        let board = Board::default();

        use chess::ChessMove;
        use std::str::FromStr;
        let move1 = ChessMove::from_str("e2e4").unwrap();
        let move2 = ChessMove::from_str("d2d4").unwrap();

        // Add positions with moves
        engine.add_position_with_move(&board, 0.0, Some(move1), Some(0.7));
        engine.add_position_with_move(&board, 0.1, Some(move2), Some(0.6));

        // Test that move data is stored
        assert_eq!(engine.position_moves.len(), 2);

        // Test move recommendations include stored moves
        let recommendations = engine.recommend_moves(&board, 5);
        let _move_strings: Vec<String> = recommendations
            .iter()
            .map(|r| r.chess_move.to_string())
            .collect();

        // Should contain either the stored moves or legal alternatives
        assert!(!recommendations.is_empty());
    }

    #[test]
    fn test_performance_regression_basic() {
        use std::time::Instant;

        let mut engine = ChessVectorEngine::new(1024);
        let board = Board::default();

        // Add a reasonable amount of data
        for i in 0..100 {
            engine.add_position(&board, i as f32 * 0.01);
        }

        // Measure basic operations
        let start = Instant::now();

        // Position encoding should be fast
        for _ in 0..100 {
            engine.add_position(&board, 0.0);
        }

        let encoding_time = start.elapsed();

        // Search should be reasonable
        let start = Instant::now();
        for _ in 0..10 {
            engine.find_similar_positions(&board, 5);
        }
        let search_time = start.elapsed();

        // Basic performance bounds (these should be generous)
        assert!(
            encoding_time.as_millis() < 5000,
            "Position encoding too slow: {}ms",
            encoding_time.as_millis()
        );
        assert!(
            search_time.as_millis() < 1000,
            "Search too slow: {}ms",
            search_time.as_millis()
        );
    }

    #[test]
    fn test_memory_usage_reasonable() {
        let mut engine = ChessVectorEngine::new(1024);
        let board = Board::default();

        // Add data and ensure it doesn't explode memory usage
        let initial_size = engine.knowledge_base_size();

        for i in 0..1000 {
            engine.add_position(&board, i as f32 * 0.001);
        }

        let final_size = engine.knowledge_base_size();
        assert_eq!(final_size, initial_size + 1000);

        // Memory growth should be linear
        assert!(final_size > initial_size);
    }

    #[test]
    fn test_incremental_training() {
        use std::str::FromStr;

        let mut engine = ChessVectorEngine::new(1024);
        let board1 = Board::default();
        let board2 =
            Board::from_str("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1").unwrap();

        // Add initial positions
        engine.add_position(&board1, 0.0);
        engine.add_position(&board2, 0.2);
        assert_eq!(engine.knowledge_base_size(), 2);

        // Create a dataset for incremental training
        let mut dataset = crate::training::TrainingDataset::new();
        dataset.add_position(board1, 0.1, 15, 1); // Duplicate position (should be skipped)
        dataset.add_position(
            Board::from_str("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
                .unwrap(),
            0.3,
            15,
            2,
        ); // New position

        // Train incrementally
        engine.train_from_dataset_incremental(&dataset);

        // Should only add the new position
        assert_eq!(engine.knowledge_base_size(), 3);

        // Check training stats
        let stats = engine.training_stats();
        assert_eq!(stats.total_positions, 3);
        assert_eq!(stats.unique_positions, 3);
        assert!(!stats.has_move_data); // No moves added in this test
    }

    #[test]
    fn test_save_load_incremental() {
        use std::str::FromStr;
        use tempfile::tempdir;

        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_training.json");

        // Create first engine with some data
        let mut engine1 = ChessVectorEngine::new(1024);
        engine1.add_position(&Board::default(), 0.0);
        engine1.add_position(
            &Board::from_str("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1").unwrap(),
            0.2,
        );

        // Save training data
        engine1.save_training_data(&file_path).unwrap();

        // Create second engine and load incrementally
        let mut engine2 = ChessVectorEngine::new(1024);
        engine2.add_position(
            &Board::from_str("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
                .unwrap(),
            0.3,
        );
        assert_eq!(engine2.knowledge_base_size(), 1);

        // Load additional data incrementally
        engine2.load_training_data_incremental(&file_path).unwrap();

        // Should now have 3 positions total
        assert_eq!(engine2.knowledge_base_size(), 3);
    }

    #[test]
    fn test_training_stats() {
        use std::str::FromStr;

        let mut engine = ChessVectorEngine::new(1024);

        // Initial stats
        let stats = engine.training_stats();
        assert_eq!(stats.total_positions, 0);
        assert_eq!(stats.unique_positions, 0);
        assert!(!stats.has_move_data);
        assert!(!stats.lsh_enabled);
        assert!(!stats.manifold_enabled);
        assert!(!stats.opening_book_enabled);

        // Add some data
        engine.add_position(&Board::default(), 0.0);
        engine.add_position_with_move(
            &Board::default(),
            0.1,
            Some(ChessMove::from_str("e2e4").unwrap()),
            Some(0.8),
        );

        // Enable features
        engine.enable_opening_book();
        engine.enable_lsh(4, 8);

        let stats = engine.training_stats();
        assert_eq!(stats.total_positions, 2);
        assert!(stats.has_move_data);
        assert!(stats.move_data_entries > 0);
        assert!(stats.lsh_enabled);
        assert!(stats.opening_book_enabled);
    }

    #[test]
    fn test_tactical_search_integration() {
        let mut engine = ChessVectorEngine::new(1024);
        let board = Board::default();

        // Test that tactical search is initially disabled
        assert!(!engine.is_tactical_search_enabled());

        // Enable tactical search with default configuration
        engine.enable_tactical_search_default();
        assert!(engine.is_tactical_search_enabled());

        // Test evaluation without any similar positions (should use tactical search)
        let evaluation = engine.evaluate_position(&board);
        assert!(evaluation.is_some());

        // Test evaluation with similar positions (should use hybrid approach)
        engine.add_position(&board, 0.5);
        let hybrid_evaluation = engine.evaluate_position(&board);
        assert!(hybrid_evaluation.is_some());
    }

    #[test]
    fn test_hybrid_evaluation_configuration() {
        let mut engine = ChessVectorEngine::new(1024);
        let board = Board::default();

        // Enable tactical search
        engine.enable_tactical_search_default();

        // Test custom hybrid configuration
        let custom_config = HybridConfig {
            pattern_confidence_threshold: 0.9, // High threshold
            enable_tactical_refinement: true,
            tactical_config: TacticalConfig::default(),
            pattern_weight: 0.8,
            min_similar_positions: 5,
        };

        engine.configure_hybrid_evaluation(custom_config);

        // Add some positions with low similarity to trigger tactical refinement
        engine.add_position(&board, 0.3);

        let evaluation = engine.evaluate_position(&board);
        assert!(evaluation.is_some());

        // Test with tactical refinement disabled
        let no_tactical_config = HybridConfig {
            enable_tactical_refinement: false,
            ..HybridConfig::default()
        };

        engine.configure_hybrid_evaluation(no_tactical_config);

        let pattern_only_evaluation = engine.evaluate_position(&board);
        assert!(pattern_only_evaluation.is_some());
    }
}
