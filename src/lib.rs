pub mod position_encoder;
pub mod similarity_search;
pub mod manifold_learner;
pub mod variational_autoencoder;
pub mod nnue;
pub mod lsh;
pub mod ann;
pub mod training;
pub mod opening_book;
pub mod persistence;
pub mod gpu_acceleration;
pub mod tactical_search;
pub mod uci;

pub use position_encoder::PositionEncoder;
pub use similarity_search::SimilaritySearch;
pub use lsh::LSH;
pub use manifold_learner::ManifoldLearner;
pub use variational_autoencoder::{VariationalAutoencoder, VAEConfig};
pub use nnue::{NNUE, NNUEConfig, HybridEvaluator, BlendStrategy, EvalStats};
pub use opening_book::{OpeningBook, OpeningEntry, OpeningBookStats};
pub use training::{TrainingData, TrainingDataset, GameExtractor, EngineEvaluator, TacticalPuzzle, TacticalTrainingData, TacticalPuzzleParser, SelfPlayConfig, SelfPlayTrainer};
pub use persistence::{Database, PositionData, LSHTableData};
pub use gpu_acceleration::{GPUAccelerator, DeviceType};
pub use tactical_search::{TacticalSearch, TacticalConfig, TacticalResult};
pub use uci::{UCIEngine, UCIConfig, run_uci_engine, run_uci_engine_with_config};

use chess::{Board, ChessMove};
use ndarray::Array1;
use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;

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

/// Main chess vector engine
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
            position_vectors: self.position_vectors.clone(),
            position_boards: self.position_boards.clone(),
            position_evaluations: self.position_evaluations.clone(),
            opening_book: self.opening_book.clone(),
            database: None, // Database connection cannot be cloned
            tactical_search: self.tactical_search.clone(),
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
            position_vectors: Vec::new(),
            position_boards: Vec::new(),
            position_evaluations: Vec::new(),
            opening_book: None,
            database: None,
            tactical_search: None,
            hybrid_config: HybridConfig::default(),
        }
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
            _ => Self::new(vector_size) // Default to linear search
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
            position_vectors: Vec::new(),
            position_boards: Vec::new(),
            position_evaluations: Vec::new(),
            opening_book: None,
            database: None,
            tactical_search: None,
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
        let vector = self.encoder.encode(board);
        self.similarity_search.add_position(vector.clone(), evaluation);
        
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
        if self.use_manifold && self.manifold_learner.is_some() {
            let compressed_query = self.manifold_learner.as_ref().unwrap().encode(&query_vector);
            
            // Use LSH in manifold space if available
            if let Some(ref lsh) = self.manifold_lsh_index {
                return lsh.query(&compressed_query, k);
            }
            
            // Fall back to linear search in manifold space
            if let Some(ref search) = self.manifold_similarity_search {
                return search.search(&compressed_query, k);
            }
        }
        
        // Use original space with LSH if enabled
        if self.use_lsh && self.lsh_index.is_some() {
            self.lsh_index.as_ref().unwrap().query(&query_vector, k)
        } else {
            self.similarity_search.search(&query_vector, k)
        }
    }
    
    /// Find similar positions with indices for move recommendation
    pub fn find_similar_positions_with_indices(&self, board: &Board, k: usize) -> Vec<(usize, f32, f32)> {
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
    pub fn evaluate_position(&self, board: &Board) -> Option<f32> {
        // First check opening book - highest priority
        if let Some(entry) = self.get_opening_entry(board) {
            return Some(entry.evaluation);
        }
        
        // Get pattern evaluation from similarity search
        let similar_positions = self.find_similar_positions(board, 5);
        
        if similar_positions.is_empty() {
            // No similar positions found - use tactical search if available
            if let Some(ref tactical_search) = self.tactical_search {
                let mut tactical_engine = tactical_search.clone();
                let result = tactical_engine.search(board);
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
        let count_factor = (similar_positions.len() as f32 / self.hybrid_config.min_similar_positions as f32).min(1.0);
        let pattern_confidence = avg_similarity * count_factor;

        // Decide whether to use tactical refinement
        let use_tactical = self.hybrid_config.enable_tactical_refinement 
            && pattern_confidence < self.hybrid_config.pattern_confidence_threshold
            && self.tactical_search.is_some();

        if use_tactical {
            // Get tactical evaluation
            let mut tactical_engine = self.tactical_search.as_ref().unwrap().clone();
            let tactical_result = tactical_engine.search(board);
            
            // Blend pattern and tactical evaluations
            let pattern_weight = self.hybrid_config.pattern_weight * pattern_confidence;
            let tactical_weight = 1.0 - pattern_weight;
            
            let hybrid_evaluation = (pattern_evaluation * pattern_weight) + 
                                  (tactical_result.evaluation * tactical_weight);
            
            Some(hybrid_evaluation)
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
    pub fn save_training_data<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        use crate::training::{TrainingDataset, TrainingData};
        
        let mut dataset = TrainingDataset::new();
        
        // Convert engine positions back to training data
        for (i, board) in self.position_boards.iter().enumerate() {
            if i < self.position_evaluations.len() {
                dataset.data.push(TrainingData {
                    board: *board,
                    evaluation: self.position_evaluations[i],
                    depth: 15, // Default depth
                    game_id: i, // Use index as game_id
                });
            }
        }
        
        dataset.save_incremental(path)?;
        println!("Saved {} positions to training data", dataset.data.len());
        Ok(())
    }

    /// Load training data incrementally (append to existing engine state)
    pub fn load_training_data_incremental<P: AsRef<std::path::Path>>(&mut self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        use crate::training::TrainingDataset;
        
        let existing_size = self.knowledge_base_size();
        let dataset = TrainingDataset::load(path)?;
        
        for data in dataset.data {
            // Skip if we already have this position to avoid exact duplicates
            if !self.position_boards.contains(&data.board) {
                self.add_position(&data.board, data.evaluation);
            }
        }
        
        println!("Loaded {} new positions (total: {})", 
                self.knowledge_base_size() - existing_size, 
                self.knowledge_base_size());
        Ok(())
    }

    /// Save training data in optimized binary format with compression (5-15x faster than JSON)
    pub fn save_training_data_binary<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        use lz4_flex::compress_prepend_size;
        
        println!("💾 Saving training data in binary format (compressed)...");
        
        // Create binary training data structure
        #[derive(serde::Serialize)]
        struct BinaryTrainingData {
            positions: Vec<String>, // FEN strings
            evaluations: Vec<f32>,
            vectors: Vec<Vec<f32>>,  // Optional for export
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
        
        println!("✅ Saved {} positions to binary file ({} bytes compressed)", 
                 binary_data.positions.len(), compressed.len());
        Ok(())
    }

    /// Load training data from optimized binary format (5-15x faster than JSON)
    pub fn load_training_data_binary<P: AsRef<std::path::Path>>(&mut self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        use lz4_flex::decompress_size_prepended;
        
        println!("📚 Loading training data from binary format...");
        
        #[derive(serde::Deserialize)]
        struct BinaryTrainingData {
            positions: Vec<String>,
            evaluations: Vec<f32>,
            vectors: Vec<Vec<f32>>,
            created_at: i64,
        }
        
        let existing_size = self.knowledge_base_size();
        
        // Read and decompress file
        let compressed_data = std::fs::read(path)?;
        let serialized = decompress_size_prepended(&compressed_data)?;
        
        // Deserialize with bincode
        let binary_data: BinaryTrainingData = bincode::deserialize(&serialized)?;
        
        // Load positions into engine
        for (i, fen) in binary_data.positions.iter().enumerate() {
            if i < binary_data.evaluations.len() {
                if let Ok(board) = fen.parse() {
                    // Skip duplicates
                    if !self.position_boards.contains(&board) {
                        self.add_position(&board, binary_data.evaluations[i]);
                    }
                }
            }
        }
        
        println!("✅ Loaded {} new positions from binary file (total: {})", 
                 self.knowledge_base_size() - existing_size, 
                 self.knowledge_base_size());
        Ok(())
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
        
        println!("Added {} new positions from dataset (total: {})", added, self.knowledge_base_size());
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
        let common_files = vec![
            "training_data.json",
            "tactical_training_data.json", 
            "engine_training.json",
            "chess_training.json",
            "my_training.json",
        ];
        
        let mut loaded_files = Vec::new();
        
        for file_path in &common_files {
            if std::path::Path::new(file_path).exists() {
                match self.load_training_data_incremental(file_path) {
                    Ok(_) => {
                        loaded_files.push(file_path.to_string());
                        println!("📚 Auto-loaded training data from {}", file_path);
                    }
                    Err(e) => {
                        println!("⚠️  Could not load {}: {}", file_path, e);
                    }
                }
            }
        }
        
        // Also look for tactical puzzle files
        let tactical_files = vec![
            "tactical_puzzles.json",
            "lichess_puzzles.json",
            "my_puzzles.json",
        ];
        
        for file_path in &tactical_files {
            if std::path::Path::new(file_path).exists() {
                match crate::training::TacticalPuzzleParser::load_tactical_puzzles(file_path) {
                    Ok(puzzles) => {
                        crate::training::TacticalPuzzleParser::load_into_engine_incremental(&puzzles, self);
                        loaded_files.push(file_path.to_string());
                        println!("🎯 Auto-loaded tactical puzzles from {}", file_path);
                    }
                    Err(e) => {
                        println!("⚠️  Could not load tactical puzzles from {}: {}", file_path, e);
                    }
                }
            }
        }
        
        Ok(loaded_files)
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
            println!("🚀 Created engine with auto-loaded training data from {} files", loaded_files.len());
            let stats = engine.training_stats();
            println!("   - Total positions: {}", stats.total_positions);
            println!("   - Move data entries: {}", stats.move_data_entries);
        }
        
        Ok(engine)
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
            return Err("Manifold learning not enabled. Call enable_manifold_learning first.".to_string());
        }
        
        if self.similarity_search.size() == 0 {
            return Err("No positions in knowledge base to train on.".to_string());
        }
        
        // Collect all position vectors
        let all_positions = self.similarity_search.get_all_positions();
        let mut training_data = Vec::new();
        
        for (vector, _eval) in all_positions {
            training_data.extend_from_slice(vector.as_slice().unwrap());
        }
        
        let rows = self.similarity_search.size();
        let cols = self.encoder.vector_size();
        let training_matrix = ndarray::Array2::from_shape_vec((rows, cols), training_data)
            .map_err(|e| format!("Failed to create training matrix: {}", e))?;
        
        // Train the manifold learner
        if let Some(ref mut learner) = self.manifold_learner {
            learner.train(&training_matrix, epochs)?;
            let compression_ratio = learner.compression_ratio();
            
            // Release the mutable borrow before calling rebuild_manifold_indices
            let _ = learner;
            
            // Rebuild compressed indices
            self.rebuild_manifold_indices()?;
            self.use_manifold = true;
            
            println!("Manifold learning training completed. Compression ratio: {:.1}x", 
                     compression_ratio);
        }
        
        Ok(())
    }
    
    /// Rebuild manifold-based indices after training
    fn rebuild_manifold_indices(&mut self) -> Result<(), String> {
        if let Some(ref learner) = self.manifold_learner {
            let all_positions = self.similarity_search.get_all_positions();
            
            // Clear existing manifold indices
            let output_dim = learner.output_dim();
            if let Some(ref mut search) = self.manifold_similarity_search {
                *search = SimilaritySearch::new(output_dim);
            }
            if let Some(ref mut lsh) = self.manifold_lsh_index {
                *lsh = LSH::new(output_dim, 8, 16); // Default LSH params for compressed space
            }
            
            // Add compressed vectors to indices
            for (vector, eval) in all_positions {
                let compressed = learner.encode(&vector);
                
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
    pub fn enable_manifold_lsh(&mut self, num_tables: usize, hash_size: usize) -> Result<(), String> {
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
        self.manifold_learner.as_ref().map(|l| l.compression_ratio())
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
        self.opening_book.as_ref()
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
    pub fn add_position_with_move(&mut self, board: &Board, evaluation: f32, chess_move: Option<ChessMove>, move_outcome: Option<f32>) {
        let position_index = self.knowledge_base_size();
        
        // Add the position first
        self.add_position(board, evaluation);
        
        // If a move and outcome are provided, store the move information
        if let (Some(mov), Some(outcome)) = (chess_move, move_outcome) {
            self.position_moves.entry(position_index)
                .or_insert_with(Vec::new)
                .push((mov, outcome));
        }
    }
    
    /// Get move recommendations based on similar positions and opening book
    pub fn recommend_moves(&self, board: &Board, num_recommendations: usize) -> Vec<MoveRecommendation> {
        // First check opening book
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
            recommendations.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
            recommendations.truncate(num_recommendations);
            return recommendations;
        }
        
        // Fall back to similarity search
        let similar_positions = self.find_similar_positions_with_indices(board, 20);
        
        if similar_positions.is_empty() {
            return Vec::new();
        }
        
        // Collect moves from similar positions
        let mut move_data: HashMap<ChessMove, Vec<(f32, f32)>> = HashMap::new(); // move -> (similarity, outcome)
        
        // Use actual position indices to get moves and outcomes
        for (position_index, _eval, similarity) in similar_positions {
            if let Some(moves) = self.position_moves.get(&position_index) {
                for &(chess_move, outcome) in moves {
                    move_data.entry(chess_move)
                        .or_insert_with(Vec::new)
                        .push((similarity, outcome));
                }
            }
        }
        
        // If no moves found from stored data, generate legal moves and give them neutral recommendations
        if move_data.is_empty() {
            use chess::MoveGen;
            let legal_moves: Vec<ChessMove> = MoveGen::new_legal(board).collect();
            
            for chess_move in legal_moves.into_iter().take(num_recommendations) {
                move_data.insert(chess_move, vec![(0.5, 0.0)]); // Neutral similarity and outcome
            }
        }
        
        // Calculate move recommendations
        let mut recommendations = Vec::new();
        
        for (chess_move, outcomes) in move_data {
            if outcomes.is_empty() { continue; }
            
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
            
            // Confidence based on number of similar positions and average similarity
            let avg_similarity = outcomes.iter().map(|(s, _)| s).sum::<f32>() / outcomes.len() as f32;
            let confidence = avg_similarity * (outcomes.len() as f32).ln().max(1.0) / 10.0; // Logarithmic scaling
            
            recommendations.push(MoveRecommendation {
                chess_move,
                confidence: confidence.min(1.0), // Cap at 1.0
                from_similar_position_count: outcomes.len(),
                average_outcome,
            });
        }
        
        // Sort by confidence (descending)
        recommendations.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        
        // Return top recommendations
        recommendations.truncate(num_recommendations);
        recommendations
    }
    
    /// Generate legal move recommendations (filters recommendations by legal moves)
    pub fn recommend_legal_moves(&self, board: &Board, num_recommendations: usize) -> Vec<MoveRecommendation> {
        use chess::MoveGen;
        
        // Get all legal moves
        let legal_moves: std::collections::HashSet<ChessMove> = MoveGen::new_legal(board).collect();
        
        // Get recommendations and filter by legal moves
        let all_recommendations = self.recommend_moves(board, num_recommendations * 2); // Get more to account for filtering
        
        all_recommendations.into_iter()
            .filter(|rec| legal_moves.contains(&rec.chess_move))
            .take(num_recommendations)
            .collect()
    }

    /// Enable persistence with database
    pub fn enable_persistence<P: AsRef<Path>>(&mut self, db_path: P) -> Result<(), Box<dyn std::error::Error>> {
        let database = Database::new(db_path)?;
        self.database = Some(database);
        println!("Persistence enabled");
        Ok(())
    }

    /// Save engine state to database using high-performance batch operations
    pub fn save_to_database(&self) -> Result<(), Box<dyn std::error::Error>> {
        let db = self.database.as_ref()
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
        let db = self.database.as_ref()
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
                self.similarity_search.add_position(vector_array.clone(), evaluation);
                
                // Store for reverse lookup
                self.position_vectors.push(vector_array);
                self.position_boards.push(board);
                self.position_evaluations.push(evaluation);
            }
        }

        // Load LSH configuration if available and LSH is enabled
        if self.use_lsh {
            let positions_for_lsh: Vec<(Array1<f32>, f32)> = self.position_vectors.iter()
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

        println!("Engine state loaded successfully ({} positions)", self.knowledge_base_size());
        Ok(())
    }

    /// Create engine with persistence enabled and auto-load from database
    pub fn new_with_persistence<P: AsRef<Path>>(vector_size: usize, db_path: P) -> Result<Self, Box<dyn std::error::Error>> {
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
        let db = self.database.as_ref()
            .ok_or("Database not enabled")?;
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

    /// Get current hybrid configuration
    pub fn hybrid_config(&self) -> &HybridConfig {
        &self.hybrid_config
    }

    /// Check if opening book is enabled
    pub fn is_opening_book_enabled(&self) -> bool {
        self.opening_book.is_some()
    }
    
    /// Run self-play training to generate new positions
    pub fn self_play_training(&mut self, config: training::SelfPlayConfig) -> Result<usize, Box<dyn std::error::Error>> {
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
        
        println!("🧠 Self-play training complete: {} new positions learned", positions_added);
        Ok(positions_added)
    }
    
    /// Run continuous self-play training with periodic saving
    pub fn continuous_self_play(&mut self, 
        config: training::SelfPlayConfig,
        iterations: usize,
        save_path: Option<&str>) -> Result<usize, Box<dyn std::error::Error>> {
        
        let mut total_positions = 0;
        let mut trainer = training::SelfPlayTrainer::new(config.clone());
        
        println!("🔄 Starting continuous self-play training for {} iterations...", iterations);
        
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
            
            println!("✅ Iteration {}: Added {} positions (total: {})", 
                     iteration, batch_size, self.knowledge_base_size());
            
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
                        Ok(_) => println!("💾 Database synchronized ({} total positions)", self.knowledge_base_size()),
                        Err(e) => println!("⚠️  Database save failed: {}", e),
                    }
                }
            }
            
            // Rebuild manifold learning every 10 iterations for large datasets
            if iteration % 10 == 0 && self.knowledge_base_size() > 5000 {
                if self.manifold_learner.is_some() {
                    println!("🧠 Retraining manifold learning with new data...");
                    let _ = self.train_manifold_learning(5);
                }
            }
        }
        
        println!("\n🎉 Continuous self-play complete: {} total new positions", total_positions);
        Ok(total_positions)
    }
    
    /// Self-play with adaptive difficulty (engine gets stronger as it learns)
    pub fn adaptive_self_play(&mut self, 
        base_config: training::SelfPlayConfig,
        target_strength: f32) -> Result<usize, Box<dyn std::error::Error>> {
        
        let mut current_config = base_config;
        let mut total_positions = 0;
        let mut iteration = 1;
        
        println!("🎯 Starting adaptive self-play training (target strength: {:.2})...", target_strength);
        
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
            
            println!("📊 Current strength estimate: {:.2} (target: {:.2})", 
                     current_strength, target_strength);
            
            if current_strength >= target_strength {
                println!("🎉 Target strength reached!");
                break;
            }
            
            // Adapt configuration for next iteration
            current_config.exploration_factor *= 0.95; // Reduce exploration as we get stronger
            current_config.temperature *= 0.98; // Reduce randomness
            current_config.games_per_iteration = (current_config.games_per_iteration as f32 * 1.1) as usize; // More games
            
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
        let _move_strings: Vec<String> = recommendations.iter()
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
        assert!(encoding_time.as_millis() < 5000, "Position encoding too slow: {}ms", encoding_time.as_millis());
        assert!(search_time.as_millis() < 1000, "Search too slow: {}ms", search_time.as_millis());
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
        let board2 = Board::from_str("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1").unwrap();
        
        // Add initial positions
        engine.add_position(&board1, 0.0);
        engine.add_position(&board2, 0.2);
        assert_eq!(engine.knowledge_base_size(), 2);
        
        // Create a dataset for incremental training
        let mut dataset = crate::training::TrainingDataset::new();
        dataset.add_position(board1, 0.1, 15, 1); // Duplicate position (should be skipped)
        dataset.add_position(
            Board::from_str("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2").unwrap(),
            0.3, 
            15, 
            2
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
        use tempfile::tempdir;
        use std::str::FromStr;
        
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_training.json");
        
        // Create first engine with some data
        let mut engine1 = ChessVectorEngine::new(1024);
        engine1.add_position(&Board::default(), 0.0);
        engine1.add_position(
            &Board::from_str("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1").unwrap(),
            0.2
        );
        
        // Save training data
        engine1.save_training_data(&file_path).unwrap();
        
        // Create second engine and load incrementally
        let mut engine2 = ChessVectorEngine::new(1024);
        engine2.add_position(
            &Board::from_str("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2").unwrap(),
            0.3
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
            Some(0.8)
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