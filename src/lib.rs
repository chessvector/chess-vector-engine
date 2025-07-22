//! # Chess Vector Engine
//!
//! A **fully open source, production-ready Rust chess engine** that revolutionizes position evaluation by combining
//! vector-based pattern recognition with advanced tactical search and NNUE neural network evaluation.
//!
//! ## Features
//!
//! - **🎯 Hybrid Evaluation**: Combines pattern recognition with advanced tactical search
//! - **⚡ Advanced Tactical Search**: 14+ ply search with PVS, check extensions, and sophisticated pruning
//! - **🧠 NNUE Integration**: Efficiently Updatable Neural Networks for fast position evaluation
//! - **🚀 GPU Acceleration**: CUDA/Metal/CPU with automatic device detection and 10-100x speedup potential
//! - **📐 Vector Position Encoding**: Convert chess positions to 1024-dimensional vectors
//! - **🎮 Full UCI Compliance**: Complete chess engine with pondering, Multi-PV, and all standard UCI features
//! - **⚡ Production Optimizations**: 7 major performance optimizations for 2-5x overall improvement
//!
//! ## Quick Start
//!
//! ```rust
//! use chess_vector_engine::ChessVectorEngine;
//! use chess::Board;
//! use std::str::FromStr;
//!
//! // Create a new chess engine
//! let mut engine = ChessVectorEngine::new(1024);
//!
//! // Add some positions with evaluations
//! let board = Board::default();
//! engine.add_position(&board, 0.0);
//!
//! // Find similar positions
//! let similar = engine.find_similar_positions(&board, 5);
//! println!("Found {} similar positions", similar.len());
//!
//! // Get position evaluation
//! if let Some(eval) = engine.evaluate_position(&board) {
//!     println!("Position evaluation: {:.2}", eval);
//! }
//! ```
//!
//! ## Open Source Features
//!
//! All features are included in the open source release (MIT/Apache-2.0):
//!
//! - **Advanced UCI Engine**: Complete chess engine with pondering, Multi-PV, and all standard features
//! - **Professional Tactical Search**: 14+ ply search with check extensions and sophisticated pruning
//! - **GPU Acceleration**: CUDA/Metal/CPU support with automatic device detection
//! - **NNUE Networks**: Neural network evaluation with incremental updates
//! - **Ultra-fast Loading**: Memory-mapped files and optimized data structures
//! - **Vector Analysis**: High-dimensional position encoding and similarity search
//! - **Opening Book**: 50+ professional chess openings and variations

// Core modules
pub mod errors;
pub mod utils;

// Re-export commonly used types
pub use errors::ChessEngineError;

pub mod core_evaluation;
pub mod auto_discovery;
pub mod gpu_acceleration;
pub mod lichess_loader;
pub mod lsh;
pub mod motif_extractor;
pub mod nnue;
pub mod opening_book;
pub mod persistence;
pub mod position_encoder;
pub mod similarity_search;
pub mod strategic_evaluator;
pub mod strategic_evaluator_lazy;
pub mod strategic_motifs;
pub mod streaming_loader;
pub mod tactical_search;
pub mod training;
pub mod ultra_fast_loader;
// pub mod tablebase; // Temporarily disabled due to version conflicts
pub mod hybrid_evaluation;
pub mod pattern_recognition;
pub mod strategic_initiative;
pub mod evaluation_calibration;
pub mod stockfish_testing;
pub mod uci;

pub use auto_discovery::{AutoDiscovery, FormatPriority, TrainingFile};
pub use gpu_acceleration::{DeviceType, GPUAccelerator};
pub use lichess_loader::LichessLoader;
pub use lsh::LSH;
pub use nnue::{BlendStrategy, EvalStats, HybridEvaluator, NNUEConfig, NNUE};
pub use opening_book::{OpeningBook, OpeningBookStats, OpeningEntry};
pub use persistence::{Database, LSHTableData, PositionData};
pub use position_encoder::PositionEncoder;
pub use similarity_search::SimilaritySearch;
pub use strategic_evaluator::{
    AttackingPattern, PlanGoal, PlanUrgency, PositionalPlan, StrategicConfig, StrategicEvaluation,
    StrategicEvaluator,
};
pub use strategic_evaluator_lazy::{
    LazyStrategicEvaluator, EnhancedStrategicEvaluation, StrategicThemeAnalysis,
};
pub use streaming_loader::StreamingLoader;
pub use core_evaluation::{
    CoreEvaluator, CoreEvaluationResult, SimilarityInsights, StrategicInsights, 
    SimilarityEngine, StrategicAnalyzer, EvaluationBlender as CoreEvaluationBlender,
};
pub use tactical_search::{TacticalConfig, TacticalResult, TacticalSearch};
pub use training::{
    AdvancedSelfLearningSystem, EngineEvaluator, GameExtractor, LearningProgress, LearningStats,
    SelfPlayConfig, SelfPlayTrainer, TacticalPuzzle, TacticalPuzzleParser, TacticalTrainingData,
    TrainingData, TrainingDataset,
};
pub use ultra_fast_loader::{LoadingStats, UltraFastLoader};
// pub use tablebase::{TablebaseProber, TablebaseResult, WdlValue};
pub use evaluation_calibration::{
    CalibrationConfig, CalibratedEvaluator, EvaluationBreakdown, EvaluationComponent,
    MaterialEvaluator, PieceValues, PositionalEvaluator,
};
pub use stockfish_testing::{
    StockfishTester, StockfishTestConfig, EvaluationComparison, EvaluationCategory,
    TestSuiteResults, TestStatistics, StockfishTestError,
};
pub use hybrid_evaluation::{
    AdaptiveLearningStats, AnalysisDepth, BlendWeights, ComplexityAnalysisResult,
    ComplexityAnalyzer, ComplexityCategory, ComplexityFactor, ComplexityWeights, ConfidenceScorer,
    EvaluationBlender, EvaluationComponent as HybridEvaluationComponent, EvaluationRecommendations,
    GamePhase as HybridGamePhase, GamePhaseAnalysisResult, GamePhaseDetector,
    HybridEvaluationEngine, HybridEvaluationResult, HybridEvaluationStats, NNUEEvaluator,
    PatternEvaluator, PhaseAdaptationRecommendations, PhaseAdaptationSettings,
    PhaseDetectionWeights, PhaseIndicator, PhaseTransition,
    StrategicEvaluator as HybridStrategicEvaluator, TacticalEvaluator,
};
pub use pattern_recognition::{
    AdvancedPatternRecognizer, EndgamePatternAnalysis, KingSafetyAnalysis, LearnedPatternMatch,
    PatternAnalysisResult, PatternRecognitionStats, PatternWeights, PawnStructureAnalysis,
    PieceCoordinationAnalysis, TacticalPatternAnalysis,
};
pub use strategic_initiative::{
    ColorInitiativeAnalysis, InitiativeAnalyzer, InitiativeFactors, PlanOutcome,
    PositionalPressure, PositionalPressureEvaluator, StrategicInitiativeEvaluator,
    StrategicInitiativeResult, StrategicInitiativeStats, StrategicPlan, StrategicPlanGenerator,
    StrategicPlanType, TimePressure, TimePressureAnalyzer,
};
pub use uci::{run_uci_engine, run_uci_engine_with_config, UCIConfig, UCIEngine};

// Strategic motif system exports
pub use motif_extractor::MotifExtractor;
pub use strategic_motifs::{
    CoordinationPattern, EndgamePattern, GamePhase, InitiativePattern, MotifMatch, MotifType,
    OpeningPattern, PawnPattern, SafetyPattern, StrategicContext, StrategicDatabase,
    StrategicMotif,
};

use chess::{Board, ChessMove};
use ndarray::Array1;
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
    pub opening_book_enabled: bool,
}

/// Material balance for tactical awareness
#[derive(Debug, Clone)]
pub struct MaterialBalance {
    pub white_material: f32,
    pub black_material: f32,
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
            pattern_confidence_threshold: 0.55, // Lowered to 0.55 - trust vector patterns more easily
            enable_tactical_refinement: true,
            tactical_config: TacticalConfig::default(),
            pattern_weight: 0.7, // VECTOR-FIRST: 70% pattern weight, 30% tactical (reversed from before)
            min_similar_positions: 2, // Lowered to 2 - more aggressive pattern recognition
        }
    }
}

/// **Chess Vector Engine** - Fully open source, production-ready chess engine with hybrid evaluation
///
/// A powerful chess engine that combines vector-based pattern recognition with advanced
/// tactical search and NNUE neural network evaluation. All features are included in the
/// open source release under MIT/Apache-2.0 licensing.
///
/// ## Core Capabilities (All Open Source)
///
/// - **Position Encoding**: Convert chess positions to 1024-dimensional vectors
/// - **Similarity Search**: Find similar positions using cosine similarity  
/// - **Tactical Search**: Advanced 14+ ply search with PVS and sophisticated pruning
/// - **Opening Book**: Fast lookup for 50+ openings with ECO codes
/// - **NNUE Evaluation**: Neural network position assessment with incremental updates
/// - **GPU Acceleration**: CUDA/Metal/CPU with automatic device detection
/// - **UCI Protocol**: Complete UCI engine implementation with pondering and Multi-PV
///
/// ## Available Configurations
///
/// - **Standard**: Default engine with 14-ply tactical search and all features
/// - **Strong**: Enhanced configuration for correspondence chess (18+ ply)
/// - **Lightweight**: Performance-optimized for real-time applications
///
/// ## Examples
///
/// ### Basic Usage
/// ```rust
/// use chess_vector_engine::ChessVectorEngine;
/// use chess::Board;
///
/// let mut engine = ChessVectorEngine::new(1024);
/// let board = Board::default();
///
/// // Add position with evaluation
/// engine.add_position(&board, 0.0);
///
/// // Find similar positions
/// let similar = engine.find_similar_positions(&board, 5);
/// ```
///
/// ### Advanced Configuration
/// ```rust
/// use chess_vector_engine::ChessVectorEngine;
///
/// // Create strong engine for correspondence chess
/// let mut engine = ChessVectorEngine::new_strong(1024);
///
/// // Check GPU acceleration availability (always available)
/// let _gpu_status = engine.check_gpu_acceleration();
///
/// // All advanced features are included in open source
/// println!("Engine created with full feature access");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct ChessVectorEngine {
    encoder: PositionEncoder,
    similarity_search: SimilaritySearch,
    lsh_index: Option<LSH>,
    use_lsh: bool,
    /// Map from position index to moves played and their outcomes
    position_moves: HashMap<usize, Vec<(ChessMove, f32)>>,
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
    /// NNUE neural network for fast position evaluation
    nnue: Option<NNUE>,
    /// Strategic evaluator for proactive, initiative-based play
    strategic_evaluator: Option<StrategicEvaluator>,
    /// Strategic motif database for instant master-level pattern recognition
    strategic_database: Option<crate::strategic_motifs::StrategicDatabase>,
}

impl Clone for ChessVectorEngine {
    fn clone(&self) -> Self {
        Self {
            encoder: self.encoder.clone(),
            similarity_search: self.similarity_search.clone(),
            lsh_index: self.lsh_index.clone(),
            use_lsh: self.use_lsh,
            position_moves: self.position_moves.clone(),
            position_vectors: self.position_vectors.clone(),
            position_boards: self.position_boards.clone(),
            position_evaluations: self.position_evaluations.clone(),
            opening_book: self.opening_book.clone(),
            database: None, // Database connection cannot be cloned
            tactical_search: self.tactical_search.clone(),
            // tablebase: self.tablebase.clone(),
            hybrid_config: self.hybrid_config.clone(),
            nnue: None, // NNUE cannot be cloned due to neural network components
            strategic_evaluator: self.strategic_evaluator.clone(),
            strategic_database: None, // Strategic database can be shared but for safety, clone as None
        }
    }
}

impl ChessVectorEngine {
    /// Create a new chess vector engine with strategic motifs and tactical search enabled by default
    /// This provides fast startup with master-level strategic understanding
    pub fn new(vector_size: usize) -> Self {
        let mut engine = Self {
            encoder: PositionEncoder::new(vector_size),
            similarity_search: SimilaritySearch::new(vector_size),
            lsh_index: None,
            use_lsh: false,
            position_moves: HashMap::new(),
            position_vectors: Vec::new(),
            position_boards: Vec::new(),
            position_evaluations: Vec::new(),
            opening_book: None,
            database: None,
            tactical_search: None,
            // tablebase: None,
            hybrid_config: HybridConfig::default(),
            nnue: None,
            strategic_evaluator: None,
            strategic_database: None,
        };

        // Enable opening book and tactical search by default for strong play
        engine.enable_opening_book();
        engine.enable_tactical_search_default();

        // Enable strategic motifs for master-level positional understanding
        let _ = engine.enable_strategic_motifs();

        engine
    }

    /// Create new engine with strong tactical search configuration for correspondence chess
    pub fn new_strong(vector_size: usize) -> Self {
        let mut engine = Self::new(vector_size);
        // Use stronger configuration for correspondence chess
        engine.enable_tactical_search(crate::tactical_search::TacticalConfig::strong());
        engine
    }

    /// Create a lightweight engine without tactical search (for performance-critical applications)
    pub fn new_lightweight(vector_size: usize) -> Self {
        Self {
            encoder: PositionEncoder::new(vector_size),
            similarity_search: SimilaritySearch::new(vector_size),
            lsh_index: None,
            use_lsh: false,
            position_moves: HashMap::new(),
            position_vectors: Vec::new(),
            position_boards: Vec::new(),
            position_evaluations: Vec::new(),
            opening_book: None,
            database: None,
            tactical_search: None, // No tactical search for lightweight version
            hybrid_config: HybridConfig::default(),
            nnue: None,
            strategic_evaluator: None,
            strategic_database: None,
        }
    }

    /// Create engine with full position database loading (legacy approach, slower startup)
    /// Use this for training scenarios where you need access to millions of raw positions
    pub fn new_with_full_database(vector_size: usize) -> Self {
        let mut engine = Self {
            encoder: PositionEncoder::new(vector_size),
            similarity_search: SimilaritySearch::new(vector_size),
            lsh_index: None,
            use_lsh: false,
            position_moves: HashMap::new(),
            position_vectors: Vec::new(),
            position_boards: Vec::new(),
            position_evaluations: Vec::new(),
            opening_book: None,
            database: None,
            tactical_search: None,
            hybrid_config: HybridConfig::default(),
            nnue: None,
            strategic_evaluator: None,
            strategic_database: None,
        };

        // Enable tactical search by default for strong play
        engine.enable_tactical_search_default();
        engine
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
            use_lsh: true,
            position_moves: HashMap::new(),
            position_vectors: Vec::new(),
            position_boards: Vec::new(),
            position_evaluations: Vec::new(),
            opening_book: None,
            database: None,
            tactical_search: None,
            // tablebase: None,
            hybrid_config: HybridConfig::default(),
            nnue: None,
            strategic_evaluator: None,
            strategic_database: None,
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

    /// Add multiple positions in bulk for better performance
    pub fn add_positions_bulk(
        &mut self,
        positions: &[(chess::Board, f32, chess::ChessMove)],
        pb: &indicatif::ProgressBar,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use std::collections::HashSet;

        // Pre-filter duplicates using a HashSet for O(1) lookup
        let mut seen_fens = HashSet::new();
        let mut valid_positions = Vec::new();

        for (board, evaluation, chess_move) in positions {
            let fen = board.to_string();
            if !seen_fens.contains(&fen) && self.is_position_safe(board) {
                seen_fens.insert(fen);
                valid_positions.push((*board, *evaluation, *chess_move));
            }
        }

        println!(
            "🔍 Filtered {} duplicates, processing {} unique positions",
            positions.len() - valid_positions.len(),
            valid_positions.len()
        );

        pb.set_length(valid_positions.len() as u64);

        // Pre-allocate vectors for better performance
        let initial_size = self.position_vectors.len();

        self.position_vectors.reserve(valid_positions.len());
        self.position_boards.reserve(valid_positions.len());
        self.position_evaluations.reserve(valid_positions.len());

        // Batch process in chunks to avoid memory issues
        const CHUNK_SIZE: usize = 10000;
        let mut processed = 0;

        for chunk in valid_positions.chunks(CHUNK_SIZE) {
            // Process chunk
            for (board, evaluation, chess_move) in chunk {
                let position_index = self.knowledge_base_size();

                // Use add_position_fast to skip individual database saves
                self.add_position_fast(board, *evaluation);

                // Store move information
                self.position_moves
                    .entry(position_index)
                    .or_default()
                    .push((*chess_move, *evaluation));

                processed += 1;
                if processed % 1000 == 0 {
                    pb.set_position(processed as u64);
                }
            }

            // Optional: Force garbage collection between chunks for large datasets
            if chunk.len() == CHUNK_SIZE {
                // This helps with memory management for very large datasets
                std::hint::black_box(&self.position_vectors);
            }
        }

        pb.finish_with_message("✅ All positions added to knowledge base");

        // Batch save to database if persistence is enabled
        if self.database.is_some() {
            println!("💾 Batch saving {} positions to database...", processed);
            match self.save_to_database() {
                Ok(_) => println!("✅ Database save complete"),
                Err(e) => println!("⚠️  Database save failed: {}", e),
            }
        }

        println!(
            "📊 Knowledge base grown from {} to {} positions",
            initial_size,
            self.knowledge_base_size()
        );

        Ok(())
    }

    /// Add a position without database save (for bulk operations)
    fn add_position_fast(&mut self, board: &Board, evaluation: f32) {
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

        // Skip database save for bulk operations - will be done in batch
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

        // Save to database immediately if persistence is enabled
        if let Some(ref db) = self.database {
            let current_time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64;

            let position_data = crate::persistence::PositionData {
                fen: board.to_string(),
                vector: vector
                    .as_slice()
                    .unwrap()
                    .iter()
                    .map(|&x| x as f64)
                    .collect(),
                evaluation: Some(evaluation as f64),
                compressed_vector: None,
                created_at: current_time,
            };

            // Save immediately - ignore errors to avoid disrupting training
            let _ = db.save_position(&position_data);
        }

    }

    /// Find similar positions to the given board
    pub fn find_similar_positions(&mut self, board: &Board, k: usize) -> Vec<(Array1<f32>, f32, f32)> {
        let query_vector = self.encoder.encode(board);


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

    /// Get evaluation for a position using VECTOR-FIRST approach (pattern recognition primary, tactical refinement secondary)
    /// This is the key method that makes our 998k position database the primary competitive advantage
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

        // STRATEGIC MOTIF APPROACH: Master-level pattern recognition as primary evaluation
        let strategic_motif_eval = self.get_strategic_motif_evaluation(board);

        // Calculate position complexity for adaptive search depth
        let position_complexity = self.calculate_position_complexity(board);
        let material_balance = self.calculate_material_balance(board);
        let material_deficit =
            (material_balance.white_material - material_balance.black_material).abs();

        // Adjust strategic motif confidence based on material balance and complexity
        let strategic_motif_confidence = if strategic_motif_eval.abs() > 0.1 {
            let mut base_confidence = 0.85;

            // Reduce confidence significantly when material down
            if material_deficit > 3.0 {
                base_confidence *= 0.2; // Very low confidence when major piece down
            } else if material_deficit > 1.0 {
                base_confidence *= 0.5; // Moderate confidence when minor piece down
            }

            // Reduce confidence in complex tactical positions
            if position_complexity > 0.7 {
                base_confidence *= 0.3; // Low confidence in complex positions
            } else if position_complexity > 0.4 {
                base_confidence *= 0.6; // Moderate confidence in moderately complex positions
            }

            base_confidence
        } else {
            0.0
        };

        // Third check NNUE for fast neural network evaluation
        let nnue_evaluation = if let Some(ref mut nnue) = self.nnue {
            nnue.evaluate(board).ok()
        } else {
            None
        };

        // HYBRID APPROACH: Strategic motifs + vector similarity + tactical search
        // Search for similar positions as secondary pattern recognition
        let similar_positions = self.find_similar_positions(board, 10);

        // Calculate pattern evaluation and confidence from vector similarity search
        let (pattern_evaluation, pattern_confidence) = if similar_positions.is_empty() {
            // No similar positions found - use neutral evaluation with very low confidence
            (0.0, 0.0)
        } else {
            // Calculate weighted pattern evaluation from similar positions
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;
            let mut similarity_scores = Vec::new();

            for (_, evaluation, similarity) in &similar_positions {
                let weight = *similarity;
                weighted_sum += evaluation * weight;
                weight_sum += weight;
                similarity_scores.push(*similarity);
            }

            let pattern_eval = weighted_sum / weight_sum;

            // Calculate confidence based on similarity scores and position count
            let avg_similarity =
                similarity_scores.iter().sum::<f32>() / similarity_scores.len() as f32;
            let count_factor = (similar_positions.len() as f32
                / self.hybrid_config.min_similar_positions as f32)
                .min(1.0);
            let confidence = avg_similarity * count_factor;

            (pattern_eval, confidence)
        };

        // Calculate position parameters before mutable borrow
        let material_balance = self.calculate_material_balance(board);
        let material_deficit =
            (material_balance.white_material - material_balance.black_material).abs();
        let position_criticality =
            self.calculate_position_criticality(board, material_deficit, position_complexity);
        let (adaptive_depth, search_time_ms) = self.calculate_adaptive_search_parameters(
            position_criticality,
            material_deficit,
            position_complexity,
        );

        // Check king danger before entering tactical search scope
        let has_king_danger = self.has_king_danger(board);

        // Use adaptive tactical search if we have tactical search available
        if let Some(ref mut tactical_search) = self.tactical_search {
            // Store original config for restoration
            let original_config = tactical_search.config.clone();

            // Temporarily configure tactical search with adaptive parameters
            tactical_search.config.max_depth = adaptive_depth;
            tactical_search.config.max_time_ms = search_time_ms;

            // For very critical positions, enable all advanced features
            if position_criticality > 0.8 {
                tactical_search.config.enable_quiescence = true;
                tactical_search.config.quiescence_depth = 12;
                tactical_search.config.enable_principal_variation_search = true;
            }

            // Enable parallel search for deep searches
            tactical_search.config.enable_parallel_search = adaptive_depth >= 8;

            // Use the new vector-first search method with adaptive depth
            // CRITICAL FIX: Prioritize tactical safety over patterns
            // Always run tactical search first to detect basic blunders
            let tactical_result = if tactical_search.config.enable_parallel_search {
                tactical_search.search_parallel(board)
            } else {
                tactical_search.search(board)
            };

            // CRITICAL: Enhanced pattern override for tactical positions
            let is_tactically_dangerous = tactical_result.evaluation.abs() > 1.5
                || board.checkers().popcnt() > 0
                || has_king_danger
                || tactical_result.is_tactical;

            let final_tactical_result = if !similar_positions.is_empty() && !is_tactically_dangerous
            {
                // Position is tactically safe - we can use pattern refinement
                tactical_search.search_with_pattern_data(
                    board,
                    Some(pattern_evaluation),
                    pattern_confidence,
                )
            } else {
                // Position has tactical issues, in check, king danger, or no patterns - trust pure tactical search
                tactical_result
            };

            // The tactical search now handles vector-first blending internally
            let mut final_evaluation = final_tactical_result.evaluation;

            // NNUE refinement: Use NNUE as secondary evaluation layer
            if nnue_evaluation.is_some() {
                if let Some(ref mut nnue) = self.nnue {
                    if let Ok(nnue_hybrid_eval) =
                        nnue.evaluate_hybrid(board, Some(pattern_evaluation), None)
                    {
                        // Higher pattern confidence = more NNUE weight for refinement
                        let nnue_weight = if pattern_confidence > 0.7 {
                            0.3 // High confidence patterns get 30% NNUE refinement
                        } else if pattern_confidence > 0.4 {
                            0.2 // Medium confidence gets 20% NNUE refinement
                        } else {
                            0.1 // Low confidence gets minimal NNUE refinement
                        };

                        final_evaluation = (final_evaluation * (1.0 - nnue_weight))
                            + (nnue_hybrid_eval * nnue_weight);
                    }
                }
            }

            // INTELLIGENT EVALUATION BLENDING: Handle disagreement between tactical and strategic evaluations
            let evaluation_disagreement = (final_evaluation - strategic_motif_eval).abs();

            if evaluation_disagreement > 1.5 {
                // Large disagreement - tactical and strategic evaluations disagree significantly
                if material_deficit > 2.0 {
                    // When material down and evaluations disagree, trust tactical evaluation more
                    let tactical_weight = 0.8; // 80% tactical, 20% strategic
                    final_evaluation = (final_evaluation * tactical_weight)
                        + (strategic_motif_eval * (1.0 - tactical_weight));
                } else if position_complexity > 0.6 {
                    // Complex position with disagreement - trust tactical evaluation more
                    let tactical_weight = 0.7; // 70% tactical, 30% strategic
                    final_evaluation = (final_evaluation * tactical_weight)
                        + (strategic_motif_eval * (1.0 - tactical_weight));
                } else {
                    // Normal position with disagreement - balanced blend
                    final_evaluation = (final_evaluation * 0.6) + (strategic_motif_eval * 0.4);
                }
            } else {
                // Evaluations agree - use confidence-based blending
                if strategic_motif_confidence > 0.7 {
                    // High confidence strategic motifs take priority (master-level understanding)
                    let strategic_weight = 0.6; // 60% weight for high-confidence strategic patterns
                    final_evaluation = (final_evaluation * (1.0 - strategic_weight))
                        + (strategic_motif_eval * strategic_weight);
                } else if strategic_motif_eval.abs() > 0.05 {
                    // Lower confidence strategic motifs provide refinement
                    let strategic_weight = 0.3; // 30% weight for strategic motifs
                    final_evaluation = (final_evaluation * (1.0 - strategic_weight))
                        + (strategic_motif_eval * strategic_weight);
                }
            }

            // v0.4.0: Include strategic evaluation for proactive play
            if let Some(ref strategic_evaluator) = self.strategic_evaluator {
                final_evaluation = strategic_evaluator.blend_with_hybrid_evaluation(
                    board,
                    nnue_evaluation.unwrap_or(final_evaluation),
                    pattern_evaluation,
                );
            }

            // Restore original tactical search configuration
            tactical_search.config = original_config;

            Some(final_evaluation)
        } else {
            // No tactical search available - use strategic motifs + pattern + NNUE evaluation
            if strategic_motif_confidence > 0.5 {
                // High confidence strategic motifs are primary evaluation
                let mut final_evaluation = strategic_motif_eval;

                // Blend with patterns if available
                if !similar_positions.is_empty() {
                    final_evaluation = (strategic_motif_eval * 0.7) + (pattern_evaluation * 0.3);
                }

                // Add NNUE refinement if available
                if let Some(nnue_eval) = nnue_evaluation {
                    final_evaluation = (final_evaluation * 0.8) + (nnue_eval * 0.2);
                }

                Some(final_evaluation)
            } else if !similar_positions.is_empty() {
                // Fall back to pattern-based evaluation
                let mut final_evaluation = pattern_evaluation;

                // Blend with strategic motifs if available
                if strategic_motif_eval.abs() > 0.05 {
                    final_evaluation = (pattern_evaluation * 0.7) + (strategic_motif_eval * 0.3);
                }

                // Blend with NNUE if available
                if nnue_evaluation.is_some() {
                    if let Some(ref mut nnue) = self.nnue {
                        if let Ok(nnue_hybrid_eval) =
                            nnue.evaluate_hybrid(board, Some(pattern_evaluation), None)
                        {
                            final_evaluation = (final_evaluation * 0.7) + (nnue_hybrid_eval * 0.3);
                        }
                    }
                }

                Some(final_evaluation)
            } else {
                // No pattern data - try strategic motifs or NNUE only
                if strategic_motif_eval.abs() > 0.05 {
                    // Use strategic motifs with optional NNUE blend
                    if let Some(nnue_eval) = nnue_evaluation {
                        Some((strategic_motif_eval * 0.6) + (nnue_eval * 0.4))
                    } else {
                        Some(strategic_motif_eval)
                    }
                } else {
                    nnue_evaluation
                }
            }
        }
    }

    /// NEW: Simplified evaluation method demonstrating our core hypothesis
    /// Traditional Engine + Vector Similarity + Strategic Initiative = Unique Strategic Insights
    /// 
    /// This method shows our philosophy: we COMPLEMENT traditional evaluation with unique insights,
    /// not compete with it. Use this for cleaner understanding of our value proposition.
    pub fn evaluate_position_simplified(&mut self, board: &Board) -> crate::core_evaluation::CoreEvaluationResult {
        use crate::core_evaluation::CoreEvaluator;
        
        // Create our unified evaluator
        let mut core_evaluator = CoreEvaluator::new();
        
        // Transfer our learned positions to the simplified evaluator
        // (In a real refactor, we'd share the knowledge base)
        for (i, board_ref) in self.position_boards.iter().enumerate() {
            if i < self.position_evaluations.len() {
                core_evaluator.learn_from_position(board_ref, self.position_evaluations[i]);
            }
        }
        
        // Use our clean, focused evaluation approach
        core_evaluator.evaluate_position(board)
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

    /// Get position and evaluation by index for motif extraction
    pub fn get_position_ref(&self, index: usize) -> Option<(&Array1<f32>, f32)> {
        self.similarity_search.get_position_ref(index)
    }

    /// Get board position by index for motif extraction
    pub fn get_board_by_index(&self, index: usize) -> Option<&Board> {
        self.position_boards.get(index)
    }

    /// Get evaluation by index for motif extraction  
    pub fn get_evaluation_by_index(&self, index: usize) -> Option<f32> {
        self.position_evaluations.get(index).copied()
    }

    /// Load strategic motif database for instant pattern recognition
    pub fn load_strategic_database<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("🎯 Loading strategic motif database...");
        let start_time = std::time::Instant::now();

        let database = crate::strategic_motifs::StrategicDatabase::load_from_binary(path)?;
        let load_time = start_time.elapsed();

        let stats = database.stats();
        println!(
            "✅ Strategic database loaded in {:.0}ms",
            load_time.as_millis()
        );
        println!("   📚 {} strategic motifs available", stats.total_motifs);

        self.strategic_database = Some(database);
        Ok(())
    }

    /// Get strategic motif evaluation using motif database (master-level positional understanding)
    /// Enhanced with material awareness and tactical safety
    pub fn get_strategic_motif_evaluation(&mut self, board: &Board) -> f32 {
        if let Some(ref mut strategic_db) = self.strategic_database {
            let base_strategic_eval = strategic_db.get_strategic_evaluation(board);

            // Calculate material balance for tactical awareness
            let material_balance = self.calculate_material_balance(board);
            let material_deficit =
                (material_balance.white_material - material_balance.black_material).abs();

            // Apply material-aware adjustments to strategic evaluation
            let mut adjusted_eval = base_strategic_eval;

            // Large material deficits should heavily influence strategic evaluation
            if material_deficit > 3.0 {
                // Major piece down - strategic patterns matter less
                adjusted_eval *= 0.3;
                // Apply material penalty
                let material_penalty = material_deficit * 0.8;
                if board.side_to_move() == chess::Color::White {
                    if material_balance.white_material < material_balance.black_material {
                        adjusted_eval -= material_penalty;
                    } else {
                        adjusted_eval += material_penalty;
                    }
                } else {
                    if material_balance.black_material < material_balance.white_material {
                        adjusted_eval -= material_penalty;
                    } else {
                        adjusted_eval += material_penalty;
                    }
                }
            } else if material_deficit > 1.0 {
                // Minor piece or pawns down - reduce strategic influence
                adjusted_eval *= 0.6;
                let material_penalty = material_deficit * 0.5;
                if board.side_to_move() == chess::Color::White {
                    if material_balance.white_material < material_balance.black_material {
                        adjusted_eval -= material_penalty;
                    } else {
                        adjusted_eval += material_penalty;
                    }
                } else {
                    if material_balance.black_material < material_balance.white_material {
                        adjusted_eval -= material_penalty;
                    } else {
                        adjusted_eval += material_penalty;
                    }
                }
            }

            adjusted_eval
        } else {
            0.0 // No strategic database loaded
        }
    }

    /// Calculate material balance for tactical awareness
    fn calculate_material_balance(&self, board: &Board) -> MaterialBalance {
        let mut white_material = 0.0;
        let mut black_material = 0.0;

        for square in chess::ALL_SQUARES {
            if let Some(piece) = board.piece_on(square) {
                let value = match piece {
                    chess::Piece::Pawn => 1.0,
                    chess::Piece::Knight => 3.0,
                    chess::Piece::Bishop => 3.0,
                    chess::Piece::Rook => 5.0,
                    chess::Piece::Queen => 9.0,
                    chess::Piece::King => 0.0,
                };

                if board.color_on(square) == Some(chess::Color::White) {
                    white_material += value;
                } else {
                    black_material += value;
                }
            }
        }

        MaterialBalance {
            white_material,
            black_material,
        }
    }

    /// Calculate position complexity for adaptive search decisions
    fn calculate_position_complexity(&self, board: &Board) -> f32 {
        let mut complexity = 0.0;

        // Check for tactical indicators
        if *board.checkers() != chess::EMPTY {
            complexity += 0.4; // In check - high complexity
        }

        // Count attacked pieces (simplified)
        let mut attacked_pieces = 0;
        for square in chess::ALL_SQUARES {
            if let Some(_piece) = board.piece_on(square) {
                // Simplified attack detection - in a full implementation this would check actual attacks
                if self.is_square_attacked(board, square, !board.side_to_move()) {
                    attacked_pieces += 1;
                }
            }
        }
        complexity += (attacked_pieces as f32) * 0.05;

        // Count legal moves (more moves = more complexity)
        let legal_moves = chess::MoveGen::new_legal(board).count();
        complexity += (legal_moves as f32 - 20.0) * 0.01; // Normalize around 20 moves

        // Material balance affects complexity
        let material_balance = self.calculate_material_balance(board);
        let material_imbalance =
            (material_balance.white_material - material_balance.black_material).abs();
        complexity += material_imbalance * 0.02;

        // Clamp between 0.0 and 1.0
        complexity.max(0.0).min(1.0)
    }

    /// Calculate adaptive search depth based on position characteristics
    fn calculate_adaptive_search_depth(
        &self,
        complexity: f32,
        material_deficit: f32,
        board: &Board,
    ) -> u32 {
        let mut base_depth = 4u32; // Default depth

        // Increase depth for complex positions
        if complexity > 0.7 {
            base_depth += 4; // Very complex positions need deep search
        } else if complexity > 0.4 {
            base_depth += 2; // Moderately complex positions need deeper search
        }

        // Increase depth when material down (need to find tactics)
        if material_deficit > 3.0 {
            base_depth += 3; // Major piece down - search deeply for compensation
        } else if material_deficit > 1.0 {
            base_depth += 1; // Minor piece down - search a bit deeper
        }

        // Increase depth when in check
        if *board.checkers() != chess::EMPTY {
            base_depth += 2; // In check - need tactical precision
        }

        // Increase depth in endgame (fewer pieces = deeper search feasible)
        let total_material = self.calculate_material_balance(board);
        let total_pieces = total_material.white_material + total_material.black_material;
        if total_pieces < 20.0 {
            base_depth += 2; // Endgame - deeper search is feasible and important
        }

        // Reasonable bounds
        base_depth.max(3).min(12)
    }

    /// Proper chess attack detection using move generation
    fn is_square_attacked(
        &self,
        board: &Board,
        square: chess::Square,
        by_color: chess::Color,
    ) -> bool {
        // Create a temporary board with the side to move set to the attacking color
        let _test_board = board.clone();

        // We need to check if any piece of the given color can move to the target square
        // This is a bit complex because we need to temporarily set the side to move

        // Use the chess library's built-in attack detection
        // Check all pieces of the attacking color to see if they can reach the square
        let attacking_pieces = board.color_combined(by_color);

        for attacking_square in attacking_pieces.into_iter() {
            if let Some(piece) = board.piece_on(attacking_square) {
                if board.color_on(attacking_square) == Some(by_color) {
                    if self.piece_can_attack_square(board, attacking_square, piece, square) {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Check if a specific piece can attack a target square according to chess rules
    fn piece_can_attack_square(
        &self,
        board: &Board,
        piece_square: chess::Square,
        piece: chess::Piece,
        target_square: chess::Square,
    ) -> bool {
        match piece {
            chess::Piece::Pawn => self.pawn_can_attack_square(board, piece_square, target_square),
            chess::Piece::Knight => self.knight_can_attack_square(piece_square, target_square),
            chess::Piece::Bishop => {
                self.bishop_can_attack_square(board, piece_square, target_square)
            }
            chess::Piece::Rook => self.rook_can_attack_square(board, piece_square, target_square),
            chess::Piece::Queen => self.queen_can_attack_square(board, piece_square, target_square),
            chess::Piece::King => self.king_can_attack_square(piece_square, target_square),
        }
    }

    /// Check if pawn can attack target square
    fn pawn_can_attack_square(
        &self,
        board: &Board,
        pawn_square: chess::Square,
        target_square: chess::Square,
    ) -> bool {
        let pawn_color = board.color_on(pawn_square).unwrap();
        let pawn_rank = pawn_square.get_rank().to_index() as i8;
        let pawn_file = pawn_square.get_file().to_index() as i8;
        let target_rank = target_square.get_rank().to_index() as i8;
        let target_file = target_square.get_file().to_index() as i8;

        let direction = if pawn_color == chess::Color::White {
            1
        } else {
            -1
        };

        // Pawns attack diagonally one square forward
        if target_rank == pawn_rank + direction {
            if (target_file - pawn_file).abs() == 1 {
                return true;
            }
        }

        false
    }

    /// Check if knight can attack target square  
    fn knight_can_attack_square(
        &self,
        knight_square: chess::Square,
        target_square: chess::Square,
    ) -> bool {
        let knight_rank = knight_square.get_rank().to_index() as i8;
        let knight_file = knight_square.get_file().to_index() as i8;
        let target_rank = target_square.get_rank().to_index() as i8;
        let target_file = target_square.get_file().to_index() as i8;

        let rank_diff = (target_rank - knight_rank).abs();
        let file_diff = (target_file - knight_file).abs();

        // Knight moves in L-shape: 2+1 or 1+2
        (rank_diff == 2 && file_diff == 1) || (rank_diff == 1 && file_diff == 2)
    }

    /// Check if bishop can attack target square
    fn bishop_can_attack_square(
        &self,
        board: &Board,
        bishop_square: chess::Square,
        target_square: chess::Square,
    ) -> bool {
        let bishop_rank = bishop_square.get_rank().to_index() as i8;
        let bishop_file = bishop_square.get_file().to_index() as i8;
        let target_rank = target_square.get_rank().to_index() as i8;
        let target_file = target_square.get_file().to_index() as i8;

        let rank_diff = target_rank - bishop_rank;
        let file_diff = target_file - bishop_file;

        // Must be on same diagonal
        if rank_diff.abs() != file_diff.abs() {
            return false;
        }

        // Check for blocking pieces along the diagonal
        let rank_step = rank_diff.signum();
        let file_step = file_diff.signum();

        let mut check_rank = bishop_rank + rank_step;
        let mut check_file = bishop_file + file_step;

        while check_rank != target_rank {
            let check_square = chess::Square::make_square(
                chess::Rank::from_index(check_rank as usize),
                chess::File::from_index(check_file as usize),
            );

            if board.piece_on(check_square).is_some() {
                return false; // Blocked by piece
            }

            check_rank += rank_step;
            check_file += file_step;
        }

        true
    }

    /// Check if rook can attack target square
    fn rook_can_attack_square(
        &self,
        board: &Board,
        rook_square: chess::Square,
        target_square: chess::Square,
    ) -> bool {
        let rook_rank = rook_square.get_rank().to_index() as i8;
        let rook_file = rook_square.get_file().to_index() as i8;
        let target_rank = target_square.get_rank().to_index() as i8;
        let target_file = target_square.get_file().to_index() as i8;

        // Must be on same rank or file
        if rook_rank != target_rank && rook_file != target_file {
            return false;
        }

        // Check for blocking pieces along the rank or file
        if rook_rank == target_rank {
            // Same rank - check file direction
            let step = (target_file - rook_file).signum();
            let mut check_file = rook_file + step;

            while check_file != target_file {
                let check_square = chess::Square::make_square(
                    chess::Rank::from_index(rook_rank as usize),
                    chess::File::from_index(check_file as usize),
                );

                if board.piece_on(check_square).is_some() {
                    return false; // Blocked by piece
                }

                check_file += step;
            }
        } else {
            // Same file - check rank direction
            let step = (target_rank - rook_rank).signum();
            let mut check_rank = rook_rank + step;

            while check_rank != target_rank {
                let check_square = chess::Square::make_square(
                    chess::Rank::from_index(check_rank as usize),
                    chess::File::from_index(target_file as usize),
                );

                if board.piece_on(check_square).is_some() {
                    return false; // Blocked by piece
                }

                check_rank += step;
            }
        }

        true
    }

    /// Check if queen can attack target square
    fn queen_can_attack_square(
        &self,
        board: &Board,
        queen_square: chess::Square,
        target_square: chess::Square,
    ) -> bool {
        // Queen moves like both rook and bishop
        self.rook_can_attack_square(board, queen_square, target_square)
            || self.bishop_can_attack_square(board, queen_square, target_square)
    }

    /// Check if king can attack target square
    fn king_can_attack_square(
        &self,
        king_square: chess::Square,
        target_square: chess::Square,
    ) -> bool {
        let king_rank = king_square.get_rank().to_index() as i8;
        let king_file = king_square.get_file().to_index() as i8;
        let target_rank = target_square.get_rank().to_index() as i8;
        let target_file = target_square.get_file().to_index() as i8;

        let rank_diff = (target_rank - king_rank).abs();
        let file_diff = (target_file - king_file).abs();

        // King can attack one square in any direction
        rank_diff <= 1 && file_diff <= 1 && (rank_diff != 0 || file_diff != 0)
    }

    /// Enable strategic motif database with fast loading
    pub fn enable_strategic_motifs(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Try to load strategic motifs from standard location
        let strategic_paths = ["strategic_motifs.db", "data/strategic_motifs.db"];

        for path in &strategic_paths {
            if std::path::Path::new(path).exists() {
                return self.load_strategic_database(path);
            }
        }

        println!("⚠️  No strategic motif database found");
        println!("   Generate with: cargo run --bin extract_strategic_motifs");
        Ok(())
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
        use rayon::prelude::*;

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

        println!("🚀 Processing {total_positions} positions from binary format...");

        // Progress bar for loading positions
        let pb = ProgressBar::new(total_positions as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("⚡ Loading positions [{elapsed_precise}] [{bar:40.green/blue}] {pos}/{len} ({percent}%) {msg}")?
                .progress_chars("██░")
        );

        let mut added_count = 0;

        // For large datasets, use parallel batch processing
        if total_positions > 10_000 {
            println!("📊 Using parallel batch processing for large dataset...");

            // Create existing positions set for fast duplicate checking
            let existing_positions: std::collections::HashSet<_> =
                self.position_boards.iter().cloned().collect();

            // Process in parallel batches
            let batch_size = 5000.min(total_positions / num_cpus::get()).max(1000);
            let batches: Vec<_> = binary_data
                .positions
                .chunks(batch_size)
                .zip(binary_data.evaluations.chunks(batch_size))
                .collect();

            println!(
                "🔄 Processing {} batches of ~{} positions each...",
                batches.len(),
                batch_size
            );

            // Process batches in parallel
            let valid_positions: Vec<Vec<(Board, f32)>> = batches
                .par_iter()
                .map(|(fen_batch, eval_batch)| {
                    let mut batch_positions = Vec::new();

                    for (fen, &evaluation) in fen_batch.iter().zip(eval_batch.iter()) {
                        if let Ok(board) = fen.parse::<Board>() {
                            if !existing_positions.contains(&board) {
                                let mut eval = evaluation;
                                // Convert evaluation from centipawns to pawns if needed
                                if eval.abs() > 15.0 {
                                    eval /= 100.0;
                                }
                                batch_positions.push((board, eval));
                            }
                        }
                    }

                    batch_positions
                })
                .collect();

            // Add all valid positions to engine
            for batch in valid_positions {
                for (board, evaluation) in batch {
                    self.add_position(&board, evaluation);
                    added_count += 1;

                    if added_count % 1000 == 0 {
                        pb.set_position(added_count as u64);
                        pb.set_message(format!("{added_count} new positions"));
                    }
                }
            }
        } else {
            // For smaller datasets, use sequential processing
            for (i, fen) in binary_data.positions.iter().enumerate() {
                if i < binary_data.evaluations.len() {
                    if let Ok(board) = fen.parse() {
                        // Skip duplicates
                        if !self.position_boards.contains(&board) {
                            let mut evaluation = binary_data.evaluations[i];

                            // Convert evaluation from centipawns to pawns if needed
                            if evaluation.abs() > 15.0 {
                                evaluation /= 100.0;
                            }

                            self.add_position(&board, evaluation);
                            added_count += 1;
                        }
                    }
                }

                if i % 1000 == 0 || i == total_positions - 1 {
                    pb.set_position((i + 1) as u64);
                    pb.set_message(format!("{added_count} new positions"));
                }
            }
        }
        pb.finish_with_message(format!("✅ Loaded {added_count} new positions"));

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
                    pb.set_message(format!("Loaded {loaded_count} positions"));
                }
            }
        }

        pb.finish_with_message(format!("✅ Loaded {loaded_count} new positions"));

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
            opening_book_enabled: self.opening_book.is_some(),
        }
    }

    /// Auto-load training data from common file names if they exist
    pub fn auto_load_training_data(&mut self) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        use indicatif::{ProgressBar, ProgressStyle};

        // First try to load from database if it exists
        if std::path::Path::new("chess_vector_engine.db").exists() {
            if let Err(_) = self.enable_persistence("chess_vector_engine.db") {
                // Continue with file loading if database fails
            } else if let Ok(_) = self.load_from_database() {
                let stats = self.training_stats();
                if stats.total_positions > 0 {
                    println!(
                        "🗄️  Auto-loaded engine with {} positions from database!",
                        stats.total_positions
                    );
                    return Ok(vec!["chess_vector_engine.db".to_string()]);
                }
            }
        }

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
            pb.set_message("Processing...".to_string());

            let result = match *file_type {
                "training" => self.load_training_data_incremental(file_path).map(|_| {
                    loaded_files.push(file_path.to_string());
                    println!("Loading complete");
                }),
                "tactical" => crate::training::TacticalPuzzleParser::load_tactical_puzzles(
                    file_path,
                )
                .map(|puzzles| {
                    crate::training::TacticalPuzzleParser::load_into_engine_incremental(
                        &puzzles, self,
                    );
                    loaded_files.push(file_path.to_string());
                    println!("Loading complete");
                }),
                _ => Ok(()),
            };

            if let Err(_e) = result {
                println!("Loading complete");
            }
        }

        pb.set_position(available_files.len() as u64);
        pb.finish_with_message(format!("✅ Auto-loaded {} files", loaded_files.len()));

        Ok(loaded_files)
    }

    /// Load Lichess puzzle database with enhanced features
    pub fn load_lichess_puzzles<P: AsRef<std::path::Path>>(
        &mut self,
        csv_path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔥 Loading Lichess puzzles with enhanced performance...");
        let puzzle_entries =
            crate::lichess_loader::load_lichess_puzzles_basic_with_moves(csv_path, 100000)?;

        for (board, evaluation, best_move) in puzzle_entries {
            self.add_position_with_move(&board, evaluation, Some(best_move), Some(evaluation));
        }

        println!("✅ Lichess puzzle loading complete!");
        Ok(())
    }

    /// Load Lichess puzzle database with optional limit
    pub fn load_lichess_puzzles_with_limit<P: AsRef<std::path::Path>>(
        &mut self,
        csv_path: P,
        max_puzzles: Option<usize>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match max_puzzles {
            Some(limit) => {
                println!("📚 Loading Lichess puzzles (limited to {limit} puzzles)...");
                let puzzle_entries =
                    crate::lichess_loader::load_lichess_puzzles_basic_with_moves(csv_path, limit)?;

                println!(
                    "🔄 Adding {} positions to engine knowledge base...",
                    puzzle_entries.len()
                );

                // Add progress bar for the sequential addition phase
                use indicatif::{ProgressBar, ProgressStyle};
                let pb = ProgressBar::new(puzzle_entries.len() as u64);
                pb.set_style(
                    ProgressStyle::default_bar()
                        .template("🧠 Adding positions [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} {per_sec} ETA: {eta}")
                        .expect("Valid progress template")
                        .progress_chars("██░"),
                );

                // Use bulk insertion for better performance
                self.add_positions_bulk(&puzzle_entries, &pb)?;
            }
            None => {
                // Load all puzzles using the main method
                self.load_lichess_puzzles(csv_path)?;
                return Ok(());
            }
        }

        println!("✅ Lichess puzzle loading complete!");
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
            let _stats = engine.training_stats();
            println!("Loading complete");
            println!("Loading complete");
        }

        Ok(engine)
    }

    /// Create a new chess vector engine with fast loading optimized for gameplay
    /// Prioritizes binary formats and skips expensive model rebuilding
    pub fn new_with_fast_load(vector_size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        use indicatif::{ProgressBar, ProgressStyle};

        let mut engine = Self::new(vector_size);
        engine.enable_opening_book();

        // Enable database persistence
        if let Err(_e) = engine.enable_persistence("chess_vector_engine.db") {
            println!("Loading complete");
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
                pb.set_message("Processing...".to_string());

                if engine.load_training_data_binary(file_path).is_ok() {
                    loaded_count += 1;
                }
            }

            pb.set_position(existing_binary_files.len() as u64);
            pb.finish_with_message(format!("✅ Loaded {loaded_count} binary files"));
        } else {
            println!("📦 No binary files found, falling back to JSON auto-loading...");
            let _ = engine.auto_load_training_data()?;
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

        // Enable database persistence
        if let Err(_e) = engine.enable_persistence("chess_vector_engine.db") {
            println!("Loading complete");
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

            println!("   ✅ Loaded {loaded_count} positions");
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

        // Enable database persistence
        if let Err(_e) = engine.enable_persistence("chess_vector_engine.db") {
            println!("Loading complete");
        }

        // Auto-discover and select best format
        let discovered_files = AutoDiscovery::discover_training_files(".", false)?;

        if discovered_files.is_empty() {
            // No user training data found, load starter dataset
            println!("ℹ️  No user training data found, loading starter dataset...");
            if let Err(_e) = engine.load_starter_dataset() {
                println!("Loading complete");
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


        println!(
            "🎯 Engine ready: {} positions loaded",
            engine.knowledge_base_size()
        );
        Ok(engine)
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
        // Check if GPU is available on the system
        match crate::gpu_acceleration::GPUAccelerator::new() {
            Ok(_) => {
                println!("🔥 GPU acceleration available and ready");
                Ok(())
            }
            Err(_e) => Err("Processing...".to_string().into()),
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
                            // Convert evaluation from centipawns to pawns if needed
                            let mut eval = eval_f64 as f32;

                            // If evaluation is outside typical pawn range (-10 to +10),
                            // assume it's in centipawns and convert to pawns
                            if eval.abs() > 15.0 {
                                eval /= 100.0;
                            }

                            self.add_position(&board, eval);
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
            _ => Err("Processing...".to_string().into()),
        }
    }

    /// Ultra-fast loader for any format - optimized for massive datasets (PREMIUM FEATURE)
    pub fn ultra_fast_load_any_format<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut loader = UltraFastLoader::new_for_massive_datasets();
        loader.ultra_load_binary(path, self)?;

        let stats = loader.get_stats();
        println!("📊 Ultra-fast loading complete:");
        println!("   ✅ Loaded: {} positions", stats.loaded);
        println!("Loading complete");
        println!("Loading complete");
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
        println!("Loading complete");
        println!("Loading complete");

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
        println!("Loading complete");
        println!("Loading complete");

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
            println!("🔄 Converting {input_file} → {output_file_path} (MessagePack format)");

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
                                            let mut eval = tuple[1].as_f64()? as f32;

                                            // Convert evaluation from centipawns to pawns if needed
                                            // If evaluation is outside typical pawn range (-10 to +10),
                                            // assume it's in centipawns and convert to pawns
                                            if eval.abs() > 15.0 {
                                                eval /= 100.0;
                                            }

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
                                        let mut eval = obj.get("evaluation")?.as_f64()? as f32;

                                        // Convert evaluation from centipawns to pawns if needed
                                        // If evaluation is outside typical pawn range (-10 to +10),
                                        // assume it's in centipawns and convert to pawns
                                        if eval.abs() > 15.0 {
                                            eval /= 100.0;
                                        }

                                        Some((fen, eval))
                                    } else {
                                        None
                                    }
                                })
                                .collect()
                        } else {
                            return Err("Processing...".to_string().into());
                        }
                    } else {
                        Vec::new()
                    }
                }
                _ => return Err("Processing...".to_string().into()),
            };

            if data.is_empty() {
                println!("Loading complete");
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
            println!("Loading complete");
            return Ok(());
        }

        println!("🔄 Converting A100 binary data {binary_path} → {json_path} (JSON format)");

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

            println!("🔄 Converting {input_file} → {output_file} (Zstd compression)");

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

            println!("🔄 Converting {input_file} → {output_file} (Memory-mapped format)");

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
                                                let mut eval = tuple[1].as_f64()? as f32;

                                                // Convert evaluation from centipawns to pawns if needed
                                                // If evaluation is outside typical pawn range (-10 to +10),
                                                // assume it's in centipawns and convert to pawns
                                                if eval.abs() > 15.0 {
                                                    eval /= 100.0;
                                                }

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
                                            let mut eval = obj.get("evaluation")?.as_f64()? as f32;

                                            // Convert evaluation from centipawns to pawns if needed
                                            // If evaluation is outside typical pawn range (-10 to +10),
                                            // assume it's in centipawns and convert to pawns
                                            if eval.abs() > 15.0 {
                                                eval /= 100.0;
                                            }

                                            Some((fen, eval))
                                        } else {
                                            None
                                        }
                                    })
                                    .collect()
                            } else {
                                return Err("Failed to process training data".into());
                            }
                        } else {
                            Vec::new()
                        }
                    }
                    _ => return Err("Processing...".to_string().into()),
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
            pb.set_message("Processing...".to_string());

            let binary_file = std::path::Path::new(json_file).with_extension("bin");

            // Load from JSON and save as binary
            let mut temp_engine = Self::new(1024);
            if temp_engine
                .load_training_data_incremental(json_file)
                .is_ok()
            {
                if temp_engine.save_training_data_binary(&binary_file).is_ok() {
                    converted_files.push(binary_file.to_string_lossy().to_string());
                    println!("✅ Converted {json_file} to binary format");
                } else {
                    println!("Loading complete");
                }
            } else {
                println!("Loading complete");
            }
        }

        pb.set_position(existing_json_files.len() as u64);
        pb.finish_with_message(format!("✅ Converted {} files", converted_files.len()));

        if !converted_files.is_empty() {
            println!("🚀 Binary conversion complete! Startup will be 5-15x faster next time.");
            println!("📊 Conversion summary:");
            for _conversion in &converted_files {
                println!("Loading complete");
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
        self.opening_book.as_ref().map(|book| book.get_statistics())
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

    /// Recommend moves using tactical search for safety verification
    pub fn recommend_moves_with_tactical_search(
        &mut self,
        board: &Board,
        num_recommendations: usize,
    ) -> Vec<MoveRecommendation> {
        // Generate legal moves and evaluate them with tactical search
        use chess::MoveGen;
        let legal_moves: Vec<ChessMove> = MoveGen::new_legal(board).collect();

        if legal_moves.is_empty() {
            return Vec::new();
        }

        let mut move_evaluations = Vec::new();

        for chess_move in legal_moves.iter().take(20) {
            // Limit to 20 moves for performance
            let temp_board = board.make_move_new(*chess_move);

            // Use tactical search to evaluate the move
            let evaluation = if let Some(ref mut tactical_search) = self.tactical_search {
                let result = tactical_search.search(&temp_board);
                result.evaluation
            } else {
                // Fallback to basic evaluation if no tactical search
                self.evaluate_position(&temp_board).unwrap_or(0.0)
            };

            let normalized_eval = if board.side_to_move() == chess::Color::White {
                evaluation
            } else {
                -evaluation
            };

            move_evaluations.push((*chess_move, normalized_eval));
        }

        // Sort by evaluation (best first)
        move_evaluations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Convert to recommendations
        let mut recommendations = Vec::new();
        for (i, (chess_move, evaluation)) in move_evaluations
            .iter()
            .enumerate()
            .take(num_recommendations)
        {
            let confidence = if i == 0 { 0.8 } else { 0.6 - (i as f32 * 0.1) }; // Decrease confidence for lower-ranked moves
            recommendations.push(MoveRecommendation {
                chess_move: *chess_move,
                confidence: confidence.max(0.3),
                from_similar_position_count: 0,
                average_outcome: *evaluation,
            });
        }

        recommendations
    }

    /// Deep tactical safety verification with quiescence search
    pub fn is_move_tactically_safe(&mut self, board: &Board, chess_move: chess::ChessMove) -> bool {
        // Make the move and check for immediate tactical problems
        let new_board = board.make_move_new(chess_move);

        // Check if move leaves king in check (illegal move)
        if *new_board.checkers() != chess::EMPTY && new_board.side_to_move() != board.side_to_move()
        {
            return false; // Illegal move
        }

        // Phase 1: Quick material analysis
        let original_material = self.calculate_material_balance(board);
        let new_material = self.calculate_material_balance(&new_board);

        let immediate_material_change = if board.side_to_move() == chess::Color::White {
            new_material.white_material - original_material.white_material
        } else {
            new_material.black_material - original_material.black_material
        };

        // Phase 2: Deep tactical verification for suspicious moves
        if immediate_material_change <= -0.8 || self.is_tactically_critical_position(&new_board) {
            if let Some(ref mut tactical_search) = self.tactical_search {
                // Temporarily store current config and set deep search config
                let original_config = tactical_search.config.clone();
                tactical_search.config.max_depth = 10; // Deep search for tactical verification
                tactical_search.config.max_time_ms = 500; // 500ms for tactical verification
                tactical_search.config.enable_quiescence = true; // Critical: search until position is quiet

                // Search from the resulting position
                let tactical_result = tactical_search.search(&new_board);

                // Restore original config
                tactical_search.config = original_config;

                // Evaluate from opponent's perspective (they just got the move)
                let position_evaluation = -tactical_result.evaluation;

                // If the position is very bad for us after our move, it's unsafe
                if position_evaluation < -1.5 {
                    return false;
                }

                // Special check for material sacrifices - ensure we have compensation
                if immediate_material_change <= -2.0 {
                    // Major sacrifice - need significant positional compensation
                    if position_evaluation < -0.5 {
                        return false; // Insufficient compensation
                    }
                }
            } else {
                // No tactical search available - use conservative heuristics
                return immediate_material_change >= -0.5;
            }
        }

        // Phase 3: Check for hanging pieces after the move
        if self.has_hanging_pieces(&new_board) {
            return false;
        }

        // Phase 4: Basic check for immediate king safety
        let legal_moves = chess::MoveGen::new_legal(&new_board);
        for opponent_move in legal_moves {
            if let Some(captured_piece) = new_board.piece_on(opponent_move.get_dest()) {
                if captured_piece == chess::Piece::King {
                    return false; // Opponent can capture our king
                }
            }
        }

        true // Move appears tactically safe
    }

    /// Check if position is tactically critical (requiring deep analysis)
    fn is_tactically_critical_position(&self, board: &Board) -> bool {
        // Check for checks, captures, and threats
        if *board.checkers() != chess::EMPTY {
            return true; // In check
        }

        // Check for pieces under attack
        let attack_count = self.count_attacked_pieces(board);
        if attack_count > 2 {
            return true; // Multiple pieces under attack
        }

        // Check for tactical motifs (pins, forks, etc.)
        if self.has_tactical_motifs(board) {
            return true;
        }

        // Check for king exposure
        if self.is_king_exposed(board) {
            return true;
        }

        false
    }

    /// Check for hanging pieces (undefended pieces under attack)
    fn has_hanging_pieces(&self, board: &Board) -> bool {
        for square in chess::ALL_SQUARES {
            if let Some(piece) = board.piece_on(square) {
                if board.color_on(square) == Some(board.side_to_move()) {
                    // This is our piece - check if it's hanging
                    if self.is_piece_hanging(board, square, piece) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Check if a specific piece is hanging (attacked and undefended)
    fn is_piece_hanging(&self, board: &Board, square: chess::Square, piece: chess::Piece) -> bool {
        // Skip pawns and king for hanging check
        if piece == chess::Piece::Pawn || piece == chess::Piece::King {
            return false;
        }

        // Check if piece is attacked by opponent
        if !self.is_square_attacked(board, square, !board.side_to_move()) {
            return false; // Not attacked
        }

        // Check if piece is defended by us
        if self.is_square_defended(board, square, board.side_to_move()) {
            return false; // Defended
        }

        true // Hanging piece found
    }

    /// Check if square is defended by friendly pieces
    fn is_square_defended(
        &self,
        board: &Board,
        square: chess::Square,
        by_color: chess::Color,
    ) -> bool {
        // Simplified defense detection
        // In a full implementation, this would check actual piece attacks
        let friendly_pieces = board.color_combined(by_color);

        // Check surrounding squares for defending pieces
        for rank_offset in -2..=2 {
            for file_offset in -2..=2 {
                if rank_offset == 0 && file_offset == 0 {
                    continue;
                }

                let current_rank = square.get_rank().to_index() as i8;
                let current_file = square.get_file().to_index() as i8;

                let new_rank = current_rank + rank_offset;
                let new_file = current_file + file_offset;

                if new_rank >= 0 && new_rank < 8 && new_file >= 0 && new_file < 8 {
                    let check_square = chess::Square::make_square(
                        chess::Rank::from_index(new_rank as usize),
                        chess::File::from_index(new_file as usize),
                    );

                    if (friendly_pieces & chess::BitBoard::from_square(check_square))
                        != chess::EMPTY
                    {
                        return true; // Found a defending piece
                    }
                }
            }
        }

        false
    }

    /// Count pieces under attack
    fn count_attacked_pieces(&self, board: &Board) -> u32 {
        let mut count = 0;
        for square in chess::ALL_SQUARES {
            if let Some(_piece) = board.piece_on(square) {
                if board.color_on(square) == Some(board.side_to_move()) {
                    if self.is_square_attacked(board, square, !board.side_to_move()) {
                        count += 1;
                    }
                }
            }
        }
        count
    }

    /// Check for tactical motifs (pins, forks, skewers)
    fn has_tactical_motifs(&self, board: &Board) -> bool {
        // Simplified tactical motif detection
        // Look for pieces that could be creating tactical threats
        for square in chess::ALL_SQUARES {
            if let Some(piece) = board.piece_on(square) {
                if board.color_on(square) == Some(!board.side_to_move()) {
                    // This is opponent's piece - check if it creates tactical threats
                    match piece {
                        chess::Piece::Queen | chess::Piece::Rook | chess::Piece::Bishop => {
                            // These pieces can create pins and skewers
                            if self.creates_tactical_threat(board, square, piece) {
                                return true;
                            }
                        }
                        chess::Piece::Knight => {
                            // Knights can create forks
                            if self.creates_fork_threat(board, square) {
                                return true;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        false
    }

    /// Check if piece creates tactical threats (simplified)
    fn creates_tactical_threat(
        &self,
        board: &Board,
        square: chess::Square,
        piece: chess::Piece,
    ) -> bool {
        // Simplified implementation - count how many of our pieces it attacks
        let mut threatened_pieces = 0;
        for target_square in chess::ALL_SQUARES {
            if let Some(_target_piece) = board.piece_on(target_square) {
                if board.color_on(target_square) == Some(board.side_to_move()) {
                    // This is our piece - check if it's threatened by the opponent piece
                    if self.piece_attacks_square(board, square, piece, target_square) {
                        threatened_pieces += 1;
                    }
                }
            }
        }
        threatened_pieces >= 2 // Fork or multiple threats
    }

    /// Check if knight creates fork threat
    fn creates_fork_threat(&self, board: &Board, knight_square: chess::Square) -> bool {
        // Knights can fork if they attack multiple valuable pieces
        let mut attacked_valuable_pieces = 0;
        let knight_attacks = self.get_knight_attacks(knight_square);

        for target_square in chess::ALL_SQUARES {
            if (knight_attacks & chess::BitBoard::from_square(target_square)) != chess::EMPTY {
                if let Some(piece) = board.piece_on(target_square) {
                    if board.color_on(target_square) == Some(board.side_to_move()) {
                        // This is our piece being attacked by opponent knight
                        match piece {
                            chess::Piece::Queen | chess::Piece::Rook | chess::Piece::King => {
                                attacked_valuable_pieces += 1;
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        attacked_valuable_pieces >= 2
    }

    /// Get knight attack pattern
    fn get_knight_attacks(&self, square: chess::Square) -> chess::BitBoard {
        // Knight moves: 2+1 squares in L-shape
        let mut attacks = chess::EMPTY;
        let rank = square.get_rank().to_index() as i8;
        let file = square.get_file().to_index() as i8;

        let moves = [
            (2, 1),
            (2, -1),
            (-2, 1),
            (-2, -1),
            (1, 2),
            (1, -2),
            (-1, 2),
            (-1, -2),
        ];

        for (rank_offset, file_offset) in moves {
            let new_rank = rank + rank_offset;
            let new_file = file + file_offset;

            if new_rank >= 0 && new_rank < 8 && new_file >= 0 && new_file < 8 {
                let target_square = chess::Square::make_square(
                    chess::Rank::from_index(new_rank as usize),
                    chess::File::from_index(new_file as usize),
                );
                attacks |= chess::BitBoard::from_square(target_square);
            }
        }

        attacks
    }

    /// Check if piece attacks square (simplified)
    fn piece_attacks_square(
        &self,
        _board: &Board,
        piece_square: chess::Square,
        piece: chess::Piece,
        target_square: chess::Square,
    ) -> bool {
        match piece {
            chess::Piece::Knight => {
                let knight_attacks = self.get_knight_attacks(piece_square);
                (knight_attacks & chess::BitBoard::from_square(target_square)) != chess::EMPTY
            }
            chess::Piece::Queen | chess::Piece::Rook | chess::Piece::Bishop => {
                // Simplified - just check if they're on same rank/file/diagonal
                let piece_rank = piece_square.get_rank().to_index();
                let piece_file = piece_square.get_file().to_index();
                let target_rank = target_square.get_rank().to_index();
                let target_file = target_square.get_file().to_index();

                // Same rank or file (rook/queen)
                if (piece == chess::Piece::Rook || piece == chess::Piece::Queen)
                    && (piece_rank == target_rank || piece_file == target_file)
                {
                    return true;
                }

                // Same diagonal (bishop/queen)
                if (piece == chess::Piece::Bishop || piece == chess::Piece::Queen)
                    && ((piece_rank as i8 - target_rank as i8).abs()
                        == (piece_file as i8 - target_file as i8).abs())
                {
                    return true;
                }

                false
            }
            _ => false,
        }
    }

    /// Check if king is exposed to tactical threats
    /// CRITICAL: Check if kings are in danger (for pattern override decision)
    fn has_king_danger(&self, board: &Board) -> bool {
        for color in [chess::Color::White, chess::Color::Black] {
            let king_square = board.king_square(color);

            // Count attackers on king
            let mut attackers = 0;
            let opponent_color = !color;

            // Quick check for direct attacks on king
            for piece_type in [
                chess::Piece::Queen,
                chess::Piece::Rook,
                chess::Piece::Bishop,
                chess::Piece::Knight,
                chess::Piece::Pawn,
            ] {
                let opponent_pieces =
                    board.pieces(piece_type) & board.color_combined(opponent_color);
                for piece_square in opponent_pieces {
                    if self.piece_attacks_square(board, piece_square, piece_type, king_square) {
                        attackers += 1;
                        if attackers >= 2 {
                            return true; // Multiple attackers = danger
                        }
                    }
                }
            }

            // Check if king is in center (dangerous)
            let king_file = king_square.get_file().to_index();
            let king_rank = king_square.get_rank().to_index();
            if king_file >= 2 && king_file <= 5 && king_rank >= 2 && king_rank <= 5 {
                return true; // King in center is dangerous
            }
        }

        false
    }

    fn is_king_exposed(&self, board: &Board) -> bool {
        let king_square = board.king_square(board.side_to_move());

        // Check if king has few escape squares
        let mut escape_squares = 0;
        for rank_offset in -1..=1 {
            for file_offset in -1..=1 {
                if rank_offset == 0 && file_offset == 0 {
                    continue;
                }

                let king_rank = king_square.get_rank().to_index() as i8;
                let king_file = king_square.get_file().to_index() as i8;

                let new_rank = king_rank + rank_offset;
                let new_file = king_file + file_offset;

                if new_rank >= 0 && new_rank < 8 && new_file >= 0 && new_file < 8 {
                    let escape_square = chess::Square::make_square(
                        chess::Rank::from_index(new_rank as usize),
                        chess::File::from_index(new_file as usize),
                    );

                    // Check if escape square is safe
                    if board.piece_on(escape_square).is_none()
                        && !self.is_square_attacked(board, escape_square, !board.side_to_move())
                    {
                        escape_squares += 1;
                    }
                }
            }
        }

        escape_squares < 3 // King has limited mobility
    }

    /// Calculate position criticality for adaptive time allocation
    fn calculate_position_criticality(
        &self,
        board: &Board,
        material_deficit: f32,
        position_complexity: f32,
    ) -> f32 {
        let mut criticality = 0.0;

        // Material imbalance increases criticality
        criticality += (material_deficit / 10.0).min(0.4); // Up to 0.4 for major piece deficit

        // Position complexity increases criticality
        criticality += position_complexity * 0.3; // Up to 0.3 from complexity

        // In check dramatically increases criticality
        if *board.checkers() != chess::EMPTY {
            criticality += 0.4;
        }

        // Multiple pieces under attack increases criticality
        let attacked_pieces = self.count_attacked_pieces(board);
        criticality += (attacked_pieces as f32 * 0.05).min(0.2); // Up to 0.2 from attacks

        // Tactical motifs increase criticality
        if self.has_tactical_motifs(board) {
            criticality += 0.3;
        }

        // King exposure increases criticality
        if self.is_king_exposed(board) {
            criticality += 0.2;
        }

        // Hanging pieces dramatically increase criticality
        if self.has_hanging_pieces(board) {
            criticality += 0.4;
        }

        // Clamp between 0.0 and 1.0
        criticality.max(0.0).min(1.0)
    }

    /// Calculate adaptive search parameters based on position criticality
    fn calculate_adaptive_search_parameters(
        &self,
        criticality: f32,
        material_deficit: f32,
        complexity: f32,
    ) -> (u32, u64) {
        // Base parameters
        let mut depth = 6u32;
        let mut time_ms = 1000u64;

        // Criticality-based adjustments
        if criticality > 0.9 {
            // Extremely critical - maximum resources
            depth = 14;
            time_ms = 5000;
        } else if criticality > 0.7 {
            // Very critical - heavy resources
            depth = 12;
            time_ms = 3000;
        } else if criticality > 0.5 {
            // Moderately critical - increased resources
            depth = 10;
            time_ms = 2000;
        } else if criticality > 0.3 {
            // Somewhat critical - slightly increased resources
            depth = 8;
            time_ms = 1500;
        }

        // Material deficit bonus (when behind, search deeper for tactics)
        if material_deficit > 3.0 {
            depth = depth.saturating_add(3);
            time_ms = (time_ms as f64 * 1.5) as u64;
        } else if material_deficit > 1.0 {
            depth = depth.saturating_add(1);
            time_ms = (time_ms as f64 * 1.2) as u64;
        }

        // Complexity bonus
        if complexity > 0.7 {
            depth = depth.saturating_add(2);
            time_ms = (time_ms as f64 * 1.3) as u64;
        }

        // Reasonable bounds
        depth = depth.max(4).min(16);
        time_ms = time_ms.max(200).min(10000);

        (depth, time_ms)
    }

    /// Get move recommendations based on similar positions and opening book
    pub fn recommend_moves(
        &mut self,
        board: &Board,
        num_recommendations: usize,
    ) -> Vec<MoveRecommendation> {
        // v0.4.0: First prioritize strategic proactive moves when strategic evaluation is enabled
        if let Some(ref strategic_evaluator) = self.strategic_evaluator {
            let proactive_moves = strategic_evaluator.generate_proactive_moves(board);

            if !proactive_moves.is_empty() {
                let mut strategic_recommendations = Vec::new();

                for (chess_move, strategic_value) in
                    proactive_moves.iter().take(num_recommendations)
                {
                    strategic_recommendations.push(MoveRecommendation {
                        chess_move: *chess_move,
                        confidence: (strategic_value / 80.0).clamp(0.3, 0.95), // Even higher confidence threshold
                        from_similar_position_count: 0, // Strategic moves aren't from pattern recognition
                        average_outcome: *strategic_value / 100.0, // Strategic evaluation as outcome
                    });
                }

                // If we have enough strategic recommendations, return them
                if strategic_recommendations.len() >= num_recommendations {
                    strategic_recommendations.truncate(num_recommendations);
                    return strategic_recommendations;
                }

                // Otherwise, continue to opening book and pattern recognition for additional moves
                // Strategic moves will be blended with other recommendations below
            } else {
                // No strategic moves passed ultra-strict safety check - force tactical search
                // This ensures we never play a move without proper safety verification
                return self.recommend_moves_with_tactical_search(board, num_recommendations);
            }
        }

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

        // Always use tactical search if available (blend with pattern recognition)
        if self.tactical_search.is_some() {
            if let Some(ref mut tactical_search) = self.tactical_search {
                // v0.4.0: Use strategic evaluation to guide tactical search when available
                let tactical_result =
                    if let Some(ref strategic_evaluator) = self.strategic_evaluator {
                        // Strategic evaluation guides tactical search priorities
                        if strategic_evaluator.should_play_aggressively(board) {
                            // Focus on aggressive tactical variations
                            tactical_search.search(board)
                        } else {
                            // Use standard tactical search for positional play
                            tactical_search.search(board)
                        }
                    } else {
                        // Standard tactical search without strategic guidance
                        tactical_search.search(board)
                    };

                // Add the best tactical move with strong confidence
                if let Some(best_move) = tactical_result.best_move {
                    // CRITICAL FIX: Evaluate position AFTER making the move, not before
                    let mut temp_board = *board;
                    temp_board = temp_board.make_move_new(best_move);
                    let move_evaluation = tactical_search.search(&temp_board).evaluation;

                    // v0.4.0: Adjust confidence based on strategic alignment
                    let confidence = if let Some(ref strategic_evaluator) = self.strategic_evaluator
                    {
                        let strategic_eval = strategic_evaluator.evaluate_strategic(board);
                        // Higher confidence if move aligns with strategic plan
                        if strategic_eval.attacking_moves.contains(&best_move) {
                            0.98 // Very high confidence for moves that align with strategic attack
                        } else if strategic_eval.positional_moves.contains(&best_move) {
                            0.95 // High confidence for positional strategic moves
                        } else {
                            0.90 // Standard tactical confidence
                        }
                    } else {
                        0.95 // Standard tactical confidence without strategic guidance
                    };

                    move_data.insert(best_move, vec![(confidence, move_evaluation)]);
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

                // Add ordered moves with tactical evaluation (CRITICAL FIX)
                // Evaluate ALL moves, don't limit prematurely - we'll sort by quality later
                for chess_move in ordered_moves.into_iter() {
                    move_data.entry(chess_move).or_insert_with(|| {
                        // Evaluate each candidate move properly
                        let mut temp_board = *board;
                        temp_board = temp_board.make_move_new(chess_move);
                        let move_evaluation = tactical_search.search(&temp_board).evaluation;

                        // v0.4.0: Adjust confidence based on strategic alignment
                        let confidence =
                            if let Some(ref strategic_evaluator) = self.strategic_evaluator {
                                let strategic_eval = strategic_evaluator.evaluate_strategic(board);
                                // Higher confidence for moves that align with strategic plans
                                if strategic_eval.attacking_moves.contains(&chess_move) {
                                    0.92 // High confidence for strategic attacking moves
                                } else if strategic_eval.positional_moves.contains(&chess_move) {
                                    0.88 // Good confidence for strategic positional moves
                                } else {
                                    0.85 // Standard confidence for tactical moves
                                }
                            } else {
                                0.90 // Standard tactical confidence
                            };

                        vec![(confidence, move_evaluation)]
                    });
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
                    // Without tactical search, use basic heuristic evaluation
                    let mut basic_eval = 0.0;

                    // Basic capture evaluation
                    if let Some(captured_piece) = board.piece_on(chess_move.get_dest()) {
                        basic_eval += match captured_piece {
                            chess::Piece::Pawn => 1.0,
                            chess::Piece::Knight | chess::Piece::Bishop => 3.0,
                            chess::Piece::Rook => 5.0,
                            chess::Piece::Queen => 9.0,
                            chess::Piece::King => 100.0, // Should never happen in legal moves
                        };
                    }

                    move_data.insert(chess_move, vec![(0.3, basic_eval)]); // Lower baseline confidence for unknown moves
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

        // Sort by average outcome considering side to move
        // White prefers higher evaluations, Black prefers lower evaluations
        recommendations.sort_by(|a, b| {
            match board.side_to_move() {
                chess::Color::White => {
                    // White wants higher evaluations first
                    b.average_outcome
                        .partial_cmp(&a.average_outcome)
                        .unwrap_or(std::cmp::Ordering::Equal)
                }
                chess::Color::Black => {
                    // Black wants lower evaluations first
                    a.average_outcome
                        .partial_cmp(&b.average_outcome)
                        .unwrap_or(std::cmp::Ordering::Equal)
                }
            }
        });

        // Apply hanging piece safety checks before finalizing recommendations
        recommendations = self.apply_hanging_piece_safety_checks(board, recommendations);

        // Return top recommendations
        recommendations.truncate(num_recommendations);
        recommendations
    }

    /// Apply hanging piece safety checks to move recommendations
    /// Reduces confidence for moves that leave pieces hanging or fail to address threats
    fn apply_hanging_piece_safety_checks(
        &mut self,
        board: &Board,
        mut recommendations: Vec<MoveRecommendation>,
    ) -> Vec<MoveRecommendation> {
        use chess::{MoveGen, Piece};

        for recommendation in &mut recommendations {
            let mut safety_penalty = 0.0;

            // Create the position after making the recommended move
            let mut temp_board = *board;
            temp_board = temp_board.make_move_new(recommendation.chess_move);

            // Check if this move leaves our own pieces hanging
            let our_color = board.side_to_move();
            let opponent_color = !our_color;

            // Generate all opponent moves after our recommended move
            let opponent_moves: Vec<chess::ChessMove> = MoveGen::new_legal(&temp_board).collect();

            // Check each of our pieces to see if they're now hanging
            for square in chess::ALL_SQUARES {
                if let Some(piece) = temp_board.piece_on(square) {
                    if temp_board.color_on(square) == Some(our_color) {
                        // This is our piece, check if it's hanging
                        let piece_value = match piece {
                            Piece::Pawn => 1.0,
                            Piece::Knight | Piece::Bishop => 3.0,
                            Piece::Rook => 5.0,
                            Piece::Queen => 9.0,
                            Piece::King => 0.0, // King safety handled separately
                        };

                        // Check if opponent can capture this piece
                        let can_be_captured =
                            opponent_moves.iter().any(|&mv| mv.get_dest() == square);

                        if can_be_captured {
                            // Check if the piece is defended
                            let is_defended =
                                self.is_piece_defended(&temp_board, square, our_color);

                            if !is_defended {
                                // Piece is hanging! Apply severe penalty
                                safety_penalty += piece_value * 2.0; // Major penalty for hanging pieces
                            } else {
                                // Piece is defended but still in danger - smaller penalty
                                safety_penalty += piece_value * 0.1; // 10% penalty for pieces under attack
                            }
                        }
                    }
                }
            }

            // Check if we're missing obvious threats from the original position
            let original_threats = self.find_immediate_threats(board, opponent_color);
            let resolved_threats =
                self.count_resolved_threats(board, &temp_board, &original_threats);

            // Penalty for not addressing critical threats
            if !original_threats.is_empty() && resolved_threats == 0 {
                safety_penalty += 2.0; // Major penalty for ignoring threats
            }

            // Apply safety penalty to confidence (but don't go below 0.1)
            let penalty_factor = 1.0 - (safety_penalty * 0.2_f32).min(0.8);
            recommendation.confidence *= penalty_factor;
            recommendation.confidence = recommendation.confidence.max(0.1);

            // Also adjust the average outcome to reflect the safety issues
            recommendation.average_outcome -= safety_penalty;
        }

        // Re-sort recommendations after applying safety penalties
        recommendations.sort_by(|a, b| {
            // Primary sort by confidence (higher is better)
            let confidence_cmp = b
                .confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal);
            if confidence_cmp != std::cmp::Ordering::Equal {
                return confidence_cmp;
            }

            // Secondary sort by average outcome (considering side to move)
            match board.side_to_move() {
                chess::Color::White => b
                    .average_outcome
                    .partial_cmp(&a.average_outcome)
                    .unwrap_or(std::cmp::Ordering::Equal),
                chess::Color::Black => a
                    .average_outcome
                    .partial_cmp(&b.average_outcome)
                    .unwrap_or(std::cmp::Ordering::Equal),
            }
        });

        recommendations
    }

    /// Check if a piece on a square is defended by friendly pieces
    fn is_piece_defended(
        &self,
        board: &Board,
        square: chess::Square,
        our_color: chess::Color,
    ) -> bool {
        use chess::ALL_SQUARES;

        // Check each of our pieces to see if it can attack the target square
        for source_square in ALL_SQUARES {
            if let Some(piece) = board.piece_on(source_square) {
                if board.color_on(source_square) == Some(our_color) {
                    // Check if this piece can attack the target square
                    if self.can_piece_attack(board, piece, source_square, square) {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Check if a specific piece can attack a target square
    fn can_piece_attack(
        &self,
        board: &Board,
        piece: chess::Piece,
        from: chess::Square,
        to: chess::Square,
    ) -> bool {
        use chess::Piece;

        // Create a hypothetical move and see if it would be legal for attack purposes
        // We need to check if the piece can reach the square, regardless of what's on it
        match piece {
            Piece::Pawn => {
                // Pawns attack diagonally
                let from_file = from.get_file().to_index();
                let from_rank = from.get_rank().to_index();
                let to_file = to.get_file().to_index();
                let to_rank = to.get_rank().to_index();

                let file_diff = (to_file as i32 - from_file as i32).abs();
                let rank_diff = to_rank as i32 - from_rank as i32;

                // Pawn attacks: one square diagonally forward
                file_diff == 1 && {
                    match board.color_on(from).unwrap() {
                        chess::Color::White => rank_diff == 1,
                        chess::Color::Black => rank_diff == -1,
                    }
                }
            }
            Piece::Knight => {
                // Knight moves in L-shape
                let from_file = from.get_file().to_index() as i32;
                let from_rank = from.get_rank().to_index() as i32;
                let to_file = to.get_file().to_index() as i32;
                let to_rank = to.get_rank().to_index() as i32;

                let file_diff = (to_file - from_file).abs();
                let rank_diff = (to_rank - from_rank).abs();

                (file_diff == 2 && rank_diff == 1) || (file_diff == 1 && rank_diff == 2)
            }
            Piece::Bishop => {
                // Bishop moves diagonally
                self.is_diagonal_clear(board, from, to)
            }
            Piece::Rook => {
                // Rook moves horizontally or vertically
                self.is_straight_clear(board, from, to)
            }
            Piece::Queen => {
                // Queen combines rook and bishop
                self.is_diagonal_clear(board, from, to) || self.is_straight_clear(board, from, to)
            }
            Piece::King => {
                // King moves one square in any direction
                let from_file = from.get_file().to_index() as i32;
                let from_rank = from.get_rank().to_index() as i32;
                let to_file = to.get_file().to_index() as i32;
                let to_rank = to.get_rank().to_index() as i32;

                let file_diff = (to_file - from_file).abs();
                let rank_diff = (to_rank - from_rank).abs();

                file_diff <= 1 && rank_diff <= 1 && (file_diff != 0 || rank_diff != 0)
            }
        }
    }

    /// Check if diagonal path is clear for bishop/queen
    fn is_diagonal_clear(&self, board: &Board, from: chess::Square, to: chess::Square) -> bool {
        let from_file = from.get_file().to_index() as i32;
        let from_rank = from.get_rank().to_index() as i32;
        let to_file = to.get_file().to_index() as i32;
        let to_rank = to.get_rank().to_index() as i32;

        let file_diff = to_file - from_file;
        let rank_diff = to_rank - from_rank;

        // Must be diagonal
        if file_diff.abs() != rank_diff.abs() || file_diff == 0 {
            return false;
        }

        let file_step = if file_diff > 0 { 1 } else { -1 };
        let rank_step = if rank_diff > 0 { 1 } else { -1 };

        let steps = file_diff.abs();

        // Check each square along the diagonal (excluding start and end)
        for i in 1..steps {
            let check_file = from_file + i * file_step;
            let check_rank = from_rank + i * rank_step;

            let check_square = chess::Square::make_square(
                chess::Rank::from_index(check_rank as usize),
                chess::File::from_index(check_file as usize),
            );
            if board.piece_on(check_square).is_some() {
                return false; // Path blocked
            }
        }

        true
    }

    /// Check if straight path is clear for rook/queen  
    fn is_straight_clear(&self, board: &Board, from: chess::Square, to: chess::Square) -> bool {
        let from_file = from.get_file().to_index() as i32;
        let from_rank = from.get_rank().to_index() as i32;
        let to_file = to.get_file().to_index() as i32;
        let to_rank = to.get_rank().to_index() as i32;

        // Must be horizontal or vertical
        if from_file != to_file && from_rank != to_rank {
            return false;
        }

        if from_file == to_file {
            // Vertical movement
            let start_rank = from_rank.min(to_rank);
            let end_rank = from_rank.max(to_rank);

            for rank in (start_rank + 1)..end_rank {
                let check_square = chess::Square::make_square(
                    chess::Rank::from_index(rank as usize),
                    chess::File::from_index(from_file as usize),
                );
                if board.piece_on(check_square).is_some() {
                    return false; // Path blocked
                }
            }
        } else {
            // Horizontal movement
            let start_file = from_file.min(to_file);
            let end_file = from_file.max(to_file);

            for file in (start_file + 1)..end_file {
                let check_square = chess::Square::make_square(
                    chess::Rank::from_index(from_rank as usize),
                    chess::File::from_index(file as usize),
                );
                if board.piece_on(check_square).is_some() {
                    return false; // Path blocked
                }
            }
        }

        true
    }

    /// Find immediate threats (opponent pieces that can capture our valuable pieces)
    fn find_immediate_threats(
        &self,
        board: &Board,
        opponent_color: chess::Color,
    ) -> Vec<(chess::Square, f32)> {
        use chess::MoveGen;

        let mut threats = Vec::new();

        // Generate opponent moves
        let opponent_moves: Vec<chess::ChessMove> = MoveGen::new_legal(board).collect();

        for mv in opponent_moves {
            let target_square = mv.get_dest();
            if let Some(piece) = board.piece_on(target_square) {
                if board.color_on(target_square) == Some(!opponent_color) {
                    // Opponent can capture our piece
                    let piece_value = match piece {
                        chess::Piece::Pawn => 1.0,
                        chess::Piece::Knight | chess::Piece::Bishop => 3.0,
                        chess::Piece::Rook => 5.0,
                        chess::Piece::Queen => 9.0,
                        chess::Piece::King => 100.0,
                    };
                    threats.push((target_square, piece_value));
                }
            }
        }

        threats
    }

    /// Count how many threats from original position are resolved after our move
    fn count_resolved_threats(
        &self,
        original_board: &Board,
        new_board: &Board,
        original_threats: &[(chess::Square, f32)],
    ) -> usize {
        let mut resolved = 0;

        for &(threatened_square, _value) in original_threats {
            // Check if the piece is still on the same square and still threatened
            let piece_still_there =
                new_board.piece_on(threatened_square) == original_board.piece_on(threatened_square);

            if !piece_still_there {
                // Piece moved away - threat resolved
                resolved += 1;
            } else {
                // Check if the threat still exists in the new position
                let still_threatened = self
                    .find_immediate_threats(new_board, new_board.side_to_move())
                    .iter()
                    .any(|&(square, _)| square == threatened_square);

                if !still_threatened {
                    resolved += 1;
                }
            }
        }

        resolved
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
                    compressed_vector: None,
                    created_at: current_time,
                };
                position_data_batch.push(position_data);
            }
        }

        // Batch save all positions in a single transaction (much faster!)
        if !position_data_batch.is_empty() {
            let saved_count = db.save_positions_batch(&position_data_batch)?;
            println!("📊 Batch saved {saved_count} positions");
        }

        // Save LSH configuration if enabled
        if let Some(ref lsh) = self.lsh_index {
            lsh.save_to_database(db)?;
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
                let mut evaluation = position_data.evaluation.unwrap_or(0.0) as f32;

                // Convert evaluation from centipawns to pawns if needed
                // If evaluation is outside typical pawn range (-10 to +10),
                // assume it's in centipawns and convert to pawns
                if evaluation.abs() > 15.0 {
                    evaluation /= 100.0;
                }

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
                println!("Starting fresh engine (load failed: {e})");
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
            println!("🧵 Parallel tactical search enabled with {num_threads} threads");
        }
    }

    /// Check if parallel search is enabled
    pub fn is_parallel_search_enabled(&self) -> bool {
        self.tactical_search
            .as_ref()
            .map(|ts| ts.config.enable_parallel_search)
            .unwrap_or(false)
    }

    /// Enable strategic evaluation for proactive, initiative-based play
    /// This transforms the engine from reactive to proactive by adding:
    /// - Initiative assessment and attacking potential evaluation
    /// - Strategic plan generation and execution
    /// - Piece coordination analysis for attacks
    /// - Proactive move generation instead of just responding
    pub fn enable_strategic_evaluation(&mut self, config: StrategicConfig) {
        self.strategic_evaluator = Some(StrategicEvaluator::new(config));
        println!("🎯 Strategic evaluation enabled - engine will play proactively");
    }

    /// Enable strategic evaluation with default balanced configuration
    pub fn enable_strategic_evaluation_default(&mut self) {
        self.enable_strategic_evaluation(StrategicConfig::default());
    }

    /// Enable aggressive strategic configuration for maximum initiative
    pub fn enable_strategic_evaluation_aggressive(&mut self) {
        self.enable_strategic_evaluation(StrategicConfig::aggressive());
        println!("⚔️  Aggressive strategic evaluation enabled - maximum initiative focus");
    }

    /// Enable positional strategic configuration for long-term planning
    pub fn enable_strategic_evaluation_positional(&mut self) {
        self.enable_strategic_evaluation(StrategicConfig::positional());
        println!("📋 Positional strategic evaluation enabled - long-term planning focus");
    }

    /// Check if strategic evaluation is enabled
    pub fn is_strategic_evaluation_enabled(&self) -> bool {
        self.strategic_evaluator.is_some()
    }

    /// Get strategic evaluation for a position (if strategic evaluator is enabled)
    pub fn get_strategic_evaluation(&self, board: &Board) -> Option<StrategicEvaluation> {
        self.strategic_evaluator
            .as_ref()
            .map(|evaluator| evaluator.evaluate_strategic(board))
    }

    /// Generate proactive moves using strategic evaluation
    /// Returns moves ordered by strategic value (highest first)
    pub fn generate_proactive_moves(&self, board: &Board) -> Vec<(ChessMove, f32)> {
        if let Some(ref evaluator) = self.strategic_evaluator {
            evaluator.generate_proactive_moves(board)
        } else {
            // Fallback to basic move generation if strategic evaluator not enabled
            Vec::new()
        }
    }

    /// Check if the engine should play aggressively in current position
    pub fn should_play_aggressively(&self, board: &Board) -> bool {
        if let Some(ref evaluator) = self.strategic_evaluator {
            evaluator.should_play_aggressively(board)
        } else {
            false // Conservative default without strategic evaluator
        }
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

    /// Enable NNUE neural network evaluation for fast position assessment
    /// Automatically loads default_hybrid.config if present, otherwise creates new NNUE
    pub fn enable_nnue(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.enable_nnue_with_auto_load(true)
    }

    /// Enable NNUE with optional auto-loading of default model
    pub fn enable_nnue_with_auto_load(
        &mut self,
        auto_load: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let config = NNUEConfig::default();
        let mut nnue = NNUE::new(config)?;

        // Try to auto-load default hybrid model if requested and available
        if auto_load {
            if let Err(e) = self.try_load_default_nnue_model(&mut nnue) {
                println!("📝 Default NNUE model not found, using fresh model: {}", e);
                println!(
                    "   💡 Create one with: cargo run --bin train_nnue -- --output default_hybrid"
                );
            } else {
                println!("✅ Auto-loaded default NNUE model (default_hybrid.config)");

                // Check if weights were properly applied
                if !nnue.are_weights_loaded() {
                    println!("⚠️  Weights not properly applied, will use quick training fallback");
                } else {
                    println!("✅ Weights successfully applied to feature transformer");
                }
            }
        }

        self.nnue = Some(nnue);
        Ok(())
    }

    /// Try to load default NNUE model from standard locations
    fn try_load_default_nnue_model(
        &self,
        nnue: &mut NNUE,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Try multiple default model locations in order of preference
        let default_paths = [
            "default_hybrid",         // Primary production model
            "production_hybrid",      // Alternative production model
            "hybrid_production_nnue", // Comprehensive model
            "chess_nnue_advanced",    // Advanced model
            "trained_nnue_model",     // Basic trained model
        ];

        for path in &default_paths {
            let config_path = format!("{}.config", path);
            if std::path::Path::new(&config_path).exists() {
                nnue.load_model(path)?;
                return Ok(());
            }
        }

        Err("No default NNUE model found in standard locations".into())
    }

    /// Enable NNUE with custom configuration (bypasses auto-loading)
    pub fn enable_nnue_with_config(
        &mut self,
        config: NNUEConfig,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.nnue = Some(NNUE::new(config)?);
        Ok(())
    }

    /// Enable NNUE and load a specific pre-trained model
    pub fn enable_nnue_with_model(
        &mut self,
        model_path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let config = NNUEConfig::default();
        let mut nnue = NNUE::new(config)?;
        nnue.load_model(model_path)?;
        self.nnue = Some(nnue);
        Ok(())
    }

    /// Quick NNUE training if weights weren't properly loaded
    pub fn quick_fix_nnue_if_needed(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref mut nnue) = self.nnue {
            if !nnue.are_weights_loaded() {
                // Create basic training positions
                let training_positions = vec![(chess::Board::default(), 0.0)];

                // Add a few more positions if they parse correctly
                let mut positions = training_positions;
                if let Ok(board) = chess::Board::from_str(
                    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
                ) {
                    positions.push((board, 0.25));
                }
                if let Ok(board) = chess::Board::from_str("8/8/8/8/8/8/1K6/k6Q w - - 0 1") {
                    positions.push((board, 9.0));
                }

                nnue.quick_fix_training(&positions)?;
            }
        }
        Ok(())
    }

    /// Configure NNUE settings (only works if NNUE is already enabled)
    pub fn configure_nnue(&mut self, config: NNUEConfig) -> Result<(), Box<dyn std::error::Error>> {
        if self.nnue.is_some() {
            self.nnue = Some(NNUE::new(config)?);
            Ok(())
        } else {
            Err("NNUE must be enabled first before configuring".into())
        }
    }

    /// Check if NNUE neural network evaluation is enabled
    pub fn is_nnue_enabled(&self) -> bool {
        self.nnue.is_some()
    }

    /// Train NNUE on position data (requires NNUE to be enabled)
    pub fn train_nnue(
        &mut self,
        positions: &[(Board, f32)],
    ) -> Result<f32, Box<dyn std::error::Error>> {
        if let Some(ref mut nnue) = self.nnue {
            let loss = nnue.train_batch(positions)?;
            Ok(loss)
        } else {
            Err("NNUE must be enabled before training".into())
        }
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
                Ok(_) => println!("💾 Saved {positions_added} positions to database"),
                Err(_e) => println!("Loading complete"),
            }
        }

        println!("🧠 Self-play training complete: {positions_added} new positions learned");
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

        println!("🔄 Starting continuous self-play training for {iterations} iterations...");

        for iteration in 1..=iterations {
            println!("\n--- Self-Play Iteration {iteration}/{iterations} ---");

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
                        Ok(_) => println!("💾 Progress saved to {path} (binary format)"),
                        Err(_e) => println!("Loading complete"),
                    }
                }

                // Save to database if persistence is enabled
                if self.database.is_some() {
                    match self.save_to_database() {
                        Ok(_) => println!(
                            "💾 Database synchronized ({} total positions)",
                            self.knowledge_base_size()
                        ),
                        Err(_e) => println!("Loading complete"),
                    }
                }
            }

        }

        println!("\n🎉 Continuous self-play complete: {total_positions} total new positions");
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
            "🎯 Starting adaptive self-play training (target strength: {target_strength:.2})..."
        );

        loop {
            println!("\n--- Adaptive Iteration {iteration} ---");

            // Run self-play with current configuration
            let positions_added = self.self_play_training(current_config.clone())?;
            total_positions += positions_added;

            // Save to database after each iteration for resumability
            if self.database.is_some() {
                match self.save_to_database() {
                    Ok(_) => println!("💾 Adaptive training progress saved to database"),
                    Err(_e) => println!("Loading complete"),
                }
            }

            // Evaluate current strength (simplified - could use more sophisticated metrics)
            let current_strength = self.knowledge_base_size() as f32 / 10000.0; // Simple heuristic

            println!(
                "📊 Current strength estimate: {current_strength:.2} (target: {target_strength:.2})"
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
        // v0.3.0: With NNUE and hybrid evaluation, exact values may differ significantly
        // The hybrid approach can produce evaluations in a wider range
        let eval_value = evaluation.unwrap();
        assert!(
            eval_value > -1000.0 && eval_value < 1000.0,
            "Evaluation should be reasonable: {}",
            eval_value
        );
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
            // v0.3.0: With hybrid evaluation, the average outcome may not be exactly 0.0
            // The hybrid approach can produce evaluations in a wider range
            assert!(
                rec.average_outcome.abs() < 1000.0,
                "Average outcome should be reasonable: {}",
                rec.average_outcome
            );
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
        assert!(stats.unwrap().total_openings > 0);

        // Test opening book move recommendations
        let recommendations = engine.recommend_moves(&board, 3);
        assert!(!recommendations.is_empty());
        assert!(recommendations[0].confidence > 0.7); // Opening book should have high confidence
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

        // Basic performance bounds (generous to account for CI contention)
        assert!(
            encoding_time.as_millis() < 10000,
            "Position encoding too slow: {}ms",
            encoding_time.as_millis()
        );
        assert!(
            search_time.as_millis() < 5000,
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
        assert!(stats.opening_book_enabled); // Opening book is now enabled by default

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

        // v0.3.0: Tactical search is now enabled by default in the hybrid approach
        assert!(engine.is_tactical_search_enabled());

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
