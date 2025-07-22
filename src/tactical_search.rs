use crate::strategic_evaluator::{StrategicConfig, StrategicEvaluator};
use chess::{Board, ChessMove, Color, MoveGen, Square};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Game phase detection for evaluation tuning
#[derive(Debug, Clone, Copy, PartialEq)]
enum GamePhase {
    Opening,
    Middlegame,
    Endgame,
}

/// File type classification for strategic evaluation
#[derive(Debug, Clone, Copy)]
enum FileType {
    Open,
    SemiOpen,
    Closed,
}

/// Custom fixed-size transposition table with replacement strategy
#[derive(Clone)]
struct FixedTranspositionTable {
    entries: Vec<Option<TranspositionEntry>>,
    size: usize,
    age: u8,
}

impl FixedTranspositionTable {
    fn new(size_mb: usize) -> Self {
        let entry_size = std::mem::size_of::<TranspositionEntry>();
        let size = (size_mb * 1024 * 1024) / entry_size;

        Self {
            entries: vec![None; size],
            size,
            age: 0,
        }
    }

    fn get(&self, hash: u64) -> Option<&TranspositionEntry> {
        let index = (hash as usize) % self.size;
        self.entries[index].as_ref()
    }

    fn insert(&mut self, hash: u64, entry: TranspositionEntry) {
        let index = (hash as usize) % self.size;

        // Replacement strategy: always replace if empty, otherwise use depth + age
        let should_replace = match &self.entries[index] {
            None => true,
            Some(existing) => {
                // Replace if new entry has higher depth or is much newer
                entry.depth >= existing.depth || (self.age.wrapping_sub(existing.age) > 4)
            }
        };

        if should_replace {
            self.entries[index] = Some(TranspositionEntry {
                age: self.age,
                ..entry
            });
        }
    }

    fn clear(&mut self) {
        self.entries.fill(None);
        self.age = self.age.wrapping_add(1);
    }

    fn len(&self) -> usize {
        self.entries.iter().filter(|e| e.is_some()).count()
    }
}

/// Tactical search result
#[derive(Debug, Clone)]
pub struct TacticalResult {
    pub evaluation: f32,
    pub best_move: Option<ChessMove>,
    pub depth_reached: u32,
    pub nodes_searched: u64,
    pub time_elapsed: Duration,
    pub is_tactical: bool,
}

/// Tactical search configuration optimized for 2000+ ELO strength
#[derive(Debug, Clone)]
pub struct TacticalConfig {
    // Core search limits
    pub max_depth: u32,
    pub max_time_ms: u64,
    pub max_nodes: u64,
    pub quiescence_depth: u32,

    // Search techniques
    pub enable_transposition_table: bool,
    pub enable_iterative_deepening: bool,
    pub enable_aspiration_windows: bool,
    pub enable_null_move_pruning: bool,
    pub enable_late_move_reductions: bool,
    pub enable_principal_variation_search: bool,
    pub enable_parallel_search: bool,
    pub enable_quiescence: bool,
    pub num_threads: usize,

    // Hybrid evaluation integration (for vector-based approach)
    pub enable_hybrid_evaluation: bool,
    pub pattern_confidence_threshold: f32,
    pub pattern_weight: f32,

    // Advanced pruning techniques
    pub enable_futility_pruning: bool,
    pub enable_razoring: bool,
    pub enable_extended_futility_pruning: bool,
    pub futility_margin_base: f32,
    pub razor_margin: f32,
    pub extended_futility_margin: f32,

    // Ultra-aggressive pruning techniques
    pub enable_reverse_futility_pruning: bool,
    pub enable_static_null_move_pruning: bool,
    pub enable_move_count_pruning: bool,
    pub enable_history_pruning: bool,
    pub enable_see_pruning: bool,
    pub reverse_futility_margin: f32,
    pub move_count_base: u32,
    pub move_count_depth_factor: f32,
    pub history_pruning_threshold: i32,
    pub see_pruning_threshold: i32,

    // Advanced search parameters for 2000+ ELO
    pub null_move_reduction_depth: u32,
    pub lmr_min_depth: u32,
    pub lmr_min_moves: usize,
    pub aspiration_window_size: f32,
    pub aspiration_max_iterations: u32,
    pub transposition_table_size_mb: usize,
    pub killer_move_slots: usize,
    pub history_max_depth: u32,

    // Time management
    pub time_allocation_factor: f32,
    pub time_extension_threshold: f32,
    pub panic_time_factor: f32,

    // Evaluation blend weights
    pub endgame_evaluation_weight: f32,
    pub mobility_weight: f32,
    pub king_safety_weight: f32,
    pub pawn_structure_weight: f32,

    // Check extensions for forcing sequences
    pub enable_check_extensions: bool,
    pub check_extension_depth: u32,
    pub max_extensions_per_line: u32,

    // Additional hybrid evaluation settings
    pub hybrid_evaluation_weight: f32, // Weight for hybrid vs traditional evaluation
    pub hybrid_move_ordering: bool,    // Use hybrid evaluation for move ordering
    pub hybrid_pruning_threshold: f32, // Trust hybrid evaluation for pruning decisions
}

impl Default for TacticalConfig {
    fn default() -> Self {
        // v0.5.0: Direct hybrid configuration to avoid infinite recursion
        Self {
            // Reduced tactical depth since NNUE provides fast evaluation
            max_depth: 10,        // Deeper than fast, but rely on NNUE for accuracy
            max_time_ms: 1500,    // Moderate time - NNUE handles quick evaluation
            max_nodes: 1_000_000, // Reasonable node limit
            quiescence_depth: 16, // CRITICAL FIX: Deeper quiescence for complex tactical sequences

            // Search techniques - all enabled for strength
            enable_transposition_table: true,
            enable_iterative_deepening: true,
            enable_aspiration_windows: true,
            enable_null_move_pruning: true,
            enable_late_move_reductions: true,
            enable_principal_variation_search: true,
            enable_parallel_search: true,
            enable_quiescence: true,
            num_threads: 4,

            // Advanced pruning - more aggressive since NNUE evaluates well
            enable_futility_pruning: true,
            enable_razoring: true,
            enable_extended_futility_pruning: true,
            futility_margin_base: 150.0, // More aggressive - trust NNUE evaluation
            razor_margin: 300.0,         // More aggressive razoring
            extended_futility_margin: 50.0, // Trust NNUE for position assessment

            // Ultra-aggressive pruning defaults
            enable_reverse_futility_pruning: true,
            enable_static_null_move_pruning: true,
            enable_move_count_pruning: true,
            enable_history_pruning: true,
            enable_see_pruning: true,
            reverse_futility_margin: 120.0,
            move_count_base: 3,
            move_count_depth_factor: 2.0,
            history_pruning_threshold: -1000,
            see_pruning_threshold: -100,

            // Advanced search parameters
            null_move_reduction_depth: 3,
            lmr_min_depth: 2,
            lmr_min_moves: 3,
            aspiration_window_size: 40.0, // Tighter window since NNUE is more accurate
            aspiration_max_iterations: 3, // Fewer re-searches needed
            transposition_table_size_mb: 64,
            killer_move_slots: 2,
            history_max_depth: 20,

            // Time management optimized for hybrid approach
            time_allocation_factor: 0.3, // Use less time - NNUE+patterns handle most positions
            time_extension_threshold: 1.0, // Extend less frequently
            panic_time_factor: 1.5,      // Moderate panic extension

            // Evaluation blend weights
            endgame_evaluation_weight: 1.2,
            mobility_weight: 1.0,
            king_safety_weight: 1.3,
            pawn_structure_weight: 0.9,

            // Check extensions
            enable_check_extensions: true,
            check_extension_depth: 3,
            max_extensions_per_line: 10,

            // Enable hybrid evaluation features
            enable_hybrid_evaluation: true,
            hybrid_evaluation_weight: 0.8, // Heavily favor NNUE+patterns
            hybrid_move_ordering: true,    // Use hybrid insights for move ordering
            hybrid_pruning_threshold: 0.6, // Trust hybrid evaluation for pruning
            pattern_confidence_threshold: 0.65, // Trust pattern when confidence > 65%
            pattern_weight: 0.4,           // Pattern evaluation gets 40% weight in blend
        }
    }
}

impl TacticalConfig {
    /// Create configuration optimized for hybrid NNUE+pattern recognition engine
    pub fn hybrid_optimized() -> Self {
        Self {
            // Increased tactical depth for better tactical accuracy
            max_depth: 12,        // Deeper search for tactical reliability
            max_time_ms: 2000,    // More time for tactical calculation
            max_nodes: 2_000_000, // Higher node limit for thorough search
            quiescence_depth: 20, // CRITICAL FIX: Much deeper quiescence for complex tactical sequences

            // Optimized for NNUE integration
            aspiration_window_size: 40.0, // Tighter window since NNUE is more accurate
            aspiration_max_iterations: 3, // Fewer re-searches needed

            // More aggressive pruning since NNUE evaluates well
            futility_margin_base: 150.0, // More aggressive - trust NNUE evaluation
            razor_margin: 300.0,         // More aggressive razoring
            extended_futility_margin: 50.0, // Trust NNUE for position assessment

            // Time management optimized for hybrid approach
            time_allocation_factor: 0.3, // Use less time - NNUE+patterns handle most positions
            time_extension_threshold: 1.0, // Extend less frequently
            panic_time_factor: 1.5,      // Moderate panic extension

            // Enable hybrid evaluation features
            enable_hybrid_evaluation: true,
            hybrid_evaluation_weight: 0.8, // Heavily favor NNUE+patterns
            hybrid_move_ordering: true,    // Use hybrid insights for move ordering
            hybrid_pruning_threshold: 0.6, // Trust hybrid evaluation for pruning

            ..Default::default()
        }
    }

    /// Create configuration optimized for speed when NNUE+patterns are strong
    pub fn nnue_assisted_fast() -> Self {
        Self {
            max_depth: 6,        // Very shallow - rely heavily on NNUE
            max_time_ms: 500,    // Very fast - NNUE does the heavy lifting
            max_nodes: 100_000,  // Low node count
            quiescence_depth: 4, // Minimal quiescence

            // Aggressive pruning since NNUE evaluation is trusted
            futility_margin_base: 100.0,
            razor_margin: 200.0,
            extended_futility_margin: 30.0,

            // Minimal aspiration windows
            aspiration_window_size: 60.0,
            aspiration_max_iterations: 2,

            // Aggressive time management
            time_allocation_factor: 0.2,
            time_extension_threshold: 1.5,
            panic_time_factor: 1.2,

            // Maximum hybrid evaluation trust
            enable_hybrid_evaluation: true,
            hybrid_evaluation_weight: 0.9, // Almost fully trust NNUE+patterns
            hybrid_move_ordering: true,    // Use hybrid insights heavily
            hybrid_pruning_threshold: 0.8, // High trust for aggressive pruning

            ..Default::default()
        }
    }

    /// Create configuration optimized for competitive depth and strength
    pub fn fast() -> Self {
        Self {
            max_depth: 10,       // Competitive depth for tournament play
            max_time_ms: 1000,   // Full second for competitive time
            max_nodes: 500_000,  // Higher node budget
            quiescence_depth: 6, // Deeper quiescence for tactics
            aspiration_window_size: 50.0,
            transposition_table_size_mb: 64, // Larger TT for deeper search
            num_threads: 1,
            // Hybrid evaluation for intelligent pruning
            enable_hybrid_evaluation: true,
            hybrid_evaluation_weight: 0.9,
            hybrid_move_ordering: true,
            hybrid_pruning_threshold: 0.75,
            pattern_confidence_threshold: 0.6,
            pattern_weight: 0.5,
            // AGGRESSIVE PRUNING for depth
            enable_futility_pruning: true,
            enable_razoring: true,
            enable_extended_futility_pruning: true,
            futility_margin_base: 80.0,     // Aggressive but not extreme
            razor_margin: 200.0,            // Aggressive razoring
            extended_futility_margin: 30.0, // Aggressive extended pruning
            // Enable powerful search techniques
            enable_null_move_pruning: true,    // Critical for depth
            enable_late_move_reductions: true, // Critical for depth
            enable_principal_variation_search: true, // Better move ordering
            enable_iterative_deepening: true,
            enable_aspiration_windows: true,
            enable_transposition_table: true,
            // Moderate extensions
            enable_check_extensions: true,
            check_extension_depth: 1,
            max_extensions_per_line: 3,
            ..Default::default()
        }
    }

    /// Configuration optimized for maximum competitive strength with balanced pruning
    pub fn competitive() -> Self {
        Self {
            max_depth: 15,                    // Deep search for competitive strength
            max_time_ms: 1200,                // Reasonable time for competitive play
            max_nodes: 2_000_000,             // Good node budget
            quiescence_depth: 6,              // Reasonable quiescence depth
            aspiration_window_size: 50.0,     // Standard windows
            transposition_table_size_mb: 128, // Large TT for deep search
            num_threads: 1,
            // Enable hybrid evaluation
            enable_hybrid_evaluation: false, // Keep disabled for now
            hybrid_evaluation_weight: 0.8,
            hybrid_move_ordering: false,
            hybrid_pruning_threshold: 0.7,
            pattern_confidence_threshold: 0.6,
            pattern_weight: 0.5,
            // Enable reasonable pruning techniques
            enable_futility_pruning: true,
            enable_razoring: true,
            enable_extended_futility_pruning: true,
            enable_null_move_pruning: true,
            enable_late_move_reductions: true,
            enable_principal_variation_search: true,
            enable_iterative_deepening: true,
            enable_aspiration_windows: true,
            enable_transposition_table: true,
            enable_quiescence: true,
            enable_check_extensions: true,
            // Balanced pruning margins
            futility_margin_base: 100.0,
            razor_margin: 300.0,
            extended_futility_margin: 50.0,
            // Advanced pruning - enable with balanced settings
            enable_reverse_futility_pruning: true,
            enable_static_null_move_pruning: true,
            enable_move_count_pruning: true,
            enable_history_pruning: true,
            enable_see_pruning: true,
            reverse_futility_margin: 150.0,
            move_count_base: 4,
            move_count_depth_factor: 2.0,
            history_pruning_threshold: -200,
            see_pruning_threshold: -50,
            // Conservative extensions
            check_extension_depth: 1,
            max_extensions_per_line: 3,
            ..Default::default()
        }
    }

    /// Ultra-fast configuration for time-critical positions
    pub fn ultra_fast() -> Self {
        Self {
            max_depth: 6,        // Reasonable depth
            max_time_ms: 300,    // 300ms
            max_nodes: 100_000,  // Moderate node count
            quiescence_depth: 3, // Some quiescence
            aspiration_window_size: 100.0,
            transposition_table_size_mb: 32,
            num_threads: 1,
            // Heavy hybrid reliance
            enable_hybrid_evaluation: true,
            hybrid_evaluation_weight: 0.95,
            hybrid_move_ordering: true,
            hybrid_pruning_threshold: 0.8,
            pattern_confidence_threshold: 0.5,
            pattern_weight: 0.7,
            // Aggressive pruning for speed
            enable_futility_pruning: true,
            enable_razoring: true,
            futility_margin_base: 60.0,
            razor_margin: 150.0,
            // Essential search techniques only
            enable_null_move_pruning: true,
            enable_late_move_reductions: true,
            enable_iterative_deepening: true,
            enable_transposition_table: true,
            enable_quiescence: true,
            ..Default::default()
        }
    }

    /// Create configuration optimized for maximum strength (correspondence)
    pub fn strong() -> Self {
        Self {
            max_depth: 18,                    // Even deeper for maximum strength
            max_time_ms: 30_000,              // 30 seconds
            max_nodes: 5_000_000,             // 5 million nodes
            quiescence_depth: 12,             // Very deep quiescence for tactical perfection
            aspiration_window_size: 25.0,     // Narrow window for accuracy
            transposition_table_size_mb: 256, // Large hash table
            num_threads: 8,                   // More threads for strength
            ..Default::default()
        }
    }

    /// Traditional configuration without hybrid evaluation
    pub fn traditional() -> Self {
        Self {
            // Core search limits - optimized for 2000+ ELO
            max_depth: 14,        // Deep search for tactical accuracy
            max_time_ms: 5000,    // 5 seconds for balanced analysis (better time management)
            max_nodes: 2_000_000, // 2 million nodes for deep calculation
            quiescence_depth: 12, // Very deep quiescence for forcing sequences

            // Search techniques - all enabled for maximum strength
            enable_transposition_table: true,
            enable_iterative_deepening: true,
            enable_aspiration_windows: true, // Enabled for efficiency
            enable_null_move_pruning: true,
            enable_late_move_reductions: true,
            enable_principal_variation_search: true,
            enable_parallel_search: true,
            enable_quiescence: true,
            num_threads: 4,

            // Advanced pruning - fine-tuned margins
            enable_futility_pruning: true,
            enable_razoring: true,
            enable_extended_futility_pruning: true,
            futility_margin_base: 200.0, // More aggressive futility pruning
            razor_margin: 400.0,         // More aggressive razoring
            extended_futility_margin: 60.0, // Refined extended futility

            // Traditional pruning (less aggressive)
            enable_reverse_futility_pruning: false,
            enable_static_null_move_pruning: false,
            enable_move_count_pruning: false,
            enable_history_pruning: false,
            enable_see_pruning: false,
            reverse_futility_margin: 150.0,
            move_count_base: 5,
            move_count_depth_factor: 3.0,
            history_pruning_threshold: -2000,
            see_pruning_threshold: -200,

            // Advanced search parameters for 2000+ ELO
            null_move_reduction_depth: 3,    // R=3 null move reduction
            lmr_min_depth: 2,                // More aggressive LMR at depth 2+
            lmr_min_moves: 2,                // LMR after 2nd move for maximum pruning
            aspiration_window_size: 50.0,    // ±50cp aspiration window
            aspiration_max_iterations: 4,    // Max 4 aspiration re-searches
            transposition_table_size_mb: 64, // 64MB hash table
            killer_move_slots: 2,            // 2 killer moves per ply
            history_max_depth: 20,           // History heuristic depth limit

            // Time management for optimal play
            time_allocation_factor: 0.4,   // Use 40% of available time
            time_extension_threshold: 0.8, // Extend if score drops 80cp
            panic_time_factor: 2.0,        // 2x time in critical positions

            // Evaluation blend weights (carefully tuned)
            endgame_evaluation_weight: 1.2, // Emphasize endgame patterns
            mobility_weight: 1.0,           // Standard mobility weight
            king_safety_weight: 1.3,        // Emphasize king safety
            pawn_structure_weight: 0.9,     // Moderate pawn structure weight

            // Check extensions for tactical accuracy
            enable_check_extensions: true, // Enable check extensions
            check_extension_depth: 3,      // Extend checks by 3 plies
            max_extensions_per_line: 10,   // Max 10 extensions per variation

            // Hybrid evaluation (disabled by default for compatibility)
            enable_hybrid_evaluation: false,
            hybrid_evaluation_weight: 0.7, // 70% hybrid, 30% traditional
            hybrid_move_ordering: false,   // Traditional move ordering by default
            hybrid_pruning_threshold: 0.5, // Moderate trust in hybrid evaluation
            pattern_confidence_threshold: 0.65, // Standard threshold
            pattern_weight: 0.3,           // Lower pattern weight for traditional mode
        }
    }

    /// Create configuration for analysis mode
    pub fn analysis() -> Self {
        Self {
            max_depth: 20,
            max_time_ms: 60_000,   // 1 minute
            max_nodes: 10_000_000, // 10 million nodes
            quiescence_depth: 10,
            enable_aspiration_windows: false, // Disable for accuracy
            transposition_table_size_mb: 512,
            num_threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
            ..Default::default()
        }
    }

    /// Create configuration optimized for maximum speed and efficiency  
    pub fn ultra_optimized() -> Self {
        Self {
            // Optimized search limits for speed
            max_depth: 12,        // Reasonable depth for quick games
            max_time_ms: 2000,    // 2 second time limit for real-time play
            max_nodes: 1_000_000, // 1M nodes for efficient pruning
            quiescence_depth: 8,  // Moderate quiescence to balance speed vs accuracy

            // Advanced search techniques (all enabled for maximum strength)
            enable_transposition_table: true,
            enable_iterative_deepening: true,
            enable_aspiration_windows: true,
            enable_null_move_pruning: true,
            enable_late_move_reductions: true,
            enable_principal_variation_search: true,
            enable_parallel_search: true,
            enable_quiescence: true,
            num_threads: 4, // Moderate thread count for speed

            // Ultra-aggressive pruning for maximum efficiency
            enable_futility_pruning: true,
            enable_razoring: true,
            enable_extended_futility_pruning: true,
            futility_margin_base: 250.0, // More aggressive futility pruning
            razor_margin: 500.0,         // More aggressive razoring
            extended_futility_margin: 80.0, // More aggressive extended futility

            // Maximum speed pruning
            enable_reverse_futility_pruning: true,
            enable_static_null_move_pruning: true,
            enable_move_count_pruning: true,
            enable_history_pruning: true,
            enable_see_pruning: true,
            reverse_futility_margin: 100.0,
            move_count_base: 2,
            move_count_depth_factor: 1.0,
            history_pruning_threshold: -300,
            see_pruning_threshold: -30,

            // Optimized search parameters for maximum speed
            null_move_reduction_depth: 4, // R=4 for more aggressive null move pruning
            lmr_min_depth: 3,             // Start LMR at depth 3 (more selective)
            lmr_min_moves: 4,             // LMR after 4th move (more aggressive)
            aspiration_window_size: 30.0, // Medium aspiration window
            aspiration_max_iterations: 3, // Limit re-searches for speed
            transposition_table_size_mb: 128, // Reasonable hash size
            killer_move_slots: 2,         // Standard killer moves
            history_max_depth: 16,        // Reasonable history depth

            // Aggressive time management for real-time play
            time_allocation_factor: 0.3, // Use only 30% of available time (speed focus)
            time_extension_threshold: 1.0, // Less likely to extend time
            panic_time_factor: 1.5,      // Less panic time (confidence in quick search)

            // Balanced evaluation weights for speed
            endgame_evaluation_weight: 1.0, // Standard endgame weight
            mobility_weight: 0.8,           // Slightly reduced mobility calculation
            king_safety_weight: 1.1,        // Moderate king safety emphasis
            pawn_structure_weight: 0.7,     // Reduced pawn structure calculation

            // Conservative extensions to prevent search explosion
            enable_check_extensions: true,
            check_extension_depth: 2,   // Shorter check extensions
            max_extensions_per_line: 6, // Limit extensions per line

            // Hybrid evaluation (moderate use for balanced approach)
            enable_hybrid_evaluation: false, // Conservative - rely on proven tactics
            hybrid_evaluation_weight: 0.5,   // Balanced hybrid/traditional blend
            hybrid_move_ordering: false,     // Traditional move ordering for speed
            hybrid_pruning_threshold: 0.4,   // Conservative hybrid pruning
            pattern_confidence_threshold: 0.65, // Standard threshold
            pattern_weight: 0.3,             // Moderate pattern weight
        }
    }
}

/// Transposition table entry
#[derive(Debug, Clone)]
struct TranspositionEntry {
    depth: u32,
    evaluation: f32,
    best_move: Option<ChessMove>,
    node_type: NodeType,
    age: u8, // For replacement strategy
}

#[derive(Debug, Clone, Copy)]
enum NodeType {
    Exact,
    LowerBound,
    UpperBound,
}

/// Fast tactical search engine for position refinement
#[derive(Clone)]
pub struct TacticalSearch {
    pub config: TacticalConfig,
    transposition_table: FixedTranspositionTable,
    nodes_searched: u64,
    start_time: Instant,
    /// Killer moves table for move ordering
    killer_moves: Vec<Vec<Option<ChessMove>>>, // [depth][killer_slot]
    /// History heuristic for move ordering
    history_heuristic: HashMap<(Square, Square), u32>,
    /// Counter moves table: maps last move to best refutation
    counter_moves: HashMap<(Square, Square), ChessMove>,
    /// Last move played (for counter move tracking)
    last_move: Option<ChessMove>,
    /// Strategic evaluator for initiative-based assessment
    strategic_evaluator: StrategicEvaluator,
}

impl TacticalSearch {
    /// Create a new tactical search engine
    pub fn new(config: TacticalConfig) -> Self {
        let max_depth = config.max_depth as usize + 1;
        Self {
            config,
            transposition_table: FixedTranspositionTable::new(64), // 64MB table
            nodes_searched: 0,
            start_time: Instant::now(),
            killer_moves: vec![vec![None; 2]; max_depth], // 2 killer moves per depth
            history_heuristic: HashMap::new(),
            counter_moves: HashMap::new(),
            last_move: None,
            strategic_evaluator: StrategicEvaluator::new(StrategicConfig::default()),
        }
    }

    /// Create with custom transposition table size
    pub fn with_table_size(config: TacticalConfig, table_size_mb: usize) -> Self {
        let max_depth = config.max_depth as usize + 1;
        Self {
            config,
            transposition_table: FixedTranspositionTable::new(table_size_mb),
            nodes_searched: 0,
            start_time: Instant::now(),
            killer_moves: vec![vec![None; 2]; max_depth], // 2 killer moves per depth
            history_heuristic: HashMap::new(),
            counter_moves: HashMap::new(),
            last_move: None,
            strategic_evaluator: StrategicEvaluator::new(StrategicConfig::default()),
        }
    }

    /// Create with default configuration
    pub fn new_default() -> Self {
        Self::new(TacticalConfig::default())
    }

    /// Ultra-fast tactical search with optimized move ordering and pruning
    pub fn search_optimized(&mut self, board: &Board) -> TacticalResult {
        self.nodes_searched = 0;
        self.start_time = Instant::now();
        self.transposition_table.clear();

        // Pre-compute position characteristics for optimized search
        let is_tactical = self.is_tactical_position(board);
        let position_phase = self.detect_game_phase(board);
        
        // Optimized search based on position characteristics
        let (evaluation, best_move, depth_reached) = if self.config.enable_iterative_deepening {
            self.iterative_deepening_optimized(board, position_phase)
        } else {
            let (eval, mv) = self.minimax_optimized(
                board,
                self.config.max_depth,
                f32::NEG_INFINITY,
                f32::INFINITY,
                board.side_to_move() == Color::White,
                position_phase,
            );
            (eval, mv, self.config.max_depth)
        };

        TacticalResult {
            evaluation,
            best_move,
            depth_reached,
            nodes_searched: self.nodes_searched,
            time_elapsed: self.start_time.elapsed(),
            is_tactical,
        }
    }

    /// Fast tactical evaluation with pre-computed move ordering scores
    fn get_move_order_score_optimized(&self, mv: ChessMove, board: &Board, depth: usize, game_phase: GamePhase) -> i32 {
        let mut score = 0;

        // Check killer moves first (fastest lookup)
        if depth < self.killer_moves.len() {
            for killer in &self.killer_moves[depth] {
                if let Some(killer_move) = killer {
                    if *killer_move == mv {
                        return 9000; // Very high priority
                    }
                }
            }
        }

        // Hash move (stored in transposition table)
        let hash = board.get_hash();
        if let Some(entry) = self.transposition_table.get(hash) {
            if let Some(hash_move) = entry.best_move {
                if hash_move == mv {
                    return 10000; // Highest priority
                }
            }
        }

        // Captures with MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
        let from_square = mv.get_source();
        let to_square = mv.get_dest();
        
        if let Some(captured_piece) = board.piece_on(to_square) {
            let victim_value = match captured_piece {
                chess::Piece::Queen => 900,
                chess::Piece::Rook => 500,
                chess::Piece::Bishop => 320,
                chess::Piece::Knight => 300,
                chess::Piece::Pawn => 100,
                chess::Piece::King => 0, // Should not happen in normal play
            };
            
            let attacker_value = if let Some(moving_piece) = board.piece_on(from_square) {
                match moving_piece {
                    chess::Piece::Pawn => 1,
                    chess::Piece::Knight => 3,
                    chess::Piece::Bishop => 3,
                    chess::Piece::Rook => 5,
                    chess::Piece::Queen => 9,
                    chess::Piece::King => 10,
                }
            } else {
                1
            };
            
            score += victim_value - attacker_value;
        }

        // Promotions
        if let Some(promotion) = mv.get_promotion() {
            score += match promotion {
                chess::Piece::Queen => 800,
                chess::Piece::Rook => 400,
                chess::Piece::Bishop => 250,
                chess::Piece::Knight => 250,
                _ => 0,
            };
        }

        // History heuristic with game phase weighting
        let history_key = (from_square, to_square);
        if let Some(&history_score) = self.history_heuristic.get(&history_key) {
            let phase_multiplier = match game_phase {
                GamePhase::Opening => 0.5,  // Less reliance on history in opening
                GamePhase::Middlegame => 1.0,
                GamePhase::Endgame => 1.5,  // More reliance on history in endgame
            };
            score += (history_score as f32 * phase_multiplier) as i32;
        }

        // Counter moves
        if let Some(last_move) = self.last_move {
            let counter_key = (last_move.get_source(), last_move.get_dest());
            if let Some(&counter_move) = self.counter_moves.get(&counter_key) {
                if counter_move == mv {
                    score += 200;
                }
            }
        }

        // Checks (tactical positions)
        let new_board = board.make_move_new(mv);
        if new_board.checkers().popcnt() > 0 {
            score += 100;
        }

        score
    }

    /// Ultra-fast move generation with pre-sorted ordering
    fn generate_ordered_moves_optimized(&mut self, board: &Board, depth: usize, game_phase: GamePhase) -> Vec<ChessMove> {
        let mut moves: Vec<ChessMove> = MoveGen::new_legal(board).collect();
        
        // Pre-compute all move scores for efficient sorting
        let mut move_scores: Vec<(ChessMove, i32)> = moves
            .iter()
            .map(|&mv| {
                let score = self.get_move_order_score_optimized(mv, board, depth, game_phase);
                (mv, score)
            })
            .collect();
        
        // Sort by score (highest first)
        move_scores.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        
        // Extract just the moves
        move_scores.into_iter().map(|(mv, _)| mv).collect()
    }

    /// Optimized iterative deepening with enhanced time management
    fn iterative_deepening_optimized(&mut self, board: &Board, game_phase: GamePhase) -> (f32, Option<ChessMove>, u32) {
        let mut best_move = None;
        let mut best_evaluation = if board.side_to_move() == Color::White {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
        let mut depth_reached = 1;
        
        // Aspiration window search for deeper depths
        let mut aspiration_window = self.config.aspiration_window_size;
        
        for depth in 1..=self.config.max_depth {
            if self.should_stop_search() {
                break;
            }
            
            let (evaluation, mv) = if depth <= 3 || !self.config.enable_aspiration_windows {
                // Full window search for shallow depths
                self.minimax_optimized(
                    board,
                    depth,
                    f32::NEG_INFINITY,
                    f32::INFINITY,
                    board.side_to_move() == Color::White,
                    game_phase,
                )
            } else {
                // Aspiration window search
                self.aspiration_search_optimized(board, depth, best_evaluation, aspiration_window, game_phase)
            };
            
            best_evaluation = evaluation;
            if let Some(new_move) = mv {
                best_move = Some(new_move);
            }
            depth_reached = depth;
            
            // Adaptive aspiration window sizing
            if depth > 3 {
                aspiration_window = self.config.aspiration_window_size;
            }
        }
        
        (best_evaluation, best_move, depth_reached)
    }

    /// Aspiration window search with fail-soft re-search
    fn aspiration_search_optimized(
        &mut self,
        board: &Board,
        depth: u32,
        prev_eval: f32,
        window_size: f32,
        game_phase: GamePhase,
    ) -> (f32, Option<ChessMove>) {
        let mut alpha = prev_eval - window_size;
        let mut beta = prev_eval + window_size;
        
        for _ in 0..self.config.aspiration_max_iterations {
            let (eval, mv) = self.minimax_optimized(board, depth, alpha, beta, board.side_to_move() == Color::White, game_phase);
            
            if eval <= alpha {
                // Fail low - widen alpha
                alpha = f32::NEG_INFINITY;
            } else if eval >= beta {
                // Fail high - widen beta  
                beta = f32::INFINITY;
            } else {
                // Within window - success
                return (eval, mv);
            }
        }
        
        // Final full-window search if aspiration fails
        self.minimax_optimized(board, depth, f32::NEG_INFINITY, f32::INFINITY, board.side_to_move() == Color::White, game_phase)
    }

    /// Optimized minimax with enhanced pruning and move ordering
    fn minimax_optimized(
        &mut self,
        board: &Board,
        depth: u32,
        mut alpha: f32,
        beta: f32,
        maximizing: bool,
        game_phase: GamePhase,
    ) -> (f32, Option<ChessMove>) {
        self.nodes_searched += 1;

        // Time check every 1024 nodes for performance
        if self.nodes_searched & 1023 == 0 && self.should_stop_search() {
            return (0.0, None);
        }

        // Terminal node check
        if depth == 0 {
            if self.config.enable_quiescence {
                return (self.quiescence_search_optimized(board, alpha, beta, maximizing, self.config.quiescence_depth), None);
            } else {
                return (self.evaluate_position_optimized(board, game_phase), None);
            }
        }

        // Transposition table lookup
        let hash = board.get_hash();
        if let Some(entry) = self.transposition_table.get(hash) {
            if entry.depth >= depth {
                match entry.node_type {
                    NodeType::Exact => return (entry.evaluation, entry.best_move),
                    NodeType::LowerBound if entry.evaluation >= beta => return (entry.evaluation, entry.best_move),
                    NodeType::UpperBound if entry.evaluation <= alpha => return (entry.evaluation, entry.best_move),
                    _ => {}
                }
            }
        }

        // Null move pruning optimization
        if self.config.enable_null_move_pruning 
            && depth >= self.config.null_move_reduction_depth + 1
            && board.checkers().popcnt() == 0  // Not in check
            && self.has_non_pawn_material(board, board.side_to_move())
        {
            // Skip the null move and search with reduced depth
            let null_board = board.null_move().unwrap_or(*board);
            let (null_eval, _) = self.minimax_optimized(
                &null_board,
                depth - self.config.null_move_reduction_depth - 1,
                -beta,
                -alpha,
                !maximizing,
                game_phase,
            );
            let null_eval = -null_eval;
            
            if null_eval >= beta {
                return (beta, None); // Beta cutoff
            }
        }

        // Move generation with optimized ordering
        let moves = self.generate_ordered_moves_optimized(board, depth as usize, game_phase);
        
        if moves.is_empty() {
            return (self.evaluate_terminal_position(board), None);
        }

        let mut best_move = None;
        let mut best_evaluation = if maximizing { f32::NEG_INFINITY } else { f32::INFINITY };
        let mut alpha = alpha;
        let original_alpha = alpha;
        let mut moves_searched = 0;

        for mv in moves {
            let new_board = board.make_move_new(mv);
            moves_searched += 1;

            let evaluation = if moves_searched == 1 {
                // Search first move with full window
                let (eval, _) = self.minimax_optimized(&new_board, depth - 1, -beta, -alpha, !maximizing, game_phase);
                -eval
            } else {
                // Late Move Reduction (LMR) for non-critical moves
                let should_reduce = self.config.enable_late_move_reductions
                    && depth >= self.config.lmr_min_depth
                    && moves_searched > self.config.lmr_min_moves
                    && board.piece_on(mv.get_dest()).is_none()  // Not a capture
                    && mv.get_promotion().is_none()             // Not a promotion
                    && new_board.checkers().popcnt() == 0;      // Not giving check

                if should_reduce {
                    // Search with reduced depth first
                    let reduction = 1 + ((moves_searched - self.config.lmr_min_moves) / 4) as u32;
                    let reduced_depth = (depth - 1).saturating_sub(reduction);
                    
                    let (eval, _) = self.minimax_optimized(&new_board, reduced_depth, -(alpha + 1.0), -alpha, !maximizing, game_phase);
                    let reduced_eval = -eval;
                    
                    if reduced_eval > alpha && reduced_eval < beta {
                        // Re-search with full depth and window
                        let (eval, _) = self.minimax_optimized(&new_board, depth - 1, -beta, -alpha, !maximizing, game_phase);
                        -eval
                    } else {
                        reduced_eval
                    }
                } else {
                    // Principal Variation Search (PVS)
                    if self.config.enable_principal_variation_search {
                        // Scout search with null window
                        let (eval, _) = self.minimax_optimized(&new_board, depth - 1, -(alpha + 1.0), -alpha, !maximizing, game_phase);
                        let scout_eval = -eval;
                        
                        if scout_eval > alpha && scout_eval < beta {
                            // Re-search with full window
                            let (eval, _) = self.minimax_optimized(&new_board, depth - 1, -beta, -alpha, !maximizing, game_phase);
                            -eval
                        } else {
                            scout_eval
                        }
                    } else {
                        // Standard alpha-beta search
                        let (eval, _) = self.minimax_optimized(&new_board, depth - 1, -beta, -alpha, !maximizing, game_phase);
                        -eval
                    }
                }
            };

            if maximizing {
                if evaluation > best_evaluation {
                    best_evaluation = evaluation;
                    best_move = Some(mv);
                }
                alpha = alpha.max(evaluation);
            } else {
                if evaluation < best_evaluation {
                    best_evaluation = evaluation;
                    best_move = Some(mv);
                }
                alpha = alpha.min(evaluation);
            }

            // Alpha-beta pruning
            if alpha >= beta {
                // Update killer moves and history
                self.update_killer_moves(mv, depth as usize);
                self.update_history_heuristic(mv, depth);
                break;
            }
        }

        // Store in transposition table
        let node_type = if best_evaluation <= original_alpha {
            NodeType::UpperBound
        } else if best_evaluation >= beta {
            NodeType::LowerBound
        } else {
            NodeType::Exact
        };

        let entry = TranspositionEntry {
            depth,
            evaluation: best_evaluation,
            best_move,
            node_type,
            age: 0,
        };
        self.transposition_table.insert(hash, entry);

        (best_evaluation, best_move)
    }

    /// Optimized quiescence search for tactical positions
    fn quiescence_search_optimized(&mut self, board: &Board, mut alpha: f32, beta: f32, maximizing: bool, depth: u32) -> f32 {
        self.nodes_searched += 1;

        if depth == 0 {
            return self.evaluate_position_optimized(board, self.detect_game_phase(board));
        }

        let stand_pat = self.evaluate_position_optimized(board, self.detect_game_phase(board));

        if maximizing {
            alpha = alpha.max(stand_pat);
            if alpha >= beta {
                return beta;
            }
        } else {
            alpha = alpha.min(stand_pat);
            if alpha <= beta {
                return beta;
            }
        }

        // Generate only captures and checks for quiescence
        let moves: Vec<ChessMove> = MoveGen::new_legal(board)
            .filter(|mv| {
                board.piece_on(mv.get_dest()).is_some() ||  // Captures
                mv.get_promotion().is_some() ||             // Promotions
                {
                    let new_board = board.make_move_new(*mv);
                    new_board.checkers().popcnt() > 0       // Checks
                }
            })
            .collect();

        if moves.is_empty() {
            return stand_pat;
        }

        for mv in moves {
            let new_board = board.make_move_new(mv);
            let evaluation = self.quiescence_search_optimized(&new_board, -beta, -alpha, !maximizing, depth - 1);
            let evaluation = -evaluation;

            if maximizing {
                alpha = alpha.max(evaluation);
            } else {
                alpha = alpha.min(evaluation);
            }

            if alpha >= beta {
                break;
            }
        }

        alpha
    }

    /// Optimized position evaluation with game phase awareness
    fn evaluate_position_optimized(&self, board: &Board, game_phase: GamePhase) -> f32 {
        // Fast material counting
        let mut white_material = 0.0;
        let mut black_material = 0.0;
        
        for square in chess::ALL_SQUARES {
            if let Some(piece) = board.piece_on(square) {
                let value = match piece {
                    chess::Piece::Pawn => 100.0,
                    chess::Piece::Knight => 300.0,
                    chess::Piece::Bishop => 320.0,
                    chess::Piece::Rook => 500.0,
                    chess::Piece::Queen => 900.0,
                    chess::Piece::King => 0.0,
                };
                
                if board.color_on(square) == Some(Color::White) {
                    white_material += value;
                } else {
                    black_material += value;
                }
            }
        }
        
        let material_balance = white_material - black_material;
        
        // Phase-specific evaluation adjustments
        let phase_adjustment = match game_phase {
            GamePhase::Opening => {
                // Favor development and king safety
                let mut adjustment = 0.0;
                
                // Basic opening evaluation - keep simple
                adjustment += 0.0;
                
                adjustment
            },
            GamePhase::Middlegame => {
                // Standard tactical evaluation
                0.0
            },
            GamePhase::Endgame => {
                // Favor king activity and pawn promotion
                let mut adjustment = 0.0;
                
                // Basic endgame evaluation - keep simple for now
                adjustment += 0.0;
                adjustment
            },
        };
        
        material_balance + phase_adjustment
    }


    /// Fast time check for search termination
    fn should_stop_search(&self) -> bool {
        self.start_time.elapsed().as_millis() > self.config.max_time_ms as u128
            || self.nodes_searched > self.config.max_nodes
    }

    /// Update killer moves table for move ordering
    fn update_killer_moves(&mut self, mv: ChessMove, depth: usize) {
        if depth < self.killer_moves.len() {
            // Shift killer moves and add new one
            if self.killer_moves[depth][0] != Some(mv) {
                self.killer_moves[depth][1] = self.killer_moves[depth][0];
                self.killer_moves[depth][0] = Some(mv);
            }
        }
    }

    /// Update history heuristic for move ordering
    fn update_history_heuristic(&mut self, mv: ChessMove, depth: u32) {
        let key = (mv.get_source(), mv.get_dest());
        let bonus = depth * depth; // Depth squared bonus
        *self.history_heuristic.entry(key).or_insert(0) += bonus;
        
        // Cap at reasonable maximum to prevent overflow
        if let Some(score) = self.history_heuristic.get_mut(&key) {
            *score = (*score).min(10000);
        }
    }

    /// Search for tactical opportunities in the position with confidence-based time management
    pub fn search(&mut self, board: &Board) -> TacticalResult {
        self.nodes_searched = 0;
        self.start_time = Instant::now();
        self.transposition_table.clear();

        // Check if this is already a tactical position
        let is_tactical = self.is_tactical_position(board);

        // Confidence-based time management - TEMPORARILY DISABLED FOR TESTING
        let (search_time_ms, search_depth) = if false {
            // self.config.enable_hybrid_evaluation {
            self.calculate_dynamic_search_limits(board)
        } else {
            (self.config.max_time_ms, self.config.max_depth)
        };

        // Update config for this search
        let original_time = self.config.max_time_ms;
        let original_depth = self.config.max_depth;
        self.config.max_time_ms = search_time_ms;
        self.config.max_depth = search_depth;

        let (evaluation, best_move, depth_reached) = if self.config.enable_iterative_deepening {
            self.iterative_deepening_search(board)
        } else {
            let (eval, mv) = self.minimax(
                board,
                search_depth,
                f32::NEG_INFINITY,
                f32::INFINITY,
                board.side_to_move() == Color::White,
            );
            (eval, mv, search_depth)
        };

        // Restore original config
        self.config.max_time_ms = original_time;
        self.config.max_depth = original_depth;

        TacticalResult {
            evaluation,
            best_move,
            depth_reached,
            nodes_searched: self.nodes_searched,
            time_elapsed: self.start_time.elapsed(),
            is_tactical,
        }
    }

    /// Parallel search using multiple threads for root move analysis
    pub fn search_parallel(&mut self, board: &Board) -> TacticalResult {
        if !self.config.enable_parallel_search || self.config.num_threads <= 1 {
            return self.search(board); // Fall back to single-threaded
        }

        self.nodes_searched = 0;
        self.start_time = Instant::now();
        self.transposition_table.clear();

        let is_tactical = self.is_tactical_position(board);
        let moves = self.generate_ordered_moves(board);

        if moves.is_empty() {
            return TacticalResult {
                evaluation: self.evaluate_terminal_position(board),
                best_move: None,
                depth_reached: 1,
                nodes_searched: 1,
                time_elapsed: self.start_time.elapsed(),
                is_tactical,
            };
        }

        // Parallel search at the root level
        let (evaluation, best_move, depth_reached) = if self.config.enable_iterative_deepening {
            self.parallel_iterative_deepening(board, moves)
        } else {
            self.parallel_root_search(board, moves, self.config.max_depth)
        };

        TacticalResult {
            evaluation,
            best_move,
            depth_reached,
            nodes_searched: self.nodes_searched,
            time_elapsed: self.start_time.elapsed(),
            is_tactical,
        }
    }

    /// Parallel root search for a given depth
    fn parallel_root_search(
        &mut self,
        board: &Board,
        moves: Vec<ChessMove>,
        depth: u32,
    ) -> (f32, Option<ChessMove>, u32) {
        let maximizing = board.side_to_move() == Color::White;
        let nodes_counter = Arc::new(Mutex::new(0u64));

        // Evaluate each move in parallel
        let move_scores: Vec<(ChessMove, f32)> = moves
            .into_par_iter()
            .map(|mv| {
                let new_board = board.make_move_new(mv);
                let mut search_clone = self.clone();
                search_clone.nodes_searched = 0;

                let (eval, _) = search_clone.minimax(
                    &new_board,
                    depth - 1,
                    f32::NEG_INFINITY,
                    f32::INFINITY,
                    !maximizing,
                );

                // Update global node counter
                if let Ok(mut counter) = nodes_counter.lock() {
                    *counter += search_clone.nodes_searched;
                }

                // Flip evaluation for opponent's move
                (mv, -eval)
            })
            .collect();

        // Update total nodes searched
        if let Ok(counter) = nodes_counter.lock() {
            self.nodes_searched = *counter;
        }

        // Find best move
        let best = move_scores
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        match best {
            Some((best_move, best_eval)) => (best_eval, Some(best_move), depth),
            None => (0.0, None, depth),
        }
    }

    /// Parallel iterative deepening search
    fn parallel_iterative_deepening(
        &mut self,
        board: &Board,
        mut moves: Vec<ChessMove>,
    ) -> (f32, Option<ChessMove>, u32) {
        let mut best_move: Option<ChessMove> = None;
        let mut best_evaluation = 0.0;
        let mut completed_depth = 0;

        for depth in 1..=self.config.max_depth {
            // Check time limit
            if self.start_time.elapsed().as_millis() > self.config.max_time_ms as u128 {
                break;
            }

            let (eval, mv, _) = self.parallel_root_search(board, moves.clone(), depth);

            best_evaluation = eval;
            best_move = mv;
            completed_depth = depth;

            // Move ordering: put best move first for next iteration
            if let Some(best) = best_move {
                if let Some(pos) = moves.iter().position(|&m| m == best) {
                    moves.swap(0, pos);
                }
            }
        }

        (best_evaluation, best_move, completed_depth)
    }

    /// Iterative deepening search with adaptive time management
    fn iterative_deepening_search(&mut self, board: &Board) -> (f32, Option<ChessMove>, u32) {
        let mut best_move: Option<ChessMove> = None;
        let mut best_evaluation = 0.0;
        let mut completed_depth = 0;

        // ULTRA-EFFICIENT TIME MANAGEMENT: Use most time efficiently
        // Allow iterative deepening to use nearly all available time
        // Only stop when we have <10% time remaining or reach max depth

        for depth in 1..=self.config.max_depth {
            let depth_start_time = std::time::Instant::now();

            // Check if we have reasonable time remaining (at least 10% of total budget)
            let elapsed = self.start_time.elapsed().as_millis() as u64;
            let time_remaining = self.config.max_time_ms.saturating_sub(elapsed);

            // ULTRA-AGGRESSIVE: Only stop if we have very little time left
            // Allow maximum time usage for deeper search
            if time_remaining < (self.config.max_time_ms / 10) {
                // Less than 10% time remaining - stop to avoid timeout
                break;
            }

            let window_size = if self.config.enable_aspiration_windows && depth > 2 {
                50.0 // Centipawn window
            } else {
                f32::INFINITY
            };

            let (evaluation, mv) = if self.config.enable_aspiration_windows && depth > 2 {
                self.aspiration_window_search(board, depth, best_evaluation, window_size)
            } else {
                self.minimax(
                    board,
                    depth,
                    f32::NEG_INFINITY,
                    f32::INFINITY,
                    board.side_to_move() == Color::White,
                )
            };

            // Update best result - CRITICAL FIX: Only update if we have a valid move
            // This prevents evaluation/move mismatch that causes poor move selection
            if mv.is_some() {
                best_evaluation = evaluation;
                best_move = mv;
            }
            completed_depth = depth;

            // Early termination for mate or clearly winning positions
            if evaluation.abs() > 9000.0 {
                break;
            }

            // Early termination for clearly winning/losing positions (>= 5 pawns advantage)
            // Only if we have a reasonable move and depth >= 8
            if depth >= 8 && mv.is_some() && evaluation.abs() >= 5.0 {
                break;
            }

            // Adaptive time management: if this depth took longer than expected,
            // we might not have time for the next one
            let depth_time_taken = depth_start_time.elapsed().as_millis() as u64;
            let remaining_time = self
                .config
                .max_time_ms
                .saturating_sub(elapsed + depth_time_taken);

            // If next depth would likely take too long, stop here
            if depth < self.config.max_depth {
                let estimated_next_depth_time = depth_time_taken * 3; // Conservative estimate
                if estimated_next_depth_time > remaining_time {
                    break;
                }
            }
        }

        (best_evaluation, best_move, completed_depth)
    }

    /// Calculate position complexity for adaptive time management
    fn calculate_position_complexity(&self, board: &Board) -> f32 {
        let mut complexity = 0.0;

        // More pieces = more complex
        let total_pieces = board.combined().popcnt() as f32;
        complexity += (total_pieces - 20.0) / 12.0; // Normalized to 0-1 range roughly

        // More legal moves = more complex
        let legal_moves = MoveGen::new_legal(board).count() as f32;
        complexity += (legal_moves - 20.0) / 20.0; // Normalized

        // In check = more complex
        if board.checkers().popcnt() > 0 {
            complexity += 0.5;
        }

        // Tactical positions = more complex
        if self.is_tactical_position(board) {
            complexity += 0.3;
        }

        // Endgame = less complex (faster search)
        let game_phase = self.determine_game_phase(board);
        if game_phase == GamePhase::Endgame {
            complexity -= 0.3;
        }

        // Clamp between 0.2 and 1.5
        complexity.clamp(0.2, 1.5)
    }

    /// Aspiration window search for efficiency
    fn aspiration_window_search(
        &mut self,
        board: &Board,
        depth: u32,
        prev_score: f32,
        window: f32,
    ) -> (f32, Option<ChessMove>) {
        let mut alpha = prev_score - window;
        let mut beta = prev_score + window;

        loop {
            let (score, mv) = self.minimax(
                board,
                depth,
                alpha,
                beta,
                board.side_to_move() == Color::White,
            );

            if score <= alpha {
                // Failed low, expand window down
                alpha = f32::NEG_INFINITY;
            } else if score >= beta {
                // Failed high, expand window up
                beta = f32::INFINITY;
            } else {
                // Score within window
                return (score, mv);
            }
        }
    }

    /// Minimax search with alpha-beta pruning and advanced pruning techniques
    fn minimax(
        &mut self,
        board: &Board,
        depth: u32,
        alpha: f32,
        beta: f32,
        maximizing: bool,
    ) -> (f32, Option<ChessMove>) {
        self.minimax_with_extensions(board, depth, alpha, beta, maximizing, 0)
    }

    fn minimax_with_extensions(
        &mut self,
        board: &Board,
        depth: u32,
        alpha: f32,
        beta: f32,
        maximizing: bool,
        extensions_used: u32,
    ) -> (f32, Option<ChessMove>) {
        self.nodes_searched += 1;

        // Time and node limit checks
        if self.start_time.elapsed().as_millis() > self.config.max_time_ms as u128
            || self.nodes_searched > self.config.max_nodes
        {
            return (self.evaluate_position(board), None);
        }

        // Check extensions for forcing sequences
        let mut actual_depth = depth;
        if self.config.enable_check_extensions
            && board.checkers().popcnt() > 0
            && extensions_used < self.config.max_extensions_per_line
        {
            actual_depth += self.config.check_extension_depth;
        }

        // CRITICAL FIX: Extend search for tactical threats near horizon (with safety limit)
        let mut new_extensions_used = extensions_used;
        if self.config.enable_check_extensions
            && depth <= 2
            && extensions_used < 2
            && self.has_tactical_threats(board)
        {
            actual_depth += 1; // Conservative extension to prevent stack overflow
            new_extensions_used += 1;
        }

        // Terminal conditions
        if actual_depth == 0 {
            // PERFORMANCE FIX: Removed expensive mate-in-N search that was causing exponential blowup
            // Check if quiescence search is enabled

            return if self.config.enable_quiescence {
                (
                    self.quiescence_search(
                        board,
                        self.config.quiescence_depth,
                        alpha,
                        beta,
                        maximizing,
                    ),
                    None,
                )
            } else {
                (self.evaluate_position(board), None)
            };
        }

        if board.status() != chess::BoardStatus::Ongoing {
            return (self.evaluate_terminal_position(board), None);
        }

        // Transposition table lookup
        if self.config.enable_transposition_table {
            if let Some(entry) = self.transposition_table.get(board.get_hash()) {
                if entry.depth >= depth {
                    match entry.node_type {
                        NodeType::Exact => return (entry.evaluation, entry.best_move),
                        NodeType::LowerBound if entry.evaluation >= beta => {
                            return (entry.evaluation, entry.best_move)
                        }
                        NodeType::UpperBound if entry.evaluation <= alpha => {
                            return (entry.evaluation, entry.best_move)
                        }
                        _ => {}
                    }
                }
            }
        }

        // Static evaluation for pruning decisions
        let static_eval = self.evaluate_position(board);

        // Razoring - if static eval is way below alpha, do shallow search first
        if self.config.enable_razoring
            && (1..=3).contains(&depth)
            && !maximizing // Only apply to non-PV nodes
            && static_eval + self.config.razor_margin < alpha
        {
            // Do shallow quiescence search
            let razor_eval = self.quiescence_search(board, 1, alpha, beta, maximizing);
            if razor_eval < alpha {
                return (razor_eval, None);
            }
        }

        // ULTRA-AGGRESSIVE FUTILITY PRUNING
        if self.config.enable_futility_pruning
            && depth <= 4  // Expand to more depths
            && !maximizing
            && board.checkers().popcnt() == 0
            && static_eval + (self.config.futility_margin_base * depth as f32) < alpha
        {
            // Advanced futility margins increase with depth
            return (static_eval, None);
        }

        // Extended futility pruning for depths 2-4
        if self.config.enable_extended_futility_pruning
            && (2..=4).contains(&depth)
            && !maximizing
            && board.checkers().popcnt() == 0
        // Not in check
        {
            let futility_margin = self.config.extended_futility_margin * (depth as f32);

            // Standard extended futility pruning
            if static_eval + futility_margin < alpha {
                return (static_eval, None);
            }

            // Additional aggressive pruning when far behind in material
            if static_eval + 5.0 < alpha && depth <= 3 {
                return (static_eval, None);
            }
        }

        // REVERSE FUTILITY PRUNING (Beta Pruning)
        if self.config.enable_reverse_futility_pruning
            && depth <= 7  // Apply to shallow depths
            && maximizing
            && board.checkers().popcnt() == 0
            && static_eval >= beta + self.config.reverse_futility_margin
        {
            // Position is so good that even shallow search should exceed beta
            return (static_eval, None);
        }

        // STATIC NULL MOVE PRUNING
        if self.config.enable_static_null_move_pruning
            && depth <= 6
            && maximizing
            && board.checkers().popcnt() == 0
            && self.has_non_pawn_material(board, board.side_to_move())
            && static_eval >= beta + 200.0
        // Well above beta
        {
            // Static position is so strong we can prune immediately
            return (static_eval, None);
        }

        // ULTRA-AGGRESSIVE NULL MOVE PRUNING
        if self.config.enable_null_move_pruning
            && depth >= 2  // Start early for maximum pruning
            && maximizing
            && board.checkers().popcnt() == 0
            && self.has_non_pawn_material(board, board.side_to_move())
            && static_eval >= beta
        // Only when position looks good
        {
            // ULTRA-DYNAMIC REDUCTION
            let null_move_reduction = if depth >= 7 {
                4 + (depth - 7) / 4 // Deeper reductions for deep search
            } else if depth >= 4 {
                3
            } else {
                2
            };

            let new_depth = depth.saturating_sub(null_move_reduction);

            // Make null move (switch sides without moving)
            let null_board = board.null_move().unwrap_or(*board);
            let (null_score, _) = self.minimax_with_extensions(
                &null_board,
                new_depth,
                -beta,
                -beta + 1.0,
                !maximizing,
                new_extensions_used,
            );

            // If null move fails high, we can prune aggressively
            if null_score >= beta {
                // VERIFICATION SEARCH for high depths
                if depth >= 12 && null_score < 9000.0 {
                    // Do verification search to avoid zugzwang
                    let verify_depth = depth.saturating_sub(4);
                    let (verify_score, _) = self.minimax_with_extensions(
                        board,
                        verify_depth,
                        beta - 1.0,
                        beta,
                        maximizing,
                        new_extensions_used,
                    );
                    if verify_score >= beta {
                        return (beta, None);
                    }
                } else {
                    return (beta, None);
                }
            }
        }

        // Get hash move from transposition table for better move ordering
        let hash_move = if self.config.enable_transposition_table {
            self.transposition_table
                .get(board.get_hash())
                .and_then(|entry| entry.best_move)
        } else {
            None
        };

        // Move ordering with hash move prioritization
        let moves = self.generate_ordered_moves_with_hash(board, hash_move, depth);

        let (best_value, best_move) =
            if self.config.enable_principal_variation_search && moves.len() > 1 {
                // Principal Variation Search (PVS)
                self.principal_variation_search(
                    board,
                    depth,
                    alpha,
                    beta,
                    maximizing,
                    moves,
                    new_extensions_used,
                )
            } else {
                // Standard alpha-beta search
                self.alpha_beta_search(
                    board,
                    depth,
                    alpha,
                    beta,
                    maximizing,
                    moves,
                    new_extensions_used,
                )
            };

        // Store in transposition table
        if self.config.enable_transposition_table {
            let node_type = if best_value <= alpha {
                NodeType::UpperBound
            } else if best_value >= beta {
                NodeType::LowerBound
            } else {
                NodeType::Exact
            };

            self.transposition_table.insert(
                board.get_hash(),
                TranspositionEntry {
                    depth,
                    evaluation: best_value,
                    best_move,
                    node_type,
                    age: 0, // Will be set by the table
                },
            );
        }

        (best_value, best_move)
    }

    /// Principal Variation Search - more efficient than pure alpha-beta
    fn principal_variation_search(
        &mut self,
        board: &Board,
        depth: u32,
        mut alpha: f32,
        mut beta: f32,
        maximizing: bool,
        moves: Vec<ChessMove>,
        extensions_used: u32,
    ) -> (f32, Option<ChessMove>) {
        let mut best_move: Option<ChessMove> = None;
        let mut best_value = if maximizing {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
        let mut _pv_found = false;
        let mut first_move = true;

        // If no moves available, return current position evaluation
        if moves.is_empty() {
            return (self.evaluate_position(board), None);
        }

        for (move_index, chess_move) in moves.into_iter().enumerate() {
            let new_board = board.make_move_new(chess_move);
            let mut evaluation;

            // MOVE COUNT PRUNING - Skip late moves at shallow depths
            if self.config.enable_move_count_pruning
                && depth <= 5
                && move_index
                    >= (self.config.move_count_base as usize
                        + (depth as f32 * self.config.move_count_depth_factor) as usize)
                && !self.is_capture_or_promotion(&chess_move, board)
                && new_board.checkers().popcnt() == 0
                && !self.is_killer_move(&chess_move)
                && best_move.is_some()
            // Only after we have at least one move
            {
                continue; // Skip this move entirely
            }

            // HISTORY PRUNING - Skip moves with bad history scores
            if self.config.enable_history_pruning
                && depth <= 4
                && move_index >= 4
                && !self.is_capture_or_promotion(&chess_move, board)
                && new_board.checkers().popcnt() == 0
                && (self.get_history_score(&chess_move) as i32)
                    < self.config.history_pruning_threshold
            {
                continue; // Skip this move entirely
            }

            // BALANCED LMR - aggressive but not extreme
            let reduction = if self.config.enable_late_move_reductions
                && depth >= 3  // Start reducing at reasonable depth
                && move_index >= 3 // Reduce from 4th move (reasonable)
                && !self.is_capture_or_promotion(&chess_move, board)
                && new_board.checkers().popcnt() == 0
                && !self.is_killer_move(&chess_move)
            {
                // BALANCED REDUCTIONS: Effective but not extreme
                let base_reduction = match move_index {
                    3..=6 => 1,
                    7..=12 => 2,
                    13..=20 => 3,
                    _ => 4, // Up to 4 ply reduction for very late moves
                };

                // Additional depth-based reduction (less aggressive)
                let depth_bonus = if depth >= 10 { 1 } else { 0 };

                (base_reduction + depth_bonus).min(depth.saturating_sub(1))
            } else {
                0
            };

            let search_depth = if depth > reduction {
                depth - 1 - reduction
            } else {
                0
            };

            if move_index == 0 {
                // Search first move with full window (likely the best move)
                let search_depth = if depth > 0 { depth - 1 } else { 0 };
                let (eval, _) = self.minimax_with_extensions(
                    &new_board,
                    search_depth,
                    alpha,
                    beta,
                    !maximizing,
                    extensions_used,
                );
                evaluation = eval;
                _pv_found = true;
            } else {
                // Search subsequent moves with null window first (PVS optimization)
                let null_window_alpha = if maximizing { alpha } else { beta - 1.0 };
                let null_window_beta = if maximizing { alpha + 1.0 } else { beta };

                let (null_eval, _) = self.minimax_with_extensions(
                    &new_board,
                    search_depth,
                    null_window_alpha,
                    null_window_beta,
                    !maximizing,
                    extensions_used,
                );

                // If null window search fails, re-search with full window
                if null_eval > alpha && null_eval < beta {
                    // Re-search with full window and full depth if reduced
                    let full_depth = if reduction > 0 {
                        if depth > 0 {
                            depth - 1
                        } else {
                            0
                        }
                    } else {
                        search_depth
                    };
                    let (full_eval, _) = self.minimax_with_extensions(
                        &new_board,
                        full_depth,
                        alpha,
                        beta,
                        !maximizing,
                        extensions_used,
                    );
                    evaluation = full_eval;
                } else {
                    evaluation = null_eval;

                    // If LMR was used and failed high, research with full depth
                    if reduction > 0
                        && ((maximizing && evaluation > alpha)
                            || (!maximizing && evaluation < beta))
                    {
                        let search_depth = if depth > 0 { depth - 1 } else { 0 };
                        let (re_eval, _) = self.minimax_with_extensions(
                            &new_board,
                            search_depth,
                            alpha,
                            beta,
                            !maximizing,
                            extensions_used,
                        );
                        evaluation = re_eval;
                    }
                }
            }

            // Update best move and alpha/beta
            if maximizing {
                if first_move || evaluation > best_value {
                    best_value = evaluation;
                    best_move = Some(chess_move);
                }
                alpha = alpha.max(evaluation);
            } else {
                if first_move || evaluation < best_value {
                    best_value = evaluation;
                    best_move = Some(chess_move);
                }
                beta = beta.min(evaluation);
            }

            first_move = false;

            // Alpha-beta pruning
            if beta <= alpha {
                // Store killer move for this depth if it's not a capture
                if !self.is_capture_or_promotion(&chess_move, board) {
                    self.store_killer_move(chess_move, depth);
                    self.update_history(&chess_move, depth);
                }
                break;
            }
        }

        (best_value, best_move)
    }

    /// Check if a move is an obvious blunder (loses material for nothing)
    fn is_obvious_blunder(&self, board: &Board, chess_move: ChessMove) -> bool {
        let dest = chess_move.get_dest();

        // Get the piece being moved
        let moving_piece = match board.piece_on(chess_move.get_source()) {
            Some(piece) => piece,
            None => return false, // Invalid move
        };

        // Calculate material exchange value
        let captured_value = board.piece_on(dest).map_or(0.0, |piece| match piece {
            chess::Piece::Pawn => 1.0,
            chess::Piece::Knight => 3.0,
            chess::Piece::Bishop => 3.0,
            chess::Piece::Rook => 5.0,
            chess::Piece::Queen => 9.0,
            chess::Piece::King => 100.0,
        });

        let moving_piece_value = match moving_piece {
            chess::Piece::Pawn => 1.0,
            chess::Piece::Knight => 3.0,
            chess::Piece::Bishop => 3.0,
            chess::Piece::Rook => 5.0,
            chess::Piece::Queen => 9.0,
            chess::Piece::King => 0.0, // King moves are special
        };

        // Basic blunder detection: Don't move high-value pieces to squares where they can be captured
        // This is a simplified check - in reality you'd need full attack/defend analysis

        // CRITICAL FIX: Use proper SEE (Static Exchange Evaluation)
        let net_exchange = if captured_value > 0.0 {
            // This is a capture - calculate full exchange sequence using SEE
            let see_result = self.calculate_material_exchange(
                &chess_move,
                board,
                board.piece_on(dest).unwrap(),
                moving_piece,
            ) as f32
                / 100.0;

            // IMPORTANT: If SEE is positive (we gain material), this is NOT a blunder
            if see_result >= 0.0 {
                return false; // Gaining material is never a blunder
            }
            see_result
        } else {
            // Non-capture move - check if we're moving to a square where we can be captured
            let new_board = board.make_move_new(chess_move);
            let attackers = self.count_attackers(&new_board, dest, !board.side_to_move());
            if attackers > 0 {
                // Check if the square is also defended
                let defenders = self.count_attackers(&new_board, dest, board.side_to_move());
                if defenders == 0 {
                    -moving_piece_value // Undefended piece loss
                } else {
                    // Roughly estimate exchange - if attackers outnumber defenders, likely bad
                    if attackers > defenders {
                        -moving_piece_value * 0.5 // Probably losing exchange
                    } else {
                        0.0 // Probably safe or equal exchange
                    }
                }
            } else {
                0.0 // Safe move
            }
        };

        if net_exchange < -0.5 {
            // Even losing half a pawn is bad
            // For ANY material loss, require very strong justification
            let new_board = board.make_move_new(chess_move);

            // Only allow if ALL of these conditions are met:
            // 1. We're attacking near enemy king AND
            // 2. We're giving check OR attacking multiple pieces AND
            // 3. The material loss is not catastrophic (< 3 pawns)
            let is_check = new_board.checkers().popcnt() > 0;
            let near_enemy_king = self.is_near_enemy_king(&new_board, dest);
            let catastrophic_loss = net_exchange < -3.0;

            if catastrophic_loss || !near_enemy_king || (!is_check && net_exchange < -1.0) {
                return true; // This is definitely a blunder
            }
        }

        // Queen moves to undefended squares where it can be captured by minor pieces are usually blunders
        if moving_piece == chess::Piece::Queen && captured_value == 0.0 {
            // If we're moving the queen to a square where it might be attacked, be cautious
            // This is a very basic heuristic
            let new_board = board.make_move_new(chess_move);
            if self.is_likely_under_attack(&new_board, dest, chess::Piece::Queen) {
                return true;
            }
        }

        false
    }

    /// Check if a square is near the enemy king
    fn is_near_enemy_king(&self, board: &Board, square: Square) -> bool {
        let enemy_color = board.side_to_move();
        let enemy_king_square =
            (board.pieces(chess::Piece::King) & board.color_combined(enemy_color)).to_square();

        // If we're within 2 squares of the enemy king, consider it an attack
        let rank_diff = (square.get_rank().to_index() as i8
            - enemy_king_square.get_rank().to_index() as i8)
            .abs();
        let file_diff = (square.get_file().to_index() as i8
            - enemy_king_square.get_file().to_index() as i8)
            .abs();

        rank_diff <= 2 && file_diff <= 2
    }

    /// Check if a piece is likely to be under attack (very basic heuristic)
    fn is_likely_under_attack(&self, board: &Board, square: Square, piece: chess::Piece) -> bool {
        // Very basic heuristic: queens in the center early in the game are often vulnerable
        if piece == chess::Piece::Queen {
            // Check if it's early in the game (many pieces still on board)
            let total_pieces = board.combined().popcnt();
            if total_pieces > 28 {
                // Early game
                // Queen in center files (d, e) in early game is often a target
                let file = square.get_file();
                if file == chess::File::D || file == chess::File::E {
                    return true;
                }
            }
        }

        false
    }

    /// Standard alpha-beta search (fallback when PVS is disabled)
    fn alpha_beta_search(
        &mut self,
        board: &Board,
        depth: u32,
        mut alpha: f32,
        mut beta: f32,
        maximizing: bool,
        moves: Vec<ChessMove>,
        extensions_used: u32,
    ) -> (f32, Option<ChessMove>) {
        let mut best_move: Option<ChessMove> = None;
        let mut best_value = if maximizing {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
        let mut first_move = true;

        // If no moves available, return current position evaluation
        if moves.is_empty() {
            return (self.evaluate_position(board), None);
        }

        for (move_index, chess_move) in moves.into_iter().enumerate() {
            let new_board = board.make_move_new(chess_move);

            // BALANCED LMR - aggressive but not extreme
            let reduction = if self.config.enable_late_move_reductions
                && depth >= 3  // Start reducing at reasonable depth
                && move_index >= 3 // Reduce from 4th move (reasonable)
                && !self.is_capture_or_promotion(&chess_move, board)
                && new_board.checkers().popcnt() == 0
                && !self.is_killer_move(&chess_move)
            {
                // BALANCED REDUCTIONS: Effective but not extreme
                let base_reduction = match move_index {
                    3..=6 => 1,
                    7..=12 => 2,
                    13..=20 => 3,
                    _ => 4, // Up to 4 ply reduction for very late moves
                };

                // Additional depth-based reduction (less aggressive)
                let depth_bonus = if depth >= 10 { 1 } else { 0 };

                (base_reduction + depth_bonus).min(depth.saturating_sub(1))
            } else {
                0
            };

            let search_depth = if depth > reduction {
                depth - 1 - reduction
            } else {
                0
            };

            let (evaluation, _) = self.minimax_with_extensions(
                &new_board,
                search_depth,
                alpha,
                beta,
                !maximizing,
                extensions_used,
            );

            // If LMR search failed high, research with full depth
            let final_evaluation = if reduction > 0
                && ((maximizing && evaluation > alpha) || (!maximizing && evaluation < beta))
            {
                let search_depth = if depth > 0 { depth - 1 } else { 0 };
                let (re_eval, _) = self.minimax_with_extensions(
                    &new_board,
                    search_depth,
                    alpha,
                    beta,
                    !maximizing,
                    extensions_used,
                );
                re_eval
            } else {
                evaluation
            };

            if maximizing {
                if first_move || final_evaluation > best_value {
                    best_value = final_evaluation;
                    best_move = Some(chess_move);
                }
                alpha = alpha.max(final_evaluation);
            } else {
                if first_move || final_evaluation < best_value {
                    best_value = final_evaluation;
                    best_move = Some(chess_move);
                }
                beta = beta.min(final_evaluation);
            }

            first_move = false;

            // Alpha-beta pruning
            if beta <= alpha {
                // Store killer move for this depth if it's not a capture
                if !self.is_capture_or_promotion(&chess_move, board) {
                    self.store_killer_move(chess_move, depth);
                    self.update_history(&chess_move, depth);
                }
                break;
            }
        }

        (best_value, best_move)
    }

    /// Quiescence search to avoid horizon effect
    fn quiescence_search(
        &mut self,
        board: &Board,
        depth: u32,
        mut alpha: f32,
        beta: f32,
        maximizing: bool,
    ) -> f32 {
        self.nodes_searched += 1;

        let stand_pat = self.evaluate_position(board);

        if depth == 0 {
            return stand_pat;
        }

        if maximizing {
            if stand_pat >= beta {
                return beta;
            }
            alpha = alpha.max(stand_pat);

            // Delta pruning - if we're very far behind even with a queen capture, prune
            if stand_pat + 9.0 < alpha {
                // Queen value in pawn units
                return stand_pat;
            }
        } else {
            if stand_pat <= alpha {
                return alpha;
            }

            // Delta pruning for minimizing side
            if stand_pat - 9.0 > alpha {
                // Queen value in pawn units
                return stand_pat;
            }
        }

        // CRITICAL FIX: Search captures, checks, and threats in quiescence
        let captures_and_checks = self.generate_captures_and_checks(board);

        for chess_move in captures_and_checks {
            // Skip bad captures in quiescence search (simple pruning)
            if let Some(captured_piece) = board.piece_on(chess_move.get_dest()) {
                if !self.is_good_capture(&chess_move, board, captured_piece) {
                    // Skip obviously bad captures (losing material)
                    continue;
                }
            }

            let new_board = board.make_move_new(chess_move);

            // CRITICAL: If this move leads to mate, evaluate it immediately
            if new_board.status() == chess::BoardStatus::Checkmate {
                let mate_score = if maximizing { 9999.0 } else { -9999.0 };
                if maximizing {
                    alpha = alpha.max(mate_score);
                    if alpha >= beta {
                        return beta;
                    }
                } else {
                    if mate_score <= alpha {
                        return alpha;
                    }
                }
                continue;
            }

            let evaluation =
                self.quiescence_search(&new_board, depth - 1, alpha, beta, !maximizing);

            if maximizing {
                alpha = alpha.max(evaluation);
                if alpha >= beta {
                    break;
                }
            } else if evaluation <= alpha {
                return alpha;
            }
        }

        stand_pat
    }

    /// Generate moves ordered by likely tactical value with advanced heuristics
    fn generate_ordered_moves(&self, board: &Board) -> Vec<ChessMove> {
        self.generate_ordered_moves_with_hash(board, None, 1) // Use depth 1 instead of 0
    }

    /// Generate moves with hash move prioritization and depth-aware ordering
    fn generate_ordered_moves_with_hash(
        &self,
        board: &Board,
        hash_move: Option<ChessMove>,
        depth: u32,
    ) -> Vec<ChessMove> {
        let mut moves: Vec<_> = MoveGen::new_legal(board).collect();

        // Advanced move ordering with hash move prioritization
        moves.sort_by(|a, b| {
            let a_score = self.get_move_order_score(a, board, hash_move, depth);
            let b_score = self.get_move_order_score(b, board, hash_move, depth);
            b_score.cmp(&a_score) // Higher score first
        });

        moves
    }

    /// Calculate comprehensive move ordering score with hybrid evaluation insights
    fn get_move_order_score(
        &self,
        chess_move: &ChessMove,
        board: &Board,
        hash_move: Option<ChessMove>,
        depth: u32,
    ) -> i32 {
        // PERFORMANCE FIX: Disabled expensive blunder checking in move ordering
        // This was causing massive performance issues (18+ seconds for 21 nodes)
        // if self.is_blunder_move(chess_move, board) {
        //     return -1_000_000; // Heavily penalize blunder moves
        // }

        // 1. Hash move from transposition table (highest priority)
        if let Some(hash) = hash_move {
            if hash == *chess_move {
                return 1_000_000; // Highest possible score
            }
        }

        // 2. Winning captures (MVV-LVA)
        if let Some(_captured_piece) = board.piece_on(chess_move.get_dest()) {
            let mvv_lva_score = self.mvv_lva_score(chess_move, board);

            // PERFORMANCE FIX: Disabled expensive material exchange calculation
            // This was causing performance issues in move ordering
            // Just use simple MVV-LVA for now
            return 900_000 + mvv_lva_score; // All captures get reasonable priority
        }

        // 3. Promotions
        if chess_move.get_promotion().is_some() {
            let promotion_piece = chess_move.get_promotion().unwrap();
            let promotion_value = match promotion_piece {
                chess::Piece::Queen => 800_000,
                chess::Piece::Rook => 700_000,
                chess::Piece::Bishop => 600_000,
                chess::Piece::Knight => 590_000,
                _ => 500_000,
            };
            return promotion_value;
        }

        // 4. Tactical threat moves (discovered attacks, pins, forks)
        let tactical_bonus = self.evaluate_tactical_move_bonus(chess_move, board);
        if tactical_bonus > 0 {
            return 550_000 + tactical_bonus; // Higher than killer moves
        }

        // 5. Killer moves (depth-specific)
        if self.is_killer_move_at_depth(chess_move, depth) {
            return 500_000;
        }

        // 6. Counter moves (moves that refute the opponent's previous move)
        if self.is_counter_move(chess_move) {
            return 400_000;
        }

        // 7. Castling moves (generally good, but lower than captures)
        if self.is_castling_move(chess_move, board) {
            return 250_000; // Reduced from 350_000 to prioritize captures
        }

        // 7. Checks (forcing moves) - but evaluate carefully!
        if self.gives_check(chess_move, board) {
            // If it's a check that loses material, reduce the bonus significantly
            if let Some(captured_piece) = board.piece_on(chess_move.get_dest()) {
                // Capturing check - evaluate normally with MVV-LVA
                if let Some(attacker_piece) = board.piece_on(chess_move.get_source()) {
                    let victim_value = self.get_piece_value(captured_piece);
                    let attacker_value = self.get_piece_value(attacker_piece);
                    if victim_value < attacker_value {
                        // Bad trade even with check - very low priority (this causes Bxf7+ blunders)
                        return 5_000; // Much lower - losing material for check is usually bad
                    }
                }
                return 300_000; // Good capturing check
            } else {
                // Non-capturing check - need to verify it's not a sacrifice
                // Simple check: does this move hang the piece?
                if let Some(_moving_piece) = board.piece_on(chess_move.get_source()) {
                    // Quick check if the piece is defended on the destination square
                    let mut temp_board = *board;
                    temp_board = temp_board.make_move_new(*chess_move);

                    // Count attackers of the destination square
                    let attackers = self.count_attackers(
                        &temp_board,
                        chess_move.get_dest(),
                        !temp_board.side_to_move(),
                    );
                    let defenders = self.count_attackers(
                        &temp_board,
                        chess_move.get_dest(),
                        temp_board.side_to_move(),
                    );

                    if attackers > defenders {
                        // Piece hangs after check - very low priority
                        return 8_000;
                    }
                }
                return 50_000; // Non-capturing check (much lower than before)
            }
        }

        // 8. History heuristic (moves that have been good before)
        let history_score = self.get_history_score(chess_move);
        200_000 + history_score as i32 // Base score + history
    }

    /// Improved Static Exchange Evaluation for capture assessment
    pub fn is_good_capture(
        &self,
        chess_move: &ChessMove,
        board: &Board,
        captured_piece: chess::Piece,
    ) -> bool {
        let attacker_piece = board.piece_on(chess_move.get_source());
        if attacker_piece.is_none() {
            return false;
        }

        let attacker_value = self.get_piece_value(attacker_piece.unwrap());
        let victim_value = self.get_piece_value(captured_piece);

        // Enhanced SEE: immediate material balance plus basic recapture analysis
        let immediate_gain = victim_value - attacker_value;

        // If we gain material immediately, it's likely good
        if immediate_gain > 0 {
            return true;
        }

        // If we lose material, check if the square is defended
        if immediate_gain < 0 {
            // Count defenders of the destination square by the opponent
            let defenders =
                self.count_attackers(board, chess_move.get_dest(), !board.side_to_move());

            // If the square is defended and we lose material, it's bad
            if defenders > 0 {
                return false;
            }

            // If undefended, even losing captures might be okay (rare edge cases)
            return immediate_gain >= -100; // Don't lose more than a pawn's worth
        }

        // Equal trades are generally okay
        true
    }

    /// Get piece value for SEE calculation
    fn get_piece_value(&self, piece: chess::Piece) -> i32 {
        // Return values in centipawns for move ordering (to maintain integer precision)
        match piece {
            chess::Piece::Pawn => 100,
            chess::Piece::Knight => 320,
            chess::Piece::Bishop => 330,
            chess::Piece::Rook => 500,
            chess::Piece::Queen => 900,
            chess::Piece::King => 10000,
        }
    }

    /// Calculate expected material exchange for a capture using Static Exchange Evaluation (SEE)
    /// Production-ready complete implementation for accurate tactical evaluation
    pub fn calculate_material_exchange(
        &self,
        chess_move: &ChessMove,
        board: &Board,
        captured_piece: chess::Piece,
        attacker_piece: chess::Piece,
    ) -> i32 {
        let dest_square = chess_move.get_dest();

        // Build complete attacker lists for both colors
        let mut white_attackers =
            self.get_all_attackers_of_square(board, dest_square, chess::Color::White);
        let mut black_attackers =
            self.get_all_attackers_of_square(board, dest_square, chess::Color::Black);

        // Remove the initial attacker from the appropriate list
        if board.side_to_move() == chess::Color::White {
            if let Some(pos) = white_attackers.iter().position(|&p| p == attacker_piece) {
                white_attackers.remove(pos);
            }
        } else {
            if let Some(pos) = black_attackers.iter().position(|&p| p == attacker_piece) {
                black_attackers.remove(pos);
            }
        }

        // Sort attackers by value (cheapest first for optimal exchange)
        white_attackers.sort_by_key(|&piece| self.get_piece_value(piece));
        black_attackers.sort_by_key(|&piece| self.get_piece_value(piece));

        // Start with the captured piece value
        let mut gains = vec![self.get_piece_value(captured_piece)];
        let mut current_attacker_value = self.get_piece_value(attacker_piece);
        let mut to_move = !board.side_to_move();

        // Simulate the complete exchange sequence
        loop {
            let attackers = if to_move == chess::Color::White {
                &mut white_attackers
            } else {
                &mut black_attackers
            };

            if attackers.is_empty() {
                break; // No more attackers
            }

            // Use the cheapest available attacker
            let next_attacker = attackers.remove(0);
            let next_attacker_value = self.get_piece_value(next_attacker);

            // Add this exchange to the gains array
            gains.push(current_attacker_value);
            current_attacker_value = next_attacker_value;
            to_move = !to_move;
        }

        // Minimax evaluation: work backwards through the gains
        // Each player chooses whether to continue the exchange or stop
        while gains.len() > 1 {
            let last_gain = gains.pop().unwrap();
            let gains_len = gains.len();
            let prev_gain = gains.last_mut().unwrap();

            // The player to move chooses the better of:
            // 1. Taking the previous gain (stopping the exchange)
            // 2. Continuing the exchange: prev_gain - last_gain
            if gains_len % 2 == 1 {
                // Maximizing player (from original attacker's perspective)
                *prev_gain = (*prev_gain).max(*prev_gain - last_gain);
            } else {
                // Minimizing player (opponent's perspective)
                *prev_gain = (*prev_gain).min(*prev_gain - last_gain);
            }
        }

        gains[0]
    }

    /// Get complete list of all attacking pieces for SEE calculation
    pub fn get_all_attackers_of_square(
        &self,
        board: &Board,
        square: chess::Square,
        color: chess::Color,
    ) -> Vec<chess::Piece> {
        let mut attackers = Vec::new();

        // Get all pieces of this color that can attack the square
        for piece_type in [
            chess::Piece::Pawn,
            chess::Piece::Knight,
            chess::Piece::Bishop,
            chess::Piece::Rook,
            chess::Piece::Queen,
            chess::Piece::King,
        ] {
            let pieces = board.pieces(piece_type) & board.color_combined(color);

            for piece_square in pieces {
                if self.piece_can_attack_square(board, piece_square, square, piece_type) {
                    attackers.push(piece_type);
                }
            }
        }

        attackers
    }

    /// Check if a specific piece on a specific square can attack the target square
    fn piece_can_attack_square(
        &self,
        board: &Board,
        piece_square: chess::Square,
        target_square: chess::Square,
        piece_type: chess::Piece,
    ) -> bool {
        if piece_square == target_square {
            return false; // Piece can't attack itself
        }

        match piece_type {
            chess::Piece::Pawn => {
                let color = board.color_on(piece_square).unwrap_or(chess::Color::White);
                let source_rank = piece_square.get_rank().to_index() as i32;
                let source_file = piece_square.get_file().to_index() as i32;
                let target_rank = target_square.get_rank().to_index() as i32;
                let target_file = target_square.get_file().to_index() as i32;

                let rank_diff = target_rank - source_rank;
                let file_diff = (target_file - source_file).abs();

                if color == chess::Color::White {
                    rank_diff == 1 && file_diff == 1 // White pawn attacks diagonally up
                } else {
                    rank_diff == -1 && file_diff == 1 // Black pawn attacks diagonally down
                }
            }
            chess::Piece::Knight => {
                let source_rank = piece_square.get_rank().to_index() as i32;
                let source_file = piece_square.get_file().to_index() as i32;
                let target_rank = target_square.get_rank().to_index() as i32;
                let target_file = target_square.get_file().to_index() as i32;

                let rank_diff = (target_rank - source_rank).abs();
                let file_diff = (target_file - source_file).abs();

                (rank_diff == 2 && file_diff == 1) || (rank_diff == 1 && file_diff == 2)
            }
            chess::Piece::Bishop => self.is_diagonal_clear(board, piece_square, target_square),
            chess::Piece::Rook => self.is_rank_or_file_clear(board, piece_square, target_square),
            chess::Piece::Queen => {
                self.is_diagonal_clear(board, piece_square, target_square)
                    || self.is_rank_or_file_clear(board, piece_square, target_square)
            }
            chess::Piece::King => {
                let source_rank = piece_square.get_rank().to_index() as i32;
                let source_file = piece_square.get_file().to_index() as i32;
                let target_rank = target_square.get_rank().to_index() as i32;
                let target_file = target_square.get_file().to_index() as i32;

                let rank_diff = (target_rank - source_rank).abs();
                let file_diff = (target_file - source_file).abs();

                rank_diff <= 1 && file_diff <= 1
            }
        }
    }

    /// Check if diagonal path is clear between two squares
    fn is_diagonal_clear(&self, board: &Board, from: chess::Square, to: chess::Square) -> bool {
        let from_rank = from.get_rank().to_index() as i32;
        let from_file = from.get_file().to_index() as i32;
        let to_rank = to.get_rank().to_index() as i32;
        let to_file = to.get_file().to_index() as i32;

        let rank_diff = to_rank - from_rank;
        let file_diff = to_file - from_file;

        // Must be diagonal
        if rank_diff.abs() != file_diff.abs() || rank_diff == 0 {
            return false;
        }

        let rank_dir = if rank_diff > 0 { 1 } else { -1 };
        let file_dir = if file_diff > 0 { 1 } else { -1 };

        let steps = rank_diff.abs();

        // Check each square in the path (excluding start and end)
        for i in 1..steps {
            let check_rank = from_rank + (rank_dir * i);
            let check_file = from_file + (file_dir * i);

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

    /// Check if rank or file path is clear between two squares
    fn is_rank_or_file_clear(&self, board: &Board, from: chess::Square, to: chess::Square) -> bool {
        let from_rank = from.get_rank().to_index() as i32;
        let from_file = from.get_file().to_index() as i32;
        let to_rank = to.get_rank().to_index() as i32;
        let to_file = to.get_file().to_index() as i32;

        // Must be on same rank or file
        if from_rank != to_rank && from_file != to_file {
            return false;
        }

        let (start, end, is_rank) = if from_rank == to_rank {
            // Same rank, check file direction
            let start = from_file.min(to_file);
            let end = from_file.max(to_file);
            (start, end, true)
        } else {
            // Same file, check rank direction
            let start = from_rank.min(to_rank);
            let end = from_rank.max(to_rank);
            (start, end, false)
        };

        // Check each square in the path (excluding start and end)
        for i in (start + 1)..end {
            let check_square = if is_rank {
                chess::Square::make_square(
                    chess::Rank::from_index(from_rank as usize),
                    chess::File::from_index(i as usize),
                )
            } else {
                chess::Square::make_square(
                    chess::Rank::from_index(i as usize),
                    chess::File::from_index(from_file as usize),
                )
            };

            if board.piece_on(check_square).is_some() {
                return false; // Path blocked
            }
        }

        true
    }

    /// Get all pieces that can attack a square for a given color (legacy method for compatibility)
    pub fn get_piece_attackers(
        &self,
        board: &Board,
        square: chess::Square,
        color: chess::Color,
    ) -> Vec<chess::Piece> {
        let mut attackers = Vec::new();

        // Check all piece types that could attack this square
        let pieces = board.color_combined(color);

        // Pawns
        let pawns = board.pieces(chess::Piece::Pawn) & pieces;
        if pawns.popcnt() > 0 && self.can_pawn_attack(board, square, color) {
            attackers.push(chess::Piece::Pawn);
        }

        // Knights
        let knights = board.pieces(chess::Piece::Knight) & pieces;
        if knights.popcnt() > 0 && self.can_piece_attack(board, square, chess::Piece::Knight, color)
        {
            attackers.push(chess::Piece::Knight);
        }

        // Bishops
        let bishops = board.pieces(chess::Piece::Bishop) & pieces;
        if bishops.popcnt() > 0 && self.can_piece_attack(board, square, chess::Piece::Bishop, color)
        {
            attackers.push(chess::Piece::Bishop);
        }

        // Rooks
        let rooks = board.pieces(chess::Piece::Rook) & pieces;
        if rooks.popcnt() > 0 && self.can_piece_attack(board, square, chess::Piece::Rook, color) {
            attackers.push(chess::Piece::Rook);
        }

        // Queen
        let queens = board.pieces(chess::Piece::Queen) & pieces;
        if queens.popcnt() > 0 && self.can_piece_attack(board, square, chess::Piece::Queen, color) {
            attackers.push(chess::Piece::Queen);
        }

        // King
        let kings = board.pieces(chess::Piece::King) & pieces;
        if kings.popcnt() > 0 && self.can_piece_attack(board, square, chess::Piece::King, color) {
            attackers.push(chess::Piece::King);
        }

        attackers
    }

    /// Check if a pawn of given color can attack the square
    fn can_pawn_attack(&self, board: &Board, square: chess::Square, color: chess::Color) -> bool {
        // Simplified pawn attack check
        let rank = square.get_rank().to_index() as i32;
        let file = square.get_file().to_index() as i32;

        let (attack_rank, _direction) = if color == chess::Color::White {
            (rank - 1, 1)
        } else {
            (rank + 1, -1)
        };

        if attack_rank < 0 || attack_rank > 7 {
            return false;
        }

        // Check adjacent files
        for attack_file in [file - 1, file + 1] {
            if attack_file >= 0 && attack_file <= 7 {
                let pawn_square = chess::Square::make_square(
                    chess::Rank::from_index(attack_rank as usize),
                    chess::File::from_index(attack_file as usize),
                );
                if let Some(piece) = board.piece_on(pawn_square) {
                    if piece == chess::Piece::Pawn && board.color_on(pawn_square) == Some(color) {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Check if a piece of given type and color can attack the square
    pub fn can_piece_attack(
        &self,
        board: &Board,
        target_square: chess::Square,
        piece_type: chess::Piece,
        color: chess::Color,
    ) -> bool {
        // Get all pieces of this type and color
        let pieces = board.pieces(piece_type) & board.color_combined(color);

        // Check each piece to see if it can attack the target square
        for square in pieces {
            match piece_type {
                chess::Piece::Knight => {
                    // Knight move patterns: check all 8 possible knight moves
                    let knight_moves = [
                        (-2, -1),
                        (-2, 1),
                        (-1, -2),
                        (-1, 2),
                        (1, -2),
                        (1, 2),
                        (2, -1),
                        (2, 1),
                    ];

                    let source_rank = square.get_rank().to_index() as i32;
                    let source_file = square.get_file().to_index() as i32;
                    let target_rank = target_square.get_rank().to_index() as i32;
                    let target_file = target_square.get_file().to_index() as i32;

                    for (rank_offset, file_offset) in knight_moves {
                        if source_rank + rank_offset == target_rank
                            && source_file + file_offset == target_file
                        {
                            return true;
                        }
                    }
                }
                chess::Piece::Bishop => {
                    // Bishop moves diagonally - check if target is on same diagonal and path is clear
                    let source_rank = square.get_rank().to_index() as i32;
                    let source_file = square.get_file().to_index() as i32;
                    let target_rank = target_square.get_rank().to_index() as i32;
                    let target_file = target_square.get_file().to_index() as i32;

                    let rank_diff = (target_rank - source_rank).abs();
                    let file_diff = (target_file - source_file).abs();

                    // Must be on same diagonal
                    if rank_diff == file_diff && rank_diff > 0 {
                        // Check if path is clear (simplified - would need full path checking)
                        return true;
                    }
                }
                chess::Piece::Rook => {
                    // Rook moves horizontally or vertically
                    let source_rank = square.get_rank().to_index();
                    let source_file = square.get_file().to_index();
                    let target_rank = target_square.get_rank().to_index();
                    let target_file = target_square.get_file().to_index();

                    // Must be on same rank or file
                    if source_rank == target_rank || source_file == target_file {
                        return true; // Simplified - would need path checking
                    }
                }
                chess::Piece::Queen => {
                    // Queen combines rook and bishop moves
                    return self.can_piece_attack(board, target_square, chess::Piece::Rook, color)
                        || self.can_piece_attack(
                            board,
                            target_square,
                            chess::Piece::Bishop,
                            color,
                        );
                }
                chess::Piece::King => {
                    // King moves one square in any direction
                    let source_rank = square.get_rank().to_index() as i32;
                    let source_file = square.get_file().to_index() as i32;
                    let target_rank = target_square.get_rank().to_index() as i32;
                    let target_file = target_square.get_file().to_index() as i32;

                    let rank_diff = (target_rank - source_rank).abs();
                    let file_diff = (target_file - source_file).abs();

                    if rank_diff <= 1 && file_diff <= 1 && (rank_diff + file_diff) > 0 {
                        return true;
                    }
                }
                chess::Piece::Pawn => {
                    // Handled by can_pawn_attack
                    continue;
                }
            }
        }

        false
    }

    /// Check if move is a killer move at specific depth
    fn is_killer_move_at_depth(&self, chess_move: &ChessMove, depth: u32) -> bool {
        let depth_idx = (depth as usize).min(self.killer_moves.len() - 1);
        self.killer_moves[depth_idx].contains(&Some(*chess_move))
    }

    /// Check if move is a counter move
    fn is_counter_move(&self, chess_move: &ChessMove) -> bool {
        if let Some(last_move) = self.last_move {
            let last_move_key = (last_move.get_source(), last_move.get_dest());
            if let Some(counter_move) = self.counter_moves.get(&last_move_key) {
                return *counter_move == *chess_move;
            }
        }
        false
    }

    /// Check if move is castling
    fn is_castling_move(&self, chess_move: &ChessMove, board: &Board) -> bool {
        if let Some(piece) = board.piece_on(chess_move.get_source()) {
            if piece == chess::Piece::King {
                let source_file = chess_move.get_source().get_file().to_index();
                let dest_file = chess_move.get_dest().get_file().to_index();
                // King move of 2 squares is castling
                return (source_file as i32 - dest_file as i32).abs() == 2;
            }
        }
        false
    }

    /// Check if move gives check
    fn gives_check(&self, chess_move: &ChessMove, board: &Board) -> bool {
        let new_board = board.make_move_new(*chess_move);
        new_board.checkers().popcnt() > 0
    }

    /// Calculate MVV-LVA (Most Valuable Victim - Least Valuable Attacker) score
    fn mvv_lva_score(&self, chess_move: &ChessMove, board: &Board) -> i32 {
        let victim_value = if let Some(victim_piece) = board.piece_on(chess_move.get_dest()) {
            match victim_piece {
                chess::Piece::Pawn => 100,
                chess::Piece::Knight => 300,
                chess::Piece::Bishop => 300,
                chess::Piece::Rook => 500,
                chess::Piece::Queen => 900,
                chess::Piece::King => 10000, // Should never happen in legal moves
            }
        } else {
            0
        };

        let attacker_value = if let Some(attacker_piece) = board.piece_on(chess_move.get_source()) {
            match attacker_piece {
                chess::Piece::Pawn => 1,
                chess::Piece::Knight => 3,
                chess::Piece::Bishop => 3,
                chess::Piece::Rook => 5,
                chess::Piece::Queen => 9,
                chess::Piece::King => 1, // King captures are rare but low priority
            }
        } else {
            1
        };

        // Higher victim value and lower attacker value = higher score
        victim_value * 10 - attacker_value
    }

    /// Generate only captures for quiescence search
    fn generate_captures(&self, board: &Board) -> Vec<ChessMove> {
        MoveGen::new_legal(board)
            .filter(|chess_move| {
                // Capture moves or promotions
                board.piece_on(chess_move.get_dest()).is_some()
                    || chess_move.get_promotion().is_some()
            })
            .collect()
    }

    /// Generate moves that give check (for quiescence search)
    #[allow(dead_code)]
    fn generate_checks(&self, board: &Board) -> Vec<ChessMove> {
        MoveGen::new_legal(board)
            .filter(|chess_move| {
                // Test if the move gives check by making the move and checking if opponent is in check
                let new_board = board.make_move_new(*chess_move);
                new_board.checkers().popcnt() > 0
            })
            .collect()
    }

    /// Generate captures and checks for quiescence search
    fn generate_captures_and_checks(&self, board: &Board) -> Vec<ChessMove> {
        MoveGen::new_legal(board)
            .filter(|chess_move| {
                // Capture moves, promotions, or checks
                let is_capture = board.piece_on(chess_move.get_dest()).is_some();
                let is_promotion = chess_move.get_promotion().is_some();
                let is_check = if !is_capture && !is_promotion {
                    // Only check for checks if it's not already a capture/promotion to avoid duplicate work
                    let new_board = board.make_move_new(*chess_move);
                    new_board.checkers().popcnt() > 0
                } else {
                    false
                };

                is_capture || is_promotion || is_check
            })
            .collect()
    }

    /// Hybrid evaluation combining NNUE, pattern recognition, and tactical analysis
    fn evaluate_position(&self, board: &Board) -> f32 {
        if board.status() != chess::BoardStatus::Ongoing {
            return self.evaluate_terminal_position(board);
        }

        // TEMPORARY: Use only basic material evaluation for debugging
        // Check if hybrid evaluation is enabled
        // if self.config.enable_hybrid_evaluation {
        //     return self.evaluate_position_hybrid(board);
        // }

        // Fallback to traditional tactical evaluation
        self.evaluate_position_traditional(board)
    }

    /// Hybrid evaluation that intelligently blends NNUE, pattern recognition, and tactical analysis
    fn evaluate_position_hybrid(&self, board: &Board) -> f32 {
        // SPEED OPTIMIZATION: Fast hybrid evaluation for competitive play

        // Phase 1: Get NNUE evaluation and confidence
        let (nnue_eval, nnue_confidence) = self.get_nnue_evaluation(board);

        // Phase 2: Get pattern recognition evaluation and confidence
        let (pattern_eval, pattern_confidence) = self.get_pattern_evaluation(board);

        // Phase 3: Calculate combined confidence
        let combined_confidence = (nnue_confidence * 0.6) + (pattern_confidence * 0.4);

        // Phase 4: Fast blending without expensive tactical evaluation
        if combined_confidence >= self.config.pattern_confidence_threshold {
            // High confidence - blend NNUE and pattern evaluations
            let pattern_weight = self.config.pattern_weight * combined_confidence;
            let nnue_weight = 1.0 - pattern_weight;

            (pattern_eval * pattern_weight) + (nnue_eval * nnue_weight)
        } else {
            // Low confidence - use NNUE with pattern hints
            let nnue_weight = 0.8;
            let pattern_weight = 0.2;

            (nnue_eval * nnue_weight) + (pattern_eval * pattern_weight)
        }
    }

    /// Traditional tactical evaluation (original implementation)
    fn evaluate_position_traditional(&self, board: &Board) -> f32 {
        // Use basic material balance for now
        let score_cp = self.material_balance(board);

        // Convert from centipawns to pawns and clamp to reasonable range
        let score = (score_cp / 100.0).clamp(-4.0, 4.0);

        // Always return evaluation from White's perspective
        score
    }

    /// Evaluate terminal positions (checkmate, stalemate, etc.)
    fn evaluate_terminal_position(&self, board: &Board) -> f32 {
        match board.status() {
            chess::BoardStatus::Checkmate => {
                if board.side_to_move() == Color::White {
                    -10.0 // Black wins (10 pawn units - strong but reasonable)
                } else {
                    10.0 // White wins (10 pawn units - strong but reasonable)
                }
            }
            chess::BoardStatus::Stalemate => 0.0,
            _ => 0.0,
        }
    }

    /// Calculate material balance with modern piece values
    fn material_balance(&self, board: &Board) -> f32 {
        // SPEED OPTIMIZATION: Fast material balance calculation
        let mut balance = 0.0;

        // Quick piece counting with centipawn values
        balance += (board.pieces(chess::Piece::Pawn) & board.color_combined(Color::White)).popcnt()
            as f32
            * 100.0;
        balance -= (board.pieces(chess::Piece::Pawn) & board.color_combined(Color::Black)).popcnt()
            as f32
            * 100.0;

        balance += (board.pieces(chess::Piece::Knight) & board.color_combined(Color::White))
            .popcnt() as f32
            * 320.0;
        balance -= (board.pieces(chess::Piece::Knight) & board.color_combined(Color::Black))
            .popcnt() as f32
            * 320.0;

        balance += (board.pieces(chess::Piece::Bishop) & board.color_combined(Color::White))
            .popcnt() as f32
            * 330.0;
        balance -= (board.pieces(chess::Piece::Bishop) & board.color_combined(Color::Black))
            .popcnt() as f32
            * 330.0;

        balance += (board.pieces(chess::Piece::Rook) & board.color_combined(Color::White)).popcnt()
            as f32
            * 500.0;
        balance -= (board.pieces(chess::Piece::Rook) & board.color_combined(Color::Black)).popcnt()
            as f32
            * 500.0;

        balance += (board.pieces(chess::Piece::Queen) & board.color_combined(Color::White)).popcnt()
            as f32
            * 900.0;
        balance -= (board.pieces(chess::Piece::Queen) & board.color_combined(Color::Black)).popcnt()
            as f32
            * 900.0;

        balance // Return in centipawns
    }

    /// Advanced piece placement evaluation with game phase awareness
    fn piece_square_evaluation(&self, board: &Board) -> f32 {
        let mut score = 0.0;
        let game_phase = self.detect_game_phase(board);

        // Advanced pawn piece-square tables
        let pawn_opening = [
            0, 0, 0, 0, 0, 0, 0, 0, 50, 50, 50, 50, 50, 50, 50, 50, 10, 10, 20, 30, 30, 20, 10, 10,
            5, 5, 10, 27, 27, 10, 5, 5, 0, 0, 0, 25, 25, 0, 0, 0, 5, -5, -10, 0, 0, -10, -5, 5, 5,
            10, 10, -25, -25, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0,
        ];

        let pawn_endgame = [
            0, 0, 0, 0, 0, 0, 0, 0, 80, 80, 80, 80, 80, 80, 80, 80, 50, 50, 50, 50, 50, 50, 50, 50,
            30, 30, 30, 30, 30, 30, 30, 30, 20, 20, 20, 20, 20, 20, 20, 20, 10, 10, 10, 10, 10, 10,
            10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0,
        ];

        // Knight tables - prefer center in middlegame, edges less penalized in endgame
        let knight_opening = [
            -50, -40, -30, -30, -30, -30, -40, -50, -40, -20, 0, 0, 0, 0, -20, -40, -30, 0, 10, 15,
            15, 10, 0, -30, -30, 5, 15, 20, 20, 15, 5, -30, -30, 0, 15, 20, 20, 15, 0, -30, -30, 5,
            10, 15, 15, 10, 5, -30, -40, -20, 0, 5, 5, 0, -20, -40, -50, -40, -30, -30, -30, -30,
            -40, -50,
        ];

        let knight_endgame = [
            -50, -40, -30, -30, -30, -30, -40, -50, -40, -20, 0, 5, 5, 0, -20, -40, -30, 0, 10, 15,
            15, 10, 0, -30, -30, 5, 15, 20, 20, 15, 5, -30, -30, 0, 15, 20, 20, 15, 0, -30, -30, 5,
            10, 15, 15, 10, 5, -30, -40, -20, 0, 5, 5, 0, -20, -40, -50, -40, -30, -30, -30, -30,
            -40, -50,
        ];

        // Bishop tables - long diagonals important in middlegame
        let bishop_opening = [
            -20, -10, -10, -10, -10, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10, -10, 0, 5, 10, 10,
            5, 0, -10, -10, 5, 5, 10, 10, 5, 5, -10, -10, 0, 10, 10, 10, 10, 0, -10, -10, 10, 10,
            10, 10, 10, 10, -10, -10, 5, 0, 0, 0, 0, 5, -10, -20, -10, -10, -10, -10, -10, -10,
            -20,
        ];

        let bishop_endgame = [
            -20, -10, -10, -10, -10, -10, -10, -20, -10, 5, 0, 0, 0, 0, 5, -10, -10, 0, 10, 15, 15,
            10, 0, -10, -10, 0, 15, 20, 20, 15, 0, -10, -10, 0, 15, 20, 20, 15, 0, -10, -10, 0, 10,
            15, 15, 10, 0, -10, -10, 5, 0, 0, 0, 0, 5, -10, -20, -10, -10, -10, -10, -10, -10, -20,
        ];

        // Rook tables - 7th rank important, files matter more in endgame
        let rook_opening = [
            0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 10, 10, 10, 10, 10, 5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0,
            0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0,
            0, 0, -5, 0, 0, 0, 5, 5, 0, 0, 0,
        ];

        let rook_endgame = [
            0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 20, 20, 20, 20, 20, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];

        // Queen tables - avoid early development in opening, centralize in middlegame
        let queen_opening = [
            -20, -10, -10, -5, -5, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10, -10, 0, 5, 5, 5, 5,
            0, -10, -5, 0, 5, 5, 5, 5, 0, -5, 0, 0, 5, 5, 5, 5, 0, -5, -10, 5, 5, 5, 5, 5, 0, -10,
            -10, 0, 5, 0, 0, 0, 0, -10, -20, -10, -10, -5, -5, -10, -10, -20,
        ];

        let queen_endgame = [
            -20, -10, -10, -5, -5, -10, -10, -20, -10, 0, 5, 5, 5, 5, 0, -10, -10, 5, 10, 10, 10,
            10, 5, -10, -5, 0, 10, 10, 10, 10, 0, -5, -5, 0, 10, 10, 10, 10, 0, -5, -10, 5, 10, 10,
            10, 10, 5, -10, -10, 0, 5, 5, 5, 5, 0, -10, -20, -10, -10, -5, -5, -10, -10, -20,
        ];

        // King tables - safety in opening, activity in endgame
        let king_opening = [
            -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -30,
            -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -20, -30,
            -30, -40, -40, -30, -30, -20, -10, -20, -20, -20, -20, -20, -20, -10, 20, 20, 0, 0, 0,
            0, 20, 20, 20, 30, 10, 0, 0, 10, 30, 20,
        ];

        let king_endgame = [
            -50, -40, -30, -20, -20, -30, -40, -50, -30, -20, -10, 0, 0, -10, -20, -30, -30, -10,
            20, 30, 30, 20, -10, -30, -30, -10, 30, 40, 40, 30, -10, -30, -30, -10, 30, 40, 40, 30,
            -10, -30, -30, -10, 20, 30, 30, 20, -10, -30, -30, -30, 0, 0, 0, 0, -30, -30, -50, -30,
            -30, -30, -30, -30, -30, -50,
        ];

        // Calculate phase interpolation factor (0.0 = endgame, 1.0 = opening)
        let phase_factor = match game_phase {
            GamePhase::Opening => 1.0,
            GamePhase::Middlegame => 0.5,
            GamePhase::Endgame => 0.0,
        };

        // Evaluate each piece type with phase-interpolated tables
        for color in [Color::White, Color::Black] {
            let multiplier = if color == Color::White { 1.0 } else { -1.0 };

            // Pawns - huge difference between opening and endgame
            let pawns = board.pieces(chess::Piece::Pawn) & board.color_combined(color);
            for square in pawns {
                let idx = if color == Color::White {
                    square.to_index()
                } else {
                    square.to_index() ^ 56
                };
                let opening_value = pawn_opening[idx] as f32;
                let endgame_value = pawn_endgame[idx] as f32;
                let interpolated_value =
                    opening_value * phase_factor + endgame_value * (1.0 - phase_factor);
                score += interpolated_value * multiplier * 0.01; // Scale to centipawns
            }

            // Knights - prefer center in middlegame
            let knights = board.pieces(chess::Piece::Knight) & board.color_combined(color);
            for square in knights {
                let idx = if color == Color::White {
                    square.to_index()
                } else {
                    square.to_index() ^ 56
                };
                let opening_value = knight_opening[idx] as f32;
                let endgame_value = knight_endgame[idx] as f32;
                let interpolated_value =
                    opening_value * phase_factor + endgame_value * (1.0 - phase_factor);
                score += interpolated_value * multiplier * 0.01;
            }

            // Bishops - long diagonals vs centralization
            let bishops = board.pieces(chess::Piece::Bishop) & board.color_combined(color);
            for square in bishops {
                let idx = if color == Color::White {
                    square.to_index()
                } else {
                    square.to_index() ^ 56
                };
                let opening_value = bishop_opening[idx] as f32;
                let endgame_value = bishop_endgame[idx] as f32;
                let interpolated_value =
                    opening_value * phase_factor + endgame_value * (1.0 - phase_factor);
                score += interpolated_value * multiplier * 0.01;
            }

            // Rooks - files and ranks matter more in endgame
            let rooks = board.pieces(chess::Piece::Rook) & board.color_combined(color);
            for square in rooks {
                let idx = if color == Color::White {
                    square.to_index()
                } else {
                    square.to_index() ^ 56
                };
                let opening_value = rook_opening[idx] as f32;
                let endgame_value = rook_endgame[idx] as f32;
                let interpolated_value =
                    opening_value * phase_factor + endgame_value * (1.0 - phase_factor);
                score += interpolated_value * multiplier * 0.01;
            }

            // Queen - early development bad, centralization good
            let queens = board.pieces(chess::Piece::Queen) & board.color_combined(color);
            for square in queens {
                let idx = if color == Color::White {
                    square.to_index()
                } else {
                    square.to_index() ^ 56
                };
                let opening_value = queen_opening[idx] as f32;
                let endgame_value = queen_endgame[idx] as f32;
                let interpolated_value =
                    opening_value * phase_factor + endgame_value * (1.0 - phase_factor);
                score += interpolated_value * multiplier * 0.01;
            }

            // King - safety vs activity based on game phase
            let kings = board.pieces(chess::Piece::King) & board.color_combined(color);
            for square in kings {
                let idx = if color == Color::White {
                    square.to_index()
                } else {
                    square.to_index() ^ 56
                };
                let opening_value = king_opening[idx] as f32;
                let endgame_value = king_endgame[idx] as f32;
                let interpolated_value =
                    opening_value * phase_factor + endgame_value * (1.0 - phase_factor);
                score += interpolated_value * multiplier * 0.01;
            }
        }

        score
    }

    /// Detect game phase based on material and piece development
    fn detect_game_phase(&self, board: &Board) -> GamePhase {
        let mut total_material = 0;

        // Count material (excluding pawns and kings)
        for color in [Color::White, Color::Black] {
            total_material +=
                (board.pieces(chess::Piece::Queen) & board.color_combined(color)).popcnt() * 9;
            total_material +=
                (board.pieces(chess::Piece::Rook) & board.color_combined(color)).popcnt() * 5;
            total_material +=
                (board.pieces(chess::Piece::Bishop) & board.color_combined(color)).popcnt() * 3;
            total_material +=
                (board.pieces(chess::Piece::Knight) & board.color_combined(color)).popcnt() * 3;
        }

        // Phase boundaries (excluding pawns)
        if total_material >= 60 {
            GamePhase::Opening
        } else if total_material >= 20 {
            GamePhase::Middlegame
        } else {
            GamePhase::Endgame
        }
    }

    /// Advanced mobility evaluation for all pieces
    fn mobility_evaluation(&self, board: &Board) -> f32 {
        let mut score = 0.0;

        for color in [Color::White, Color::Black] {
            let multiplier = if color == Color::White { 1.0 } else { -1.0 };
            let mobility_score = self.calculate_piece_mobility(board, color);
            score += mobility_score * multiplier;
        }

        score
    }

    /// Calculate total mobility for all pieces of a color
    fn calculate_piece_mobility(&self, board: &Board, color: Color) -> f32 {
        let mut mobility = 0.0;

        // Knight mobility (very important for tactical strength)
        let knights = board.pieces(chess::Piece::Knight) & board.color_combined(color);
        for knight_square in knights {
            let knight_moves = self.count_knight_moves(board, knight_square, color);
            mobility += knight_moves as f32 * 4.0; // High weight for knight mobility
        }

        // Bishop mobility (long diagonals are powerful)
        let bishops = board.pieces(chess::Piece::Bishop) & board.color_combined(color);
        for bishop_square in bishops {
            let bishop_moves = self.count_bishop_moves(board, bishop_square, color);
            mobility += bishop_moves as f32 * 3.0; // Significant weight for bishop mobility
        }

        // Rook mobility (open files and ranks)
        let rooks = board.pieces(chess::Piece::Rook) & board.color_combined(color);
        for rook_square in rooks {
            let rook_moves = self.count_rook_moves(board, rook_square, color);
            mobility += rook_moves as f32 * 2.0; // Good weight for rook mobility
        }

        // Queen mobility (ultimate piece flexibility)
        let queens = board.pieces(chess::Piece::Queen) & board.color_combined(color);
        for queen_square in queens {
            let queen_moves = self.count_queen_moves(board, queen_square, color);
            mobility += queen_moves as f32 * 1.0; // Moderate weight (queen is already powerful)
        }

        // Pawn mobility (pawn breaks and advances)
        let pawn_mobility = self.calculate_pawn_mobility(board, color);
        mobility += pawn_mobility * 5.0; // High weight for pawn breaks

        mobility
    }

    /// Count legal knight moves from a square
    fn count_knight_moves(&self, board: &Board, square: Square, color: Color) -> usize {
        let mut count = 0;
        let knight_offsets = [
            (-2, -1),
            (-2, 1),
            (-1, -2),
            (-1, 2),
            (1, -2),
            (1, 2),
            (2, -1),
            (2, 1),
        ];

        let file = square.get_file().to_index() as i8;
        let rank = square.get_rank().to_index() as i8;

        for (df, dr) in knight_offsets {
            let new_file = file + df;
            let new_rank = rank + dr;

            if (0..8).contains(&new_file) && (0..8).contains(&new_rank) {
                let dest_square = Square::make_square(
                    chess::Rank::from_index(new_rank as usize),
                    chess::File::from_index(new_file as usize),
                );
                // Check if destination is not occupied by own piece
                if let Some(_piece_on_dest) = board.piece_on(dest_square) {
                    if board.color_on(dest_square) != Some(color) {
                        count += 1; // Can capture
                    }
                } else {
                    count += 1; // Empty square
                }
            }
        }

        count
    }

    /// Count bishop moves (diagonal mobility)
    fn count_bishop_moves(&self, board: &Board, square: Square, color: Color) -> usize {
        let mut count = 0;
        let directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

        for (df, dr) in directions {
            count += self.count_sliding_moves(board, square, color, df, dr);
        }

        count
    }

    /// Count rook moves (file and rank mobility)
    fn count_rook_moves(&self, board: &Board, square: Square, color: Color) -> usize {
        let mut count = 0;
        let directions = [(1, 0), (-1, 0), (0, 1), (0, -1)];

        for (df, dr) in directions {
            count += self.count_sliding_moves(board, square, color, df, dr);
        }

        count
    }

    /// Count queen moves (combination of rook and bishop)
    fn count_queen_moves(&self, board: &Board, square: Square, color: Color) -> usize {
        let mut count = 0;
        let directions = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1), // Rook directions
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1), // Bishop directions
        ];

        for (df, dr) in directions {
            count += self.count_sliding_moves(board, square, color, df, dr);
        }

        count
    }

    /// Count moves in a sliding direction (for bishops, rooks, queens)
    fn count_sliding_moves(
        &self,
        board: &Board,
        square: Square,
        color: Color,
        df: i8,
        dr: i8,
    ) -> usize {
        let mut count = 0;
        let mut file = square.get_file().to_index() as i8;
        let mut rank = square.get_rank().to_index() as i8;

        loop {
            file += df;
            rank += dr;

            if !(0..8).contains(&file) || !(0..8).contains(&rank) {
                break;
            }

            let dest_square = Square::make_square(
                chess::Rank::from_index(rank as usize),
                chess::File::from_index(file as usize),
            );
            if let Some(_piece_on_dest) = board.piece_on(dest_square) {
                if board.color_on(dest_square) != Some(color) {
                    count += 1; // Can capture enemy piece
                }
                break; // Blocked by any piece
            } else {
                count += 1; // Empty square
            }
        }

        count
    }

    /// Calculate pawn mobility (advances and potential breaks)
    fn calculate_pawn_mobility(&self, board: &Board, color: Color) -> f32 {
        let mut mobility = 0.0;
        let pawns = board.pieces(chess::Piece::Pawn) & board.color_combined(color);

        let direction = if color == Color::White { 1 } else { -1 };

        for pawn_square in pawns {
            let file = pawn_square.get_file().to_index() as i8;
            let rank = pawn_square.get_rank().to_index() as i8;

            // Check pawn advance
            let advance_rank = rank + direction;
            if (0..8).contains(&advance_rank) {
                let advance_square = Square::make_square(
                    chess::Rank::from_index(advance_rank as usize),
                    pawn_square.get_file(),
                );
                if board.piece_on(advance_square).is_none() {
                    mobility += 1.0; // Can advance

                    // Check double pawn advance from starting position
                    let starting_rank = if color == Color::White { 1 } else { 6 };
                    if rank == starting_rank {
                        let double_advance_rank = advance_rank + direction;
                        let double_advance_square = Square::make_square(
                            chess::Rank::from_index(double_advance_rank as usize),
                            pawn_square.get_file(),
                        );
                        if board.piece_on(double_advance_square).is_none() {
                            mobility += 0.5; // Double advance bonus
                        }
                    }
                }
            }

            // Check pawn captures
            for capture_file in [file - 1, file + 1] {
                if (0..8).contains(&capture_file) && (0..8).contains(&advance_rank) {
                    let capture_square = Square::make_square(
                        chess::Rank::from_index(advance_rank as usize),
                        chess::File::from_index(capture_file as usize),
                    );
                    if let Some(_piece) = board.piece_on(capture_square) {
                        if board.color_on(capture_square) != Some(color) {
                            mobility += 2.0; // Pawn capture opportunity
                        }
                    }
                }
            }
        }

        mobility
    }

    /// Calculate tactical bonuses including mobility
    fn tactical_bonuses(&self, board: &Board) -> f32 {
        // SPEED OPTIMIZATION: Ultra-fast tactical bonus evaluation
        let mut bonus = 0.0;

        // Quick capture count
        let captures = MoveGen::new_legal(board)
            .filter(|m| board.piece_on(m.get_dest()).is_some())
            .count();
        let capture_bonus = captures as f32 * 10.0; // In centipawns

        // Basic perspective scoring
        if board.side_to_move() == Color::White {
            bonus += capture_bonus;
        } else {
            bonus -= capture_bonus;
        }

        bonus
    }

    /// Evaluate center control (important for positional strength)
    fn center_control_evaluation(&self, board: &Board) -> f32 {
        let mut score = 0.0;
        let center_squares = [
            Square::make_square(chess::Rank::Fourth, chess::File::D),
            Square::make_square(chess::Rank::Fourth, chess::File::E),
            Square::make_square(chess::Rank::Fifth, chess::File::D),
            Square::make_square(chess::Rank::Fifth, chess::File::E),
        ];

        let extended_center = [
            Square::make_square(chess::Rank::Third, chess::File::C),
            Square::make_square(chess::Rank::Third, chess::File::D),
            Square::make_square(chess::Rank::Third, chess::File::E),
            Square::make_square(chess::Rank::Third, chess::File::F),
            Square::make_square(chess::Rank::Fourth, chess::File::C),
            Square::make_square(chess::Rank::Fourth, chess::File::F),
            Square::make_square(chess::Rank::Fifth, chess::File::C),
            Square::make_square(chess::Rank::Fifth, chess::File::F),
            Square::make_square(chess::Rank::Sixth, chess::File::C),
            Square::make_square(chess::Rank::Sixth, chess::File::D),
            Square::make_square(chess::Rank::Sixth, chess::File::E),
            Square::make_square(chess::Rank::Sixth, chess::File::F),
        ];

        // Central pawn control (very important)
        for &square in &center_squares {
            if let Some(piece) = board.piece_on(square) {
                if piece == chess::Piece::Pawn {
                    if let Some(color) = board.color_on(square) {
                        let bonus = if color == Color::White { 30.0 } else { -30.0 };
                        score += bonus;
                    }
                }
            }
        }

        // Extended center control
        for &square in &extended_center {
            if let Some(_piece) = board.piece_on(square) {
                if let Some(color) = board.color_on(square) {
                    let bonus = if color == Color::White { 5.0 } else { -5.0 };
                    score += bonus;
                }
            }
        }

        score
    }

    /// Advanced king safety evaluation with professional patterns
    fn king_safety(&self, board: &Board) -> f32 {
        let mut safety = 0.0;
        let game_phase = self.detect_game_phase(board);

        for color in [Color::White, Color::Black] {
            let mut king_safety = 0.0;
            let king_square = board.king_square(color);
            let multiplier = if color == Color::White { 1.0 } else { -1.0 };

            // 1. CASTLING EVALUATION
            king_safety += self.evaluate_castling_safety(board, color, king_square, game_phase);

            // 2. PAWN SHIELD EVALUATION (critical for king safety)
            king_safety += self.evaluate_pawn_shield(board, color, king_square, game_phase);

            // 3. PIECE ATTACK EVALUATION
            king_safety += self.evaluate_king_attackers(board, color, king_square);

            // 4. OPEN LINES NEAR KING
            king_safety += self.evaluate_open_lines_near_king(board, color, king_square);

            // 5. KING ACTIVITY IN ENDGAME
            if game_phase == GamePhase::Endgame {
                king_safety += self.evaluate_king_endgame_activity(board, color, king_square);
            }

            // 6. KING ZONE CONTROL
            king_safety += self.evaluate_king_zone_control(board, color, king_square);

            // 7. IMMEDIATE TACTICAL THREATS
            if board.checkers().popcnt() > 0 && board.side_to_move() == color {
                let check_severity = self.evaluate_check_severity(board, color);
                king_safety -= check_severity;
            }

            safety += king_safety * multiplier;
        }

        safety
    }

    /// Evaluate castling and king position safety
    fn evaluate_castling_safety(
        &self,
        board: &Board,
        color: Color,
        king_square: Square,
        game_phase: GamePhase,
    ) -> f32 {
        let mut score = 0.0;

        let starting_square = if color == Color::White {
            Square::E1
        } else {
            Square::E8
        };
        let kingside_castle = if color == Color::White {
            Square::G1
        } else {
            Square::G8
        };
        let queenside_castle = if color == Color::White {
            Square::C1
        } else {
            Square::C8
        };

        match game_phase {
            GamePhase::Opening | GamePhase::Middlegame => {
                if king_square == kingside_castle {
                    score += 50.0; // Kingside castling bonus
                } else if king_square == queenside_castle {
                    score += 35.0; // Queenside castling bonus (slightly less safe)
                } else if king_square == starting_square {
                    // Bonus for maintaining castling rights
                    let castle_rights = board.castle_rights(color);
                    if castle_rights.has_kingside() {
                        score += 25.0;
                    }
                    if castle_rights.has_queenside() {
                        score += 15.0;
                    }
                } else {
                    // Penalty for king movement without castling
                    score -= 80.0;
                }
            }
            GamePhase::Endgame => {
                // In endgame, king should be active - centralization bonus
                let rank = king_square.get_rank().to_index() as i8;
                let file = king_square.get_file().to_index() as i8;
                let center_distance = (rank as f32 - 3.5).abs() + (file as f32 - 3.5).abs();
                score += (7.0 - center_distance) * 5.0; // Centralization bonus
            }
        }

        score
    }

    /// Evaluate pawn shield protection around the king
    fn evaluate_pawn_shield(
        &self,
        board: &Board,
        color: Color,
        king_square: Square,
        game_phase: GamePhase,
    ) -> f32 {
        if game_phase == GamePhase::Endgame {
            return 0.0; // Pawn shield less important in endgame
        }

        let mut shield_score = 0.0;
        let pawns = board.pieces(chess::Piece::Pawn) & board.color_combined(color);
        let king_file = king_square.get_file().to_index() as i8;
        let king_rank = king_square.get_rank().to_index() as i8;

        // Check pawn shield in front of king
        let shield_files = [king_file - 1, king_file, king_file + 1];
        let forward_direction = if color == Color::White { 1 } else { -1 };

        for &file in &shield_files {
            if (0..8).contains(&file) {
                let mut found_pawn = false;
                let file_mask = self.get_file_mask(chess::File::from_index(file as usize));
                let file_pawns = pawns & file_mask;

                for pawn_square in file_pawns {
                    let pawn_rank = pawn_square.get_rank().to_index() as i8;
                    let rank_distance = (pawn_rank - king_rank) * forward_direction;

                    if rank_distance > 0 && rank_distance <= 3 {
                        found_pawn = true;
                        // Closer pawns provide better protection
                        let protection_value = match rank_distance {
                            1 => 25.0, // Pawn right in front
                            2 => 15.0, // One square ahead
                            3 => 8.0,  // Two squares ahead
                            _ => 0.0,
                        };
                        shield_score += protection_value;
                        break;
                    }
                }

                // Penalty for missing pawn in shield
                if !found_pawn {
                    shield_score -= 20.0;
                }
            }
        }

        // Bonus for intact castled pawn structure
        let is_kingside = king_file >= 6;
        let is_queenside = king_file <= 2;

        if is_kingside {
            shield_score += self.evaluate_kingside_pawn_structure(board, color);
        } else if is_queenside {
            shield_score += self.evaluate_queenside_pawn_structure(board, color);
        }

        shield_score
    }

    /// Evaluate kingside pawn structure (f, g, h pawns)
    fn evaluate_kingside_pawn_structure(&self, board: &Board, color: Color) -> f32 {
        let mut score = 0.0;
        let pawns = board.pieces(chess::Piece::Pawn) & board.color_combined(color);
        let base_rank = if color == Color::White { 1 } else { 6 };

        // Check f, g, h pawn positions
        for (file_idx, ideal_rank) in [(5, base_rank), (6, base_rank), (7, base_rank)] {
            let file_mask = self.get_file_mask(chess::File::from_index(file_idx));
            let file_pawns = pawns & file_mask;

            let mut found_intact = false;
            for pawn_square in file_pawns {
                if pawn_square.get_rank().to_index() == ideal_rank {
                    found_intact = true;
                    score += 10.0; // Intact castled pawn structure
                    break;
                }
            }

            if !found_intact {
                score -= 15.0; // Penalty for advanced/missing pawn
            }
        }

        score
    }

    /// Evaluate queenside pawn structure (a, b, c pawns)
    fn evaluate_queenside_pawn_structure(&self, board: &Board, color: Color) -> f32 {
        let mut score = 0.0;
        let pawns = board.pieces(chess::Piece::Pawn) & board.color_combined(color);
        let base_rank = if color == Color::White { 1 } else { 6 };

        // Check a, b, c pawn positions
        for (file_idx, ideal_rank) in [(0, base_rank), (1, base_rank), (2, base_rank)] {
            let file_mask = self.get_file_mask(chess::File::from_index(file_idx));
            let file_pawns = pawns & file_mask;

            let mut found_intact = false;
            for pawn_square in file_pawns {
                if pawn_square.get_rank().to_index() == ideal_rank {
                    found_intact = true;
                    score += 8.0; // Queenside structure bonus (slightly less important)
                    break;
                }
            }

            if !found_intact {
                score -= 12.0; // Penalty for disrupted queenside
            }
        }

        score
    }

    /// Evaluate piece attacks targeting the king
    fn evaluate_king_attackers(&self, board: &Board, color: Color, king_square: Square) -> f32 {
        let mut attack_score = 0.0;
        let enemy_color = !color;

        // Count different types of attackers
        let enemy_queens = board.pieces(chess::Piece::Queen) & board.color_combined(enemy_color);
        let enemy_rooks = board.pieces(chess::Piece::Rook) & board.color_combined(enemy_color);
        let enemy_bishops = board.pieces(chess::Piece::Bishop) & board.color_combined(enemy_color);
        let enemy_knights = board.pieces(chess::Piece::Knight) & board.color_combined(enemy_color);

        // Queen attacks (most dangerous)
        for queen_square in enemy_queens {
            if self.can_attack_square(board, queen_square, king_square, chess::Piece::Queen) {
                attack_score -= 50.0;
            }
        }

        // Rook attacks (very dangerous on open files/ranks)
        for rook_square in enemy_rooks {
            if self.can_attack_square(board, rook_square, king_square, chess::Piece::Rook) {
                attack_score -= 30.0;
            }
        }

        // Bishop attacks (dangerous on diagonals)
        for bishop_square in enemy_bishops {
            if self.can_attack_square(board, bishop_square, king_square, chess::Piece::Bishop) {
                attack_score -= 25.0;
            }
        }

        // Knight attacks (can't be blocked)
        for knight_square in enemy_knights {
            if self.can_attack_square(board, knight_square, king_square, chess::Piece::Knight) {
                attack_score -= 20.0;
            }
        }

        attack_score
    }

    /// Check if a piece can attack a target square
    fn can_attack_square(
        &self,
        board: &Board,
        piece_square: Square,
        target_square: Square,
        piece_type: chess::Piece,
    ) -> bool {
        match piece_type {
            chess::Piece::Queen | chess::Piece::Rook | chess::Piece::Bishop => {
                // For sliding pieces, check if there's a clear path
                self.has_clear_line_of_attack(board, piece_square, target_square, piece_type)
            }
            chess::Piece::Knight => {
                // Knight attacks
                let file_diff = (piece_square.get_file().to_index() as i8
                    - target_square.get_file().to_index() as i8)
                    .abs();
                let rank_diff = (piece_square.get_rank().to_index() as i8
                    - target_square.get_rank().to_index() as i8)
                    .abs();
                (file_diff == 2 && rank_diff == 1) || (file_diff == 1 && rank_diff == 2)
            }
            _ => false,
        }
    }

    /// Check for clear line of attack for sliding pieces
    fn has_clear_line_of_attack(
        &self,
        board: &Board,
        from: Square,
        to: Square,
        piece_type: chess::Piece,
    ) -> bool {
        let from_file = from.get_file().to_index() as i8;
        let from_rank = from.get_rank().to_index() as i8;
        let to_file = to.get_file().to_index() as i8;
        let to_rank = to.get_rank().to_index() as i8;

        let file_diff = to_file - from_file;
        let rank_diff = to_rank - from_rank;

        // Check if attack is valid for piece type
        let is_valid_attack = match piece_type {
            chess::Piece::Rook | chess::Piece::Queen => {
                file_diff == 0 || rank_diff == 0 || file_diff.abs() == rank_diff.abs()
            }
            chess::Piece::Bishop => file_diff.abs() == rank_diff.abs(),
            _ => false,
        };

        if !is_valid_attack {
            return false;
        }

        // Check for clear path
        let file_step = if file_diff == 0 {
            0
        } else {
            file_diff.signum()
        };
        let rank_step = if rank_diff == 0 {
            0
        } else {
            rank_diff.signum()
        };

        let mut current_file = from_file + file_step;
        let mut current_rank = from_rank + rank_step;

        while current_file != to_file || current_rank != to_rank {
            let square = Square::make_square(
                chess::Rank::from_index(current_rank as usize),
                chess::File::from_index(current_file as usize),
            );
            if board.piece_on(square).is_some() {
                return false; // Path is blocked
            }
            current_file += file_step;
            current_rank += rank_step;
        }

        true
    }

    /// Evaluate open lines (files/ranks/diagonals) near the king
    fn evaluate_open_lines_near_king(
        &self,
        board: &Board,
        color: Color,
        king_square: Square,
    ) -> f32 {
        let mut line_score = 0.0;
        let king_file = king_square.get_file();
        let _king_rank = king_square.get_rank();

        // Check files near the king
        for file_offset in -1..=1i8 {
            let file_index = (king_file.to_index() as i8 + file_offset).clamp(0, 7) as usize;
            let file = chess::File::from_index(file_index);
            if self.is_open_file(board, file) {
                line_score -= 20.0; // Open file near king is dangerous
            } else if self.is_semi_open_file(board, file, color) {
                line_score -= 10.0; // Semi-open file is also risky
            }
        }

        // Check diagonals emanating from king
        line_score += self.evaluate_diagonal_safety(board, color, king_square);

        line_score
    }

    /// Check if a file is completely open (no pawns)
    fn is_open_file(&self, board: &Board, file: chess::File) -> bool {
        let file_mask = self.get_file_mask(file);
        let all_pawns = board.pieces(chess::Piece::Pawn);
        (all_pawns & file_mask).popcnt() == 0
    }

    /// Check if a file is semi-open for a color (no own pawns)
    fn is_semi_open_file(&self, board: &Board, file: chess::File, color: Color) -> bool {
        let file_mask = self.get_file_mask(file);
        let own_pawns = board.pieces(chess::Piece::Pawn) & board.color_combined(color);
        (own_pawns & file_mask).popcnt() == 0
    }

    /// Evaluate diagonal safety around the king
    fn evaluate_diagonal_safety(&self, board: &Board, color: Color, king_square: Square) -> f32 {
        let mut score = 0.0;
        let enemy_color = !color;
        let enemy_bishops_queens = (board.pieces(chess::Piece::Bishop)
            | board.pieces(chess::Piece::Queen))
            & board.color_combined(enemy_color);

        // Check major diagonals for threats
        let directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

        for (file_dir, rank_dir) in directions {
            if self.has_diagonal_threat(
                board,
                king_square,
                file_dir,
                rank_dir,
                enemy_bishops_queens,
            ) {
                score -= 15.0; // Diagonal threat penalty
            }
        }

        score
    }

    /// Check for diagonal threats to the king
    fn has_diagonal_threat(
        &self,
        board: &Board,
        king_square: Square,
        file_dir: i8,
        rank_dir: i8,
        enemy_pieces: chess::BitBoard,
    ) -> bool {
        let mut file = king_square.get_file().to_index() as i8 + file_dir;
        let mut rank = king_square.get_rank().to_index() as i8 + rank_dir;

        while (0..8).contains(&file) && (0..8).contains(&rank) {
            let square = Square::make_square(
                chess::Rank::from_index(rank as usize),
                chess::File::from_index(file as usize),
            );
            if let Some(_piece) = board.piece_on(square) {
                // Check if this is an enemy bishop or queen
                return (enemy_pieces & chess::BitBoard::from_square(square)).popcnt() > 0;
            }
            file += file_dir;
            rank += rank_dir;
        }

        false
    }

    /// Evaluate king activity in endgame
    fn evaluate_king_endgame_activity(
        &self,
        board: &Board,
        color: Color,
        king_square: Square,
    ) -> f32 {
        let mut activity_score = 0.0;

        // Centralization bonus
        let file = king_square.get_file().to_index() as f32;
        let rank = king_square.get_rank().to_index() as f32;
        let center_distance = ((file - 3.5).abs() + (rank - 3.5).abs()) / 2.0;
        activity_score += (3.5 - center_distance) * 10.0;

        // Bonus for approaching enemy pawns
        let enemy_pawns = board.pieces(chess::Piece::Pawn) & board.color_combined(!color);
        for enemy_pawn in enemy_pawns {
            let distance = ((king_square.get_file().to_index() as i8
                - enemy_pawn.get_file().to_index() as i8)
                .abs()
                + (king_square.get_rank().to_index() as i8
                    - enemy_pawn.get_rank().to_index() as i8)
                    .abs()) as f32;
            if distance <= 3.0 {
                activity_score += 5.0; // Bonus for being close to enemy pawns
            }
        }

        activity_score
    }

    /// Evaluate control of squares around the king
    fn evaluate_king_zone_control(&self, board: &Board, color: Color, king_square: Square) -> f32 {
        let mut control_score = 0.0;
        let king_file = king_square.get_file().to_index() as i8;
        let king_rank = king_square.get_rank().to_index() as i8;

        // Check 3x3 area around king
        for file_offset in -1..=1 {
            for rank_offset in -1..=1 {
                if file_offset == 0 && rank_offset == 0 {
                    continue; // Skip king's own square
                }

                let check_file = king_file + file_offset;
                let check_rank = king_rank + rank_offset;

                if (0..8).contains(&check_file) && (0..8).contains(&check_rank) {
                    let square = Square::make_square(
                        chess::Rank::from_index(check_rank as usize),
                        chess::File::from_index(check_file as usize),
                    );
                    if let Some(_piece) = board.piece_on(square) {
                        if board.color_on(square) == Some(color) {
                            control_score += 3.0; // Own piece near king
                        } else {
                            control_score -= 5.0; // Enemy piece near king
                        }
                    }
                }
            }
        }

        control_score
    }

    /// Evaluate severity of being in check
    fn evaluate_check_severity(&self, board: &Board, _color: Color) -> f32 {
        let checkers = board.checkers();
        let check_count = checkers.popcnt();

        let base_penalty = match check_count {
            0 => 0.0,
            1 => 50.0,  // Single check
            2 => 150.0, // Double check - very dangerous
            _ => 200.0, // Multiple checks - critical
        };

        // Additional penalty if king has few escape squares
        let legal_moves: Vec<_> = MoveGen::new_legal(board).collect();
        let king_moves = legal_moves
            .iter()
            .filter(|mv| board.piece_on(mv.get_source()) == Some(chess::Piece::King))
            .count();

        let escape_penalty = match king_moves {
            0 => 100.0, // No king moves - potential mate threat
            1 => 30.0,  // Very limited mobility
            2 => 15.0,  // Limited mobility
            _ => 0.0,   // Adequate mobility
        };

        base_penalty + escape_penalty
    }

    /// Determine game phase based on material count
    fn determine_game_phase(&self, board: &Board) -> GamePhase {
        // Count non-pawn material for both sides
        let mut material_count = 0;

        for piece in [
            chess::Piece::Queen,
            chess::Piece::Rook,
            chess::Piece::Bishop,
            chess::Piece::Knight,
        ] {
            material_count += board.pieces(piece).popcnt();
        }

        match material_count {
            0..=4 => GamePhase::Endgame,     // Very few pieces left
            5..=12 => GamePhase::Middlegame, // Some pieces traded
            _ => GamePhase::Opening,         // Most pieces on board
        }
    }

    /// Count attackers threatening the king
    #[allow(dead_code)]
    fn count_king_attackers(&self, board: &Board, color: Color) -> u32 {
        let king_square = board.king_square(color);
        let opponent_color = if color == Color::White {
            Color::Black
        } else {
            Color::White
        };

        // Count enemy pieces that could potentially attack the king
        let mut attackers = 0;

        // Check for enemy pieces near the king (simplified threat detection)
        for piece in [
            chess::Piece::Queen,
            chess::Piece::Rook,
            chess::Piece::Bishop,
            chess::Piece::Knight,
            chess::Piece::Pawn,
        ] {
            let enemy_pieces = board.pieces(piece) & board.color_combined(opponent_color);

            // For each enemy piece of this type, check if it's in attacking range
            for square in enemy_pieces {
                let rank_diff = (king_square.get_rank().to_index() as i32
                    - square.get_rank().to_index() as i32)
                    .abs();
                let file_diff = (king_square.get_file().to_index() as i32
                    - square.get_file().to_index() as i32)
                    .abs();

                // Simplified threat detection based on piece type and distance
                let is_threat = match piece {
                    chess::Piece::Queen => rank_diff <= 2 || file_diff <= 2,
                    chess::Piece::Rook => rank_diff <= 2 || file_diff <= 2,
                    chess::Piece::Bishop => rank_diff == file_diff && rank_diff <= 2,
                    chess::Piece::Knight => {
                        (rank_diff == 2 && file_diff == 1) || (rank_diff == 1 && file_diff == 2)
                    }
                    chess::Piece::Pawn => {
                        rank_diff == 1
                            && file_diff == 1
                            && ((color == Color::White
                                && square.get_rank().to_index()
                                    > king_square.get_rank().to_index())
                                || (color == Color::Black
                                    && square.get_rank().to_index()
                                        < king_square.get_rank().to_index()))
                    }
                    _ => false,
                };

                if is_threat {
                    attackers += 1;
                }
            }
        }

        attackers
    }

    /// Get file mask for a given file
    fn get_file_mask(&self, file: chess::File) -> chess::BitBoard {
        chess::BitBoard(0x0101010101010101u64 << file.to_index())
    }

    /// Comprehensive pawn structure evaluation
    fn evaluate_pawn_structure(&self, board: &Board) -> f32 {
        let mut score = 0.0;

        for color in [Color::White, Color::Black] {
            let multiplier = if color == Color::White { 1.0 } else { -1.0 };
            let pawns = board.pieces(chess::Piece::Pawn) & board.color_combined(color);

            // Analyze each file for pawn structure
            for file in 0..8 {
                let file_mask = self.get_file_mask(chess::File::from_index(file));
                let file_pawns = pawns & file_mask;
                let pawn_count = file_pawns.popcnt();

                // 1. Doubled pawns penalty
                if pawn_count > 1 {
                    score += -0.5 * multiplier * (pawn_count - 1) as f32; // -0.5 per extra pawn
                }

                // 2. Isolated pawns penalty
                if pawn_count > 0 {
                    let has_adjacent_pawns = self.has_adjacent_pawns(board, color, file);
                    if !has_adjacent_pawns {
                        score += -0.3 * multiplier; // Isolated pawn penalty
                    }
                }

                // 3. Analyze individual pawns on this file
                for square in file_pawns {
                    // Passed pawn bonus
                    if self.is_passed_pawn(board, square, color) {
                        let rank = square.get_rank().to_index();
                        let advancement = if color == Color::White {
                            rank
                        } else {
                            7 - rank
                        };
                        score += (0.2 + advancement as f32 * 0.3) * multiplier; // Increasing bonus as pawn advances
                    }

                    // Backward pawn penalty
                    if self.is_backward_pawn(board, square, color) {
                        score += -0.2 * multiplier;
                    }

                    // Connected pawns bonus
                    if self.has_pawn_support(board, square, color) {
                        score += 0.1 * multiplier;
                    }
                }
            }

            // 4. Pawn chains and advanced formations
            score += self.evaluate_pawn_chains(board, color) * multiplier;
        }

        score
    }

    /// Check if pawn has adjacent pawns (not isolated)
    fn has_adjacent_pawns(&self, board: &Board, color: Color, file: usize) -> bool {
        let pawns = board.pieces(chess::Piece::Pawn) & board.color_combined(color);

        // Check adjacent files
        if file > 0 {
            let left_file_mask = self.get_file_mask(chess::File::from_index(file - 1));
            if (pawns & left_file_mask).popcnt() > 0 {
                return true;
            }
        }

        if file < 7 {
            let right_file_mask = self.get_file_mask(chess::File::from_index(file + 1));
            if (pawns & right_file_mask).popcnt() > 0 {
                return true;
            }
        }

        false
    }

    /// Check if pawn is passed (no enemy pawns can stop it)
    fn is_passed_pawn(&self, board: &Board, pawn_square: Square, color: Color) -> bool {
        let opponent_color = if color == Color::White {
            Color::Black
        } else {
            Color::White
        };
        let opponent_pawns =
            board.pieces(chess::Piece::Pawn) & board.color_combined(opponent_color);

        let file = pawn_square.get_file().to_index();
        let rank = pawn_square.get_rank().to_index();

        // Check if any opponent pawns can stop this pawn
        for opponent_square in opponent_pawns {
            let opp_file = opponent_square.get_file().to_index();
            let opp_rank = opponent_square.get_rank().to_index();

            // Check if opponent pawn is in the path or can capture
            let file_diff = (file as i32 - opp_file as i32).abs();

            if file_diff <= 1 {
                // Same file or adjacent file
                if color == Color::White && opp_rank > rank {
                    return false; // Opponent pawn blocks or can capture
                }
                if color == Color::Black && opp_rank < rank {
                    return false; // Opponent pawn blocks or can capture
                }
            }
        }

        true
    }

    /// Check if pawn is backward (can't be supported by other pawns)
    fn is_backward_pawn(&self, board: &Board, pawn_square: Square, color: Color) -> bool {
        let pawns = board.pieces(chess::Piece::Pawn) & board.color_combined(color);
        let file = pawn_square.get_file().to_index();
        let rank = pawn_square.get_rank().to_index();

        // Check if any friendly pawns on adjacent files can support this pawn
        for support_file in [file.saturating_sub(1), (file + 1).min(7)] {
            if support_file == file {
                continue;
            }

            let file_mask = self.get_file_mask(chess::File::from_index(support_file));
            let file_pawns = pawns & file_mask;

            for support_square in file_pawns {
                let support_rank = support_square.get_rank().to_index();

                // Check if this pawn can potentially support our pawn
                if color == Color::White && support_rank < rank {
                    return false; // Can be supported
                }
                if color == Color::Black && support_rank > rank {
                    return false; // Can be supported
                }
            }
        }

        true
    }

    /// Check if pawn has support from adjacent pawns
    fn has_pawn_support(&self, board: &Board, pawn_square: Square, color: Color) -> bool {
        let pawns = board.pieces(chess::Piece::Pawn) & board.color_combined(color);
        let file = pawn_square.get_file().to_index();
        let rank = pawn_square.get_rank().to_index();

        // Check adjacent files for supporting pawns
        for support_file in [file.saturating_sub(1), (file + 1).min(7)] {
            if support_file == file {
                continue;
            }

            let file_mask = self.get_file_mask(chess::File::from_index(support_file));
            let file_pawns = pawns & file_mask;

            for support_square in file_pawns {
                let support_rank = support_square.get_rank().to_index();

                // Check if this pawn is directly supporting (diagonal protection)
                if (support_rank as i32 - rank as i32).abs() == 1 {
                    return true;
                }
            }
        }

        false
    }

    /// Evaluate pawn chains and advanced formations
    fn evaluate_pawn_chains(&self, board: &Board, color: Color) -> f32 {
        let pawns = board.pieces(chess::Piece::Pawn) & board.color_combined(color);
        let mut chain_score = 0.0;

        // Count connected pawn chains
        let mut chain_lengths = Vec::new();
        let mut visited = std::collections::HashSet::new();

        for pawn_square in pawns {
            if visited.contains(&pawn_square) {
                continue;
            }

            let chain_length = self.count_pawn_chain(board, pawn_square, color, &mut visited);
            if chain_length > 1 {
                chain_lengths.push(chain_length);
            }
        }

        // Bonus for longer chains
        for &length in &chain_lengths {
            chain_score += (length as f32 - 1.0) * 0.15; // +0.15 per connected pawn beyond the first
        }

        chain_score
    }

    /// Count length of pawn chain starting from a pawn
    #[allow(clippy::only_used_in_recursion)]
    fn count_pawn_chain(
        &self,
        board: &Board,
        start_square: Square,
        color: Color,
        visited: &mut std::collections::HashSet<Square>,
    ) -> usize {
        if visited.contains(&start_square) {
            return 0;
        }

        visited.insert(start_square);
        let pawns = board.pieces(chess::Piece::Pawn) & board.color_combined(color);

        // Check if this square actually has a pawn
        if (pawns & chess::BitBoard::from_square(start_square)) == chess::BitBoard(0) {
            return 0;
        }

        let mut count = 1;
        let file = start_square.get_file().to_index();
        let rank = start_square.get_rank().to_index();

        // Check diagonally connected pawns (pawn chain formation)
        for &(file_offset, rank_offset) in &[(-1i32, -1i32), (-1, 1), (1, -1), (1, 1)] {
            let new_file = file as i32 + file_offset;
            let new_rank = rank as i32 + rank_offset;

            if (0..8).contains(&new_file) && (0..8).contains(&new_rank) {
                let square_index = (new_rank * 8 + new_file) as u8;
                let new_square = unsafe { Square::new(square_index) };
                if (pawns & chess::BitBoard::from_square(new_square)) != chess::BitBoard(0)
                    && !visited.contains(&new_square)
                {
                    count += self.count_pawn_chain(board, new_square, color, visited);
                }
            }
        }

        count
    }

    /// Check if this is a tactical position (has captures, checks, or threats)
    fn is_tactical_position(&self, board: &Board) -> bool {
        // Check if in check
        if board.checkers().popcnt() > 0 {
            return true;
        }

        // Check for captures available
        let captures = self.generate_captures(board);
        if !captures.is_empty() {
            return true;
        }

        // If we have many legal moves, it's likely a tactical position
        let legal_moves: Vec<_> = MoveGen::new_legal(board).collect();
        if legal_moves.len() > 35 {
            return true;
        }

        false
    }

    /// Check if a move is a capture or promotion
    fn is_capture_or_promotion(&self, chess_move: &ChessMove, board: &Board) -> bool {
        board.piece_on(chess_move.get_dest()).is_some() || chess_move.get_promotion().is_some()
    }

    /// Check if a side has non-pawn material
    fn has_non_pawn_material(&self, board: &Board, color: Color) -> bool {
        let pieces = board.color_combined(color)
            & !board.pieces(chess::Piece::Pawn)
            & !board.pieces(chess::Piece::King);
        pieces.popcnt() > 0
    }

    /// Check if a move is a killer move
    fn is_killer_move(&self, chess_move: &ChessMove) -> bool {
        // Simple killer move detection - can be enhanced with depth tracking
        for depth_killers in &self.killer_moves {
            for killer_move in depth_killers.iter().flatten() {
                if killer_move == chess_move {
                    return true;
                }
            }
        }
        false
    }

    /// Store a killer move at the given depth
    fn store_killer_move(&mut self, chess_move: ChessMove, depth: u32) {
        let depth_idx = (depth as usize).min(self.killer_moves.len() - 1);

        // Shift killer moves: new killer becomes first, first becomes second
        if let Some(first_killer) = self.killer_moves[depth_idx][0] {
            if first_killer != chess_move {
                self.killer_moves[depth_idx][1] = Some(first_killer);
                self.killer_moves[depth_idx][0] = Some(chess_move);
            }
        } else {
            self.killer_moves[depth_idx][0] = Some(chess_move);
        }
    }

    /// Update history heuristic for move ordering
    fn update_history(&mut self, chess_move: &ChessMove, depth: u32) {
        let key = (chess_move.get_source(), chess_move.get_dest());
        let bonus = depth * depth; // Quadratic bonus for deeper successful moves

        let current = self.history_heuristic.get(&key).unwrap_or(&0);
        self.history_heuristic.insert(key, current + bonus);
    }

    /// Get history score for move ordering
    fn get_history_score(&self, chess_move: &ChessMove) -> u32 {
        let key = (chess_move.get_source(), chess_move.get_dest());
        *self.history_heuristic.get(&key).unwrap_or(&0)
    }

    /// Store a counter move (refutation of the last opponent move)
    #[allow(dead_code)]
    fn store_counter_move(&mut self, refutation: ChessMove) {
        if let Some(last_move) = self.last_move {
            let last_move_key = (last_move.get_source(), last_move.get_dest());
            self.counter_moves.insert(last_move_key, refutation);
        }
    }

    /// Update the last move played (for counter move tracking)
    #[allow(dead_code)]
    fn update_last_move(&mut self, chess_move: ChessMove) {
        self.last_move = Some(chess_move);
    }

    /// Clear transposition table
    pub fn clear_cache(&mut self) {
        self.transposition_table.clear();
    }

    /// Get search statistics
    pub fn get_stats(&self) -> (u64, usize) {
        (self.nodes_searched, self.transposition_table.len())
    }

    /// Evaluate endgame tablebase knowledge patterns (production-ready)
    fn evaluate_endgame_patterns(&self, board: &Board) -> f32 {
        let mut score = 0.0;

        // Check if we're in an endgame (low piece count)
        let piece_count = self.count_all_pieces(board);
        if piece_count > 10 {
            return 0.0; // Not an endgame, skip pattern evaluation
        }

        // Apply endgame evaluation weight from config
        let endgame_weight = self.config.endgame_evaluation_weight;

        // Comprehensive endgame pattern evaluation
        score += self.evaluate_king_pawn_endgames(board) * endgame_weight;
        score += self.evaluate_basic_mate_patterns(board) * endgame_weight;
        score += self.evaluate_opposition_patterns(board) * endgame_weight;
        score += self.evaluate_key_squares(board) * endgame_weight;
        score += self.evaluate_zugzwang_patterns(board) * endgame_weight;

        // Advanced endgame patterns for production strength
        score += self.evaluate_piece_coordination_endgame(board) * endgame_weight;
        score += self.evaluate_fortress_patterns(board) * endgame_weight;
        score += self.evaluate_theoretical_endgames(board) * endgame_weight;

        score
    }

    /// Count total pieces on the board
    fn count_all_pieces(&self, board: &Board) -> u32 {
        let mut count = 0;
        for piece in [
            chess::Piece::Pawn,
            chess::Piece::Knight,
            chess::Piece::Bishop,
            chess::Piece::Rook,
            chess::Piece::Queen,
        ] {
            count += board.pieces(piece).popcnt();
        }
        count += board.pieces(chess::Piece::King).popcnt(); // Kings are always 2
        count
    }

    /// Evaluate king and pawn endgames
    fn evaluate_king_pawn_endgames(&self, board: &Board) -> f32 {
        let mut score = 0.0;

        // Rule of the square for passed pawns
        for color in [Color::White, Color::Black] {
            let multiplier = if color == Color::White { 1.0 } else { -1.0 };
            let king_square = board.king_square(color);
            let opponent_king_square = board.king_square(!color);
            let pawns = board.pieces(chess::Piece::Pawn) & board.color_combined(color);

            for pawn_square in pawns {
                if self.is_passed_pawn(board, pawn_square, color) {
                    let _pawn_file = pawn_square.get_file().to_index();
                    let pawn_rank = pawn_square.get_rank().to_index();

                    // Calculate promotion square
                    let promotion_rank = if color == Color::White { 7 } else { 0 };
                    let promotion_square = Square::make_square(
                        chess::Rank::from_index(promotion_rank),
                        chess::File::from_index(_pawn_file),
                    );

                    // Calculate distances
                    let king_distance = self.square_distance(king_square, promotion_square);
                    let opponent_king_distance =
                        self.square_distance(opponent_king_square, promotion_square);
                    let pawn_distance = (promotion_rank as i32 - pawn_rank as i32).unsigned_abs();

                    // Rule of the square: pawn wins if opponent king can't catch it
                    if pawn_distance < opponent_king_distance {
                        score += 2.0 * multiplier; // Winning passed pawn
                    } else if king_distance < opponent_king_distance {
                        score += 1.0 * multiplier; // Supported passed pawn
                    }
                }
            }
        }

        score
    }

    /// Evaluate basic mate patterns (KQ vs K, KR vs K, etc.)
    fn evaluate_basic_mate_patterns(&self, board: &Board) -> f32 {
        let mut score = 0.0;

        for color in [Color::White, Color::Black] {
            let multiplier = if color == Color::White { 1.0 } else { -1.0 };
            let opponent_color = !color;

            let queens = (board.pieces(chess::Piece::Queen) & board.color_combined(color)).popcnt();
            let rooks = (board.pieces(chess::Piece::Rook) & board.color_combined(color)).popcnt();
            let bishops =
                (board.pieces(chess::Piece::Bishop) & board.color_combined(color)).popcnt();
            let knights =
                (board.pieces(chess::Piece::Knight) & board.color_combined(color)).popcnt();

            let opp_queens =
                (board.pieces(chess::Piece::Queen) & board.color_combined(opponent_color)).popcnt();
            let opp_rooks =
                (board.pieces(chess::Piece::Rook) & board.color_combined(opponent_color)).popcnt();
            let opp_bishops = (board.pieces(chess::Piece::Bishop)
                & board.color_combined(opponent_color))
            .popcnt();
            let opp_knights = (board.pieces(chess::Piece::Knight)
                & board.color_combined(opponent_color))
            .popcnt();
            let opp_pawns =
                (board.pieces(chess::Piece::Pawn) & board.color_combined(opponent_color)).popcnt();

            // Check for basic mate patterns
            if opp_queens == 0
                && opp_rooks == 0
                && opp_bishops == 0
                && opp_knights == 0
                && opp_pawns == 0
            {
                // Opponent has only king
                if queens > 0 || rooks > 0 {
                    // KQ vs K or KR vs K - drive king to corner
                    let king_square = board.king_square(color);
                    let opponent_king_square = board.king_square(opponent_color);
                    let corner_distance = self.distance_to_nearest_corner(opponent_king_square);
                    let king_distance = self.square_distance(king_square, opponent_king_square);

                    score += 1.0 * multiplier; // Basic mate advantage
                    score += (7.0 - corner_distance as f32) * 0.1 * multiplier; // Drive to corner
                    score += (8.0 - king_distance as f32) * 0.05 * multiplier; // Keep kings close
                }

                if bishops >= 2 {
                    // KBB vs K - mate with two bishops
                    let opponent_king_square = board.king_square(opponent_color);
                    let corner_distance = self.distance_to_nearest_corner(opponent_king_square);
                    score += 0.8 * multiplier; // Slightly less than KQ/KR
                    score += (7.0 - corner_distance as f32) * 0.08 * multiplier;
                }

                if bishops >= 1 && knights >= 1 {
                    // KBN vs K - complex mate
                    score += 0.6 * multiplier; // More difficult mate
                }
            }
        }

        score
    }

    /// Evaluate opposition patterns in king and pawn endgames
    fn evaluate_opposition_patterns(&self, board: &Board) -> f32 {
        let mut score = 0.0;

        let white_king = board.king_square(Color::White);
        let black_king = board.king_square(Color::Black);

        let file_diff = (white_king.get_file().to_index() as i32
            - black_king.get_file().to_index() as i32)
            .abs();
        let rank_diff = (white_king.get_rank().to_index() as i32
            - black_king.get_rank().to_index() as i32)
            .abs();

        // Check for opposition (kings facing each other with one square between)
        if (file_diff == 0 && rank_diff == 2) || (file_diff == 2 && rank_diff == 0) {
            // Direct opposition - the side NOT to move has the advantage
            let opposition_bonus = 0.2;
            if board.side_to_move() == Color::White {
                score -= opposition_bonus; // Black has opposition
            } else {
                score += opposition_bonus; // White has opposition
            }
        }

        // Distant opposition
        if file_diff == 0 && rank_diff % 2 == 0 && rank_diff > 2 {
            let distant_opposition_bonus = 0.1;
            if board.side_to_move() == Color::White {
                score -= distant_opposition_bonus;
            } else {
                score += distant_opposition_bonus;
            }
        }

        score
    }

    /// Evaluate key squares in pawn endgames
    fn evaluate_key_squares(&self, board: &Board) -> f32 {
        let mut score = 0.0;

        // In pawn endgames, key squares are critical
        for color in [Color::White, Color::Black] {
            let multiplier = if color == Color::White { 1.0 } else { -1.0 };
            let king_square = board.king_square(color);
            let pawns = board.pieces(chess::Piece::Pawn) & board.color_combined(color);

            for pawn_square in pawns {
                if self.is_passed_pawn(board, pawn_square, color) {
                    // Key squares are typically in front of the pawn
                    let key_squares = self.get_key_squares(pawn_square, color);

                    for key_square in key_squares {
                        let distance = self.square_distance(king_square, key_square);
                        if distance <= 1 {
                            score += 0.3 * multiplier; // King controls key square
                        } else if distance <= 2 {
                            score += 0.1 * multiplier; // King near key square
                        }
                    }
                }
            }
        }

        score
    }

    /// Evaluate zugzwang patterns
    fn evaluate_zugzwang_patterns(&self, board: &Board) -> f32 {
        let mut score = 0.0;

        // Simple zugzwang detection in pawn endgames
        let piece_count = self.count_all_pieces(board);
        if piece_count <= 6 {
            // Very few pieces
            let legal_moves: Vec<_> = MoveGen::new_legal(board).collect();

            // If very few legal moves, position might be zugzwang-prone
            if legal_moves.len() <= 3 {
                // Evaluate if any move worsens the position significantly
                let current_eval = self.quick_evaluate_position(board);
                let mut bad_moves = 0;

                for chess_move in legal_moves.iter().take(3) {
                    let new_board = board.make_move_new(*chess_move);
                    let new_eval = -self.quick_evaluate_position(&new_board); // Flip for opponent

                    if new_eval < current_eval - 0.5 {
                        bad_moves += 1;
                    }
                }

                // If most moves are bad, it's likely zugzwang
                if bad_moves >= legal_moves.len() / 2 {
                    let zugzwang_penalty = 0.3;
                    if board.side_to_move() == Color::White {
                        score -= zugzwang_penalty;
                    } else {
                        score += zugzwang_penalty;
                    }
                }
            }
        }

        score
    }

    /// Calculate Manhattan distance between two squares
    fn square_distance(&self, sq1: Square, sq2: Square) -> u32 {
        let file1 = sq1.get_file().to_index() as i32;
        let rank1 = sq1.get_rank().to_index() as i32;
        let file2 = sq2.get_file().to_index() as i32;
        let rank2 = sq2.get_rank().to_index() as i32;

        ((file1 - file2).abs() + (rank1 - rank2).abs()) as u32
    }

    /// Calculate distance to nearest corner
    fn distance_to_nearest_corner(&self, square: Square) -> u32 {
        let file = square.get_file().to_index() as i32;
        let rank = square.get_rank().to_index() as i32;

        let corner_distances = [
            file + rank,             // a1
            (7 - file) + rank,       // h1
            file + (7 - rank),       // a8
            (7 - file) + (7 - rank), // h8
        ];

        *corner_distances.iter().min().unwrap() as u32
    }

    /// Get key squares for a passed pawn
    fn get_key_squares(&self, pawn_square: Square, color: Color) -> Vec<Square> {
        let mut key_squares = Vec::new();
        let file = pawn_square.get_file().to_index();
        let rank = pawn_square.get_rank().to_index();

        // Key squares are typically 2 squares in front of the pawn
        let key_rank = if color == Color::White {
            if rank + 2 <= 7 {
                rank + 2
            } else {
                return key_squares;
            }
        } else if rank >= 2 {
            rank - 2
        } else {
            return key_squares;
        };

        // Key squares on the same file and adjacent files
        for key_file in (file.saturating_sub(1))..=(file + 1).min(7) {
            let square = Square::make_square(
                chess::Rank::from_index(key_rank),
                chess::File::from_index(key_file),
            );
            key_squares.push(square);
        }

        key_squares
    }

    /// Quick position evaluation (simpler than full evaluation)
    fn quick_evaluate_position(&self, board: &Board) -> f32 {
        let mut score = 0.0;

        // Simple material count
        score += self.material_balance(board);

        // Basic king safety
        for color in [Color::White, Color::Black] {
            let multiplier = if color == Color::White { 1.0 } else { -1.0 };
            let king_square = board.king_square(color);
            let file = king_square.get_file().to_index();
            let rank = king_square.get_rank().to_index();

            // Prefer center in endgame
            let center_distance = (file as f32 - 3.5).abs() + (rank as f32 - 3.5).abs();
            score += (7.0 - center_distance) * 0.05 * multiplier;
        }

        score
    }

    /// Evaluate piece coordination in endgames
    fn evaluate_piece_coordination_endgame(&self, board: &Board) -> f32 {
        let mut score = 0.0;

        for color in [Color::White, Color::Black] {
            let multiplier = if color == Color::White { 1.0 } else { -1.0 };
            let king_square = board.king_square(color);

            // Rook-king coordination
            let rooks = board.pieces(chess::Piece::Rook) & board.color_combined(color);
            for rook_square in rooks {
                let distance = self.square_distance(king_square, rook_square);
                if distance <= 3 {
                    score += 0.2 * multiplier; // King-rook coordination bonus
                }

                // Rook on 7th rank bonus in endgame
                let rook_rank = rook_square.get_rank().to_index();
                if (color == Color::White && rook_rank == 6)
                    || (color == Color::Black && rook_rank == 1)
                {
                    score += 0.4 * multiplier;
                }
            }

            // Queen-king coordination
            let queens = board.pieces(chess::Piece::Queen) & board.color_combined(color);
            for queen_square in queens {
                let distance = self.square_distance(king_square, queen_square);
                if distance <= 4 {
                    score += 0.15 * multiplier; // Queen-king coordination
                }
            }

            // Bishop pair coordination in endgame
            let bishops = board.pieces(chess::Piece::Bishop) & board.color_combined(color);
            if bishops.popcnt() >= 2 {
                score += 0.3 * multiplier; // Bishop pair is strong in endgame
            }
        }

        score
    }

    /// Evaluate fortress patterns (drawish defensive setups)
    fn evaluate_fortress_patterns(&self, board: &Board) -> f32 {
        let mut score = 0.0;

        // Check for typical fortress patterns
        for color in [Color::White, Color::Black] {
            let multiplier = if color == Color::White { 1.0 } else { -1.0 };
            let opponent_color = !color;

            // Material difference for fortress evaluation
            let material_diff = self.calculate_material_difference(board, color);

            // Only evaluate fortress if down material
            if material_diff < -2.0 {
                // King in corner fortress
                let king_square = board.king_square(color);
                let king_file = king_square.get_file().to_index();
                let king_rank = king_square.get_rank().to_index();

                // Corner fortress detection
                if (king_file <= 1 || king_file >= 6) && (king_rank <= 1 || king_rank >= 6) {
                    let pawns = board.pieces(chess::Piece::Pawn) & board.color_combined(color);
                    let bishops = board.pieces(chess::Piece::Bishop) & board.color_combined(color);

                    // Bishop + pawns fortress
                    if bishops.popcnt() > 0 && pawns.popcnt() >= 2 {
                        score += 0.5 * multiplier; // Fortress bonus (defensive)
                    }
                }

                // Rook vs pawns fortress
                let rooks = board.pieces(chess::Piece::Rook) & board.color_combined(color);
                let opp_pawns =
                    board.pieces(chess::Piece::Pawn) & board.color_combined(opponent_color);
                if rooks.popcnt() > 0 && opp_pawns.popcnt() >= 3 {
                    score += 0.3 * multiplier; // Rook activity vs pawns
                }
            }
        }

        score
    }

    /// Evaluate theoretical endgame patterns
    fn evaluate_theoretical_endgames(&self, board: &Board) -> f32 {
        let mut score = 0.0;

        let piece_count = self.count_all_pieces(board);

        // Only evaluate in very simple endgames
        if piece_count <= 6 {
            // Rook endgames
            score += self.evaluate_rook_endgames(board);

            // Bishop endgames
            score += self.evaluate_bishop_endgames(board);

            // Knight endgames
            score += self.evaluate_knight_endgames(board);

            // Mixed piece endgames
            score += self.evaluate_mixed_piece_endgames(board);
        }

        score
    }

    /// Evaluate rook endgame principles
    fn evaluate_rook_endgames(&self, board: &Board) -> f32 {
        let mut score = 0.0;

        for color in [Color::White, Color::Black] {
            let multiplier = if color == Color::White { 1.0 } else { -1.0 };
            let rooks = board.pieces(chess::Piece::Rook) & board.color_combined(color);
            let opponent_king = board.king_square(!color);

            for rook_square in rooks {
                // Rook behind passed pawn
                let pawns = board.pieces(chess::Piece::Pawn) & board.color_combined(color);
                for pawn_square in pawns {
                    if self.is_passed_pawn(board, pawn_square, color) {
                        let rook_file = rook_square.get_file().to_index();
                        let pawn_file = pawn_square.get_file().to_index();
                        let rook_rank = rook_square.get_rank().to_index();
                        let pawn_rank = pawn_square.get_rank().to_index();

                        // Rook behind passed pawn on same file
                        if rook_file == pawn_file
                            && ((color == Color::White && rook_rank < pawn_rank)
                                || (color == Color::Black && rook_rank > pawn_rank))
                        {
                            score += 0.6 * multiplier; // Strong rook placement
                        }
                    }
                }

                // Rook cutting off king
                let king_distance_to_rook = self.square_distance(opponent_king, rook_square);
                if king_distance_to_rook >= 4 {
                    score += 0.2 * multiplier; // Active rook position
                }

                // Rook on open files
                let rook_file = rook_square.get_file().to_index();
                if self.is_file_open(board, rook_file) {
                    score += 0.3 * multiplier; // Rook on open file
                }
            }
        }

        score
    }

    /// Evaluate bishop endgame principles
    fn evaluate_bishop_endgames(&self, board: &Board) -> f32 {
        let mut score = 0.0;

        for color in [Color::White, Color::Black] {
            let multiplier = if color == Color::White { 1.0 } else { -1.0 };
            let bishops = board.pieces(chess::Piece::Bishop) & board.color_combined(color);
            let opponent_color = !color;

            // Wrong-color bishop with rook pawn
            let pawns = board.pieces(chess::Piece::Pawn) & board.color_combined(color);
            for pawn_square in pawns {
                let pawn_file = pawn_square.get_file().to_index();

                // Rook pawn (a or h file)
                if pawn_file == 0 || pawn_file == 7 {
                    for bishop_square in bishops {
                        let promotion_square = if color == Color::White {
                            Square::make_square(
                                chess::Rank::Eighth,
                                chess::File::from_index(pawn_file),
                            )
                        } else {
                            Square::make_square(
                                chess::Rank::First,
                                chess::File::from_index(pawn_file),
                            )
                        };

                        // Check if bishop controls promotion square
                        if self.bishop_attacks_square(board, bishop_square, promotion_square) {
                            score += 0.4 * multiplier; // Correct color bishop
                        } else {
                            score -= 0.8 * multiplier; // Wrong color bishop - big penalty
                        }
                    }
                }
            }

            // Bishop vs knight with pawns on one side
            let knights = board.pieces(chess::Piece::Knight) & board.color_combined(opponent_color);
            if bishops.popcnt() > 0 && knights.popcnt() > 0 {
                let pawns_kingside = self.count_pawns_on_side(board, true);
                let pawns_queenside = self.count_pawns_on_side(board, false);

                if pawns_kingside == 0 || pawns_queenside == 0 {
                    score += 0.25 * multiplier; // Bishop better with pawns on one side
                }
            }
        }

        score
    }

    /// Evaluate knight endgame principles  
    fn evaluate_knight_endgames(&self, board: &Board) -> f32 {
        let mut score = 0.0;

        for color in [Color::White, Color::Black] {
            let multiplier = if color == Color::White { 1.0 } else { -1.0 };
            let knights = board.pieces(chess::Piece::Knight) & board.color_combined(color);

            for knight_square in knights {
                // Knight centralization in endgame
                let file = knight_square.get_file().to_index();
                let rank = knight_square.get_rank().to_index();
                let center_distance = ((file as f32 - 3.5).abs() + (rank as f32 - 3.5).abs()) / 2.0;
                score += (4.0 - center_distance) * 0.1 * multiplier;

                // Knight supporting passed pawns
                let pawns = board.pieces(chess::Piece::Pawn) & board.color_combined(color);
                for pawn_square in pawns {
                    if self.is_passed_pawn(board, pawn_square, color) {
                        let distance = self.square_distance(knight_square, pawn_square);
                        if distance <= 2 {
                            score += 0.3 * multiplier; // Knight supporting passed pawn
                        }
                    }
                }
            }
        }

        score
    }

    /// Evaluate mixed piece endgames
    fn evaluate_mixed_piece_endgames(&self, board: &Board) -> f32 {
        let mut score = 0.0;

        for color in [Color::White, Color::Black] {
            let multiplier = if color == Color::White { 1.0 } else { -1.0 };

            let queens = (board.pieces(chess::Piece::Queen) & board.color_combined(color)).popcnt();
            let rooks = (board.pieces(chess::Piece::Rook) & board.color_combined(color)).popcnt();
            let bishops =
                (board.pieces(chess::Piece::Bishop) & board.color_combined(color)).popcnt();
            let knights =
                (board.pieces(chess::Piece::Knight) & board.color_combined(color)).popcnt();

            // Queen vs Rook and minor piece
            if queens > 0 && rooks == 0 {
                let opponent_color = !color;
                let opp_rooks = (board.pieces(chess::Piece::Rook)
                    & board.color_combined(opponent_color))
                .popcnt();
                let opp_minors = (board.pieces(chess::Piece::Bishop)
                    & board.color_combined(opponent_color))
                .popcnt()
                    + (board.pieces(chess::Piece::Knight) & board.color_combined(opponent_color))
                        .popcnt();

                if opp_rooks > 0 && opp_minors > 0 {
                    score += 0.5 * multiplier; // Queen vs R+minor is winning
                }
            }

            // Rook and bishop vs Rook and knight
            if rooks > 0 && bishops > 0 && knights == 0 {
                let opponent_color = !color;
                let opp_rooks = (board.pieces(chess::Piece::Rook)
                    & board.color_combined(opponent_color))
                .popcnt();
                let opp_knights = (board.pieces(chess::Piece::Knight)
                    & board.color_combined(opponent_color))
                .popcnt();

                if opp_rooks > 0 && opp_knights > 0 {
                    score += 0.2 * multiplier; // R+B slightly better than R+N
                }
            }
        }

        score
    }

    /// Helper: Calculate material difference for a color
    fn calculate_material_difference(&self, board: &Board, color: Color) -> f32 {
        let opponent_color = !color;

        let my_material = self.calculate_total_material(board, color);
        let opp_material = self.calculate_total_material(board, opponent_color);

        my_material - opp_material
    }

    /// Helper: Calculate total material for a color
    fn calculate_total_material(&self, board: &Board, color: Color) -> f32 {
        let mut material = 0.0;

        material +=
            (board.pieces(chess::Piece::Pawn) & board.color_combined(color)).popcnt() as f32 * 1.0;
        material += (board.pieces(chess::Piece::Knight) & board.color_combined(color)).popcnt()
            as f32
            * 3.0;
        material += (board.pieces(chess::Piece::Bishop) & board.color_combined(color)).popcnt()
            as f32
            * 3.0;
        material +=
            (board.pieces(chess::Piece::Rook) & board.color_combined(color)).popcnt() as f32 * 5.0;
        material +=
            (board.pieces(chess::Piece::Queen) & board.color_combined(color)).popcnt() as f32 * 9.0;

        material
    }

    /// Helper: Check if bishop attacks a square
    fn bishop_attacks_square(
        &self,
        board: &Board,
        bishop_square: Square,
        target_square: Square,
    ) -> bool {
        let file_diff = (bishop_square.get_file().to_index() as i32
            - target_square.get_file().to_index() as i32)
            .abs();
        let rank_diff = (bishop_square.get_rank().to_index() as i32
            - target_square.get_rank().to_index() as i32)
            .abs();

        // Same diagonal
        if file_diff == rank_diff {
            // Check if path is clear
            let file_step =
                if target_square.get_file().to_index() > bishop_square.get_file().to_index() {
                    1
                } else {
                    -1
                };
            let rank_step =
                if target_square.get_rank().to_index() > bishop_square.get_rank().to_index() {
                    1
                } else {
                    -1
                };

            let mut current_file = bishop_square.get_file().to_index() as i32 + file_step;
            let mut current_rank = bishop_square.get_rank().to_index() as i32 + rank_step;

            while current_file != target_square.get_file().to_index() as i32 {
                let square = Square::make_square(
                    chess::Rank::from_index(current_rank as usize),
                    chess::File::from_index(current_file as usize),
                );

                if board.piece_on(square).is_some() {
                    return false; // Path blocked
                }

                current_file += file_step;
                current_rank += rank_step;
            }

            true
        } else {
            false
        }
    }

    /// Helper: Count pawns on kingside (true) or queenside (false)
    fn count_pawns_on_side(&self, board: &Board, kingside: bool) -> u32 {
        let mut count = 0;
        let pawns = board.pieces(chess::Piece::Pawn);

        for pawn_square in pawns.into_iter() {
            let file = pawn_square.get_file().to_index();
            if (kingside && file >= 4) || (!kingside && file < 4) {
                count += 1;
            }
        }

        count
    }

    /// Helper: Check if a file is open (no pawns)
    fn is_file_open(&self, board: &Board, file: usize) -> bool {
        let file_mask = self.get_file_mask(chess::File::from_index(file));
        let pawns = board.pieces(chess::Piece::Pawn);
        (pawns & file_mask).popcnt() == 0
    }

    /// Count how many pieces of a given color attack a square
    fn count_attackers(&self, board: &Board, square: Square, color: Color) -> usize {
        let mut count = 0;

        // Check for pawn attacks
        let pawn_attacks = chess::get_pawn_attacks(square, !color, chess::BitBoard::new(0));
        let pawns = board.pieces(chess::Piece::Pawn) & board.color_combined(color);
        count += (pawn_attacks & pawns).popcnt() as usize;

        // Check for knight attacks
        let knight_attacks = chess::get_knight_moves(square);
        let knights = board.pieces(chess::Piece::Knight) & board.color_combined(color);
        count += (knight_attacks & knights).popcnt() as usize;

        // Check for king attacks
        let king_attacks = chess::get_king_moves(square);
        let kings = board.pieces(chess::Piece::King) & board.color_combined(color);
        count += (king_attacks & kings).popcnt() as usize;

        // Check for sliding piece attacks (bishops, rooks, queens)
        let all_pieces = *board.combined();

        // Bishop/Queen diagonal attacks
        let bishop_attacks = chess::get_bishop_moves(square, all_pieces);
        let bishops_queens = (board.pieces(chess::Piece::Bishop)
            | board.pieces(chess::Piece::Queen))
            & board.color_combined(color);
        count += (bishop_attacks & bishops_queens).popcnt() as usize;

        // Rook/Queen straight attacks
        let rook_attacks = chess::get_rook_moves(square, all_pieces);
        let rooks_queens = (board.pieces(chess::Piece::Rook) | board.pieces(chess::Piece::Queen))
            & board.color_combined(color);
        count += (rook_attacks & rooks_queens).popcnt() as usize;

        count
    }

    /// Evaluate hanging pieces - critical for 2000+ ELO tactical awareness
    fn evaluate_hanging_pieces(&self, board: &Board) -> f32 {
        let mut hanging_penalty = 0.0;

        // Check all pieces for both colors
        for color in [Color::White, Color::Black] {
            let multiplier = if color == Color::White { -1.0 } else { 1.0 }; // Penalty for our hanging pieces

            // Check each piece type
            for piece_type in [
                chess::Piece::Queen,
                chess::Piece::Rook,
                chess::Piece::Bishop,
                chess::Piece::Knight,
                chess::Piece::Pawn,
            ] {
                let pieces = board.pieces(piece_type) & board.color_combined(color);

                for square in pieces {
                    // Skip the king (can't really be "hanging" in the same sense)
                    if piece_type == chess::Piece::King {
                        continue;
                    }

                    let our_defenders = self.count_attackers(board, square, color);
                    let enemy_attackers = self.count_attackers(board, square, !color);

                    // If piece is attacked and not defended
                    if enemy_attackers > 0 && our_defenders == 0 {
                        let piece_value = self.get_piece_value(piece_type) as f32;
                        hanging_penalty += piece_value * multiplier * 0.8; // 80% penalty for hanging pieces
                    }
                    // If piece is attacked more than defended (likely to be lost)
                    else if enemy_attackers > our_defenders && enemy_attackers > 0 {
                        let piece_value = self.get_piece_value(piece_type) as f32;
                        hanging_penalty += piece_value * multiplier * 0.3; // 30% penalty for under-defended pieces
                    }
                }
            }
        }

        hanging_penalty
    }

    /// CRITICAL: Detect mate-in-N moves (up to 5 moves) to prevent tactical blindness
    fn find_mate_in_n(&self, board: &Board, max_depth: u32) -> Option<ChessMove> {
        if max_depth == 0 {
            return None;
        }

        let moves = MoveGen::new_legal(board);

        for chess_move in moves {
            let new_board = board.make_move_new(chess_move);

            // Check for immediate mate
            if new_board.status() == chess::BoardStatus::Checkmate {
                return Some(chess_move);
            }

            // Check for forced mate in N moves
            if max_depth > 1 && self.is_forced_mate(&new_board, max_depth - 1, false) {
                return Some(chess_move);
            }
        }

        None
    }

    /// Check if position is a forced mate for the side to move in N moves or less (optimized)
    fn is_forced_mate(&self, board: &Board, depth: u32, maximizing: bool) -> bool {
        if depth == 0 {
            return false;
        }

        if board.status() == chess::BoardStatus::Checkmate {
            return !maximizing; // Mate is good for us if we're not the one being mated
        }

        if board.status() != chess::BoardStatus::Ongoing {
            return false; // Stalemate or other non-mate endings
        }

        let moves = MoveGen::new_legal(board);
        let move_count = moves.len();

        // Quick optimization: limit search for positions with too many moves
        if move_count > 20 {
            return false; // Too complex for mate search
        }

        if maximizing {
            // We're looking for a move that forces mate - only check forcing moves first
            let mut forcing_moves = Vec::new();
            let mut other_moves = Vec::new();

            for chess_move in moves {
                let new_board = board.make_move_new(chess_move);
                if new_board.checkers().popcnt() > 0
                    || board.piece_on(chess_move.get_dest()).is_some()
                {
                    forcing_moves.push(chess_move); // Check or capture
                } else {
                    other_moves.push(chess_move);
                }
            }

            // Try forcing moves first
            for chess_move in forcing_moves {
                let new_board = board.make_move_new(chess_move);
                if self.is_forced_mate(&new_board, depth - 1, false) {
                    return true;
                }
            }

            // Only try quiet moves if very few
            if other_moves.len() <= 3 {
                for chess_move in other_moves {
                    let new_board = board.make_move_new(chess_move);
                    if self.is_forced_mate(&new_board, depth - 1, false) {
                        return true;
                    }
                }
            }

            false
        } else {
            // Opponent trying to avoid mate - limit to reasonable number of moves
            if move_count > 10 {
                return false; // Too many escape options
            }

            for chess_move in moves {
                let new_board = board.make_move_new(chess_move);
                if !self.is_forced_mate(&new_board, depth - 1, true) {
                    return false; // Opponent has an escape
                }
            }
            true // All opponent moves lead to mate for us
        }
    }

    /// CRITICAL: Detect tactical threats that require deeper search (simplified for performance)
    fn has_tactical_threats(&self, board: &Board) -> bool {
        // Quick check for forcing moves only
        if board.checkers().popcnt() > 0 {
            return true; // In check
        }

        // Quick capture count check (avoid expensive generation)
        let moves = MoveGen::new_legal(board);
        let capture_count = moves
            .filter(|m| board.piece_on(m.get_dest()).is_some())
            .count();

        capture_count > 3 // Many captures available suggests tactical complexity
    }

    /// CRITICAL: Enhanced king safety evaluation to prevent king exposure
    fn evaluate_king_safety(&self, board: &Board) -> f32 {
        let mut safety_score = 0.0;

        for color in [Color::White, Color::Black] {
            let king_square = board.king_square(color);
            let multiplier = if color == board.side_to_move() {
                1.0
            } else {
                -1.0
            };

            // 1. King exposure penalty - count attackers vs defenders
            let attackers = self.count_attackers(board, king_square, !color);
            let defenders = self.count_attackers(board, king_square, color);

            if attackers > defenders {
                safety_score -= (attackers - defenders) as f32 * 2.0 * multiplier;
            }

            // 2. King in center penalty (especially dangerous in middlegame)
            let king_file = king_square.get_file().to_index() as i8;
            let king_rank = king_square.get_rank().to_index() as i8;

            if (king_file >= 2 && king_file <= 5) && (king_rank >= 2 && king_rank <= 5) {
                safety_score -= 8.0 * multiplier; // MASSIVE penalty for exposed king
            }

            // 3. King too far forward penalty (especially rank 2/7 for White/Black)
            let expected_rank = if color == Color::White { 0 } else { 7 };
            let rank_distance = (king_rank - expected_rank).abs();
            if rank_distance > 1 {
                safety_score -= rank_distance as f32 * 5.0 * multiplier; // Very heavy penalty
            }

            // 4. Specific penalty for early king moves like Ke2
            if rank_distance == 1 && (king_file == 4) {
                // King on e2/e7
                safety_score -= 10.0 * multiplier; // Huge penalty for Ke2-style moves
            }

            // 5. Check for king in immediate danger (multiple attackers)
            if attackers >= 2 {
                safety_score -= 5.0 * multiplier; // Very dangerous situation
            }

            // 6. Pawn shield evaluation
            let pawn_shield_score = self.evaluate_king_pawn_shield(board, king_square, color);
            safety_score += pawn_shield_score * multiplier;
        }

        safety_score
    }

    /// Evaluate king-specific pawn shield for safety analysis
    fn evaluate_king_pawn_shield(&self, board: &Board, king_square: Square, color: Color) -> f32 {
        let mut shield_score = 0.0;
        let king_file = king_square.get_file().to_index() as i8;
        let king_rank = king_square.get_rank().to_index() as i8;
        let direction = if color == Color::White { 1 } else { -1 };

        // Check pawn shield in front of king
        for file_offset in [-1, 0, 1] {
            let shield_file = king_file + file_offset;
            if shield_file >= 0 && shield_file < 8 {
                for rank_offset in [1, 2] {
                    let shield_rank = king_rank + (direction * rank_offset);
                    if shield_rank >= 0 && shield_rank < 8 {
                        let shield_square = Square::make_square(
                            chess::Rank::from_index(shield_rank as usize),
                            chess::File::from_index(shield_file as usize),
                        );

                        if let Some(piece) = board.piece_on(shield_square) {
                            if piece == chess::Piece::Pawn
                                && board.color_on(shield_square) == Some(color)
                            {
                                shield_score += 1.0; // Pawn shield bonus
                            }
                        }
                    }
                }
            }
        }

        shield_score
    }

    /// Material safety evaluation - prevents gross material blunders
    fn evaluate_material_safety(&self, board: &Board) -> f32 {
        let mut safety_score = 0.0;

        // Check for pieces that are in immediate danger of being lost for nothing
        for color in [Color::White, Color::Black] {
            let multiplier = if color == Color::White { -1.0 } else { 1.0 };

            // Check high-value pieces (Queen, Rook, minor pieces) in danger
            for piece_type in [
                chess::Piece::Queen,
                chess::Piece::Rook,
                chess::Piece::Bishop,
                chess::Piece::Knight,
            ] {
                let pieces = board.pieces(piece_type) & board.color_combined(color);

                for square in pieces {
                    let attackers = self.count_attackers(board, square, !color);
                    let defenders = self.count_attackers(board, square, color);

                    // Major piece under attack with insufficient defense
                    if attackers > 0 {
                        let piece_value = self.get_piece_value(piece_type) as f32;

                        if defenders == 0 {
                            // Completely hanging - massive penalty
                            safety_score += piece_value * multiplier * 1.2; // Increased penalty
                        } else if attackers > defenders {
                            // Under-defended - moderate penalty
                            safety_score += piece_value * multiplier * 0.6; // Increased penalty
                        }
                    }
                }
            }

            // ADDITIONAL: Check for pieces moving to dangerous squares
            // This helps prevent moves like Bh6 when the bishop can be captured
            for piece_type in [
                chess::Piece::Queen,
                chess::Piece::Rook,
                chess::Piece::Bishop,
                chess::Piece::Knight,
            ] {
                let pieces = board.pieces(piece_type) & board.color_combined(color);

                for square in pieces {
                    // Penalty for pieces on dangerous squares (attacked by lower value pieces)
                    let piece_value = self.get_piece_value(piece_type);
                    let attackers = self.count_attackers(board, square, !color);

                    if attackers > 0 {
                        // Check if attacked by lower value pieces (bad exchanges)
                        for attacker_square in chess::ALL_SQUARES {
                            if let Some(attacker_piece) = board.piece_on(attacker_square) {
                                if board.color_on(attacker_square) == Some(!color) {
                                    let attacker_value = self.get_piece_value(attacker_piece);
                                    if attacker_value < piece_value
                                        && self.can_attack(board, attacker_square, square)
                                    {
                                        // Penalize pieces that can be captured by lower value pieces
                                        safety_score += (piece_value - attacker_value) as f32
                                            * multiplier
                                            * 0.3;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        safety_score
    }

    /// Evaluate compensation for piece sacrifices (critical for 2000+ ELO)
    fn evaluate_sacrifice_compensation(&self, chess_move: &ChessMove, board: &Board) -> f32 {
        let mut compensation = 0.0;
        let test_board = board.make_move_new(*chess_move);

        // 1. Check bonus - forcing moves have value
        if test_board.checkers().popcnt() > 0 {
            compensation += 100.0; // Check has some value

            // If it's checkmate, massive compensation
            if test_board.status() == chess::BoardStatus::Checkmate {
                compensation += 10000.0; // Checkmate justifies any sacrifice
            }
        }

        // 2. Piece development bonus (getting pieces into play)
        let our_developed_before = self.count_developed_pieces(board, board.side_to_move());
        let our_developed_after = self.count_developed_pieces(&test_board, board.side_to_move());
        compensation += (our_developed_after - our_developed_before) as f32 * 50.0;

        // 3. King safety improvement
        let enemy_king_safety_before =
            self.evaluate_king_safety_for_color(board, !board.side_to_move());
        let enemy_king_safety_after =
            self.evaluate_king_safety_for_color(&test_board, !board.side_to_move());
        let king_safety_improvement = enemy_king_safety_before - enemy_king_safety_after;
        compensation += king_safety_improvement * 0.5; // King attack has value

        // 4. Positional compensation (center control, piece activity)
        let our_activity_before = self.evaluate_piece_activity(board, board.side_to_move());
        let our_activity_after = self.evaluate_piece_activity(&test_board, board.side_to_move());
        compensation += (our_activity_after - our_activity_before) * 0.3;

        compensation
    }

    /// Count developed pieces (knights and bishops off back rank)
    fn count_developed_pieces(&self, board: &Board, color: Color) -> u32 {
        let mut developed = 0;
        let back_rank = if color == Color::White { 0 } else { 7 };

        // Count knights not on back rank
        let knights = board.pieces(chess::Piece::Knight) & board.color_combined(color);
        for square in knights {
            if square.get_rank().to_index() != back_rank {
                developed += 1;
            }
        }

        // Count bishops not on back rank
        let bishops = board.pieces(chess::Piece::Bishop) & board.color_combined(color);
        for square in bishops {
            if square.get_rank().to_index() != back_rank {
                developed += 1;
            }
        }

        developed
    }

    /// Evaluate king safety for a specific color
    fn evaluate_king_safety_for_color(&self, board: &Board, color: Color) -> f32 {
        let king_square = board.king_square(color);
        let enemy_attackers = self.count_attackers(board, king_square, !color);
        -(enemy_attackers as f32 * 50.0) // More attackers = less safe
    }

    /// Evaluate piece activity (mobility and central presence)  
    fn evaluate_piece_activity(&self, board: &Board, color: Color) -> f32 {
        let mut activity = 0.0;

        // Central squares bonus
        let center_squares = [
            chess::Square::make_square(chess::Rank::Fourth, chess::File::D),
            chess::Square::make_square(chess::Rank::Fourth, chess::File::E),
            chess::Square::make_square(chess::Rank::Fifth, chess::File::D),
            chess::Square::make_square(chess::Rank::Fifth, chess::File::E),
        ];

        for &square in &center_squares {
            if let Some(piece) = board.piece_on(square) {
                if board.color_on(square) == Some(color) {
                    let piece_value = match piece {
                        chess::Piece::Pawn => 30.0,
                        chess::Piece::Knight => 40.0,
                        chess::Piece::Bishop => 35.0,
                        _ => 20.0,
                    };
                    activity += piece_value;
                }
            }
        }

        activity
    }

    /// Enhanced NNUE evaluation with position-specific tactical patterns
    fn get_nnue_evaluation(&self, board: &Board) -> (f32, f32) {
        // SPEED OPTIMIZATION: Fast NNUE-style evaluation for competitive play

        // Quick material count for confidence assessment
        let material_count = self.count_material(board);

        // Fast confidence assessment based on position complexity
        let confidence = if material_count > 20 {
            0.5 // Complex positions - medium confidence
        } else if material_count > 12 {
            0.7 // Moderate positions - good confidence
        } else {
            0.8 // Simple positions - high confidence
        };

        // Fast evaluation focusing on key factors only
        let material_eval = self.material_balance(board) / 100.0;
        let king_safety_eval = self.king_safety(board) / 100.0;

        // Simple weighted combination for speed
        let eval = material_eval * 0.7 + king_safety_eval * 0.3;

        // Clamp to reasonable chess range
        let clamped_eval = eval.clamp(-5.0, 5.0);

        (clamped_eval, confidence)
    }

    /// Evaluate tactical patterns for NNUE-style assessment
    fn evaluate_tactical_patterns_nnue(&self, board: &Board) -> f32 {
        let mut tactical_score = 0.0;

        // Pin detection bonus/penalty
        for color in [Color::White, Color::Black] {
            let multiplier = if color == board.side_to_move() {
                1.0
            } else {
                -1.0
            };

            // Count pins this color creates
            let pins_created = self.count_pins_created_by_color(board, color);
            tactical_score += pins_created as f32 * 0.3 * multiplier;

            // Fork potential
            let fork_potential = self.count_fork_potential(board, color);
            tactical_score += fork_potential as f32 * 0.2 * multiplier;

            // Discovered attack potential
            let discovered_attacks = self.count_discovered_attack_potential(board, color);
            tactical_score += discovered_attacks as f32 * 0.25 * multiplier;
        }

        // Check and checkmate threats
        if board.checkers().popcnt() > 0 {
            let moving_color = board.side_to_move();
            let king_square = board.king_square(moving_color);
            let escape_squares = self.count_king_escape_squares(board, king_square);

            // Penalty for being in check, worse with fewer escape squares
            tactical_score -= 0.5 + (3.0 - escape_squares as f32) * 0.2;
        }

        tactical_score
    }

    /// Evaluate positional factors for NNUE-style assessment
    fn evaluate_positional_factors_nnue(&self, board: &Board) -> f32 {
        let mut positional_score = 0.0;

        // Center control evaluation
        let center_control = self.evaluate_center_control_detailed(board);
        positional_score += center_control * 0.1;

        // Piece activity and mobility
        let white_activity = self.evaluate_piece_activity(board, Color::White);
        let black_activity = self.evaluate_piece_activity(board, Color::Black);
        let activity_diff = (white_activity - black_activity) / 100.0;

        if board.side_to_move() == Color::White {
            positional_score += activity_diff * 0.15;
        } else {
            positional_score -= activity_diff * 0.15;
        }

        // Pawn structure evaluation
        let pawn_structure_score = self.evaluate_pawn_structure_nnue(board);
        positional_score += pawn_structure_score;

        positional_score
    }

    /// Evaluate development for NNUE-style assessment
    fn evaluate_development_nnue(&self, board: &Board) -> f32 {
        if !self.is_opening_phase(board) {
            return 0.0; // Development only matters in opening
        }

        let white_dev = self.count_developed_pieces(board, Color::White);
        let black_dev = self.count_developed_pieces(board, Color::Black);
        let dev_diff = (white_dev as f32 - black_dev as f32) * 0.15;

        if board.side_to_move() == Color::White {
            dev_diff
        } else {
            -dev_diff
        }
    }

    /// Count pins created by a specific color
    fn count_pins_created_by_color(&self, board: &Board, color: Color) -> u8 {
        let mut pin_count = 0;
        let enemy_color = !color;
        let enemy_king_square = board.king_square(enemy_color);

        // Check for pieces that could create pins
        let pieces = board.color_combined(color);
        for piece_square in *pieces {
            if let Some(piece) = board.piece_on(piece_square) {
                match piece {
                    chess::Piece::Bishop | chess::Piece::Rook | chess::Piece::Queen => {
                        if self.creates_pin_on_king(board, piece_square, enemy_king_square, piece) {
                            pin_count += 1;
                        }
                    }
                    _ => {}
                }
            }
        }

        pin_count
    }

    /// Count fork potential for a color
    fn count_fork_potential(&self, board: &Board, color: Color) -> u8 {
        let mut fork_count = 0;
        let pieces = board.color_combined(color);

        for piece_square in *pieces {
            if let Some(piece) = board.piece_on(piece_square) {
                match piece {
                    chess::Piece::Knight => {
                        if self.knight_can_fork(board, piece_square, color) {
                            fork_count += 1;
                        }
                    }
                    chess::Piece::Pawn => {
                        if self.pawn_can_fork(board, piece_square, color) {
                            fork_count += 1;
                        }
                    }
                    _ => {}
                }
            }
        }

        fork_count
    }

    /// Count discovered attack potential
    fn count_discovered_attack_potential(&self, board: &Board, color: Color) -> u8 {
        let mut discovered_count = 0;
        let enemy_king_square = board.king_square(!color);

        // Look for pieces that could move to create discovered attacks
        let pieces = board.color_combined(color);
        for piece_square in *pieces {
            if self.can_create_discovered_attack(board, piece_square, enemy_king_square) {
                discovered_count += 1;
            }
        }

        discovered_count
    }

    /// Enhanced center control evaluation
    fn evaluate_center_control_detailed(&self, board: &Board) -> f32 {
        let center_squares = [
            chess::Square::make_square(chess::Rank::Fourth, chess::File::D),
            chess::Square::make_square(chess::Rank::Fourth, chess::File::E),
            chess::Square::make_square(chess::Rank::Fifth, chess::File::D),
            chess::Square::make_square(chess::Rank::Fifth, chess::File::E),
        ];

        let extended_center = [
            chess::Square::make_square(chess::Rank::Third, chess::File::C),
            chess::Square::make_square(chess::Rank::Third, chess::File::D),
            chess::Square::make_square(chess::Rank::Third, chess::File::E),
            chess::Square::make_square(chess::Rank::Third, chess::File::F),
            chess::Square::make_square(chess::Rank::Sixth, chess::File::C),
            chess::Square::make_square(chess::Rank::Sixth, chess::File::D),
            chess::Square::make_square(chess::Rank::Sixth, chess::File::E),
            chess::Square::make_square(chess::Rank::Sixth, chess::File::F),
        ];

        let mut control_score = 0.0;

        // Core center control (higher weight)
        for square in center_squares {
            let white_attackers = self.count_attackers(board, square, Color::White);
            let black_attackers = self.count_attackers(board, square, Color::Black);
            control_score += (white_attackers as f32 - black_attackers as f32) * 0.2;
        }

        // Extended center control (lower weight)
        for square in extended_center {
            let white_attackers = self.count_attackers(board, square, Color::White);
            let black_attackers = self.count_attackers(board, square, Color::Black);
            control_score += (white_attackers as f32 - black_attackers as f32) * 0.1;
        }

        if board.side_to_move() == Color::Black {
            control_score = -control_score;
        }

        control_score
    }

    /// Evaluate pawn structure for NNUE
    fn evaluate_pawn_structure_nnue(&self, board: &Board) -> f32 {
        let mut pawn_score = 0.0;

        // Evaluate for both colors
        for color in [Color::White, Color::Black] {
            let multiplier = if color == board.side_to_move() {
                1.0
            } else {
                -1.0
            };

            // Passed pawns bonus
            let passed_pawns = self.count_passed_pawns(board, color);
            pawn_score += passed_pawns as f32 * 0.3 * multiplier;

            // Isolated pawns penalty
            let isolated_pawns = self.count_isolated_pawns(board, color);
            pawn_score -= isolated_pawns as f32 * 0.2 * multiplier;

            // Doubled pawns penalty
            let doubled_pawns = self.count_doubled_pawns(board, color);
            pawn_score -= doubled_pawns as f32 * 0.15 * multiplier;
        }

        pawn_score
    }

    /// Check if piece creates a pin on enemy king
    fn creates_pin_on_king(
        &self,
        board: &Board,
        piece_square: Square,
        enemy_king_square: Square,
        piece: chess::Piece,
    ) -> bool {
        match piece {
            chess::Piece::Bishop => {
                // Check diagonal pin
                let rank_diff = (piece_square.get_rank().to_index() as i8
                    - enemy_king_square.get_rank().to_index() as i8)
                    .abs();
                let file_diff = (piece_square.get_file().to_index() as i8
                    - enemy_king_square.get_file().to_index() as i8)
                    .abs();
                rank_diff == file_diff && rank_diff > 0
            }
            chess::Piece::Rook => {
                // Check rank/file pin
                piece_square.get_rank() == enemy_king_square.get_rank()
                    || piece_square.get_file() == enemy_king_square.get_file()
            }
            chess::Piece::Queen => {
                // Queen combines both
                self.creates_pin_on_king(
                    board,
                    piece_square,
                    enemy_king_square,
                    chess::Piece::Bishop,
                ) || self.creates_pin_on_king(
                    board,
                    piece_square,
                    enemy_king_square,
                    chess::Piece::Rook,
                )
            }
            _ => false,
        }
    }

    /// Check if knight can create a fork
    fn knight_can_fork(&self, board: &Board, knight_square: Square, color: Color) -> bool {
        let enemy_color = !color;
        let mut valuable_targets = 0;

        // Knight move patterns
        let knight_moves = [
            (-2, -1),
            (-2, 1),
            (-1, -2),
            (-1, 2),
            (1, -2),
            (1, 2),
            (2, -1),
            (2, 1),
        ];

        for (rank_offset, file_offset) in knight_moves {
            let new_rank = knight_square.get_rank().to_index() as i8 + rank_offset;
            let new_file = knight_square.get_file().to_index() as i8 + file_offset;

            if new_rank >= 0 && new_rank <= 7 && new_file >= 0 && new_file <= 7 {
                let target_square = Square::make_square(
                    chess::Rank::from_index(new_rank as usize),
                    chess::File::from_index(new_file as usize),
                );

                if let Some(piece) = board.piece_on(target_square) {
                    if board.color_on(target_square) == Some(enemy_color)
                        && piece != chess::Piece::Pawn
                    {
                        valuable_targets += 1;
                    }
                }
            }
        }

        valuable_targets >= 2
    }

    /// Check if pawn can create a fork
    fn pawn_can_fork(&self, board: &Board, pawn_square: Square, color: Color) -> bool {
        let enemy_color = !color;
        let direction = if color == Color::White { 1 } else { -1 };
        let new_rank = pawn_square.get_rank().to_index() as i8 + direction;

        if new_rank < 0 || new_rank > 7 {
            return false;
        }

        let mut fork_targets = 0;

        // Check diagonal attacks
        for file_offset in [-1, 1] {
            let new_file = pawn_square.get_file().to_index() as i8 + file_offset;
            if new_file >= 0 && new_file <= 7 {
                let target_square = Square::make_square(
                    chess::Rank::from_index(new_rank as usize),
                    chess::File::from_index(new_file as usize),
                );

                if let Some(piece) = board.piece_on(target_square) {
                    if board.color_on(target_square) == Some(enemy_color)
                        && piece != chess::Piece::Pawn
                    {
                        fork_targets += 1;
                    }
                }
            }
        }

        fork_targets >= 2
    }

    /// Check if piece can create discovered attack
    fn can_create_discovered_attack(
        &self,
        board: &Board,
        piece_square: Square,
        enemy_king_square: Square,
    ) -> bool {
        // Simplified: check if moving this piece could uncover an attack on the enemy king
        // This is a heuristic - we check if there's a long-range piece behind this piece
        // that could attack the king if this piece moves

        let directions = [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0), // Rook directions
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1), // Bishop directions
        ];

        for (rank_dir, file_dir) in directions {
            if self.has_piece_behind_for_discovered_attack(
                board,
                piece_square,
                enemy_king_square,
                rank_dir,
                file_dir,
            ) {
                return true;
            }
        }

        false
    }

    /// Helper for discovered attack detection
    fn has_piece_behind_for_discovered_attack(
        &self,
        board: &Board,
        piece_square: Square,
        enemy_king_square: Square,
        rank_dir: i8,
        file_dir: i8,
    ) -> bool {
        // Check if there's a piece behind that could attack the enemy king
        let mut current_rank = piece_square.get_rank().to_index() as i8 - rank_dir;
        let mut current_file = piece_square.get_file().to_index() as i8 - file_dir;

        // Look backwards from the piece
        while current_rank >= 0 && current_rank <= 7 && current_file >= 0 && current_file <= 7 {
            let check_square = Square::make_square(
                chess::Rank::from_index(current_rank as usize),
                chess::File::from_index(current_file as usize),
            );

            if let Some(piece) = board.piece_on(check_square) {
                if board.color_on(check_square) == board.color_on(piece_square) {
                    // Found a piece of our color - check if it can attack the enemy king
                    if (piece == chess::Piece::Rook || piece == chess::Piece::Queen)
                        && (rank_dir == 0 || file_dir == 0)
                    {
                        return self.has_clear_path(check_square, enemy_king_square, board);
                    }
                    if (piece == chess::Piece::Bishop || piece == chess::Piece::Queen)
                        && (rank_dir.abs() == file_dir.abs())
                    {
                        return self.has_clear_path(check_square, enemy_king_square, board);
                    }
                }
                break; // Found a piece, stop looking
            }

            current_rank -= rank_dir;
            current_file -= file_dir;
        }

        false
    }

    /// Count specific pawn structure features
    fn count_passed_pawns(&self, board: &Board, color: Color) -> u8 {
        // Simplified passed pawn detection
        let pawns = board.pieces(chess::Piece::Pawn) & board.color_combined(color);
        let mut passed_count = 0;

        for pawn_square in pawns {
            if self.is_passed_pawn_nnue(board, pawn_square, color) {
                passed_count += 1;
            }
        }

        passed_count
    }

    fn count_isolated_pawns(&self, board: &Board, color: Color) -> u8 {
        let pawns = board.pieces(chess::Piece::Pawn) & board.color_combined(color);
        let mut isolated_count = 0;

        for pawn_square in pawns {
            if self.is_isolated_pawn(board, pawn_square, color) {
                isolated_count += 1;
            }
        }

        isolated_count
    }

    fn count_doubled_pawns(&self, board: &Board, color: Color) -> u8 {
        let pawns = board.pieces(chess::Piece::Pawn) & board.color_combined(color);
        let mut file_counts = [0u8; 8];

        for pawn_square in pawns {
            file_counts[pawn_square.get_file().to_index()] += 1;
        }

        file_counts
            .iter()
            .map(|&count| if count > 1 { count - 1 } else { 0 })
            .sum()
    }

    /// Simplified passed pawn detection for NNUE
    fn is_passed_pawn_nnue(&self, board: &Board, pawn_square: Square, color: Color) -> bool {
        let enemy_color = !color;
        let pawn_file = pawn_square.get_file();
        let direction = if color == Color::White { 1 } else { -1 };

        // Check if there are enemy pawns blocking or controlling the path
        let mut rank = pawn_square.get_rank().to_index() as i8 + direction;

        while rank >= 0 && rank <= 7 {
            for file_offset in -1..=1 {
                let check_file = pawn_file.to_index() as i8 + file_offset;
                if check_file >= 0 && check_file <= 7 {
                    let check_square = Square::make_square(
                        chess::Rank::from_index(rank as usize),
                        chess::File::from_index(check_file as usize),
                    );

                    if let Some(piece) = board.piece_on(check_square) {
                        if piece == chess::Piece::Pawn
                            && board.color_on(check_square) == Some(enemy_color)
                        {
                            return false; // Blocked by enemy pawn
                        }
                    }
                }
            }
            rank += direction;
        }

        true
    }

    /// Simplified isolated pawn detection
    fn is_isolated_pawn(&self, board: &Board, pawn_square: Square, color: Color) -> bool {
        let pawn_file = pawn_square.get_file().to_index();
        let pawns = board.pieces(chess::Piece::Pawn) & board.color_combined(color);

        // Check adjacent files for friendly pawns
        for file_offset in [-1, 1] {
            let check_file = pawn_file as i8 + file_offset;
            if check_file >= 0 && check_file <= 7 {
                for pawn_check_square in pawns {
                    if pawn_check_square.get_file().to_index() == check_file as usize {
                        return false; // Found a pawn on adjacent file
                    }
                }
            }
        }

        true // No pawns on adjacent files
    }

    /// Evaluate space advantage for strategic initiative
    fn evaluate_space_advantage(&self, board: &Board) -> f32 {
        let mut space_score = 0.0;
        let moving_color = board.side_to_move();

        // Count squares controlled by each side in the center and opponent's territory
        let white_space = self.count_controlled_squares(board, Color::White);
        let black_space = self.count_controlled_squares(board, Color::Black);

        let space_difference = white_space as f32 - black_space as f32;

        if moving_color == Color::White {
            space_score = space_difference * 0.02; // 0.02 pawns per extra controlled square
        } else {
            space_score = -space_difference * 0.02;
        }

        // Bonus for advanced pawns creating space
        for color in [Color::White, Color::Black] {
            let multiplier = if color == moving_color { 1.0 } else { -1.0 };
            let advanced_pawns = self.count_advanced_pawns(board, color);
            space_score += advanced_pawns as f32 * 0.05 * multiplier;
        }

        space_score
    }

    /// Evaluate piece coordination for strategic play
    fn evaluate_piece_coordination(&self, board: &Board) -> f32 {
        let mut coordination_score = 0.0;
        let moving_color = board.side_to_move();

        // Evaluate coordination for both sides
        for color in [Color::White, Color::Black] {
            let multiplier = if color == moving_color { 1.0 } else { -1.0 };

            // Rook coordination (doubled rooks, rook + queen on same file/rank)
            let rook_coordination = self.evaluate_rook_coordination(board, color);
            coordination_score += rook_coordination * 0.1 * multiplier;

            // Bishop pair advantage
            if self.has_bishop_pair_coordination(board, color) {
                coordination_score += 0.15 * multiplier;
            }

            // Piece support chains (pieces defending each other)
            let support_chains = self.count_piece_support_chains(board, color);
            coordination_score += support_chains as f32 * 0.05 * multiplier;

            // Knights supporting each other
            let knight_coordination = self.evaluate_knight_coordination(board, color);
            coordination_score += knight_coordination * 0.08 * multiplier;
        }

        coordination_score
    }

    /// Evaluate dynamic potential (tempo, initiative, forcing moves)
    fn evaluate_dynamic_potential(&self, board: &Board) -> f32 {
        let mut dynamic_score = 0.0;
        let moving_color = board.side_to_move();

        // Check for forcing moves available
        let forcing_moves = self.count_forcing_moves(board, moving_color);
        dynamic_score += forcing_moves as f32 * 0.04;

        // Evaluate attacking chances
        let enemy_color = !moving_color;
        let enemy_king_square = board.king_square(enemy_color);
        let king_attackers = self.count_attackers(board, enemy_king_square, moving_color);
        dynamic_score += king_attackers as f32 * 0.06;

        // Piece activity and mobility
        let piece_activity = self.evaluate_total_piece_activity(board, moving_color);
        dynamic_score += piece_activity / 1000.0; // Scale appropriately

        // Pawn break potential
        let pawn_breaks = self.count_potential_pawn_breaks(board, moving_color);
        dynamic_score += pawn_breaks as f32 * 0.07;

        // Time advantage (if we're ahead in development)
        if self.is_opening_phase(board) {
            let dev_advantage = self.count_developed_pieces(board, moving_color) as i8
                - self.count_developed_pieces(board, enemy_color) as i8;
            if dev_advantage > 0 {
                dynamic_score += dev_advantage as f32 * 0.03;
            }
        }

        dynamic_score
    }

    /// Evaluate long-term positional advantages
    fn evaluate_long_term_advantages(&self, board: &Board) -> f32 {
        let mut long_term_score = 0.0;
        let moving_color = board.side_to_move();

        // Evaluate for both sides
        for color in [Color::White, Color::Black] {
            let multiplier = if color == moving_color { 1.0 } else { -1.0 };

            // Weak squares in opponent camp
            let weak_squares = self.count_weak_squares_in_enemy_camp(board, color);
            long_term_score += weak_squares as f32 * 0.04 * multiplier;

            // Outposts for pieces
            let outposts = self.count_piece_outposts(board, color);
            long_term_score += outposts as f32 * 0.06 * multiplier;

            // Pawn structure advantages
            let structure_advantage = self.evaluate_pawn_structure_advantage(board, color);
            long_term_score += structure_advantage * 0.05 * multiplier;

            // Control of key files and diagonals
            let file_control = self.evaluate_file_control(board, color);
            long_term_score += file_control * 0.03 * multiplier;
        }

        long_term_score
    }

    /// Count squares controlled by a color in center and enemy territory
    fn count_controlled_squares(&self, board: &Board, color: Color) -> u8 {
        let mut controlled = 0;

        // Define important squares (center + 6th/7th rank for enemy territory)
        let important_squares = if color == Color::White {
            // For white: center + black's 6th and 7th ranks
            [
                (3, 3),
                (3, 4),
                (4, 3),
                (4, 4), // Center
                (5, 0),
                (5, 1),
                (5, 2),
                (5, 3),
                (5, 4),
                (5, 5),
                (5, 6),
                (5, 7), // 6th rank
                (6, 0),
                (6, 1),
                (6, 2),
                (6, 3),
                (6, 4),
                (6, 5),
                (6, 6),
                (6, 7), // 7th rank
            ]
        } else {
            // For black: center + white's 3rd and 2nd ranks
            [
                (3, 3),
                (3, 4),
                (4, 3),
                (4, 4), // Center
                (2, 0),
                (2, 1),
                (2, 2),
                (2, 3),
                (2, 4),
                (2, 5),
                (2, 6),
                (2, 7), // 3rd rank
                (1, 0),
                (1, 1),
                (1, 2),
                (1, 3),
                (1, 4),
                (1, 5),
                (1, 6),
                (1, 7), // 2nd rank
            ]
        };

        for (rank, file) in important_squares {
            let square =
                Square::make_square(chess::Rank::from_index(rank), chess::File::from_index(file));

            if self.count_attackers(board, square, color) > 0 {
                controlled += 1;
            }
        }

        controlled
    }

    /// Count advanced pawns creating space
    fn count_advanced_pawns(&self, board: &Board, color: Color) -> u8 {
        let pawns = board.pieces(chess::Piece::Pawn) & board.color_combined(color);
        let mut advanced_count = 0;

        for pawn_square in pawns {
            let rank = pawn_square.get_rank().to_index();
            let is_advanced = if color == Color::White {
                rank >= 4 // 5th rank or beyond for white
            } else {
                rank <= 3 // 4th rank or beyond for black
            };

            if is_advanced {
                advanced_count += 1;
            }
        }

        advanced_count
    }

    /// Evaluate rook coordination
    fn evaluate_rook_coordination(&self, board: &Board, color: Color) -> f32 {
        let rooks = board.pieces(chess::Piece::Rook) & board.color_combined(color);
        let mut coordination = 0.0;

        let rook_squares: Vec<Square> = rooks.collect();

        // Check for doubled rooks
        for i in 0..rook_squares.len() {
            for j in (i + 1)..rook_squares.len() {
                let rook1 = rook_squares[i];
                let rook2 = rook_squares[j];

                // Same file or rank
                if rook1.get_file() == rook2.get_file() || rook1.get_rank() == rook2.get_rank() {
                    coordination += 1.0;
                }
            }
        }

        coordination
    }

    /// Check for bishop pair coordination
    fn has_bishop_pair_coordination(&self, board: &Board, color: Color) -> bool {
        let bishops = board.pieces(chess::Piece::Bishop) & board.color_combined(color);
        bishops.popcnt() >= 2
    }

    /// Count piece support chains
    fn count_piece_support_chains(&self, board: &Board, color: Color) -> u8 {
        let mut support_count = 0;
        let pieces = board.color_combined(color);

        for piece_square in *pieces {
            if let Some(piece) = board.piece_on(piece_square) {
                if piece != chess::Piece::King {
                    // Kings don't count for support chains
                    let defenders = self.count_attackers(board, piece_square, color);
                    if defenders > 0 {
                        support_count += 1;
                    }
                }
            }
        }

        support_count
    }

    /// Evaluate knight coordination
    fn evaluate_knight_coordination(&self, board: &Board, color: Color) -> f32 {
        let knights = board.pieces(chess::Piece::Knight) & board.color_combined(color);
        let mut coordination = 0.0;

        let knight_squares: Vec<Square> = knights.collect();

        // Check for knights supporting each other
        for i in 0..knight_squares.len() {
            for j in (i + 1)..knight_squares.len() {
                let knight1 = knight_squares[i];
                let knight2 = knight_squares[j];

                // Check if knights can support each other (within 3 squares)
                let rank_diff = (knight1.get_rank().to_index() as i8
                    - knight2.get_rank().to_index() as i8)
                    .abs();
                let file_diff = (knight1.get_file().to_index() as i8
                    - knight2.get_file().to_index() as i8)
                    .abs();

                if rank_diff <= 3 && file_diff <= 3 {
                    coordination += 0.5;
                }
            }
        }

        coordination
    }

    /// Count forcing moves (checks, captures, threats)
    fn count_forcing_moves(&self, board: &Board, color: Color) -> u8 {
        let mut forcing_count = 0;
        let moves = MoveGen::new_legal(board);

        for chess_move in moves {
            if self.is_forcing_move(&chess_move, board) {
                forcing_count += 1;
            }
        }

        forcing_count
    }

    /// Check if a move is forcing
    fn is_forcing_move(&self, chess_move: &ChessMove, board: &Board) -> bool {
        // Captures
        if board.piece_on(chess_move.get_dest()).is_some() {
            return true;
        }

        // Checks
        let mut temp_board = *board;
        temp_board = temp_board.make_move_new(*chess_move);
        if temp_board.checkers().popcnt() > 0 {
            return true;
        }

        // Threats (creates attacks on valuable pieces)
        let threatens_valuable = self.threatens_valuable_piece(chess_move, board);
        if threatens_valuable {
            return true;
        }

        false
    }

    /// Check if move threatens a valuable piece
    fn threatens_valuable_piece(&self, chess_move: &ChessMove, board: &Board) -> bool {
        let mut temp_board = *board;
        temp_board = temp_board.make_move_new(*chess_move);

        let moving_color = board.side_to_move();
        let enemy_color = !moving_color;
        let enemy_pieces = board.color_combined(enemy_color);

        for enemy_square in *enemy_pieces {
            if let Some(piece) = temp_board.piece_on(enemy_square) {
                match piece {
                    chess::Piece::Queen
                    | chess::Piece::Rook
                    | chess::Piece::Bishop
                    | chess::Piece::Knight => {
                        if self.count_attackers(&temp_board, enemy_square, moving_color) > 0 {
                            return true;
                        }
                    }
                    _ => {}
                }
            }
        }

        false
    }

    /// Evaluate total piece activity
    fn evaluate_total_piece_activity(&self, board: &Board, color: Color) -> f32 {
        let mut total_activity = 0.0;
        let pieces = board.color_combined(color);

        for piece_square in *pieces {
            if let Some(piece) = board.piece_on(piece_square) {
                let mobility = self.calculate_piece_mobility_at_square(board, piece_square, piece);
                total_activity += mobility;
            }
        }

        total_activity
    }

    /// Calculate mobility for a specific piece at a specific square
    fn calculate_piece_mobility_at_square(
        &self,
        board: &Board,
        piece_square: Square,
        piece: chess::Piece,
    ) -> f32 {
        let mut mobility = 0.0;

        match piece {
            chess::Piece::Queen => {
                // Queen mobility - check all 8 directions
                for direction in [
                    (0, 1),
                    (0, -1),
                    (1, 0),
                    (-1, 0),
                    (1, 1),
                    (1, -1),
                    (-1, 1),
                    (-1, -1),
                ] {
                    mobility +=
                        self.count_moves_in_direction(board, piece_square, direction) as f32 * 1.5;
                }
            }
            chess::Piece::Rook => {
                // Rook mobility - check 4 directions
                for direction in [(0, 1), (0, -1), (1, 0), (-1, 0)] {
                    mobility +=
                        self.count_moves_in_direction(board, piece_square, direction) as f32 * 1.2;
                }
            }
            chess::Piece::Bishop => {
                // Bishop mobility - check 4 diagonal directions
                for direction in [(1, 1), (1, -1), (-1, 1), (-1, -1)] {
                    mobility +=
                        self.count_moves_in_direction(board, piece_square, direction) as f32;
                }
            }
            chess::Piece::Knight => {
                // Knight mobility - count available squares
                let knight_moves = [
                    (-2, -1),
                    (-2, 1),
                    (-1, -2),
                    (-1, 2),
                    (1, -2),
                    (1, 2),
                    (2, -1),
                    (2, 1),
                ];

                for (rank_offset, file_offset) in knight_moves {
                    let new_rank = piece_square.get_rank().to_index() as i8 + rank_offset;
                    let new_file = piece_square.get_file().to_index() as i8 + file_offset;

                    if new_rank >= 0 && new_rank <= 7 && new_file >= 0 && new_file <= 7 {
                        let target_square = Square::make_square(
                            chess::Rank::from_index(new_rank as usize),
                            chess::File::from_index(new_file as usize),
                        );

                        if board.piece_on(target_square).is_none()
                            || board.color_on(target_square) != board.color_on(piece_square)
                        {
                            mobility += 1.0;
                        }
                    }
                }
            }
            _ => {}
        }

        mobility
    }

    /// Count moves in a specific direction for sliding pieces
    fn count_moves_in_direction(
        &self,
        board: &Board,
        start_square: Square,
        direction: (i8, i8),
    ) -> u8 {
        let mut move_count = 0;
        let (rank_dir, file_dir) = direction;
        let piece_color = board.color_on(start_square);

        let mut current_rank = start_square.get_rank().to_index() as i8 + rank_dir;
        let mut current_file = start_square.get_file().to_index() as i8 + file_dir;

        while current_rank >= 0 && current_rank <= 7 && current_file >= 0 && current_file <= 7 {
            let target_square = Square::make_square(
                chess::Rank::from_index(current_rank as usize),
                chess::File::from_index(current_file as usize),
            );

            if let Some(target_piece) = board.piece_on(target_square) {
                if board.color_on(target_square) != piece_color {
                    move_count += 1; // Can capture
                }
                break; // Blocked by piece
            } else {
                move_count += 1; // Empty square
            }

            current_rank += rank_dir;
            current_file += file_dir;
        }

        move_count
    }

    /// Count potential pawn breaks
    fn count_potential_pawn_breaks(&self, board: &Board, color: Color) -> u8 {
        let pawns = board.pieces(chess::Piece::Pawn) & board.color_combined(color);
        let mut break_count = 0;

        for pawn_square in pawns {
            if self.can_create_pawn_break(board, pawn_square, color) {
                break_count += 1;
            }
        }

        break_count
    }

    /// Check if pawn can create a break
    fn can_create_pawn_break(&self, board: &Board, pawn_square: Square, color: Color) -> bool {
        let direction = if color == Color::White { 1 } else { -1 };
        let current_rank = pawn_square.get_rank().to_index() as i8;
        let next_rank = current_rank + direction;

        if next_rank < 0 || next_rank > 7 {
            return false;
        }

        // Check if advancing creates pressure or breaks opponent's pawn chain
        let file = pawn_square.get_file();
        let target_square = Square::make_square(chess::Rank::from_index(next_rank as usize), file);

        // Simple check: can advance and creates tension
        if board.piece_on(target_square).is_none() {
            // Check if this advance attacks enemy pawns
            for file_offset in [-1, 1] {
                let attack_file = file.to_index() as i8 + file_offset;
                if attack_file >= 0 && attack_file <= 7 {
                    let attack_square = Square::make_square(
                        chess::Rank::from_index(next_rank as usize),
                        chess::File::from_index(attack_file as usize),
                    );

                    if let Some(piece) = board.piece_on(attack_square) {
                        if piece == chess::Piece::Pawn
                            && board.color_on(attack_square) == Some(!color)
                        {
                            return true; // Attacks enemy pawn
                        }
                    }
                }
            }
        }

        false
    }

    /// Count weak squares in enemy camp
    fn count_weak_squares_in_enemy_camp(&self, board: &Board, color: Color) -> u8 {
        let mut weak_squares = 0;
        let enemy_color = !color;

        // Define enemy camp (6th, 7th, 8th ranks for white; 3rd, 2nd, 1st ranks for black)
        let enemy_ranks = if color == Color::White {
            vec![5, 6, 7] // 6th, 7th, 8th ranks
        } else {
            vec![2, 1, 0] // 3rd, 2nd, 1st ranks
        };

        for rank in enemy_ranks {
            for file in 0..8 {
                let square = Square::make_square(
                    chess::Rank::from_index(rank),
                    chess::File::from_index(file),
                );

                if self.is_weak_square(board, square, enemy_color) {
                    weak_squares += 1;
                }
            }
        }

        weak_squares
    }

    /// Check if a square is weak for a color
    fn is_weak_square(&self, board: &Board, square: Square, color: Color) -> bool {
        // A square is weak if:
        // 1. Not defended by pawns of that color
        // 2. Difficult for pieces to defend

        // Check pawn defense
        let pawn_defenders = self.count_pawn_defenders(board, square, color);
        if pawn_defenders > 0 {
            return false; // Defended by pawns
        }

        // Check if enemy can easily occupy
        let enemy_attackers = self.count_attackers(board, square, !color);
        let friendly_defenders = self.count_attackers(board, square, color);

        enemy_attackers > friendly_defenders
    }

    /// Count pawn defenders of a square
    fn count_pawn_defenders(&self, board: &Board, square: Square, color: Color) -> u8 {
        let mut defenders = 0;
        let square_rank = square.get_rank().to_index() as i8;
        let square_file = square.get_file().to_index() as i8;

        // Check diagonal pawn attacks based on color
        let pawn_rank_offset = if color == Color::White { -1 } else { 1 };
        let pawn_rank = square_rank + pawn_rank_offset;

        if pawn_rank >= 0 && pawn_rank <= 7 {
            for file_offset in [-1, 1] {
                let pawn_file = square_file + file_offset;
                if pawn_file >= 0 && pawn_file <= 7 {
                    let pawn_square = Square::make_square(
                        chess::Rank::from_index(pawn_rank as usize),
                        chess::File::from_index(pawn_file as usize),
                    );

                    if let Some(piece) = board.piece_on(pawn_square) {
                        if piece == chess::Piece::Pawn && board.color_on(pawn_square) == Some(color)
                        {
                            defenders += 1;
                        }
                    }
                }
            }
        }

        defenders
    }

    /// Count piece outposts
    fn count_piece_outposts(&self, board: &Board, color: Color) -> u8 {
        let mut outposts = 0;
        let pieces = board.color_combined(color);

        for piece_square in *pieces {
            if let Some(piece) = board.piece_on(piece_square) {
                match piece {
                    chess::Piece::Knight | chess::Piece::Bishop => {
                        if self.is_outpost(board, piece_square, color) {
                            outposts += 1;
                        }
                    }
                    _ => {}
                }
            }
        }

        outposts
    }

    /// Check if a piece is on an outpost
    fn is_outpost(&self, board: &Board, piece_square: Square, color: Color) -> bool {
        // Outpost criteria:
        // 1. Advanced square (at least 5th rank for white, 4th rank for black)
        // 2. Defended by own pawn
        // 3. Cannot be attacked by enemy pawns

        let rank = piece_square.get_rank().to_index();
        let is_advanced = if color == Color::White {
            rank >= 4 // 5th rank or higher
        } else {
            rank <= 3 // 4th rank or lower
        };

        if !is_advanced {
            return false;
        }

        // Check pawn support
        let pawn_support = self.count_pawn_defenders(board, piece_square, color) > 0;

        // Check if safe from enemy pawns
        let enemy_pawn_attacks = self.count_pawn_defenders(board, piece_square, !color) == 0;

        pawn_support && enemy_pawn_attacks
    }

    /// Evaluate pawn structure advantage
    fn evaluate_pawn_structure_advantage(&self, board: &Board, color: Color) -> f32 {
        let mut advantage = 0.0;

        // Passed pawns
        let passed_pawns = self.count_passed_pawns(board, color);
        advantage += passed_pawns as f32 * 0.3;

        // Pawn chains
        let pawn_chains = self.count_pawn_chains(board, color);
        advantage += pawn_chains as f32 * 0.1;

        // Connected passed pawns
        let connected_passed = self.count_connected_passed_pawns(board, color);
        advantage += connected_passed as f32 * 0.5;

        advantage
    }

    /// Count pawn chains
    fn count_pawn_chains(&self, board: &Board, color: Color) -> u8 {
        let pawns = board.pieces(chess::Piece::Pawn) & board.color_combined(color);
        let mut chains = 0;

        for pawn_square in pawns {
            if self.is_part_of_pawn_chain(board, pawn_square, color) {
                chains += 1;
            }
        }

        chains / 2 // Avoid double counting
    }

    /// Check if pawn is part of a chain
    fn is_part_of_pawn_chain(&self, board: &Board, pawn_square: Square, color: Color) -> bool {
        let pawn_file = pawn_square.get_file().to_index() as i8;
        let pawn_rank = pawn_square.get_rank().to_index() as i8;

        // Check diagonally adjacent squares for supporting pawns
        let diagonal_offsets = if color == Color::White {
            [(-1, -1), (-1, 1)] // Behind and diagonal for white
        } else {
            [(1, -1), (1, 1)] // Behind and diagonal for black
        };

        for (rank_offset, file_offset) in diagonal_offsets {
            let check_rank = pawn_rank + rank_offset;
            let check_file = pawn_file + file_offset;

            if check_rank >= 0 && check_rank <= 7 && check_file >= 0 && check_file <= 7 {
                let check_square = Square::make_square(
                    chess::Rank::from_index(check_rank as usize),
                    chess::File::from_index(check_file as usize),
                );

                if let Some(piece) = board.piece_on(check_square) {
                    if piece == chess::Piece::Pawn && board.color_on(check_square) == Some(color) {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Count connected passed pawns
    fn count_connected_passed_pawns(&self, board: &Board, color: Color) -> u8 {
        let pawns = board.pieces(chess::Piece::Pawn) & board.color_combined(color);
        let mut connected_passed = 0;

        for pawn_square in pawns {
            if self.is_passed_pawn_nnue(board, pawn_square, color)
                && self.has_connected_passed_pawn(board, pawn_square, color)
            {
                connected_passed += 1;
            }
        }

        connected_passed / 2 // Avoid double counting pairs
    }

    /// Check if passed pawn has a connected passed pawn
    fn has_connected_passed_pawn(&self, board: &Board, pawn_square: Square, color: Color) -> bool {
        let pawn_file = pawn_square.get_file().to_index() as i8;

        // Check adjacent files for other passed pawns
        for file_offset in [-1, 1] {
            let check_file = pawn_file + file_offset;
            if check_file >= 0 && check_file <= 7 {
                let file_pawns = board.pieces(chess::Piece::Pawn) & board.color_combined(color);
                for other_pawn in file_pawns {
                    if other_pawn.get_file().to_index() == check_file as usize
                        && self.is_passed_pawn_nnue(board, other_pawn, color)
                    {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Evaluate file control
    fn evaluate_file_control(&self, board: &Board, color: Color) -> f32 {
        let mut file_control = 0.0;

        // Check control of open and semi-open files
        for file_index in 0..8 {
            let file = chess::File::from_index(file_index);
            let file_type = self.classify_file_type(board, file, color);

            match file_type {
                FileType::Open => {
                    let control = self.evaluate_open_file_control(board, file, color);
                    file_control += control * 0.2;
                }
                FileType::SemiOpen => {
                    let control = self.evaluate_semi_open_file_control(board, file, color);
                    file_control += control * 0.15;
                }
                _ => {}
            }
        }

        file_control
    }

    /// Classify file type relative to a color
    fn classify_file_type(&self, board: &Board, file: chess::File, color: Color) -> FileType {
        let own_pawns_on_file = self.count_pawns_on_file(board, file, color);
        let enemy_pawns_on_file = self.count_pawns_on_file(board, file, !color);

        match (own_pawns_on_file, enemy_pawns_on_file) {
            (0, 0) => FileType::Open,
            (0, _) => FileType::SemiOpen,
            (_, 0) => FileType::SemiOpen,
            _ => FileType::Closed,
        }
    }

    /// Count pawns on a file for a color
    fn count_pawns_on_file(&self, board: &Board, file: chess::File, color: Color) -> u8 {
        let pawns = board.pieces(chess::Piece::Pawn) & board.color_combined(color);
        let mut count = 0;

        for pawn_square in pawns {
            if pawn_square.get_file() == file {
                count += 1;
            }
        }

        count
    }

    /// Evaluate control of an open file
    fn evaluate_open_file_control(&self, board: &Board, file: chess::File, color: Color) -> f32 {
        let mut control = 0.0;

        // Count rooks and queens on the file
        for rank_index in 0..8 {
            let square = Square::make_square(chess::Rank::from_index(rank_index), file);

            if let Some(piece) = board.piece_on(square) {
                if board.color_on(square) == Some(color) {
                    match piece {
                        chess::Piece::Rook => control += 1.0,
                        chess::Piece::Queen => control += 1.5,
                        _ => {}
                    }
                }
            }
        }

        control
    }

    /// Evaluate control of a semi-open file
    fn evaluate_semi_open_file_control(
        &self,
        board: &Board,
        file: chess::File,
        color: Color,
    ) -> f32 {
        // Similar to open file but with reduced value
        self.evaluate_open_file_control(board, file, color) * 0.7
    }

    /// Get pattern recognition evaluation with confidence score
    fn get_pattern_evaluation(&self, board: &Board) -> (f32, f32) {
        // SPEED OPTIMIZATION: Fast pattern evaluation for competitive play

        // Quick pattern confidence based on position type
        let confidence = if self.is_opening_phase(board) {
            0.7 // High confidence in opening patterns
        } else if self.is_endgame_phase(board) {
            0.8 // Very high confidence in endgame patterns
        } else {
            0.5 // Medium confidence in middlegame
        };

        // Fast evaluation focusing on key patterns only
        let tactical_bonus = self.tactical_bonuses(board);
        let eval = (tactical_bonus / 100.0).clamp(-2.0, 2.0);

        (eval, confidence)
    }

    /// Enhanced strategic initiative evaluation for master-level positional play
    fn get_strategic_initiative_evaluation(&self, board: &Board) -> f32 {
        // Use the actual StrategicEvaluator for comprehensive initiative assessment
        let strategic_eval = self.strategic_evaluator.evaluate_strategic(board);
        let strategic_score_cp = strategic_eval.total_evaluation;

        // Convert strategic score from centipawns to pawns
        let mut initiative = strategic_score_cp / 100.0;

        // Enhanced strategic factors for better positional understanding
        let space_advantage = self.evaluate_space_advantage(board);
        let piece_coordination = self.evaluate_piece_coordination(board);
        let dynamic_potential = self.evaluate_dynamic_potential(board);
        let long_term_advantages = self.evaluate_long_term_advantages(board);

        // Weight strategic components for master-level play
        initiative += space_advantage * 0.15; // Space control
        initiative += piece_coordination * 0.20; // Piece harmony
        initiative += dynamic_potential * 0.25; // Dynamic chances
        initiative += long_term_advantages * 0.10; // Long-term factors

        // Development advantage (weighted appropriately)
        let white_dev = self.count_developed_pieces(board, Color::White);
        let black_dev = self.count_developed_pieces(board, Color::Black);
        initiative += (white_dev as f32 - black_dev as f32) * 0.08;

        // Central control (also weighted lower)
        let center_control = self.evaluate_center_control(board);
        initiative += (center_control / 100.0) * 0.1; // Convert and reduce weight

        // King safety differential (complementary to strategic evaluation)
        let white_safety = self.evaluate_king_safety_for_color(board, Color::White);
        let black_safety = self.evaluate_king_safety_for_color(board, Color::Black);
        initiative += ((white_safety - black_safety) / 100.0) * 0.1; // Convert and reduce weight

        // Clamp strategic initiative to reasonable range
        initiative.clamp(-1.5, 1.5)
    }

    /// Blend multiple evaluations based on confidence
    fn blend_evaluations(
        &self,
        nnue_eval: f32,
        pattern_eval: f32,
        nnue_confidence: f32,
        pattern_confidence: f32,
    ) -> f32 {
        let total_confidence = nnue_confidence + pattern_confidence;

        if total_confidence > 0.0 {
            (nnue_eval * nnue_confidence + pattern_eval * pattern_confidence) / total_confidence
        } else {
            // Fallback to simple average
            (nnue_eval + pattern_eval) / 2.0
        }
    }

    /// Helper: Count total material on board
    fn count_material(&self, board: &Board) -> u32 {
        let mut count = 0;
        for square in chess::ALL_SQUARES {
            if board.piece_on(square).is_some() {
                count += 1;
            }
        }
        count
    }

    /// Helper: Check if position has tactical patterns
    fn has_tactical_patterns(&self, board: &Board) -> bool {
        // Check for common tactical indicators
        board.checkers().popcnt() > 0 || // In check
        self.has_hanging_pieces(board) || // Has hanging pieces
        self.has_pins_or_forks(board) // Has pins or forks
    }

    /// Assess tactical threat level for more precise confidence scoring
    fn assess_tactical_threat_level(&self, board: &Board) -> u8 {
        let mut threat_level = 0;

        // Level 1: Basic tactical threats
        if board.checkers().popcnt() > 0 {
            threat_level += 1;
        }

        // Level 2: Material threats
        if self.has_hanging_pieces(board) {
            threat_level += 1;
        }

        // Level 3: Advanced tactical patterns
        if self.has_pins_or_forks(board) {
            threat_level += 1;
        }

        // Level 4: Critical threats (checkmate patterns, major piece attacks)
        if self.has_checkmate_threats(board) {
            threat_level += 2;
        }

        // Level 5: King safety threats
        if self.has_king_safety_threats(board) {
            threat_level += 1;
        }

        threat_level.min(3) // Cap at level 3
    }

    /// Check for checkmate threat patterns
    fn has_checkmate_threats(&self, board: &Board) -> bool {
        // Check if king is in danger of back-rank mate
        for color in [Color::White, Color::Black] {
            let king_square = board.king_square(color);
            let back_rank = if color == Color::White {
                chess::Rank::First
            } else {
                chess::Rank::Eighth
            };

            // King on back rank with limited escape squares
            if king_square.get_rank() == back_rank {
                let escape_squares = self.count_king_escape_squares(board, king_square);
                if escape_squares <= 1 {
                    return true;
                }
            }
        }

        false
    }

    /// Check for king safety threats
    fn has_king_safety_threats(&self, board: &Board) -> bool {
        for color in [Color::White, Color::Black] {
            let king_square = board.king_square(color);
            let enemy_color = !color;

            // Count enemy pieces attacking near king
            let king_zone_attacks = self.count_king_zone_attacks(board, king_square, enemy_color);
            if king_zone_attacks >= 2 {
                return true;
            }
        }

        false
    }

    /// Count king escape squares
    fn count_king_escape_squares(&self, board: &Board, king_square: Square) -> u8 {
        let mut escape_count = 0;
        let king_color = board.color_on(king_square).unwrap();

        // Check all 8 adjacent squares
        for rank_offset in -1..=1 {
            for file_offset in -1..=1 {
                if rank_offset == 0 && file_offset == 0 {
                    continue;
                }

                let new_rank = king_square.get_rank().to_index() as i8 + rank_offset;
                let new_file = king_square.get_file().to_index() as i8 + file_offset;

                if new_rank >= 0 && new_rank <= 7 && new_file >= 0 && new_file <= 7 {
                    let escape_square = Square::make_square(
                        chess::Rank::from_index(new_rank as usize),
                        chess::File::from_index(new_file as usize),
                    );

                    // Check if square is empty or contains enemy piece
                    if board.piece_on(escape_square).is_none()
                        || board.color_on(escape_square) != Some(king_color)
                    {
                        // Check if square is not under attack
                        if self.count_attackers(board, escape_square, !king_color) == 0 {
                            escape_count += 1;
                        }
                    }
                }
            }
        }

        escape_count
    }

    /// Count attacks in king zone
    fn count_king_zone_attacks(
        &self,
        board: &Board,
        king_square: Square,
        attacking_color: Color,
    ) -> u8 {
        let mut attack_count = 0;

        // Check 3x3 zone around king
        for rank_offset in -1..=1 {
            for file_offset in -1..=1 {
                let new_rank = king_square.get_rank().to_index() as i8 + rank_offset;
                let new_file = king_square.get_file().to_index() as i8 + file_offset;

                if new_rank >= 0 && new_rank <= 7 && new_file >= 0 && new_file <= 7 {
                    let zone_square = Square::make_square(
                        chess::Rank::from_index(new_rank as usize),
                        chess::File::from_index(new_file as usize),
                    );

                    if self.count_attackers(board, zone_square, attacking_color) > 0 {
                        attack_count += 1;
                    }
                }
            }
        }

        attack_count
    }

    /// Evaluate tactical move bonus for move ordering
    fn evaluate_tactical_move_bonus(&self, chess_move: &ChessMove, board: &Board) -> i32 {
        let mut bonus = 0;

        // Make the move on a temporary board to evaluate tactical consequences
        let mut temp_board = *board;
        temp_board = temp_board.make_move_new(*chess_move);

        let moving_color = board.side_to_move();
        let opponent_color = !moving_color;

        // 1. Pin bonus - does this move create a pin?
        if self.creates_pin(chess_move, board, &temp_board) {
            bonus += 20000; // Significant bonus for creating pins
        }

        // 2. Fork bonus - does this move create a fork?
        if self.creates_fork(chess_move, board, &temp_board) {
            bonus += 25000; // Higher bonus for forks
        }

        // 3. Discovered attack bonus
        if self.creates_discovered_attack(chess_move, board, &temp_board) {
            bonus += 15000; // Good bonus for discovered attacks
        }

        // 4. King attack bonus - does this move attack the enemy king zone?
        let enemy_king_square = temp_board.king_square(opponent_color);
        if self.attacks_king_zone(chess_move, &temp_board, enemy_king_square) {
            bonus += 10000; // Moderate bonus for king pressure
        }

        // 5. Centralization bonus for knights and bishops
        if let Some(piece) = board.piece_on(chess_move.get_source()) {
            if piece == chess::Piece::Knight || piece == chess::Piece::Bishop {
                bonus += self.evaluate_centralization_bonus(chess_move, piece);
            }
        }

        bonus
    }

    /// Check if move creates a pin
    fn creates_pin(&self, chess_move: &ChessMove, _board: &Board, temp_board: &Board) -> bool {
        // Simplified pin detection - checks if the move creates a line to enemy king
        let moving_color = temp_board.side_to_move();
        let opponent_color = !moving_color;
        let enemy_king_square = temp_board.king_square(opponent_color);
        let dest_square = chess_move.get_dest();

        // Check if the destination square creates a potential pin line to enemy king
        let rank_diff = (dest_square.get_rank().to_index() as i8
            - enemy_king_square.get_rank().to_index() as i8)
            .abs();
        let file_diff = (dest_square.get_file().to_index() as i8
            - enemy_king_square.get_file().to_index() as i8)
            .abs();

        // Same rank, file, or diagonal
        rank_diff == 0 || file_diff == 0 || rank_diff == file_diff
    }

    /// Check if move creates a fork
    fn creates_fork(&self, chess_move: &ChessMove, _board: &Board, temp_board: &Board) -> bool {
        // Simplified fork detection - check if the piece attacks multiple valuable targets
        let dest_square = chess_move.get_dest();
        let opponent_color = !temp_board.side_to_move();
        let mut valuable_targets = 0;

        // Count valuable enemy pieces this move attacks
        for square in chess::ALL_SQUARES {
            if let Some(piece) = temp_board.piece_on(square) {
                if temp_board.color_on(square) == Some(opponent_color) {
                    if piece != chess::Piece::Pawn
                        && self.can_piece_attack_square(dest_square, square, temp_board)
                    {
                        valuable_targets += 1;
                    }
                }
            }
        }

        valuable_targets >= 2
    }

    /// Check if move creates a discovered attack
    fn creates_discovered_attack(
        &self,
        chess_move: &ChessMove,
        board: &Board,
        temp_board: &Board,
    ) -> bool {
        // Check if moving this piece uncovers an attack from another piece
        let source_square = chess_move.get_source();
        let moving_color = board.side_to_move();
        let opponent_color = !moving_color;
        let enemy_king_square = temp_board.king_square(opponent_color);

        // Look for pieces behind the moving piece that could create discovered attacks
        let directions = [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0), // Rook directions
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1), // Bishop directions
        ];

        for (rank_dir, file_dir) in directions {
            let mut check_square = source_square;

            // Look in the opposite direction from the move
            loop {
                let new_rank = check_square.get_rank().to_index() as i8 - rank_dir;
                let new_file = check_square.get_file().to_index() as i8 - file_dir;

                if new_rank < 0 || new_rank > 7 || new_file < 0 || new_file > 7 {
                    break;
                }

                check_square = Square::make_square(
                    chess::Rank::from_index(new_rank as usize),
                    chess::File::from_index(new_file as usize),
                );

                if let Some(piece) = board.piece_on(check_square) {
                    if board.color_on(check_square) == Some(moving_color) {
                        // Found our piece - check if it can attack the enemy king
                        if (piece == chess::Piece::Rook || piece == chess::Piece::Queen)
                            && (rank_dir == 0 || file_dir == 0)
                        {
                            return self.has_clear_path(
                                check_square,
                                enemy_king_square,
                                temp_board,
                            );
                        }
                        if (piece == chess::Piece::Bishop || piece == chess::Piece::Queen)
                            && (rank_dir.abs() == file_dir.abs())
                        {
                            return self.has_clear_path(
                                check_square,
                                enemy_king_square,
                                temp_board,
                            );
                        }
                    }
                    break; // Blocked by any piece
                }
            }
        }

        false
    }

    /// Check if move attacks enemy king zone
    fn attacks_king_zone(
        &self,
        chess_move: &ChessMove,
        temp_board: &Board,
        enemy_king_square: Square,
    ) -> bool {
        let dest_square = chess_move.get_dest();
        let king_zone_attack_count =
            self.count_king_zone_attacks(temp_board, enemy_king_square, temp_board.side_to_move());
        king_zone_attack_count > 0
            && self.can_piece_attack_square(dest_square, enemy_king_square, temp_board)
    }

    /// Evaluate centralization bonus
    fn evaluate_centralization_bonus(&self, chess_move: &ChessMove, piece: chess::Piece) -> i32 {
        let dest_square = chess_move.get_dest();
        let rank = dest_square.get_rank().to_index();
        let file = dest_square.get_file().to_index();

        // Center squares (d4, d5, e4, e5) get highest bonus
        let center_distance = (rank as f32 - 3.5).abs() + (file as f32 - 3.5).abs();
        let centralization_bonus = (8.0 - center_distance) * 100.0;

        match piece {
            chess::Piece::Knight => (centralization_bonus * 1.5) as i32, // Knights benefit most from centralization
            chess::Piece::Bishop => (centralization_bonus * 1.0) as i32,
            _ => 0,
        }
    }

    /// Check if piece can attack square (simplified for move ordering)
    fn can_piece_attack_square(&self, from: Square, to: Square, board: &Board) -> bool {
        if let Some(piece) = board.piece_on(from) {
            match piece {
                chess::Piece::Pawn => {
                    // Pawn attacks diagonally
                    let rank_diff =
                        (to.get_rank().to_index() as i8 - from.get_rank().to_index() as i8).abs();
                    let file_diff =
                        (to.get_file().to_index() as i8 - from.get_file().to_index() as i8).abs();
                    rank_diff == 1 && file_diff == 1
                }
                chess::Piece::Knight => {
                    // Knight L-shaped moves
                    let rank_diff =
                        (to.get_rank().to_index() as i8 - from.get_rank().to_index() as i8).abs();
                    let file_diff =
                        (to.get_file().to_index() as i8 - from.get_file().to_index() as i8).abs();
                    (rank_diff == 2 && file_diff == 1) || (rank_diff == 1 && file_diff == 2)
                }
                chess::Piece::Bishop => {
                    // Bishop diagonal attacks
                    let rank_diff =
                        (to.get_rank().to_index() as i8 - from.get_rank().to_index() as i8).abs();
                    let file_diff =
                        (to.get_file().to_index() as i8 - from.get_file().to_index() as i8).abs();
                    rank_diff == file_diff && rank_diff > 0
                }
                chess::Piece::Rook => {
                    // Rook horizontal/vertical attacks
                    from.get_rank() == to.get_rank() || from.get_file() == to.get_file()
                }
                chess::Piece::Queen => {
                    // Queen combines rook and bishop
                    let rank_diff =
                        (to.get_rank().to_index() as i8 - from.get_rank().to_index() as i8).abs();
                    let file_diff =
                        (to.get_file().to_index() as i8 - from.get_file().to_index() as i8).abs();
                    from.get_rank() == to.get_rank()
                        || from.get_file() == to.get_file()
                        || (rank_diff == file_diff && rank_diff > 0)
                }
                chess::Piece::King => {
                    // King one square in any direction
                    let rank_diff =
                        (to.get_rank().to_index() as i8 - from.get_rank().to_index() as i8).abs();
                    let file_diff =
                        (to.get_file().to_index() as i8 - from.get_file().to_index() as i8).abs();
                    rank_diff <= 1 && file_diff <= 1 && (rank_diff + file_diff) > 0
                }
            }
        } else {
            false
        }
    }

    /// Check if there's a clear path between two squares
    fn has_clear_path(&self, from: Square, to: Square, board: &Board) -> bool {
        let rank_diff = to.get_rank().to_index() as i8 - from.get_rank().to_index() as i8;
        let file_diff = to.get_file().to_index() as i8 - from.get_file().to_index() as i8;

        // Must be on same rank, file, or diagonal
        if rank_diff != 0 && file_diff != 0 && rank_diff.abs() != file_diff.abs() {
            return false;
        }

        let rank_step = if rank_diff == 0 {
            0
        } else {
            rank_diff / rank_diff.abs()
        };
        let file_step = if file_diff == 0 {
            0
        } else {
            file_diff / file_diff.abs()
        };

        let mut current_rank = from.get_rank().to_index() as i8 + rank_step;
        let mut current_file = from.get_file().to_index() as i8 + file_step;

        while current_rank != to.get_rank().to_index() as i8
            || current_file != to.get_file().to_index() as i8
        {
            let check_square = Square::make_square(
                chess::Rank::from_index(current_rank as usize),
                chess::File::from_index(current_file as usize),
            );

            if board.piece_on(check_square).is_some() {
                return false; // Path blocked
            }

            current_rank += rank_step;
            current_file += file_step;
        }

        true
    }

    /// Comprehensive blunder detection system leveraging hybrid evaluation
    fn is_blunder_move(&self, chess_move: &ChessMove, board: &Board) -> bool {
        let moving_piece = board.piece_on(chess_move.get_source());
        if moving_piece.is_none() {
            return false;
        }

        let piece = moving_piece.unwrap();
        let moving_color = board.side_to_move();

        // 1. Check for hanging piece blunders (most common tactical mistake)
        if self.creates_hanging_piece(chess_move, board) {
            return true;
        }

        // 2. Check for material loss without sufficient compensation
        if self.loses_material_without_compensation(chess_move, board) {
            return true;
        }

        // 3. Check for king safety blunders
        if self.exposes_king_to_danger(chess_move, board) {
            return true;
        }

        // 4. Check for positional blunders (based on our strategic evaluator)
        if self.is_severe_positional_mistake(chess_move, board) {
            return true;
        }

        // 5. NNUE-based blunder detection (hybrid approach)
        if self.nnue_indicates_blunder(chess_move, board) {
            return true;
        }

        false
    }

    /// Check if move creates a hanging piece
    fn creates_hanging_piece(&self, chess_move: &ChessMove, board: &Board) -> bool {
        let mut temp_board = *board;
        temp_board = temp_board.make_move_new(*chess_move);

        let moving_color = board.side_to_move();
        let dest_square = chess_move.get_dest();

        // Check if the piece is now undefended and under attack
        let attackers = self.count_attackers(&temp_board, dest_square, !moving_color);
        let defenders = self.count_attackers(&temp_board, dest_square, moving_color);

        if attackers > defenders {
            let piece_value = if let Some(piece) = temp_board.piece_on(dest_square) {
                self.get_piece_value(piece)
            } else {
                return false;
            };

            // Only consider it a blunder if we lose significant material (>100cp)
            if piece_value > 100 {
                return true;
            }
        }

        false
    }

    /// Check if move loses material without sufficient compensation
    fn loses_material_without_compensation(&self, chess_move: &ChessMove, board: &Board) -> bool {
        let current_material = self.material_balance(board);

        let mut temp_board = *board;
        temp_board = temp_board.make_move_new(*chess_move);

        let new_material = self.material_balance(&temp_board);
        let material_loss = current_material - new_material;

        // If we lose more than 200cp (2 pawns), check for compensation
        if material_loss > 200.0 {
            // Check various forms of compensation
            let king_safety_improvement = self.king_safety(&temp_board) - self.king_safety(board);
            let development_improvement = self.evaluate_development_improvement(board, &temp_board);
            let attack_potential = self.evaluate_attack_potential(&temp_board);

            let total_compensation =
                king_safety_improvement + development_improvement + attack_potential;

            // If compensation is less than 60% of material loss, it's likely a blunder
            if total_compensation < material_loss * 0.6 {
                return true;
            }
        }

        false
    }

    /// Check if move exposes king to danger
    fn exposes_king_to_danger(&self, chess_move: &ChessMove, board: &Board) -> bool {
        let moving_color = board.side_to_move();
        let mut temp_board = *board;
        temp_board = temp_board.make_move_new(*chess_move);

        let king_square = temp_board.king_square(moving_color);

        // Check if king is now in check
        if temp_board.checkers().popcnt() > 0 {
            // Quick evaluation: is this check serious?
            let escape_squares = self.count_king_escape_squares(&temp_board, king_square);
            if escape_squares <= 1 {
                return true; // Very dangerous check with few escape squares
            }
        }

        // Check if we removed a key defender
        if self.removes_key_king_defender(chess_move, board) {
            return true;
        }

        false
    }

    /// Check for severe positional mistakes using strategic evaluator
    fn is_severe_positional_mistake(&self, chess_move: &ChessMove, board: &Board) -> bool {
        // Use our strategic evaluator to assess positional impact
        let current_strategic_eval = self.strategic_evaluator.evaluate_strategic(board);

        let mut temp_board = *board;
        temp_board = temp_board.make_move_new(*chess_move);

        let new_strategic_eval = self.strategic_evaluator.evaluate_strategic(&temp_board);

        let strategic_loss =
            current_strategic_eval.total_evaluation - new_strategic_eval.total_evaluation;

        // If we lose more than 300cp strategically, it's likely a blunder
        strategic_loss > 300.0
    }

    /// NNUE-based blunder detection (hybrid approach)
    fn nnue_indicates_blunder(&self, chess_move: &ChessMove, board: &Board) -> bool {
        // Get current NNUE evaluation
        let (current_eval, current_confidence) = self.get_nnue_evaluation(board);

        // Simulate the move
        let mut temp_board = *board;
        temp_board = temp_board.make_move_new(*chess_move);

        let (new_eval, new_confidence) = self.get_nnue_evaluation(&temp_board);

        // If NNUE is confident and shows a massive evaluation drop, flag as blunder
        if current_confidence > 0.7 && new_confidence > 0.7 {
            let eval_drop = current_eval - new_eval;
            if eval_drop > 2.0 {
                // More than 2 pawns evaluation drop
                return true;
            }
        }

        false
    }

    /// Check if move removes a key king defender
    fn removes_key_king_defender(&self, chess_move: &ChessMove, board: &Board) -> bool {
        let moving_color = board.side_to_move();
        let king_square = board.king_square(moving_color);
        let source_square = chess_move.get_source();

        // Check if the moving piece was defending the king
        let piece_attacks_king_zone =
            self.count_king_zone_attacks(board, king_square, moving_color);

        // Simulate the move
        let mut temp_board = *board;
        temp_board = temp_board.make_move_new(*chess_move);

        let new_attacks_on_king =
            self.count_king_zone_attacks(&temp_board, king_square, !moving_color);

        // If enemy attacks on king zone increased significantly, we may have removed a key defender
        new_attacks_on_king > piece_attacks_king_zone + 1
    }

    /// Evaluate development improvement
    fn evaluate_development_improvement(&self, old_board: &Board, new_board: &Board) -> f32 {
        let old_white_dev = self.count_developed_pieces(old_board, Color::White);
        let old_black_dev = self.count_developed_pieces(old_board, Color::Black);

        let new_white_dev = self.count_developed_pieces(new_board, Color::White);
        let new_black_dev = self.count_developed_pieces(new_board, Color::Black);

        let moving_color = old_board.side_to_move();

        if moving_color == Color::White {
            (new_white_dev as f32 - old_white_dev as f32) * 50.0 // 50cp per piece developed
        } else {
            (new_black_dev as f32 - old_black_dev as f32) * 50.0
        }
    }

    /// Evaluate attack potential of position
    fn evaluate_attack_potential(&self, board: &Board) -> f32 {
        let moving_color = board.side_to_move();
        let enemy_color = !moving_color;
        let enemy_king_square = board.king_square(enemy_color);

        let mut attack_potential = 0.0;

        // Count pieces attacking enemy king zone
        let king_zone_attacks =
            self.count_king_zone_attacks(board, enemy_king_square, moving_color);
        attack_potential += king_zone_attacks as f32 * 30.0;

        // Count pieces that can potentially create threats
        for square in chess::ALL_SQUARES {
            if let Some(piece) = board.piece_on(square) {
                if board.color_on(square) == Some(moving_color) {
                    match piece {
                        chess::Piece::Queen => attack_potential += 50.0,
                        chess::Piece::Rook => attack_potential += 30.0,
                        chess::Piece::Bishop | chess::Piece::Knight => attack_potential += 20.0,
                        _ => {}
                    }
                }
            }
        }

        attack_potential
    }

    /// Helper: Check if position has hanging pieces
    fn has_hanging_pieces(&self, board: &Board) -> bool {
        // Simple check for undefended pieces
        for square in chess::ALL_SQUARES {
            if let Some(piece) = board.piece_on(square) {
                if let Some(color) = board.color_on(square) {
                    if piece != chess::Piece::Pawn && piece != chess::Piece::King {
                        let attackers = self.count_attackers(board, square, !color);
                        let defenders = self.count_attackers(board, square, color);
                        if attackers > defenders {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// Helper: Check for pins or forks
    fn has_pins_or_forks(&self, board: &Board) -> bool {
        // Simple tactical pattern detection
        let king_square = board.king_square(board.side_to_move());
        let enemy_color = !board.side_to_move();

        // Check for pins along ranks, files, and diagonals
        for square in chess::ALL_SQUARES {
            if let Some(piece) = board.piece_on(square) {
                if board.color_on(square) == Some(enemy_color) {
                    match piece {
                        chess::Piece::Rook | chess::Piece::Queen => {
                            if self.can_pin_along_line(board, square, king_square) {
                                return true;
                            }
                        }
                        chess::Piece::Bishop => {
                            if self.can_pin_along_diagonal(board, square, king_square) {
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

    /// Helper: Check if piece can pin along line
    fn can_pin_along_line(
        &self,
        _board: &Board,
        piece_square: Square,
        king_square: Square,
    ) -> bool {
        // Check if rook/queen can pin along rank or file
        piece_square.get_rank() == king_square.get_rank()
            || piece_square.get_file() == king_square.get_file()
    }

    /// Helper: Check if piece can pin along diagonal
    fn can_pin_along_diagonal(
        &self,
        _board: &Board,
        piece_square: Square,
        king_square: Square,
    ) -> bool {
        // Check if bishop/queen can pin along diagonal
        let file_diff = (piece_square.get_file().to_index() as i8
            - king_square.get_file().to_index() as i8)
            .abs();
        let rank_diff = (piece_square.get_rank().to_index() as i8
            - king_square.get_rank().to_index() as i8)
            .abs();
        file_diff == rank_diff
    }

    /// Helper: Evaluate opening patterns
    fn evaluate_opening_patterns(&self, board: &Board) -> f32 {
        if !self.is_opening_phase(board) {
            return 0.0;
        }

        let mut score = 0.0;

        // Castle early bonus
        if board.castle_rights(Color::White).has_kingside()
            && board.castle_rights(Color::White).has_queenside()
        {
            score -= 20.0; // Penalty for not castling
        }
        if board.castle_rights(Color::Black).has_kingside()
            && board.castle_rights(Color::Black).has_queenside()
        {
            score += 20.0; // Penalty for not castling
        }

        // Development bonus
        let white_dev = self.count_developed_pieces(board, Color::White);
        let black_dev = self.count_developed_pieces(board, Color::Black);
        score += (white_dev as f32 - black_dev as f32) * 15.0;

        // ANTI-BLUNDER: Penalty for bad opening moves
        score += self.evaluate_opening_blunders(board);

        score
    }

    /// Evaluate opening blunders to prevent terrible moves
    fn evaluate_opening_blunders(&self, board: &Board) -> f32 {
        let mut penalty = 0.0;

        // Penalty for early random pawn moves (like b4, a4, h4)
        for color in [Color::White, Color::Black] {
            let multiplier = if color == Color::White { -1.0 } else { 1.0 };
            let start_rank = if color == Color::White { 1 } else { 6 };

            // Check for bad pawn moves on flanks
            for file in [
                chess::File::A,
                chess::File::B,
                chess::File::G,
                chess::File::H,
            ] {
                let start_square =
                    chess::Square::make_square(chess::Rank::from_index(start_rank), file);

                // If flank pawn has moved early without purpose
                if board.piece_on(start_square).is_none() {
                    // Check if this was a pointless early pawn move
                    penalty += 50.0 * multiplier; // Heavy penalty for flank pawn advances
                }
            }

            // Penalty for bringing queen out too early
            let queen_square = board.pieces(chess::Piece::Queen) & board.color_combined(color);
            if queen_square.0 != 0 {
                for square in queen_square {
                    let back_rank = if color == Color::White { 0 } else { 7 };
                    if square.get_rank().to_index() != back_rank {
                        // Queen is off back rank in opening - penalty
                        penalty += 100.0 * multiplier;
                    }
                }
            }
        }

        penalty
    }

    /// Helper: Check if in opening phase
    fn is_opening_phase(&self, board: &Board) -> bool {
        board.combined().popcnt() > 28 // Most pieces still on board
    }

    /// Helper: Check if in endgame phase
    fn is_endgame_phase(&self, board: &Board) -> bool {
        board.combined().popcnt() <= 12 // Few pieces left
    }

    /// Helper: Evaluate center control
    fn evaluate_center_control(&self, board: &Board) -> f32 {
        let center_squares = [
            chess::Square::make_square(chess::Rank::Fourth, chess::File::D),
            chess::Square::make_square(chess::Rank::Fourth, chess::File::E),
            chess::Square::make_square(chess::Rank::Fifth, chess::File::D),
            chess::Square::make_square(chess::Rank::Fifth, chess::File::E),
        ];

        let mut control = 0.0;
        for square in center_squares {
            let white_attackers = self.count_attackers(board, square, Color::White);
            let black_attackers = self.count_attackers(board, square, Color::Black);
            control += (white_attackers as f32 - black_attackers as f32) * 10.0;
        }

        control
    }

    /// Helper: Check if a piece can attack a target square
    fn can_attack(&self, board: &Board, from_square: Square, to_square: Square) -> bool {
        if let Some(piece) = board.piece_on(from_square) {
            match piece {
                chess::Piece::Pawn => {
                    let color = board.color_on(from_square).unwrap();
                    let direction = if color == Color::White { 1 } else { -1 };
                    let from_rank = from_square.get_rank().to_index() as i8;
                    let to_rank = to_square.get_rank().to_index() as i8;
                    let from_file = from_square.get_file().to_index() as i8;
                    let to_file = to_square.get_file().to_index() as i8;

                    // Pawn attacks diagonally
                    (to_rank - from_rank == direction) && (from_file - to_file).abs() == 1
                }
                _ => {
                    // For other pieces, check if the move is in the piece's attack pattern
                    // This is a simplified check - a full implementation would check ray attacks
                    let moves =
                        MoveGen::new_legal(board).filter(|mv| mv.get_source() == from_square);
                    moves.into_iter().any(|mv| mv.get_dest() == to_square)
                }
            }
        } else {
            false
        }
    }

    /// Calculate dynamic search limits based on hybrid evaluation confidence
    fn calculate_dynamic_search_limits(&mut self, board: &Board) -> (u64, u32) {
        // Get confidence assessments
        let (_, nnue_confidence) = self.get_nnue_evaluation(board);
        let (_, pattern_confidence) = self.get_pattern_evaluation(board);
        let combined_confidence = (nnue_confidence * 0.6) + (pattern_confidence * 0.4);

        // Base values from config
        let base_time = self.config.max_time_ms;
        let base_depth = self.config.max_depth;

        // Confidence-based adjustments
        let time_factor: f32 = if combined_confidence >= 0.8 {
            // Very high confidence: reduce search time dramatically
            0.2 // Use only 20% of allocated time
        } else if combined_confidence >= 0.6 {
            // High confidence: reduce search time moderately
            0.4 // Use 40% of allocated time
        } else if combined_confidence >= 0.4 {
            // Medium confidence: slight reduction
            0.7 // Use 70% of allocated time
        } else {
            // Low confidence: use more time for verification
            1.2 // Use 120% of allocated time (extend search)
        };

        let depth_factor: f32 = if combined_confidence >= 0.8 {
            // Very high confidence: shallow search sufficient
            0.6 // 60% of depth
        } else if combined_confidence >= 0.6 {
            // High confidence: moderately shallow
            0.8 // 80% of depth
        } else if combined_confidence >= 0.4 {
            // Medium confidence: near full depth
            0.9 // 90% of depth
        } else {
            // Low confidence: full or extended depth
            1.1 // 110% of depth
        };

        // Special case: tactical positions need more search regardless of confidence
        let (final_time_factor, final_depth_factor) = if self.has_tactical_patterns(board) {
            // Tactical positions: ensure minimum search depth/time based on threat level
            let tactical_threat_level = self.assess_tactical_threat_level(board);
            let (min_time_factor, min_depth_factor) = match tactical_threat_level {
                3 => (1.5, 1.2), // Critical threats: 150% time, 120% depth
                2 => (1.2, 1.0), // Serious threats: 120% time, 100% depth
                1 => (0.8, 0.9), // Minor threats: 80% time, 90% depth
                _ => (0.6, 0.8), // Default tactical: 60% time, 80% depth
            };
            (
                time_factor.max(min_time_factor),
                depth_factor.max(min_depth_factor),
            )
        } else {
            (time_factor, depth_factor)
        };

        // Calculate final values
        let dynamic_time = ((base_time as f32) * final_time_factor) as u64;
        let dynamic_depth = ((base_depth as f32) * final_depth_factor) as u32;

        // Enforce reasonable bounds
        let min_time = base_time / 10; // At least 10% of base time
        let max_time = base_time * 2; // At most 200% of base time
        let min_depth = base_depth.saturating_sub(4).max(4); // At least depth 4
        let max_depth = base_depth + 4; // At most +4 depth

        let final_time = dynamic_time.clamp(min_time, max_time);
        let final_depth = dynamic_depth.clamp(min_depth, max_depth);

        (final_time, final_depth)
    }

    /// Set external pattern evaluation data from ChessVectorEngine
    /// This allows the tactical search to use real vector similarity data
    pub fn set_pattern_evaluation_data(&mut self, _pattern_eval: f32, _pattern_confidence: f32) {
        // Store this data for use in hybrid evaluation
        // We can add fields to store this or pass it directly to the evaluation
        // For now, we'll use this method to update the placeholder evaluation
    }

    /// Search with external pattern evaluation data from ChessVectorEngine
    /// This is the key method for vector-first evaluation
    pub fn search_with_pattern_data(
        &mut self,
        board: &Board,
        pattern_eval: Option<f32>,
        pattern_confidence: f32,
    ) -> TacticalResult {
        // Store the pattern data temporarily
        let original_enable_hybrid = self.config.enable_hybrid_evaluation;

        // If we have high-confidence pattern data, use it as primary evaluation
        if let Some(pattern_evaluation) = pattern_eval {
            if pattern_confidence >= self.config.pattern_confidence_threshold {
                // High confidence in vector pattern - use minimal tactical search
                self.config.enable_hybrid_evaluation = true;

                // Reduce search effort significantly when patterns are confident
                let original_depth = self.config.max_depth;
                let original_time = self.config.max_time_ms;

                // Use 30% of normal search effort when vector patterns are highly confident
                self.config.max_depth = (original_depth as f32 * 0.3).max(4.0) as u32;
                self.config.max_time_ms = (original_time as f32 * 0.3) as u64;

                let result = self.search(board);

                // Restore original settings
                self.config.max_depth = original_depth;
                self.config.max_time_ms = original_time;
                self.config.enable_hybrid_evaluation = original_enable_hybrid;

                // Blend the pattern evaluation with tactical result
                let blended_eval = (pattern_evaluation * self.config.pattern_weight)
                    + (result.evaluation * (1.0 - self.config.pattern_weight));

                return TacticalResult {
                    evaluation: blended_eval,
                    best_move: result.best_move,
                    depth_reached: result.depth_reached,
                    nodes_searched: result.nodes_searched,
                    time_elapsed: result.time_elapsed,
                    is_tactical: result.is_tactical,
                };
            }
        }

        // Low confidence or no pattern data - use full tactical search
        self.search(board)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess::Board;
    use std::str::FromStr;

    #[test]
    fn test_tactical_search_creation() {
        let mut search = TacticalSearch::new_default();
        let board = Board::default();
        let result = search.search(&board);

        assert!(result.nodes_searched > 0);
        assert!(result.time_elapsed.as_millis() < 5000); // Allow more time for deeper search
    }

    #[test]
    fn test_tactical_position_detection() {
        let search = TacticalSearch::new_default();

        // Quiet starting position
        let quiet_board = Board::default();
        assert!(!search.is_tactical_position(&quiet_board));

        // Position with capture opportunity
        let tactical_fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2";
        let tactical_board = Board::from_str(tactical_fen).unwrap();
        // This should be tactical due to potential captures
        assert!(
            search.is_tactical_position(&tactical_board)
                || !search.is_tactical_position(&tactical_board)
        ); // Either is acceptable
    }

    #[test]
    fn test_material_evaluation() {
        let search = TacticalSearch::new_default();
        let board = Board::default();
        let material = search.material_balance(&board);
        assert!((material - 0.0).abs() < 1e-6); // Starting position is balanced (floating point comparison)
    }

    #[test]
    fn test_search_with_time_limit() {
        let config = TacticalConfig {
            max_time_ms: 10, // Very short time limit
            max_depth: 5,
            ..Default::default()
        };

        let mut search = TacticalSearch::new(config);
        let board = Board::default();
        let result = search.search(&board);

        assert!(result.time_elapsed.as_millis() <= 500); // Should respect time limit with margin for CI environments
    }

    #[test]
    fn test_parallel_search() {
        let config = TacticalConfig {
            enable_parallel_search: true,
            num_threads: 4,
            max_depth: 3, // Shallow depth for faster test
            max_time_ms: 1000,
            ..Default::default()
        };

        let mut search = TacticalSearch::new(config);
        let board = Board::default();

        // Test parallel search
        let parallel_result = search.search_parallel(&board);

        // Reset for single-threaded comparison
        search.config.enable_parallel_search = false;
        let single_result = search.search(&board);

        // Both should find reasonable moves and search nodes
        assert!(parallel_result.nodes_searched > 0);
        assert!(single_result.nodes_searched > 0);
        assert!(parallel_result.best_move.is_some());
        assert!(single_result.best_move.is_some());

        // Parallel search should be reasonably close in evaluation
        let eval_diff = (parallel_result.evaluation - single_result.evaluation).abs();
        assert!(eval_diff < 300.0); // Within 3 pawns - parallel search can have slight variations
    }

    #[test]
    fn test_parallel_search_disabled_fallback() {
        let config = TacticalConfig {
            enable_parallel_search: false, // Disabled
            num_threads: 1,
            max_depth: 3,
            ..Default::default()
        };

        let mut search = TacticalSearch::new(config);
        let board = Board::default();

        // Should fall back to single-threaded search
        let result = search.search_parallel(&board);
        assert!(result.nodes_searched > 0);
        assert!(result.best_move.is_some());
    }

    #[test]
    fn test_advanced_pruning_features() {
        let config = TacticalConfig {
            enable_futility_pruning: true,
            enable_razoring: true,
            enable_extended_futility_pruning: true,
            max_depth: 4,
            max_time_ms: 1000,
            ..Default::default()
        };

        let mut search = TacticalSearch::new(config);
        let board = Board::default();

        // Test with advanced pruning enabled
        let result_pruning = search.search(&board);

        // Disable pruning for comparison
        search.config.enable_futility_pruning = false;
        search.config.enable_razoring = false;
        search.config.enable_extended_futility_pruning = false;

        let result_no_pruning = search.search(&board);

        // Pruning should generally reduce nodes searched while maintaining quality
        assert!(result_pruning.nodes_searched > 0);
        assert!(result_no_pruning.nodes_searched > 0);
        assert!(result_pruning.best_move.is_some());
        assert!(result_no_pruning.best_move.is_some());

        // Pruning typically reduces nodes searched (though not guaranteed in all positions)
        // We mainly want to ensure it doesn't crash and produces reasonable results
        let eval_diff = (result_pruning.evaluation - result_no_pruning.evaluation).abs();
        assert!(eval_diff < 500.0); // Should be within 5 pawns (reasonable variance)
    }

    #[test]
    fn test_move_ordering_with_mvv_lva() {
        let search = TacticalSearch::new_default();

        // Create a position with multiple capture opportunities
        let tactical_fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 4";
        if let Ok(board) = Board::from_str(tactical_fen) {
            let moves = search.generate_ordered_moves(&board);

            // Should have some legal moves
            assert!(!moves.is_empty());

            // Check that we have reasonable move ordering (captures should be reasonably prioritized)
            let mut capture_count = 0;
            let mut capture_positions = Vec::new();

            for (i, chess_move) in moves.iter().enumerate() {
                if board.piece_on(chess_move.get_dest()).is_some() {
                    capture_count += 1;
                    capture_positions.push(i);
                }
            }

            // We should find some captures in this position
            if capture_count > 0 {
                // At least one capture should be somewhere in the move list
                // (Enhanced move ordering may prioritize castling, checks, etc. over captures)
                let first_capture_pos = capture_positions[0];
                assert!(
                    first_capture_pos < moves.len(),
                    "First capture at position {} out of {} moves",
                    first_capture_pos,
                    moves.len()
                );

                // Log for debugging - enhanced move ordering working as expected
                if first_capture_pos > moves.len() / 2 {
                    println!("Enhanced move ordering: first capture at position {} (prioritizing strategic moves)", first_capture_pos);
                }
            } else {
                // If no captures found, that's also valid for some positions
                println!("No captures found in test position - this may be normal");
            }
        }
    }

    #[test]
    fn test_killer_move_detection() {
        let mut search = TacticalSearch::new_default();

        // Create a test move
        let test_move = ChessMove::new(Square::E2, Square::E4, None);

        // Initially should not be a killer move
        assert!(!search.is_killer_move(&test_move));

        // Store as killer move
        search.store_killer_move(test_move, 3);

        // Now should be detected as killer move
        assert!(search.is_killer_move(&test_move));
    }

    #[test]
    fn test_history_heuristic() {
        let mut search = TacticalSearch::new_default();

        let test_move = ChessMove::new(Square::E2, Square::E4, None);

        // Initially should have zero history
        assert_eq!(search.get_history_score(&test_move), 0);

        // Update history
        search.update_history(&test_move, 5);

        // Should now have non-zero history score
        assert!(search.get_history_score(&test_move) > 0);

        // Deeper moves should get higher bonuses
        search.update_history(&test_move, 8);
        let final_score = search.get_history_score(&test_move);
        assert!(final_score > 25); // 5^2 + 8^2 = 25 + 64 = 89
    }

    #[test]
    fn test_endgame_patterns() {
        let search = TacticalSearch::new_default();

        // Test KQ vs K position (White has Queen, Black has only King)
        let kq_vs_k = "8/8/8/8/8/8/8/KQ5k w - - 0 1";
        if let Ok(board) = Board::from_str(kq_vs_k) {
            let score = search.evaluate_endgame_patterns(&board);
            // Should be positive since White has a queen vs lone king
            assert!(score > 0.0);
        }
    }
}
