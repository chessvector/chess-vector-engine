# Chess Vector Engine - API Reference

A comprehensive API reference for the Chess Vector Engine, covering all public types, methods, and usage patterns.

## Table of Contents

1. [Core Engine API](#core-engine-api)
2. [Position Encoding](#position-encoding)
3. [Similarity Search](#similarity-search)
4. [Tactical Search](#tactical-search)
5. [Strategic Evaluation](#strategic-evaluation)
6. [Opening Book](#opening-book)
7. [UCI Engine](#uci-engine)
8. [Training & Data Loading](#training--data-loading)
9. [GPU Acceleration](#gpu-acceleration)
10. [Error Handling](#error-handling)

## Core Engine API

### ChessVectorEngine

The main engine struct that coordinates all chess analysis functionality.

```rust
use chess_vector_engine::ChessVectorEngine;
use chess::Board;

// Create new engine
let mut engine = ChessVectorEngine::new(1024);

// Basic operations
engine.add_position(&board, evaluation);
let similar = engine.find_similar_positions(&board, k);
let eval = engine.evaluate_position(&board);
```

#### Constructor Methods

```rust
// Basic constructor with vector dimension
ChessVectorEngine::new(vector_size: usize) -> Self

// Constructor with auto-loading of training data
ChessVectorEngine::new_with_auto_load(vector_size: usize) -> Result<Self, ChessEngineError>

// Constructor with custom configuration
ChessVectorEngine::new_with_config(config: EngineConfig) -> Self
```

#### Core Methods

```rust
// Position management
add_position(&mut self, board: &Board, evaluation: f32)
add_position_from_vector(&mut self, vector: Array1<f32>, evaluation: f32)

// Similarity search
find_similar_positions(&self, board: &Board, k: usize) -> Vec<(Board, f32, f32)>
find_similar_vectors(&self, query: &Array1<f32>, k: usize) -> Vec<(Array1<f32>, f32, f32)>

// Position evaluation
evaluate_position(&self, board: &Board) -> Option<f32>
evaluate_position_detailed(&self, board: &Board) -> Option<DetailedEvaluation>

// Position encoding
encode_position(&self, board: &Board) -> Array1<f32>

// Statistics and information
get_stats(&self) -> EngineStats
knowledge_base_size(&self) -> usize
```

#### Feature Control Methods

```rust
// Enable/disable features
enable_opening_book(&mut self)
enable_tactical_search_default(&mut self)
enable_tactical_search(&mut self, config: TacticalConfig)
enable_gpu_acceleration(&mut self) -> Result<(), ChessEngineError>

// Configuration
configure_similarity_search(&mut self, config: SimilarityConfig)
configure_tactical_search(&mut self, config: TacticalConfig)
set_pattern_weight(&mut self, weight: f32) // 0.0 to 1.0
```

#### Data Loading Methods

```rust
// Load training data
load_positions(&mut self, positions: Vec<(Board, f32)>) -> Result<(), ChessEngineError>
load_from_file<P: AsRef<Path>>(&mut self, path: P) -> Result<(), ChessEngineError>
auto_load_training_data(&mut self) -> Result<LoadingStats, ChessEngineError>

// Save/load engine state
save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), ChessEngineError>
load_from_file<P: AsRef<Path>>(&mut self, path: P) -> Result<(), ChessEngineError>
```

## Position Encoding

### PositionEncoder

Converts chess positions to high-dimensional vectors for similarity analysis.

```rust
use chess_vector_engine::PositionEncoder;

let encoder = PositionEncoder::new(1024);
let vector = encoder.encode(&board);
```

#### Methods

```rust
// Constructor
PositionEncoder::new(vector_size: usize) -> Self

// Encoding methods
encode(&self, board: &Board) -> Array1<f32>
encode_batch(&self, boards: &[Board]) -> Vec<Array1<f32>>

// Utility methods
vector_size(&self) -> usize
encoding_stats(&self) -> EncodingStats
```

#### Encoding Features

- **Piece positions**: 12 piece types × 64 squares = 768 features
- **Game state**: Castling rights, en passant, side to move = 8 features  
- **Material balance**: Relative piece values = 32 features
- **Positional features**: King safety, pawn structure, mobility = 216 features

## Similarity Search

### SimilaritySearch

High-performance k-nearest neighbor search for chess positions.

```rust
use chess_vector_engine::SimilaritySearch;

let mut search = SimilaritySearch::new(1024);
search.add_position(vector, evaluation);
let results = search.search(&query_vector, k);
```

#### Methods

```rust
// Constructor and configuration
SimilaritySearch::new(vector_size: usize) -> Self
SimilaritySearch::with_capacity(vector_size: usize, capacity: usize) -> Self

// Adding positions
add_position(&mut self, vector: Array1<f32>, evaluation: f32)
add_positions(&mut self, positions: Vec<(Array1<f32>, f32)>)

// Search methods
search(&self, query: &Array1<f32>, k: usize) -> Vec<(Array1<f32>, f32, f32)>
search_with_threshold(&self, query: &Array1<f32>, k: usize, threshold: f32) -> Vec<(Array1<f32>, f32, f32)>

// Optimization
enable_lsh(&mut self, num_tables: usize, hash_size: usize)
optimize_memory(&mut self) -> MemoryOptimizationResult
```

#### Performance Features

- **Linear search**: Exact k-NN with cosine similarity
- **LSH acceleration**: Approximate search for large databases
- **Memory optimization**: 75% memory reduction through compression
- **SIMD optimization**: Vectorized operations for 2-4x speedup
- **GPU acceleration**: CUDA/Metal support for 10-100x speedup

## Tactical Search

### TacticalSearch

Advanced chess engine with professional-level tactical analysis.

```rust
use chess_vector_engine::{TacticalSearch, TacticalConfig};

let config = TacticalConfig::default();
let mut search = TacticalSearch::new(config);
let result = search.search(&board, depth);
```

#### TacticalConfig

```rust
pub struct TacticalConfig {
    pub depth: i32,                      // Search depth in ply
    pub time_limit_ms: u64,              // Time limit in milliseconds
    pub use_iterative_deepening: bool,   // Enable iterative deepening
    pub enable_quiescence_search: bool,  // Search tactical sequences to quiet positions
    pub enable_transposition_table: bool, // Hash table for position caching
    pub enable_move_ordering: bool,      // Intelligent move ordering
    pub aspiration_window: i32,          // Aspiration window size
    pub null_move_pruning: bool,         // Null move pruning
    pub late_move_reduction: bool,       // Late move reduction (LMR)
    pub futility_pruning: bool,          // Futility pruning
    pub enable_check_extensions: bool,   // Extend search in check
    pub max_extensions: i32,             // Maximum extension depth
    pub enable_singular_extensions: bool, // Singular move extensions
    pub razoring_enabled: bool,          // Razoring pruning
    pub hybrid_move_ordering: bool,      // Use pattern recognition for move ordering
}
```

#### Predefined Configurations

```rust
// Fast configuration (blitz games)
TacticalConfig::fast() -> Self

// Default configuration (standard games)  
TacticalConfig::default() -> Self

// Strong configuration (correspondence)
TacticalConfig::strong() -> Self

// Analysis configuration (deep analysis)
TacticalConfig::analysis() -> Self

// Hybrid configuration (pattern-optimized)
TacticalConfig::hybrid_optimized() -> Self
```

#### Search Methods

```rust
// Main search interface
search(&mut self, board: &Board, depth: i32) -> TacticalResult
search_with_time_limit(&mut self, board: &Board, time_ms: u64) -> TacticalResult

// Advanced search
iterative_deepening(&mut self, board: &Board, max_depth: i32) -> TacticalResult
analyze_position(&mut self, board: &Board, multi_pv: usize) -> Vec<TacticalResult>

// Utility methods
evaluate_move(&mut self, board: &Board, chess_move: ChessMove) -> f32
get_best_line(&mut self, board: &Board, depth: i32) -> Vec<ChessMove>
```

#### TacticalResult

```rust
pub struct TacticalResult {
    pub best_move: Option<ChessMove>,
    pub evaluation: f32,
    pub depth: i32,
    pub nodes_searched: u64,
    pub time_ms: u64,
    pub principal_variation: Vec<ChessMove>,
    pub mate_in: Option<i32>,
}
```

## Strategic Evaluation

### Strategic Initiative System

Advanced strategic analysis that goes beyond traditional evaluation.

```rust
use chess_vector_engine::{StrategicInitiativeEvaluator, InitiativeFactors};

let evaluator = StrategicInitiativeEvaluator::new();
let result = evaluator.analyze_position(&board);
```

#### StrategicInitiativeResult

```rust
pub struct StrategicInitiativeResult {
    pub initiative_score: f32,          // Overall initiative evaluation
    pub color_initiative: ColorInitiativeAnalysis, // Per-color analysis
    pub positional_pressure: PositionalPressure,   // Pressure analysis
    pub strategic_plans: Vec<StrategicPlan>,        // Recommended plans
    pub time_pressure: TimePressure,                // Time-based factors
    pub plan_urgency: PlanUrgency,                  // Urgency assessment
}
```

#### Strategic Plans

```rust
pub enum StrategicPlanType {
    KingsideAttack,     // Attack opponent's king
    QueensidePlay,      // Queenside pawn storm
    CenterControl,      // Control central squares
    PieceImprovement,   // Improve piece positions
    PawnStructure,      // Fix pawn weaknesses
    Endgame,           // Transition to favorable endgame
}

pub struct StrategicPlan {
    pub plan_type: StrategicPlanType,
    pub priority: f32,
    pub moves: Vec<ChessMove>,
    pub evaluation_change: f32,
}
```

### Pattern Recognition

```rust
use chess_vector_engine::{AdvancedPatternRecognizer, PatternAnalysisResult};

let recognizer = AdvancedPatternRecognizer::new();
let analysis = recognizer.analyze_position(&board);
```

#### Pattern Types

- **Tactical patterns**: Pins, forks, skewers, discovered attacks
- **Pawn structure**: Isolated, doubled, passed, backward pawns
- **King safety**: Pawn shield, piece attacks, escape squares
- **Piece coordination**: Piece activity, harmony, outposts
- **Endgame patterns**: Known theoretical positions

## Opening Book

### OpeningBook

Fast lookup of opening theory with 50+ major chess openings.

```rust
use chess_vector_engine::{OpeningBook, OpeningEntry};

let mut book = OpeningBook::new();
book.load_default_openings();
let entry = book.lookup(&board);
```

#### Methods

```rust
// Constructor and loading
OpeningBook::new() -> Self
load_default_openings(&mut self)
load_from_file<P: AsRef<Path>>(&mut self, path: P) -> Result<(), ChessEngineError>

// Lookup methods
lookup(&self, board: &Board) -> Option<&OpeningEntry>
lookup_by_fen(&self, fen: &str) -> Option<&OpeningEntry>
get_opening_moves(&self, board: &Board) -> Vec<ChessMove>

// Statistics
get_stats(&self) -> OpeningBookStats
total_openings(&self) -> usize
```

#### OpeningEntry

```rust
pub struct OpeningEntry {
    pub name: String,           // Opening name (e.g., "Sicilian Defense")
    pub eco_code: String,       // ECO code (e.g., "B20")
    pub evaluation: f32,        // Opening evaluation
    pub popularity: f32,        // Frequency in master games
    pub success_rate: f32,      // Win rate for white
    pub moves: Vec<ChessMove>,  // Principal line
}
```

#### Supported Openings

- **e4 openings**: Italian Game, Ruy Lopez, Sicilian Defense
- **d4 openings**: Queen's Gambit, King's Indian, Nimzo-Indian
- **Other**: English Opening, French Defense, Caro-Kann
- **ECO codes**: Complete A00-E99 classification

## UCI Engine

### UCI Protocol Implementation

Full Universal Chess Interface support for chess GUI integration.

```rust
use chess_vector_engine::{run_uci_engine, UCIConfig, UCIEngine};

// Run with default configuration
run_uci_engine();

// Run with custom configuration
let config = UCIConfig::default();
run_uci_engine_with_config(config);
```

#### UCI Options

| Option | Type | Default | Range | Description |
|--------|------|---------|-------|-------------|
| MultiPV | spin | 1 | 1-10 | Number of principal variations |
| Pattern_Weight | spin | 60 | 0-100 | Pattern recognition weight (%) |
| Ponder | check | true | - | Think on opponent's time |
| Load_Position_Data | button | - | - | Load training data |
| Threads | spin | 1 | 1-64 | Number of search threads |
| Enable_LSH | check | true | - | Use approximate search |
| Hash | spin | 128 | 1-2048 | Hash table size (MB) |
| Tactical_Depth | spin | 3 | 1-10 | Default search depth |
| Pattern_Confidence_Threshold | spin | 75 | 0-100 | Pattern confidence minimum |
| Enable_GPU | check | true | - | Use GPU acceleration |

#### UCI Commands

```rust
// Standard UCI commands
uci                 // Initialize UCI mode
isready            // Engine ready check
position <pos>     // Set board position
go <params>        // Start search
stop               // Stop search
quit               // Exit engine

// Engine-specific extensions
load_data <file>   // Load additional training data
pattern_weight <n> // Set pattern weight (0-100)
analyze_similarity // Show similar positions
```

## Training & Data Loading

### Training Data Formats

The engine supports multiple training data formats for optimal performance.

```rust
use chess_vector_engine::{UltraFastLoader, LoadingStats, AutoDiscovery};

// Auto-discover and load training data
let mut discovery = AutoDiscovery::new();
let files = discovery.discover_training_files("training_data/")?;
let stats = engine.load_training_data(files)?;

// Manual loading
let loader = UltraFastLoader::new();
let stats = loader.load_any_format("data.bin")?;
```

#### Supported Formats

1. **Binary (.bin)**: LZ4-compressed binary format (5-15x faster than JSON)
2. **MessagePack (.msgpack)**: Binary serialization (10-20% faster than bincode)
3. **Memory-mapped (.mmap)**: Zero-copy loading for instant startup
4. **Zstd compressed (.zst)**: Best compression ratios with fast decompression
5. **JSON (.json)**: Human-readable format with streaming support

#### Format Priority

1. Memory-mapped files (instant)
2. MessagePack binary (fastest)
3. Zstd compressed (smallest)
4. LZ4 binary (balanced)
5. JSON streaming (fallback)

#### Performance Comparison

| Format | Load Time | File Size | Use Case |
|--------|-----------|-----------|----------|
| Memory-mapped | Instant | Largest | Production servers |
| MessagePack | Very fast | Small | General use |
| Zstd | Fast | Smallest | Bandwidth-limited |
| LZ4 Binary | Fast | Medium | Development |
| JSON | Slow | Large | Human editing |

### Training Data Structure

```rust
// Position with evaluation
pub struct PositionData {
    pub fen: String,
    pub evaluation: f32,
    pub game_phase: GamePhase,
    pub metadata: HashMap<String, Value>,
}

// Training dataset
pub struct TrainingDataset {
    pub positions: Vec<PositionData>,
    pub metadata: DatasetMetadata,
    pub statistics: DatasetStats,
}
```

## GPU Acceleration

### GPU Support

Optional GPU acceleration for massive performance improvements.

```rust
use chess_vector_engine::{GPUAccelerator, DeviceType};

// Check for GPU availability
if GPUAccelerator::is_available() {
    let device = GPUAccelerator::detect_device()?;
    engine.enable_gpu_acceleration()?;
}
```

#### Supported Devices

- **CUDA**: NVIDIA GPUs with CUDA 11.0+
- **Metal**: Apple Silicon and Intel Macs
- **CPU fallback**: Always available

#### Performance Gains

- **Similarity search**: 10-100x speedup for large datasets
- **Neural networks**: 5-50x speedup for NNUE evaluation
- **Batch operations**: 20-200x speedup for multiple positions

#### GPU Configuration

```rust
// Enable with automatic device detection
engine.enable_gpu_acceleration()?;

// Manual device selection
engine.enable_gpu_with_device(DeviceType::CUDA)?;
engine.enable_gpu_with_device(DeviceType::Metal)?;

// Configure memory limits
engine.set_gpu_memory_limit(1024 * 1024 * 1024)?; // 1GB
```

## Error Handling

### ChessEngineError

Comprehensive error handling for all engine operations.

```rust
use chess_vector_engine::ChessEngineError;

match engine.load_training_data(path) {
    Ok(stats) => println!("Loaded {} positions", stats.total_positions),
    Err(ChessEngineError::InvalidFormat(msg)) => eprintln!("Format error: {}", msg),
    Err(ChessEngineError::FileNotFound(path)) => eprintln!("File not found: {}", path),
    Err(ChessEngineError::InsufficientMemory(required)) => eprintln!("Need {} MB", required),
    Err(e) => eprintln!("Engine error: {}", e),
}
```

#### Error Types

```rust
pub enum ChessEngineError {
    // File and I/O errors
    FileNotFound(String),
    InvalidFormat(String),
    CorruptedData(String),
    
    // Memory and resource errors
    InsufficientMemory(usize),
    VectorSizeMismatch { expected: usize, actual: usize },
    
    // Configuration errors
    InvalidConfiguration(String),
    FeatureNotAvailable(String),
    
    // GPU and acceleration errors
    GPUNotAvailable,
    CUDAError(String),
    MetalError(String),
    
    // Chess-specific errors
    InvalidPosition(String),
    InvalidMove(String),
    
    // Internal errors
    InternalError(String),
}
```

#### Error Context

```rust
// Errors include full context for debugging
let result = engine.load_training_data("invalid.bin")
    .map_err(|e| e.with_context("Loading training data"));

// Chain multiple error sources
let result = engine.enable_gpu_acceleration()
    .map_err(|e| e.with_context("Initializing GPU acceleration"))
    .and_then(|_| engine.load_large_dataset("data.bin"))
    .map_err(|e| e.with_context("Loading large dataset"));
```

## Performance Tuning

### Memory Optimization

```rust
// Enable memory optimization
engine.optimize_memory()?;

// Configure memory limits
engine.set_memory_limit(512 * 1024 * 1024)?; // 512MB

// Get memory statistics
let stats = engine.memory_stats();
println!("Memory usage: {} MB", stats.total_mb);
```

### Search Optimization

```rust
// Configure similarity search
engine.configure_similarity_search(SimilarityConfig {
    enable_lsh: true,
    num_hash_tables: 8,
    hash_size: 16,
    similarity_threshold: 0.7,
})?;

// Configure tactical search
engine.configure_tactical_search(TacticalConfig {
    depth: 10,
    enable_transposition_table: true,
    hash_size_mb: 256,
    enable_multi_threading: true,
    thread_count: 4,
    ..Default::default()
})?;
```

### Batch Operations

```rust
// Batch position encoding
let vectors = encoder.encode_batch(&boards);

// Batch similarity search
let results = search.search_batch(&query_vectors, k);

// Batch evaluation
let evaluations = engine.evaluate_positions_batch(&boards);
```

## Examples

See the `examples/` directory for complete usage examples:

- `basic_usage.rs`: Fundamental API usage
- `advanced_usage.rs`: Advanced features and optimization
- `uci_integration.rs`: UCI engine integration
- `training_data.rs`: Training data loading and management
- `gpu_acceleration.rs`: GPU acceleration setup
- `performance_tuning.rs`: Performance optimization techniques

## Version Compatibility

- **Rust**: 1.81+ (MSRV due to ML dependencies)
- **Chess**: 3.2+ (stable API)
- **CUDA**: 11.0+ (optional)
- **Metal**: macOS 10.15+ (optional)

## License

MIT OR Apache-2.0 - Choose the license that best fits your project.