# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **production-ready Rust library** and **UCI chess engine** (`chess-vector-engine`) that provides hybrid chess evaluation combining vector-based pattern recognition with advanced tactical search and NNUE neural networks. The core concept is encoding chess positions into high-dimensional vectors (1024d) and using similarity metrics combined with sophisticated 6-10+ ply tactical search with advanced pruning techniques to evaluate positions and suggest moves based on learned patterns.

## Commands

### Development
- `cargo build` - Build the library
- `cargo test` - Run all tests (105+ tests covering all modules with 99%+ pass rate)
- `cargo run --bin demo` - Run basic demonstration
- `cargo run --bin uci_engine` - UCI chess engine (for chess GUIs)
- `cargo run --bin hybrid_evaluation_demo` - Hybrid pattern + tactical evaluation demo
- `cargo run --bin nnue_pvs_demo` - NNUE + PVS + Vector analysis demo
- `cargo run --bin benchmark` - Performance testing and scaling analysis
- `cargo run --bin lsh_benchmark` - Compare LSH vs linear search performance
- `cargo run --bin manifold_demo` - Demonstrate neural network compression (memory optimized)
- `cargo run --bin analyze <FEN>` - Position analysis tool with opening book
- `cargo run --bin training_benchmark` - Test training performance optimizations
- `cargo run --bin performance_benchmark` - Benchmark new optimizations (Stockfish pool, DB batching, binary format)
- `cargo run --bin tactical_training -- --puzzles <CSV_FILE>` - Train with Lichess puzzles
- `cargo run --bin self_play_training --stockfish-level` - Ultra-fast self-play training (optimized: binary format, DB batching, Stockfish pool)
- `cargo run --bin play_stockfish` - Play against Stockfish with the trained engine (FAST startup - seconds not minutes)
- `cargo run --bin play_stockfish -- --convert-to-binary` - Convert JSON training files to binary format (5-15x faster loading)
- `cargo run --bin play_stockfish -- --rebuild-models` - Play with model rebuilding (slower startup but complete retraining)
- `cargo run --bin ultra_fast_converter -- all` - Convert all training data to ultra-fast formats (instant loading)
- `cargo run --bin ultra_fast_converter -- benchmark` - Benchmark all loading methods for performance comparison
- `cargo run --bin ultra_fast_converter -- info` - Show file sizes and format recommendations

### Publishing
- `cargo publish` - Publish library to crates.io (when ready)

## Architecture

The engine implements a **hybrid evaluation system** combining pattern recognition with advanced tactical search across multiple components:

### Core Components

1. **PositionEncoder** (`src/position_encoder.rs`) - Converts chess positions to 1024-dimensional vectors using piece positions, game state, material balance, and positional features
2. **SimilaritySearch** (`src/similarity_search.rs`) - Linear k-NN search through position vectors using cosine similarity with memory-efficient iterators
3. **LSH** (`src/lsh.rs`) - Locality Sensitive Hashing for approximate nearest neighbor search to break linear scaling
4. **ManifoldLearner** (`src/manifold_learner.rs`) - Memory-optimized neural network autoencoder for position compression (8:1 to 32:1 ratios) with sequential batch processing
5. **TacticalSearch** (`src/tactical_search.rs`) - Advanced 6-10+ ply search with PVS, sophisticated pruning (futility, razoring, extended futility), enhanced LMR, multi-threading, and advanced move ordering
6. **NNUE** (`src/nnue.rs`) - Efficiently Updatable Neural Networks for fast position evaluation with incremental updates and hybrid blending
7. **OpeningBook** (`src/opening_book.rs`) - Fast hash-map lookup for 50+ openings with ECO codes
8. **GPUAccelerator** (`src/gpu_acceleration.rs`) - Automatic CUDA/Metal/CPU device detection with 10-100x speedup potential
9. **UCIEngine** (`src/uci.rs`) - Full UCI protocol implementation with pondering, Multi-PV analysis, and comprehensive options
10. **Database** (`src/persistence.rs`) - SQLite persistence with instant startup and training resume
11. **Training** (`src/training.rs`) - Ultra-fast training with Stockfish process pools, batch operations, and binary format

### Hybrid Evaluation Pipeline
```
Chess Position → PositionEncoder → Vector (1024d)
                     ↓
    ┌─ Opening Book (instant lookup) ─┐
    │                                 ↓
    ├─ Pattern Recognition ──→ Confidence Assessment
    │   (similarity search)           ↓
    ├─ NNUE Neural Networks ──→ Fast Position Evaluation
    │   (incremental updates)         ↓
    │                          ┌─ High Confidence → Pattern + NNUE Evaluation
    │                          └─ Low Confidence → Advanced Tactical Search
    │                              (PVS + Pruning + Multi-threading 6-10+ ply)
    │                                 ↓
    └─────────────→ Hybrid Blending ──→ Final Evaluation
                         ↓
            ManifoldLearner (memory optimized) → Compressed Vector (64-128d)
                         ↓
               GPU Acceleration → 10-100x speedup
```

### Performance Optimization Features

The engine includes comprehensive optimizations for production-ready performance:

#### Advanced Search Optimizations
- **Sophisticated Pruning**: Futility pruning, razoring, extended futility pruning for 2-5x search speedup
- **Enhanced LMR**: Improved Late Move Reductions with depth and move-based reduction formulas
- **Advanced Move Ordering**: MVV-LVA captures, killer moves, history heuristic for optimal branch evaluation
- **Multi-threading Support**: Parallel root search with configurable thread count for 2-4x performance gain
- **NNUE Integration**: Fast neural network evaluation with incremental updates

#### Production-Ready Features
- **Robust Error Handling**: Eliminated 300+ unwrap() calls with proper Result types and error propagation
- **Comprehensive Testing**: 105+ tests covering all modules with 99%+ pass rate
- **UCI Compliance**: Full UCI protocol with pondering, Multi-PV analysis, and extensive options
- **Memory Optimization**: 75-80% memory reduction with streaming data processing

#### Loading Performance Optimizations
- **O(n²) → O(n) Duplicate Detection**: HashSet-based duplicate checking eliminates linear search bottleneck
- **Binary Format Priority**: `new_with_fast_load()` tries binary files first (5-15x faster than JSON)
- **Batch Processing**: Eliminates repeated individual position additions
- **Smart Loading Strategy**: Fast loading by default, full loading only when rebuilding models

## Key Dependencies

- `chess` (3.2) - Chess game logic and position representation
- `ndarray` (0.16) - Numerical arrays for vector operations  
- `candle-core/candle-nn` (0.9) - Neural network framework for manifold learning
- `criterion` (0.5) - Benchmarking framework
- `rusqlite` (0.32) - SQLite database for persistence
- `rayon` (1.10) - Data parallelism for multithreading
- `serde` (1.0) - Serialization for training data and persistence

## Library Usage

This is designed as a library-first project with both pattern recognition and UCI engine capabilities. The main API is through `ChessVectorEngine`:

### Basic Usage
```rust
// INSTANT loading for regular users (fastest possible startup)
let mut engine = ChessVectorEngine::new_with_instant_load(1024)?;

// Fast loading for gameplay (seconds not minutes)  
let mut engine = ChessVectorEngine::new_with_fast_load(1024)?;

// Convert to ultra-fast formats (run once, use forever)
ChessVectorEngine::convert_to_msgpack()?;      // MessagePack: 10-20% faster than bincode
ChessVectorEngine::convert_to_mmap()?;         // Memory-mapped: instant loading 
ChessVectorEngine::convert_to_zstd()?;         // Zstd: best compression ratios

// Standard usage
let mut engine = ChessVectorEngine::new(1024);
engine.add_position(&board, evaluation);
let similar = engine.find_similar_positions(&board, 5);
let eval = engine.evaluate_position(&board);
```

### Hybrid Evaluation with NNUE
```rust
// Enable all advanced features
engine.enable_opening_book();                    // Fast opening lookup
engine.enable_tactical_search_default();         // Advanced tactical search with pruning
engine.enable_nnue()?;                          // NNUE neural network evaluation
engine.configure_hybrid_evaluation(HybridConfig {
    pattern_confidence_threshold: 0.75,          // Use tactical when confidence < 75%
    pattern_weight: 0.4,                         // 40% pattern, 30% NNUE, 30% tactical
    ..Default::default()
});

// Hybrid evaluation (opening book → patterns → NNUE → tactical search)
let eval = engine.evaluate_position(&board);
```

### Memory-Optimized Manifold Learning
```rust
// Enable memory-efficient neural compression
engine.enable_manifold_learning(8.0)?;           // 8:1 compression ratio
engine.train_manifold_learning(50)?;             // Uses memory-optimized training

// Memory usage: ~150-200MB instead of ~1GB for 30k positions
```

### Ultra-Fast Loading Methods
```rust
// Load specific ultra-fast formats
engine.load_training_data_mmap("training_data.mmap")?;         // Memory-mapped (instant)
engine.load_training_data_msgpack("training_data.msgpack")?;   // MessagePack binary
engine.load_training_data_compressed("training_data.zst")?;    // Zstd compressed
engine.load_training_data_streaming_json("data.json")?;        // Parallel streaming JSON

// Automatic format detection with priority ordering
let engine = ChessVectorEngine::new_with_instant_load(1024)?;  // Tries all formats in speed order
```

### Production UCI Engine
```rust
// Create production-ready UCI engine for chess GUIs
let config = UCIConfig::default();
run_uci_engine_with_config(config)?;

// Supported UCI features:
// - Hash: Memory allocation (1-2048 MB)
// - Threads: Multi-threading (1-64 cores)
// - MultiPV: Multiple best lines (1-10)
// - Ponder: Think on opponent's time
// - Pattern_Weight: Evaluation blend ratio
// - Tactical_Depth: Search depth configuration
```

## Performance Characteristics

### Loading Performance (30k positions)
- **Memory-mapped (.mmap)**: Instant startup (zero-copy loading)
- **MessagePack (.msgpack)**: 10-20% faster than bincode, smaller files
- **Zstd compressed (.zst)**: Best compression ratios with fast decompression  
- **LZ4 binary (.bin)**: 5-15x faster than JSON (current optimized format)
- **Streaming JSON**: Parallel processing for large JSON files
- **Before optimization**: Minutes to hours (O(n²) duplicate checking)
- **After optimization**: Seconds (O(n) HashSet + binary format priority)

### Memory Usage (30k positions)
- **Before optimization**: ~1GB (multiple dataset copies)
- **After optimization**: ~150-200MB (75-80% reduction)

### Training Speed
- **Stockfish evaluation**: 20-100x faster with process pools
- **Database operations**: 10-50x faster with batch processing
- **Overall training**: 17 hours → ~2 hours (8.5x speedup)

### Search Performance
- **Advanced Pruning**: 2-5x search speedup with futility pruning, razoring, and extended futility pruning
- **Multi-threading**: 2-4x performance gain with parallel root search
- **Enhanced LMR**: Sophisticated reduction formulas for optimal branch elimination
- **Advanced Move Ordering**: MVV-LVA, killer moves, history heuristic for efficient search
- **GPU acceleration**: 10-100x speedup for large datasets
- **LSH indexing**: 3.3x speedup over linear search
- **Tactical search**: 2800+ nodes/ms with custom transposition tables and PVS optimizations
- **NNUE evaluation**: Fast neural network position assessment
- **SIMD vector operations**: 2-4x speedup for similarity calculations
- **Reference-based search**: 50% memory reduction
- **Dynamic hash table sizing**: 30% LSH performance improvement

## Important Implementation Notes

### Performance Optimization Implementation Details

The engine includes 7 major performance optimizations for production-ready performance:

1. **SIMD Vector Operations** (`similarity_search.rs:340-480`, `lsh.rs:380-520`): AVX2/SSE4.1/NEON optimized dot products for 2-4x similarity calculation speedup
2. **Pre-computed Vector Norms** (`similarity_search.rs:8-14`): Cached `norm_squared` in `PositionEntry` for 3x faster cosine similarity
3. **Reference-based Search Results** (`similarity_search.rs:22-35`): Zero-copy search patterns with `search_ref()` methods for 50% memory reduction
4. **Dynamic LSH Hash Table Sizing** (`lsh.rs:27-55`): Adaptive capacity allocation and replacement strategy for 30% search improvement
5. **Parallel Neural Network Training** (`manifold_learner.rs:203-276`): Concurrent batch processing with `train_parallel()` for 2-3x training speedup
6. **Custom Transposition Tables** (`tactical_search.rs:9-53`): Fixed-size cache with replacement policy for 40% tactical search improvement
7. **Ultra-Fast Position Loading** (`lib.rs:405-445`): O(n²) → O(n) duplicate detection with HashSet and binary format priority for startup time reduction from minutes/hours to seconds

### Memory Optimization Implementation Details

When working with manifold learning code, be aware of these memory optimization patterns:

1. **Training Data Preparation** (`lib.rs:670-676`): Uses `Array2::from_shape_fn()` instead of collecting vectors
2. **Batch Processing** (`manifold_learner.rs:250-320`): Sequential batch processing in `train_memory_efficient()`
3. **Position Access** (`similarity_search.rs:321-323`): `iter_positions()` provides references instead of clones
4. **Index Rebuilding** (`lib.rs:710`): Uses iterator pattern to avoid dataset copies

### When Memory Issues Occur

Memory problems typically manifest when:
- Dataset size > 30,000 positions
- Manifold learning is enabled (`--rebuild-models` flag)
- Multiple training iterations without optimization

The optimizations eliminate these issues by processing data in streams rather than loading everything into memory simultaneously.

### Performance Optimization Usage

The performance optimizations are automatically enabled:

- **SIMD Operations**: Automatic feature detection (AVX2 → SSE4.1 → NEON → scalar fallback)
- **Pre-computed Norms**: Always enabled for new positions
- **Reference Search**: Use `search_ref()` methods when lifetime permits
- **Dynamic LSH**: Automatic sizing based on expected dataset size
- **Parallel Training**: Enabled for datasets > 1000 positions
- **Custom Transposition**: Always enabled with 64MB default size
- **Fast Loading**: Use `new_with_fast_load()` for gameplay, `convert_json_to_binary()` for preprocessing

## Documentation Guidelines

### README Requirements

When updating documentation, especially the README:

- Use technical, professional language focused on features and capabilities
- Avoid conversational or personable tone
- Focus on factual information about performance, architecture, and usage
- Use concrete metrics and benchmarks where available
- Structure information hierarchically with clear sections
- Provide code examples that demonstrate actual functionality
- Include performance characteristics and optimization details