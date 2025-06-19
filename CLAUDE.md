# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Rust library** (`chess-vector-engine`) that provides vector-based chess position analysis using similarity search. The core concept is encoding chess positions into high-dimensional vectors (1024d) and using similarity metrics to evaluate positions and suggest moves based on learned patterns.

## Commands

### Development
- `cargo build` - Build the library
- `cargo test` - Run all tests (22 tests covering all modules)
- `cargo run --bin demo` - Run basic demonstration
- `cargo run --bin benchmark` - Performance testing and scaling analysis
- `cargo run --bin lsh_benchmark` - Compare LSH vs linear search performance
- `cargo run --bin manifold_demo` - Demonstrate neural network compression
- `cargo run --bin analyze <FEN>` - Position analysis tool with opening book
- `cargo run --bin training_benchmark` - Test training performance optimizations
- `cargo run --bin performance_benchmark` - Benchmark new optimizations (Stockfish pool, DB batching, binary format)
- `cargo run --bin tactical_training -- --puzzles <CSV_FILE>` - Train with Lichess puzzles
- `cargo run --bin self_play_training` - Self-play training (optimized: binary format, DB batching, Stockfish pool)
- `cargo run --bin play_stockfish` - Play against Stockfish with the trained engine

### Publishing
- `cargo publish` - Publish library to crates.io (when ready)

## Architecture

The engine consists of five main components:

1. **PositionEncoder** (`src/position_encoder.rs`) - Converts chess positions to 1024-dimensional vectors using piece positions, game state, material balance, and positional features
2. **SimilaritySearch** (`src/similarity_search.rs`) - Linear k-NN search through position vectors using cosine similarity
3. **LSH** (`src/lsh.rs`) - Locality Sensitive Hashing for approximate nearest neighbor search to break linear scaling
4. **ManifoldLearner** (`src/manifold_learner.rs`) - Neural network autoencoder for position compression (8:1 to 32:1 ratios)
5. **ANN** (`src/ann.rs`) - Comprehensive approximate nearest neighbor system with multiple search strategies

### Core Data Flow
```
Chess Position → PositionEncoder → Vector (1024d) → [LSH/ANN/Linear] → Similar Positions → Weighted Evaluation
                                      ↓
                              ManifoldLearner → Compressed Vector (64-128d)
```

## Key Dependencies

- `chess` (3.2) - Chess game logic and position representation
- `ndarray` (0.16) - Numerical arrays for vector operations  
- `candle-core/candle-nn` (0.9) - Neural network framework for future manifold learning
- `criterion` (0.5) - Benchmarking framework

## Library Usage

This is designed as a library-first project. The main API is through `ChessVectorEngine`:

```rust
let mut engine = ChessVectorEngine::new(1024);
engine.add_position(&board, evaluation);
let similar = engine.find_similar_positions(&board, 5);
let eval = engine.evaluate_position(&board);
```

## Future Architecture Goals

The project roadmap includes implementing manifold learning to compress 1024d vectors into 64-128d space using autoencoders, enabling more efficient search through millions of positions rather than the current linear search approach.