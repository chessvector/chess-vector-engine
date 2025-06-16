# Chess Vector Engine

A **Rust library** for vector-based chess position analysis using similarity search, neural compression, and opening book integration to evaluate positions and suggest moves based on learned patterns.

[![Tests](https://img.shields.io/badge/tests-26%20passing-brightgreen)](#testing)
[![Rust](https://img.shields.io/badge/rust-stable-orange)](https://www.rust-lang.org/)

## 🚀 Features

- **📐 Vector Position Encoding** - Convert chess positions to 1024-dimensional vectors capturing piece positions, game state, and strategic features
- **🔍 Similarity Search** - Find similar positions using cosine similarity with linear and LSH-based search
- **🧠 Neural Compression** - Autoencoder networks compress vectors 8:1 (1024d → 128d) while maintaining accuracy
- **📖 Opening Book** - Integrated opening book with standard chess openings (8 positions, 7 ECO codes) for fast lookup
- **🎯 Move Recommendations** - Intelligent move suggestions based on similar positions with confidence scoring
- **⚡ Performance Optimized** - LSH indexing provides 3.3x speedup, opening book gives 7.7x speedup over linear search

## 🏗️ Architecture

The engine provides a complete pipeline from chess positions to intelligent recommendations:

```
Chess Position → Position Encoder → Vector (1024d)
                                     ↓
                 ┌─ Opening Book (fast lookup)
                 │                  ↓
                 └─ Manifold Learner → Compressed Vector (128d)
                                     ↓
                   LSH Index → Similar Positions → Move Recommendations
```

## 🎮 Quick Start

### Using as a Library

Add to your `Cargo.toml`:
```toml
[dependencies]
chess-vector-engine = "0.1.0"
chess = "3.2"
```

Basic usage:
```rust
use chess_vector_engine::ChessVectorEngine;
use chess::Board;

// Create engine with opening book
let mut engine = ChessVectorEngine::new(1024);
engine.enable_opening_book();

// Add positions to knowledge base
let board = Board::default();
engine.add_position(&board, 0.0);

// Get evaluation (checks opening book first, then similarity search)
if let Some(eval) = engine.evaluate_position(&board) {
    println!("Position evaluation: {:.2}", eval);
}

// Get move recommendations
let recommendations = engine.recommend_moves(&board, 3);
for (i, rec) in recommendations.iter().enumerate() {
    println!("{}. {} (confidence: {:.2})", 
             i + 1, rec.chess_move, rec.confidence);
}
```

### Advanced Features

```rust
// Enable neural compression (8:1 ratio)
engine.enable_manifold_learning(8.0)?;
engine.train_manifold_learning(50)?; // 50 epochs

// Enable LSH for fast similarity search
engine.enable_lsh(8, 16); // 8 tables, 16-bit hashes

// Enable LSH in compressed manifold space
engine.enable_manifold_lsh(8, 16)?;

// Check compression ratio
if let Some(ratio) = engine.manifold_compression_ratio() {
    println!("Compression ratio: {:.1}x", ratio);
}
```

## 🏃‍♂️ Demo Applications

Run the included demos to see the engine in action:

```bash
# Basic engine demonstration
cargo run --bin demo

# Position analysis with opening book
cargo run --bin analyze "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"

# Performance benchmarking
cargo run --bin benchmark

# LSH vs linear search comparison
cargo run --bin lsh_benchmark

# Neural compression demonstration
cargo run --bin manifold_demo

# Move recommendations
cargo run --bin move_recommendation_demo

# Manifold learning + LSH integration
cargo run --bin manifold_lsh_demo

# Opening book demonstration
cargo run --bin opening_book_demo
```

## 🎓 Training the Engine

### 1. Load Training Data

The engine includes quality training data with 147 diverse chess positions:

```rust
use chess_vector_engine::ChessVectorEngine;
use serde_json::Value;

let mut engine = ChessVectorEngine::new(1024);

// Load from included training data
let content = std::fs::read_to_string("training_data.json")?;
let positions: Vec<Value> = serde_json::from_str(&content)?;

for position in &positions {
    if let (Some(fen), Some(eval)) = (position["fen"].as_str(), position["evaluation"].as_f64()) {
        if let Ok(board) = Board::from_str(fen) {
            engine.add_position(&board, eval as f32);
        }
    }
}
```

### 2. Train Neural Compression

```rust
// Enable manifold learning with 8:1 compression
engine.enable_manifold_learning(8.0)?;

// Train autoencoder (adjust epochs based on data size)
engine.train_manifold_learning(50)?;

// Training output shows loss reduction:
// Epoch 0: Loss = 1.984517
// Epoch 10: Loss = 1.141460
// Epoch 20: Loss = 1.071337
// Training completed!
```

### 3. Enable Fast Search

```rust
// Enable LSH for original vectors
engine.enable_lsh(8, 16);

// Enable LSH for compressed vectors (best performance)
engine.enable_manifold_lsh(8, 16)?;
```

### 4. Add Custom Training Data

```rust
// From PGN files using the training module
use chess_vector_engine::TrainingDataset;

let mut dataset = TrainingDataset::new();
dataset.load_from_pgn("games.pgn", Some(100), 30)?; // 100 games, 30 moves each
dataset.evaluate_with_stockfish(15)?; // Evaluate with Stockfish depth 15

// Add to engine
for data in &dataset.data {
    engine.add_position(&data.board, data.evaluation);
    
    // Optionally add move information
    engine.add_position_with_move(
        &data.board,
        data.evaluation,
        Some(best_move),
        Some(outcome)
    );
}
```

### 5. Training with the CLI

Use the built-in training binary:

```bash
# Train from PGN file
cargo run --bin train -- --pgn games.pgn --evaluate --output trained_data.json

# Train from existing dataset
cargo run --bin train -- --dataset existing_data.json --max-games 500

# Enable LSH during training
cargo run --bin train -- --dataset data.json --enable-lsh
```

## 📊 Performance Characteristics

| Component | Speed | Memory | Accuracy |
|-----------|-------|---------|----------|
| Linear Search | 154-421 qps | 4KB/position | 100% |
| LSH Search | 3.3x speedup | +overhead | 95%+ |
| Opening Book | 7.7x speedup | minimal | 100% |
| Neural Compression | varies | 8:1 reduction | 95%+ |
| Full Pipeline | up to 10x faster | 50% less memory | 95%+ |

### Benchmarks

```bash
# Position encoding: 7,812 positions/second
# Search performance: 154-421 queries/second (dataset dependent)
# Neural training: ~1 minute for 147 positions, 50 epochs
# Opening book lookup: 7.7x faster than similarity search
```

## 🧠 How It Works

### Position Encoding

Converts chess positions to 1024-dimensional vectors capturing:
- **Piece positions** (64 squares × 12 piece types × 1 dimension = 768d)
- **Game state** (castling rights, en passant, turn to move = 8d)
- **Material balance** (piece values = 6d)
- **Positional features** (center control, development, king safety = 242d)

### Similarity Search

Uses cosine similarity to find positions with similar strategic characteristics:
```rust
similarity = dot_product(v1, v2) / (norm(v1) * norm(v2))
```

### Neural Compression

Autoencoder neural network compresses vectors while preserving strategic information:
```
Input (1024d) → Hidden (512d) → Compressed (128d) → Hidden (512d) → Output (1024d)
```

### Opening Book Integration

Fast hash-map lookup provides instant access to opening theory:
- **8 standard opening positions** with ECO codes
- **High-quality move recommendations** from chess theory
- **Accurate evaluations** for opening phases
- **Seamless fallback** to similarity search for non-opening positions

## 🔧 API Reference

### Core Engine

```rust
// Creation
let mut engine = ChessVectorEngine::new(vector_size);
let mut engine = ChessVectorEngine::new_with_lsh(vector_size, num_tables, hash_size);

// Position management
engine.add_position(&board, evaluation);
engine.add_position_with_move(&board, eval, Some(chess_move), Some(outcome));

// Analysis
let eval = engine.evaluate_position(&board);
let similar = engine.find_similar_positions(&board, k);
let recommendations = engine.recommend_moves(&board, num_moves);
let legal_recs = engine.recommend_legal_moves(&board, num_moves);

// Configuration
engine.enable_opening_book();
engine.enable_lsh(num_tables, hash_size);
engine.enable_manifold_learning(compression_ratio)?;
engine.train_manifold_learning(epochs)?;
engine.enable_manifold_lsh(num_tables, hash_size)?;

// Information
let is_opening = engine.is_opening_position(&board);
let entry = engine.get_opening_entry(&board);
let stats = engine.opening_book_stats();
let ratio = engine.manifold_compression_ratio();
```

### Data Structures

```rust
// Move recommendation with confidence
pub struct MoveRecommendation {
    pub chess_move: ChessMove,
    pub confidence: f32,
    pub from_similar_position_count: usize,
    pub average_outcome: f32,
}

// Opening book entry
pub struct OpeningEntry {
    pub evaluation: f32,
    pub best_moves: Vec<(ChessMove, f32)>,
    pub name: String,
    pub eco_code: Option<String>,
}
```

## 🏗️ Project Structure

```
src/
├── lib.rs                    # Main engine API
├── position_encoder.rs       # Chess → Vector conversion
├── similarity_search.rs      # Linear k-NN search
├── lsh.rs                   # Locality Sensitive Hashing
├── manifold_learner.rs      # Neural autoencoder
├── ann.rs                   # Approximate Nearest Neighbors
├── opening_book.rs          # Opening book integration
├── training.rs              # Training utilities
└── bin/
    ├── demo.rs              # Basic demonstration
    ├── analyze.rs           # Position analysis tool
    ├── benchmark.rs         # Performance testing
    ├── lsh_benchmark.rs     # LSH comparison
    ├── manifold_demo.rs     # Neural compression demo
    ├── manifold_lsh_demo.rs # Integrated demo
    ├── move_recommendation_demo.rs # Move suggestions
    ├── opening_book_demo.rs # Opening book demo
    └── train.rs             # Training CLI
```

## 🧪 Testing

```bash
# Run all tests (26 tests covering all modules)
cargo test

# Run specific module tests
cargo test position_encoder
cargo test similarity_search
cargo test manifold_learner
cargo test opening_book

# Run with output
cargo test -- --nocapture
```

Tests cover:
- ✅ Position encoding consistency and accuracy
- ✅ Similarity calculation correctness
- ✅ Search functionality and performance
- ✅ Neural network training and compression
- ✅ Opening book lookup and integration
- ✅ Move recommendation accuracy
- ✅ End-to-end engine functionality

## 🔬 Research Applications

This engine is designed for:

### Chess AI Research
- **Pattern Recognition** - Identify strategic patterns across large game databases
- **Position Evaluation** - Learn evaluation functions from master games
- **Opening Analysis** - Discover new opening ideas through similarity search
- **Endgame Studies** - Find similar endgame positions for learning

### Machine Learning
- **Representation Learning** - Chess positions as a benchmark for vector representations
- **Manifold Learning** - Explore high-dimensional chess position space
- **Transfer Learning** - Apply learned chess patterns to other domains
- **Neural Architecture** - Test compression and search architectures

### Chess Education
- **Training Tools** - Help players recognize similar positions
- **Analysis Software** - Provide context for position evaluation
- **Pattern Recognition** - Identify tactical and strategic themes
- **Opening Preparation** - Find similar positions in opening repertoire

## 🤝 Contributing

This library is designed for extension and contribution:

### Adding New Features
- **Position Encoders** - Implement new ways to convert positions to vectors
- **Similarity Metrics** - Add alternative distance/similarity functions
- **Search Algorithms** - Integrate new approximate nearest neighbor methods
- **Neural Architectures** - Experiment with different compression networks

### Performance Improvements
- **Parallel Processing** - Add multi-threading to similarity search
- **GPU Acceleration** - Implement CUDA/OpenCL for vector operations
- **Memory Optimization** - Reduce memory footprint for large databases
- **Cache Optimization** - Add intelligent caching for frequent queries

### Integration Enhancements
- **UCI Protocol** - Add UCI engine integration for chess GUIs
- **Web Assembly** - Compile to WASM for browser applications
- **Python Bindings** - Add PyO3 bindings for Python integration
- **C API** - Provide C interface for broader language support

## 📈 Roadmap

### Completed ✅
- **Core engine architecture** with position encoding and similarity search
- **LSH implementation** for approximate nearest neighbor search
- **Neural compression** using autoencoder networks (8:1 ratio)
- **Opening book integration** with standard chess openings
- **Move recommendation system** with confidence scoring
- **Comprehensive testing** (26 tests) and documentation

### Next Steps
- **Variational Autoencoders** - Better compression with probabilistic models
- **Transformer Architecture** - Attention-based position understanding
- **Advanced Opening Book** - Expand to thousands of opening positions
- **Tactical Pattern Recognition** - Specialized encoding for tactical motifs
- **Game Phase Detection** - Separate models for opening/middlegame/endgame

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **chess crate** - Excellent Rust chess library foundation
- **candle** - Modern neural network framework for Rust
- **ndarray** - Efficient numerical computing in Rust
- **Chess community** - For opening theory and position databases

---

*Built with ❤️ for the chess and Rust communities*