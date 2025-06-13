# Chess Vector Engine

A **library** for vector-based chess position analysis using similarity search to evaluate positions and suggest moves based on learned patterns.

## Library-First Design

This is primarily a **Rust library** (`chess-vector-engine`) that other projects can use. It comes with several example binaries that demonstrate different use cases:

- **`demo`** - Basic demonstration of the core concepts
- **`benchmark`** - Performance testing and scaling analysis  
- **`analyze`** - Position analysis tool with opening book

## Usage as a Library

Add to your `Cargo.toml`:
```toml
[dependencies]
chess-vector-engine = "0.1.0"
```

Basic usage:
```rust
use chess::{Board, ChessMove};
use chess_vector_engine::ChessVectorEngine;
use std::str::FromStr;

// Create engine with 1024-dimensional vectors
let mut engine = ChessVectorEngine::new(1024);

// Add known positions with evaluations
let board = Board::default();
engine.add_position(&board, 0.0);  // Starting position is equal

// Find similar positions
let similar = engine.find_similar_positions(&board, 5);

// Get evaluation for a position
if let Some(eval) = engine.evaluate_position(&board) {
    println!("Predicted evaluation: {}", eval);
}

// Calculate similarity between positions
let similarity = engine.calculate_similarity(&board1, &board2);
```

## Publishing the Library

Once you're ready to share this with the chess community:

```bash
# Add metadata to Cargo.toml (already done above)
# Then publish to crates.io
cargo publish
```

Other developers could then use it:
```toml
[dependencies]
chess-vector-engine = "0.1.0"
```

## Integration Examples

This library is designed to integrate with:
- **Chess GUIs** (like existing UCI engines)
- **Web applications** (chess analysis tools)
- **Research projects** (AI/ML chess experiments)
- **Chess databases** (position similarity search)
- **Training tools** (pattern recognition for players)

## API Design Philosophy

The library provides:
- **Simple interface** - Easy to get started with basic functionality
- **Extensible architecture** - Easy to add new encoding strategies or similarity metrics
- **Performance focus** - Designed for real-time position analysis
- **Type safety** - Rust's type system prevents common chess programming errors

## Next Steps (Your Original Vision)

### 1. Manifold Learning (`manifold_learner.rs` - placeholder)
- Implement autoencoder neural networks using `candle-core`
- Compress 1024-dimensional vectors into 64-128 dimensional manifold space
- Learn the underlying structure of "good" chess positions
- Navigate along the manifold for more efficient search

### 2. Advanced Similarity Search
- Implement Locality Sensitive Hashing (LSH) for faster approximate search
- Add hierarchical clustering for better position organization
- Use learned hash functions specific to chess position similarity

### 3. Strategic Sampling
- Instead of storing every position, store strategic "anchor points"
- Build hierarchy: opening patterns → middlegame themes → endgame structures
- Use representative positions to guide search

### 4. Move Recommendation
- Extend from position evaluation to move suggestion
- Find similar positions and their best moves
- Weight recommendations by position similarity

## Mathematical Foundation

The approach addresses your key insight about finding "deterministic paths through seemingly undeterministic collections":

- **Manifold Hypothesis**: Good chess positions lie on a lower-dimensional manifold
- **Similarity Metrics**: Cosine similarity captures strategic resemblance
- **Interpolation**: Navigate between known good positions in vector space
- **Dimensionality Reduction**: Compress to essential strategic features

## Architecture

```
Chess Position → Position Encoder → Vector (1024d)
                                      ↓
Knowledge Base ← Similarity Search ← Query Vector
      ↓
Similar Positions → Weighted Average → Evaluation/Move
```

## Files Structure

- `src/lib.rs` - Main engine interface
- `src/position_encoder.rs` - Chess position to vector conversion
- `src/similarity_search.rs` - Efficient k-NN search
- `src/manifold_learner.rs` - Future manifold learning (placeholder)
- `src/main.rs` - Demo and usage examples

## Dependencies

- `chess` - Chess game logic and position representation
- `ndarray` - Numerical arrays for vector operations
- `candle-core` - Neural network framework for future manifold learning
- `serde` - Serialization for model persistence

## Testing

```bash
cargo test
```

Tests cover:
- Position encoding consistency
- Similarity calculations
- Search functionality
- Integration between components

## Performance Considerations

- **Current**: Linear search through all positions (fine for < 10k positions)
- **Future**: LSH and manifold compression for millions of positions
- **Memory**: ~4KB per position vector (1024 floats)
- **Search Time**: O(n) currently, target O(log n) with advanced indexing