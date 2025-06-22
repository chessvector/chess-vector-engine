# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **production-ready Rust library** and **UCI chess engine** (`chess-vector-engine`) that implements an **open-core business model** with hybrid chess evaluation combining vector-based pattern recognition with advanced tactical search and NNUE neural networks. The core concept is encoding chess positions into high-dimensional vectors (1024d) and using similarity metrics combined with sophisticated 6-10+ ply tactical search with advanced pruning techniques to evaluate positions and suggest moves based on learned patterns.

### Open-Core Architecture

The project uses an **open-core business model** with three tiers:

- **Open Source** (MIT/Apache-2.0): Basic UCI engine, position encoding, similarity search, opening book, 6-ply tactical search, JSON training data support
- **Premium** (Commercial License): GPU acceleration, NNUE networks, ultra-fast loading, 10+ ply search, multi-threading, advanced pruning, pondering, Multi-PV
- **Enterprise** (Enterprise License): Distributed training, cloud deployment, enterprise analytics, custom algorithms, unlimited positions, dedicated support

All features are developed in a single repository with runtime license verification controlling access to premium features.

## Commands

### Development
- `cargo build` - Build the library
- `cargo test` - Run all tests (105+ tests covering all modules with 99%+ pass rate)
- `cargo run --bin demo` - Run basic demonstration
- `cargo run --bin uci_engine` - UCI chess engine (for chess GUIs)
- `cargo run --bin analyze <FEN>` - Position analysis tool with opening book
- `cargo run --bin benchmark` - Performance testing and scaling analysis
- `cargo run --bin feature_demo` - Demonstrate feature gating system across all tiers
- `cargo run --bin license_demo` - Comprehensive license verification system demonstration

### Publishing and Release
- `cargo package --allow-dirty` - Create publishable package (excludes development binaries and training data)
- `cargo publish` - Publish open source version to crates.io
- `cargo build --release` - Build commercial version with all features

## Architecture

The engine implements a **hybrid evaluation system** with **open-core architecture** combining pattern recognition with advanced tactical search across multiple components:

### Core Components

1. **PositionEncoder** (`src/position_encoder.rs`) - Converts chess positions to 1024-dimensional vectors using piece positions, game state, material balance, and positional features
2. **SimilaritySearch** (`src/similarity_search.rs`) - Linear k-NN search through position vectors using cosine similarity with memory-efficient iterators and SIMD optimizations (Premium+)
3. **LSH** (`src/lsh.rs`) - Locality Sensitive Hashing for approximate nearest neighbor search to break linear scaling
4. **ManifoldLearner** (`src/manifold_learner.rs`) - Memory-optimized neural network autoencoder for position compression (8:1 to 32:1 ratios) with sequential batch processing
5. **TacticalSearch** (`src/tactical_search.rs`) - Advanced 6-10+ ply search with PVS, sophisticated pruning (futility, razoring, extended futility), enhanced LMR, multi-threading (Premium+), and advanced move ordering
6. **NNUE** (`src/nnue.rs`) - Efficiently Updatable Neural Networks for fast position evaluation with incremental updates and hybrid blending (Premium+)
7. **OpeningBook** (`src/opening_book.rs`) - Fast hash-map lookup for 50+ openings with ECO codes
8. **GPUAccelerator** (`src/gpu_acceleration.rs`) - Automatic CUDA/Metal/CPU device detection with 10-100x speedup potential (Premium+)
9. **UCIEngine** (`src/uci.rs`) - Full UCI protocol implementation with pondering (Premium+), Multi-PV analysis (Premium+), and comprehensive options
10. **Database** (`src/persistence.rs`) - SQLite persistence with instant startup and training resume
11. **Training** (`src/training.rs`) - Ultra-fast training with Stockfish process pools, batch operations, and binary format
12. **FeatureSystem** (`src/features.rs`) - Runtime feature gating with tier-based access control (OpenSource/Premium/Enterprise)
13. **LicenseVerifier** (`src/license.rs`) - License verification system with online/offline validation, caching, and subscription management
14. **AutoDiscovery** (`src/auto_discovery.rs`) - Intelligent training data file discovery and format prioritization
15. **UltraFastLoader** (`src/ultra_fast_loader.rs`) - Memory-mapped and streaming loaders for massive datasets (900k+ positions) (Premium+)

### Hybrid Evaluation Pipeline
```
Chess Position → PositionEncoder → Vector (1024d)
                     ↓
    ┌─ Opening Book (instant lookup) ─┐
    │                                 ↓
    ├─ Pattern Recognition ──→ Confidence Assessment
    │   (similarity search)           ↓
    │                          ┌─ High Confidence → Pattern Evaluation
    │                          └─ Low Confidence → Tactical Search (PVS 6-10+ ply)
    │                                 ↓
    └─────────────→ Hybrid Blending ──→ Final Evaluation
                         ↓
            NNUE Evaluation (Premium+) → Neural Position Assessment
                         ↓
            ManifoldLearner (memory optimized) → Compressed Vector (64-128d)
                         ↓
               GPU Acceleration (Premium+) → 10-100x speedup
```

### Open-Core Business Model Integration

#### Feature Gating Implementation
All premium features are protected by runtime license verification:

```rust
// Basic feature gating
pub fn ultra_fast_load_any_format<P: AsRef<Path>>(&mut self, path: P) -> Result<(), Box<dyn std::error::Error>> {
    self.require_feature("ultra_fast_loading")?;  // Premium+ required
    // ... implementation
}

// License verification
pub async fn activate_license(&mut self, key: &str) -> Result<FeatureTier, LicenseError> {
    if let Some(ref mut checker) = self.licensed_feature_checker {
        let tier = checker.activate_license(key).await?;
        self.feature_checker.upgrade_tier(tier.clone());
        Ok(tier)
    } else {
        Err(LicenseError::InvalidFormat("No license checker initialized".to_string()))
    }
}
```

#### Release Strategy
1. **Single Repository Development**: All features developed in this repository with feature gating
2. **Open Source Release**: `cargo publish` releases only open source features to crates.io
3. **Commercial Distribution**: Full binaries distributed via web platform with license keys
4. **Version Synchronization**: Both open source and commercial versions share same version numbers

#### Repository Structure for Business Model
- **This Repository**: Complete open-core codebase with all features and license verification
- **Public Cargo Release**: Automated publishing of open source features only
- **Web Platform** (Separate Repo): Nuxt.js + TailwindCSS frontend for subscription management
- **Backend API** (Separate Repo): Axum-based license server with Stripe integration

## Key Dependencies

- `chess` (3.2) - Chess game logic and position representation
- `ndarray` (0.16) - Numerical arrays for vector operations  
- `candle-core/candle-nn` (0.9) - Neural network framework for manifold learning and NNUE
- `criterion` (0.5) - Benchmarking framework
- `rusqlite` (0.32) - SQLite database for persistence
- `rayon` (1.10) - Data parallelism for multithreading
- `serde` (1.0) - Serialization for training data and persistence
- `tokio` (1.0) - Async runtime for license verification

## Library Usage

This is designed as a library-first project with both pattern recognition and UCI engine capabilities. The main API is through `ChessVectorEngine` with feature gating:

### Open Source Usage
```rust
// Open source features (always available)
let mut engine = ChessVectorEngine::new(1024);
engine.enable_opening_book();
let evaluation = engine.evaluate_position(&board);
let similar = engine.find_similar_positions(&board, 5);
```

### Premium Usage (License Required)
```rust
// Premium features require license activation
let mut engine = ChessVectorEngine::new_with_offline_license(1024);
engine.activate_license("PREMIUM-LICENSE-KEY").await?;

// Now premium features are available
engine.enable_gpu_acceleration()?;  // Requires Premium+
engine.ultra_fast_load_any_format("data.bin")?;  // Requires Premium+
engine.configure_hybrid_evaluation(HybridConfig {
    pattern_confidence_threshold: 0.75,
    pattern_weight: 0.6,
    ..Default::default()
});
```

### License Management
```rust
// License verification and caching
let mut engine = ChessVectorEngine::new_with_license(1024, "https://api.yourdomain.com/license".to_string());
engine.load_license_cache("license_cache.json")?;

match engine.activate_license("YOUR-LICENSE-KEY").await {
    Ok(tier) => println!("Activated {:?} tier", tier),
    Err(e) => println!("License error: {}", e),
}

engine.save_license_cache("license_cache.json")?;
```

### Feature Availability Checking
```rust
// Check feature availability before use
if engine.is_feature_available("gpu_acceleration") {
    engine.enable_gpu_acceleration()?;
} else {
    println!("GPU acceleration requires Premium+ license");
}

// Or use direct feature checking with error handling
engine.require_feature("ultra_fast_loading")?;  // Throws error if not licensed
```

## Performance Characteristics

### Loading Performance (30k+ positions)
- **Memory-mapped (.mmap)**: Instant startup (zero-copy loading) - *Premium+*
- **MessagePack (.msgpack)**: 10-20% faster than bincode, smaller files - *Premium+*
- **Zstd compressed (.zst)**: Best compression ratios with fast decompression - *Premium+*
- **LZ4 binary (.bin)**: 5-15x faster than JSON (current optimized format) - *All tiers*
- **Streaming JSON**: Parallel processing for large JSON files - *All tiers*
- **900k Position Loading**: Solved performance bottleneck (3.5+ hours → ~2 minutes with ultra-fast loader)

### Memory Usage (30k+ positions)
- **Before optimization**: ~1GB (multiple dataset copies)
- **After optimization**: ~150-200MB (75-80% reduction)
- **Streaming processing**: Handles arbitrarily large datasets efficiently

### Search Performance
- **Basic tactical search**: 1000+ nodes/ms (6-ply) - *Open Source*
- **Advanced tactical search**: 2800+ nodes/ms with PVS (10+ ply) - *Premium+*
- **GPU acceleration**: 10-100x speedup for large datasets - *Premium+*
- **Multi-threading**: 2-4x speedup with parallel search - *Premium+*
- **SIMD vector operations**: 2-4x speedup for similarity calculations - *Premium+*

## Important Implementation Notes

### Open-Core Feature Implementation

The engine includes comprehensive feature gating for the open-core business model:

1. **Feature Registry** (`features.rs:16-84`): Maps features to required tiers (OpenSource/Premium/Enterprise)
2. **License Verification** (`license.rs:82-380`): Online/offline license validation with caching and expiration handling
3. **Runtime Feature Checking** (`lib.rs:207-210`): All premium features protected by `require_feature()` calls
4. **Tier Management** (`lib.rs:185-200`): License activation automatically upgrades feature tier
5. **Commercial Integration** (`lib.rs:1172-1222`): License verification integrated into engine constructors

### Performance Optimization Implementation Details

The engine includes 7 major performance optimizations for production-ready performance:

1. **SIMD Vector Operations** (`similarity_search.rs:340-480`, `lsh.rs:380-520`): AVX2/SSE4.1/NEON optimized dot products for 2-4x similarity calculation speedup
2. **Pre-computed Vector Norms** (`similarity_search.rs:8-14`): Cached `norm_squared` in `PositionEntry` for 3x faster cosine similarity
3. **Reference-based Search Results** (`similarity_search.rs:22-35`): Zero-copy search patterns with `search_ref()` methods for 50% memory reduction
4. **Dynamic LSH Hash Table Sizing** (`lsh.rs:27-55`): Adaptive capacity allocation and replacement strategy for 30% search improvement
5. **Parallel Neural Network Training** (`manifold_learner.rs:203-276`): Concurrent batch processing with `train_parallel()` for 2-3x training speedup
6. **Custom Transposition Tables** (`tactical_search.rs:9-53`): Fixed-size cache with replacement policy for 40% tactical search improvement
7. **Ultra-Fast Position Loading** (`ultra_fast_loader.rs`, `lib.rs:1238-1285`): O(n²) → O(n) duplicate detection with HashSet and memory-mapped files for startup time reduction from hours to seconds

### Memory Optimization Implementation Details

When working with manifold learning code, be aware of these memory optimization patterns:

1. **Training Data Preparation** (`lib.rs:670-676`): Uses `Array2::from_shape_fn()` instead of collecting vectors
2. **Batch Processing** (`manifold_learner.rs:250-320`): Sequential batch processing in `train_memory_efficient()`
3. **Position Access** (`similarity_search.rs:321-323`): `iter_positions()` provides references instead of clones
4. **Index Rebuilding** (`lib.rs:710`): Uses iterator pattern to avoid dataset copies

### When Memory Issues Occur

Memory problems typically manifest when:
- Dataset size > 30,000 positions
- Manifold learning is enabled without license verification
- Multiple training iterations without optimization

The optimizations eliminate these issues by processing data in streams rather than loading everything into memory simultaneously.

### Cargo Release Configuration

The published package excludes development and testing binaries:

- **Included Binaries**: `demo`, `uci_engine`, `analyze`, `benchmark`, `feature_demo`, `license_demo`
- **Excluded Files**: All training data, database files, development binaries, test utilities
- **Package Size**: ~740KB (159KB compressed)
- **License**: Open source features only (MIT OR Apache-2.0)

## Documentation Guidelines

### README Requirements

When updating documentation, especially the README:

- Use technical, professional language focused on features and capabilities
- Clearly distinguish between open source and premium features
- Include comprehensive business model and release strategy sections
- Provide concrete code examples for both open source and premium usage
- Document the license verification system and subscription tiers
- Include performance metrics and optimization details
- Structure information hierarchically with clear sections
- Focus on factual information about performance, architecture, and usage

### Business Model Documentation

Always include:
- Clear feature tier comparison table
- License verification examples
- Release process explanation
- Repository structure for multi-tier development
- Version synchronization strategy
- Commercial distribution approach

## Release Strategy

### Open Source Release (Cargo)
1. **Package Creation**: `cargo package --allow-dirty` (excludes commercial development files)
2. **Publication**: `cargo publish` (automated in CI/CD)
3. **Version Management**: Semantic versioning synchronized with commercial releases
4. **Feature Set**: Open source tier only, premium features gated

### Commercial Release (Web Platform)
1. **Full Binary Build**: `cargo build --release` with all features enabled
2. **License Integration**: Distributed with license verification system
3. **Subscription Management**: Handled by separate web platform (Nuxt.js + Axum)
4. **Key Distribution**: License keys generated server-side, validated locally with caching

### Development Workflow
1. **Single Repository**: All development in this repository with feature gating
2. **Feature Branches**: Develop premium features with appropriate gating from start
3. **Testing**: Both open source and premium features tested together
4. **Release**: Simultaneous open source and commercial releases with same version number

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.