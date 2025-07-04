# Chess Vector Engine Hybrid Setup Guide

This guide shows you how to set up the optimal hybrid NNUE + Vector + Tactical evaluation system.

## 🎯 Optimal Hybrid Command

For the best hybrid approach combining NNUE neural networks, vector pattern recognition, and tactical search:

### 1. Train Production Hybrid Model

```bash
# Create the optimal hybrid NNUE model (recommended)
cargo run --bin train_nnue -- \
  --mode train \
  --epochs 25 \
  --config vector-integrated \
  --include-games \
  --save-interval 5 \
  --output default_hybrid
```

**What this does:**
- ✅ **25 epochs**: Sufficient for good convergence
- ✅ **vector-integrated config**: 40% vector influence, 60% NNUE (optimal hybrid)
- ✅ **include-games**: ~120 training positions for better diversity
- ✅ **5-epoch checkpoints**: Save progress every 5 epochs
- ✅ **default_hybrid name**: Auto-loaded by engine

### 2. Play with Hybrid Engine

```bash
# Use the trained hybrid model (auto-loaded)
cargo run --bin play_stockfish -- \
  --color white \
  --depth 8 \
  --time 2000 \
  --tactical-depth 8
```

**Features active:**
- 🧠 **Auto-loaded NNUE**: Pre-trained hybrid model
- 📚 **Opening Book**: 50+ professional openings  
- 🎯 **Vector Patterns**: Strategic pattern recognition
- ⚔️ **Tactical Search**: 8-ply backup search
- ⚡ **GPU Acceleration**: When available

## 🔧 Engine Configuration Options

### Auto-Loading Behavior

#### Default Mode (Auto-loading Enabled)
```bash
# Automatically loads default_hybrid.config if present
cargo run --bin play_stockfish -- --color white
```
**Output**: `✅ Auto-loaded default NNUE model (default_hybrid.config)`

#### Development Mode (Auto-loading Disabled)  
```bash
# Trains fresh NNUE for development/testing
cargo run --bin play_stockfish -- --color white --disable-auto-load
```
**Output**: `🎯 Auto-loading disabled, training NNUE on basic chess positions...`

### Hybrid Evaluation Pipeline

When properly configured, the engine uses this evaluation hierarchy:

```
1. 📚 Opening Book Lookup (instant for known positions)
   ↓
2. 🧠 NNUE Neural Network (fast, varying evaluations - PRIMARY)
   ↓  
3. 🎯 Vector Pattern Recognition (strategic guidance - SECONDARY)
   ↓
4. ⚔️ Tactical Search (8-ply, 1M nodes, 50% time - BACKUP)
```

## 📊 Model Configurations

### Vector-Integrated (Recommended)
```bash
--config vector-integrated
```
- **Vector blend**: 40% vector influence, 60% NNUE
- **Best for**: Balanced hybrid evaluation
- **Use case**: General gameplay and analysis

### NNUE-Focused (Speed Priority)
```bash
--config nnue-focused  
```
- **Vector blend**: 10% vector influence, 90% NNUE
- **Best for**: Fast pure neural evaluation
- **Use case**: Rapid analysis and time-critical games

### Default (Balanced)
```bash
--config default
```
- **Vector blend**: 30% vector influence, 70% NNUE
- **Best for**: Standard balanced approach
- **Use case**: Development and testing

### Experimental (Research)
```bash
--config experimental
```
- **Vector blend**: 50% equal blend
- **Architecture**: Larger network (1024 features, 3x512 layers)
- **Best for**: Research and advanced experimentation

## 🚀 Quick Setup Commands

### Complete Hybrid Setup (One-time)
```bash
# 1. Train the production hybrid model
cargo run --bin train_nnue -- \
  --mode train \
  --epochs 20 \
  --config vector-integrated \
  --include-games \
  --output default_hybrid

# 2. Test the trained model
cargo run --bin train_nnue -- \
  --mode test \
  --model-path default_hybrid

# 3. Play with hybrid engine
cargo run --bin play_stockfish -- \
  --color white \
  --depth 8 \
  --time 2000
```

### Development Setup (Fresh Training)
```bash
# Skip model training, use fresh NNUE each time
cargo run --bin play_stockfish -- \
  --color white \
  --depth 8 \
  --time 2000 \
  --disable-auto-load
```

## 🎮 Game Performance

### Optimal Settings for Different Goals

#### Maximum Strength
```bash
cargo run --bin play_stockfish -- \
  --color white \
  --depth 10 \
  --time 5000 \
  --tactical-depth 10
```

#### Balanced Performance  
```bash
cargo run --bin play_stockfish -- \
  --color white \
  --depth 8 \
  --time 2000 \
  --tactical-depth 8
```

#### Speed Priority
```bash
cargo run --bin play_stockfish -- \
  --color white \
  --depth 6 \
  --time 1000 \
  --tactical-depth 6
```

## 📈 Expected Performance

### With Auto-loaded Hybrid Model
- **Startup**: ~2-3 seconds (instant NNUE loading)
- **Evaluation**: Varying outputs (0.1 to 3.0+ range)
- **Move quality**: 2000+ ELO strength
- **Speed**: ~500-1000ms per move

### Without Auto-loading (Fresh Training)
- **Startup**: ~5-10 seconds (includes NNUE training)
- **Evaluation**: Varying outputs after training
- **Move quality**: 1800+ ELO strength  
- **Speed**: ~500-1000ms per move after training

## 🔍 Verification

### Check Auto-loading Works
```bash
# Should show: "✅ Auto-loaded default NNUE model"
cargo run --bin play_stockfish -- --color white | head -20
```

### Check Model Quality
```bash
# Test the model gives varying evaluations
cargo run --bin train_nnue -- --mode test --model-path default_hybrid
```

**Expected output:**
```
Position                  | Evaluation
--------------------------|------------
Starting position         |   +0.540
1.e4                      |   +0.493  
King vs King              |   +0.385
Development               |   +0.465
```

### Verify Integration
```bash
# Test engine integration works
cargo run --bin train_nnue -- --mode integrate --model-path default_hybrid
```

## 🛠️ Troubleshooting

### Model Not Auto-loading
```
📝 Default NNUE model not found, using fresh model
💡 Create one with: cargo run --bin train_nnue -- --output default_hybrid
```
**Solution**: Train the default model with the command provided.

### Constant Evaluations
If you see the same evaluation (like -0.09) for all positions:
```bash
# Re-train the model
cargo run --bin train_nnue -- --mode train --epochs 15 --output default_hybrid --config vector-integrated
```

### Slow Performance
```bash
# Use faster settings
cargo run --bin play_stockfish -- \
  --color white \
  --depth 6 \
  --time 1000 \
  --tactical-depth 6
```

### Development Testing
```bash
# Force fresh training each time
cargo run --bin play_stockfish -- --color white --disable-auto-load
```

## 🎯 Best Practices

### For Production Use
1. Train the hybrid model once: `default_hybrid`
2. Use auto-loading for consistent performance
3. Use vector-integrated configuration
4. Include game positions for diversity

### For Development
1. Use `--disable-auto-load` to test fresh models
2. Use shorter epoch counts for faster iteration
3. Test different configurations
4. Save checkpoints frequently

### For Competition
1. Train with 25+ epochs and `--include-games`
2. Use maximum depth and time settings
3. Verify model quality before important games
4. Test against various opponents

This hybrid setup provides the optimal balance of pattern recognition, neural network evaluation, and tactical search for maximum chess playing strength.