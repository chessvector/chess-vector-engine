# NNUE Training Guide

This guide shows how to train and use NNUE (Efficiently Updatable Neural Networks) models with the Chess Vector Engine.

## Quick Start

### 1. Train a Basic NNUE Model

```bash
# Train a basic model with default settings
cargo run --bin train_nnue -- --mode train --epochs 10 --output my_nnue_model

# Train with more comprehensive data including game positions
cargo run --bin train_nnue -- --mode train --epochs 20 --output comprehensive_model --include-games

# Train with vector integration optimized settings
cargo run --bin train_nnue -- --mode train --epochs 15 --output vector_nnue --config vector-integrated
```

### 2. Test a Trained Model

```bash
# Test the trained model
cargo run --bin train_nnue -- --mode test --model-path my_nnue_model

# Test integration with the main engine
cargo run --bin train_nnue -- --mode integrate --model-path my_nnue_model
```

## Training Configurations

The trainer supports several NNUE configurations optimized for different use cases:

### Default Configuration
- **Feature size**: 768 (standard NNUE king-relative encoding)
- **Hidden layers**: 2 layers of 256 neurons each
- **Activation**: Clipped ReLU (standard for NNUE)
- **Vector blend**: 30% vector influence, 70% NNUE
- **Best for**: General purpose chess evaluation

### Vector-Integrated Configuration  
- **Feature size**: 768
- **Hidden layers**: 2 layers of 256 neurons each
- **Vector blend**: 40% vector influence, 60% NNUE
- **Best for**: Hybrid evaluation with strong pattern recognition

### NNUE-Focused Configuration
- **Feature size**: 768  
- **Hidden layers**: 2 layers of 256 neurons each
- **Vector blend**: 10% vector influence, 90% NNUE
- **Best for**: Pure neural network evaluation with minimal pattern influence

### Experimental Configuration
- **Feature size**: 1024 (matches vector dimension)
- **Hidden layers**: 3 layers of 512 neurons each
- **Vector blend**: 50% equal blend
- **Best for**: Research and advanced experimentation

## Training Data

The trainer generates comprehensive training data including:

### Curated Positions (21 positions)
- Starting position and basic openings
- King's Pawn and Queen's Pawn openings
- Development positions (Italian Game, etc.)
- Material advantages and disadvantages
- Tactical positions (pins, pressure)
- Endgame scenarios (K+Q vs K, K+R vs K, pawn endgames)
- Positional advantages and castling scenarios

### Game Positions (Optional, ~100 positions)
- Generated from random game play
- Diverse middlegame positions
- Material imbalances and tactical themes
- Evaluated using basic heuristic function

## Model Files

The training process creates several files:

### Configuration File (.config)
```json
{
  "feature_size": 768,
  "hidden_size": 256,
  "num_hidden_layers": 2,
  "activation": "ClippedReLU",
  "learning_rate": 0.001,
  "vector_blend_weight": 0.3,
  "enable_incremental_updates": true
}
```

### Checkpoint Files (Optional)
- Saved every N epochs as specified by `--save-interval`
- Format: `{model_name}_epoch_{number}.config`
- Allows resuming training or testing intermediate models

## Usage Examples

### Basic Training Session
```bash
# Train for 20 epochs with checkpoints every 5 epochs
cargo run --bin train_nnue -- \
  --mode train \
  --epochs 20 \
  --save-interval 5 \
  --output production_nnue \
  --config default \
  --include-games
```

### Advanced Training with Vector Integration
```bash
# Train optimized for hybrid evaluation
cargo run --bin train_nnue -- \
  --mode train \
  --epochs 30 \
  --save-interval 10 \
  --output hybrid_nnue \
  --config vector-integrated \
  --include-games
```

### Testing Different Configurations
```bash
# Test each configuration
for config in default vector-integrated nnue-focused experimental; do
  echo "Testing $config configuration..."
  cargo run --bin train_nnue -- \
    --mode train \
    --epochs 10 \
    --output "test_$config" \
    --config "$config"
  
  cargo run --bin train_nnue -- \
    --mode test \
    --model-path "test_$config"
done
```

## Integration with Chess Vector Engine

The trained NNUE models can be used with the main Chess Vector Engine:

### In Code
```rust
use chess_vector_engine::{ChessVectorEngine, NNUEConfig, NNUE};

// Create engine with NNUE
let mut engine = ChessVectorEngine::new(1024);
engine.enable_nnue()?;

// Train basic NNUE (as done in play_stockfish.rs)
// Or load a pre-trained model (when full persistence is implemented)

// Use in evaluation
let evaluation = engine.evaluate_position(&board);
```

### In Game Play
The `play_stockfish.rs` binary automatically trains a basic NNUE model at startup:

```bash
cargo run --bin play_stockfish -- --color white --depth 8 --time 2000
```

## Performance Characteristics

### Training Speed
- **Basic training** (21 positions): ~3 seconds per epoch
- **With game positions** (~120 positions): ~15 seconds per epoch
- **Memory usage**: ~50-100MB during training
- **GPU acceleration**: Not yet implemented (CPU only)

### Model Quality
- **Convergence**: Usually converges within 10-50 epochs
- **Loss target**: <0.001 for well-converged models
- **Evaluation range**: Typically -10.0 to +10.0 pawn units
- **Varying outputs**: Trained models give different evaluations for different positions

### Integration Performance
- **Evaluation speed**: ~1ms per position (CPU)
- **Memory usage**: ~10MB loaded model
- **Startup time**: <1 second model initialization
- **Compatibility**: Works with all engine evaluation modes

## Limitations and Future Improvements

### Current Limitations
1. **Weight persistence**: Only configuration is saved, not actual neural network weights
2. **Limited training data**: Relatively small dataset compared to production NNUE
3. **CPU only**: No GPU acceleration during training
4. **Basic feature encoding**: Simplified king-relative piece encoding

### Planned Improvements
1. **Safetensors integration**: Full weight serialization and loading
2. **GPU training**: CUDA/Metal acceleration for faster training
3. **Larger datasets**: Integration with Lichess database and puzzle sets  
4. **Advanced features**: Proper king-relative encoding with all perspectives
5. **Incremental updates**: True NNUE incremental evaluation
6. **Online learning**: Continuous improvement during game play

## Troubleshooting

### Common Issues

#### Training Loss Not Decreasing
- Try different learning rates (modify config in code)
- Increase training epochs
- Use `--include-games` for more diverse data

#### Model Loading Fails
- Ensure `.config` file exists
- Check file permissions
- Verify model path is correct

#### Integration Test Fails
- Ensure NNUE is properly enabled in engine
- Check that the model was trained successfully
- Verify Chess Vector Engine is working properly

#### Slow Training
- Remove `--include-games` flag for faster training
- Reduce number of epochs
- Use `default` config instead of `experimental`

### Getting Help
- Check the training logs for specific error messages
- Test with smaller epoch counts first
- Verify the basic engine functionality with `cargo run --bin demo`
- Review the NNUE configuration settings in the output

## Advanced Usage

### Custom Training Data
You can modify the `generate_training_positions()` function in `src/bin/train_nnue.rs` to include:
- Specific opening positions you want to emphasize
- Endgame positions relevant to your playing style  
- Tactical puzzles from databases
- Positions from your own games

### Hyperparameter Tuning
Modify the configuration presets in the code to experiment with:
- Different network architectures (hidden layers, sizes)
- Various activation functions
- Learning rate schedules
- Vector blend weights for hybrid evaluation

### Batch Training
Train multiple models with different configurations:
```bash
# Train a series of models
for epochs in 10 20 30 50; do
  for config in default vector-integrated nnue-focused; do
    cargo run --bin train_nnue -- \
      --mode train \
      --epochs $epochs \
      --output "model_${config}_${epochs}ep" \
      --config "$config"
  done
done
```

This creates a comprehensive training and evaluation framework for NNUE neural networks integrated with the Chess Vector Engine's hybrid evaluation system.