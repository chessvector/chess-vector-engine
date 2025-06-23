# Chess Vector vs Stockfish - Engine Battle Binary

Play Chess Vector Engine against Stockfish with comprehensive analysis and PGN output.

## 🚀 Features

- **Hybrid AI vs Traditional Engine**: Chess Vector's unique hybrid approach vs Stockfish's pure search
- **Full Hybrid Pipeline**: Opening book → Pattern recognition → Tactical search → Confidence blending  
- **Interactive Engine Battle**: Real-time display of which evaluation method is being used
- **Color Selection**: Choose whether Chess Vector plays White or Black
- **Comprehensive Analysis**: Move-by-move evaluation and centipawn loss calculation
- **Performance Statistics**: Accuracy percentage and total centipawn loss for both engines
- **PGN Output**: Valid PGN format with evaluation annotations
- **Configurable Settings**: Adjust Stockfish depth and time controls

## 📋 Prerequisites

1. **Install Stockfish**: Download from [https://stockfishchess.org/download/](https://stockfishchess.org/download/)
2. **Add to PATH**: Make sure `stockfish` command is available in your terminal
3. **Rust Environment**: Ensure you have Rust installed

## 🎮 Usage

### Basic Usage
```bash
# Chess Vector plays White against Stockfish
cargo run --bin play_stockfish

# Chess Vector plays Black against Stockfish  
cargo run --bin play_stockfish -- --color black

# Adjust Stockfish strength and time control
cargo run --bin play_stockfish -- --depth 15 --time 5000
```

### Command Line Options

| Option | Short | Description | Default | Range |
|--------|-------|-------------|---------|-------|
| `--color` | `-c` | Color for Chess Vector Engine | `white` | `white`, `black` |
| `--depth` | `-d` | Stockfish search depth | `10` | `1-20` |
| `--time` | `-t` | Time per move (milliseconds) | `3000` | `100-60000` |

### Examples

```bash
# Quick game - Chess Vector as Black, fast time control
cargo run --bin play_stockfish -- -c black -d 8 -t 1000

# Strong Stockfish - Deep search with more time
cargo run --bin play_stockfish -- -c white -d 16 -t 10000

# Balanced game - Medium strength
cargo run --bin play_stockfish -- -d 12 -t 3000
```

## 📊 Output Analysis

### Game Progress
- **Real-time Updates**: Move-by-move game progress
- **Position Display**: FEN notation for each position
- **Engine Thinking**: Shows which engine is calculating
- **Move Evaluation**: Confidence scores and evaluations

### Final Statistics
```
📊 Game Statistics:
Chess Vector Engine:
  - Total centipawn loss: 45.2
  - Accuracy: 87.3%

Stockfish:
  - Total centipawn loss: 23.1
  - Accuracy: 94.5%
```

### PGN Output
```pgn
[Event "Chess Vector vs Stockfish"]
[Site "CLI Match"]
[Date "2024.12.22"]
[Round "1"]
[White "Chess Vector Engine"]
[Black "Stockfish"]
[Result "0-1"]

1. e4 { eval: 0.25, loss: 0.0cp } e5 { eval: -0.15, loss: 5.2cp }
2. Nf3 { eval: 0.30, loss: 8.1cp } Nc6 { eval: -0.20, loss: 3.4cp }
...
0-1
```

## 🎯 Understanding the Analysis

### Centipawn Loss
- **0-10cp**: Excellent move
- **10-20cp**: Good move  
- **20-50cp**: Questionable move (?)
- **50+cp**: Blunder (??)

### Accuracy Percentage
- **90%+**: Excellent play
- **80-90%**: Strong play
- **70-80%**: Good play
- **60-70%**: Average play
- **<60%**: Weak play

### Move Annotations
- **No annotation**: Best or near-best move
- **?**: Questionable move (20-50cp loss)
- **??**: Blunder (50+cp loss)

## 🔧 Engine Configuration

### Chess Vector Hybrid Architecture
The engine uses a sophisticated 4-tier evaluation pipeline:

1. **📚 Opening Book Priority**: Instant lookup for 50+ known opening positions
2. **🧠 Pattern Recognition**: Similarity search through learned position patterns  
3. **⚔️ Tactical Search Fallback**: 6+ ply minimax with advanced pruning when patterns are uncertain
4. **🎯 Confidence Blending**: Dynamic weighting between pattern recognition and tactical analysis

**Initialization Output:**
```
🤖 Initializing Chess Vector Engine with hybrid evaluation...
📚 Opening book enabled (50+ openings)
⚔️  Tactical search enabled (6+ ply depth)
🧠 Loading training data for pattern recognition...
✅ Training data loaded - pattern recognition active

🎯 Hybrid Evaluation Pipeline Active:
   1. Opening Book Lookup (instant for known positions)
   2. Pattern Recognition (similarity search in position space)
   3. Tactical Search Fallback (6+ ply minimax with pruning)
   4. Confidence-Based Blending (combines pattern + tactical analysis)
```

**Advanced Features** (Premium+ license):
- **🚀 GPU Acceleration**: 10-100x speedup for similarity search
- **⚡ Advanced Tactical Search**: Enhanced algorithms with deeper analysis
- **🧠 NNUE Networks**: Neural network position evaluation

### Stockfish Configuration
- **Hash Size**: 256MB (automatically set)
- **Threads**: 1 (for fair comparison)
- **Time Control**: Configurable per move
- **Depth Control**: 1-20 ply search depth

## 📁 File Output

### Automatic PGN Saving
Games are automatically saved with timestamped filenames:
```
chess_vector_vs_stockfish_20241222_143052.pgn
```

### File Location
PGN files are saved in the current working directory where you run the command.

## 🐛 Troubleshooting

### Common Issues

#### "Stockfish not found"
```bash
# Check if Stockfish is installed
stockfish --help

# Install on macOS
brew install stockfish

# Install on Ubuntu/Debian
sudo apt-get install stockfish

# Install on Windows
# Download from https://stockfishchess.org/download/
```

#### "Engine has no legal moves"
This can happen in certain endgame positions. The game will terminate gracefully and show the result.

#### Long thinking times
- Reduce `--depth` for faster games
- Reduce `--time` for quicker moves
- Chess Vector may take longer in complex positions

### Performance Tips

1. **Faster Games**: Use `--depth 8 --time 1000`
2. **Stronger Games**: Use `--depth 16 --time 5000`  
3. **Balanced Games**: Use default settings
4. **Analysis Games**: Use `--depth 20 --time 10000`

## 🎯 Sample Game Session

```bash
$ cargo run --bin play_stockfish -- --color white --depth 12 --time 3000

🎮 Starting Chess Vector vs Stockfish match!
♟️  Chess Vector plays as: White
🐟 Stockfish depth: 12, Time per move: 3000ms

🤖 Initializing Chess Vector Engine with hybrid evaluation...
📚 Opening book enabled (50+ openings)
⚔️  Tactical search enabled (6+ ply depth)
🧠 Loading training data for pattern recognition...
✅ Training data loaded - pattern recognition active

🎯 Hybrid Evaluation Pipeline Active:
   1. Opening Book Lookup (instant for known positions)
   2. Pattern Recognition (similarity search in position space)
   3. Tactical Search Fallback (6+ ply minimax with pruning)
   4. Confidence-Based Blending (combines pattern + tactical analysis)

✅ Stockfish engine ready (depth 12)

--- Move 1 ---
Position: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
Turn: White
🤖 Chess Vector is thinking... (using hybrid evaluation)
🎯 Chess Vector plays: e2e4 | Method: 📚 Opening Book | Confidence: 0.95 | Eval: 0.25

--- Move 2 ---
Position: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1
Turn: Black
🐟 Stockfish is thinking...
🎯 Stockfish plays: e7e5 (eval: -0.15)

--- Move 3 ---
Position: rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2
Turn: White
🤖 Chess Vector is thinking... (using hybrid evaluation)
🎯 Chess Vector plays: Nf3 | Method: 📚 Opening Book | Confidence: 0.88 | Eval: 0.30

--- Move 4 ---
Position: rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2
Turn: Black
🐟 Stockfish is thinking...
🎯 Stockfish plays: Nc6 (eval: -0.20)

--- Move 5 ---
Position: r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3
Turn: White
🤖 Chess Vector is thinking... (using hybrid evaluation)
🎯 Chess Vector plays: Bc4 | Method: 🧠 Pattern Recognition + ⚔️ Tactical Search | Confidence: 0.72 | Eval: 0.45

...

🏁 Game finished!
🏆 Black wins by checkmate!

📊 Game Statistics:
Chess Vector Engine:
  - Total centipawn loss: 67.3
  - Accuracy: 82.1%

Stockfish:
  - Total centipawn loss: 31.2
  - Accuracy: 91.7%

💾 Game saved to: chess_vector_vs_stockfish_20241222_143052.pgn
```

## 🚀 Advanced Usage

### Tournament Testing
Run multiple games to get statistical data:
```bash
# Run 10 games with different colors
for i in {1..5}; do
    cargo run --bin play_stockfish -- -c white -d 10 -t 2000
    cargo run --bin play_stockfish -- -c black -d 10 -t 2000
done
```

### Strength Testing
Test against different Stockfish strengths:
```bash
# Weak Stockfish
cargo run --bin play_stockfish -- -d 6 -t 1000

# Medium Stockfish  
cargo run --bin play_stockfish -- -d 10 -t 3000

# Strong Stockfish
cargo run --bin play_stockfish -- -d 16 -t 5000
```

This binary provides a comprehensive platform for testing Chess Vector Engine's performance against one of the world's strongest chess engines!