# Lichess Puzzle Database Loader

The `load_lichess_puzzles` binary downloads and processes Lichess puzzles directly into the chess vector engine database.

## Usage

### Basic Usage
```bash
# Load puzzles from a local CSV file
cargo run --bin load_lichess_puzzles -- --input my_puzzles.csv --max-puzzles 10000

# Download and load puzzles automatically
cargo run --bin load_lichess_puzzles -- --download --max-puzzles 10000

# Use custom database and rating range
cargo run --bin load_lichess_puzzles -- --input puzzles.csv --database my_engine.db --min-rating 1200 --max-rating 2200
```

### Command Line Options

- `--input <FILE>`: Path to Lichess puzzle CSV file (default: `lichess_db_puzzle.csv`)
- `--max-puzzles <N>`: Maximum number of puzzles to load (default: 10,000)
- `--database <PATH>`: Database path for persistence (default: `chess_vector_engine.db`)
- `--download`: Download the puzzle database if not found locally
- `--min-rating <N>`: Minimum puzzle rating to include (default: 1000)
- `--max-rating <N>`: Maximum puzzle rating to include (default: 2000)
- `--vector-size <N>`: Vector size for the engine (default: 1024)

### Manual Download

If you prefer to download manually:

```bash
# Download the compressed puzzle database
curl -L -o lichess_db_puzzle.csv.bz2 https://database.lichess.org/lichess_db_puzzle.csv.bz2

# Decompress the file
bunzip2 lichess_db_puzzle.csv.bz2

# Load puzzles into the engine
cargo run --bin load_lichess_puzzles -- --input lichess_db_puzzle.csv --max-puzzles 10000
```

### Features

- **Automatic Download**: Can download the latest Lichess puzzle database automatically
- **Database Persistence**: Saves positions to SQLite database for fast loading
- **Progress Tracking**: Shows loading progress and statistics
- **Rating Filter**: Filter puzzles by rating range
- **Parallel Processing**: Uses all available CPU cores for fast loading
- **Smart Deduplication**: Skips loading if database already contains enough positions
- **Integration**: Uses the engine's existing infrastructure for puzzle processing

### Performance

- Loads 1000+ puzzles per second on modern hardware
- Uses parallel processing with all available CPU cores
- Saves positions to database immediately for persistence
- Memory efficient streaming processing

### Database Integration

The loader:
1. Enables database persistence in the engine
2. Loads existing positions from database if available
3. Processes puzzles using the engine's `load_lichess_puzzles_with_limit` method
4. Saves positions to database automatically via `add_position_with_move`
5. Provides final statistics and similarity search test

### Example Output

```
🔥 Chess Vector Engine - Lichess Puzzle Database Loader
========================================================

Configuration:
  • Input file: sample_puzzles.csv
  • Max puzzles: 10000
  • Database: chess_vector_engine.db
  • Rating range: 1000-2000
  • Vector size: 1024

🚀 Initializing Chess Vector Engine...
✅ Database persistence enabled at: chess_vector_engine.db
📁 Starting with empty database

🧩 Loading Lichess puzzles...
🎉 Lichess puzzle loading complete!
====================================
⏱️  Total time: 2.45s
📊 Final statistics:
  • Total positions: 8642
  • Unique positions: 8642
  • Has move data: true
  • Vector dimension: 1024
  • Database: chess_vector_engine.db
🚀 Loading speed: 3527 positions/second

🔍 Testing similarity search...
  • Found 5 similar positions to starting position
  • Example similarities:
    - Position 1: evaluation = 1.700, similarity = 0.989
    - Position 2: evaluation = 1.988, similarity = 0.974
    - Position 3: evaluation = -0.017, similarity = 0.687

✅ Engine is ready for chess analysis with 8642 positions!
```

## Integration with Chess Vector Engine

The loaded puzzles are immediately available for:
- Similarity search via `find_similar_positions`
- Move recommendations via `recommend_moves`
- Position evaluation via `evaluate_position`
- Tactical analysis via the engine's hybrid evaluation system

The engine combines the loaded puzzle patterns with its NNUE neural network evaluation and tactical search for comprehensive chess analysis.