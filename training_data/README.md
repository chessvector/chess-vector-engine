# Starter Training Dataset

This directory contains a curated starter dataset for the Chess Vector Engine.

## Files

- `starter_dataset.json` - Human-readable JSON format (47 positions)

## Content

The starter dataset includes:

### Opening Positions (~30 positions)
- King's Pawn openings (e4)
- Queen's Pawn openings (d4)  
- Sicilian Defense variations
- French Defense
- Caro-Kann Defense
- English Opening
- Queen's Gambit

### Middlegame Patterns (~12 positions)
- Italian Game developments
- Spanish Opening (Ruy Lopez)
- Tactical motifs
- Development principles
- Pawn structure examples

### Endgame Fundamentals (~5 positions)
- Basic king and pawn endings
- Rook endgames
- Queen endgames
- Checkmate patterns

## Usage

This dataset provides immediate chess knowledge for the open source version of Chess Vector Engine. Premium and Enterprise tiers include much larger, professionally curated datasets.

### For Open Source Users
The starter dataset gives you:
- ✅ **Immediate functionality** - Engine works out of the box
- ✅ **Basic chess knowledge** - Understands common openings and patterns
- ✅ **Foundation for learning** - Good starting point for your own training

### Premium vs Enterprise Datasets
- **Premium** (~30,000 positions): Master-level games, advanced patterns, optimized formats
- **Enterprise** (500,000+ positions): Comprehensive databases, custom datasets, ultra-fast formats

## Integration

The engine automatically loads this starter dataset when using:

```rust
use chess_vector_engine::ChessVectorEngine;

// Automatically includes starter dataset
let mut engine = ChessVectorEngine::new_with_instant_load(1024)?;
```

## Data Format

Each position includes:
- **FEN notation** - Standard chess position format
- **Evaluation** - Centipawn evaluation (-8.0 to +8.0)
- **Game context** - Opening name, game result
- **Metadata** - Depth, game ID for tracking

## License

This starter dataset is released under the same MIT/Apache-2.0 license as the open source Chess Vector Engine.

---

**Want more chess knowledge?** Consider upgrading to Premium for 30k positions or Enterprise for 500k+ positions at [chessvector.ai](https://chessvector.ai)