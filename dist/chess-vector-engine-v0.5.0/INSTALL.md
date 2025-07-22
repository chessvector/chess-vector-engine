# Chess Vector Engine - Installation & Setup Guide

**A production-ready chess engine with 1600-1800 ELO validated strength**

## 🚀 Quick Installation

### Option 1: Download Pre-built Binary (Recommended)

1. **Download** the latest release for your system:
   - **Linux**: `chess-vector-engine-linux-x64`
   - **Windows**: `chess-vector-engine-windows-x64.exe` 
   - **macOS**: `chess-vector-engine-macos-x64`

2. **Make executable** (Linux/macOS only):
   ```bash
   chmod +x chess-vector-engine-linux-x64
   ```

3. **Test the engine**:
   ```bash
   ./chess-vector-engine-linux-x64
   ```

### Option 2: Install via Cargo (From Source)

If you have Rust installed:

```bash
# Install the engine
cargo install chess-vector-engine

# The binary will be available as:
chess-vector-engine-uci
```

## 🎯 Chess GUI Setup

### Arena Chess GUI

1. **Download Arena**: [http://www.playwitharena.de/](http://www.playwitharena.de/)
2. **Install Arena** and start the application
3. **Add Engine**:
   - Go to `Engines` → `Install New Engine`
   - Browse to your chess-vector-engine binary
   - Select the file and click "Open"
4. **Configure Engine**:
   - Name: `Chess Vector Engine`
   - Command: Path to your binary
   - Directory: Directory containing the binary
5. **Test**: Go to `Engine` → `Manage` and click "Test" to verify connection

### SCID vs. PC

1. **Download SCID**: [https://scidvspc.sourceforge.net/](https://scidvspc.sourceforge.net/)
2. **Add Engine**:
   - Go to `Tools` → `Analysis Engines`
   - Click "Add Engine"
   - Browse to chess-vector-engine binary
   - Name: `Chess Vector Engine`
3. **Start Analysis**: Use `Tools` → `Start Engine #1` to begin analysis

### ChessBase (Windows)

1. **Open ChessBase** or any ChessBase program
2. **Install Engine**:
   - Go to `Home` → `Engines`
   - Click "Create UCI Engine"
   - Browse to `chess-vector-engine.exe`
   - Name: `Chess Vector Engine`
3. **Test**: The engine should appear in your engines list

### Banksia GUI

1. **Download Banksia**: [https://banksiagui.com/](https://banksiagui.com/)
2. **Add Engine**:
   - Go to `Engines` → `Install`
   - Select "Add from executable file"
   - Browse to chess-vector-engine binary
3. **Play**: Select the engine for analysis or games

## ⚙️ Engine Configuration

### Available UCI Options

| Option | Type | Default | Range | Description |
|--------|------|---------|-------|-------------|
| **Hash** | Spin | 128 | 1-2048 | Hash table size in MB |
| **Threads** | Spin | 1 | 1-64 | Number of search threads |
| **MultiPV** | Spin | 1 | 1-10 | Number of principal variations |
| **Pattern_Weight** | Spin | 60 | 0-100 | Weight for pattern evaluation |
| **Pattern_Confidence_Threshold** | Spin | 75 | 0-100 | Minimum confidence for patterns |
| **Tactical_Depth** | Spin | 3 | 1-10 | Maximum tactical search depth |
| **Enable_LSH** | Check | true | - | Use LSH acceleration |
| **Enable_GPU** | Check | true | - | Use GPU when available |
| **Ponder** | Check | true | - | Think on opponent's time |

### Recommended Settings

#### For Fast Games (Blitz)
- Hash: 64 MB
- Threads: 1
- Tactical_Depth: 2

#### For Standard Games
- Hash: 128 MB (default)
- Threads: 2-4
- Tactical_Depth: 3 (default)

#### For Analysis/Correspondence
- Hash: 512 MB
- Threads: 4-8
- Tactical_Depth: 5
- MultiPV: 3-5

## 📊 Performance Characteristics

### Engine Strength
- **ELO Rating**: 1600-1800 (validated against controlled Stockfish)
- **Tactical Accuracy**: 90.9% on calibrated test positions
- **Material Recognition**: 100% accuracy on test positions
- **Playing Style**: Positional with tactical awareness

### System Requirements
- **Memory**: ~70 MB RAM usage
- **Startup**: Near-instantaneous (1ms)
- **Disk Space**: ~10 MB
- **Dependencies**: None (statically linked)

### Performance Tuning

#### Memory Settings
```
Hash = 128 MB (standard)
Hash = 512 MB (analysis)
Hash = 64 MB (limited RAM)
```

#### CPU Settings  
```
Threads = 1 (single core)
Threads = 4 (quad core)
Threads = 8 (high-end CPU)
```

## 🎮 First Game Setup

### Playing Against the Engine

1. **Start your chess GUI**
2. **Set up a new game**:
   - White: Human Player
   - Black: Chess Vector Engine
   - Time Control: 5+3 or 15+10
3. **Configure engine**:
   - Set Hash to 128 MB
   - Set Threads to 2-4
   - Enable all features
4. **Start playing** and enjoy!

### Analyzing Your Games

1. **Load your PGN file** in the chess GUI
2. **Start engine analysis**:
   - Select Chess Vector Engine
   - Set MultiPV to 3-5 for multiple variations
   - Let it analyze for 10-30 seconds per move
3. **Review suggestions**:
   - Look for tactical patterns
   - Compare with engine's material evaluation
   - Learn from positional assessments

## 🔧 Troubleshooting

### Engine Won't Start
- **Check file permissions**: Make sure binary is executable
- **Verify path**: Ensure GUI has correct path to binary
- **Test manually**: Run engine from command line first

### Engine Responds Slowly
- **Increase Hash**: Set Hash to 256-512 MB for better performance
- **Check Tactical_Depth**: Lower to 2-3 for faster responses
- **Reduce MultiPV**: Set to 1 for single-line analysis

### No Move Suggestions
- **Check UCI protocol**: Engine may need UCI initialization
- **Verify position**: Ensure valid chess position is loaded
- **Reset engine**: Restart engine in GUI

### Memory Issues
- **Lower Hash size**: Reduce to 64 MB for limited systems
- **Disable GPU**: Set Enable_GPU to false if causing issues
- **Single threading**: Set Threads to 1

## 🌟 Advanced Usage

### Command Line Testing

Test engine functionality directly:

```bash
# Start engine and send UCI commands
./chess-vector-engine

# Example UCI session:
uci                          # Initialize engine
isready                      # Check if ready
position startpos            # Set starting position
go movetime 1000            # Search for 1 second
quit                        # Exit engine
```

### Engine vs Engine Games

Set up engine tournaments in your GUI:
1. **Create tournament** with multiple engines
2. **Add Chess Vector Engine** as participant
3. **Configure time controls** (5+3 recommended)
4. **Run tournament** and compare performance

### Position Analysis

For deep analysis:
1. **Set MultiPV to 5** for multiple variations
2. **Increase Hash to 512 MB** for complex positions
3. **Use higher Tactical_Depth (4-5)** for tactical positions
4. **Enable pondering** for continuous analysis

## ❓ FAQ

**Q: How strong is the engine?**
A: 1600-1800 ELO validated strength, suitable for club-level play and analysis.

**Q: Does it support Chess960/Fischer Random?**
A: Yes, the engine handles any valid chess position via UCI protocol.

**Q: Can I use it for analysis?**
A: Absolutely! The engine excels at positional analysis and tactical evaluation.

**Q: Is it completely free?**
A: Yes, 100% open source under MIT/Apache-2.0 license with all features included.

**Q: How does it compare to Stockfish?**
A: Different approach - focuses on pattern recognition and strategic evaluation rather than pure search depth.

**Q: Can I run it on my phone/tablet?**
A: Currently designed for desktop systems with UCI-compatible chess GUIs.

## 📞 Support

- **GitHub Issues**: [Report bugs and feature requests](https://github.com/chessvector/chess-vector-engine/issues)
- **Documentation**: Complete API docs at [docs.rs](https://docs.rs/chess-vector-engine)
- **Community**: Chess programming discussions and help

---

**Ready to play with a unique chess engine?** Start with Arena Chess GUI and our recommended settings for the best experience!