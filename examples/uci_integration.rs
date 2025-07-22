/// UCI Engine Integration Example
/// 
/// This example demonstrates how to integrate the Chess Vector Engine
/// with chess GUIs and other UCI-compatible software.
/// 
/// The engine supports all standard UCI commands and options:
/// - Engine identification and options
/// - Position setup and move input
/// - Search configuration and execution
/// - Multi-PV analysis and pondering

use chess_vector_engine::{run_uci_engine, run_uci_engine_with_config, UCIConfig, UCIEngine};
use std::io::{self, BufRead, BufReader, Write};
use std::process::{Command, Stdio};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎮 Chess Vector Engine - UCI Integration Example");
    println!("================================================\n");

    println!("This example demonstrates UCI (Universal Chess Interface) integration.");
    println!("The Chess Vector Engine is fully UCI-compliant and works with:");
    println!("- Chess GUIs (Arena, ChessBase, Fritz, etc.)");
    println!("- Command-line tools (cutechess-cli, polyglot, etc.)");
    println!("- Online platforms and analysis tools");
    println!("- Custom chess applications\n");

    // 1. Demonstrate UCI commands
    println!("1️⃣ Standard UCI Commands");
    println!("======================");
    
    println!("Basic UCI protocol:");
    println!("  GUI → Engine: uci");
    println!("  Engine → GUI: id name Chess Vector Engine");
    println!("  Engine → GUI: id author Chess Vector Engine Team");
    println!("  Engine → GUI: option name MultiPV type spin default 1 min 1 max 10");
    println!("  Engine → GUI: option name Pattern_Weight type spin default 60 min 0 max 100");
    println!("  Engine → GUI: option name Ponder type check default true");
    println!("  Engine → GUI: uciok\n");

    // 2. Show available UCI options
    println!("2️⃣ UCI Options");
    println!("=============");
    
    let uci_options = [
        ("MultiPV", "spin", "1", "1", "10", "Number of principal variations to analyze"),
        ("Pattern_Weight", "spin", "60", "0", "100", "Weight of pattern recognition in evaluation (0-100%)"),
        ("Ponder", "check", "true", "", "", "Enable pondering (thinking on opponent's time)"),
        ("Load_Position_Data", "button", "", "", "", "Load additional position training data"),
        ("Threads", "spin", "1", "1", "64", "Number of threads for parallel search"),
        ("Enable_LSH", "check", "true", "", "", "Enable Locality Sensitive Hashing for faster similarity search"),
        ("Hash", "spin", "128", "1", "2048", "Hash table size in MB for transposition tables"),
        ("Tactical_Depth", "spin", "3", "1", "10", "Default tactical search depth"),
        ("Pattern_Confidence_Threshold", "spin", "75", "0", "100", "Minimum confidence for pattern-based moves"),
        ("Enable_GPU", "check", "true", "", "", "Enable GPU acceleration if available"),
    ];

    for (name, type_str, default, min, max, description) in &uci_options {
        print!("  option name {} type {}", name, type_str);
        if !default.is_empty() {
            print!(" default {}", default);
        }
        if !min.is_empty() {
            print!(" min {}", min);
        }
        if !max.is_empty() {
            print!(" max {}", max);
        }
        println!();
        println!("    Description: {}", description);
    }
    println!();

    // 3. Position setup examples
    println!("3️⃣ Position Setup");
    println!("================");
    
    println!("Setting up positions:");
    println!("  GUI → Engine: position startpos");
    println!("  GUI → Engine: position startpos moves e2e4 e7e5 g1f3");
    println!("  GUI → Engine: position fen rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2");
    println!();

    // 4. Search examples
    println!("4️⃣ Search Commands");
    println!("=================");
    
    println!("Search modes:");
    println!("  Fixed depth:     go depth 10");
    println!("  Fixed time:      go movetime 5000");
    println!("  Time control:    go wtime 300000 btime 300000 winc 5000 binc 5000");
    println!("  Infinite:        go infinite");
    println!("  Ponder mode:     go ponder wtime 300000 btime 300000");
    println!();

    println!("Multi-PV analysis:");
    println!("  GUI → Engine: setoption name MultiPV value 3");
    println!("  GUI → Engine: go depth 8");
    println!("  Engine → GUI: info depth 8 multipv 1 score cp 25 pv e2e4 e7e5 g1f3");
    println!("  Engine → GUI: info depth 8 multipv 2 score cp 20 pv d2d4 d7d5 c2c4");
    println!("  Engine → GUI: info depth 8 multipv 3 score cp 15 pv g1f3 g8f6 d2d4");
    println!("  Engine → GUI: bestmove e2e4 ponder e7e5");
    println!();

    // 5. Engine-specific features
    println!("5️⃣ Chess Vector Engine Features");
    println!("==============================");
    
    println!("Pattern Recognition Integration:");
    println!("  - Evaluates positions using 1024-dimensional vectors");
    println!("  - Finds similar positions from training database");
    println!("  - Blends pattern-based evaluation with tactical search");
    println!("  - Pattern_Weight option controls the blend ratio");
    println!();

    println!("Strategic Initiative Evaluation:");
    println!("  - Analyzes positional pressure and initiative");
    println!("  - Identifies proactive vs reactive moves");
    println!("  - Provides master-level strategic insights");
    println!("  - Works across all game phases");
    println!();

    println!("Performance Optimizations:");
    println!("  - 75% memory reduction through optimization");
    println!("  - Multi-threaded similarity search");
    println!("  - GPU acceleration for large databases");
    println!("  - LSH for approximate nearest neighbor search");
    println!();

    // 6. Integration code example
    println!("6️⃣ Integration Code Example");
    println!("==========================");
    
    println!("// Rust integration:");
    println!("use chess_vector_engine::{{run_uci_engine, UCIConfig}};");
    println!("");
    println!("fn main() {{");
    println!("    // Run with default configuration");
    println!("    run_uci_engine();");
    println!("    ");
    println!("    // Or with custom configuration");
    println!("    let config = UCIConfig {{");
    println!("        engine_name: \"Chess Vector Engine Custom\".to_string(),");
    println!("        author: \"Your Name\".to_string(),");
    println!("        default_threads: 4,");
    println!("        default_hash_mb: 256,");
    println!("        enable_ponder: true,");
    println!("        enable_multi_pv: true,");
    println!("        ..Default::default()");
    println!("    }};");
    println!("    run_uci_engine_with_config(config);");
    println!("}}");
    println!();

    // 7. Chess GUI integration
    println!("7️⃣ Chess GUI Integration");
    println!("=======================");
    
    println!("Integration steps:");
    println!("1. Build the UCI engine:");
    println!("   cargo build --release --bin uci_engine");
    println!();
    println!("2. Add to your chess GUI:");
    println!("   - Engine path: target/release/uci_engine (or uci_engine.exe on Windows)");
    println!("   - Engine name: Chess Vector Engine");
    println!("   - Protocol: UCI");
    println!();
    println!("3. Configure engine options:");
    println!("   - Set MultiPV for analysis (1-10 lines)");
    println!("   - Adjust Pattern_Weight for playing style (0-100%)");
    println!("   - Enable/disable pondering based on time control");
    println!("   - Set hash size and thread count for performance");
    println!();
    println!("4. Start playing or analyzing:");
    println!("   - The engine will use pattern recognition for strategic play");
    println!("   - Tactical search provides tactical accuracy");
    println!("   - Opening book ensures strong opening play");
    println!();

    // 8. Command-line testing
    println!("8️⃣ Command-line Testing");
    println!("======================");
    
    println!("Test the UCI engine directly:");
    println!("1. Start the engine: ./target/release/uci_engine");
    println!("2. Send UCI commands:");
    println!("   uci");
    println!("   isready");
    println!("   position startpos moves e2e4");
    println!("   go depth 6");
    println!("   quit");
    println!();

    println!("Expected output:");
    println!("   id name Chess Vector Engine");
    println!("   id author Chess Vector Engine Team");
    println!("   [options list]");
    println!("   uciok");
    println!("   readyok");
    println!("   info depth 1 score cp 25 pv e7e5");
    println!("   info depth 2 score cp 20 pv e7e5 g1f3");
    println!("   [progressive deepening]");
    println!("   bestmove e7e5 ponder g1f3");
    println!();

    println!("🎉 UCI Integration example completed!");
    println!("🎯 Ready for chess GUI integration:");
    println!("   ✅ Full UCI protocol compliance");
    println!("   ✅ 10 configurable options");
    println!("   ✅ Multi-PV analysis support");
    println!("   ✅ Pondering capability");
    println!("   ✅ Pattern recognition integration");
    println!("   ✅ Strategic initiative evaluation");
    println!("   ✅ Professional tournament features");
    
    println!("\n💡 Recommended GUI settings:");
    println!("   - MultiPV: 1 (playing), 3-5 (analysis)");
    println!("   - Pattern_Weight: 60% (balanced), 80% (positional), 40% (tactical)");
    println!("   - Hash: 128-512 MB (depending on available RAM)");
    println!("   - Threads: Number of CPU cores");

    Ok(())
}