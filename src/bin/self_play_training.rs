use chess_vector_engine::{ChessVectorEngine, SelfPlayConfig};
use clap::{Arg, Command};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("Self-Play Training")
        .version("1.0")
        .about("Train chess vector engine through self-play games")
        .arg(
            Arg::new("games")
                .long("games")
                .short('g')
                .value_name("NUMBER")
                .help("Number of games per iteration")
                .value_parser(clap::value_parser!(usize))
                .default_value("50")
        )
        .arg(
            Arg::new("iterations")
                .long("iterations")
                .short('i')
                .value_name("NUMBER")
                .help("Number of training iterations")
                .value_parser(clap::value_parser!(usize))
                .default_value("10")
        )
        .arg(
            Arg::new("exploration")
                .long("exploration")
                .short('e')
                .value_name("FACTOR")
                .help("Exploration factor (0.0-1.0)")
                .value_parser(clap::value_parser!(f32))
                .default_value("0.3")
        )
        .arg(
            Arg::new("temperature")
                .long("temperature")
                .short('t')
                .value_name("VALUE")
                .help("Temperature for move selection")
                .value_parser(clap::value_parser!(f32))
                .default_value("0.8")
        )
        .arg(
            Arg::new("output")
                .long("output")
                .short('o')
                .value_name("FILE")
                .help("Output file for training data")
                .default_value("self_play_training.bin")
        )
        .arg(
            Arg::new("existing")
                .long("existing")
                .value_name("FILE")
                .help("Load existing training data from file")
        )
        .arg(
            Arg::new("continuous")
                .long("continuous")
                .help("Run continuous self-play training")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("adaptive")
                .long("adaptive")
                .help("Use adaptive difficulty training")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("target-strength")
                .long("target-strength")
                .value_name("STRENGTH")
                .help("Target strength for adaptive training")
                .value_parser(clap::value_parser!(f32))
                .default_value("5.0")
        )
        .arg(
            Arg::new("enable-lsh")
                .long("enable-lsh")
                .help("Enable LSH for large-scale training")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("enable-manifold")
                .long("enable-manifold")
                .help("Enable manifold learning for compression")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("enable-persistence")
                .long("enable-persistence")
                .help("Enable database persistence")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("stockfish-level")
                .long("stockfish-level")
                .help("Train to match Stockfish at medium level")
                .action(clap::ArgAction::SetTrue)
        )
        .get_matches();

    let games_per_iteration = *matches.get_one::<usize>("games").unwrap();
    let iterations = *matches.get_one::<usize>("iterations").unwrap();
    let exploration_factor = *matches.get_one::<f32>("exploration").unwrap();
    let temperature = *matches.get_one::<f32>("temperature").unwrap();
    let output_file = matches.get_one::<String>("output").unwrap();
    let existing_file = matches.get_one::<String>("existing");
    let continuous = matches.get_flag("continuous");
    let adaptive = matches.get_flag("adaptive");
    let target_strength = *matches.get_one::<f32>("target-strength").unwrap();
    let enable_lsh = matches.get_flag("enable-lsh");
    let enable_manifold = matches.get_flag("enable-manifold");
    let enable_persistence = matches.get_flag("enable-persistence");
    let stockfish_level = matches.get_flag("stockfish-level");

    println!("🧠 Chess Vector Engine - Self-Play Training");
    println!("============================================");
    println!("Games per iteration: {}", games_per_iteration);
    println!("Exploration factor: {}", exploration_factor);
    println!("Temperature: {}", temperature);
    if continuous {
        println!("Training mode: Continuous ({} iterations)", iterations);
    } else if adaptive {
        println!("Training mode: Adaptive (target strength: {})", target_strength);
    } else {
        println!("Training mode: Single iteration");
    }
    println!();

    // Initialize engine with advanced features for serious training
    println!("🚀 Initializing chess vector engine...");
    
    // Configure for Stockfish-level training if requested
    let (mut engine, games_per_iter, max_iterations, target_str) = if stockfish_level {
        println!("🎯 Configuring for Stockfish medium-level training...");
        let engine = ChessVectorEngine::new_with_lsh(1024, 16, 24); // Large LSH for serious training
        (engine, 200, 1000, 15.0) // More aggressive training
    } else {
        let engine = if enable_lsh {
            ChessVectorEngine::new_with_lsh(1024, 12, 20)
        } else {
            ChessVectorEngine::new(1024)
        };
        (engine, games_per_iteration, iterations, target_strength)
    };
    
    // Enable advanced features
    engine.enable_opening_book();
    
    if enable_persistence || stockfish_level {
        let db_path = if stockfish_level {
            "stockfish_training.db"
        } else {
            "self_play_training.db"
        };
        
        if let Err(e) = engine.enable_persistence(db_path) {
            println!("⚠️  Could not enable persistence: {}", e);
        } else {
            println!("💾 Database persistence enabled ({})", db_path);
            
            // Load existing data from database if it exists
            let initial_size = engine.knowledge_base_size();
            match engine.load_from_database() {
                Ok(_) => {
                    let loaded_count = engine.knowledge_base_size() - initial_size;
                    if loaded_count > 0 {
                        println!("📚 Loaded {} existing positions from database", loaded_count);
                        println!("🔄 Resuming training from previous state");
                    } else {
                        println!("📝 Starting fresh training (no existing data)");
                    }
                }
                Err(e) => {
                    println!("⚠️  Could not load existing data: {}", e);
                    println!("📝 Starting fresh training");
                }
            }
        }
    }
    
    if enable_manifold || stockfish_level {
        if let Err(e) = engine.enable_manifold_learning(8.0) {
            println!("⚠️  Could not enable manifold learning: {}", e);
        } else {
            println!("🧠 Manifold learning enabled (8:1 compression)");
        }
    }

    // Load existing training data if provided (auto-detect format)
    if let Some(existing_path) = existing_file {
        // Try binary format first, fallback to JSON for backwards compatibility
        let load_result = if existing_path.ends_with(".bin") || existing_path.ends_with(".binary") {
            engine.load_training_data_binary(existing_path)
        } else if existing_path.ends_with(".json") {
            engine.load_training_data_incremental(existing_path)
        } else {
            // Auto-detect: try binary first, then JSON
            engine.load_training_data_binary(existing_path)
                .or_else(|_| engine.load_training_data_incremental(existing_path))
        };
        
        match load_result {
            Ok(_) => {
                println!("📚 Loaded existing training data from {}", existing_path);
                println!("   Starting knowledge base size: {}", engine.knowledge_base_size());
            }
            Err(e) => {
                println!("⚠️  Could not load existing data: {}", e);
                println!("   Starting with empty knowledge base");
            }
        }
    }

    // Configure self-play (use optimized values for Stockfish-level training)
    let config = if stockfish_level {
        SelfPlayConfig {
            games_per_iteration: games_per_iter,
            max_moves_per_game: 300, // Longer games for deeper learning
            exploration_factor: 0.4, // Higher exploration
            min_confidence: 0.05, // Lower threshold for more positions
            use_opening_book: true,
            temperature: 1.0, // Higher temperature for diversity
        }
    } else {
        SelfPlayConfig {
            games_per_iteration: games_per_iter,
            max_moves_per_game: 200,
            exploration_factor,
            min_confidence: 0.1,
            use_opening_book: true,
            temperature,
        }
    };

    println!("🎮 Self-play configuration:");
    println!("   Games per iteration: {}", config.games_per_iteration);
    println!("   Max moves per game: {}", config.max_moves_per_game);
    println!("   Exploration factor: {}", config.exploration_factor);
    println!("   Temperature: {}", config.temperature);
    println!();

    // Run training based on mode
    let total_positions = if stockfish_level {
        println!("🎯 Starting Stockfish-level training (adaptive, continuous, persistent)...");
        println!("   Target: {} strength over {} max iterations", target_str, max_iterations);
        println!("   This will run indefinitely until target is reached or max iterations hit");
        
        // For Stockfish level, always use adaptive continuous training
        engine.adaptive_self_play(config, target_str)?
    } else if adaptive {
        println!("🎯 Starting adaptive self-play training...");
        engine.adaptive_self_play(config, target_str)?
    } else if continuous {
        println!("🔄 Starting continuous self-play training...");
        engine.continuous_self_play(config, max_iterations, Some(output_file))?
    } else {
        println!("🎮 Starting single self-play iteration...");
        engine.self_play_training(config)?
    };

    println!("\n📊 Training Results:");
    println!("   Total new positions: {}", total_positions);
    println!("   Final knowledge base size: {}", engine.knowledge_base_size());

    // Save final training data
    if !continuous { // Continuous mode saves automatically
        println!("\n💾 Saving training data to {}...", output_file);
        match engine.save_training_data_binary(output_file) {
            Ok(_) => println!("✅ Training data saved successfully (binary format)!"),
            Err(e) => println!("⚠️  Failed to save training data: {}", e),
        }
    }

    // Test engine strength with a few sample positions
    println!("\n🧪 Testing engine with sample positions...");
    test_engine_strength(&mut engine);

    println!("\n🎉 Self-play training complete!");
    println!("\n🎮 You can now test the engine with:");
    println!("cargo run --bin analyze <FEN>");
    println!("cargo run --bin play_stockfish");

    Ok(())
}

fn test_engine_strength(engine: &mut ChessVectorEngine) {
    use chess::Board;
    use std::str::FromStr;

    let test_positions = vec![
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Starting position"),
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "1.e4"),
        ("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 2 3", "1.e4 e5 2.Nf3"),
        ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4", "Italian Game"),
    ];

    for (fen, description) in test_positions {
        if let Ok(board) = Board::from_str(fen) {
            if let Some(eval) = engine.evaluate_position(&board) {
                println!("   {}: {:.3}", description, eval);
                
                let recommendations = engine.recommend_legal_moves(&board, 3);
                if !recommendations.is_empty() {
                    print!("     Best moves: ");
                    for (i, rec) in recommendations.iter().take(3).enumerate() {
                        if i > 0 { print!(", "); }
                        print!("{}", rec.chess_move);
                    }
                    println!();
                }
            }
        }
    }
}