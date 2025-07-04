use chess_vector_engine::{AdvancedSelfLearningSystem, ChessVectorEngine};
use std::env;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Chess Vector Engine - Advanced Self-Learning System");
    println!("======================================================");

    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        return Ok(());
    }

    match args[1].as_str() {
        "train" => run_training(&args[2..])?,
        "fast-train" => run_fast_training(&args[2..])?,
        "progress" => show_progress(&args[2..])?,
        "load-pgn" => load_pgn_games(&args[2..])?,
        "continuous" => run_continuous_learning(&args[2..])?,
        _ => {
            println!("❌ Unknown command: {}", args[1]);
            print_usage();
        }
    }

    Ok(())
}

fn print_usage() {
    println!("Usage: cargo run --bin self_learning <command> [options]");
    println!();
    println!("Commands:");
    println!(
        "  train [iterations] [games_per_iter]   - Run training (default: 1 iteration, 20 games)"
    );
    println!("  fast-train [iterations]               - Quick training with 5 games per iteration");
    println!("  progress [progress_file]              - Show training progress (default: learning_progress.json)");
    println!("  load-pgn <file> [max_games] [max_moves] - Load master games from PGN file");
    println!(
        "  continuous [max_iterations]           - Run continuous learning (default: infinite)"
    );
    println!();
    println!("Examples:");
    println!(
        "  cargo run --bin self_learning fast-train 10   # 10 iterations, 5 games each (~2 min)"
    );
    println!(
        "  cargo run --bin self_learning train 5 10      # 5 iterations, 10 games each (~5 min)"
    );
    println!(
        "  cargo run --bin self_learning train 10        # 10 iterations, 20 games each (~15 min)"
    );
    println!("  cargo run --bin self_learning progress");
    println!("  cargo run --bin self_learning load-pgn master_games.pgn 5000");
    println!("  cargo run --bin self_learning continuous 100");
}

fn run_training(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let iterations = if args.is_empty() {
        1
    } else {
        args[0].parse::<usize>().unwrap_or(1)
    };
    let games_per_iteration = if args.len() > 1 {
        args[1].parse::<usize>().unwrap_or(20)
    } else {
        20
    };
    let progress_file = "learning_progress.json";

    println!(
        "🚀 Starting training for {} iterations ({} games each)...",
        iterations, games_per_iteration
    );

    // Load or create learning system with custom config
    let mut learning_system = if std::path::Path::new(progress_file).exists() {
        let mut system = AdvancedSelfLearningSystem::load_progress(progress_file)?;
        system.games_per_iteration = games_per_iteration; // Update games per iteration
        system
    } else {
        AdvancedSelfLearningSystem::new_with_config(0.6, 500_000, games_per_iteration)
    };

    // Create or load engine
    let mut engine = ChessVectorEngine::new(1024);
    engine.enable_opening_book();

    // Try to load existing training data first
    if Path::new("chess_knowledge.bin").exists() {
        println!("📂 Loading existing training data...");
        if let Err(e) = engine.load_training_data_binary("chess_knowledge.bin") {
            println!("⚠️  Failed to load existing data: {}", e);
        }
    }

    // Initialize with some basic positions if empty
    println!("🎯 Initializing with starting position...");
    let starting_board = chess::Board::default();
    engine.add_position(&starting_board, 0.0);

    println!("🎯 Current engine initialized with positions");

    // Run training iterations
    for i in 1..=iterations {
        println!("\n📚 === Learning Iteration {}/{} ===", i, iterations);

        let stats = learning_system.continuous_learning_iteration(&mut engine)?;

        println!(
            "📊 Stats: {} gen, {} kept, {} high quality",
            stats.positions_generated, stats.positions_kept, stats.high_quality_positions
        );

        // Save progress every iteration
        learning_system.save_progress(progress_file)?;

        // Save updated engine every 5 iterations
        if i % 5 == 0 || i == iterations {
            println!("💾 Saving engine knowledge...");
            engine.save_training_data_binary("chess_knowledge.bin")?;
        }
    }

    println!("\n🎉 Training complete! Use 'progress' command to see detailed report.");
    Ok(())
}

fn run_fast_training(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let iterations = if args.is_empty() {
        5
    } else {
        args[0].parse::<usize>().unwrap_or(5)
    };
    let games_per_iteration = 5; // Fixed fast mode
    let progress_file = "learning_progress.json";

    println!(
        "⚡ Starting FAST training for {} iterations ({} games each)...",
        iterations, games_per_iteration
    );
    println!("🕐 Estimated time: ~{} minutes", (iterations * 2).max(1)); // Rough estimate

    // Load or create learning system with fast config
    let mut learning_system = if std::path::Path::new(progress_file).exists() {
        let mut system = AdvancedSelfLearningSystem::load_progress(progress_file)?;
        system.games_per_iteration = games_per_iteration;
        system
    } else {
        AdvancedSelfLearningSystem::new_with_config(0.5, 500_000, games_per_iteration)
        // Lower quality threshold for speed
    };

    // Create or load engine
    let mut engine = ChessVectorEngine::new(1024);
    engine.enable_opening_book();

    // Try to load existing training data first
    if Path::new("chess_knowledge.bin").exists() {
        println!("📂 Loading existing training data...");
        if let Err(e) = engine.load_training_data_binary("chess_knowledge.bin") {
            println!("⚠️  Failed to load existing data: {}", e);
        }
    }

    // Initialize with some basic positions if empty
    println!("🎯 Initializing with starting position...");
    let starting_board = chess::Board::default();
    engine.add_position(&starting_board, 0.0);

    println!("🎯 Current engine initialized with positions");

    // Run fast training iterations
    for i in 1..=iterations {
        println!("\n📚 === Fast Learning Iteration {}/{} ===", i, iterations);

        let stats = learning_system.continuous_learning_iteration(&mut engine)?;

        println!(
            "📊 Stats: {} gen, {} kept, {} high quality",
            stats.positions_generated, stats.positions_kept, stats.high_quality_positions
        );

        // Save progress every iteration
        learning_system.save_progress(progress_file)?;

        // Save updated engine every 3 iterations or at end
        if i % 3 == 0 || i == iterations {
            println!("💾 Saving engine knowledge...");
            engine.save_training_data_binary("chess_knowledge.bin")?;
        }
    }

    println!("\n🎉 Fast training complete! Use 'progress' command to see detailed report.");
    Ok(())
}

fn show_progress(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let progress_file = if args.is_empty() {
        "learning_progress.json"
    } else {
        &args[0]
    };

    if !Path::new(progress_file).exists() {
        println!("❌ No progress file found at: {}", progress_file);
        println!("💡 Run 'train' command first to start learning.");
        return Ok(());
    }

    let learning_system = AdvancedSelfLearningSystem::load_progress(progress_file)?;
    println!("{}", learning_system.get_progress_report());

    // Show ELO progression graph
    if !learning_system.learning_stats.elo_progression.is_empty() {
        println!("\n📈 ELO Progression:");
        for (iteration, elo) in &learning_system.learning_stats.elo_progression {
            println!("  Iteration {}: {:.0} ELO", iteration, elo);
        }
    }

    Ok(())
}

fn load_pgn_games(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.is_empty() {
        println!("❌ Please provide PGN file path");
        println!(
            "💡 Usage: cargo run --bin self_learning load-pgn <file.pgn> [max_games] [max_moves]"
        );
        println!("💡 Examples:");
        println!("     cargo run --bin self_learning load-pgn games.pgn");
        println!("     cargo run --bin self_learning load-pgn games.pgn 5000");
        println!("     cargo run --bin self_learning load-pgn games.pgn 10000 40");
        return Ok(());
    }

    let pgn_file = &args[0];
    let max_games = if args.len() > 1 {
        args[1].parse::<usize>().unwrap_or(1000)
    } else {
        1000
    };
    let max_moves = if args.len() > 2 {
        args[2].parse::<usize>().unwrap_or(50)
    } else {
        50
    };

    if !Path::new(pgn_file).exists() {
        println!("❌ PGN file not found: {}", pgn_file);
        return Ok(());
    }

    println!("📥 Loading master games from: {}", pgn_file);
    println!(
        "🎯 Limits: {} games max, {} moves per game max",
        max_games, max_moves
    );

    // Create or load engine
    let mut engine = ChessVectorEngine::new(1024);
    engine.enable_opening_book();

    // Initialize with some basic positions if empty
    if engine.database_position_count().unwrap_or(0) == 0 {
        println!("🎯 Initializing with starting position...");
        let starting_board = chess::Board::default();
        engine.add_position(&starting_board, 0.0);
    }

    // Load existing data if available
    if Path::new("chess_knowledge.bin").exists() {
        println!("📂 Loading existing knowledge base...");
        if let Err(e) = engine.load_training_data_binary("chess_knowledge.bin") {
            println!("⚠️  Failed to load existing data: {}", e);
        }
    }

    println!("🎯 Starting PGN loading process...");

    // Use TrainingDataset to load PGN games
    let mut dataset = chess_vector_engine::TrainingDataset::new();

    // Extract positions from PGN with quality filtering
    println!("🔍 Extracting positions from PGN (this may take a while)...");
    dataset.load_from_pgn(pgn_file, Some(max_games), max_moves)?;

    let positions_to_add = dataset.data.len();
    println!("📦 Extracted {} positions from PGN", positions_to_add);

    // Add the positions to the engine
    for training_data in &dataset.data {
        engine.add_position(&training_data.board, training_data.evaluation);
    }

    println!(
        "✅ Successfully loaded {} new positions from master games!",
        positions_to_add
    );
    println!("🎯 Positions added to knowledge base");

    // Save updated engine
    println!("💾 Saving enhanced knowledge base...");
    engine.save_training_data_binary("chess_knowledge.bin")?;

    println!("🎉 Master games loaded! Ready for enhanced learning and Stockfish testing.");

    Ok(())
}

fn run_continuous_learning(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let max_iterations = if args.is_empty() {
        usize::MAX
    } else {
        args[0].parse::<usize>().unwrap_or(usize::MAX)
    };

    let progress_file = "learning_progress.json";

    println!(
        "🔄 Starting continuous learning (max {} iterations)...",
        if max_iterations == usize::MAX {
            "infinite".to_string()
        } else {
            max_iterations.to_string()
        }
    );
    println!("⏹️  Press Ctrl+C to stop and save progress");

    // Load or create learning system
    let mut learning_system = AdvancedSelfLearningSystem::load_progress(progress_file)?;

    // Create or load engine
    let mut engine = ChessVectorEngine::new(1024);
    engine.enable_opening_book();

    // Initialize with some basic positions if empty
    if engine.database_position_count().unwrap_or(0) == 0 {
        println!("🎯 Initializing with starting position...");
        let starting_board = chess::Board::default();
        engine.add_position(&starting_board, 0.0);
    }

    // Load existing data
    if Path::new("chess_knowledge.bin").exists() {
        println!("📂 Loading existing training data...");
        if let Err(e) = engine.load_training_data_binary("chess_knowledge.bin") {
            println!("⚠️  Failed to load existing data: {}", e);
        }
    }

    println!("🎯 Starting with {} positions", 0);

    // Set up Ctrl+C handler for graceful shutdown
    let mut iteration = 0;

    while iteration < max_iterations {
        iteration += 1;

        println!(
            "\n🔄 === Continuous Learning Iteration {} ===",
            learning_system.learning_stats.iterations_completed + 1
        );

        let stats = learning_system.continuous_learning_iteration(&mut engine)?;

        println!(
            "📊 Stats: {} gen, {} kept, {} high quality",
            stats.positions_generated, stats.positions_kept, stats.high_quality_positions
        );

        // Save progress every iteration
        learning_system.save_progress(progress_file)?;

        // Save engine every 10 iterations
        if iteration % 10 == 0 {
            println!("💾 Saving engine knowledge...");
            engine.save_training_data_binary("chess_knowledge.bin")?;

            // Show progress report every 10 iterations
            println!("\n{}", learning_system.get_progress_report());
        }

        // Small delay to allow Ctrl+C
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    println!("🎉 Continuous learning completed!");
    Ok(())
}
