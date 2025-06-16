use chess_vector_engine::{TrainingDataset, EngineEvaluator, ChessVectorEngine};
use clap::{Arg, Command};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("Chess Vector Engine Trainer")
        .version("0.1.0")
        .about("Train the chess vector engine with PGN games and Stockfish evaluations")
        .arg(
            Arg::new("pgn")
                .long("pgn")
                .value_name("FILE")
                .help("PGN file to load games from")
                .required(false),
        )
        .arg(
            Arg::new("dataset")
                .long("dataset")
                .value_name("FILE")
                .help("Pre-evaluated dataset file to load")
                .required(false),
        )
        .arg(
            Arg::new("output")
                .long("output")
                .short('o')
                .value_name("FILE")
                .help("Output file to save the trained dataset")
                .default_value("training_data.json"),
        )
        .arg(
            Arg::new("max-games")
                .long("max-games")
                .value_name("N")
                .help("Maximum number of games to process")
                .default_value("100"),
        )
        .arg(
            Arg::new("max-moves-per-game")
                .long("max-moves-per-game")
                .value_name("N")
                .help("Maximum moves to extract per game")
                .default_value("30"),
        )
        .arg(
            Arg::new("stockfish-depth")
                .long("depth")
                .short('d')
                .value_name("N")
                .help("Stockfish search depth for evaluation")
                .default_value("10"),
        )
        .arg(
            Arg::new("evaluate")
                .long("evaluate")
                .help("Evaluate positions with Stockfish (requires stockfish in PATH)")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("test")
                .long("test")
                .help("Test the trained engine against Stockfish")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("enable-lsh")
                .long("enable-lsh")
                .help("Enable LSH indexing for faster similarity search")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("deduplicate")
                .long("deduplicate")
                .help("Remove near-duplicate positions (similarity > 0.95)")
                .action(clap::ArgAction::SetTrue),
        )
        .get_matches();

    let max_games: usize = matches.get_one::<String>("max-games").unwrap().parse()?;
    let max_moves_per_game: usize = matches.get_one::<String>("max-moves-per-game").unwrap().parse()?;
    let stockfish_depth: u8 = matches.get_one::<String>("stockfish-depth").unwrap().parse()?;
    let output_file = matches.get_one::<String>("output").unwrap();
    let enable_lsh = matches.get_flag("enable-lsh");

    println!("🚀 Chess Vector Engine Trainer");
    println!("================================");

    let mut dataset = TrainingDataset::new();

    // Load data from PGN or existing dataset
    if let Some(pgn_file) = matches.get_one::<String>("pgn") {
        println!("📖 Loading games from PGN: {}", pgn_file);
        dataset.load_from_pgn(pgn_file, Some(max_games), max_moves_per_game)?;
        println!("✅ Loaded {} positions", dataset.data.len());

        if matches.get_flag("evaluate") {
            println!("🤖 Evaluating positions with Stockfish (depth {})...", stockfish_depth);
            println!("⚠️  This may take a while for large datasets!");
            dataset.evaluate_with_stockfish(stockfish_depth)?;
            
            // Save the evaluated dataset
            println!("💾 Saving evaluated dataset to: {}", output_file);
            dataset.save(output_file)?;
        }
    } else if let Some(dataset_file) = matches.get_one::<String>("dataset") {
        println!("📁 Loading pre-evaluated dataset: {}", dataset_file);
        dataset = TrainingDataset::load(dataset_file)?;
        println!("✅ Loaded {} positions", dataset.data.len());
    } else {
        println!("❌ Either --pgn or --dataset must be specified");
        std::process::exit(1);
    }

    if dataset.data.is_empty() {
        println!("❌ No training data available");
        std::process::exit(1);
    }

    // Deduplicate positions if requested
    if matches.get_flag("deduplicate") {
        println!("🔄 Removing near-duplicate positions...");
        dataset.deduplicate(0.95);
    }

    // Split into train/test sets
    println!("🔀 Splitting dataset (80% train, 20% test)...");
    let (train_dataset, test_dataset) = dataset.split(0.8);
    println!("📊 Train: {} positions, Test: {} positions", 
             train_dataset.data.len(), test_dataset.data.len());

    // Create and train the engine
    println!("🧠 Creating chess vector engine...");
    let mut engine = if enable_lsh {
        println!("🔍 LSH indexing enabled");
        ChessVectorEngine::new_with_lsh(1024, 16, 16)
    } else {
        ChessVectorEngine::new(1024)
    };

    println!("🎯 Training engine with {} positions...", train_dataset.data.len());
    train_dataset.train_engine(&mut engine);

    // Test the engine if requested
    if matches.get_flag("test") && !test_dataset.data.is_empty() {
        println!("📈 Testing engine accuracy against Stockfish...");
        let evaluator = EngineEvaluator::new(stockfish_depth);
        let mae = evaluator.evaluate_accuracy(&engine, &test_dataset)?;
        
        println!("📊 Engine Performance:");
        println!("   Mean Absolute Error: {:.3} pawns", mae);
        println!("   Knowledge Base Size: {} positions", engine.knowledge_base_size());
        
        if engine.is_lsh_enabled() {
            if let Some(stats) = engine.lsh_stats() {
                println!("   LSH Tables: {}", stats.num_tables);
                println!("   LSH Hash Size: {}", stats.hash_size);
                println!("   LSH Non-empty Buckets: {}", stats.non_empty_buckets);
            }
        }
    }

    println!("✅ Training complete!");
    
    if matches.get_one::<String>("pgn").is_some() && matches.get_flag("evaluate") {
        println!("💡 Next steps:");
        println!("   1. Test the engine: cargo run --bin train -- --dataset {} --test", output_file);
        println!("   2. Analyze positions: cargo run --bin analyze <FEN>");
        println!("   3. Run benchmarks: cargo run --bin benchmark");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    // Integration tests would go here
}