use chess_vector_engine::{ChessVectorEngine, TacticalPuzzleParser};
use clap::{Arg, Command};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("Tactical Training")
        .version("1.0")
        .about("Train chess vector engine with tactical puzzles from Lichess database")
        .arg(
            Arg::new("puzzles")
                .long("puzzles")
                .value_name("CSV_FILE")
                .help("Path to Lichess puzzle CSV file")
                .required(true)
        )
        .arg(
            Arg::new("output")
                .long("output")
                .short('o')
                .value_name("JSON_FILE")
                .help("Output file for combined training data")
                .default_value("tactical_training_data.json")
        )
        .arg(
            Arg::new("existing")
                .long("existing")
                .value_name("JSON_FILE")
                .help("Existing training data to merge with")
                .default_value("training_data.json")
        )
        .arg(
            Arg::new("max-puzzles")
                .long("max-puzzles")
                .value_name("NUMBER")
                .help("Maximum number of puzzles to process")
                .value_parser(clap::value_parser!(usize))
                .default_value("10000")
        )
        .arg(
            Arg::new("min-rating")
                .long("min-rating")
                .value_name("RATING")
                .help("Minimum puzzle rating")
                .value_parser(clap::value_parser!(u32))
                .default_value("1000")
        )
        .arg(
            Arg::new("max-rating")
                .long("max-rating")
                .value_name("RATING")
                .help("Maximum puzzle rating")
                .value_parser(clap::value_parser!(u32))
                .default_value("2500")
        )
        .arg(
            Arg::new("test-engine")
                .long("test")
                .help("Test the engine with tactical puzzles after training")
                .action(clap::ArgAction::SetTrue)
        )
        .get_matches();

    let puzzle_file = matches.get_one::<String>("puzzles").unwrap();
    let output_file = matches.get_one::<String>("output").unwrap();
    let existing_file = matches.get_one::<String>("existing").unwrap();
    let max_puzzles = *matches.get_one::<usize>("max-puzzles").unwrap();
    let min_rating = *matches.get_one::<u32>("min-rating").unwrap();
    let max_rating = *matches.get_one::<u32>("max-rating").unwrap();
    let test_engine = matches.get_flag("test-engine");

    println!("🧩 Chess Vector Engine - Tactical Training");
    println!("==========================================");
    println!("Puzzle file: {}", puzzle_file);
    println!("Max puzzles: {}", max_puzzles);
    println!("Rating range: {}-{}", min_rating, max_rating);
    println!();

    // Check if puzzle file exists
    if !Path::new(puzzle_file).exists() {
        eprintln!("❌ Error: Puzzle file '{}' not found", puzzle_file);
        eprintln!();
        eprintln!("To download Lichess puzzle database:");
        eprintln!("wget https://database.lichess.org/lichess_db_puzzle.csv.bz2");
        eprintln!("bunzip2 lichess_db_puzzle.csv.bz2");
        return Ok(());
    }

    // Parse tactical puzzles
    println!("📊 Parsing tactical puzzles...");
    let tactical_data = TacticalPuzzleParser::parse_csv(
        puzzle_file,
        Some(max_puzzles),
        Some(min_rating),
        Some(max_rating),
    )?;

    if tactical_data.is_empty() {
        eprintln!("❌ No tactical puzzles found with the given criteria");
        return Ok(());
    }

    println!("✅ Parsed {} tactical puzzles", tactical_data.len());

    // Create engine and load existing data if available
    println!("🚀 Initializing chess vector engine...");
    
    // Use LSH for large datasets to improve performance
    let expected_positions = tactical_data.len() + 60000; // Account for existing data
    let mut engine = if expected_positions > 10000 {
        println!("📊 Large dataset detected ({} positions), enabling LSH for performance", expected_positions);
        // LSH parameters: 12 hash tables, 20 bits per hash for good recall with large datasets
        ChessVectorEngine::new_with_lsh(1024, 12, 20)
    } else {
        println!("📊 Small dataset ({} positions), using linear search", expected_positions);
        ChessVectorEngine::new(1024)
    };
    
    engine.enable_opening_book();

    // Load existing training data incrementally
    let mut total_positions = 0;
    if Path::new(existing_file).exists() {
        match engine.load_training_data_incremental(existing_file) {
            Ok(_) => {
                total_positions = engine.knowledge_base_size();
                println!("📚 Loaded existing training data (total positions: {})", total_positions);
            }
            Err(e) => {
                println!("⚠️  Warning: Could not load existing data from {}: {}", existing_file, e);
            }
        }
    }

    // Load tactical patterns incrementally (preserves existing data)
    println!("🎯 Loading tactical patterns into engine (incremental)...");
    TacticalPuzzleParser::load_into_engine_incremental(&tactical_data, &mut engine);
    let final_stats = engine.training_stats();
    
    println!("✅ Engine loaded with {} total positions", final_stats.total_positions);
    println!("   - Unique positions: {}", final_stats.unique_positions);
    println!("   - Move data entries: {}", final_stats.move_data_entries);
    if let Some(stats) = engine.opening_book_stats() {
        println!("📖 Opening book: {} positions with {} ECO classifications", 
                 stats.total_positions, stats.eco_classifications);
    }

    // Enable manifold learning for very large datasets to compress vectors
    if total_positions > 100000 { // Increased threshold to avoid training on smaller datasets
        println!("🧠 Large dataset detected, enabling manifold learning for vector compression...");
        let _ = engine.enable_manifold_learning(8.0); // 8:1 compression ratio (1024d -> 128d)
        
        println!("🏋️  Training manifold compression model...");
        let epochs = if total_positions > 50000 { 15 } else { 10 }; // More training for larger datasets
        let _ = engine.train_manifold_learning(epochs);
        
        println!("✅ Manifold learning enabled - vectors compressed to 128 dimensions");
    }

    // Test engine if requested
    if test_engine {
        println!("\n🧪 Testing engine with tactical puzzles...");
        test_tactical_performance(&engine, &tactical_data.iter().take(100).cloned().collect::<Vec<_>>());
    }

    // Save combined training data for future use (incremental save)
    println!("\n💾 Saving combined training data to {}...", output_file);
    match engine.save_training_data(output_file) {
        Ok(_) => println!("✅ Successfully saved training data incrementally!"),
        Err(e) => println!("⚠️  Warning: Could not save training data: {}", e),
    }
    
    println!("✅ Training complete! Engine now has tactical knowledge.");
    println!("\n🎮 You can now test the engine with:");
    println!("cargo run --bin play_stockfish");
    println!("cargo run --bin analyze <FEN>");

    Ok(())
}

fn test_tactical_performance(
    engine: &ChessVectorEngine,
    test_puzzles: &[chess_vector_engine::TacticalTrainingData],
) {
    let mut correct = 0;
    let mut total = 0;

    println!("Testing on {} puzzles...", test_puzzles.len());

    for puzzle in test_puzzles {
        let recommendations = engine.recommend_legal_moves(&puzzle.position, 3);
        
        if let Some(top_recommendation) = recommendations.first() {
            if top_recommendation.chess_move == puzzle.solution_move {
                correct += 1;
            }
        }
        total += 1;
    }

    let accuracy = if total > 0 {
        (correct as f32 / total as f32) * 100.0
    } else {
        0.0
    };

    println!("🎯 Tactical accuracy: {}/{} ({:.1}%)", correct, total, accuracy);
}