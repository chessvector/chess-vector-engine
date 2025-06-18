use chess::{Board, ChessMove};
use chess_vector_engine::{ChessVectorEngine, TrainingDataset};
use std::str::FromStr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Chess Vector Engine - Incremental Training Example");
    println!("================================================");
    
    // Create engine
    let mut engine = ChessVectorEngine::new(1024);
    
    // Step 1: Initial training with a few positions
    println!("\n1. Initial Training:");
    let initial_positions = vec![
        (Board::default(), 0.0),
        (Board::from_str("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1").unwrap(), 0.2),
        (Board::from_str("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2").unwrap(), 0.1),
    ];
    
    for (board, eval) in initial_positions {
        engine.add_position(&board, eval);
    }
    
    let stats = engine.training_stats();
    println!("Initial positions: {}", stats.total_positions);
    
    // Step 2: Save current training data
    println!("\n2. Saving training data...");
    engine.save_training_data("training_progress.json")?;
    
    // Step 3: Simulate adding new training data from another source
    println!("\n3. Creating additional training data:");
    let mut new_dataset = TrainingDataset::new();
    
    // Add some new positions (simulating data from PGN analysis, etc.)
    let new_positions = vec![
        "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3", // Italian Game
        "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 4", // French Defense
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 4", // Scotch Game
    ];
    
    for (i, fen) in new_positions.iter().enumerate() {
        let board = Board::from_str(fen).unwrap();
        let evaluation = (i as f32) * 0.1 + 0.3; // Some example evaluations
        new_dataset.add_position(board, evaluation, 20, i + 10);
    }
    
    println!("New positions to add: {}", new_dataset.data.len());
    
    // Step 4: Add new data incrementally (preserves existing progress)
    println!("\n4. Adding new data incrementally:");
    engine.train_from_dataset_incremental(&new_dataset);
    
    let stats = engine.training_stats();
    println!("Total positions after incremental training: {}", stats.total_positions);
    
    // Step 5: Save incrementally (appends to existing file)
    println!("\n5. Saving incremental progress:");
    new_dataset.save_incremental("training_progress.json")?;
    
    // Step 6: Demonstrate that we can load and continue from saved state
    println!("\n6. Loading from saved state:");
    let mut fresh_engine = ChessVectorEngine::new(1024);
    fresh_engine.load_training_data_incremental("training_progress.json")?;
    
    let final_stats = fresh_engine.training_stats();
    println!("Loaded engine positions: {}", final_stats.total_positions);
    
    // Step 7: Test the trained engine
    println!("\n7. Testing trained engine:");
    let test_board = Board::default();
    
    if let Some(evaluation) = fresh_engine.evaluate_position(&test_board) {
        println!("Starting position evaluation: {:.2}", evaluation);
    }
    
    let similar = fresh_engine.find_similar_positions(&test_board, 3);
    println!("Found {} similar positions", similar.len());
    
    // Step 8: Add some moves and save again
    println!("\n8. Adding move recommendations:");
    fresh_engine.add_position_with_move(
        &test_board,
        0.0,
        Some(ChessMove::from_str("e2e4").unwrap()),
        Some(0.8) // Good move outcome
    );
    
    let recommendations = fresh_engine.recommend_moves(&test_board, 3);
    println!("Move recommendations:");
    for (i, rec) in recommendations.iter().enumerate() {
        println!("  {}. {} (confidence: {:.2}, outcome: {:.2})", 
                i + 1, rec.chess_move, rec.confidence, rec.average_outcome);
    }
    
    // Final save with move data
    fresh_engine.save_training_data("training_progress.json")?;
    
    let final_stats = fresh_engine.training_stats();
    println!("\nFinal Training Statistics:");
    println!("- Total positions: {}", final_stats.total_positions);
    println!("- Unique positions: {}", final_stats.unique_positions);
    println!("- Has move data: {}", final_stats.has_move_data);
    println!("- Move data entries: {}", final_stats.move_data_entries);
    println!("- LSH enabled: {}", final_stats.lsh_enabled);
    println!("- Manifold enabled: {}", final_stats.manifold_enabled);
    println!("- Opening book enabled: {}", final_stats.opening_book_enabled);
    
    println!("\n✅ Incremental training complete!");
    println!("Your training progress is saved in 'training_progress.json'");
    println!("You can continue adding more data anytime without losing progress.");
    
    // Cleanup
    std::fs::remove_file("training_progress.json").ok();
    
    Ok(())
}