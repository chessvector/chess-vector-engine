use chess::{Board, ChessMove};
use chess_vector_engine::{ChessVectorEngine, TacticalPuzzleParser, TacticalTrainingData};
use std::str::FromStr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Chess Vector Engine - Incremental Puzzle Training Example");
    println!("========================================================");
    
    // Create engine
    let mut engine = ChessVectorEngine::new(1024);
    engine.enable_opening_book();
    
    // Step 1: Create some sample tactical puzzles
    println!("\n1. Creating sample tactical puzzles:");
    let initial_puzzles = vec![
        TacticalTrainingData {
            position: Board::from_str("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 4").unwrap(),
            solution_move: ChessMove::from_str("f6d5").unwrap(), // Fork attack
            move_theme: "fork".to_string(),
            difficulty: 1.5,
            tactical_value: 3.0,
        },
        TacticalTrainingData {
            position: Board::from_str("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5").unwrap(),
            solution_move: ChessMove::from_str("f3g5").unwrap(), // Pin attack
            move_theme: "pin".to_string(),
            difficulty: 1.8,
            tactical_value: 3.5,
        },
        TacticalTrainingData {
            position: Board::from_str("rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 4").unwrap(),
            solution_move: ChessMove::from_str("c4d5").unwrap(), // Pawn break
            move_theme: "pawn_break".to_string(),
            difficulty: 1.2,
            tactical_value: 2.5,
        },
    ];
    
    println!("Created {} initial tactical puzzles", initial_puzzles.len());
    
    // Step 2: Load initial puzzles into engine
    println!("\n2. Loading initial puzzles into engine:");
    TacticalPuzzleParser::load_into_engine(&initial_puzzles, &mut engine);
    
    let stats = engine.training_stats();
    println!("Engine stats after initial load:");
    println!("  - Total positions: {}", stats.total_positions);
    println!("  - Move data entries: {}", stats.move_data_entries);
    
    // Step 3: Save puzzles for later use
    println!("\n3. Saving puzzles to file:");
    TacticalPuzzleParser::save_tactical_puzzles(&initial_puzzles, "puzzle_progress.json")?;
    
    // Step 4: Save engine state
    println!("\n4. Saving engine training state:");
    engine.save_training_data("engine_progress.json")?;
    
    // Step 5: Simulate adding new puzzles later
    println!("\n5. Creating additional tactical puzzles:");
    let additional_puzzles = vec![
        TacticalTrainingData {
            position: Board::from_str("r2qkb1r/ppp2ppp/2np1n2/4p1B1/2B1P3/3P1N2/PPP2PPP/RN1QK2R w KQkq - 0 6").unwrap(),
            solution_move: ChessMove::from_str("g5f6").unwrap(), // Discovered attack
            move_theme: "discovered_attack".to_string(),
            difficulty: 2.0,
            tactical_value: 4.0,
        },
        TacticalTrainingData {
            position: Board::from_str("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 b kq - 5 5").unwrap(),
            solution_move: ChessMove::from_str("c5d4").unwrap(), // Deflection
            move_theme: "deflection".to_string(),
            difficulty: 2.2,
            tactical_value: 4.2,
        },
    ];
    
    println!("Created {} additional tactical puzzles", additional_puzzles.len());
    
    // Step 6: Add new puzzles incrementally (preserves existing progress)
    println!("\n6. Adding new puzzles incrementally:");
    TacticalPuzzleParser::load_into_engine_incremental(&additional_puzzles, &mut engine);
    
    // Step 7: Save puzzles incrementally (appends to existing file)
    println!("\n7. Saving additional puzzles incrementally:");
    TacticalPuzzleParser::save_tactical_puzzles_incremental(&additional_puzzles, "puzzle_progress.json")?;
    
    // Step 8: Demonstrate loading from saved state
    println!("\n8. Creating fresh engine and loading all progress:");
    let mut fresh_engine = ChessVectorEngine::new(1024);
    fresh_engine.enable_opening_book();
    
    // Load all training progress
    fresh_engine.load_training_data_incremental("engine_progress.json")?;
    
    // Load additional puzzles saved incrementally
    let all_puzzles = TacticalPuzzleParser::load_tactical_puzzles("puzzle_progress.json")?;
    println!("Loaded {} total puzzles from file", all_puzzles.len());
    
    TacticalPuzzleParser::load_into_engine_incremental(&all_puzzles, &mut fresh_engine);
    
    // Step 9: Compare final stats
    let final_stats = fresh_engine.training_stats();
    println!("\n9. Final engine statistics:");
    println!("  - Total positions: {}", final_stats.total_positions);
    println!("  - Unique positions: {}", final_stats.unique_positions);
    println!("  - Move data entries: {}", final_stats.move_data_entries);
    println!("  - Has tactical data: {}", final_stats.has_move_data);
    
    // Step 10: Test tactical recommendations
    println!("\n10. Testing tactical move recommendations:");
    
    // Test on a tactical position
    let test_position = Board::from_str("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 4").unwrap();
    
    let recommendations = fresh_engine.recommend_legal_moves(&test_position, 3);
    println!("Tactical recommendations for test position:");
    for (i, rec) in recommendations.iter().enumerate() {
        println!("  {}. {} (confidence: {:.2}, value: {:.2})", 
                i + 1, rec.chess_move, rec.confidence, rec.average_outcome);
    }
    
    // Step 11: Show puzzle theme distribution
    println!("\n11. Puzzle theme distribution:");
    let mut theme_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for puzzle in &all_puzzles {
        *theme_counts.entry(puzzle.move_theme.clone()).or_insert(0) += 1;
    }
    
    for (theme, count) in &theme_counts {
        println!("  - {}: {} puzzles", theme, count);
    }
    
    // Step 12: Save final combined state
    println!("\n12. Saving final combined state:");
    fresh_engine.save_training_data("final_engine_state.json")?;
    
    println!("\n✅ Incremental puzzle training complete!");
    println!("\nKey features demonstrated:");
    println!("  ✓ Load puzzles incrementally without losing existing data");
    println!("  ✓ Automatic deduplication prevents storing duplicate puzzles"); 
    println!("  ✓ Save/load puzzle collections for reuse");
    println!("  ✓ Combine engine state with tactical knowledge");
    println!("  ✓ Get tactical move recommendations based on learned patterns");
    
    println!("\nFiles created:");
    println!("  - puzzle_progress.json: Incremental puzzle collection");
    println!("  - engine_progress.json: Engine training state");
    println!("  - final_engine_state.json: Combined final state");
    
    // Cleanup
    std::fs::remove_file("puzzle_progress.json").ok();
    std::fs::remove_file("engine_progress.json").ok();
    std::fs::remove_file("final_engine_state.json").ok();
    
    Ok(())
}