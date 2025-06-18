use chess::{Board, ChessMove, MoveGen};
use chess_vector_engine::ChessVectorEngine;
use std::str::FromStr;

fn main() {
    println!("Chess Vector Engine - Move Recommendation Demo");
    println!("==============================================");
    
    // Try to use auto-loading to get all available training data including tactical patterns
    let mut engine = match ChessVectorEngine::new_with_auto_load(1024) {
        Ok(engine) => {
            let stats = engine.training_stats();
            println!("🚀 Auto-loaded engine with {} positions!", stats.total_positions);
            if stats.has_move_data {
                println!("🎯 Includes tactical training data with {} move entries", stats.move_data_entries);
                println!("   This will enhance move recommendations with tactical patterns!");
            }
            engine
        }
        Err(_) => {
            println!("🤖 Creating fresh engine (no training data found)");
            ChessVectorEngine::new(1024)
        }
    };
    
    // Load some example positions with moves and outcomes
    load_example_games(&mut engine);
    
    println!("Loaded {} positions into knowledge base", engine.knowledge_base_size());
    
    // Test move recommendations on a specific position
    test_move_recommendations(&engine);
}

/// Load example games with moves and outcomes
fn load_example_games(engine: &mut ChessVectorEngine) {
    // Starting position - e4 is a good opening move
    let starting_board = Board::default();
    let e4_move = ChessMove::from_str("e2e4").expect("Valid move");
    engine.add_position_with_move(&starting_board, 0.0, Some(e4_move), Some(0.2));
    
    // Starting position - d4 is also good
    let d4_move = ChessMove::from_str("d2d4").expect("Valid move");
    engine.add_position_with_move(&starting_board, 0.0, Some(d4_move), Some(0.15));
    
    // Starting position - Nf3 is decent
    let nf3_move = ChessMove::from_str("g1f3").expect("Valid move");
    engine.add_position_with_move(&starting_board, 0.0, Some(nf3_move), Some(0.1));
    
    // After 1.e4 - e5 is a classical response
    let after_e4 = Board::from_str("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
        .expect("Valid FEN");
    let e5_move = ChessMove::from_str("e7e5").expect("Valid move");
    engine.add_position_with_move(&after_e4, 0.2, Some(e5_move), Some(0.0));
    
    // After 1.e4 - c5 (Sicilian) is sharp
    let c5_move = ChessMove::from_str("c7c5").expect("Valid move");
    engine.add_position_with_move(&after_e4, 0.2, Some(c5_move), Some(-0.1));
    
    // Add some tactical positions
    add_tactical_examples(engine);
}

/// Add some tactical position examples
fn add_tactical_examples(engine: &mut ChessVectorEngine) {
    // Example: Scholar's mate threat
    let scholars_setup = Board::from_str("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2")
        .expect("Valid FEN");
    
    // Developing the bishop is good here
    let bc4_move = ChessMove::from_str("f1c4").expect("Valid move");
    engine.add_position_with_move(&scholars_setup, 0.0, Some(bc4_move), Some(0.3));
    
    // Developing the knight is also good
    let nf3_move = ChessMove::from_str("g1f3").expect("Valid move");
    engine.add_position_with_move(&scholars_setup, 0.0, Some(nf3_move), Some(0.25));
}

/// Test move recommendations on a specific position
fn test_move_recommendations(engine: &ChessVectorEngine) {
    println!("\n=== Move Recommendation Test ===");
    
    // Test on starting position
    let test_board = Board::default();
    println!("Position: Starting position");
    println!("FEN: {}", test_board.to_string());
    
    // Get move recommendations
    let recommendations = engine.recommend_legal_moves(&test_board, 5);
    
    if recommendations.is_empty() {
        println!("No move recommendations found.");
    } else {
        println!("\nTop move recommendations:");
        for (i, rec) in recommendations.iter().enumerate() {
            println!("  {}. {} - Confidence: {:.3}, Avg Outcome: {:+.3}, From {} similar positions",
                     i + 1,
                     rec.chess_move,
                     rec.confidence,
                     rec.average_outcome,
                     rec.from_similar_position_count);
        }
    }
    
    // Test on a different position
    println!("\n=== Testing After 1.e4 ===");
    let after_e4 = Board::from_str("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
        .expect("Valid FEN");
    
    let recommendations = engine.recommend_legal_moves(&after_e4, 5);
    
    if recommendations.is_empty() {
        println!("No move recommendations found for this position.");
    } else {
        println!("Top move recommendations for Black after 1.e4:");
        for (i, rec) in recommendations.iter().enumerate() {
            println!("  {}. {} - Confidence: {:.3}, Avg Outcome: {:+.3}, From {} similar positions",
                     i + 1,
                     rec.chess_move,
                     rec.confidence,
                     rec.average_outcome,
                     rec.from_similar_position_count);
        }
    }
    
    // Show all legal moves for comparison
    let legal_moves: Vec<ChessMove> = MoveGen::new_legal(&after_e4).collect();
    println!("\nAll legal moves ({} total): {:?}", legal_moves.len(), legal_moves.iter().take(10).collect::<Vec<_>>());
}