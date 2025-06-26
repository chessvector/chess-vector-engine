use chess::{Board, ChessMove, MoveGen};
use chess_vector_engine::ChessVectorEngine;
use std::str::FromStr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Debugging chess position that causes panic...");

    // This is the position where the panic occurs (from the game log)
    let fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1";

    println!("Complete");

    // Try to create the board
    let board = match Board::from_str(fen) {
        Ok(board) => {
            println!("✅ Board created successfully");
            board
        }
        Err(e) => {
            println!("Complete");
            return Err(format!("Failed to parse FEN: {e}").into());
        }
    };

    // Try to generate legal moves
    println!("🔍 Generating legal moves...");
    let legal_moves: Vec<ChessMove> = MoveGen::new_legal(&board).collect();
    println!("✅ Generated {} legal moves", legal_moves.len());

    for (i, mv) in legal_moves.iter().take(10).enumerate() {
        println!("Complete");
    }

    // Test the engine with this specific position
    println!("\n🤖 Testing Chess Vector Engine with this position...");
    let mut engine = ChessVectorEngine::new(1024);
    engine.enable_opening_book();

    // Test position evaluation
    println!("🔍 Testing position evaluation...");
    match engine.evaluate_position(&board) {
        Some(eval) => println!("✅ Evaluation: {:.2}", eval),
        None => println!("❌ Evaluation failed: No evaluation available"),
    }

    // Test move recommendations
    println!("🔍 Testing move recommendations...");
    let recommendations = engine.recommend_moves(&board, 3);
    println!("✅ Got {} move recommendations", recommendations.len());

    for (i, rec) in recommendations.iter().enumerate() {
        println!(
            "  {}: {} (confidence: {:.2})",
            i + 1,
            rec.chess_move,
            rec.confidence
        );
    }

    // Test making moves to see if any specific move causes issues
    println!("\n🔍 Testing individual moves...");
    for (i, &mv) in legal_moves.iter().take(5).enumerate() {
        println!("Complete");

        // Try to make the move
        let new_board = board.make_move_new(mv);
        println!("  ✅ Move executed successfully");

        // Try to evaluate the new position
        match engine.evaluate_position(&new_board) {
            Some(eval) => println!("  ✅ New position evaluation: {:.2}", eval),
            None => println!("  ❌ New position evaluation failed: No evaluation available"),
        }
    }

    println!("\n🎉 All tests passed! The position itself is not the issue.");
    println!("💡 The panic might be caused by interaction with Lichess puzzle data or specific move sequences.");

    Ok(())
}
