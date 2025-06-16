use chess_vector_engine::ChessVectorEngine;
use chess::Board;
use std::str::FromStr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Chess Vector Engine - Opening Book Integration Demo");
    println!("=================================================");
    
    // Create engine with opening book
    let mut engine = ChessVectorEngine::new(1024);
    engine.enable_opening_book();
    
    // Test positions
    let test_positions = vec![
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Starting Position"),
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "After 1.e4"),
        ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "After 1.e4 e5"),
        ("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "Sicilian Defense"),
        ("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1", "After 1.d4"),
        ("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3", "Ruy Lopez"),
        ("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3", "Italian Game"),
    ];
    
    println!("\n=== Opening Book Statistics ===");
    if let Some(stats) = engine.opening_book_stats() {
        println!("Total positions: {}", stats.total_positions);
        println!("ECO classifications: {}", stats.eco_classifications);
        println!("Average moves per position: {:.1}", stats.avg_moves_per_position);
    }
    
    println!("\n=== Position Analysis ===");
    
    for (fen, description) in test_positions {
        println!("\n--- {} ---", description);
        println!("FEN: {}", fen);
        
        if let Ok(board) = Board::from_str(fen) {
            // Check if in opening book
            if engine.is_opening_position(&board) {
                println!("✓ Found in opening book");
                
                if let Some(entry) = engine.get_opening_entry(&board) {
                    println!("  Opening: {}", entry.name);
                    if let Some(ref eco) = entry.eco_code {
                        println!("  ECO Code: {}", eco);
                    }
                    println!("  Book Evaluation: {:.2}", entry.evaluation);
                    
                    println!("  Best moves:");
                    for (i, (chess_move, strength)) in entry.best_moves.iter().enumerate() {
                        println!("    {}. {} (strength: {:.1})", i + 1, chess_move, strength);
                    }
                }
                
                // Get move recommendations
                let recommendations = engine.recommend_moves(&board, 3);
                println!("  Engine recommendations:");
                for (i, rec) in recommendations.iter().enumerate() {
                    println!("    {}. {} - Confidence: {:.2}, Outcome: {:.2}", 
                             i + 1, rec.chess_move, rec.confidence, rec.average_outcome);
                }
                
                // Get evaluation
                if let Some(eval) = engine.evaluate_position(&board) {
                    println!("  Position evaluation: {:.2}", eval);
                }
            } else {
                println!("✗ Not in opening book");
                
                // Still try to get evaluation from similarity search
                if let Some(eval) = engine.evaluate_position(&board) {
                    println!("  Similarity-based evaluation: {:.2}", eval);
                } else {
                    println!("  No evaluation available");
                }
            }
        }
    }
    
    println!("\n=== Move Sequence Test ===");
    
    // Test a sequence of moves to see opening book integration
    let mut current_board = Board::default();
    let moves = vec!["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"];
    
    println!("Following opening sequence:");
    
    for (move_num, move_str) in moves.iter().enumerate() {
        if move_num > 0 {
            println!("\nAfter {}...", move_str);
        } else {
            println!("\nStarting position:");
        }
        
        // Show current position analysis
        if engine.is_opening_position(&current_board) {
            if let Some(entry) = engine.get_opening_entry(&current_board) {
                println!("  Opening: {} (eval: {:.2})", entry.name, entry.evaluation);
                
                if !entry.best_moves.is_empty() {
                    let (best_move, strength) = &entry.best_moves[0];
                    println!("  Book suggests: {} (strength: {:.1})", best_move, strength);
                }
            }
        }
        
        // Make the move if not the last iteration
        if move_num < moves.len() {
            if let Ok(chess_move) = chess::ChessMove::from_str(move_str) {
                current_board = current_board.make_move_new(chess_move);
            }
        }
    }
    
    println!("\n=== Performance Test ===");
    
    // Test performance difference between opening book lookup and similarity search
    let start_pos = Board::default();
    let iterations = 1000;
    
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        engine.is_opening_position(&start_pos);
    }
    let opening_book_time = start.elapsed();
    
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        engine.find_similar_positions(&start_pos, 5);
    }
    let similarity_time = start.elapsed();
    
    println!("Opening book lookup ({} iterations): {:?}", iterations, opening_book_time);
    println!("Similarity search ({} iterations): {:?}", iterations, similarity_time);
    println!("Speedup: {:.1}x", 
             similarity_time.as_nanos() as f64 / opening_book_time.as_nanos() as f64);
    
    println!("\n=== Demo Complete ===");
    println!("The opening book integration provides:");
    println!("1. Fast lookup for known opening positions");
    println!("2. High-quality move recommendations from chess theory");
    println!("3. Accurate evaluations for opening positions");
    println!("4. Fallback to similarity search for non-opening positions");
    
    Ok(())
}