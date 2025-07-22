/// Basic usage example for Chess Vector Engine
/// 
/// This example demonstrates the fundamental API usage patterns:
/// - Creating an engine instance
/// - Adding positions to the knowledge base
/// - Finding similar positions
/// - Evaluating positions
/// - Using the opening book

use chess_vector_engine::{ChessVectorEngine, PositionEncoder};
use chess::Board;
use std::str::FromStr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Chess Vector Engine - Basic Usage Example");
    println!("==============================================\n");

    // 1. Create a new chess engine with 1024-dimensional vectors
    println!("1️⃣ Creating Chess Vector Engine...");
    let mut engine = ChessVectorEngine::new(1024);
    println!("✅ Engine created with vector dimension: 1024\n");

    // 2. Enable opening book for better opening play
    println!("2️⃣ Enabling opening book...");
    engine.enable_opening_book();
    let opening_stats = engine.opening_book_stats();
    if let Some(stats) = opening_stats {
        println!("✅ Opening book enabled with {} openings\n", stats.total_openings);
    } else {
        println!("✅ Opening book enabled\n");
    }

    // 3. Add some example positions to the knowledge base
    println!("3️⃣ Adding positions to knowledge base...");
    
    let positions = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 0.0, "Starting position"),
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", 0.2, "1.e4"),
        ("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2", 0.0, "Scandinavian Defense"),
        ("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 2 3", 0.1, "Italian Game setup"),
        ("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2", -0.1, "Sicilian Defense"),
    ];

    for (fen, evaluation, description) in &positions {
        if let Ok(board) = Board::from_str(fen) {
            engine.add_position(&board, *evaluation);
            println!("   Added: {} (eval: {:.1})", description, evaluation);
        }
    }
    println!("✅ Added {} positions to knowledge base\n", positions.len());

    // 4. Demonstrate position encoding
    println!("4️⃣ Demonstrating position encoding...");
    let encoder = PositionEncoder::new(1024);
    let starting_board = Board::default();
    let vector = encoder.encode(&starting_board);
    println!("✅ Encoded starting position as {}-dimensional vector", vector.len());
    println!("   First 10 elements: {:?}\n", &vector.as_slice().unwrap_or(&[])[..10.min(vector.len())]);

    // 5. Find similar positions
    println!("5️⃣ Finding similar positions...");
    
    // Test with starting position
    let similar = engine.find_similar_positions(&starting_board, 3);
    println!("Similar positions to starting position:");
    for (i, (board, eval, similarity)) in similar.iter().enumerate() {
        println!("   {}. Evaluation: {:.3}, Similarity: {:.3}", i + 1, eval, similarity);
    }
    
    // Test with 1.e4 position
    if let Ok(e4_board) = Board::from_str("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1") {
        let similar_e4 = engine.find_similar_positions(&e4_board, 3);
        println!("\nSimilar positions to 1.e4:");
        for (i, (board, eval, similarity)) in similar_e4.iter().enumerate() {
            println!("   {}. Evaluation: {:.3}, Similarity: {:.3}", i + 1, eval, similarity);
        }
    }
    println!();

    // 6. Evaluate positions
    println!("6️⃣ Evaluating positions...");
    
    if let Some(eval) = engine.evaluate_position(&starting_board) {
        println!("Starting position evaluation: {:.3}", eval);
    } else {
        println!("Starting position: No evaluation available");
    }

    if let Ok(e4_board) = Board::from_str("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1") {
        if let Some(eval) = engine.evaluate_position(&e4_board) {
            println!("1.e4 position evaluation: {:.3}", eval);
        } else {
            println!("1.e4 position: No evaluation available");
        }
    }
    println!();

    // 7. Get engine statistics
    println!("7️⃣ Engine statistics...");
    let training_stats = engine.training_stats();
    println!("✅ Total positions in knowledge base: {}", training_stats.total_positions);
    println!("✅ Engine has move data: {}", training_stats.has_move_data);
    println!();

    // 8. Demonstrate vector similarity calculation
    println!("8️⃣ Direct vector similarity...");
    if let Ok(sicilian_board) = Board::from_str("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2") {
        let starting_vector = encoder.encode(&starting_board);
        let sicilian_vector = encoder.encode(&sicilian_board);
        
        // Calculate cosine similarity manually
        let dot_product: f32 = starting_vector.iter()
            .zip(sicilian_vector.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm_a: f32 = starting_vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = sicilian_vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let similarity = dot_product / (norm_a * norm_b);
        
        println!("Manual similarity calculation:");
        println!("   Starting position vs Sicilian Defense: {:.3}", similarity);
    }
    println!();

    println!("🎉 Basic usage example completed!");
    println!("💡 Next steps:");
    println!("   - Try adding more positions to improve evaluations");
    println!("   - Enable tactical search for deeper analysis");
    println!("   - Use the UCI engine for chess GUI integration");
    println!("   - Explore advanced features like NNUE and GPU acceleration");

    Ok(())
}