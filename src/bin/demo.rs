use chess::Board;
use chess_vector_engine::ChessVectorEngine;
use std::str::FromStr;

fn main() {
    println!("Chess Vector Engine Demo");
    println!("========================");

    // Try to create engine with auto-loading first
    let mut engine = match ChessVectorEngine::new_with_auto_load(1024) {
        Ok(engine) => {
            let stats = engine.training_stats();
            println!(
                "🚀 Auto-loaded engine with {} positions!",
                stats.total_positions
            );
            if stats.has_move_data {
                println!("🎯 Includes tactical training data!");
            }
            engine
        }
        Err(_) => {
            println!("🤖 Creating fresh engine (no training data found)");
            ChessVectorEngine::new(1024)
        }
    };

    // Define some sample positions with simple evaluations
    let positions_with_evals = vec![
        // Starting position
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            0.0,
        ),
        // After 1.e4
        (
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            0.2,
        ),
        // After 1.e4 e5
        (
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
            0.0,
        ),
        // After 1.e4 e5 2.Nf3
        (
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
            0.1,
        ),
        // Sicilian Defense
        (
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
            -0.1,
        ),
        // French Defense
        (
            "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            -0.05,
        ),
    ];

    println!(
        "Adding {} positions to the knowledge base...",
        positions_with_evals.len()
    );

    // Add positions to the engine
    for (fen, eval) in &positions_with_evals {
        let board = Board::from_str(fen).expect("Valid FEN");
        engine.add_position(&board, *eval);
        println!("Added position: {} (eval: {})", fen, eval);
    }

    println!("Training data loaded");

    // Test similarity search
    println!("\n=== Similarity Search Demo ===");

    // Test with the starting position
    let test_board = Board::default();
    println!("Searching for positions similar to starting position...");

    let similar_positions = engine.find_similar_positions(&test_board, 3);
    println!("Found {} similar positions:", similar_positions.len());

    for (i, (_vector, evaluation, similarity)) in similar_positions.iter().enumerate() {
        println!(
            "  {}. Evaluation: {:.3}, Similarity: {:.3}",
            i + 1,
            evaluation,
            similarity
        );
    }

    // Test evaluation prediction
    if let Some(predicted_eval) = engine.evaluate_position(&test_board) {
        println!(
            "Predicted evaluation for starting position: {:.3}",
            predicted_eval
        );
    }

    // Test with a different position (after 1.e4)
    println!("\n=== Testing with 1.e4 position ===");
    let e4_board = Board::from_str("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
        .expect("Valid FEN");

    let similar_to_e4 = engine.find_similar_positions(&e4_board, 3);
    println!("Positions similar to 1.e4:");

    for (i, (_, evaluation, similarity)) in similar_to_e4.iter().enumerate() {
        println!(
            "  {}. Evaluation: {:.3}, Similarity: {:.3}",
            i + 1,
            evaluation,
            similarity
        );
    }

    if let Some(predicted_eval) = engine.evaluate_position(&e4_board) {
        println!("Predicted evaluation for 1.e4: {:.3}", predicted_eval);
    }

    // Demonstrate position similarity calculation
    println!("\n=== Position Similarity Demo ===");
    let board1 = Board::default();
    let board2 = Board::from_str("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
        .expect("Valid FEN");
    let board3 = Board::from_str("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2")
        .expect("Valid FEN");

    let sim_1_2 = engine.calculate_similarity(&board1, &board2);
    let sim_1_3 = engine.calculate_similarity(&board1, &board3);
    let sim_2_3 = engine.calculate_similarity(&board2, &board3);

    println!(
        "Similarity between starting position and 1.e4: {:.3}",
        sim_1_2
    );
    println!(
        "Similarity between starting position and Sicilian: {:.3}",
        sim_1_3
    );
    println!("Similarity between 1.e4 and Sicilian: {:.3}", sim_2_3);

    // Show vector encoding example
    println!("\n=== Vector Encoding Demo ===");
    let start_vector = engine.encode_position(&board1);
    println!(
        "Starting position vector (first 10 elements): {:?}",
        &start_vector.as_slice().unwrap()[0..10]
    );

    let e4_vector = engine.encode_position(&board2);
    println!(
        "1.e4 position vector (first 10 elements): {:?}",
        &e4_vector.as_slice().unwrap()[0..10]
    );

    println!("\n=== Demo Complete ===");
    println!("This demonstrates the basic vector similarity approach!");
    println!("Next steps would be:");
    println!("1. Implement manifold learning for better compression");
    println!("2. Add more sophisticated similarity search (LSH, approximate NN)");
    println!("3. Train on larger datasets of positions with known evaluations");
    println!("4. Implement move recommendation based on similar positions");
}
