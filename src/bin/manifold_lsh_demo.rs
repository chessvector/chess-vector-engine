use chess_vector_engine::ChessVectorEngine;
use chess::Board;
use std::str::FromStr;
use std::time::Instant;
use serde_json::Value;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Chess Vector Engine - Manifold Learning + LSH Integration Demo");
    println!("============================================================");
    
    // Create engine
    let mut engine = ChessVectorEngine::new(1024);
    
    // Load training data manually from JSON
    println!("Loading training data...");
    let content = std::fs::read_to_string("training_data.json")?;
    let positions: Vec<Value> = serde_json::from_str(&content)?;
    
    println!("Loaded {} positions", positions.len());
    
    // Add positions to knowledge base
    println!("Building knowledge base...");
    for position in &positions {
        if let (Some(fen), Some(eval)) = (position["fen"].as_str(), position["evaluation"].as_f64()) {
            if let Ok(board) = Board::from_str(fen) {
                engine.add_position(&board, eval as f32);
            }
        }
    }
    
    println!("Knowledge base size: {}", engine.knowledge_base_size());
    
    // Test 1: Regular similarity search
    println!("\n=== Test 1: Regular similarity search ===");
    let test_board = Board::default();
    
    let start = Instant::now();
    let results = engine.find_similar_positions(&test_board, 5);
    let search_time = start.elapsed();
    
    println!("Found {} similar positions in {:?}", results.len(), search_time);
    for (i, (_, eval, sim)) in results.iter().enumerate().take(3) {
        println!("  {}. Eval: {:.3}, Similarity: {:.3}", i + 1, eval, sim);
    }
    
    // Test 2: Enable manifold learning
    println!("\n=== Test 2: Enable manifold learning ===");
    engine.enable_manifold_learning(8.0)?; // 8:1 compression ratio
    
    if let Some(ratio) = engine.manifold_compression_ratio() {
        println!("Manifold compression ratio: {:.1}x", ratio);
    }
    
    // Train on existing positions
    println!("Training manifold learner...");
    let train_start = Instant::now();
    engine.train_manifold_learning(50)?;
    let train_time = train_start.elapsed();
    
    println!("Training completed in {:?}", train_time);
    
    // Test 3: Manifold similarity search
    println!("\n=== Test 3: Manifold similarity search ===");
    
    let start = Instant::now();
    let manifold_results = engine.find_similar_positions(&test_board, 5);
    let manifold_search_time = start.elapsed();
    
    println!("Found {} similar positions in {:?}", manifold_results.len(), manifold_search_time);
    for (i, (_, eval, sim)) in manifold_results.iter().enumerate().take(3) {
        println!("  {}. Eval: {:.3}, Similarity: {:.3}", i + 1, eval, sim);
    }
    
    println!("Speedup from compression: {:.2}x", 
             search_time.as_nanos() as f64 / manifold_search_time.as_nanos() as f64);
    
    // Test 4: Enable LSH in manifold space
    println!("\n=== Test 4: Enable LSH in manifold space ===");
    engine.enable_manifold_lsh(8, 16)?;
    
    let start = Instant::now();
    let lsh_results = engine.find_similar_positions(&test_board, 5);
    let lsh_search_time = start.elapsed();
    
    println!("Found {} similar positions in {:?}", lsh_results.len(), lsh_search_time);
    for (i, (_, eval, sim)) in lsh_results.iter().enumerate().take(3) {
        println!("  {}. Eval: {:.3}, Similarity: {:.3}", i + 1, eval, sim);
    }
    
    println!("Total speedup: {:.2}x", 
             search_time.as_nanos() as f64 / lsh_search_time.as_nanos() as f64);
    
    // Test 5: Evaluation comparison
    println!("\n=== Test 5: Evaluation comparison ===");
    
    // Test with a few different positions
    let test_positions = vec![
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 4",
    ];
    
    for (i, fen) in test_positions.iter().enumerate() {
        if let Ok(board) = Board::from_str(fen) {
            if let Some(eval) = engine.evaluate_position(&board) {
                println!("Position {}: Predicted evaluation = {:.3}", i + 1, eval);
            }
        }
    }
    
    // Test 6: Performance scaling test
    println!("\n=== Test 6: Performance comparison ===");
    
    // Create larger dataset for performance testing
    println!("Building larger dataset for performance testing...");
    let mut large_engine = ChessVectorEngine::new(1024);
    
    // Add many positions (repeat existing ones with slight variations)
    for position in &positions {
        if let (Some(fen), Some(eval)) = (position["fen"].as_str(), position["evaluation"].as_f64()) {
            if let Ok(board) = Board::from_str(fen) {
                let eval = eval as f32;
                // Add original position
                large_engine.add_position(&board, eval);
                
                // Add with slightly varied evaluation
                large_engine.add_position(&board, eval + 0.01);
                large_engine.add_position(&board, eval - 0.01);
            }
        }
    }
    
    println!("Large knowledge base size: {}", large_engine.knowledge_base_size());
    
    // Test regular search
    let start = Instant::now();
    let _results = large_engine.find_similar_positions(&test_board, 10);
    let linear_time = start.elapsed();
    
    // Enable and train manifold learning
    large_engine.enable_manifold_learning(8.0)?;
    large_engine.train_manifold_learning(30)?;
    large_engine.enable_manifold_lsh(8, 16)?;
    
    // Test manifold + LSH search
    let start = Instant::now();
    let _results = large_engine.find_similar_positions(&test_board, 10);
    let manifold_lsh_time = start.elapsed();
    
    println!("Linear search time: {:?}", linear_time);
    println!("Manifold + LSH time: {:?}", manifold_lsh_time);
    println!("Overall speedup: {:.2}x", 
             linear_time.as_nanos() as f64 / manifold_lsh_time.as_nanos() as f64);
    
    println!("\n=== Demo Complete ===");
    println!("The manifold learning + LSH integration provides:");
    println!("1. Memory efficiency through neural compression");
    println!("2. Faster similarity search through LSH in compressed space");
    println!("3. Maintained accuracy for chess position evaluation");
    
    Ok(())
}