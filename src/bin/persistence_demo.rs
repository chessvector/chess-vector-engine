use chess_vector_engine::ChessVectorEngine;
use chess::Board;
use std::str::FromStr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🗄️  Chess Vector Engine - Persistence Demo");
    println!("============================================");

    // Database file path
    let db_path = "chess_engine.db";

    // Create engine with persistence enabled
    println!("\n📚 Creating engine with persistence...");
    let mut engine = ChessVectorEngine::new_with_persistence(1024, db_path)?;

    // Check if we loaded existing data
    let initial_positions = engine.knowledge_base_size();
    println!("   Initial positions in engine: {}", initial_positions);

    // Add some sample positions if database is empty
    if initial_positions == 0 {
        println!("\n➕ Adding sample positions...");
        
        // Starting position
        let starting_board = Board::default();
        engine.add_position(&starting_board, 0.0);
        
        // Some common openings
        let positions = vec![
            ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", 0.2), // e4
            ("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1", 0.1), // d4
            ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", 0.0), // e4 e5
            ("rnbqkbnr/pppp1ppp/8/4p3/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2", 0.1), // e4 e5 d4
        ];

        for (fen, eval) in positions {
            if let Ok(board) = Board::from_str(fen) {
                engine.add_position(&board, eval);
                println!("   Added position: {} (eval: {})", fen, eval);
            }
        }

        println!("   Total positions added: {}", engine.knowledge_base_size());
    }

    // Enable LSH for larger datasets
    if engine.knowledge_base_size() > 3 {
        println!("\n🔍 Enabling LSH for improved search performance...");
        engine.enable_lsh(4, 8);
        println!("   LSH enabled with 4 tables and 8 hash functions");
    }

    // Test manifold learning if we have enough data
    if engine.knowledge_base_size() >= 4 {
        println!("\n🧠 Testing manifold learning...");
        match engine.enable_manifold_learning(8.0) {
            Ok(_) => {
                println!("   Manifold learning enabled with 8:1 compression ratio");
                
                // Train with minimal epochs for demo
                match engine.train_manifold_learning(10) {
                    Ok(_) => println!("   Manifold learning training completed"),
                    Err(e) => println!("   Manifold training failed: {}", e),
                }
            }
            Err(e) => println!("   Could not enable manifold learning: {}", e),
        }
    }

    // Save current state to database
    println!("\n💾 Saving engine state to database...");
    engine.save_to_database()?;

    // Test database position count
    let db_count = engine.database_position_count()?;
    println!("   Positions in database: {}", db_count);

    // Demonstrate similarity search
    println!("\n🔎 Testing similarity search...");
    let test_position = Board::from_str("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
        .map_err(|e| format!("Invalid FEN: {}", e))?;
    let similar_positions = engine.find_similar_positions(&test_position, 3);
    
    println!("   Query position: {}", test_position);
    println!("   Found {} similar positions:", similar_positions.len());
    for (i, (_, evaluation, similarity)) in similar_positions.iter().enumerate() {
        println!("     {}. Eval: {:.3}, Similarity: {:.3}", i + 1, evaluation, similarity);
    }

    // Test position evaluation
    if let Some(eval) = engine.evaluate_position(&test_position) {
        println!("   Position evaluation: {:.3}", eval);
    }

    // Test move recommendations
    println!("\n♟️  Testing move recommendations...");
    let recommendations = engine.recommend_legal_moves(&test_position, 3);
    println!("   Recommended moves for position:");
    for (i, rec) in recommendations.iter().enumerate() {
        println!("     {}. {} (confidence: {:.3}, from {} similar positions)", 
                i + 1, rec.chess_move, rec.confidence, rec.from_similar_position_count);
    }

    // Create a second engine instance to test loading
    println!("\n🔄 Testing database loading with new engine instance...");
    let mut engine2 = ChessVectorEngine::new(1024);
    engine2.enable_persistence(db_path)?;
    engine2.load_from_database()?;
    
    println!("   Second engine loaded {} positions from database", engine2.knowledge_base_size());
    
    // Verify the data is the same
    let eval1 = engine.evaluate_position(&test_position);
    let eval2 = engine2.evaluate_position(&test_position);
    
    match (eval1, eval2) {
        (Some(e1), Some(e2)) => {
            let diff = (e1 - e2).abs();
            if diff < 0.001 {
                println!("   ✅ Position evaluations match: {:.3}", e1);
            } else {
                println!("   ⚠️  Position evaluations differ: {:.3} vs {:.3}", e1, e2);
            }
        }
        _ => println!("   ⚠️  Could not compare evaluations"),
    }

    // Show engine statistics
    println!("\n📊 Engine Statistics:");
    let stats = engine.training_stats();
    println!("   Total positions: {}", stats.total_positions);
    println!("   Unique positions: {}", stats.unique_positions);
    println!("   LSH enabled: {}", stats.lsh_enabled);
    println!("   Manifold enabled: {}", stats.manifold_enabled);
    println!("   Opening book enabled: {}", stats.opening_book_enabled);
    println!("   Move data entries: {}", stats.move_data_entries);

    if let Some(lsh_stats) = engine.lsh_stats() {
        println!("\n🔍 LSH Statistics:");
        println!("   Number of vectors: {}", lsh_stats.num_vectors);
        println!("   Number of tables: {}", lsh_stats.num_tables);
        println!("   Hash size: {}", lsh_stats.hash_size);
        println!("   Non-empty buckets: {}", lsh_stats.non_empty_buckets);
        println!("   Average bucket size: {:.2}", lsh_stats.avg_bucket_size);
    }

    if let Some(ratio) = engine.manifold_compression_ratio() {
        println!("\n🧠 Manifold Learning:");
        println!("   Compression ratio: {:.1}:1", ratio);
        println!("   Status: {}", if engine.is_manifold_enabled() { "Trained" } else { "Not trained" });
    }

    println!("\n✅ Persistence demo completed successfully!");
    println!("   Database file: {}", db_path);
    println!("   The engine state has been saved and can be loaded on future runs.");

    Ok(())
}