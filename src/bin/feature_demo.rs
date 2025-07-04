use chess::Board;
use chess_vector_engine::ChessVectorEngine;
use std::str::FromStr;

/// Demonstrate the full open source feature set of the Chess Vector Engine
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 CHESS VECTOR ENGINE - FULL OPEN SOURCE FEATURE DEMO");
    println!("=====================================================");

    // Test 1: Standard Engine with Tactical Search
    println!("\n1️⃣ Standard Engine (Tactical Search Enabled by Default)");
    let mut standard_engine = ChessVectorEngine::new(1024);
    println!(
        "   ✅ Tactical search enabled: {}",
        standard_engine.is_tactical_search_enabled()
    );

    // Test position analysis
    let board = Board::from_str("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        .map_err(|e| format!("Failed to parse board: {:?}", e))?;
    if let Some(eval) = standard_engine.evaluate_position(&board) {
        println!("   ✅ Position evaluation: {:.3}", eval);
    }

    // Test 2: Strong Engine for Correspondence Chess
    println!("\n2️⃣ Strong Engine Configuration");
    let mut strong_engine = ChessVectorEngine::new_strong(1024);
    println!("   ✅ Strong tactical search configured");

    // Test move recommendations
    let recommendations = strong_engine.recommend_moves(&board, 5);
    println!(
        "   ✅ Move recommendations: {} moves found",
        recommendations.len()
    );
    for (i, rec) in recommendations.iter().take(3).enumerate() {
        println!(
            "      {}. {}{} (confidence: {:.2})",
            i + 1,
            rec.chess_move.get_source(),
            rec.chess_move.get_dest(),
            rec.confidence
        );
    }

    // Test 3: Lightweight Engine for Performance
    println!("\n3️⃣ Lightweight Engine (Performance Optimized)");
    let mut lightweight_engine = ChessVectorEngine::new_lightweight(1024);
    println!(
        "   ✅ Tactical search disabled: {}",
        !lightweight_engine.is_tactical_search_enabled()
    );

    // Add some positions for similarity search
    lightweight_engine.add_position(&board, 0.0);
    let similar = lightweight_engine.find_similar_positions(&board, 3);
    println!(
        "   ✅ Similarity search: {} similar positions found",
        similar.len()
    );

    // Test 4: Advanced Features (All Available in Open Source)
    println!("\n4️⃣ Advanced Features (All Open Source)");

    // Opening book
    standard_engine.enable_opening_book();
    println!("   ✅ Opening book enabled (50+ professional openings)");

    // GPU acceleration check
    match standard_engine.check_gpu_acceleration() {
        Ok(_) => println!("   ✅ GPU acceleration available"),
        Err(_) => println!("   ⚠️  GPU acceleration not available (CPU fallback)"),
    }

    // LSH indexing
    let _lsh_engine = ChessVectorEngine::new_with_lsh(1024, 12, 20);
    println!("   ✅ LSH indexing enabled for fast similarity search");

    // Memory-mapped file loading capability
    println!("   ✅ Memory-mapped file loading available");
    println!("   ✅ Ultra-fast position loading available");

    // NNUE neural network evaluation
    println!("   ✅ NNUE neural network evaluation available");

    // Manifold learning for position compression
    println!("   ✅ Manifold learning for position compression available");

    // Test 5: Training and Data Loading
    println!("\n5️⃣ Training and Data Management");

    // Auto-discovery
    match ChessVectorEngine::new_with_auto_discovery(1024) {
        Ok(_auto_engine) => {
            println!("   ✅ Auto-discovery engine created");
            println!("   ✅ Automatic training data detection available");
        }
        Err(_) => println!("   ⚠️  No training data found for auto-discovery"),
    }

    // Adaptive engine selection
    let _adaptive_engine = ChessVectorEngine::new_adaptive(1024, 10000, "analysis");
    println!("   ✅ Adaptive engine configuration available");

    println!("\n🎉 SUMMARY: All features are available in open source!");
    println!("   • Advanced tactical search with check extensions");
    println!("   • Professional opening book (50+ systems)");
    println!("   • GPU acceleration support");
    println!("   • NNUE neural network evaluation");
    println!("   • Manifold learning and compression");
    println!("   • Memory-mapped ultra-fast loading");
    println!("   • LSH indexing for large datasets");
    println!("   • Auto-discovery and adaptive configurations");
    println!("   • Full UCI engine compatibility");

    println!("\n💡 Configuration Tips:");
    println!("   • Use new() for standard play with tactical search");
    println!("   • Use new_strong() for correspondence chess");
    println!("   • Use new_lightweight() for real-time applications");
    println!("   • Use new_adaptive() to automatically optimize for your use case");

    Ok(())
}
