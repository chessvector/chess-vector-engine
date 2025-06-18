use chess_vector_engine::{ChessVectorEngine, TacticalConfig, HybridConfig};
use chess::Board;
use std::str::FromStr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Chess Vector Engine: Hybrid Evaluation Demo ===\n");

    // Initialize engine with tactical search and hybrid evaluation
    let mut engine = ChessVectorEngine::new(1024);
    
    // Enable opening book
    engine.enable_opening_book();
    
    // Configure tactical search
    let tactical_config = TacticalConfig {
        max_depth: 3,
        max_time_ms: 100,
        max_nodes: 10_000,
        quiescence_depth: 2,
        enable_transposition_table: true,
    };
    engine.enable_tactical_search(tactical_config);
    
    // Configure hybrid evaluation
    let hybrid_config = HybridConfig {
        pattern_confidence_threshold: 0.75,
        enable_tactical_refinement: true,
        tactical_config: TacticalConfig::default(),
        pattern_weight: 0.6,
        min_similar_positions: 2,
    };
    engine.configure_hybrid_evaluation(hybrid_config);
    
    println!("✅ Engine initialized with:");
    println!("   - Tactical search: 3-ply depth, 100ms limit");
    println!("   - Hybrid evaluation: 75% confidence threshold");
    println!("   - Pattern weight: 60%, min similar positions: 2");
    println!("   - Opening book enabled");
    
    // Add some training positions
    add_training_positions(&mut engine);
    
    println!("\n📚 Added {} training positions to knowledge base", engine.knowledge_base_size());
    
    // Test different types of positions
    test_position_evaluations(&engine);
    
    // Test move recommendations with hybrid evaluation
    test_move_recommendations(&engine);
    
    // Performance comparison
    test_performance_comparison(&mut engine);
    
    Ok(())
}

fn add_training_positions(engine: &mut ChessVectorEngine) {
    let training_data = vec![
        // Opening positions
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", 0.2),
        ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", 0.0),
        
        // Middlegame positions
        ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 4", 0.3),
        ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 b kq - 0 5", 0.1),
        
        // Tactical positions
        ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPP2PPP/RNB1K1NR b KQkq - 0 4", -1.5), // White threatens checkmate
        ("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQ - 0 6", 0.4),
        
        // Endgame positions
        ("8/8/8/8/8/3k4/3P4/3K4 w - - 0 1", 0.8), // King and pawn vs king
        ("8/8/8/8/8/3k4/8/3K3R w - - 0 1", 5.0), // Rook endgame
    ];
    
    for (fen, eval) in training_data {
        if let Ok(board) = Board::from_str(fen) {
            engine.add_position(&board, eval);
        }
    }
}

fn test_position_evaluations(engine: &ChessVectorEngine) {
    println!("\n🎯 Testing Hybrid Evaluation on Different Position Types:\n");
    
    let test_positions = vec![
        ("Starting position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("London System", "rnbqkb1r/ppp1pppp/5n2/3p4/3P1B2/5N2/PPP1PPPP/RN1QKB1R b KQkq - 0 3"),
        ("Tactical puzzle", "r1bqk2r/pppp1ppp/2n2n2/2b1p2Q/2B1P3/8/PPP2PPP/RNB1K1NR b KQkq - 0 4"),
        ("Queen's Gambit", "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2"),
        ("Complex middlegame", "r2q1rk1/ppp2ppp/2npbn2/2b1p3/2B1P3/2NP1N2/PPP1QPPP/R1B2RK1 w - - 0 8"),
        ("Rook endgame", "8/8/8/8/8/3k4/8/3K3R w - - 0 1"),
    ];
    
    for (name, fen) in test_positions {
        if let Ok(board) = Board::from_str(fen) {
            print!("{:<20} | ", name);
            
            if let Some(evaluation) = engine.evaluate_position(&board) {
                let eval_type = if engine.is_opening_position(&board) {
                    "Opening Book"
                } else if engine.knowledge_base_size() > 0 {
                    "Hybrid (Pattern+Tactical)"
                } else {
                    "Tactical Only"
                };
                println!("Eval: {:+6.2} | Type: {}", evaluation, eval_type);
            } else {
                println!("No evaluation available");
            }
        }
    }
}

fn test_move_recommendations(engine: &ChessVectorEngine) {
    println!("\n♟️  Testing Move Recommendations with Hybrid Evaluation:\n");
    
    let test_positions = vec![
        ("Opening position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("After 1.e4", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
        ("Tactical position", "r1bqk2r/pppp1ppp/2n2n2/2b1p2Q/2B1P3/8/PPP2PPP/RNB1K1NR b KQkq - 0 4"),
    ];
    
    for (name, fen) in test_positions {
        if let Ok(board) = Board::from_str(fen) {
            println!("Position: {}", name);
            
            let recommendations = engine.recommend_legal_moves(&board, 3);
            if recommendations.is_empty() {
                println!("  No move recommendations available\n");
                continue;
            }
            
            for (i, rec) in recommendations.iter().enumerate() {
                println!("  {}. {} (confidence: {:.2}, similar positions: {}, avg outcome: {:.2})", 
                    i + 1, rec.chess_move, rec.confidence, 
                    rec.from_similar_position_count, rec.average_outcome);
            }
            println!();
        }
    }
}

fn test_performance_comparison(engine: &mut ChessVectorEngine) {
    println!("⚡ Performance Comparison:\n");
    
    let test_fen = "r2q1rk1/ppp2ppp/2npbn2/2b1p3/2B1P3/2NP1N2/PPP1QPPP/R1B2RK1 w - - 0 8";
    let board = Board::from_str(test_fen).unwrap();
    
    // Test with hybrid evaluation enabled
    println!("🔄 Hybrid Evaluation (Pattern + Tactical):");
    let start = std::time::Instant::now();
    if let Some(eval) = engine.evaluate_position(&board) {
        let duration = start.elapsed();
        println!("  Evaluation: {:+.2}", eval);
        println!("  Time: {:?}", duration);
    }
    
    // Test with tactical only (disable pattern matching temporarily)
    println!("\n🎯 Tactical-Only Evaluation:");
    let original_config = HybridConfig::default();
    let tactical_only_config = HybridConfig {
        pattern_confidence_threshold: 0.0, // Force tactical search
        enable_tactical_refinement: true,
        pattern_weight: 0.0, // No pattern weight
        min_similar_positions: 1000, // Impossible threshold
        tactical_config: TacticalConfig::default(),
    };
    engine.configure_hybrid_evaluation(tactical_only_config);
    
    let start = std::time::Instant::now();
    if let Some(eval) = engine.evaluate_position(&board) {
        let duration = start.elapsed();
        println!("  Evaluation: {:+.2}", eval);
        println!("  Time: {:?}", duration);
    }
    
    // Restore original configuration
    engine.configure_hybrid_evaluation(original_config);
    
    // GPU acceleration info
    let gpu_accelerator = chess_vector_engine::GPUAccelerator::global();
    println!("\n🖥️  GPU Acceleration Status:");
    println!("  Device: {:?}", gpu_accelerator.device_type());
    println!("  Memory: {}", gpu_accelerator.memory_info());
    
    if gpu_accelerator.is_gpu_enabled() {
        if let Ok(gflops) = gpu_accelerator.benchmark() {
            println!("  Performance: {:.2} GFLOPS", gflops);
        }
    }
    
    println!("\n📊 Engine Statistics:");
    println!("  Knowledge base size: {}", engine.knowledge_base_size());
    println!("  Tactical search enabled: {}", engine.is_tactical_search_enabled());
    println!("  Opening book enabled: {}", engine.is_opening_book_enabled());
    
    let stats = engine.training_stats();
    println!("  Training statistics:");
    println!("    Total positions: {}", stats.total_positions);
    println!("    Has move data: {}", stats.has_move_data);
    println!("    LSH enabled: {}", stats.lsh_enabled);
    println!("    Manifold enabled: {}", stats.manifold_enabled);
}