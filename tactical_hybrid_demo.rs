use chess_vector_engine::{ChessVectorEngine, HybridConfig, TacticalConfig};
use chess::Board;
use std::str::FromStr;

fn main() {
    println!("=== Chess Vector Engine: Tactical Search & Hybrid Evaluation Demo ===\n");

    // Create a new engine
    let mut engine = ChessVectorEngine::new(1024);
    
    // Enable opening book for comprehensive evaluation
    engine.enable_opening_book();
    
    // Enable tactical search with custom configuration
    let tactical_config = TacticalConfig {
        max_depth: 4,
        max_time_ms: 200,  // 200ms limit
        max_nodes: 20_000,
        quiescence_depth: 3,
        enable_transposition_table: true,
    };
    
    engine.enable_tactical_search(tactical_config);
    println!("✓ Tactical search enabled with custom configuration");

    // Configure hybrid evaluation
    let hybrid_config = HybridConfig {
        pattern_confidence_threshold: 0.75,
        enable_tactical_refinement: true,
        tactical_config: TacticalConfig::default(),
        pattern_weight: 0.6,  // 60% pattern, 40% tactical when blending
        min_similar_positions: 2,
    };
    
    engine.configure_hybrid_evaluation(hybrid_config);
    println!("✓ Hybrid evaluation configured");
    
    // Test positions
    let positions = vec![
        ("Starting position", Board::default()),
        ("After 1.e4", Board::from_str("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1").unwrap()),
        ("Tactical position", Board::from_str("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 4").unwrap()),
        ("Endgame position", Board::from_str("8/2k5/3p4/p2P1p2/P2P1P2/8/2K5/8 w - - 0 1").unwrap()),
    ];

    // Add some training positions to demonstrate pattern recognition
    println!("\n--- Adding training positions ---");
    engine.add_position(&Board::default(), 0.0);
    engine.add_position(&Board::from_str("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1").unwrap(), 0.2);
    engine.add_position(&Board::from_str("rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2").unwrap(), 0.1);
    println!("✓ Added {} positions to knowledge base", engine.knowledge_base_size());

    // Evaluate each position with detailed output
    println!("\n--- Position Evaluations ---");
    for (name, board) in positions {
        println!("\n{}: {}", name, board);
        
        // Check if opening book position
        let is_opening = engine.is_opening_position(&board);
        println!("  Opening book position: {}", is_opening);
        
        // Get evaluation
        if let Some(evaluation) = engine.evaluate_position(&board) {
            println!("  Evaluation: {:.3}", evaluation);
            
            // Show similar positions for context
            let similar = engine.find_similar_positions(&board, 3);
            println!("  Similar positions found: {}", similar.len());
            
            if !similar.is_empty() {
                for (i, (_, eval, similarity)) in similar.iter().enumerate() {
                    println!("    {}. Eval: {:.3}, Similarity: {:.3}", i + 1, eval, similarity);
                }
            }
        } else {
            println!("  No evaluation available");
        }
    }

    // Demonstrate tactical-only evaluation by disabling pattern matching
    println!("\n--- Tactical-Only Evaluation Test ---");
    let mut tactical_only_config = HybridConfig {
        enable_tactical_refinement: true,
        pattern_confidence_threshold: 0.0, // Force tactical evaluation
        pattern_weight: 0.0, // Pure tactical evaluation
        min_similar_positions: 100, // Impossible to meet
        tactical_config: TacticalConfig::default(),
    };
    
    engine.configure_hybrid_evaluation(tactical_only_config);
    
    let tactical_board = Board::from_str("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 4").unwrap();
    if let Some(tactical_eval) = engine.evaluate_position(&tactical_board) {
        println!("Tactical-only evaluation: {:.3}", tactical_eval);
    }

    println!("\n=== Demo Complete ===");
    println!("✓ Tactical search integration working");
    println!("✓ Hybrid evaluation (opening book + pattern + tactical) working");
    println!("✓ Configurable confidence thresholds and blending weights");
}