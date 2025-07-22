use chess::Board;
use chess_vector_engine::ChessVectorEngine;
use std::str::FromStr;

/// Simplified demo that clearly demonstrates our unique value proposition:
/// Traditional Chess Engine + Vector Similarity + Strategic Initiative = Unique Strategic Insights
///
/// This demo shows how we COMPLEMENT traditional engines rather than compete with them.
fn main() {
    println!("🎯 Chess Vector Engine - Simplified Approach Demo");
    println!("================================================");
    println!("Our Philosophy: Traditional + Vector Similarity + Strategic Initiative = Unique Insights\n");

    // Create engine and load some strategic knowledge
    let mut engine = ChessVectorEngine::new(1024);
    load_strategic_knowledge(&mut engine);

    // Test positions that showcase our unique approach
    let test_positions = vec![
        ("Opening Choice", "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"),
        ("Strategic Decision", "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
        ("Positional Understanding", "rnbqk2r/ppp2ppp/3bpn2/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQkq - 2 6"),
    ];

    for (description, fen) in test_positions {
        println!("🔍 Analyzing: {}", description);
        println!("Position: {}", fen);
        
        let board = Board::from_str(fen).unwrap();
        
        // Use our simplified evaluation that shows our core value clearly
        let result = engine.evaluate_position_simplified(&board);
        
        println!("  📊 Evaluation Results:");
        println!("    Traditional Component: {:.2}", result.tactical_component);
        println!("    Final Blended Result: {:.2}", result.final_evaluation);
        
        // Show our unique vector similarity insights
        if !result.similarity_insights.similar_positions.is_empty() {
            println!("    🎯 Vector Similarity Insights:");
            println!("      - Found {} similar positions", result.similarity_insights.similar_positions.len());
            println!("      - Suggested evaluation: {:.2}", result.similarity_insights.suggested_evaluation);
            println!("      - Confidence: {:.1}%", result.similarity_insights.confidence * 100.0);
        }
        
        // Show our unique strategic initiative insights
        println!("    ⚡ Strategic Initiative Analysis:");
        println!("      - Initiative advantage: {:.2}", result.strategic_insights.initiative_advantage);
        println!("      - Development: {:.2}", result.strategic_insights.development_advantage);
        println!("      - Center control: {:.2}", result.strategic_insights.center_control);
        println!("      - Recommendation: {}", result.strategic_insights.strategic_recommendation);
        
        // Highlight our unique value
        if result.unique_insights_provided {
            println!("    ✅ UNIQUE VALUE: Our approach provided strategic insights beyond traditional analysis");
        } else {
            println!("    ℹ️  Standard position - traditional analysis sufficient");
        }
        
        println!();
    }
    
    println!("🎯 Summary of Our Approach:");
    println!("==========================");
    println!("✅ We DON'T try to beat traditional engines at their own game");
    println!("✅ We ADD unique strategic insights through vector similarity");
    println!("✅ We ADD proactive evaluation through strategic initiative analysis");
    println!("✅ We COMPLEMENT traditional tactical search with pattern recognition");
    println!("✅ Our value is in providing insights that pure tactical engines cannot offer");
    println!("\n🚀 This is what makes our engine unique!");
}

/// Load strategic positions to demonstrate our pattern recognition capabilities
fn load_strategic_knowledge(engine: &mut ChessVectorEngine) {
    println!("📚 Loading strategic knowledge base...");
    
    let strategic_positions = vec![
        // Opening patterns
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 0.0),
        ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2", 0.0),
        ("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2", -0.1),
        
        // Initiative patterns
        ("r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", 0.3),
        ("rnbqk2r/ppp2ppp/3bpn2/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQkq - 2 6", 0.2),
        
        // Strategic patterns
        ("r1bq1rk1/ppp2ppp/2npbn2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQR1K1 w - - 0 8", 0.1),
        ("r2q1rk1/ppp2ppp/2npbn2/2b1p3/2B1P3/3P1N2/PPP1NPPP/R1BQR1K1 w - - 4 9", 0.4),
        
        // Positional understanding
        ("r1bqr1k1/pp3ppp/2npbn2/2p1p3/2B1P3/2NP1N2/PPP2PPP/R1BQR1K1 w - - 0 10", 0.2),
    ];
    
    for (fen, eval) in strategic_positions {
        if let Ok(board) = Board::from_str(fen) {
            engine.add_position(&board, eval);
        }
    }
    
    println!("✅ Strategic knowledge loaded\n");
}