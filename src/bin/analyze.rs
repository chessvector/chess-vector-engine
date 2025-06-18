use chess::{Board, MoveGen, Color};
use chess_vector_engine::ChessVectorEngine;
use std::env;
use std::str::FromStr;

fn main() {
    println!("Chess Vector Engine - Position Analyzer");
    println!("=======================================");
    
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        println!("Usage: {} <FEN_position>", args[0]);
        println!("Example: {} \"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1\"", args[0]);
        return;
    }
    
    let fen = &args[1];
    let board = Board::from_str(fen).expect("Valid FEN");
    
    println!("Analyzing position: {}", fen);
    println!("Side to move: {:?}", board.side_to_move());
    
    // Initialize engine with intelligent architecture selection
    let engine = initialize_analysis_engine();
    
    println!("\nKnowledge base loaded with {} positions", engine.knowledge_base_size());
    
    // Analyze the position
    analyze_position(&engine, &board);
}

/// Initialize analysis engine with advanced features based on available data
fn initialize_analysis_engine() -> ChessVectorEngine {
    // Check available training data to determine optimal architecture
    let mut total_positions = 0;
    
    // Count positions in training data
    if let Ok(dataset) = chess_vector_engine::TrainingDataset::load("training_data.json") {
        total_positions += dataset.data.len();
    }
    
    // Estimate positions from opening book
    total_positions += 100; // Approximate opening book size
    
    println!("🔍 Initializing analysis engine for {} estimated positions...", total_positions);
    
    let mut engine = if total_positions > 10000 {
        println!("📊 Large dataset detected, using LSH for fast analysis");
        // LSH optimized for analysis: more hash tables for better recall
        ChessVectorEngine::new_with_lsh(1024, 14, 22)
    } else {
        println!("📊 Standard dataset, using linear search");
        ChessVectorEngine::new(1024)
    };
    
    // Enable opening book
    engine.enable_opening_book();
    
    // Load training data if available
    match chess_vector_engine::TrainingDataset::load("training_data.json") {
        Ok(dataset) => {
            println!("📚 Loading {} positions from training_data.json", dataset.data.len());
            total_positions = dataset.data.len();
            for training_data in dataset.data {
                engine.add_position(&training_data.board, training_data.evaluation);
            }
        }
        Err(_) => {
            println!("📖 No training data found, using opening book only");
        }
    }
    
    // Load basic opening positions
    load_opening_book(&mut engine);
    
    // Enable manifold learning for large analysis datasets
    if total_positions > 15000 {
        println!("🧠 Large analysis dataset, enabling manifold learning for deeper pattern recognition...");
        let _ = engine.enable_manifold_learning(4.0); // 4:1 compression for analysis depth (1024d -> 256d)
        
        println!("🏋️  Training manifold compression for analysis optimization...");
        let _ = engine.train_manifold_learning(12); // More epochs for better analysis quality
        
        println!("✅ Manifold learning enabled - optimized for position analysis");
    }
    
    engine
}

/// Load a basic opening book with evaluations
fn load_opening_book(engine: &mut ChessVectorEngine) {
    let openings = vec![
        // Basic opening principles: control center, develop pieces
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 0.0),
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", 0.2),
        ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2", 0.0),
        ("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2", 0.15),
        ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", 0.1),
        
        // Sicilian variations
        ("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2", -0.1),
        ("rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2", 0.0),
        ("r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", 0.05),
        
        // French Defense
        ("rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", -0.05),
        ("rnbqkbnr/pppp1ppp/4p3/8/4P3/3P4/PPP2PPP/RNBQKBNR b KQkq - 0 2", 0.1),
        
        // Caro-Kann
        ("rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", 0.0),
        ("rnbqkbnr/pp1ppppp/2p5/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq d3 0 2", 0.1),
        
        // King's Indian Setup
        ("rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3", 0.2),
        ("rnbqk2r/ppppppbp/5np1/8/2PP4/2N2N2/PP2PPPP/R1BQKB1R b KQkq - 3 4", 0.15),
        
        // Queen's Gambit
        ("rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2", 0.1),
        ("rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3", 0.05),
        
        // Common tactical patterns
        ("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 3", 0.0),
        ("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3", 0.3),
    ];
    
    for (fen, eval) in openings {
        let board = Board::from_str(fen).expect("Valid FEN");
        engine.add_position(&board, eval);
    }
}

/// Comprehensive position analysis
fn analyze_position(engine: &ChessVectorEngine, board: &Board) {
    println!("\n=== Position Analysis ===");
    
    // Basic position info
    let legal_moves: Vec<_> = MoveGen::new_legal(board).collect();
    println!("Legal moves: {}", legal_moves.len());
    
    if board.checkers().popcnt() > 0 {
        println!("⚠️  King is in check!");
    }
    
    // Vector representation
    let position_vector = engine.encode_position(board);
    println!("Position encoded as {}-dimensional vector", position_vector.len());
    
    // Find similar positions
    println!("\n=== Similar Positions ===");
    let similar_positions = engine.find_similar_positions(board, 5);
    
    if similar_positions.is_empty() {
        println!("No similar positions found in knowledge base");
    } else {
        println!("Found {} similar positions:", similar_positions.len());
        for (i, (_, evaluation, similarity)) in similar_positions.iter().enumerate() {
            let similarity_desc = match similarity {
                s if *s > 0.95 => "Nearly identical",
                s if *s > 0.85 => "Very similar", 
                s if *s > 0.70 => "Similar",
                s if *s > 0.50 => "Somewhat similar",
                _ => "Distantly related",
            };
            
            println!("  {}. Eval: {:+.2}, Similarity: {:.3} ({})", 
                     i + 1, evaluation, similarity, similarity_desc);
        }
    }
    
    // Position evaluation
    println!("\n=== Position Evaluation ===");
    if let Some(predicted_eval) = engine.evaluate_position(board) {
        let eval_desc = match predicted_eval {
            e if e > 1.0 => "Strong advantage for White",
            e if e > 0.5 => "Moderate advantage for White", 
            e if e > 0.1 => "Slight advantage for White",
            e if e > -0.1 => "Roughly equal",
            e if e > -0.5 => "Slight advantage for Black",
            e if e > -1.0 => "Moderate advantage for Black",
            _ => "Strong advantage for Black",
        };
        
        println!("Predicted evaluation: {:+.3} ({})", predicted_eval, eval_desc);
    } else {
        println!("Unable to evaluate - no similar positions in knowledge base");
    }
    
    // Material analysis
    analyze_material(board);
    
    // Positional features
    analyze_positional_features(board);
}

/// Analyze material balance
fn analyze_material(board: &Board) {
    println!("\n=== Material Analysis ===");
    
    use chess::{Piece, Color};
    
    let pieces = [Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen];
    let values = [1, 3, 3, 5, 9];
    
    let mut white_material = 0;
    let mut black_material = 0;
    
    for (piece, value) in pieces.iter().zip(values.iter()) {
        let white_count = (board.pieces(*piece) & board.color_combined(Color::White)).popcnt();
        let black_count = (board.pieces(*piece) & board.color_combined(Color::Black)).popcnt();
        
        white_material += white_count * value;
        black_material += black_count * value;
        
        if white_count != black_count {
            println!("{:?}: White {} vs Black {} (difference: {:+})", 
                     piece, white_count, black_count, white_count as i32 - black_count as i32);
        }
    }
    
    let material_balance = white_material as i32 - black_material as i32;
    println!("Total material: White {} vs Black {} (balance: {:+})", 
             white_material, black_material, material_balance);
}

/// Analyze basic positional features
fn analyze_positional_features(board: &Board) {
    println!("\n=== Positional Features ===");
    
    use chess::Color;
    
    // King safety
    for color in [Color::White, Color::Black] {
        let king_sq = board.king_square(color);
        let file = king_sq.get_file().to_index();
        let rank = king_sq.get_rank().to_index();
        
        let safety_desc = match (color, file, rank) {
            (Color::White, 0..=2, 0..=1) => "Queenside castle",
            (Color::White, 5..=7, 0..=1) => "Kingside castle", 
            (Color::Black, 0..=2, 6..=7) => "Queenside castle",
            (Color::Black, 5..=7, 6..=7) => "Kingside castle",
            (_, 3..=4, _) => "Exposed in center",
            _ => "Custom position",
        };
        
        println!("{:?} king: {} ({})", color, king_sq, safety_desc);
    }
    
    // Development
    let white_developed = count_developed_pieces(board, Color::White);
    let black_developed = count_developed_pieces(board, Color::Black);
    
    println!("Developed pieces: White {} vs Black {}", white_developed, black_developed);
}

/// Count developed pieces (not on starting squares)
fn count_developed_pieces(board: &Board, color: Color) -> u32 {
    use chess::Square;
    
    let mut developed = 0;
    
    // Check if minor pieces have moved from starting squares
    let back_rank = if color == Color::White { 0 } else { 7 };
    let starting_squares = [
        Square::make_square(chess::Rank::from_index(back_rank), chess::File::B),
        Square::make_square(chess::Rank::from_index(back_rank), chess::File::C),
        Square::make_square(chess::Rank::from_index(back_rank), chess::File::F),
        Square::make_square(chess::Rank::from_index(back_rank), chess::File::G),
    ];
    
    for square in starting_squares {
        let sq = square;
        if board.piece_on(sq).is_none() || board.color_on(sq) != Some(color) {
            developed += 1;
        }
    }
    
    developed
}