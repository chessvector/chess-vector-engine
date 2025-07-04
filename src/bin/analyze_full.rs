use chess::{Board, ChessMove, Color, Game, MoveGen};
use chess_vector_engine::ChessVectorEngine;
use chess_vector_engine::HybridConfig;
use chrono::{DateTime, Utc};
use std::env;
use std::str::FromStr;

#[derive(Debug, Clone)]
struct MoveAnalysis {
    #[allow(dead_code)]
    chess_move: ChessMove,
    #[allow(dead_code)]
    evaluation_before: f32,
    #[allow(dead_code)]
    evaluation_after: f32,
    #[allow(dead_code)]
    centipawn_loss: f32,
    #[allow(dead_code)]
    engine_name: String,
    #[allow(dead_code)]
    time_taken: u64, // milliseconds
    #[allow(dead_code)]
    depth_searched: u8,
}

#[allow(dead_code)]
struct GameState {
    game: Game,
    move_history: Vec<MoveAnalysis>,
    chess_vector_color: Color,
    start_time: DateTime<Utc>,
}

#[allow(dead_code)]
impl GameState {
    fn new(chess_vector_color: Color) -> Self {
        Self {
            game: Game::new(),
            move_history: Vec::new(),
            chess_vector_color,
            start_time: Utc::now(),
        }
    }

    fn current_board(&self) -> Board {
        self.game.current_position()
    }

    fn make_move(&mut self, analysis: MoveAnalysis) -> Result<(), &'static str> {
        if self.game.make_move(analysis.chess_move) {
            self.move_history.push(analysis);
            Ok(())
        } else {
            Err("Invalid move")
        }
    }

    fn is_chess_vector_turn(&self) -> bool {
        self.current_board().side_to_move() == self.chess_vector_color
    }

    fn calculate_statistics(&self) -> (f32, f32, f32, f32) {
        let chess_vector_moves: Vec<_> = self
            .move_history
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                let move_color = if i % 2 == 0 {
                    Color::White
                } else {
                    Color::Black
                };
                move_color == self.chess_vector_color
            })
            .map(|(_, analysis)| analysis)
            .collect();

        let stockfish_moves: Vec<_> = self
            .move_history
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                let move_color = if i % 2 == 0 {
                    Color::White
                } else {
                    Color::Black
                };
                move_color != self.chess_vector_color
            })
            .map(|(_, analysis)| analysis)
            .collect();

        let chess_vector_total_loss: f32 =
            chess_vector_moves.iter().map(|m| m.centipawn_loss).sum();
        let chess_vector_good_moves = chess_vector_moves
            .iter()
            .filter(|m| m.centipawn_loss < 20.0)
            .count();
        let chess_vector_accuracy = if chess_vector_moves.is_empty() {
            0.0
        } else {
            chess_vector_good_moves as f32 / chess_vector_moves.len() as f32 * 100.0
        };

        let stockfish_total_loss: f32 = stockfish_moves.iter().map(|m| m.centipawn_loss).sum();
        let stockfish_good_moves = stockfish_moves
            .iter()
            .filter(|m| m.centipawn_loss < 20.0)
            .count();
        let stockfish_accuracy = if stockfish_moves.is_empty() {
            0.0
        } else {
            stockfish_good_moves as f32 / stockfish_moves.len() as f32 * 100.0
        };

        (
            chess_vector_total_loss,
            chess_vector_accuracy,
            stockfish_total_loss,
            stockfish_accuracy,
        )
    }
}

fn main() {
    println!("Chess Vector Engine - Position Analyzer");
    println!("=======================================");

    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        println!("Usage: {} <FEN_position>", args[0]);
        println!(
            "Example: {} \"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1\"",
            args[0]
        );
        return;
    }

    let fen = &args[1];
    let board = Board::from_str(fen).expect("Valid FEN");

    println!("Complete");
    println!("Side to move: {:?}", board.side_to_move());

    // Initialize engine with intelligent architecture selection
    let mut engine = initialize_analysis_engine();

    println!(
        "\nKnowledge base loaded with {} positions",
        engine.knowledge_base_size()
    );

    // Analyze the position
    analyze_position(&mut engine, &board);
}

/// Initialize analysis engine with advanced features based on available data
fn initialize_analysis_engine() -> ChessVectorEngine {
    println!("🔍 Initializing analysis engine with auto-loading...");

    let _game_state = GameState::new(Color::White);

    // Initialize Chess Vector Engine with premium features unlocked
    // Note: This is a full-featured open source demonstration tool
    println!("🤖 Initializing Chess Vector Engine (Full Feature Demo)...");
    let mut chess_vector_engine = ChessVectorEngine::new_strong(1024);

    // Enable opening book
    chess_vector_engine.enable_opening_book();
    println!("📚 Opening book enabled (50+ openings)");

    // Configure strong tactical search to match Stockfish strength
    use chess_vector_engine::TacticalConfig;
    let mut strong_tactical_config = TacticalConfig::strong(); // Use optimized strong config
    strong_tactical_config.max_depth = 16_u32; // Match or exceed Stockfish depth
    strong_tactical_config.max_time_ms = 1000_u64; // Match Stockfish thinking time
    strong_tactical_config.num_threads = 8; // More threads for stronger search

    chess_vector_engine.enable_tactical_search(strong_tactical_config.clone());

    // Configure hybrid evaluation with balanced pattern recognition and tactical search
    let hybrid_config = HybridConfig {
        pattern_confidence_threshold: 0.65, // Balanced trust in patterns - use tactical search when uncertain
        enable_tactical_refinement: true,
        tactical_config: strong_tactical_config,
        pattern_weight: 0.6, // Give patterns significant influence while still using tactical search
        min_similar_positions: 3,
    };
    chess_vector_engine.configure_hybrid_evaluation(hybrid_config);
    println!("🎯 Hybrid evaluation configured (confidence threshold: 0.65, pattern weight: 0.6)");

    // Try to load training data for pattern recognition
    println!("🧠 Loading training data for pattern recognition...");
    if let Err(e) = chess_vector_engine.auto_load_training_data() {
        println!(
            "⚠️  No training data found, using tactical search fallback: {}",
            e
        );
        println!("   (Pattern recognition will fall back to tactical search)");
    } else {
        println!("✅ Training data loaded - pattern recognition active");
    }

    // Try to load Lichess puzzles if available
    let lichess_paths = vec![
        "lichess_db_puzzle.csv",
        "Downloads/lichess_db_puzzle.csv",
        "~/Downloads/lichess_db_puzzle.csv",
    ];

    for lichess_path in lichess_paths {
        let expanded_path = if lichess_path.starts_with("~/") {
            if let Ok(home) = std::env::var("HOME") {
                lichess_path.replace("~", &home)
            } else {
                continue;
            }
        } else {
            lichess_path.to_string()
        };

        if std::path::Path::new(&expanded_path).exists() {
            println!("🧠 Loading Lichess puzzles from {expanded_path}...");

            // Try premium loading first, fall back to basic
            if true {
                // Ultra-fast loading always available in open source
                println!("🚀 Using premium ultra-fast loading...");
                match chess_vector_engine.load_lichess_puzzles(&expanded_path) {
                    Ok(()) => {
                        println!("✅ Premium Lichess loading successful!");
                        break;
                    }
                    Err(e) => println!("⚠️ Premium loading failed: {e}"),
                }
            } else {
                println!("📚 Using basic puzzle loading (limited to 50,000 puzzles)...");
                match chess_vector_engine
                    .load_lichess_puzzles_with_limit(&expanded_path, Some(50_000))
                {
                    Ok(()) => {
                        println!("✅ Basic Lichess loading successful!");
                        break;
                    }
                    Err(e) => println!("⚠️ Basic loading failed: {e}"),
                }
            }
        }
    }

    // Check for available advanced features
    if chess_vector_engine.check_gpu_acceleration().is_ok() {
        println!("🚀 GPU acceleration available (Premium feature)");
    }
    if chess_vector_engine.is_tactical_search_enabled() {
        println!("⚡ Advanced tactical search available (Premium feature)");
        println!("   (Using enhanced search algorithms and deeper analysis)");
    }

    println!("🎯 Hybrid Evaluation Pipeline Active:");
    println!("   1. Opening Book Lookup (instant for known positions)");
    println!("   2. Pattern Recognition (similarity search in position space)");
    println!("   3. Tactical Search Fallback (6+ ply minimax with pruning)");
    println!("   4. Confidence-Based Blending (combines pattern + tactical analysis)");

    chess_vector_engine
}

/// Load a basic opening book with evaluations
#[allow(dead_code)]
fn load_opening_book(engine: &mut ChessVectorEngine) {
    let openings = vec![
        // Basic opening principles: control center, develop pieces
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            0.0,
        ),
        (
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            0.2,
        ),
        (
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
            0.0,
        ),
        (
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
            0.15,
        ),
        (
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            0.1,
        ),
        // Sicilian variations
        (
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
            -0.1,
        ),
        (
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
            0.0,
        ),
        (
            "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            0.05,
        ),
        // French Defense
        (
            "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            -0.05,
        ),
        (
            "rnbqkbnr/pppp1ppp/4p3/8/4P3/3P4/PPP2PPP/RNBQKBNR b KQkq - 0 2",
            0.1,
        ),
        // Caro-Kann
        (
            "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            0.0,
        ),
        (
            "rnbqkbnr/pp1ppppp/2p5/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq d3 0 2",
            0.1,
        ),
        // King's Indian Setup
        (
            "rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
            0.2,
        ),
        (
            "rnbqk2r/ppppppbp/5np1/8/2PP4/2N2N2/PP2PPPP/R1BQKB1R b KQkq - 3 4",
            0.15,
        ),
        // Queen's Gambit
        (
            "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2",
            0.1,
        ),
        (
            "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
            0.05,
        ),
        // Common tactical patterns
        (
            "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 3",
            0.0,
        ),
        (
            "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
            0.3,
        ),
    ];

    for (fen, eval) in openings {
        let board = Board::from_str(fen).expect("Valid FEN");
        engine.add_position(&board, eval);
    }
}

/// Comprehensive position analysis
fn analyze_position(engine: &mut ChessVectorEngine, board: &Board) {
    println!("\n=== Position Analysis ===");

    // Basic position info
    let _legal_moves: Vec<_> = MoveGen::new_legal(board).collect();
    println!("Analysis starting");

    if board.checkers().popcnt() > 0 {
        println!("⚠️  King is in check!");
    }

    // Vector representation
    let position_vector = engine.encode_position(board);
    println!(
        "Position encoded as {}-dimensional vector",
        position_vector.len()
    );

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

            println!(
                "  {}. Eval: {:+.2}, Similarity: {:.3} ({})",
                i + 1,
                evaluation,
                similarity,
                similarity_desc
            );
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

        println!(
            "Predicted evaluation: {:+.3} ({})",
            predicted_eval, eval_desc
        );
    } else {
        println!("Unable to evaluate - no similar positions in knowledge base");
    }

    // Get best move from Chess Vector Engine (uses hybrid recommendation system)
    let recommendations = engine.recommend_moves(board, 5);

    if recommendations.is_empty() {
        println!("❌ Chess Vector has no legal moves!");
    }

    let _best_move = recommendations[0].chess_move;
    let _confidence = recommendations[0].confidence;

    println!("\n=== Recommended Moves In Order ===\n\n");

    println!("{:?}", recommendations);

    // Material analysis
    analyze_material(board);

    // Positional features
    analyze_positional_features(board);
}

/// Analyze material balance
fn analyze_material(board: &Board) {
    println!("\n=== Material Analysis ===");

    use chess::{Color, Piece};

    let pieces = [
        Piece::Pawn,
        Piece::Knight,
        Piece::Bishop,
        Piece::Rook,
        Piece::Queen,
    ];
    let values = [1, 3, 3, 5, 9];

    let mut white_material = 0;
    let mut black_material = 0;

    for (piece, value) in pieces.iter().zip(values.iter()) {
        let white_count = (board.pieces(*piece) & board.color_combined(Color::White)).popcnt();
        let black_count = (board.pieces(*piece) & board.color_combined(Color::Black)).popcnt();

        white_material += white_count * value;
        black_material += black_count * value;

        if white_count != black_count {
            println!(
                "{:?}: White {} vs Black {} (difference: {:+})",
                piece,
                white_count,
                black_count,
                white_count as i32 - black_count as i32
            );
        }
    }

    let material_balance = white_material as i32 - black_material as i32;
    println!(
        "Total material: White {} vs Black {} (balance: {:+})",
        white_material, black_material, material_balance
    );
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

    println!(
        "Developed pieces: White {} vs Black {}",
        white_developed, black_developed
    );
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
