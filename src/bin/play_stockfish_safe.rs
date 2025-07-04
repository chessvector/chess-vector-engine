// Copy of play_stockfish.rs but WITHOUT Lichess puzzle loading to test if that's the issue

use chess::{Board, ChessMove, Color, Game, MoveGen, Piece, Square};
use chess_vector_engine::ChessVectorEngine;
use chrono::{DateTime, Utc};
use clap::{Arg, Command};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::process::{Child, Command as ProcessCommand, Stdio};
use std::str::FromStr;

/// Normalize evaluation to be from White's perspective and in reasonable units
fn normalize_evaluation(raw_eval: f32, _side_to_move: Color) -> f32 {
    // Tactical search now returns pawn values directly (not centipawns)
    // No conversion needed - just clamp to reasonable range

    // The tactical search always returns evaluation from White's perspective
    // (positive = good for White, negative = good for Black)
    // No perspective conversion needed

    // Clamp to reasonable range (-10 to +10 pawns)
    raw_eval.clamp(-10.0, 10.0)
}

/// Convert a ChessMove to Standard Algebraic Notation (SAN)
fn convert_to_san(board: &Board, chess_move: ChessMove) -> String {
    let from_square = chess_move.get_source();
    let to_square = chess_move.get_dest();
    let piece = board.piece_on(from_square);

    if piece.is_none() {
        return chess_move.to_string(); // Fallback to long algebraic notation
    }

    let piece = piece.unwrap();
    let promotion = chess_move.get_promotion();

    // Handle castling
    if piece == Piece::King {
        if from_square == Square::E1 && to_square == Square::G1 {
            return "O-O".to_string();
        }
        if from_square == Square::E1 && to_square == Square::C1 {
            return "O-O-O".to_string();
        }
        if from_square == Square::E8 && to_square == Square::G8 {
            return "O-O".to_string();
        }
        if from_square == Square::E8 && to_square == Square::C8 {
            return "O-O-O".to_string();
        }
    }

    let mut san = String::new();

    // Add piece letter (except for pawns)
    if piece != Piece::Pawn {
        san.push(match piece {
            Piece::King => 'K',
            Piece::Queen => 'Q',
            Piece::Rook => 'R',
            Piece::Bishop => 'B',
            Piece::Knight => 'N',
            Piece::Pawn => unreachable!(),
        });
    }

    // Handle disambiguation for pieces (simplified)
    if piece != Piece::Pawn {
        let legal_moves: Vec<ChessMove> = MoveGen::new_legal(board).collect();
        let same_piece_moves: Vec<ChessMove> = legal_moves
            .iter()
            .filter(|&mv| {
                board.piece_on(mv.get_source()) == Some(piece)
                    && mv.get_dest() == to_square
                    && mv.get_source() != from_square
            })
            .cloned()
            .collect();

        if !same_piece_moves.is_empty() {
            // Need disambiguation - use file if different files, otherwise use rank
            let need_file_disambiguation = same_piece_moves
                .iter()
                .any(|mv| mv.get_source().get_file() != from_square.get_file());

            if need_file_disambiguation {
                san.push((b'a' + from_square.get_file().to_index() as u8) as char);
            } else {
                san.push((b'1' + from_square.get_rank().to_index() as u8) as char);
            }
        }
    }

    // Handle captures
    let is_capture = board.piece_on(to_square).is_some()
        || (piece == Piece::Pawn && from_square.get_file() != to_square.get_file());

    if is_capture {
        // For pawn captures, add the file of departure
        if piece == Piece::Pawn {
            san.push((b'a' + from_square.get_file().to_index() as u8) as char);
        }
        san.push('x');
    }

    // Add destination square
    san.push((b'a' + to_square.get_file().to_index() as u8) as char);
    san.push((b'1' + to_square.get_rank().to_index() as u8) as char);

    // Handle promotion
    if let Some(promo_piece) = promotion {
        san.push('=');
        san.push(match promo_piece {
            Piece::Queen => 'Q',
            Piece::Rook => 'R',
            Piece::Bishop => 'B',
            Piece::Knight => 'N',
            _ => 'Q', // Default to queen
        });
    }

    // Check for check/checkmate (simplified - just add if next position is in check)
    let next_board = board.make_move_new(chess_move);
    if next_board.checkers().0 != 0 {
        // Simple check detection - would need full game state to determine checkmate
        san.push('+');
    }

    san
}

/// Enhanced Stockfish process for interactive gameplay
struct StockfishPlayer {
    child: Child,
    stdin: BufWriter<std::process::ChildStdin>,
    stdout: BufReader<std::process::ChildStdout>,
    #[allow(dead_code)]
    name: String,
    #[allow(dead_code)]
    depth: u8,
    time_per_move: u32, // milliseconds
}

impl StockfishPlayer {
    fn new(depth: u8, time_per_move: u32) -> Result<Self, Box<dyn std::error::Error>> {
        let mut child = ProcessCommand::new("stockfish")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                format!(
                    "Failed to start Stockfish. Make sure it's installed and in PATH. Error: {}",
                    e
                )
            })?;

        let stdin = BufWriter::new(
            child
                .stdin
                .take()
                .ok_or("Failed to get stdin handle for Stockfish process")?,
        );
        let stdout = BufReader::new(
            child
                .stdout
                .take()
                .ok_or("Failed to get stdout handle for Stockfish process")?,
        );

        let mut player = Self {
            child,
            stdin,
            stdout,
            name: "Stockfish".to_string(),
            depth,
            time_per_move,
        };

        // Initialize UCI
        player.send_command("uci")?;
        player.wait_for_uciok()?;
        player.send_command("setoption name Hash value 256")?;
        player.send_command("setoption name Threads value 1")?;
        player.send_command("isready")?;
        player.wait_for_readyok()?;

        println!("✅ Stockfish engine ready (depth {})", depth);

        Ok(player)
    }

    fn send_command(&mut self, command: &str) -> Result<(), Box<dyn std::error::Error>> {
        writeln!(self.stdin, "{command}")?;
        self.stdin.flush()?;
        Ok(())
    }

    fn wait_for_uciok(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut line = String::new();
        loop {
            line.clear();
            self.stdout.read_line(&mut line)?;
            if line.trim() == "uciok" {
                break;
            }
        }
        Ok(())
    }

    fn wait_for_readyok(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut line = String::new();
        loop {
            line.clear();
            self.stdout.read_line(&mut line)?;
            if line.trim() == "readyok" {
                break;
            }
        }
        Ok(())
    }

    fn get_best_move(
        &mut self,
        board: &Board,
    ) -> Result<(ChessMove, f32), Box<dyn std::error::Error>> {
        let fen = board.to_string();

        // Send position
        self.send_command(&format!("position fen {fen}"))?;

        // Start search with time control
        self.send_command(&format!("go movetime {}", self.time_per_move))?;

        let mut line = String::new();
        let mut best_move = None;
        let mut evaluation = 0.0;

        loop {
            line.clear();
            self.stdout.read_line(&mut line)?;
            let line = line.trim();

            if line.starts_with("info") && line.contains("depth") && line.contains("pv") {
                // Extract evaluation
                if line.contains("score cp") {
                    if let Some(cp_pos) = line.find("score cp ") {
                        let cp_str = &line[cp_pos + 9..];
                        if let Some(end) = cp_str.find(' ') {
                            if let Ok(cp_value) = cp_str[..end].parse::<i32>() {
                                evaluation = cp_value as f32 / 100.0;
                            }
                        }
                    }
                } else if line.contains("score mate") {
                    if let Some(mate_pos) = line.find("score mate ") {
                        let mate_str = &line[mate_pos + 11..];
                        if let Some(end) = mate_str.find(' ') {
                            if let Ok(mate_moves) = mate_str[..end].parse::<i32>() {
                                evaluation = if mate_moves > 0 { 100.0 } else { -100.0 };
                            }
                        }
                    }
                }

                // Extract principal variation
                if let Some(pv_pos) = line.find(" pv ") {
                    let _pv_line = line[pv_pos + 4..].to_string();
                }
            } else if line.starts_with("bestmove") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 && parts[1] != "(none)" {
                    best_move = ChessMove::from_str(parts[1]).ok();
                }
                break;
            }
        }

        match best_move {
            Some(mv) => Ok((mv, evaluation)),
            None => Err("Stockfish did not return a valid move".into()),
        }
    }
}

impl Drop for StockfishPlayer {
    fn drop(&mut self) {
        let _ = self.send_command("quit");
        let _ = self.child.wait();
    }
}

/// Game state and analysis
#[derive(Debug, Clone)]
struct MoveAnalysis {
    chess_move: ChessMove,
    #[allow(dead_code)]
    evaluation_before: f32,
    evaluation_after: f32,
    centipawn_loss: f32,
    #[allow(dead_code)]
    engine_name: String,
    #[allow(dead_code)]
    time_taken: u64, // milliseconds
    #[allow(dead_code)]
    depth_searched: u8,
}

struct GameState {
    game: Game,
    move_history: Vec<MoveAnalysis>,
    chess_vector_color: Color,
    start_time: DateTime<Utc>,
}

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

    fn generate_pgn(&self) -> String {
        let mut pgn = String::new();

        // PGN headers
        pgn.push_str("[Event \"Chess Vector vs Stockfish (Safe Mode)\"]\n");
        pgn.push_str("[Site \"CLI Match\"]\n");
        pgn.push_str(&format!(
            "[Date \"{}\"]\n",
            self.start_time.format("%Y.%m.%d")
        ));
        pgn.push_str("[Round \"1\"]\n");

        match self.chess_vector_color {
            Color::White => {
                pgn.push_str("[White \"Chess Vector Engine\"]\n");
                pgn.push_str("[Black \"Stockfish\"]\n");
            }
            Color::Black => {
                pgn.push_str("[White \"Stockfish\"]\n");
                pgn.push_str("[Black \"Chess Vector Engine\"]\n");
            }
        }

        // Game result
        let result = match self.game.result() {
            Some(chess::GameResult::WhiteCheckmates) => "1-0",
            Some(chess::GameResult::BlackCheckmates) => "0-1",
            Some(chess::GameResult::WhiteResigns) => "0-1",
            Some(chess::GameResult::BlackResigns) => "1-0",
            Some(chess::GameResult::Stalemate) => "1/2-1/2",
            Some(chess::GameResult::DrawAccepted) => "1/2-1/2",
            Some(chess::GameResult::DrawDeclared) => "1/2-1/2",
            None => "*",
        };
        pgn.push_str(&format!("[Result \"{}\"]\n\n", result));

        // Moves with analysis annotations - using proper SAN notation
        let mut move_number = 1;
        let mut current_board = Board::default(); // Start with initial position

        for (i, analysis) in self.move_history.iter().enumerate() {
            if i % 2 == 0 {
                pgn.push_str(&format!("{}. ", move_number));
            }

            // Convert move to proper SAN notation
            let move_san = convert_to_san(&current_board, analysis.chess_move);
            pgn.push_str(&move_san);

            // Make the move to update the board for the next iteration
            current_board = current_board.make_move_new(analysis.chess_move);

            // Add evaluation annotation
            if analysis.centipawn_loss > 50.0 {
                pgn.push_str("??"); // Blunder
            } else if analysis.centipawn_loss > 20.0 {
                pgn.push('?'); // Questionable move
            }

            // Add evaluation comment
            pgn.push_str(&format!(
                " {{ eval: {:.2}, loss: {:.1}cp }} ",
                analysis.evaluation_after, analysis.centipawn_loss
            ));

            if i % 2 == 1 {
                move_number += 1;
                pgn.push('\n');
            }
        }

        pgn.push_str(&format!(" {}\n", result));
        pgn
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

fn play_game(
    chess_vector_color: Color,
    stockfish_depth: u8,
    time_per_move: u32,
    tactical_depth: u8,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎮 Starting Chess Vector vs Stockfish match! (SAFE MODE - No Lichess Loading)");
    println!(
        "♟️  Chess Vector plays as: {}",
        if chess_vector_color == Color::White {
            "White"
        } else {
            "Black"
        }
    );
    println!(
        "🐟 Stockfish: depth {}, {}ms per move",
        stockfish_depth, time_per_move
    );
    println!(
        "🤖 Chess Vector: tactical depth {}, {}ms per move\n",
        tactical_depth, time_per_move
    );

    let mut game_state = GameState::new(chess_vector_color);

    // Initialize Chess Vector Engine WITHOUT Lichess loading
    println!("🤖 Initializing Chess Vector Engine (Safe Mode)...");
    let mut chess_vector_engine = ChessVectorEngine::new(1024);

    // Enable opening book
    chess_vector_engine.enable_opening_book();
    println!("📚 Opening book enabled (50+ openings)");

    // Configure strong tactical search to match Stockfish strength
    use chess_vector_engine::TacticalConfig;
    let mut strong_tactical_config = TacticalConfig::strong(); // Use optimized strong config
    strong_tactical_config.max_depth = tactical_depth as u32; // Match or exceed Stockfish depth
    strong_tactical_config.max_time_ms = time_per_move as u64; // Match Stockfish thinking time
    strong_tactical_config.num_threads = 8; // More threads for stronger search

    chess_vector_engine.enable_tactical_search(strong_tactical_config);
    println!(
        "⚔️  Strong tactical search enabled ({} ply depth, {}ms time, 2M nodes, 8 threads)",
        tactical_depth, time_per_move
    );

    // Load only basic training data for pattern recognition
    println!("🧠 Loading basic training data for pattern recognition...");
    if let Err(e) = chess_vector_engine.auto_load_training_data() {
        println!(
            "⚠️  No training data found, using tactical search fallback: {}",
            e
        );
        println!("   (Pattern recognition will fall back to tactical search)");
    } else {
        println!("✅ Basic training data loaded - pattern recognition active");
    }

    println!("🎯 Hybrid Evaluation Pipeline Active:");
    println!("   1. Opening Book Lookup (instant for known positions)");
    println!("   2. Pattern Recognition (basic similarity search)");
    println!(
        "   3. Tactical Search Fallback ({}+ ply minimax with pruning)",
        tactical_depth
    );

    let mut stockfish = StockfishPlayer::new(stockfish_depth, time_per_move)?;

    // Game loop
    let mut move_count = 0;
    while game_state.game.result().is_none() && move_count < 200 {
        let current_board = game_state.current_board();
        let is_chess_vector_turn = game_state.is_chess_vector_turn();

        println!("\n--- Move {} ---", move_count + 1);
        println!("Position: {}", current_board);
        println!(
            "Turn: {}",
            if current_board.side_to_move() == Color::White {
                "White"
            } else {
                "Black"
            }
        );

        let start_time = std::time::Instant::now();

        if is_chess_vector_turn {
            println!("🤖 Chess Vector is thinking... (safe mode)");

            // Check which evaluation method will be used
            let eval_method = if chess_vector_engine
                .get_opening_entry(&current_board)
                .is_some()
            {
                "📚 Opening Book"
            } else {
                "🧠 Basic Pattern Recognition + ⚔️ Tactical Search"
            };

            // Get current evaluation using hybrid approach
            let eval_before = chess_vector_engine
                .evaluate_position(&current_board)
                .unwrap_or(0.0);

            // Get best move from Chess Vector Engine (uses hybrid recommendation system)
            let recommendations = chess_vector_engine.recommend_moves(&current_board, 1);

            if recommendations.is_empty() {
                println!("❌ Chess Vector has no legal moves!");
                break;
            }

            let best_move = recommendations[0].chess_move;
            let confidence = recommendations[0].confidence;

            // Make the move to evaluate the resulting position
            let mut temp_board = current_board;
            temp_board = temp_board.make_move_new(best_move);
            let eval_after_raw = chess_vector_engine
                .evaluate_position(&temp_board)
                .unwrap_or(0.0);

            // Normalize evaluation: convert from centipawns to pawns and fix perspective
            let eval_after = normalize_evaluation(eval_after_raw, temp_board.side_to_move());

            let time_taken = start_time.elapsed().as_millis() as u64;

            // Show which hybrid component was primarily used
            println!(
                "🎯 Chess Vector plays: {} | Method: {} | Confidence: {:.2} | Eval: {:.2}",
                best_move, eval_method, confidence, eval_after
            );

            let analysis = MoveAnalysis {
                chess_move: best_move,
                evaluation_before: normalize_evaluation(eval_before, current_board.side_to_move()),
                evaluation_after: eval_after,
                centipawn_loss: 0.0, // Will be calculated later
                engine_name: "Chess Vector".to_string(),
                time_taken,
                depth_searched: tactical_depth,
            };

            game_state.make_move(analysis)?;
        } else {
            println!("🐟 Stockfish is thinking...");

            // Get Stockfish's evaluation before move
            let (best_move, eval_after_raw) = stockfish.get_best_move(&current_board)?;
            let time_taken = start_time.elapsed().as_millis() as u64;

            // Normalize Stockfish evaluation (it's already in pawns, just clamp it)
            let eval_after = eval_after_raw.clamp(-10.0, 10.0);

            println!(
                "🎯 Stockfish plays: {} (eval: {:.2})",
                best_move, eval_after
            );

            let analysis = MoveAnalysis {
                chess_move: best_move,
                evaluation_before: 0.0, // Will be set from previous move
                evaluation_after: eval_after,
                centipawn_loss: 0.0, // Will be calculated later
                engine_name: "Stockfish".to_string(),
                time_taken,
                depth_searched: stockfish_depth,
            };

            game_state.make_move(analysis)?;
        }

        move_count += 1;

        // Show current game status
        let board = game_state.current_board();
        if board.checkers().0 != 0 {
            println!("⚠️  Check!");
        }
    }

    // Calculate centipawn losses
    calculate_centipawn_losses(&mut game_state.move_history);

    // Game finished
    println!("\n🏁 Game finished!");
    match game_state.game.result() {
        Some(chess::GameResult::WhiteCheckmates) => println!("🏆 White wins by checkmate!"),
        Some(chess::GameResult::BlackCheckmates) => println!("🏆 Black wins by checkmate!"),
        Some(chess::GameResult::WhiteResigns) => println!("🏆 Black wins by resignation!"),
        Some(chess::GameResult::BlackResigns) => println!("🏆 White wins by resignation!"),
        Some(chess::GameResult::Stalemate) => println!("🤝 Draw by stalemate"),
        Some(chess::GameResult::DrawAccepted) => println!("🤝 Draw agreed"),
        Some(chess::GameResult::DrawDeclared) => println!("🤝 Draw declared"),
        None => println!("🤝 Game incomplete (move limit reached)"),
    }

    // Show statistics
    let (cv_loss, cv_accuracy, sf_loss, sf_accuracy) = game_state.calculate_statistics();

    println!("\n📊 Game Statistics:");
    println!("Chess Vector Engine:");
    println!("  - Total centipawn loss: {:.1}", cv_loss);
    println!("  - Accuracy: {:.1}%", cv_accuracy);
    println!("\nStockfish:");
    println!("  - Total centipawn loss: {:.1}", sf_loss);
    println!("  - Accuracy: {:.1}%", sf_accuracy);

    // Generate and display PGN
    println!("\n📋 PGN:");
    println!("{}", game_state.generate_pgn());

    // Save PGN to file
    let filename = format!(
        "chess_vector_vs_stockfish_safe_{}.pgn",
        game_state.start_time.format("%Y%m%d_%H%M%S")
    );
    std::fs::write(&filename, game_state.generate_pgn())?;
    println!("📁 Game saved to {filename}");

    Ok(())
}

fn calculate_centipawn_losses(move_history: &mut [MoveAnalysis]) {
    for i in 0..move_history.len() {
        if i > 0 {
            let prev_eval = move_history[i - 1].evaluation_after;
            let current_eval = move_history[i].evaluation_after;

            // Calculate centipawn loss properly
            // All evaluations are from White's perspective
            // So for ANY move, if the evaluation gets worse for the side to move, that's a loss
            let eval_diff = if i % 2 == 1 {
                // Odd index = White just moved (0-indexed)
                // White move: if evaluation decreased, that's bad for white
                prev_eval - current_eval
            } else {
                // Even index = Black just moved
                // Black move: if evaluation increased (better for white), that's bad for black
                current_eval - prev_eval
            };

            // Convert to centipawns and ensure non-negative
            move_history[i].centipawn_loss = (eval_diff * 100.0).max(0.0);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("Chess Vector vs Stockfish (Safe Mode)")
        .version("1.0")
        .author("Chess Vector Team")
        .about("Play Chess Vector Engine against Stockfish with analysis (without Lichess puzzles)")
        .arg(
            Arg::new("color")
                .short('c')
                .long("color")
                .value_name("COLOR")
                .help("Color for Chess Vector Engine (white/black)")
                .value_parser(["white", "black"])
                .default_value("white"),
        )
        .arg(
            Arg::new("depth")
                .short('d')
                .long("depth")
                .value_name("DEPTH")
                .help("Stockfish search depth")
                .value_parser(clap::value_parser!(u8).range(1..=20))
                .default_value("10"),
        )
        .arg(
            Arg::new("time")
                .short('t')
                .long("time")
                .value_name("MILLISECONDS")
                .help("Time per move in milliseconds")
                .value_parser(clap::value_parser!(u32).range(100..=60000))
                .default_value("3000"),
        )
        .arg(
            Arg::new("tactical_depth")
                .long("tactical-depth")
                .value_name("DEPTH")
                .help("Chess Vector tactical search depth (higher = stronger but slower)")
                .value_parser(clap::value_parser!(u8).range(4..=16))
                .default_value("10"),
        )
        .get_matches();

    let color_str = matches.get_one::<String>("color").unwrap();
    let chess_vector_color = match color_str.as_str() {
        "white" => Color::White,
        "black" => Color::Black,
        _ => unreachable!(),
    };

    let stockfish_depth = *matches.get_one::<u8>("depth").unwrap();
    let time_per_move = *matches.get_one::<u32>("time").unwrap();
    let tactical_depth = *matches.get_one::<u8>("tactical_depth").unwrap();

    // Check if Stockfish is available
    match ProcessCommand::new("stockfish").arg("--help").output() {
        Ok(_) => {}
        Err(_) => {
            eprintln!(
                "❌ Stockfish not found! Please install Stockfish and make sure it's in your PATH."
            );
            eprintln!("   Download from: https://stockfishchess.org/download/");
            std::process::exit(1);
        }
    }

    play_game(
        chess_vector_color,
        stockfish_depth,
        time_per_move,
        tactical_depth,
    )
}
