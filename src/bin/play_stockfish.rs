use chess::{Board, ChessMove, Color, Game, MoveGen};
use chess_vector_engine::ChessVectorEngine;
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::str::FromStr;
use rand::seq::SliceRandom;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    
    // Check for help flag
    if args.contains(&"--help".to_string()) || args.contains(&"-h".to_string()) {
        print_usage();
        return Ok(());
    }
    
    let rebuild_models = args.contains(&"--rebuild-models".to_string());
    let convert_to_binary = args.contains(&"--convert-to-binary".to_string());
    let force_training_files = args.contains(&"--force-training-files".to_string());
    
    // Handle binary conversion and exit
    if convert_to_binary {
        println!("🔄 Converting JSON training files to binary format...");
        match ChessVectorEngine::convert_json_to_binary() {
            Ok(converted) => {
                if converted.is_empty() {
                    println!("ℹ️  No JSON training files found to convert");
                } else {
                    for conversion in converted {
                        println!("   {}", conversion);
                    }
                }
            }
            Err(e) => println!("❌ Conversion failed: {}", e),
        }
        return Ok(());
    }
    
    println!("Starting game: Chess Vector Engine vs Stockfish (Improved Version)");
    if rebuild_models {
        println!("🔄 Rebuild models flag detected - will retrain LSH and manifold learning");
    } else {
        println!("⚡ Using fast startup optimized for gameplay (use --rebuild-models to retrain)");
    }
    
    // Initialize the chess vector engine with opening book and substantial knowledge base
    let mut engine = create_engine_with_knowledge(rebuild_models || force_training_files);
    
    // Start Stockfish process
    let mut stockfish = Command::new("stockfish")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    
    let mut stockfish_stdin = stockfish.stdin.take().unwrap();
    let stockfish_stdout = stockfish.stdout.take().unwrap();
    let mut stockfish_reader = BufReader::new(stockfish_stdout);
    
    // Configure Stockfish
    writeln!(stockfish_stdin, "uci")?;
    wait_for_uciok(&mut stockfish_reader)?;
    writeln!(stockfish_stdin, "setoption name Skill Level value 8")?; // Moderate strength
    writeln!(stockfish_stdin, "ucinewgame")?;
    
    let mut game = Game::new();
    let mut move_number = 1;
    
    // Choose random opening for variety
    let opening_move = choose_random_opening(&mut engine);
    
    println!("\nGame begins! Vector Engine plays White, Stockfish plays Black");
    if let Some(opening) = opening_move {
        println!("Vector Engine will try to play opening starting with: {}", opening);
    }
    println!();
    
    loop {
        let current_board = game.current_position();
        
        if current_board.status() != chess::BoardStatus::Ongoing {
            break;
        }
        
        let mv = if current_board.side_to_move() == Color::White {
            // Vector engine's turn (White) - USE THE PROPER ENGINE API
            print!("{}. ", move_number);
            let engine_move = get_engine_move_smart(&mut engine, &current_board)?;
            println!("{}", engine_move);
            engine_move
        } else {
            // Stockfish's turn (Black)
            let stockfish_move = get_stockfish_move(&mut stockfish_stdin, &mut stockfish_reader, &current_board)?;
            println!("{}... {}", move_number, stockfish_move);
            move_number += 1;
            stockfish_move
        };
        
        game.make_move(mv);
    }
    
    // Game over
    let final_board = game.current_position();
    println!("\nGame Over!");
    
    match final_board.status() {
        chess::BoardStatus::Checkmate => {
            if final_board.side_to_move() == Color::White {
                println!("Stockfish wins by checkmate!");
            } else {
                println!("Vector Engine wins by checkmate!");
            }
        }
        chess::BoardStatus::Stalemate => println!("Draw by stalemate!"),
        _ => println!("Game ended"),
    }
    
    println!("\nGame moves (algebraic notation):");
    for (i, mv) in game.actions().iter().enumerate() {
        if i % 2 == 0 {
            print!("{}. {:?}", (i / 2) + 1, mv);
        } else {
            println!(" {:?}", mv);
        }
    }
    
    // Generate PGN format
    println!("\nPGN Format:");
    generate_pgn(&game);
    
    // Save game data to database for future learning
    println!("\n💾 Saving game data to database for future learning...");
    save_game_to_database(&mut engine, &game);
    
    // Clean up Stockfish
    writeln!(stockfish_stdin, "quit")?;
    stockfish.wait()?;
    
    Ok(())
}

fn create_engine_with_knowledge(rebuild_models: bool) -> ChessVectorEngine {
    println!("🚀 Initializing Chess Vector Engine for gameplay...");
    
    // Smart loading strategy: Check database first, fallback to training files
    let engine_result = if rebuild_models {
        println!("🔄 Rebuild flag detected - loading from training files and retraining models");
        ChessVectorEngine::new_with_auto_load(1024)
    } else {
        // Check if we have sufficient data in database first
        let mut engine = ChessVectorEngine::new(1024);
        engine.enable_opening_book();
        
        match engine.enable_persistence("chess_vector_engine.db") {
            Ok(_) => {
                match engine.load_from_database() {
                    Ok(_) => {
                        let stats = engine.training_stats();
                        if stats.total_positions >= 1000 {
                            println!("📊 Found {} positions in database - using fast database-only loading", stats.total_positions);
                            Ok(engine)
                        } else {
                            println!("📊 Database has only {} positions - loading training files to supplement", stats.total_positions);
                            ChessVectorEngine::new_with_fast_load(1024)
                        }
                    }
                    Err(_) => {
                        println!("📦 No existing database found - loading from training files");
                        ChessVectorEngine::new_with_fast_load(1024)
                    }
                }
            }
            Err(_) => {
                println!("⚠️  Could not access database - loading from training files");
                ChessVectorEngine::new_with_fast_load(1024)
            }
        }
    };
    
    match engine_result {
        Ok(mut engine) => {
            let stats = engine.training_stats();
            println!("🎯 Engine initialized with {} total positions", stats.total_positions);
            
            if stats.has_move_data {
                println!("⚔️  Includes tactical training data with {} move entries", stats.move_data_entries);
                println!("   This gives the engine tactical awareness for aggressive play!");
            }
            
            if let Some(book_stats) = engine.opening_book_stats() {
                println!("📖 Opening book: {} positions with {} ECO classifications", 
                         book_stats.total_positions, book_stats.eco_classifications);
            }
            
            // Ensure database persistence is enabled for saving game data
            if engine.enable_persistence("chess_vector_engine.db").is_err() {
                println!("⚠️  Could not enable database persistence");
            }
            
            // Enable tactical search for better chess play
            engine.enable_tactical_search_default();
            println!("⚔️  Tactical search enabled for strategic gameplay");
            
            // Enable LSH for large datasets to maintain game speed
            let mut engine = engine;
            if rebuild_models && stats.total_positions > 15000 {
                println!("📊 Large knowledge base detected, rebuilding LSH for optimal gameplay performance");
                engine.enable_lsh(10, 18); // Optimized for speed vs accuracy balance
            } else if !rebuild_models && stats.total_positions > 15000 {
                println!("📊 Large knowledge base detected, LSH models should be pre-built (use --rebuild-models to retrain)");
            }
            
            // Enable manifold learning for very large datasets during gameplay
            if rebuild_models && stats.total_positions > 30000 {
                println!("🧠 Very large knowledge base, rebuilding manifold learning for faster similarity search...");
                let _ = engine.enable_manifold_learning(6.0); // 6:1 compression for speed (1024d -> ~170d)
                
                println!("🏋️  Training manifold compression for gameplay optimization...");
                let _ = engine.train_manifold_learning(8); // Fewer epochs for faster startup
                
                println!("✅ Manifold learning enabled - optimized for real-time gameplay");
            } else if !rebuild_models && stats.total_positions > 30000 {
                println!("🧠 Very large knowledge base detected, manifold learning models should be pre-built (use --rebuild-models to retrain)");
            }
            
            engine
        }
        Err(e) => {
            println!("⚠️  Auto-loading failed: {}", e);
            println!("🔄 Falling back to basic engine with opening book...");
            
            let mut engine = ChessVectorEngine::new(1024);
            engine.enable_opening_book();
            add_position_knowledge(&mut engine);
            engine
        }
    }
}

fn add_position_knowledge(_engine: &mut ChessVectorEngine) {
    // Let the vector system learn patterns naturally through position similarity
    // The engine will use its opening book and similarity search to discover
    // tactical patterns, endgame principles, and positional concepts organically
    // based on the encoded position vectors and their inherent relationships
}

fn print_usage() {
    println!("Usage: cargo run --bin play_stockfish [OPTIONS]");
    println!();
    println!("Options:");
    println!("  --rebuild-models        Force rebuild of LSH and manifold learning models");
    println!("                          (normally models are pre-built to avoid startup delay)");
    println!("  --force-training-files  Force loading from training files instead of database");
    println!("                          (useful for getting fresh training data)");
    println!("  --convert-to-binary     Convert JSON training files to binary format");
    println!("                          (provides 5-15x faster loading for future runs)");
    println!("  --help                  Show this help message");
    println!();
    println!("Loading Strategy:");
    println!("  1st run: Loads from training files, saves to database");
    println!("  Later runs: Uses database (much faster), saves new game data");
    println!();
    println!("Examples:");
    println!("  cargo run --bin play_stockfish                        # Fast startup with optimized loading");
    println!("  cargo run --bin play_stockfish -- --rebuild-models      # Rebuild models before playing");
    println!("  cargo run --bin play_stockfish -- --convert-to-binary   # Convert JSON to binary format");
}

fn choose_random_opening(engine: &mut ChessVectorEngine) -> Option<ChessMove> {
    let starting_board = Board::default();
    
    // Try to get opening moves from the engine's opening book
    let recommendations = engine.recommend_legal_moves(&starting_board, 4);
    
    if !recommendations.is_empty() {
        // Weight the selection by confidence
        let weighted_moves: Vec<(ChessMove, u32)> = recommendations
            .iter()
            .map(|rec| (rec.chess_move, (rec.confidence * 100.0) as u32 + 1))
            .collect();
        
        // Create a weighted random selection
        let mut choices = Vec::new();
        for (mv, weight) in weighted_moves {
            for _ in 0..weight {
                choices.push(mv);
            }
        }
        
        choices.choose(&mut rand::thread_rng()).copied()
    } else {
        // Fallback to basic opening moves
        let basic_openings = vec![
            ChessMove::from_str("e2e4").ok(),
            ChessMove::from_str("d2d4").ok(), 
            ChessMove::from_str("g1f3").ok(),
            ChessMove::from_str("c2c4").ok(),
        ];
        
        basic_openings.into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .choose(&mut rand::thread_rng())
            .copied()
    }
}

fn get_engine_move_smart(engine: &mut ChessVectorEngine, board: &Board) -> Result<ChessMove, Box<dyn std::error::Error>> {
    // PROPER CHESS ENGINE APPROACH - Evaluate all legal moves using hybrid evaluation
    let legal_moves: Vec<ChessMove> = MoveGen::new_legal(board).collect();
    
    if legal_moves.is_empty() {
        return Err("No legal moves available".into());
    }
    
    let mut best_move: Option<ChessMove> = None;
    let mut best_evaluation: Option<f32> = None;
    let mut move_evaluations = Vec::new();
    
    println!("    Engine evaluating {} legal moves using hybrid system...", legal_moves.len());
    
    // Evaluate each legal move
    for chess_move in &legal_moves {
        let temp_board = board.make_move_new(*chess_move);
            // Use the engine's hybrid evaluation (opening book + patterns + tactical search)
            if let Some(position_eval) = engine.evaluate_position(&temp_board) {
                // Flip evaluation for opponent's perspective
                let eval_for_us = if board.side_to_move() == Color::White {
                    position_eval
                } else {
                    -position_eval
                };
                
                move_evaluations.push((*chess_move, eval_for_us));
                
                // Update best move
                if best_evaluation.is_none() || eval_for_us > best_evaluation.unwrap() {
                    best_move = Some(*chess_move);
                    best_evaluation = Some(eval_for_us);
                }
            }
    }
    
    // Debug output showing top moves
    if !move_evaluations.is_empty() {
        move_evaluations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        println!("    Top moves by evaluation:");
        for (i, (mv, eval)) in move_evaluations.iter().take(3).enumerate() {
            println!("      {}. {} (eval: {:.3})", i + 1, mv, eval);
        }
    }
    
    if let Some(best) = best_move {
        println!("    Selected: {} (eval: {:.3})", best, best_evaluation.unwrap_or(0.0));
        Ok(best)
    } else {
        // Fallback to first legal move if no evaluations worked
        println!("    Warning: No position evaluations available, using first legal move");
        Ok(legal_moves[0])
    }
}

fn get_stockfish_move(
    stdin: &mut std::process::ChildStdin,
    reader: &mut BufReader<std::process::ChildStdout>,
    board: &Board,
) -> Result<ChessMove, Box<dyn std::error::Error>> {
    writeln!(stdin, "position fen {}", board.to_string())?;
    writeln!(stdin, "go movetime 2000")?; // 2 second think time
    
    let mut line = String::new();
    loop {
        line.clear();
        reader.read_line(&mut line)?;
        
        if line.starts_with("bestmove") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let move_str = parts[1];
                return ChessMove::from_str(move_str)
                    .map_err(|e| format!("Chess move parse error: {:?}", e).into());
            }
        }
    }
}

fn wait_for_uciok(reader: &mut BufReader<std::process::ChildStdout>) -> Result<(), Box<dyn std::error::Error>> {
    let mut line = String::new();
    loop {
        line.clear();
        reader.read_line(&mut line)?;
        if line.trim() == "uciok" {
            break;
        }
    }
    Ok(())
}

fn generate_pgn(game: &Game) {
    use chrono::Utc;
    let current_date = Utc::now().format("%Y.%m.%d").to_string();
    
    println!("[Event \"Vector Engine vs Stockfish\"]");
    println!("[Site \"Local Analysis\"]");
    println!("[Date \"{}\"]", current_date);
    println!("[Round \"1\"]");
    println!("[White \"Chess Vector Engine\"]");
    println!("[Black \"Stockfish Level 8\"]");
    
    // Determine result
    let final_board = game.current_position();
    let result = match final_board.status() {
        chess::BoardStatus::Checkmate => {
            if final_board.side_to_move() == Color::White {
                "0-1"
            } else {
                "1-0"
            }
        }
        chess::BoardStatus::Stalemate => "1/2-1/2",
        _ => "*",
    };
    
    println!("[Result \"{}\"]", result);
    println!();
    
    // Convert moves to proper algebraic notation
    let mut board = Board::default();
    let mut pgn_moves = Vec::new();
    
    for (i, action) in game.actions().iter().enumerate() {
        if let chess::Action::MakeMove(chess_move) = action {
            // Convert to algebraic notation using the board state
            let san = move_to_san(&board, *chess_move);
            
            if i % 2 == 0 {
                pgn_moves.push(format!("{}. {}", (i / 2) + 1, san));
            } else {
                pgn_moves.push(san);
            }
            
            // Update board for next move
            board = board.make_move_new(*chess_move);
        }
    }
    
    // Print moves in lines of reasonable length (80 chars max)
    let mut line = String::new();
    for mv in pgn_moves {
        if line.len() + mv.len() + 1 > 80 {
            println!("{}", line.trim());
            line = String::new();
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(&mv);
    }
    if !line.is_empty() {
        println!("{}", line.trim());
    }
    println!("{}", result);
}

fn move_to_san(board: &Board, chess_move: ChessMove) -> String {
    // Simple conversion to algebraic notation
    // This is a basic implementation - could be enhanced with proper SAN
    let from = chess_move.get_source();
    let to = chess_move.get_dest();
    
    // Get piece on source square
    let piece = board.piece_on(from);
    let piece_char = match piece {
        Some(chess::Piece::King) => "K",
        Some(chess::Piece::Queen) => "Q", 
        Some(chess::Piece::Rook) => "R",
        Some(chess::Piece::Bishop) => "B",
        Some(chess::Piece::Knight) => "N",
        Some(chess::Piece::Pawn) => "",
        None => "",
    };
    
    // Check for captures
    let capture = if board.piece_on(to).is_some() { "x" } else { "" };
    
    // Format square names
    let from_square = square_to_algebraic(from);
    let to_square = square_to_algebraic(to);
    
    // Handle pawn captures (need file specification)
    if piece == Some(chess::Piece::Pawn) && !capture.is_empty() {
        format!("{}x{}", from_square.chars().next().unwrap(), to_square)
    } else if piece == Some(chess::Piece::Pawn) {
        to_square
    } else {
        format!("{}{}{}", piece_char, capture, to_square)
    }
}

fn square_to_algebraic(square: chess::Square) -> String {
    let file = (square.get_file().to_index() as u8 + b'a') as char;
    let rank = (square.get_rank().to_index() + 1).to_string();
    format!("{}{}", file, rank)
}

/// Save game positions and moves to database for future learning
fn save_game_to_database(engine: &mut ChessVectorEngine, game: &Game) {
    let mut board = Board::default();
    let moves = game.actions();
    
    // Determine game outcome for learning
    let final_board = game.current_position();
    let game_outcome = match final_board.status() {
        chess::BoardStatus::Checkmate => {
            if final_board.side_to_move() == Color::White {
                -1.0 // Black (Stockfish) won
            } else {
                1.0  // White (Vector Engine) won
            }
        }
        chess::BoardStatus::Stalemate => 0.0, // Draw
        _ => 0.0, // Other endings treated as draw
    };
    
    println!("    Analyzing {} moves for learning...", moves.len());
    
    // Process each position in the game
    for (i, action) in moves.iter().enumerate() {
        if let chess::Action::MakeMove(chess_move) = action {
            // Calculate position evaluation based on game outcome and move number
            // Early game moves are less decisive than late game moves
            let move_weight = (i as f32 + 1.0) / moves.len() as f32;
            let position_eval = game_outcome * move_weight * 0.1; // Scale down for training
            
            // Add position with the move and outcome
            let move_outcome = if i % 2 == 0 { 
                game_outcome  // White move
            } else { 
                -game_outcome // Black move (flip perspective)
            };
            
            engine.add_position_with_move(&board, position_eval, Some(*chess_move), Some(move_outcome));
            
            // Move to next position
            board = board.make_move_new(*chess_move);
        }
    }
    
    // Save to database
    match engine.save_to_database() {
        Ok(_) => println!("    ✅ Game data saved successfully"),
        Err(e) => println!("    ⚠️  Failed to save game data: {}", e),
    }
}