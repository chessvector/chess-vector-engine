use chess::{Board, ChessMove, Color, Game, MoveGen};
use chess_vector_engine::ChessVectorEngine;
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::str::FromStr;
use rand::seq::SliceRandom;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting game: Chess Vector Engine vs Stockfish (Improved Version)");
    
    // Initialize the chess vector engine with opening book and substantial knowledge base
    let engine = create_engine_with_knowledge();
    
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
    let opening_move = choose_random_opening(&engine);
    
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
            let engine_move = get_engine_move_smart(&engine, &current_board)?;
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
    
    // Clean up Stockfish
    writeln!(stockfish_stdin, "quit")?;
    stockfish.wait()?;
    
    Ok(())
}

fn create_engine_with_knowledge() -> ChessVectorEngine {
    // Initialize with intelligent architecture selection based on dataset size
    let mut initial_positions = 0;
    
    // Check training data size first to determine architecture
    if let Ok(dataset) = chess_vector_engine::TrainingDataset::load("training_data.json") {
        initial_positions = dataset.data.len();
    }
    
    let mut engine = if initial_positions > 15000 {
        println!("🚀 Large knowledge base detected ({} positions), using LSH for optimal performance", initial_positions);
        // Optimized LSH parameters for gameplay (faster search, good recall)
        ChessVectorEngine::new_with_lsh(1024, 10, 18)
    } else {
        println!("🚀 Standard knowledge base size ({} positions), using linear search", initial_positions);
        ChessVectorEngine::new(1024)
    };
    
    // Enable opening book with expanded openings
    engine.enable_opening_book();
    
    // Load training data if available
    match chess_vector_engine::TrainingDataset::load("training_data.json") {
        Ok(dataset) => {
            println!("Loading {} positions from training_data.json", dataset.data.len());
            for training_data in dataset.data {
                engine.add_position(&training_data.board, training_data.evaluation);
            }
        }
        Err(e) => {
            println!("Failed to load training_data.json: {}", e);
        }
    }
    
    // Let the engine rely purely on opening book and vector similarity
    add_position_knowledge(&mut engine);
    
    let total_positions = engine.knowledge_base_size();
    println!("Engine initialized with {} positions in knowledge base", total_positions);
    if let Some(stats) = engine.opening_book_stats() {
        println!("Opening book contains {} positions with {} ECO classifications", 
                 stats.total_positions, stats.eco_classifications);
    }
    
    // Enable manifold learning for very large datasets during gameplay
    if total_positions > 30000 {
        println!("🧠 Very large knowledge base, enabling manifold learning for faster similarity search...");
        let _ = engine.enable_manifold_learning(6.0); // 6:1 compression for speed (1024d -> ~170d)
        
        println!("🏋️  Training manifold compression for gameplay optimization...");
        let _ = engine.train_manifold_learning(8); // Fewer epochs for faster startup
        
        println!("✅ Manifold learning enabled - optimized for real-time gameplay");
    }
    
    engine
}

fn add_position_knowledge(_engine: &mut ChessVectorEngine) {
    // Let the vector system learn patterns naturally through position similarity
    // The engine will use its opening book and similarity search to discover
    // tactical patterns, endgame principles, and positional concepts organically
    // based on the encoded position vectors and their inherent relationships
}

fn choose_random_opening(engine: &ChessVectorEngine) -> Option<ChessMove> {
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

fn get_engine_move_smart(engine: &ChessVectorEngine, board: &Board) -> Result<ChessMove, Box<dyn std::error::Error>> {
    // THIS IS THE CORRECT WAY - Use the engine's sophisticated recommendation system
    let recommendations = engine.recommend_legal_moves(board, 5);
    
    if recommendations.is_empty() {
        // Fallback: generate any legal move
        let legal_moves: Vec<ChessMove> = MoveGen::new_legal(board).collect();
        if legal_moves.is_empty() {
            return Err("No legal moves available".into());
        }
        return Ok(legal_moves[0]);
    }
    
    // Debug output to see what the engine is thinking
    if recommendations.len() > 1 {
        println!("    Engine considering: {} moves", recommendations.len());
        for (i, rec) in recommendations.iter().take(3).enumerate() {
            println!("      {}. {} (confidence: {:.2}, from {} similar positions, avg outcome: {:.2})", 
                     i + 1, rec.chess_move, rec.confidence, rec.from_similar_position_count, rec.average_outcome);
        }
    }
    
    // Select move based on confidence, with some randomness for variety
    let best_move = if recommendations[0].confidence > 0.6 {
        // High confidence - take the best move
        recommendations[0].chess_move
    } else if recommendations.len() > 1 && recommendations[1].confidence > 0.3 {
        // Medium confidence - choose between top 2 moves randomly
        let top_two = &recommendations[0..2.min(recommendations.len())];
        top_two.choose(&mut rand::thread_rng()).unwrap().chess_move
    } else {
        // Low confidence - add some randomness among top moves
        let top_moves = &recommendations[0..3.min(recommendations.len())];
        top_moves.choose(&mut rand::thread_rng()).unwrap().chess_move
    };
    
    Ok(best_move)
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