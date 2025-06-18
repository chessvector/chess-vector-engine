use chess::{Board, ChessMove, Game, Square};
use std::str::FromStr;

fn main() {
    let moves = vec![
        "e2e4", "e7e6", "d2d4", "a7a6", "f2f4", "d7d5", "e4d5", "e6d5",
        "g1f3", "g8f6", "f3e5", "c7c5", "b1c3", "c5d4", "d1d4", "f8e7",
        "c3d5", "d8d5", "d4d5", "f6d5", "f1b5", "a6b5", "c2c4", "d5b4",
        "a2a4", "b4c2", "e1e2", "c2a1", "f4f5", "a8a4", "c4c5", "f7f6",
        "c1h6", "a1b3", "h1b1", "g7h6", "c5c6", "a4e4", "e2d1", "e4e5",
        "b1a1", "b3a1", "d1d2", "e7b4", "d2d1", "e5e1"
    ];
    
    let mut game = Game::new();
    let mut pgn_moves = Vec::new();
    
    for (i, move_str) in moves.iter().enumerate() {
        let current_board = game.current_position();
        
        if let Ok(chess_move) = ChessMove::from_str(move_str) {
            // Convert to standard algebraic notation
            let san = move_to_san(&current_board, chess_move);
            
            if i % 2 == 0 {
                pgn_moves.push(format!("{}.", (i / 2) + 1));
                pgn_moves.push(san);
            } else {
                pgn_moves.push(san);
            }
            
            game.make_move(chess_move);
        }
    }
    
    println!("Game Result: Stockfish wins by checkmate!");
    println!("\nPGN Format:");
    println!("[Event \"Vector Engine vs Stockfish\"]");
    println!("[Site \"Local\"]");
    println!("[Date \"2024.01.01\"]");
    println!("[Round \"1\"]");
    println!("[White \"Chess Vector Engine\"]");
    println!("[Black \"Stockfish\"]");
    println!("[Result \"0-1\"]");
    println!("");
    
    let mut line = String::new();
    for (_i, mv) in pgn_moves.iter().enumerate() {
        if line.len() + mv.len() > 80 {
            println!("{}", line);
            line = String::new();
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(mv);
    }
    if !line.is_empty() {
        println!("{}", line);
    }
    println!("0-1");
}

fn move_to_san(board: &Board, chess_move: ChessMove) -> String {
    let piece = board.piece_on(chess_move.get_source());
    let dest = chess_move.get_dest();
    let source = chess_move.get_source();
    
    if let Some(piece) = piece {
        match piece {
            chess::Piece::Pawn => {
                if board.piece_on(dest).is_some() {
                    // Capture
                    format!("{}x{}", 
                        square_to_file(source), 
                        square_to_algebraic(dest))
                } else {
                    // Normal pawn move
                    square_to_algebraic(dest)
                }
            },
            chess::Piece::King => {
                // Check for castling
                if source == Square::E1 && dest == Square::G1 {
                    "O-O".to_string()
                } else if source == Square::E1 && dest == Square::C1 {
                    "O-O-O".to_string()
                } else if source == Square::E8 && dest == Square::G8 {
                    "O-O".to_string()
                } else if source == Square::E8 && dest == Square::C8 {
                    "O-O-O".to_string()
                } else {
                    let capture = if board.piece_on(dest).is_some() { "x" } else { "" };
                    format!("K{}{}", capture, square_to_algebraic(dest))
                }
            },
            _ => {
                let piece_symbol = match piece {
                    chess::Piece::Queen => "Q",
                    chess::Piece::Rook => "R",
                    chess::Piece::Bishop => "B",
                    chess::Piece::Knight => "N",
                    _ => "",
                };
                let capture = if board.piece_on(dest).is_some() { "x" } else { "" };
                format!("{}{}{}", piece_symbol, capture, square_to_algebraic(dest))
            }
        }
    } else {
        format!("{}-{}", square_to_algebraic(source), square_to_algebraic(dest))
    }
}

fn square_to_algebraic(square: Square) -> String {
    let file = (square.to_index() % 8) as u8 + b'a';
    let rank = (square.to_index() / 8) as u8 + b'1';
    format!("{}{}", file as char, rank as char)
}

fn square_to_file(square: Square) -> char {
    ((square.to_index() % 8) as u8 + b'a') as char
}