use chess::{Board, MoveGen};
use chess_vector_engine::ChessVectorEngine;
use rand::Rng;
use std::str::FromStr;

fn main() {
    println!("=== Chess Position Similarity Analysis ===");
    
    let engine = ChessVectorEngine::new(1024);
    
    // Test 1: Identical positions
    let board1 = Board::default();
    let board2 = Board::default();
    let sim1 = engine.calculate_similarity(&board1, &board2);
    println!("1. Identical positions: {:.6}", sim1);
    
    // Test 2: Starting position vs one move
    let board3 = Board::from_str("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1").unwrap();
    let sim2 = engine.calculate_similarity(&board1, &board3);
    println!("2. Starting vs e4: {:.6}", sim2);
    
    // Test 3: Starting position vs two moves
    let board4 = Board::from_str("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2").unwrap();
    let sim3 = engine.calculate_similarity(&board1, &board4);
    println!("3. Starting vs e4 e5: {:.6}", sim3);
    
    // Test 4: Completely different positions
    let board5 = Board::from_str("8/8/8/8/8/8/8/8 w - - 0 1").unwrap(); // Empty board
    let sim4 = engine.calculate_similarity(&board1, &board5);
    println!("4. Starting vs empty: {:.6}", sim4);
    
    // Test 5: Random positions
    println!("\n=== Random Position Similarities ===");
    let mut rng = rand::thread_rng();
    
    for i in 0..5 {
        let random_board1 = generate_random_position(&mut rng);
        let random_board2 = generate_random_position(&mut rng);
        let sim = engine.calculate_similarity(&random_board1, &random_board2);
        println!("Random pair {}: {:.6}", i + 1, sim);
    }
    
    // Test 6: Check actual vector values
    println!("\n=== Vector Analysis ===");
    let vec1 = engine.encode_position(&board1);
    let vec2 = engine.encode_position(&board3);
    let vec3 = engine.encode_position(&board5);
    
    println!("Starting position vector stats:");
    println!("  Non-zero values: {}", vec1.iter().filter(|&&x| x != 0.0).count());
    println!("  Sum: {:.6}", vec1.sum());
    println!("  Max: {:.6}", vec1.iter().fold(0.0f32, |a, &b| a.max(b)));
    println!("  Min: {:.6}", vec1.iter().fold(0.0f32, |a, &b| a.min(b)));
    
    println!("After e4 vector stats:");
    println!("  Non-zero values: {}", vec2.iter().filter(|&&x| x != 0.0).count());
    println!("  Sum: {:.6}", vec2.sum());
    println!("  Max: {:.6}", vec2.iter().fold(0.0f32, |a, &b| a.max(b)));
    println!("  Min: {:.6}", vec2.iter().fold(0.0f32, |a, &b| a.min(b)));
    
    println!("Empty board vector stats:");
    println!("  Non-zero values: {}", vec3.iter().filter(|&&x| x != 0.0).count());
    println!("  Sum: {:.6}", vec3.sum());
    println!("  Max: {:.6}", vec3.iter().fold(0.0f32, |a, &b| a.max(b)));
    println!("  Min: {:.6}", vec3.iter().fold(0.0f32, |a, &b| a.min(b)));
    
    // Vector norms
    println!("\nVector norms:");
    println!("  Starting position norm: {:.6}", vec1.dot(&vec1).sqrt());
    println!("  After e4 norm: {:.6}", vec2.dot(&vec2).sqrt());
    println!("  Empty board norm: {:.6}", vec3.dot(&vec3).sqrt());
}

fn generate_random_position(rng: &mut impl Rng) -> Board {
    let mut board = Board::default();
    let moves_to_play = rng.gen_range(0..15);
    
    for _ in 0..moves_to_play {
        let legal_moves: Vec<_> = MoveGen::new_legal(&board).collect();
        if legal_moves.is_empty() {
            break;
        }
        
        let random_move = legal_moves[rng.gen_range(0..legal_moves.len())];
        board = board.make_move_new(random_move);
    }
    
    board
}