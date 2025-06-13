use chess::{Board, MoveGen};
use chess_vector_engine::ChessVectorEngine;
use std::time::Instant;
use rand::Rng;

fn main() {
    println!("LSH vs Linear Search Benchmark");
    println!("==============================");
    
    let sizes = vec![1000, 5000];
    
    for &size in &sizes {
        println!("\n=== Testing with {} positions ===", size);
        
        // Test linear search
        println!("Building linear search index...");
        let mut linear_engine = ChessVectorEngine::new(1024);
        let build_start = Instant::now();
        populate_engine(&mut linear_engine, size);
        let linear_build_time = build_start.elapsed();
        
        // Test LSH search
        println!("Building LSH index...");
        let mut lsh_engine = ChessVectorEngine::new_with_lsh(1024, 8, 20);
        let build_start = Instant::now();
        populate_engine(&mut lsh_engine, size);
        let lsh_build_time = build_start.elapsed();
        
        // Test query performance
        let query_board = Board::default();
        let k = 10;
        let num_queries = 50;
        
        // Linear search queries
        let query_start = Instant::now();
        for _ in 0..num_queries {
            let _results = linear_engine.find_similar_positions(&query_board, k);
        }
        let linear_query_time = query_start.elapsed();
        let linear_qps = num_queries as f32 / linear_query_time.as_secs_f32();
        
        // LSH queries
        let query_start = Instant::now();
        for _ in 0..num_queries {
            let _results = lsh_engine.find_similar_positions(&query_board, k);
        }
        let lsh_query_time = query_start.elapsed();
        let lsh_qps = num_queries as f32 / lsh_query_time.as_secs_f32();
        
        // Compare result quality
        let linear_results = linear_engine.find_similar_positions(&query_board, k);
        let lsh_results = lsh_engine.find_similar_positions(&query_board, k);
        
        let recall = calculate_recall(&linear_results, &lsh_results);
        
        // Print results
        println!("Size: {}", size);
        println!("Linear  - Build: {:?}, Query: {:.1} qps", linear_build_time, linear_qps);
        println!("LSH     - Build: {:?}, Query: {:.1} qps, Recall: {:.2}", 
                 lsh_build_time, lsh_qps, recall);
        println!("Speedup: {:.1}x", lsh_qps / linear_qps);
        
        // Show LSH stats
        if let Some(stats) = lsh_engine.lsh_stats() {
            println!("LSH Stats - Tables: {}, Hash size: {}, Avg bucket size: {:.1}", 
                     stats.num_tables, stats.hash_size, stats.avg_bucket_size);
        }
    }
}

fn populate_engine(engine: &mut ChessVectorEngine, size: usize) {
    let mut rng = rand::thread_rng();
    
    for _ in 0..size {
        let board = generate_random_position(&mut rng).expect("Valid random position");
        let eval = rng.gen_range(-2.0..2.0);
        engine.add_position(&board, eval);
    }
}

fn generate_random_position(rng: &mut impl Rng) -> Result<Board, Box<dyn std::error::Error>> {
    let mut board = Board::default();
    let moves_to_play = rng.gen_range(0..20);
    
    for _ in 0..moves_to_play {
        let legal_moves: Vec<_> = MoveGen::new_legal(&board).collect();
        if legal_moves.is_empty() {
            break;
        }
        
        let random_move = legal_moves[rng.gen_range(0..legal_moves.len())];
        board = board.make_move_new(random_move);
    }
    
    Ok(board)
}

fn calculate_recall(linear_results: &[(ndarray::Array1<f32>, f32, f32)], 
                   lsh_results: &[(ndarray::Array1<f32>, f32, f32)]) -> f32 {
    if linear_results.is_empty() {
        return 1.0;
    }
    
    // Simple recall: check how many of the top LSH results are also in top linear results
    // We'll use similarity threshold for approximate matching since floating point comparison is tricky
    let mut matches = 0;
    
    for (lsh_vec, lsh_eval, _) in lsh_results {
        for (lin_vec, lin_eval, _) in linear_results {
            // Check if vectors are approximately equal and evaluations match
            let vec_diff: f32 = (lsh_vec - lin_vec).mapv(|x| x * x).sum().sqrt();
            if vec_diff < 1e-6 && (lsh_eval - lin_eval).abs() < 1e-6 {
                matches += 1;
                break;
            }
        }
    }
    
    if lsh_results.is_empty() {
        1.0
    } else {
        matches as f32 / lsh_results.len().min(linear_results.len()) as f32
    }
}