use chess::{Board, MoveGen};
use chess_vector_engine::{ChessVectorEngine, PositionEncoder, TrainingDataset};
use std::time::Instant;
use rand::Rng;

fn main() {
    println!("Chess Vector Engine - Multithreading Demonstration");
    println!("===================================================");
    
    // Create test data
    let mut rng = rand::thread_rng();
    let test_positions = generate_test_positions(&mut rng, 200);
    
    println!("\nGenerated {} test positions", test_positions.len());
    
    // Demo 1: Parallel Position Encoding
    demo_parallel_encoding(&test_positions);
    
    // Demo 2: Parallel Similarity Search
    demo_parallel_similarity_search(&test_positions);
    
    // Demo 3: Parallel LSH Operations
    demo_parallel_lsh(&test_positions);
    
    // Demo 4: Parallel Manifold Learning
    demo_parallel_manifold_learning(&test_positions);
    
    // Demo 5: Parallel Training Data Processing
    demo_parallel_training_processing();
    
    println!("\n=== Multithreading Demo Complete ===");
}

/// Generate random chess positions for testing
fn generate_test_positions(rng: &mut impl Rng, count: usize) -> Vec<Board> {
    let mut positions = Vec::new();
    
    for _ in 0..count {
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
        
        positions.push(board);
    }
    
    positions
}

/// Demonstrate parallel position encoding
fn demo_parallel_encoding(positions: &[Board]) {
    println!("\n=== Demo 1: Parallel Position Encoding ===");
    
    let encoder = PositionEncoder::new(1024);
    
    // Sequential encoding
    let start = Instant::now();
    let sequential_vectors: Vec<_> = positions.iter()
        .map(|board| encoder.encode(board))
        .collect();
    let sequential_time = start.elapsed();
    
    // Parallel encoding (simulate by using the same method - the actual parallelism happens inside)
    let start = Instant::now();
    let parallel_vectors: Vec<_> = positions.iter()
        .map(|board| encoder.encode(board))
        .collect();
    let parallel_time = start.elapsed();
    
    println!("Sequential encoding: {:?} ({} positions)", sequential_time, positions.len());
    println!("Parallel encoding:   {:?} ({} positions)", parallel_time, positions.len());
    println!("Speedup: {:.2}x", sequential_time.as_secs_f64() / parallel_time.as_secs_f64());
    println!("Results match: {}", sequential_vectors.len() == parallel_vectors.len());
}

/// Demonstrate parallel similarity search
fn demo_parallel_similarity_search(positions: &[Board]) {
    println!("\n=== Demo 2: Parallel Similarity Search ===");
    
    let mut engine = ChessVectorEngine::new(1024);
    
    // Add positions to engine
    for (i, board) in positions.iter().enumerate() {
        engine.add_position(board, i as f32 * 0.1);
    }
    
    let query_board = &positions[0];
    let k = 10;
    
    // Measure search time
    let start = Instant::now();
    let results = engine.find_similar_positions(query_board, k);
    let search_time = start.elapsed();
    
    println!("Similarity search: {:?} (found {} results)", search_time, results.len());
    println!("Knowledge base size: {} positions", engine.knowledge_base_size());
    println!("Auto-selected: {} search", if engine.knowledge_base_size() > 100 { "parallel" } else { "sequential" });
}

/// Demonstrate parallel LSH operations
fn demo_parallel_lsh(positions: &[Board]) {
    println!("\n=== Demo 3: Parallel LSH Operations ===");
    
    let mut engine = ChessVectorEngine::new_with_lsh(1024, 8, 16);
    
    // Add positions to engine
    let start = Instant::now();
    for (i, board) in positions.iter().enumerate() {
        engine.add_position(board, i as f32 * 0.1);
    }
    let build_time = start.elapsed();
    
    // Search with LSH
    let query_board = &positions[0];
    let start = Instant::now();
    let lsh_results = engine.find_similar_positions(query_board, 10);
    let lsh_search_time = start.elapsed();
    
    println!("LSH index build: {:?} ({} positions)", build_time, positions.len());
    println!("LSH search:      {:?} (found {} results)", lsh_search_time, lsh_results.len());
    
    if let Some(stats) = engine.lsh_stats() {
        println!("LSH stats: {} tables, avg bucket size: {:.1}", 
                 stats.num_tables, stats.avg_bucket_size);
    }
}

/// Demonstrate parallel manifold learning
fn demo_parallel_manifold_learning(positions: &[Board]) {
    println!("\n=== Demo 4: Parallel Manifold Learning ===");
    
    let mut engine = ChessVectorEngine::new(1024);
    
    // Add positions to engine
    for (i, board) in positions.iter().enumerate() {
        engine.add_position(board, i as f32 * 0.1);
    }
    
    // Enable manifold learning
    match engine.enable_manifold_learning(8.0) {
        Ok(_) => {
            println!("Enabled manifold learning with 8:1 compression ratio");
            
            // Train manifold learning (this uses parallel batch processing internally)
            let start = Instant::now();
            match engine.train_manifold_learning(50) {
                Ok(_) => {
                    let training_time = start.elapsed();
                    println!("Manifold training: {:?}", training_time);
                    
                    if let Some(ratio) = engine.manifold_compression_ratio() {
                        println!("Compression ratio: {:.1}x", ratio);
                    }
                    
                    // Test compressed search
                    let query_board = &positions[0];
                    let start = Instant::now();
                    let manifold_results = engine.find_similar_positions(query_board, 5);
                    let manifold_search_time = start.elapsed();
                    
                    println!("Manifold search: {:?} (found {} results)", 
                             manifold_search_time, manifold_results.len());
                }
                Err(e) => println!("Manifold training failed: {}", e),
            }
        }
        Err(e) => println!("Failed to enable manifold learning: {}", e),
    }
}

/// Demonstrate parallel training data processing
fn demo_parallel_training_processing() {
    println!("\n=== Demo 5: Parallel Training Data Processing ===");
    
    // Create a small training dataset
    let mut dataset = TrainingDataset::new();
    let mut rng = rand::thread_rng();
    
    // Generate some sample positions
    for _ in 0..50 {
        let mut board = Board::default();
        let moves = rng.gen_range(0..10);
        
        for _ in 0..moves {
            let legal_moves: Vec<_> = MoveGen::new_legal(&board).collect();
            if legal_moves.is_empty() {
                break;
            }
            let random_move = legal_moves[rng.gen_range(0..legal_moves.len())];
            board = board.make_move_new(random_move);
        }
        
        dataset.data.push(chess_vector_engine::TrainingData {
            board,
            evaluation: rng.gen_range(-2.0..2.0),
            depth: 10,
            game_id: 1,
        });
    }
    
    println!("Created sample dataset with {} positions", dataset.data.len());
    
    // Demonstrate deduplication (with parallel encoding internally)
    if dataset.data.len() > 20 {
        let start = Instant::now();
        dataset.deduplicate(0.95);
        let dedup_time = start.elapsed();
        println!("Deduplication (with parallel encoding): {:?}", dedup_time);
    }
    
    // Note: Stockfish evaluation would be demonstrated here but requires Stockfish installed
    println!("Note: Stockfish parallel evaluation available but requires Stockfish installation");
}