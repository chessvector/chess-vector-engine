use chess::{Board, MoveGen};
use chess_vector_engine::ChessVectorEngine;
use rand::Rng;
use std::time::Instant;

fn main() {
    println!("Chess Vector Engine Benchmark");
    println!("=============================");

    // Use LSH-enabled engine for realistic performance testing
    let mut engine = ChessVectorEngine::new(1024);
    engine.enable_lsh(8, 16); // Enable LSH with reasonable parameters
    let mut rng = rand::thread_rng();

    // Benchmark 1: Position encoding speed
    println!("\n=== Position Encoding Benchmark ===");
    let board = Board::default();
    let iterations = 10_000;

    let start = Instant::now();
    for _ in 0..iterations {
        let _vector = engine.encode_position(&board);
    }
    let encoding_time = start.elapsed();

    println!("Encoded {} positions in {:?}", iterations, encoding_time);
    println!(
        "Average encoding time: {:?} per position",
        encoding_time / iterations
    );
    println!(
        "Encoding rate: {:.0} positions/second",
        iterations as f64 / encoding_time.as_secs_f64()
    );

    // Benchmark 2: Knowledge base scaling
    println!("\n=== Knowledge Base Scaling Benchmark ===");
    let test_sizes = vec![100, 500, 1000];

    for size in test_sizes {
        let mut test_engine = ChessVectorEngine::new(1024);
        test_engine.enable_lsh(8, 16); // Enable LSH for scaling tests

        // Add random positions
        println!("Building knowledge base with {} positions...", size);
        let build_start = Instant::now();

        for _ in 0..size {
            let board = generate_random_position(&mut rng).expect("Valid random position");
            let eval = rng.gen_range(-2.0..2.0);
            test_engine.add_position(&board, eval);
        }

        // Show LSH stats if enabled
        if test_engine.is_lsh_enabled() {
            if let Some(stats) = test_engine.lsh_stats() {
                println!(
                    "  LSH Stats - Tables: {}, Hash size: {}, Avg bucket: {:.1}",
                    stats.num_tables, stats.hash_size, stats.avg_bucket_size
                );
            }
        }
        let build_time = build_start.elapsed();

        // Test search speed
        let query_board = Board::default();
        let search_start = Instant::now();
        let results = test_engine.find_similar_positions(&query_board, 10);
        let search_time = search_start.elapsed();

        println!(
            "  Size: {}, Build: {:?}, Search: {:?}, Found: {} results",
            size,
            build_time,
            search_time,
            results.len()
        );
        println!(
            "  Search rate: {:.0} queries/second",
            1.0 / search_time.as_secs_f64()
        );
    }

    // Benchmark 3: Similarity calculation accuracy
    println!("\n=== Similarity Accuracy Test ===");
    test_similarity_accuracy(&engine);

    println!("\n=== Benchmark Complete ===");
}

/// Generate a semi-random chess position by playing random moves
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

/// Test similarity calculation accuracy with known similar/dissimilar positions
fn test_similarity_accuracy(engine: &ChessVectorEngine) {
    use std::str::FromStr;

    // Test cases: positions that should be similar vs dissimilar
    let similar_pairs = vec![
        // Starting position vs 1.e4
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        ),
        // 1.e4 e5 vs 1.d4 d5 (symmetric openings)
        (
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
            "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2",
        ),
    ];

    let dissimilar_pairs = vec![
        // Opening vs endgame
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
        ),
        // Opening vs middlegame with major piece differences
        (
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQ - 4 6",
        ),
    ];

    println!("Testing similarity on known similar pairs:");
    for (pos1, pos2) in similar_pairs {
        let board1 = Board::from_str(pos1).expect("Valid FEN");
        let board2 = Board::from_str(pos2).expect("Valid FEN");
        let similarity = engine.calculate_similarity(&board1, &board2);
        println!("  Similarity: {:.3} (should be high)", similarity);
    }

    println!("Testing similarity on known dissimilar pairs:");
    for (pos1, pos2) in dissimilar_pairs {
        let board1 = Board::from_str(pos1).expect("Valid FEN");
        let board2 = Board::from_str(pos2).expect("Valid FEN");
        let similarity = engine.calculate_similarity(&board1, &board2);
        println!("  Similarity: {:.3} (should be low)", similarity);
    }
}
