use chess_vector_engine::core_evaluation::{CoreEvaluator, CacheStats};
use chess::Board;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::str::FromStr;

fn benchmark_core_evaluation(c: &mut Criterion) {
    // Test positions of varying complexity
    let test_positions = vec![
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", // Starting position
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2", // King's pawn opening
        "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 3", // Italian game
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 4", // Middle game
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", // Endgame position
    ];

    let boards: Vec<Board> = test_positions
        .iter()
        .map(|fen| Board::from_str(fen).expect("Valid FEN"))
        .collect();

    // Benchmark without optimizations (fresh evaluator each time)
    c.bench_function("core_evaluation_without_cache", |b| {
        b.iter(|| {
            let mut evaluator = CoreEvaluator::new();
            
            // Add some training data
            for (i, board) in boards.iter().enumerate() {
                evaluator.learn_from_position(board, i as f32 * 0.1);
            }
            
            // Evaluate all positions
            for board in &boards {
                black_box(evaluator.evaluate_position(board));
            }
        })
    });

    // Benchmark with optimizations (cached evaluator)
    c.bench_function("core_evaluation_with_cache", |b| {
        let mut evaluator = CoreEvaluator::new_with_cache_config(1000, 60);
        
        // Pre-populate with training data
        for (i, board) in boards.iter().enumerate() {
            evaluator.learn_from_position(board, i as f32 * 0.1);
        }
        
        b.iter(|| {
            // Evaluate all positions (should benefit from caching)
            for board in &boards {
                black_box(evaluator.evaluate_position(&board));
            }
        })
    });

    // Benchmark cache performance specifically
    c.bench_function("cache_hit_performance", |b| {
        let mut evaluator = CoreEvaluator::new_with_cache_config(1000, 60);
        
        // Pre-populate with training data and cache
        for (i, board) in boards.iter().enumerate() {
            evaluator.learn_from_position(board, i as f32 * 0.1);
            evaluator.evaluate_position(board); // Prime the cache
        }
        
        b.iter(|| {
            // These should all be cache hits
            for board in &boards {
                black_box(evaluator.evaluate_position(&board));
            }
        })
    });

    // Benchmark position encoding performance
    c.bench_function("position_encoding", |b| {
        let evaluator = CoreEvaluator::new();
        
        b.iter(|| {
            for board in &boards {
                // This tests the optimized position encoder
                let similarity_insights = evaluator.similarity_engine.find_strategic_insights(black_box(board));
                black_box(similarity_insights);
            }
        })
    });
}

fn benchmark_cache_management(c: &mut Criterion) {
    let boards: Vec<Board> = (0..100)
        .map(|i| {
            // Create varied positions by making random legal moves
            let fen = format!("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKB{}R w KQkq - 0 1", i % 2);
            Board::from_str(&fen).unwrap_or_else(|_| Board::default())
        })
        .collect();

    c.bench_function("cache_with_eviction", |b| {
        b.iter(|| {
            let mut evaluator = CoreEvaluator::new_with_cache_config(50, 1); // Small cache, short TTL
            
            // Force cache evictions by exceeding cache size
            for (i, board) in boards.iter().enumerate() {
                evaluator.learn_from_position(board, i as f32 * 0.01);
                black_box(evaluator.evaluate_position(board));
            }
            
            let stats = evaluator.get_cache_stats();
            black_box(stats);
        })
    });

    c.bench_function("cache_statistics", |b| {
        let mut evaluator = CoreEvaluator::new_with_cache_config(1000, 60);
        
        // Populate cache
        for (i, board) in boards.iter().enumerate() {
            evaluator.learn_from_position(board, i as f32 * 0.01);
            evaluator.evaluate_position(board);
        }
        
        b.iter(|| {
            let stats = evaluator.get_cache_stats();
            black_box(stats);
        })
    });
}

criterion_group!(
    benches,
    benchmark_core_evaluation,
    benchmark_cache_management
);
criterion_main!(benches);