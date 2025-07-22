use chess::Board;
use chess_vector_engine::{PositionEncoder, SimilaritySearch};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array1;
use std::str::FromStr;

fn bench_similarity_search_methods(c: &mut Criterion) {
    let encoder = PositionEncoder::new(1024);
    let mut search = SimilaritySearch::new(1024);
    
    // Create test dataset
    let test_positions = vec![
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/3P1N2/PPP2PPP/RNBQKB1R b KQkq - 0 3",
    ];
    
    // Add positions to search index
    for (i, fen) in test_positions.iter().enumerate() {
        let board = Board::from_str(fen).unwrap();
        let vector = encoder.encode(&board);
        search.add_position(vector, i as f32 * 0.1);
    }
    
    // Create larger dataset for more realistic benchmarking
    for i in 0..100 {
        let mut variations = Vec::new();
        for base_fen in &test_positions {
            for _ in 0..10 {
                let board = Board::from_str(base_fen).unwrap();
                let vector = encoder.encode(&board);
                // Add small random variations
                let modified_vector = vector.map(|x| x + (i as f32 * 0.001));
                variations.push((modified_vector, (i as f32 + 0.5) * 0.01));
            }
        }
        
        for (vector, eval) in variations {
            search.add_position(vector, eval);
        }
    }
    
    let query_board = Board::from_str("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
    let query_vector = encoder.encode(&query_board);
    
    println!("Dataset size: {} positions", search.size());
    
    let mut group = c.benchmark_group("similarity_search_methods");
    
    // Benchmark different search methods
    group.bench_function("search_optimized", |b| {
        b.iter(|| black_box(search.search_optimized(&query_vector, 10)))
    });
    
    group.bench_function("search_standard", |b| {
        b.iter(|| black_box(search.sequential_search(&query_vector, 10)))
    });
    
    group.bench_function("search_parallel", |b| {
        b.iter(|| black_box(search.parallel_search(&query_vector, 10)))
    });
    
    group.finish();
}

fn bench_search_with_different_k_values(c: &mut Criterion) {
    let encoder = PositionEncoder::new(1024);
    let mut search = SimilaritySearch::new(1024);
    
    // Create test dataset
    let base_positions = vec![
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
        "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    ];
    
    // Add 1000 variations
    for i in 0..200 {
        for base_fen in &base_positions {
            let board = Board::from_str(base_fen).unwrap();
            let mut vector = encoder.encode(&board);
            // Add variations
            for j in 0..vector.len() {
                vector[j] += (i as f32 * 0.001) + (j as f32 * 0.0001);
            }
            search.add_position(vector, i as f32 * 0.01);
        }
    }
    
    let query_board = Board::from_str("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
    let query_vector = encoder.encode(&query_board);
    
    println!("Large dataset size: {} positions", search.size());
    
    let mut group = c.benchmark_group("search_k_values");
    
    for k in [1, 5, 10, 20, 50, 100] {
        group.bench_with_input(
            BenchmarkId::new("optimized_search", k),
            &k,
            |b, &k| {
                b.iter(|| black_box(search.search_optimized(&query_vector, k)))
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("parallel_search", k),
            &k,
            |b, &k| {
                b.iter(|| black_box(search.parallel_search(&query_vector, k)))
            },
        );
    }
    
    group.finish();
}

fn bench_batch_search(c: &mut Criterion) {
    let encoder = PositionEncoder::new(1024);
    let mut search = SimilaritySearch::new(1024);
    
    // Create dataset
    let base_positions = vec![
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
    ];
    
    for i in 0..100 {
        for base_fen in &base_positions {
            let board = Board::from_str(base_fen).unwrap();
            let vector = encoder.encode(&board);
            search.add_position(vector, i as f32 * 0.01);
        }
    }
    
    // Create query batch
    let query_vectors: Vec<Array1<f32>> = base_positions
        .iter()
        .map(|fen| {
            let board = Board::from_str(fen).unwrap();
            encoder.encode(&board)
        })
        .collect();
    
    c.bench_function("batch_search_optimized", |b| {
        b.iter(|| black_box(search.batch_search_optimized(&query_vectors, 10)))
    });
}

criterion_group!(
    benches,
    bench_similarity_search_methods,
    bench_search_with_different_k_values,
    bench_batch_search
);
criterion_main!(benches);