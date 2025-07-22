use chess_vector_engine::{
    similarity_search::SimilaritySearch,
    PositionEncoder,
};
use chess::Board;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array1;
use std::str::FromStr;

/// Simplified benchmark suite for core functionality

fn benchmark_basic_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("basic_operations");
    
    // Test different vector sizes commonly used in the engine
    for size in [64, 256, 512, 1024] {
        let a = Array1::from_vec((0..size).map(|i| (i as f32).sin()).collect());
        let b = Array1::from_vec((0..size).map(|i| (i as f32).cos()).collect());
        
        group.bench_with_input(
            BenchmarkId::new("dot_product", size),
            &size,
            |bench, _| {
                bench.iter(|| black_box(a.dot(&b)))
            },
        );
    }
    
    group.finish();
}

fn benchmark_similarity_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity_search");
    
    // Create test data
    let mut search = SimilaritySearch::new(1024);
    
    // Add some test positions
    for i in 0..100 {
        let vector = Array1::from_vec(
            (0..1024).map(|j| ((i * 1024 + j) as f32).sin() * 0.1).collect()
        );
        search.add_position(vector, i as f32 * 0.001);
    }
    
    let query = Array1::from_vec((0..1024).map(|i| (i as f32).cos() * 0.1).collect());
    
    group.bench_function("search_k5", |bench| {
        bench.iter(|| black_box(search.search(&query, 5)))
    });
    
    group.bench_function("search_k10", |bench| {
        bench.iter(|| black_box(search.search(&query, 10)))
    });
    
    group.finish();
}

fn benchmark_position_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("position_encoding");
    
    let encoder = PositionEncoder::new(1024);
    let board = Board::default();
    
    group.bench_function("encode_default_position", |bench| {
        bench.iter(|| black_box(encoder.encode(&board)))
    });
    
    // Test with different positions
    let positions = [
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
        "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 2 3",
    ];
    
    for (i, fen) in positions.iter().enumerate() {
        if let Ok(board) = Board::from_str(fen) {
            group.bench_with_input(
                BenchmarkId::new("encode_position", i),
                &board,
                |bench, board| {
                    bench.iter(|| black_box(encoder.encode(board)))
                },
            );
        }
    }
    
    group.finish();
}

criterion_group!(benches, benchmark_basic_operations, benchmark_similarity_search, benchmark_position_encoding);
criterion_main!(benches);