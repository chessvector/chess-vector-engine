use chess::Board;
use chess_vector_engine::PositionEncoder;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::str::FromStr;

fn bench_position_encoding(c: &mut Criterion) {
    let encoder = PositionEncoder::new(1024);
    
    // Test positions with varying complexity
    let test_positions = vec![
        ("Starting Position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Middle Game", "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
        ("Complex Position", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"),
        ("Endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
    ];
    
    let mut group = c.benchmark_group("position_encoding");
    
    for (name, fen) in test_positions {
        let board = Board::from_str(fen).unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("standard_encoding", name),
            &board,
            |b, board| {
                b.iter(|| black_box(encoder.encode(board)))
            },
        );
    }
    
    group.finish();
}

fn bench_batch_encoding(c: &mut Criterion) {
    let encoder = PositionEncoder::new(1024);
    
    // Create batches of different sizes
    let batch_sizes = vec![1, 5, 10, 20, 50, 100];
    let base_position = Board::from_str("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
    
    let mut group = c.benchmark_group("batch_encoding");
    
    for batch_size in batch_sizes {
        let positions: Vec<Board> = (0..batch_size).map(|_| base_position.clone()).collect();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &positions,
            |b, positions| {
                b.iter(|| black_box(encoder.encode_batch(positions)))
            },
        );
    }
    
    group.finish();
}

fn bench_similarity_calculations(c: &mut Criterion) {
    let encoder = PositionEncoder::new(1024);
    let board1 = Board::from_str("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
    let board2 = Board::from_str("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1").unwrap();
    
    let vec1 = encoder.encode(&board1);
    let vec2 = encoder.encode(&board2);
    
    let mut group = c.benchmark_group("similarity_calculation");
    
    group.bench_function("cosine_similarity", |b| {
        b.iter(|| black_box(encoder.similarity(&vec1, &vec2)))
    });
    
    group.bench_function("euclidean_distance", |b| {
        b.iter(|| black_box(encoder.distance(&vec1, &vec2)))
    });
    
    // Batch similarity testing
    let vectors: Vec<_> = (0..50).map(|_| vec2.clone()).collect();
    
    group.bench_function("batch_similarity_50", |b| {
        b.iter(|| black_box(encoder.batch_similarity(&vec1, &vectors)))
    });
    
    group.finish();
}

fn bench_memory_usage(c: &mut Criterion) {
    let encoder = PositionEncoder::new(1024);
    let board = Board::from_str("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
    
    c.bench_function("memory_allocation_overhead", |b| {
        b.iter(|| {
            // This tests the memory allocation overhead of creating new vectors
            let vec = black_box(encoder.encode(&board));
            drop(vec);
        })
    });
}

criterion_group!(
    benches,
    bench_position_encoding,
    bench_batch_encoding,
    bench_similarity_calculations,
    bench_memory_usage
);
criterion_main!(benches);