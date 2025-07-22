use chess_vector_engine::similarity_search::SimilaritySearch;
use ndarray::Array1;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn generate_test_vectors(vector_size: usize, count: usize) -> Vec<Array1<f32>> {
    (0..count)
        .map(|i| {
            let mut vector = Array1::zeros(vector_size);
            for j in 0..vector_size {
                vector[j] = ((i * vector_size + j) as f32).sin() * 0.1;
            }
            vector
        })
        .collect()
}

fn benchmark_similarity_search_caching(c: &mut Criterion) {
    let vector_size = 1024;
    let dataset_size = 1000;
    let query_count = 50;
    
    // Generate test data
    let test_vectors = generate_test_vectors(vector_size, dataset_size);
    let query_vectors = generate_test_vectors(vector_size, query_count);
    
    // Benchmark without caching (fresh engine each time)
    c.bench_function("similarity_search_without_cache", |b| {
        b.iter(|| {
            let mut engine = SimilaritySearch::new(vector_size);
            
            // Add positions to search engine
            for (i, vector) in test_vectors.iter().enumerate() {
                engine.add_position(vector.clone(), i as f32 * 0.01);
            }
            
            // Perform searches
            for query in &query_vectors {
                black_box(engine.search(query, 5));
            }
        })
    });
    
    // Benchmark with caching (reuse same engine)
    c.bench_function("similarity_search_with_cache", |b| {
        let mut engine = SimilaritySearch::with_cache_config(vector_size, 1000, 60);
        
        // Pre-populate with data
        for (i, vector) in test_vectors.iter().enumerate() {
            engine.add_position(vector.clone(), i as f32 * 0.01);
        }
        
        b.iter(|| {
            // Perform searches (should benefit from caching on repeated runs)
            for query in &query_vectors {
                black_box(engine.search(query, 5));
            }
        })
    });
    
    // Benchmark cache hit performance specifically
    c.bench_function("cache_hit_performance", |b| {
        let mut engine = SimilaritySearch::with_cache_config(vector_size, 1000, 60);
        
        // Pre-populate with data
        for (i, vector) in test_vectors.iter().enumerate() {
            engine.add_position(vector.clone(), i as f32 * 0.01);
        }
        
        // Prime the cache
        for query in &query_vectors {
            engine.search(query, 5);
        }
        
        b.iter(|| {
            // These should all be cache hits
            for query in &query_vectors {
                black_box(engine.search(query, 5));
            }
        })
    });
    
    // Benchmark cache management overhead
    c.bench_function("cache_management_overhead", |b| {
        let mut engine = SimilaritySearch::with_cache_config(vector_size, 100, 1); // Small cache, short TTL
        
        b.iter(|| {
            // Add positions and perform searches to force cache evictions
            for (i, vector) in test_vectors.iter().enumerate().take(200) {
                engine.add_position(vector.clone(), i as f32 * 0.01);
                
                // Perform a search every few additions
                if i % 5 == 0 && i < query_vectors.len() * 5 {
                    let query_idx = (i / 5) % query_vectors.len();
                    black_box(engine.search(&query_vectors[query_idx], 3));
                }
            }
            
            let stats = engine.get_cache_stats();
            black_box(stats);
        })
    });
}

fn benchmark_cache_statistics(c: &mut Criterion) {
    let vector_size = 1024;
    let test_vectors = generate_test_vectors(vector_size, 100);
    let query_vectors = generate_test_vectors(vector_size, 20);
    
    c.bench_function("cache_statistics", |b| {
        let mut engine = SimilaritySearch::with_cache_config(vector_size, 1000, 60);
        
        // Populate engine and perform some searches
        for (i, vector) in test_vectors.iter().enumerate() {
            engine.add_position(vector.clone(), i as f32 * 0.01);
        }
        
        for query in &query_vectors {
            engine.search(query, 5);
        }
        
        b.iter(|| {
            let stats = engine.get_cache_stats();
            black_box(stats);
        })
    });
    
    c.bench_function("cache_clear_and_reset", |b| {
        let mut engine = SimilaritySearch::with_cache_config(vector_size, 1000, 60);
        
        // Populate engine
        for (i, vector) in test_vectors.iter().enumerate() {
            engine.add_position(vector.clone(), i as f32 * 0.01);
        }
        
        // Fill cache
        for query in &query_vectors {
            engine.search(query, 5);
        }
        
        b.iter(|| {
            engine.clear_caches();
            
            // Repopulate cache
            for query in query_vectors.iter().take(5) {
                engine.search(query, 3);
            }
            
            engine.reset_cache_stats();
        })
    });
}

fn benchmark_cache_effectiveness(c: &mut Criterion) {
    let vector_size = 512; // Smaller for faster benchmarking
    let dataset_size = 500;
    let test_vectors = generate_test_vectors(vector_size, dataset_size);
    
    // Create queries that are slight variations of dataset vectors (should have good cache locality)
    let similar_queries: Vec<Array1<f32>> = test_vectors.iter().take(10).map(|v| {
        let mut query = v.clone();
        // Add small noise
        for i in 0..query.len() {
            query[i] += 0.001 * ((i as f32).sin());
        }
        query
    }).collect();
    
    c.bench_function("cache_effectiveness_similar_queries", |b| {
        let mut engine = SimilaritySearch::with_cache_config(vector_size, 1000, 300);
        
        // Populate engine
        for (i, vector) in test_vectors.iter().enumerate() {
            engine.add_position(vector.clone(), i as f32 * 0.01);
        }
        
        b.iter(|| {
            // Perform searches with similar queries (should benefit from caching)
            for query in &similar_queries {
                black_box(engine.search(query, 5));
            }
            
            // Check cache effectiveness
            let stats = engine.get_cache_stats();
            black_box(stats.hit_ratio);
        })
    });
}

criterion_group!(
    benches,
    benchmark_similarity_search_caching,
    benchmark_cache_statistics,
    benchmark_cache_effectiveness
);
criterion_main!(benches);