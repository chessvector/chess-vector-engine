/// Example demonstrating the production-ready object pool implementation
use chess_vector_engine::utils::object_pool::{
    clear_vector_pools, get_vector, get_vector_pool_stats, get_zeroed_vector, return_vector,
    PooledVector,
};

fn main() {
    println!("=== Chess Vector Engine Object Pool Demo ===\n");

    // Clear pools to start fresh
    clear_vector_pools();

    // Method 1: Manual pool management
    println!("1. Manual pool management:");
    let mut vectors = Vec::new();

    // Get vectors from pool
    for i in 0..5 {
        let mut vec = get_vector(1024);
        vec[0] = i as f32;
        vectors.push(vec);
    }

    println!("   Created {} vectors", vectors.len());

    // Return vectors to pool
    for vec in vectors {
        return_vector(vec);
    }

    // Check pool stats
    let stats = get_vector_pool_stats();
    println!("   Pool stats: {:?}", stats);

    // Method 2: RAII with PooledVector
    println!("\n2. RAII with PooledVector:");
    {
        let mut pooled1 = PooledVector::new(1024);
        let mut pooled2 = PooledVector::zeroed(512);

        // Use the vectors
        pooled1[0] = 42.0;
        pooled2[0] = 7.0;

        println!("   pooled1[0] = {}", pooled1[0]);
        println!("   pooled2[0] = {}", pooled2[0]);

        // Vectors automatically returned to pool when dropping
    }

    // Check pool stats after RAII
    let stats_after = get_vector_pool_stats();
    println!("   Pool stats after RAII: {:?}", stats_after);

    // Method 3: Taking ownership
    println!("\n3. Taking ownership:");
    let pooled = PooledVector::new(256);
    let owned = pooled.take(); // Takes ownership, won't return to pool
    println!("   Owned vector length: {}", owned.len());

    // Method 4: Performance comparison
    println!("\n4. Performance comparison:");

    // Time vector creation from pool
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        let vec = get_vector(1024);
        return_vector(vec);
    }
    let pool_time = start.elapsed();

    // Time direct vector creation
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        let _vec = ndarray::Array1::<f32>::zeros(1024);
    }
    let direct_time = start.elapsed();

    println!("   Pool creation time: {:?}", pool_time);
    println!("   Direct creation time: {:?}", direct_time);

    if pool_time < direct_time {
        println!("   Pool is faster by {:?}", direct_time - pool_time);
    } else {
        println!("   Direct is faster by {:?}", pool_time - direct_time);
    }

    // Method 5: Pool size limits
    println!("\n5. Pool size limits:");

    // Fill pool beyond limit
    for i in 0..15 {
        let mut vec = get_vector(64);
        vec[0] = i as f32;
        return_vector(vec);
    }

    let final_stats = get_vector_pool_stats();
    println!("   Final pool stats: {:?}", final_stats);
    println!("   Note: Pool size is limited to prevent memory bloat");

    // Method 6: Non-standard sizes
    println!("\n6. Non-standard sizes:");
    let weird_size = get_vector(100); // Not pooled
    println!("   Non-standard size vector: {} elements", weird_size.len());
    return_vector(weird_size); // Won't be pooled

    let stats_weird = get_vector_pool_stats();
    println!("   Pool stats (no change): {:?}", stats_weird);

    println!("\n=== Demo Complete ===");
}
