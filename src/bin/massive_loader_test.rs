use chess_vector_engine::ChessVectorEngine;
use std::time::Instant;

/// Test loading massive datasets with the new streaming loader
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 MASSIVE DATASET LOADING TEST");
    println!("=================================");
    
    // Test 1: Auto-discovery with consolidation
    println!("\n1️⃣ Testing auto-discovery with format consolidation...");
    let start = Instant::now();
    let _engine1 = ChessVectorEngine::new_with_auto_discovery(1024)?;
    let auto_discovery_time = start.elapsed();
    println!("   ⏱️  Auto-discovery time: {:.2}s", auto_discovery_time.as_secs_f64());
    
    // Test 2: Instant load (best format only)
    println!("\n2️⃣ Testing instant load (best format only)...");
    let start = Instant::now();
    let _engine2 = ChessVectorEngine::new_with_instant_load(1024)?;
    let instant_load_time = start.elapsed();
    println!("   ⏱️  Instant load time: {:.2}s", instant_load_time.as_secs_f64());
    
    // Test 3: Massive dataset optimized loader
    println!("\n3️⃣ Testing massive dataset loader...");
    let start = Instant::now();
    let engine3 = ChessVectorEngine::new_for_massive_datasets(1024)?;
    let massive_load_time = start.elapsed();
    println!("   ⏱️  Massive load time: {:.2}s", massive_load_time.as_secs_f64());
    println!("   📊 Final positions loaded: {}", engine3.knowledge_base_size());
    
    // Performance comparison
    println!("\n📈 PERFORMANCE COMPARISON:");
    println!("   Auto-discovery: {:.2}s", auto_discovery_time.as_secs_f64());
    println!("   Instant load:   {:.2}s", instant_load_time.as_secs_f64());
    println!("   Massive load:   {:.2}s", massive_load_time.as_secs_f64());
    
    let times = [
        ("Auto-discovery", auto_discovery_time.as_secs_f64()),
        ("Instant load", instant_load_time.as_secs_f64()),
        ("Massive load", massive_load_time.as_secs_f64()),
    ];
    let fastest = times.iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    
    println!("   🏆 Fastest: {} ({:.2}s)", fastest.0, fastest.1);
    
    // Memory usage estimate
    let positions = engine3.knowledge_base_size();
    let estimated_memory_mb = (positions * 1024 * 4) / (1024 * 1024); // 4 bytes per f32, 1024 dimensions
    println!("\n💾 MEMORY USAGE ESTIMATE:");
    println!("   Positions: {}", positions);
    println!("   Vector memory: ~{} MB", estimated_memory_mb);
    
    Ok(())
}