use chess_vector_engine::ChessVectorEngine;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Testing INSTANT loading performance...\n");
    
    // Test the new instant loading method
    println!("Testing new_with_instant_load()...");
    let start = Instant::now();
    let engine = ChessVectorEngine::new_with_instant_load(1024)?;
    let instant_load_time = start.elapsed();
    let instant_positions = engine.knowledge_base_size();
    
    println!("✅ Instant load: {:.3}s ({} positions)\n", 
             instant_load_time.as_secs_f64(), instant_positions);
    
    // For comparison, test the previous fast loading method  
    println!("Testing new_with_fast_load() for comparison...");
    let start = Instant::now();
    let fast_engine = ChessVectorEngine::new_with_fast_load(1024)?;
    let fast_load_time = start.elapsed();
    let fast_positions = fast_engine.knowledge_base_size();
    
    println!("✅ Fast load: {:.3}s ({} positions)\n", 
             fast_load_time.as_secs_f64(), fast_positions);
    
    // Calculate improvement
    if instant_load_time < fast_load_time {
        let speedup = fast_load_time.as_secs_f64() / instant_load_time.as_secs_f64();
        println!("🎯 INSTANT loading is {:.1}x FASTER than fast loading!", speedup);
    } else {
        println!("ℹ️  Both methods loaded similar amounts of data at similar speeds");
    }
    
    println!("\n📊 Performance Summary:");
    println!("   Instant load: {:.3}s", instant_load_time.as_secs_f64());
    println!("   Fast load:    {:.3}s", fast_load_time.as_secs_f64());
    println!("   Positions:    {} / {}", instant_positions, fast_positions);
    
    if instant_positions > 0 {
        let positions_per_sec = instant_positions as f64 / instant_load_time.as_secs_f64();
        println!("   Speed:        {:.0} positions/second", positions_per_sec);
        
        if positions_per_sec > 100_000.0 {
            println!("   🚀 ULTRA-FAST loading achieved!");
        } else if positions_per_sec > 10_000.0 {
            println!("   ⚡ Fast loading achieved!");
        }
    }
    
    Ok(())
}