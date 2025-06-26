use chess_vector_engine::{ChessVectorEngine, FeatureTier};

/// Demo showing how to access all premium features as the creator
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Chess Vector Engine - Creator Premium Access Demo");
    println!("==================================================");

    // Method 1: Direct tier setting (for development/testing)
    println!("\n🔧 Method 1: Direct Premium Tier Access (Development Mode)");
    let mut engine_dev = ChessVectorEngine::new_with_tier(1024, FeatureTier::Premium);

    println!("✅ Engine created with Premium tier directly");
    println!("📊 Available features:");

    let premium_features = vec![
        "gpu_acceleration",
        "ultra_fast_loading",
        "advanced_tactical_search",
        "multi_threading",
        "nnue_evaluation",
        "manifold_learning",
        "pondering",
        "multi_pv",
        "custom_algorithms",
    ];

    for feature in &premium_features {
        let available = engine_dev.is_feature_available(feature);
        println!("   {} {}", if available { "✅" } else { "❌" }, feature);
    }

    // Method 2: Using creator license key (for production)
    println!("\n🔑 Method 2: Creator License Key (Production Mode)");
    let mut engine_licensed = ChessVectorEngine::new_with_offline_license(1024);

    // Creator license key (this would be your actual license in production)
    let creator_key = "CREATOR-PREMIUM-UNLIMITED-ACCESS-KEY";

    // Note: In production, use proper license activation:
    // engine_licensed.activate_license(creator_key).await?;

    // Demo premium features
    println!("\n🚀 Demonstrating Premium Features:");

    // 1. GPU Acceleration check
    if engine_licensed.is_feature_available("gpu_acceleration") {
        println!("✅ GPU Acceleration: Available");
        match engine_licensed.check_gpu_acceleration() {
            Ok(()) => println!("   🔥 GPU acceleration ready"),
            Err(e) => println!("   ⚠️  GPU not available (using CPU): {}", e),
        }
    }

    // 2. Ultra-fast loading
    if engine_licensed.is_feature_available("ultra_fast_loading") {
        println!("✅ Ultra-Fast Loading: Available");
        println!("   ⚡ Can load millions of positions in seconds");
    }

    // 3. Advanced tactical search
    if engine_licensed.is_feature_available("advanced_tactical_search") {
        println!("✅ Advanced Tactical Search: Available");
        println!("   🧠 Enhanced algorithms with deeper analysis");
    }

    // 4. Lichess puzzle loading (Premium vs Basic)
    println!("\n📚 Lichess Puzzle Loading Comparison:");

    // Basic tier loading (limited)
    let basic_engine = ChessVectorEngine::new(1024);
    println!("❌ Basic Tier: Limited to 50,000 puzzles max");
    println!("   📊 Rating range: 1000-2000");
    println!("   ⚡ Batch size: 10,000");

    // Premium tier loading (unlimited)
    println!("✅ Premium Tier: Unlimited puzzle loading");
    println!("   📊 Rating range: 1200-2400 (higher quality)");
    println!("   ⚡ Batch size: 100,000 (10x faster)");
    println!("   🎯 Theme filtering for tactical patterns");

    // 5. Advanced evaluation features
    println!("\n🎯 Advanced Evaluation Features:");

    if engine_licensed.is_feature_available("nnue_evaluation") {
        println!("✅ NNUE Neural Networks: Available");
        println!("   🧠 Deep learning position evaluation");
    }

    if engine_licensed.is_feature_available("manifold_learning") {
        println!("✅ Manifold Learning: Available");
        println!("   📐 Advanced dimensionality reduction");
    }

    if engine_licensed.is_feature_available("multi_threading") {
        println!("✅ Multi-Threading: Available");
        println!("   ⚡ Parallel search with up to 32 threads");
    }

    // 6. UCI Engine features
    println!("\n♟️  UCI Engine Premium Features:");

    if engine_licensed.is_feature_available("pondering") {
        println!("✅ Pondering: Available");
        println!("   🤔 Think on opponent's time");
    }

    if engine_licensed.is_feature_available("multi_pv") {
        println!("✅ Multi-PV Analysis: Available");
        println!("   📊 Show multiple best move variations");
    }

    // Demo actual premium loading (if file exists)
    let lichess_path = "~/Downloads/lichess_db_puzzle.csv";
    let expanded_path = if lichess_path.starts_with("~/") {
        if let Some(home) = std::env::var("HOME").ok() {
            lichess_path.replace("~", &home)
        } else {
            lichess_path.to_string()
        }
    } else {
        lichess_path.to_string()
    };

    if std::path::Path::new(&expanded_path).exists() {
        println!("\n🔥 Demo: Premium Lichess Loading");
        println!("Found Lichess database at: {}", expanded_path);

        // This would work if the feature was properly activated
        if engine_licensed.is_feature_available("ultra_fast_loading") {
            println!("✅ Premium loading capability confirmed");
            println!("   💡 Use: engine.load_lichess_puzzles_premium(path) for unlimited loading");
        }
    }

    // Show the difference in practice
    println!("\n⚡ Performance Comparison:");
    println!("┌─────────────────┬──────────────┬──────────────┐");
    println!("│ Feature         │ Basic Tier   │ Premium Tier │");
    println!("├─────────────────┼──────────────┼──────────────┤");
    println!("│ Puzzle Limit    │ 50,000       │ Unlimited    │");
    println!("│ Loading Speed   │ 10k batches  │ 100k batches │");
    println!("│ Thread Count    │ 8 max        │ 32 max       │");
    println!("│ Search Depth    │ 10 ply       │ 16+ ply      │");
    println!("│ GPU Support     │ ❌           │ ✅           │");
    println!("│ NNUE Networks   │ ❌           │ ✅           │");
    println!("│ Pondering       │ ❌           │ ✅           │");
    println!("│ Multi-PV        │ ❌           │ ✅           │");
    println!("└─────────────────┴──────────────┴──────────────┘");

    println!("\n🎉 Creator Access Summary:");
    println!("✅ Full access to all premium features");
    println!("✅ Unlimited Lichess puzzle loading");
    println!("✅ Maximum performance optimizations");
    println!("✅ Advanced neural network evaluation");
    println!("✅ Professional UCI engine capabilities");

    println!("\n💡 Quick Start for Creator:");
    println!("```rust");
    println!("// Method 1: Direct tier (development)");
    println!("let engine = ChessVectorEngine::new_with_tier(1024, FeatureTier::Premium);");
    println!();
    println!("// Method 2: Creator access (development/production)");
    println!("let mut engine = ChessVectorEngine::new_with_offline_license(1024);");
    println!("engine.unlock_creator_features(); // Unlock all premium features");
    println!();
    println!("// Load unlimited puzzles");
    println!("engine.load_lichess_puzzles_premium(\"lichess_db_puzzle.csv\")?;");
    println!("```");

    Ok(())
}
