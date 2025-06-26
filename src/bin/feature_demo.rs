use chess_vector_engine::{ChessVectorEngine, FeatureTier};

/// Demonstrate the feature gating system
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 FEATURE GATING DEMONSTRATION");
    println!("===============================");

    // Test 1: Open Source Engine
    println!("\n1️⃣ Open Source Engine (Free Tier)");
    let mut open_engine = ChessVectorEngine::new(1024);
    println!("   Current tier: {:?}", open_engine.get_feature_tier());

    // These should work (open source features)
    println!(
        "   ✅ Basic position encoding: {}",
        open_engine.is_feature_available("basic_position_encoding")
    );
    println!(
        "   ✅ Similarity search: {}",
        open_engine.is_feature_available("similarity_search")
    );
    println!(
        "   ✅ Opening book: {}",
        open_engine.is_feature_available("opening_book")
    );

    // These should fail (premium features)
    println!(
        "   ❌ GPU acceleration: {}",
        open_engine.is_feature_available("gpu_acceleration")
    );
    println!(
        "   ❌ Ultra-fast loading: {}",
        open_engine.is_feature_available("ultra_fast_loading")
    );
    println!(
        "   ❌ Memory-mapped files: {}",
        open_engine.is_feature_available("memory_mapped_files")
    );

    // Try to use a premium feature
    println!("\n   🚫 Attempting to use ultra-fast loading...");
    if let Some(test_file) = find_test_file() {
        match open_engine.ultra_fast_load_any_format(&test_file) {
            Ok(()) => println!("      ✅ Unexpected success!"),
            Err(e) => println!("❌ Error: {e}"),
        }
    } else {
        println!("      ⚠️  No test file found, creating fake error...");
        match open_engine.require_feature("ultra_fast_loading") {
            Ok(()) => println!("      ✅ Unexpected success!"),
            Err(e) => println!("❌ Error: {e}"),
        }
    }

    // Test 2: Premium Engine
    println!("\n2️⃣ Premium Engine (Paid Tier)");
    let premium_engine = ChessVectorEngine::new_with_tier(1024, FeatureTier::Premium);
    println!("   Current tier: {:?}", premium_engine.get_feature_tier());

    // These should work (open source + premium features)
    println!(
        "   ✅ Basic position encoding: {}",
        premium_engine.is_feature_available("basic_position_encoding")
    );
    println!(
        "   ✅ GPU acceleration: {}",
        premium_engine.is_feature_available("gpu_acceleration")
    );
    println!(
        "   ✅ Ultra-fast loading: {}",
        premium_engine.is_feature_available("ultra_fast_loading")
    );
    println!(
        "   ✅ Memory-mapped files: {}",
        premium_engine.is_feature_available("memory_mapped_files")
    );
    println!(
        "   ✅ Advanced tactical search: {}",
        premium_engine.is_feature_available("advanced_tactical_search")
    );

    // These should fail (enterprise features)
    println!(
        "   ❌ Distributed training: {}",
        premium_engine.is_feature_available("distributed_training")
    );
    println!(
        "   ❌ Enterprise analytics: {}",
        premium_engine.is_feature_available("enterprise_analytics")
    );

    // Test 3: Enterprise Engine
    println!("\n3️⃣ Enterprise Engine (Full Access)");
    let enterprise_engine = ChessVectorEngine::new_with_tier(1024, FeatureTier::Enterprise);
    println!(
        "   Current tier: {:?}",
        enterprise_engine.get_feature_tier()
    );

    // All features should work
    println!(
        "   ✅ Basic position encoding: {}",
        enterprise_engine.is_feature_available("basic_position_encoding")
    );
    println!(
        "   ✅ GPU acceleration: {}",
        enterprise_engine.is_feature_available("gpu_acceleration")
    );
    println!(
        "   ✅ Ultra-fast loading: {}",
        enterprise_engine.is_feature_available("ultra_fast_loading")
    );
    println!(
        "   ✅ Distributed training: {}",
        enterprise_engine.is_feature_available("distributed_training")
    );
    println!(
        "   ✅ Enterprise analytics: {}",
        enterprise_engine.is_feature_available("enterprise_analytics")
    );

    // Test 4: Tier Upgrade
    println!("\n4️⃣ Tier Upgrade Simulation");
    let mut upgradeable_engine = ChessVectorEngine::new(1024);
    println!(
        "   Initial tier: {:?}",
        upgradeable_engine.get_feature_tier()
    );
    println!(
        "   GPU acceleration available: {}",
        upgradeable_engine.is_feature_available("gpu_acceleration")
    );

    // Simulate license activation
    println!("   🔄 Upgrading to Premium tier...");
    upgradeable_engine.upgrade_tier(FeatureTier::Premium);
    println!("   New tier: {:?}", upgradeable_engine.get_feature_tier());
    println!(
        "   GPU acceleration available: {}",
        upgradeable_engine.is_feature_available("gpu_acceleration")
    );

    // Feature summary
    println!("\n📊 FEATURE SUMMARY BY TIER:");
    let registry = chess_vector_engine::FeatureRegistry::new();

    for tier in [
        FeatureTier::OpenSource,
        FeatureTier::Premium,
        FeatureTier::Enterprise,
    ] {
        let features = registry.get_features_for_tier(&tier);
        println!("   {:?}: {} features", tier, features.len());
        for feature in features.iter().take(5) {
            println!("Feature not available");
        }
        if features.len() > 5 {
            println!("     ... and {} more", features.len() - 5);
        }
    }

    Ok(())
}

fn find_test_file() -> Option<std::path::PathBuf> {
    // Look for any training file to test with
    for file in [
        "test_loading.json",
        "training_data.bin",
        "training_data_a100.bin",
    ] {
        if std::path::Path::new(file).exists() {
            return Some(std::path::PathBuf::from(file));
        }
    }
    None
}
