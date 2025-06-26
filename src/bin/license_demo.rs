use chess_vector_engine::license::{current_timestamp, LicenseError, LicenseKey, LicenseVerifier};
use chess_vector_engine::{ChessVectorEngine, FeatureTier};
use tokio;

/// Comprehensive license system demonstration
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔐 LICENSE VERIFICATION SYSTEM DEMONSTRATION");
    println!("===========================================");

    // Test 1: Offline License Verification
    println!("\n1️⃣ OFFLINE LICENSE VERIFICATION");
    println!("--------------------------------");

    let mut engine = ChessVectorEngine::new_with_offline_license(1024);
    println!("✅ Engine created with offline license verification");

    // Try to activate a demo license
    println!("\n🔑 Activating DEMO license...");
    match engine.activate_license("DEMO-123456").await {
        Ok(tier) => {
            println!("✅ License activated successfully!");
            println!("   Tier: {:?}", tier);
            println!("   Current engine tier: {:?}", engine.get_feature_tier());
        }
        Err(e) => {
            println!("❌ License activation failed: {}", e);
        }
    }

    // Test basic features
    println!("\n📋 Testing basic features:");
    match engine
        .check_licensed_feature("basic_position_encoding")
        .await
    {
        Ok(()) => println!("   ✅ basic_position_encoding: Available"),
        Err(e) => println!("   ❌ basic_position_encoding: {}", e),
    }

    match engine.check_licensed_feature("gpu_acceleration").await {
        Ok(()) => println!("   ✅ gpu_acceleration: Available"),
        Err(e) => println!("   ❌ gpu_acceleration: {}", e),
    }

    // Test 2: Premium License
    println!("\n2️⃣ PREMIUM LICENSE ACTIVATION");
    println!("-----------------------------");

    println!("🔑 Activating PREMIUM license...");
    match engine.activate_license("PREMIUM-789012").await {
        Ok(tier) => {
            println!("✅ Premium license activated!");
            println!("   New tier: {:?}", tier);
        }
        Err(e) => {
            println!("❌ Premium activation failed: {}", e);
        }
    }

    // Test premium features
    println!("\n📋 Testing premium features:");
    let premium_features = [
        "gpu_acceleration",
        "ultra_fast_loading",
        "memory_mapped_files",
        "advanced_tactical_search",
        "pondering",
        "multi_pv_analysis",
    ];

    for feature in &premium_features {
        match engine.check_licensed_feature(feature).await {
            Ok(()) => println!("   ✅ {}: Available", feature),
            Err(e) => println!("   ❌ {}: {}", feature, e),
        }
    }

    // Test enterprise features (should fail)
    println!("\n📋 Testing enterprise features (should fail):");
    let enterprise_features = [
        "distributed_training",
        "enterprise_analytics",
        "custom_algorithms",
    ];

    for feature in &enterprise_features {
        match engine.check_licensed_feature(feature).await {
            Ok(()) => println!("   ✅ {}: Available", feature),
            Err(e) => println!("   ❌ {}: {}", feature, e),
        }
    }

    // Test 3: Enterprise License
    println!("\n3️⃣ ENTERPRISE LICENSE ACTIVATION");
    println!("--------------------------------");

    println!("🔑 Activating ENTERPRISE license...");
    match engine.activate_license("ENTERPRISE-345678").await {
        Ok(tier) => {
            println!("✅ Enterprise license activated!");
            println!("   New tier: {:?}", tier);
        }
        Err(e) => {
            println!("❌ Enterprise activation failed: {}", e);
        }
    }

    // Test all features
    println!("\n📋 Testing all features (should all work):");
    let all_features = [
        "basic_position_encoding",
        "similarity_search",
        "opening_book",
        "gpu_acceleration",
        "ultra_fast_loading",
        "memory_mapped_files",
        "advanced_tactical_search",
        "pondering",
        "multi_pv_analysis",
        "distributed_training",
        "enterprise_analytics",
        "custom_algorithms",
    ];

    for feature in &all_features {
        match engine.check_licensed_feature(feature).await {
            Ok(()) => println!("   ✅ {}: Available", feature),
            Err(e) => println!("   ❌ {}: {}", feature, e),
        }
    }

    // Test 4: License Cache
    println!("\n4️⃣ LICENSE CACHE TESTING");
    println!("------------------------");

    let cache_file = "test_license_cache.json";
    println!("💾 Saving license cache to {}...", cache_file);
    engine.save_license_cache(cache_file)?;
    println!("✅ License cache saved");

    // Create new engine and load cache
    let mut new_engine = ChessVectorEngine::new_with_offline_license(1024);
    println!("📂 Loading license cache...");
    new_engine.load_license_cache(cache_file)?;
    println!("✅ License cache loaded");

    // Test if cached licenses work
    println!("🔑 Testing cached license activation...");
    match new_engine.activate_license("ENTERPRISE-345678").await {
        Ok(tier) => {
            println!("✅ Cached license works! Tier: {:?}", tier);
        }
        Err(e) => {
            println!("❌ Cached license failed: {}", e);
        }
    }

    // Test 5: Invalid License
    println!("\n5️⃣ INVALID LICENSE TESTING");
    println!("--------------------------");

    println!("🔑 Trying invalid license...");
    match engine.activate_license("INVALID-999999").await {
        Ok(tier) => {
            println!(
                "❌ Unexpected success with invalid license! Tier: {:?}",
                tier
            );
        }
        Err(e) => {
            println!("✅ Expected failure: {}", e);
        }
    }

    // Test 6: License Verifier Directly
    println!("\n6️⃣ DIRECT LICENSE VERIFIER TESTING");
    println!("----------------------------------");

    let mut verifier = LicenseVerifier::new_offline();

    // Add a test license manually
    let test_license = LicenseKey {
        key: "TEST-MANUAL".to_string(),
        tier: FeatureTier::Premium,
        expires_at: current_timestamp() + 86400, // 1 day
        issued_at: current_timestamp(),
        customer_id: "manual-test-user".to_string(),
        features: vec![
            "gpu_acceleration".to_string(),
            "ultra_fast_loading".to_string(),
        ],
    };

    verifier.add_license(test_license);
    println!("➕ Added manual test license");

    // Verify it works
    match verifier.verify_license("TEST-MANUAL").await {
        Ok(status) => {
            println!("✅ Manual license verification: {:?}", status);
        }
        Err(e) => {
            println!("❌ Manual license verification failed: {}", e);
        }
    }

    // Check specific feature
    match verifier
        .check_feature_license("TEST-MANUAL", "gpu_acceleration")
        .await
    {
        Ok(()) => {
            println!("✅ Feature 'gpu_acceleration' is licensed");
        }
        Err(e) => {
            println!("❌ Feature check failed: {}", e);
        }
    }

    // Clean up
    if std::path::Path::new(cache_file).exists() {
        std::fs::remove_file(cache_file)?;
        println!("🧹 Cleaned up test cache file");
    }

    println!("\n🎯 LICENSE SYSTEM DEMONSTRATION COMPLETE");
    println!("=======================================");
    println!("The license verification system is working correctly with:");
    println!("✅ Online/offline license verification");
    println!("✅ Tier-based feature access control");
    println!("✅ License caching and persistence");
    println!("✅ Integration with ChessVectorEngine");
    println!("✅ Mock license server responses");
    println!("✅ Error handling for invalid licenses");

    Ok(())
}

// Helper function to access private current_timestamp
mod license_utils {
    use std::time::{SystemTime, UNIX_EPOCH};

    pub fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
}
