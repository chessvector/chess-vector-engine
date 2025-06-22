use chess_vector_engine::{ChessVectorEngine, Database};
use std::collections::HashSet;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Database vs Training Files Analysis");
    println!("======================================");

    // Check what's in the database
    println!("\n📊 Analyzing chess_vector_engine.db...");
    let db = Database::new("chess_vector_engine.db")?;
    let db_positions = db.load_all_positions()?;
    println!("   Database contains {} positions", db_positions.len());
    
    let mut db_fens = HashSet::new();
    for pos in &db_positions {
        db_fens.insert(pos.fen.clone());
    }
    
    println!("   Unique FENs in database: {}", db_fens.len());

    // Load positions from various training files to compare
    let mut engine = ChessVectorEngine::new(1024);
    
    // Count positions in different file formats
    let training_files = [
        ("training_data_a100.json", "A100 JSON"),
        ("training_data.json", "Main JSON"),
        ("tactical_training_data.json", "Tactical JSON"),
    ];
    
    let mut total_file_positions = 0;
    let mut all_file_fens: HashSet<String> = HashSet::new();
    
    for (file_path, description) in &training_files {
        if std::path::Path::new(file_path).exists() {
            println!("\n📂 Analyzing {}...", description);
            
            let mut temp_engine = ChessVectorEngine::new(1024);
            match temp_engine.load_training_data_incremental(file_path) {
                Ok(()) => {
                    let file_positions = temp_engine.knowledge_base_size();
                    println!("   {} contains {} positions", description, file_positions);
                    total_file_positions += file_positions;
                    
                    // Can't access private fields directly - skip detailed FEN comparison
                    // This would require modifying the public API
                },
                Err(e) => println!("   Failed to load {}: {}", description, e),
            }
        } else {
            println!("   {} not found", file_path);
        }
    }
    
    println!("\n📊 Summary:");
    println!("   Total positions across all files: {}", total_file_positions);
    println!("   Unique FENs across all files: {}", all_file_fens.len());
    println!("   Database positions: {}", db_positions.len());
    println!("   Database unique FENs: {}", db_fens.len());
    
    // Skip detailed overlap analysis due to private field access limitation
    println!("\n🔍 Comparison (based on counts only):");
    println!("   Database positions: {}", db_positions.len());
    println!("   File positions total: {}", total_file_positions);
    
    // Show sample database FENs
    if !db_fens.is_empty() {
        println!("\n📄 Sample database positions:");
        for (i, fen) in db_fens.iter().take(5).enumerate() {
            println!("   {}. {}", i + 1, fen);
        }
    }
    
    // Test loading with different methods
    println!("\n🚀 Testing loading methods...");
    
    // Test new_with_fast_load
    println!("   Testing new_with_fast_load...");
    match ChessVectorEngine::new_with_fast_load(1024) {
        Ok(fast_engine) => {
            println!("     Fast load: {} positions", fast_engine.knowledge_base_size());
        },
        Err(e) => println!("     Fast load failed: {}", e),
    }
    
    // Test new_with_instant_load
    println!("   Testing new_with_instant_load...");
    match ChessVectorEngine::new_with_instant_load(1024) {
        Ok(instant_engine) => {
            println!("     Instant load: {} positions", instant_engine.knowledge_base_size());
        },
        Err(e) => println!("     Instant load failed: {}", e),
    }
    
    // Check if database loading is automatic
    println!("\n🗄️  Testing database auto-loading...");
    let mut persistence_engine = ChessVectorEngine::new(1024);
    if let Ok(()) = persistence_engine.enable_persistence("chess_vector_engine.db") {
        match persistence_engine.load_from_database() {
            Ok(()) => {
                println!("   Manual database load: {} positions", persistence_engine.knowledge_base_size());
            },
            Err(e) => println!("   Manual database load failed: {}", e),
        }
    }

    println!("\n✅ Analysis complete!");
    
    Ok(())
}