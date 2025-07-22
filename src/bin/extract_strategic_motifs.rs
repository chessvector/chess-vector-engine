#!/usr/bin/env rust
//! Strategic Motif Extraction Binary
//!
//! This binary extracts strategic motifs from the existing chess engine database
//! and creates a fast-loading strategic database for instant UCI startup.

use chess_vector_engine::{
    motif_extractor::MotifExtractor, strategic_motifs::StrategicDatabase, ChessVectorEngine,
};
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Strategic Motif Extraction Tool");
    println!("{}", "=".repeat(50));

    let args: Vec<String> = std::env::args().collect();
    let database_path = args
        .get(1)
        .unwrap_or(&"chess_vector_engine.db".to_string())
        .clone();
    let output_path = args
        .get(2)
        .unwrap_or(&"strategic_motifs.db".to_string())
        .clone();

    // Check if database exists
    if !Path::new(&database_path).exists() {
        eprintln!("❌ Database not found: {}", database_path);
        eprintln!("Usage: {} [database_path] [output_path]", args[0]);
        std::process::exit(1);
    }

    let db_size = std::fs::metadata(&database_path)?.len();
    println!(
        "📊 Input database: {} ({:.1} GB)",
        database_path,
        db_size as f64 / 1_000_000_000.0
    );

    // Load the existing engine database
    println!("🔄 Loading existing chess engine database...");
    let start_time = Instant::now();

    let mut engine = ChessVectorEngine::new(1024);

    // Enable persistence and load from database
    if let Err(e) = engine.enable_persistence(&database_path) {
        eprintln!("❌ Failed to enable persistence: {}", e);
        std::process::exit(1);
    }

    if let Err(e) = engine.load_from_database() {
        eprintln!("❌ Failed to load database: {}", e);
        std::process::exit(1);
    }

    let load_time = start_time.elapsed();
    let total_positions = engine.knowledge_base_size();

    println!(
        "✅ Loaded {} positions in {:.1}s",
        total_positions,
        load_time.as_secs_f32()
    );

    if total_positions == 0 {
        eprintln!("❌ No positions found in database");
        std::process::exit(1);
    }

    // Extract strategic motifs
    println!("\n🎯 Extracting strategic motifs...");
    let extraction_start = Instant::now();

    let mut extractor = MotifExtractor::new();
    let motifs = extractor.extract_from_engine(&engine)?;

    let extraction_time = extraction_start.elapsed();
    println!(
        "⏱️  Extraction completed in {:.1}s",
        extraction_time.as_secs_f32()
    );

    // Create strategic database
    println!("\n📚 Creating strategic database...");
    let mut strategic_db = StrategicDatabase::new();

    for motif in motifs {
        strategic_db.add_motif(motif);
    }

    // Save to binary file
    println!("💾 Saving strategic database to {}...", output_path);
    strategic_db.save_to_binary(&output_path)?;

    let output_size = std::fs::metadata(&output_path)?.len();
    let compression_ratio = db_size as f64 / output_size as f64;

    // Print results
    println!("\n🎉 Strategic Motif Extraction Complete!");
    println!("{}", "=".repeat(50));
    println!("📊 Results:");
    println!(
        "   📥 Input size:     {:.1} GB",
        db_size as f64 / 1_000_000_000.0
    );
    println!(
        "   📤 Output size:    {:.1} MB",
        output_size as f64 / 1_000_000.0
    );
    println!("   📈 Compression:    {:.1}x smaller", compression_ratio);
    println!(
        "   🎯 Motifs:         {}",
        strategic_db.stats().total_motifs
    );
    println!(
        "   ⏱️  Total time:     {:.1}s",
        (load_time + extraction_time).as_secs_f32()
    );

    // Test loading speed
    println!("\n⚡ Testing strategic database load speed...");
    let load_test_start = Instant::now();
    let _test_db = StrategicDatabase::load_from_binary(&output_path)?;
    let load_test_time = load_test_start.elapsed();

    println!(
        "✅ Strategic database loads in {:.0}ms ({}x faster than original)",
        load_test_time.as_millis(),
        (load_time.as_millis() as f64 / load_test_time.as_millis() as f64) as u32
    );

    // Performance comparison
    println!("\n📈 Performance Improvement:");
    println!(
        "   🚀 Startup time:   {:.0}ms vs {:.1}s ({:.0}x faster)",
        load_test_time.as_millis(),
        load_time.as_secs_f32(),
        load_time.as_secs_f32() * 1000.0 / load_test_time.as_millis() as f32
    );
    println!(
        "   💾 Storage size:   {:.1} MB vs {:.1} GB ({:.0}x smaller)",
        output_size as f64 / 1_000_000.0,
        db_size as f64 / 1_000_000_000.0,
        compression_ratio
    );

    // UCI compliance check
    if load_test_time.as_millis() < 1000 {
        println!("   ✅ UCI compliance: PASSED (startup < 1 second)");
    } else {
        println!("   ⚠️  UCI compliance: Consider further optimization");
    }

    println!("\n🎯 Strategic motifs extracted successfully!");
    println!("   Use 'strategic_motifs.db' for instant UCI startup with master-level strategic understanding");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motif_extraction_with_small_dataset() {
        // Test extraction with a small dataset
        let mut engine = ChessVectorEngine::new(128);

        // Add a few test positions
        use chess::Board;
        use std::str::FromStr;

        let boards = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
        ];

        for (i, fen) in boards.iter().enumerate() {
            if let Ok(board) = Board::from_str(fen) {
                engine.add_position(&board, i as f32 * 0.1);
            }
        }

        let mut extractor = MotifExtractor::new();
        let result = extractor.extract_from_engine(&engine);

        assert!(result.is_ok());
        // With such a small dataset, we might not find significant patterns
        // but the extraction should complete without errors
    }

    #[test]
    fn test_strategic_database_save_load() {
        use tempfile::NamedTempFile;

        let mut db = StrategicDatabase::new();

        // The database starts empty but should save/load correctly
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        let save_result = db.save_to_binary(path);
        assert!(save_result.is_ok());

        let load_result = StrategicDatabase::load_from_binary(path);
        assert!(load_result.is_ok());
    }
}
