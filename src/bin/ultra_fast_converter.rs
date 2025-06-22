use chess_vector_engine::ChessVectorEngine;
use clap::{Parser, Subcommand};
use std::time::Instant;

/// Ultra-fast training data converter for maximum loading performance
/// Convert your training data to the fastest possible formats for instant startup
#[derive(Parser)]
#[command(name = "ultra-fast-converter")]
#[command(about = "Convert training data to ultra-fast formats for instant loading")]
#[command(long_about = "
🚀 ULTRA-FAST TRAINING DATA CONVERTER 🚀

This tool converts your chess training data to the fastest possible formats:

📦 FORMATS (ranked by speed):
  1. Memory-mapped (.mmap)    - INSTANT loading, zero-copy access
  2. MessagePack (.msgpack)   - 10-20% faster than bincode, smaller files  
  3. Zstd compressed (.zst)   - Best compression ratio, fast decompression
  4. LZ4 binary (.bin)        - Current binary format, good balance

🎯 USAGE EXAMPLES:
  # Convert all formats for maximum compatibility
  cargo run --bin ultra_fast_converter -- all
  
  # Convert to specific format
  cargo run --bin ultra_fast_converter -- msgpack
  cargo run --bin ultra_fast_converter -- mmap
  cargo run --bin ultra_fast_converter -- zstd
  
  # Convert and benchmark
  cargo run --bin ultra_fast_converter -- benchmark

⚡ PERFORMANCE IMPROVEMENTS:
  - Memory-mapped: Instant startup (zero loading time)
  - MessagePack: 10-20% faster loading than bincode
  - Zstd: 2-5x smaller files than JSON
  - Parallel processing: Multi-threaded JSON parsing

The converted files work with new_with_instant_load() for maximum performance.
")]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Convert to all ultra-fast formats
    All,
    /// Convert to MessagePack format (fastest binary)
    Msgpack,
    /// Convert to memory-mapped format (instant loading)
    Mmap,
    /// Convert to Zstd compressed format (smallest files)
    Zstd,
    /// Convert and benchmark all formats
    Benchmark,
    /// Show file sizes and recommendations
    Info,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    
    println!("🚀 Ultra-Fast Training Data Converter");
    println!("=====================================");
    
    match args.command {
        Commands::All => {
            convert_all_formats()?;
        },
        Commands::Msgpack => {
            let start = Instant::now();
            ChessVectorEngine::convert_to_msgpack()?;
            println!("✅ MessagePack conversion completed in {:.2}s", start.elapsed().as_secs_f64());
        },
        Commands::Mmap => {
            let start = Instant::now();
            ChessVectorEngine::convert_to_mmap()?;
            println!("✅ Memory-mapped conversion completed in {:.2}s", start.elapsed().as_secs_f64());
        },
        Commands::Zstd => {
            let start = Instant::now();
            ChessVectorEngine::convert_to_zstd()?;
            println!("✅ Zstd compression completed in {:.2}s", start.elapsed().as_secs_f64());
        },
        Commands::Benchmark => {
            benchmark_all_formats()?;
        },
        Commands::Info => {
            show_file_info()?;
        },
    }
    
    Ok(())
}

fn convert_all_formats() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Converting to all ultra-fast formats...\n");
    
    // Convert to MessagePack first (fastest binary format)
    println!("1️⃣ Converting to MessagePack format...");
    let start = Instant::now();
    ChessVectorEngine::convert_to_msgpack()?;
    println!("   ✅ MessagePack conversion: {:.2}s\n", start.elapsed().as_secs_f64());
    
    // Convert to memory-mapped format (instant loading)
    println!("2️⃣ Converting to memory-mapped format...");
    let start = Instant::now();
    ChessVectorEngine::convert_to_mmap()?;
    println!("   ✅ Memory-mapped conversion: {:.2}s\n", start.elapsed().as_secs_f64());
    
    // Convert to Zstd compressed format (smallest files)
    println!("3️⃣ Converting to Zstd compressed format...");
    let start = Instant::now();
    ChessVectorEngine::convert_to_zstd()?;
    println!("   ✅ Zstd compression: {:.2}s\n", start.elapsed().as_secs_f64());
    
    println!("🎯 All conversions complete! Files are ready for ultra-fast loading.");
    println!("   Use ChessVectorEngine::new_with_instant_load() for maximum speed.");
    
    Ok(())
}

fn benchmark_all_formats() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏁 Benchmarking all loading formats...\n");
    
    // First convert all formats
    convert_all_formats()?;
    
    println!("\n📊 LOADING PERFORMANCE BENCHMARK");
    println!("================================\n");
    
    let vector_size = 1024;
    let formats = [
        ("Memory-mapped", "training_data.mmap", "load_training_data_mmap"),
        ("MessagePack", "training_data.msgpack", "load_training_data_msgpack"), 
        ("Zstd compressed", "training_data.zst", "load_training_data_compressed"),
        ("LZ4 binary", "training_data.bin", "load_training_data_binary"),
        ("Streaming JSON", "training_data.json", "load_training_data_streaming_json"),
    ];
    
    for (name, file_path, _method) in &formats {
        if std::path::Path::new(file_path).exists() {
            println!("🔄 Testing {} format...", name);
            
            let start = Instant::now();
            let mut engine = ChessVectorEngine::new(vector_size);
            
            let load_result = match *file_path {
                f if f.ends_with(".mmap") => engine.load_training_data_mmap(f),
                f if f.ends_with(".msgpack") => engine.load_training_data_msgpack(f),
                f if f.ends_with(".zst") => engine.load_training_data_compressed(f),
                f if f.ends_with(".bin") => engine.load_training_data_binary(f),
                f if f.ends_with(".json") => engine.load_training_data_streaming_json(f),
                _ => continue,
            };
            
            match load_result {
                Ok(()) => {
                    let duration = start.elapsed();
                    let positions = engine.knowledge_base_size();
                    let positions_per_sec = positions as f64 / duration.as_secs_f64();
                    
                    println!("   ✅ {}: {:.3}s ({} positions, {:.0} pos/sec)", 
                             name, duration.as_secs_f64(), positions, positions_per_sec);
                },
                Err(e) => {
                    println!("   ❌ {}: Failed - {}", name, e);
                }
            }
        } else {
            println!("   ⚠️  {}: File not found ({})", name, file_path);
        }
    }
    
    println!("\n🎯 Benchmark complete! Use the fastest format for your use case.");
    
    Ok(())
}

fn show_file_info() -> Result<(), Box<dyn std::error::Error>> {
    println!("📁 TRAINING DATA FILE ANALYSIS");
    println!("==============================\n");
    
    let files = [
        ("JSON Original", "training_data.json"),
        ("JSON Tactical", "tactical_training_data.json"),
        ("JSON A100", "training_data_a100.json"),
        ("LZ4 Binary", "training_data.bin"),
        ("LZ4 Tactical", "tactical_training_data.bin"),
        ("LZ4 A100", "training_data_a100.bin"),
        ("MessagePack", "training_data.msgpack"),
        ("MessagePack Tactical", "tactical_training_data.msgpack"),
        ("MessagePack A100", "training_data_a100.msgpack"),
        ("Memory-mapped", "training_data.mmap"),
        ("Memory-mapped Tactical", "tactical_training_data.mmap"),
        ("Memory-mapped A100", "training_data_a100.mmap"),
        ("Zstd Compressed", "training_data.zst"),
        ("Zstd Tactical", "tactical_training_data.zst"),
        ("Zstd A100", "training_data_a100.zst"),
    ];
    
    let mut found_files = Vec::new();
    
    for (name, path) in &files {
        if let Ok(metadata) = std::fs::metadata(path) {
            let size = metadata.len();
            found_files.push((name, path, size));
        }
    }
    
    if found_files.is_empty() {
        println!("⚠️  No training data files found in current directory.");
        println!("   Expected files: training_data.json, tactical_training_data.json");
        return Ok(());
    }
    
    // Sort by file size
    found_files.sort_by_key(|(_, _, size)| *size);
    
    println!("📊 File sizes (smallest to largest):");
    for (name, path, size) in &found_files {
        println!("   {} - {} ({})", format_bytes(*size), name, path);
    }
    
    // Show recommendations
    println!("\n💡 RECOMMENDATIONS:");
    println!("   🚀 For fastest loading: Use .mmap files (memory-mapped)");
    println!("   📦 For smallest files: Use .zst files (Zstd compressed)");
    println!("   ⚖️  For best balance: Use .msgpack files (MessagePack binary)");
    println!("   🎯 For instant startup: Use new_with_instant_load() method");
    
    Ok(())
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;
    
    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }
    
    format!("{:>8.1} {}", size, UNITS[unit_index])
}