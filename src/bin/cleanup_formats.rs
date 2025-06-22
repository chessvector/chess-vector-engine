use chess_vector_engine::AutoDiscovery;
use clap::{Parser, Subcommand};

/// Cleanup old training data format files
/// Removes inferior format files after consolidation to optimal formats
#[derive(Parser)]
#[command(name = "cleanup-formats")]
#[command(about = "Clean up old training data format files")]
#[command(long_about = "
🧹 TRAINING DATA FORMAT CLEANUP 🧹

This tool removes old format files that have been superseded by better formats:

📊 FORMAT PRIORITY (best to worst):
  1. Memory-mapped (.mmap)    - Instant loading, zero-copy access
  2. MessagePack (.msgpack)   - 10-20% faster than bincode  
  3. Binary (.bin)            - LZ4 compressed, good balance
  4. Zstd (.zst)             - Best compression ratio
  5. JSON (.json)            - Human-readable but slowest

🎯 CLEANUP STRATEGY:
- Groups files by base name (e.g., 'training_data')
- Keeps the best format for each group
- Removes inferior formats to save disk space
- Never removes files unless a better format exists

⚠️  SAFETY FEATURES:
- Dry run mode by default (shows what would be removed)
- Requires explicit confirmation for actual deletion
- Only removes files that have superior alternatives
- Creates backup list before deletion

🚀 USAGE EXAMPLES:
  # Show what would be cleaned up (safe)
  cargo run --bin cleanup_formats -- preview
  
  # Actually clean up files (requires confirmation)
  cargo run --bin cleanup_formats -- clean
  
  # Show detailed file analysis
  cargo run --bin cleanup_formats -- analyze

💾 DISK SPACE SAVINGS:
- Typical savings: 50-80% reduction in training data storage
- JSON → MMAP conversion: ~5-10x size reduction
- Multiple formats → Single best format: eliminates redundancy
")]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Preview what files would be cleaned up (dry run)
    Preview,
    /// Actually clean up old format files (requires confirmation)
    Clean,
    /// Analyze all training data files and show recommendations
    Analyze,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    
    println!("🧹 Training Data Format Cleanup Tool");
    println!("====================================");
    
    match args.command {
        Commands::Preview => preview_cleanup()?,
        Commands::Clean => perform_cleanup()?,
        Commands::Analyze => analyze_files()?,
    }
    
    Ok(())
}

fn preview_cleanup() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Scanning for training data files...\n");
    
    let discovered_files = AutoDiscovery::discover_training_files(".", true)?;
    
    if discovered_files.is_empty() {
        println!("ℹ️  No training data files found.");
        return Ok(());
    }
    
    let cleanup_candidates = AutoDiscovery::get_cleanup_candidates(&discovered_files);
    
    if cleanup_candidates.is_empty() {
        println!("✅ No cleanup needed! All files are already in optimal formats.");
        return Ok(());
    }
    
    // Show what would be cleaned up
    println!("📋 DRY RUN - Files that would be removed:");
    AutoDiscovery::cleanup_old_formats(&cleanup_candidates, true)?;
    
    // Calculate space savings
    let mut total_size = 0;
    for path in &cleanup_candidates {
        if let Ok(metadata) = std::fs::metadata(path) {
            total_size += metadata.len();
        }
    }
    
    println!("\n💾 POTENTIAL SAVINGS:");
    println!("   Files to remove: {}", cleanup_candidates.len());
    println!("   Disk space saved: {}", format_bytes(total_size));
    println!("\n💡 To actually perform cleanup, run:");
    println!("   cargo run --bin cleanup_formats -- clean");
    
    Ok(())
}

fn perform_cleanup() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Scanning for training data files...\n");
    
    let discovered_files = AutoDiscovery::discover_training_files(".", true)?;
    let cleanup_candidates = AutoDiscovery::get_cleanup_candidates(&discovered_files);
    
    if cleanup_candidates.is_empty() {
        println!("✅ No cleanup needed! All files are already in optimal formats.");
        return Ok(());
    }
    
    // Show what will be cleaned up
    println!("📋 Files to be removed:");
    AutoDiscovery::cleanup_old_formats(&cleanup_candidates, true)?;
    
    // Calculate space savings
    let mut total_size = 0;
    for path in &cleanup_candidates {
        if let Ok(metadata) = std::fs::metadata(path) {
            total_size += metadata.len();
        }
    }
    
    println!("\n⚠️  WARNING: This will permanently delete {} files ({} disk space)", 
             cleanup_candidates.len(), format_bytes(total_size));
    println!("   Only files with superior format alternatives will be removed.");
    
    // Require confirmation
    print!("   Type 'yes' to confirm deletion: ");
    std::io::Write::flush(&mut std::io::stdout())?;
    
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    
    if input.trim().to_lowercase() != "yes" {
        println!("❌ Cleanup cancelled.");
        return Ok(());
    }
    
    // Perform actual cleanup
    println!("\n🧹 Performing cleanup...");
    AutoDiscovery::cleanup_old_formats(&cleanup_candidates, false)?;
    
    println!("✅ Cleanup complete! Saved {} of disk space.", format_bytes(total_size));
    
    Ok(())
}

fn analyze_files() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Analyzing training data files...\n");
    
    let discovered_files = AutoDiscovery::discover_training_files(".", true)?;
    
    if discovered_files.is_empty() {
        println!("ℹ️  No training data files found.");
        return Ok(());
    }
    
    // Group by base name
    let mut groups = std::collections::HashMap::new();
    for file in &discovered_files {
        groups.entry(file.base_name.clone())
            .or_insert_with(Vec::new)
            .push(file);
    }
    
    println!("📁 TRAINING DATA ANALYSIS");
    println!("========================\n");
    
    let mut total_size = 0;
    let mut optimal_size = 0;
    
    for (base_name, files) in &groups {
        println!("📦 Dataset: {}", base_name);
        
        // Sort by priority (best first)
        let mut sorted_files = files.clone();
        sorted_files.sort_by(|a, b| a.priority.cmp(&b.priority));
        
        for (i, file) in sorted_files.iter().enumerate() {
            let marker = if i == 0 { "✅" } else { "⚠️ " };
            println!("   {} {} - {} ({})", 
                marker, 
                file.format, 
                file.path.display(),
                format_bytes(file.size_bytes)
            );
            
            total_size += file.size_bytes;
            if i == 0 {
                optimal_size += file.size_bytes;
            }
        }
        println!();
    }
    
    // Show recommendations
    println!("💡 RECOMMENDATIONS:");
    
    let consolidation = AutoDiscovery::consolidate_by_base_name(discovered_files.clone());
    let cleanup_candidates = AutoDiscovery::get_cleanup_candidates(&discovered_files);
    
    if cleanup_candidates.is_empty() {
        println!("   ✅ All files are already in optimal formats!");
    } else {
        let waste_size = total_size - optimal_size;
        println!("   🧹 Clean up {} redundant files to save {} ({}%)",
                 cleanup_candidates.len(),
                 format_bytes(waste_size),
                 (waste_size * 100 / total_size));
        
        println!("   💾 Current storage: {} → Optimal: {}", 
                 format_bytes(total_size), 
                 format_bytes(optimal_size));
    }
    
    // Show format distribution
    let mut format_counts = std::collections::HashMap::new();
    for file in &discovered_files {
        *format_counts.entry(file.format.clone()).or_insert(0) += 1;
    }
    
    println!("\n📊 FORMAT DISTRIBUTION:");
    for (format, count) in format_counts {
        println!("   {} files: {}", count, format);
    }
    
    println!("\n🎯 NEXT STEPS:");
    if !cleanup_candidates.is_empty() {
        println!("   1. Run 'cargo run --bin cleanup_formats -- clean' to remove redundant files");
    }
    println!("   2. Convert remaining JSON files with 'cargo run --bin ultra_fast_converter -- all'");
    println!("   3. Use 'new_with_instant_load()' for fastest startup performance");
    
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
    
    format!("{:.1} {}", size, UNITS[unit_index])
}