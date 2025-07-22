use chess_vector_engine::ChessVectorEngine;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::Command;
use std::time::Instant;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the Lichess puzzle CSV file (or download URL)
    #[arg(short, long, default_value = "lichess_db_puzzle.csv")]
    input: String,

    /// Maximum number of puzzles to load (default: 10,000 for testing)
    #[arg(short, long, default_value = "10000")]
    max_puzzles: usize,

    /// Database path for persistence
    #[arg(short, long, default_value = "chess_vector_engine.db")]
    database: String,

    /// Download the puzzle database if not found locally
    #[arg(long)]
    download: bool,

    /// Minimum puzzle rating to include
    #[arg(long, default_value = "1000")]
    min_rating: u32,

    /// Maximum puzzle rating to include
    #[arg(long, default_value = "2000")]
    max_rating: u32,

    /// Vector size for the engine
    #[arg(long, default_value = "1024")]
    vector_size: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("🔥 Chess Vector Engine - Lichess Puzzle Database Loader");
    println!("========================================================");
    println!();
    println!("Configuration:");
    println!("  • Input file: {}", args.input);
    println!("  • Max puzzles: {}", args.max_puzzles);
    println!("  • Database: {}", args.database);
    println!("  • Rating range: {}-{}", args.min_rating, args.max_rating);
    println!("  • Vector size: {}", args.vector_size);
    println!();

    // Check if we need to download the puzzle database
    let puzzle_file = if args.download || !Path::new(&args.input).exists() {
        download_lichess_puzzles(&args.input)?
    } else {
        args.input.clone()
    };

    // Initialize the chess engine with persistence
    println!("🚀 Initializing Chess Vector Engine...");
    let mut engine = ChessVectorEngine::new(args.vector_size);

    // Enable persistence to save positions to database
    engine.enable_persistence(&args.database)?;
    println!("✅ Database persistence enabled at: {}", args.database);

    // Check if we have existing data
    match engine.load_from_database() {
        Ok(_) => {
            let stats = engine.training_stats();
            println!(
                "📊 Loaded existing engine from database with {} positions",
                stats.total_positions
            );

            if stats.total_positions >= args.max_puzzles {
                println!(
                    "🎯 Database already contains {} positions (>= {}), skipping load",
                    stats.total_positions, args.max_puzzles
                );
                return Ok(());
            }
        }
        Err(_) => {
            println!("📁 Starting with empty database");
        }
    }

    // Load puzzles using the engine's built-in loader
    println!();
    println!("🧩 Loading Lichess puzzles...");
    let start_time = Instant::now();

    // Use the engine's load_lichess_puzzles_with_limit method
    engine.load_lichess_puzzles_with_limit(&puzzle_file, Some(args.max_puzzles))?;

    let elapsed = start_time.elapsed();
    let final_stats = engine.training_stats();

    println!();
    println!("🎉 Lichess puzzle loading complete!");
    println!("====================================");
    println!("⏱️  Total time: {:.2}s", elapsed.as_secs_f64());
    println!("📊 Final statistics:");
    println!("  • Total positions: {}", final_stats.total_positions);
    println!("  • Unique positions: {}", final_stats.unique_positions);
    println!("  • Has move data: {}", final_stats.has_move_data);
    println!("  • Vector dimension: {}", args.vector_size);
    println!("  • Database: {}", args.database);

    if final_stats.total_positions > 0 {
        println!(
            "🚀 Loading speed: {:.0} positions/second",
            final_stats.total_positions as f64 / elapsed.as_secs_f64()
        );
    }

    // Test the engine with a few similarity searches
    println!();
    println!("🔍 Testing similarity search...");
    let test_board = chess::Board::default();
    let similar_positions = engine.find_similar_positions(&test_board, 5);
    println!(
        "  • Found {} similar positions to starting position",
        similar_positions.len()
    );

    // Show a few examples if we have them
    if similar_positions.len() > 0 {
        println!("  • Example similarities:");
        for (i, (_, evaluation, similarity)) in similar_positions.iter().enumerate().take(3) {
            println!(
                "    - Position {}: evaluation = {:.3}, similarity = {:.3}",
                i + 1,
                evaluation,
                similarity
            );
        }
    }

    println!();
    println!(
        "✅ Engine is ready for chess analysis with {} positions!",
        final_stats.total_positions
    );
    println!("   Use the engine in your applications or run other binaries to analyze positions.");

    Ok(())
}

fn download_lichess_puzzles(_output_path: &str) -> Result<String, Box<dyn std::error::Error>> {
    let url = "https://database.lichess.org/lichess_db_puzzle.csv.bz2";
    let compressed_file = "lichess_db_puzzle.csv.bz2";

    println!("🌐 Downloading Lichess puzzle database...");
    println!("   URL: {}", url);
    println!("   This may take several minutes depending on your connection.");

    // Check if we have curl or wget available
    let download_cmd = if Command::new("curl").arg("--version").output().is_ok() {
        println!("   Using curl for download...");
        Command::new("curl")
            .arg("-L") // Follow redirects
            .arg("-o")
            .arg(compressed_file)
            .arg(url)
            .status()
    } else if Command::new("wget").arg("--version").output().is_ok() {
        println!("   Using wget for download...");
        Command::new("wget")
            .arg("-O")
            .arg(compressed_file)
            .arg(url)
            .status()
    } else {
        return Err("Neither curl nor wget found. Please install one of these tools or manually download the file from: https://database.lichess.org/lichess_db_puzzle.csv.bz2".into());
    };

    match download_cmd {
        Ok(status) if status.success() => {
            println!("✅ Download completed successfully");
        }
        Ok(status) => {
            return Err(format!(
                "Download failed with exit code: {}",
                status.code().unwrap_or(-1)
            )
            .into());
        }
        Err(e) => {
            return Err(format!("Download command failed: {}", e).into());
        }
    }

    // Decompress the file
    println!("📦 Decompressing puzzle database...");
    let decompress_status = if Command::new("bunzip2").arg("--version").output().is_ok() {
        println!("   Using bunzip2 for decompression...");
        Command::new("bunzip2")
            .arg("-k") // Keep original file
            .arg(compressed_file)
            .status()
    } else if Command::new("bzip2").arg("--version").output().is_ok() {
        println!("   Using bzip2 for decompression...");
        Command::new("bzip2")
            .arg("-d")
            .arg("-k") // Keep original file
            .arg(compressed_file)
            .status()
    } else {
        return Err("Neither bunzip2 nor bzip2 found. Please install bzip2 tools or manually decompress the file".into());
    };

    match decompress_status {
        Ok(status) if status.success() => {
            println!("✅ Decompression completed successfully");
            Ok("lichess_db_puzzle.csv".to_string())
        }
        Ok(status) => Err(format!(
            "Decompression failed with exit code: {}",
            status.code().unwrap_or(-1)
        )
        .into()),
        Err(e) => Err(format!("Decompression command failed: {}", e).into()),
    }
}

fn _get_csv_line_count(file_path: &str) -> Result<usize, Box<dyn std::error::Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let mut count: usize = 0;

    for line in reader.lines() {
        line?;
        count += 1;
    }

    // Subtract 1 for header line
    Ok(count.saturating_sub(1))
}

fn _create_progress_bar(total: usize, message: &str) -> ProgressBar {
    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(&format!(
                "{} {{bar:40.cyan/blue}} {{pos:>7}}/{{len:7}} {{msg}}",
                message
            ))
            .unwrap()
            .progress_chars("##-"),
    );
    pb
}
