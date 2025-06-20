use chess_vector_engine::{ChessVectorEngine, GPUAccelerator};
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};
use std::time::Instant;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "comprehensive_training")]
#[command(about = "Complete chess engine training pipeline with GPU acceleration and persistence")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run complete training pipeline (recommended)
    Complete {
        /// Number of self-play games to generate
        #[arg(long, default_value = "10000")]
        games: usize,
        
        /// Number of training iterations
        #[arg(long, default_value = "50")]
        iterations: usize,
        
        /// Path to Lichess puzzle CSV file
        #[arg(long)]
        puzzles: Option<String>,
        
        /// Maximum puzzles to load
        #[arg(long, default_value = "100000")]
        max_puzzles: usize,
        
        /// Output directory for all training artifacts
        #[arg(long, default_value = "training_output")]
        output_dir: String,
        
        /// Force GPU acceleration (fail if not available)
        #[arg(long)]
        force_gpu: bool,
        
        /// Use multi-GPU acceleration when available
        #[arg(long)]
        multi_gpu: bool,
        
        /// Enable all optimizations (LSH, manifold learning, binary formats)
        #[arg(long, default_value = "true")]
        enable_optimizations: bool,
        
        /// Enable intelligent position curation to reduce dataset size
        #[arg(long)]
        enable_curation: bool,
        
        /// Target number of positions after curation
        #[arg(long, default_value = "100000")]
        curation_target: usize,
        
        /// Minimum tactical score for curation
        #[arg(long, default_value = "0.3")]
        curation_min_score: f32,
    },
    
    /// Run specific training phase only
    Phase {
        #[command(subcommand)]
        phase: TrainingPhase,
    },
    
    /// Convert existing training data to optimized formats
    Optimize {
        /// Input directory containing training files
        #[arg(long, default_value = ".")]
        input_dir: String,
        
        /// Output directory for optimized files
        #[arg(long, default_value = "optimized")]
        output_dir: String,
    },
}

#[derive(Subcommand)]
enum TrainingPhase {
    /// Self-play game generation
    SelfPlay {
        #[arg(long, default_value = "5000")]
        games: usize,
        #[arg(long, default_value = "training_output")]
        output_dir: String,
    },
    
    /// Tactical puzzle training
    Tactical {
        #[arg(long)]
        puzzles_file: String,
        #[arg(long, default_value = "50000")]
        max_puzzles: usize,
        #[arg(long, default_value = "training_output")]
        output_dir: String,
    },
    
    /// Neural network training (LSH + Manifold learning)
    Neural {
        #[arg(long, default_value = "training_output")]
        input_dir: String,
        #[arg(long, default_value = "8.0")]
        compression_ratio: f32,
        #[arg(long, default_value = "50")]
        epochs: usize,
    },
    
    /// Model optimization and export
    Export {
        #[arg(long, default_value = "training_output")]
        input_dir: String,
        #[arg(long, default_value = "exported_model")]
        output_file: String,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    
    // Initialize GPU acceleration
    let gpu = GPUAccelerator::global();
    println!("🖥️  Compute device: {:?} ({} device(s))", 
             gpu.device_type(), gpu.device_count());
    
    if gpu.is_multi_gpu_available() {
        println!("🚀 Multi-GPU acceleration available with {} devices!", gpu.device_count());
    }
    
    match cli.command {
        Commands::Complete { 
            games, iterations, puzzles, max_puzzles, output_dir, 
            force_gpu, multi_gpu, enable_optimizations,
            enable_curation, curation_target, curation_min_score
        } => {
            run_complete_training(ComprehensiveTrainingConfig {
                games,
                iterations,
                puzzles_file: puzzles,
                max_puzzles,
                output_dir,
                force_gpu,
                multi_gpu,
                enable_optimizations,
                enable_curation,
                curation_target,
                curation_min_score,
            })
        }
        Commands::Phase { phase } => {
            run_training_phase(phase)
        }
        Commands::Optimize { input_dir, output_dir } => {
            run_optimization(input_dir, output_dir)
        }
    }
}

struct ComprehensiveTrainingConfig {
    games: usize,
    iterations: usize,
    puzzles_file: Option<String>,
    max_puzzles: usize,
    output_dir: String,
    force_gpu: bool,
    multi_gpu: bool,
    enable_optimizations: bool,
    enable_curation: bool,
    curation_target: usize,
    curation_min_score: f32,
}

fn run_complete_training(config: ComprehensiveTrainingConfig) -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    
    // Validate GPU requirements
    let gpu = GPUAccelerator::global();
    if config.force_gpu && !gpu.is_gpu_enabled() {
        return Err("GPU acceleration required but not available".into());
    }
    
    if config.multi_gpu && !gpu.is_multi_gpu_available() {
        println!("⚠️  Multi-GPU requested but only {} device(s) available", gpu.device_count());
    }
    
    // Create output directory
    std::fs::create_dir_all(&config.output_dir)?;
    
    // Initialize comprehensive progress tracking
    let multi_progress = Arc::new(MultiProgress::new());
    let total_phases = if config.enable_curation { 7 } else { 6 };
    let main_pb = multi_progress.add(ProgressBar::new(total_phases));
    main_pb.set_style(
        ProgressStyle::default_bar()
            .template("🎯 Overall Progress [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")?
            .progress_chars("██░")
    );
    
    println!("🚀 Starting comprehensive chess engine training pipeline");
    println!("📊 Configuration:");
    println!("   - Games: {}", config.games);
    println!("   - Iterations: {}", config.iterations);
    println!("   - Puzzles: {}", config.puzzles_file.as_ref().map_or("None", |s| s));
    println!("   - GPU: {} (Multi-GPU: {})", gpu.is_gpu_enabled(), config.multi_gpu);
    println!("   - Curation: {} (Target: {})", config.enable_curation, config.curation_target);
    println!("   - Output: {}", config.output_dir);
    println!();
    
    // Phase 1: Initialize engine with optimizations
    main_pb.set_message("Initializing engine");
    let mut engine = if config.enable_optimizations {
        println!("📦 Attempting fast loading with existing data...");
        ChessVectorEngine::new_with_fast_load(1024).unwrap_or_else(|_| {
            println!("🆕 Creating fresh engine");
            ChessVectorEngine::new(1024)
        })
    } else {
        ChessVectorEngine::new(1024)
    };
    
    engine.enable_opening_book();
    main_pb.inc(1);
    
    // Phase 2: Self-play training
    if config.games > 0 {
        main_pb.set_message("Self-play training");
        println!("🎮 Starting self-play training: {} games across {} iterations", 
                 config.games, config.iterations);
        
        run_self_play_training(&mut engine, config.games, config.iterations, &config.output_dir)?;
    }
    main_pb.inc(1);
    
    // Phase 3: Tactical puzzle training
    if let Some(puzzles_file) = &config.puzzles_file {
        main_pb.set_message("Tactical training");
        println!("🎯 Loading tactical puzzles from {}", puzzles_file);
        
        run_tactical_training(&mut engine, puzzles_file, config.max_puzzles, &config.output_dir)?;
    }
    main_pb.inc(1);
    
    // Phase 4: Intelligent position curation (if enabled)
    if config.enable_curation {
        main_pb.set_message("Position curation");
        println!("🧠 Running intelligent position curation");
        
        run_position_curation(&mut engine, &config)?;
        main_pb.inc(1);
    }
    
    // Phase 5: Neural network optimizations
    if config.enable_optimizations {
        main_pb.set_message("Neural optimizations");
        println!("🧠 Training neural networks (LSH + Manifold learning)");
        
        run_neural_training(&mut engine, &config)?;
    }
    main_pb.inc(1);
    
    // Phase 6: Export optimized models
    main_pb.set_message("Exporting models");
    println!("📦 Exporting optimized models and data");
    
    export_comprehensive_model(&engine, &config.output_dir)?;
    main_pb.inc(1);
    
    // Phase 7: Validation and benchmarking
    main_pb.set_message("Validation");
    println!("✅ Running validation and benchmarks");
    
    run_validation(&mut engine)?;
    main_pb.inc(1);
    
    main_pb.finish_with_message("✅ Training complete!");
    
    let total_time = start_time.elapsed();
    println!();
    println!("🎉 Comprehensive training completed!");
    println!("⏱️  Total time: {:?}", total_time);
    println!("📊 Final engine stats:");
    
    let stats = engine.training_stats();
    println!("   - Total positions: {}", stats.total_positions);
    println!("   - Move data entries: {}", stats.move_data_entries);
    println!("   - LSH enabled: {}", stats.lsh_enabled);
    println!("   - Manifold enabled: {}", stats.manifold_enabled);
    
    if gpu.is_gpu_enabled() {
        println!("   - GPU device: {:?}", gpu.device_type());
        if let Ok(gflops) = gpu.benchmark() {
            println!("   - GPU performance: {:.2} GFLOPS", gflops);
        }
    }
    
    println!();
    println!("📁 Output files in: {}", config.output_dir);
    println!("🚀 Use the exported model for fast loading in production!");
    
    Ok(())
}

fn run_self_play_training(
    engine: &mut ChessVectorEngine, 
    games: usize, 
    iterations: usize, 
    output_dir: &str
) -> Result<(), Box<dyn std::error::Error>> {
    
    use std::process::Command;
    
    // Run self-play training with all optimizations
    let output_file = format!("{}/self_play_training.json", output_dir);
    
    let mut cmd = Command::new("cargo");
    cmd.arg("run")
       .arg("--release")
       .arg("--features")
       .arg("cuda")
       .arg("--bin")
       .arg("self_play_training")
       .arg("--")
       .arg("--games")
       .arg(&games.to_string())
       .arg("--iterations")
       .arg(&iterations.to_string())
       .arg("--output")
       .arg(&output_file)
       .arg("--enable-lsh")
       .arg("--enable-manifold")
       .arg("--enable-persistence");
    
    let status = cmd.status()?;
    if !status.success() {
        return Err("Self-play training failed".into());
    }
    
    // Load the results into our engine
    if std::path::Path::new(&output_file).exists() {
        engine.load_training_data_incremental(&output_file)?;
    }
    
    Ok(())
}

fn run_tactical_training(
    engine: &mut ChessVectorEngine,
    puzzles_file: &str,
    max_puzzles: usize,
    output_dir: &str
) -> Result<(), Box<dyn std::error::Error>> {
    
    use std::process::Command;
    
    let output_file = format!("{}/tactical_training.json", output_dir);
    
    let mut cmd = Command::new("cargo");
    cmd.arg("run")
       .arg("--release")
       .arg("--features")
       .arg("cuda")
       .arg("--bin")
       .arg("tactical_training")
       .arg("--")
       .arg("--puzzles")
       .arg(puzzles_file)
       .arg("--max-puzzles")
       .arg(&max_puzzles.to_string())
       .arg("--output")
       .arg(&output_file);
    
    let status = cmd.status()?;
    if !status.success() {
        return Err("Tactical training failed".into());
    }
    
    // Load tactical training results
    if std::path::Path::new(&output_file).exists() {
        engine.load_training_data_incremental(&output_file)?;
    }
    
    Ok(())
}

fn run_position_curation(
    engine: &mut ChessVectorEngine,
    config: &ComprehensiveTrainingConfig
) -> Result<(), Box<dyn std::error::Error>> {
    
    let stats = engine.training_stats();
    
    // Only run curation if we have more positions than the target
    if stats.total_positions > config.curation_target {
        println!("📊 Running intelligent curation: {} → {} positions", 
                 stats.total_positions, config.curation_target);
        
        // Export current training data
        let temp_input = format!("{}/temp_positions_for_curation.json", config.output_dir);
        let curated_output = format!("{}/curated_positions.json", config.output_dir);
        
        engine.save_training_data(&temp_input)?;
        
        // Run intelligent curation
        let mut cmd = std::process::Command::new("cargo");
        cmd.arg("run")
           .arg("--release")
           .arg("--features")
           .arg("cuda")
           .arg("--bin")
           .arg("intelligent_curation")
           .arg("--")
           .arg("--input")
           .arg(&temp_input)
           .arg("--output")
           .arg(&curated_output)
           .arg("--target-size")
           .arg(&config.curation_target.to_string())
           .arg("--min-tactical-score")
           .arg(&config.curation_min_score.to_string());
        
        let status = cmd.status()?;
        if !status.success() {
            return Err("Intelligent curation failed".into());
        }
        
        // Replace engine data with curated positions
        *engine = ChessVectorEngine::new(1024);
        engine.enable_opening_book();
        engine.load_training_data_incremental(&curated_output)?;
        
        // Clean up temporary files
        let _ = std::fs::remove_file(temp_input);
        
        println!("✅ Curation complete: retained {} high-quality positions", config.curation_target);
    } else {
        println!("ℹ️  Skipping curation: {} positions already under target of {}", 
                 stats.total_positions, config.curation_target);
    }
    
    Ok(())
}

fn run_neural_training(
    engine: &mut ChessVectorEngine,
    config: &ComprehensiveTrainingConfig
) -> Result<(), Box<dyn std::error::Error>> {
    
    let stats = engine.training_stats();
    
    // Enable LSH for large datasets
    if stats.total_positions > 10000 {
        println!("📊 Enabling LSH for {} positions", stats.total_positions);
        let tables = if config.multi_gpu { 16 } else { 12 };
        let hash_size = if stats.total_positions > 100000 { 24 } else { 20 };
        engine.enable_lsh(tables, hash_size);
    }
    
    // Enable manifold learning for very large datasets
    if stats.total_positions > 30000 {
        println!("🧠 Enabling manifold learning for {} positions", stats.total_positions);
        let compression_ratio = if config.multi_gpu { 16.0 } else { 8.0 };
        engine.enable_manifold_learning(compression_ratio)?;
        
        let epochs = if config.multi_gpu { 100 } else { 50 };
        println!("🏋️  Training manifold learning ({} epochs)", epochs);
        engine.train_manifold_learning(epochs)?;
    }
    
    Ok(())
}

fn export_comprehensive_model(
    engine: &ChessVectorEngine,
    output_dir: &str
) -> Result<(), Box<dyn std::error::Error>> {
    
    // Save in multiple formats for flexibility
    let binary_file = format!("{}/comprehensive_model.bin", output_dir);
    let json_file = format!("{}/comprehensive_model.json", output_dir);
    
    engine.save_training_data_binary(&binary_file)?;
    engine.save_training_data(&json_file)?;
    
    // Create a summary file
    let summary_file = format!("{}/training_summary.json", output_dir);
    let stats = engine.training_stats();
    let summary = serde_json::json!({
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "total_positions": stats.total_positions,
        "unique_positions": stats.unique_positions,
        "move_data_entries": stats.move_data_entries,
        "lsh_enabled": stats.lsh_enabled,
        "manifold_enabled": stats.manifold_enabled,
        "opening_book_enabled": stats.opening_book_enabled,
        "gpu_info": {
            "device_type": format!("{:?}", GPUAccelerator::global().device_type()),
            "device_count": GPUAccelerator::global().device_count(),
            "multi_gpu_available": GPUAccelerator::global().is_multi_gpu_available()
        }
    });
    
    std::fs::write(summary_file, serde_json::to_string_pretty(&summary)?)?;
    
    println!("✅ Exported comprehensive model:");
    println!("   - Binary: {}", binary_file);
    println!("   - JSON: {}", json_file);
    
    Ok(())
}

fn run_validation(engine: &mut ChessVectorEngine) -> Result<(), Box<dyn std::error::Error>> {
    use chess::Board;
    
    // Test engine performance with a few positions
    let test_positions = vec![
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 3",
    ];
    
    println!("🧪 Testing engine performance...");
    
    for (i, fen) in test_positions.iter().enumerate() {
        if let Ok(board) = fen.parse::<Board>() {
            let start = Instant::now();
            let similar = engine.find_similar_positions(&board, 5);
            let duration = start.elapsed();
            
            println!("   Test {}: {} similar positions found in {:?}", 
                     i + 1, similar.len(), duration);
            
            if let Some(eval) = engine.evaluate_position(&board) {
                println!("   Position evaluation: {:.3}", eval);
            }
        }
    }
    
    Ok(())
}

fn run_training_phase(phase: TrainingPhase) -> Result<(), Box<dyn std::error::Error>> {
    match phase {
        TrainingPhase::SelfPlay { games, output_dir } => {
            println!("🎮 Running self-play phase: {} games", games);
            std::fs::create_dir_all(&output_dir)?;
            
            let mut engine = ChessVectorEngine::new(1024);
            run_self_play_training(&mut engine, games, 10, &output_dir)?;
        }
        
        TrainingPhase::Tactical { puzzles_file, max_puzzles, output_dir } => {
            println!("🎯 Running tactical training phase");
            std::fs::create_dir_all(&output_dir)?;
            
            let mut engine = ChessVectorEngine::new(1024);
            run_tactical_training(&mut engine, &puzzles_file, max_puzzles, &output_dir)?;
        }
        
        TrainingPhase::Neural { input_dir, compression_ratio, epochs } => {
            println!("🧠 Running neural training phase");
            
            let mut engine = ChessVectorEngine::new_with_fast_load(1024)?;
            engine.enable_manifold_learning(compression_ratio)?;
            engine.train_manifold_learning(epochs)?;
        }
        
        TrainingPhase::Export { input_dir, output_file } => {
            println!("📦 Running export phase");
            
            let engine = ChessVectorEngine::new_with_fast_load(1024)?;
            export_comprehensive_model(&engine, &input_dir)?;
        }
    }
    
    Ok(())
}

fn run_optimization(input_dir: String, output_dir: String) -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Optimizing training data formats");
    
    std::fs::create_dir_all(&output_dir)?;
    
    // Convert JSON to binary
    println!("🔄 Converting JSON files to binary format...");
    let converted = ChessVectorEngine::convert_json_to_binary()?;
    
    for conversion in converted {
        println!("   ✅ {}", conversion);
    }
    
    println!("🚀 Optimization complete!");
    
    Ok(())
}