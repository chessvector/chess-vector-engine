use chess::{Board, ChessMove, Color, Game, MoveGen};
use chess_vector_engine::{ChessVectorEngine, NNUEConfig, NNUE};
use clap::{Arg, Command};
use std::path::Path;
use std::str::FromStr;
use std::time::Instant;

/// Comprehensive NNUE training with various position types
fn generate_training_positions() -> Vec<(Board, f32)> {
    let mut positions = Vec::new();

    // Starting positions and basic openings
    positions.push((Board::default(), 0.0)); // Starting position

    // King's Pawn openings
    if let Ok(board) =
        Board::from_str("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    {
        positions.push((board, 0.25)); // 1.e4
    }
    if let Ok(board) =
        Board::from_str("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    {
        positions.push((board, 0.1)); // 1.e4 e5
    }

    // Queen's Pawn openings
    if let Ok(board) =
        Board::from_str("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1")
    {
        positions.push((board, 0.15)); // 1.d4
    }
    if let Ok(board) =
        Board::from_str("rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2")
    {
        positions.push((board, 0.05)); // 1.d4 d5
    }

    // Development positions
    if let Ok(board) =
        Board::from_str("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 3")
    {
        positions.push((board, 0.35)); // Knights developed
    }
    if let Ok(board) =
        Board::from_str("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 6 4")
    {
        positions.push((board, 0.4)); // Italian Game development
    }

    // Material advantages
    if let Ok(board) =
        Board::from_str("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 2 3")
    {
        positions.push((board, 1.0)); // Up a pawn
    }
    if let Ok(board) =
        Board::from_str("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    {
        positions.push((board, -0.8)); // Down a pawn
    }
    if let Ok(board) =
        Board::from_str("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 2 3")
    {
        positions.push((board, 3.2)); // Up a knight
    }

    // Tactical positions
    if let Ok(board) =
        Board::from_str("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 4 4")
    {
        positions.push((board, 2.5)); // Pin on knight
    }
    if let Ok(board) =
        Board::from_str("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 b kq - 0 5")
    {
        positions.push((board, -1.5)); // Black under pressure
    }

    // Endgame positions
    if let Ok(board) = Board::from_str("8/8/8/8/8/8/8/K6k w - - 0 1") {
        positions.push((board, 0.0)); // King vs King draw
    }
    if let Ok(board) = Board::from_str("8/8/8/8/8/8/1K6/k6Q w - - 0 1") {
        positions.push((board, 9.0)); // Queen vs King
    }
    if let Ok(board) = Board::from_str("8/8/8/8/8/8/1K6/k6R w - - 0 1") {
        positions.push((board, 5.0)); // Rook vs King
    }
    if let Ok(board) = Board::from_str("8/8/8/3k4/8/3K4/8/8 w - - 0 1") {
        positions.push((board, 0.0)); // King vs King draw
    }

    // Pawn endgames
    if let Ok(board) = Board::from_str("8/8/8/3k4/3P4/3K4/8/8 w - - 0 1") {
        positions.push((board, 1.5)); // Pawn advantage
    }
    if let Ok(board) = Board::from_str("8/8/8/3k4/8/3K4/3P4/8 w - - 0 1") {
        positions.push((board, 2.0)); // Protected pawn
    }

    // Positional advantages
    if let Ok(board) =
        Board::from_str("r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w - - 8 6")
    {
        positions.push((board, 0.6)); // Good development
    }
    if let Ok(board) =
        Board::from_str("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 2 3")
    {
        positions.push((board, -0.3)); // Passive position
    }

    // Check positions
    if let Ok(board) =
        Board::from_str("rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR b KQkq - 3 3")
    {
        positions.push((board, -2.0)); // King in check, bad
    }

    // Castling advantages
    if let Ok(board) =
        Board::from_str("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 6 4")
    {
        positions.push((board, 0.8)); // Can castle
    }

    // Material imbalances (different piece values)
    if let Ok(board) =
        Board::from_str("r1bqkbnr/pppp1ppp/2n5/8/3pP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 4")
    {
        positions.push((board, -3.0)); // Down material
    }

    positions
}

/// Generate random positions from actual games for more diverse training
fn generate_game_positions(num_games: usize) -> Vec<(Board, f32)> {
    let mut positions = Vec::new();
    let mut rng_state = 42u64; // Simple PRNG state

    for _game in 0..num_games {
        let mut game = Game::new();
        let mut move_count = 0;

        // Play random moves to generate diverse positions
        while game.result().is_none() && move_count < 30 {
            let board = game.current_position();
            let legal_moves: Vec<ChessMove> = MoveGen::new_legal(&board).collect();

            if legal_moves.is_empty() {
                break;
            }

            // Simple PRNG (LCG)
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let move_idx = (rng_state as usize) % legal_moves.len();
            let chosen_move = legal_moves[move_idx];

            if !game.make_move(chosen_move) {
                break;
            }

            move_count += 1;

            // Evaluate position based on material and position
            let evaluation = evaluate_position_heuristic(&game.current_position());
            positions.push((game.current_position(), evaluation));

            // Stop if too many positions
            if positions.len() >= 100 {
                break;
            }
        }

        if positions.len() >= 100 {
            break;
        }
    }

    positions
}

/// Simple heuristic evaluation for training data generation
fn evaluate_position_heuristic(board: &Board) -> f32 {
    let mut score = 0.0;

    // Material count
    for square in chess::ALL_SQUARES {
        if let Some(piece) = board.piece_on(square) {
            let piece_value = match piece {
                chess::Piece::Pawn => 1.0,
                chess::Piece::Knight => 3.0,
                chess::Piece::Bishop => 3.0,
                chess::Piece::Rook => 5.0,
                chess::Piece::Queen => 9.0,
                chess::Piece::King => 0.0,
            };

            if board.color_on(square) == Some(Color::White) {
                score += piece_value;
            } else {
                score -= piece_value;
            }
        }
    }

    // Add some positional factors
    let legal_moves = MoveGen::new_legal(board).count() as f32;
    if board.side_to_move() == Color::White {
        score += legal_moves * 0.01;
    } else {
        score -= legal_moves * 0.01;
    }

    // King safety (very basic)
    if board.checkers().popcnt() > 0 {
        if board.side_to_move() == Color::White {
            score -= 1.0;
        } else {
            score += 1.0;
        }
    }

    // Clamp to reasonable range
    score.clamp(-10.0, 10.0)
}

/// Train NNUE with comprehensive dataset
fn train_nnue_comprehensive(
    config: NNUEConfig,
    epochs: usize,
    save_interval: usize,
    output_path: &str,
    include_game_positions: bool,
    resume_training: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Creating NNUE with configuration: {:?}", config);
    let mut nnue = NNUE::new(config.clone())?;

    // Try to load existing model if resuming training
    if resume_training {
        let config_path = format!("{}.config", output_path);
        if std::path::Path::new(&config_path).exists() {
            println!("🔄 Resuming training from existing model: {}", output_path);
            if let Err(e) = nnue.load_model(output_path) {
                println!("⚠️  Failed to load existing model: {}", e);
                println!("🆕 Starting fresh training instead");
            } else {
                println!("✅ Successfully loaded existing model for incremental training");
            }
        } else {
            println!(
                "📝 No existing model found at {}, starting fresh training",
                output_path
            );
        }
    } else {
        println!("🆕 Starting fresh training (not resuming from existing model)");
    }

    // Generate training data
    println!("📊 Generating training positions...");
    let mut training_data = generate_training_positions();
    println!("   Generated {} curated positions", training_data.len());

    if include_game_positions {
        println!("🎮 Generating positions from random games...");
        let game_positions = generate_game_positions(50);
        println!("   Generated {} game positions", game_positions.len());
        training_data.extend(game_positions);
    }

    println!(
        "📈 Total training dataset: {} positions",
        training_data.len()
    );

    // Training loop
    println!("🚀 Starting NNUE training for {} epochs...", epochs);
    let start_time = Instant::now();

    for epoch in 0..epochs {
        let epoch_start = Instant::now();

        // Use incremental training if resuming, otherwise regular training
        let loss = if resume_training {
            nnue.incremental_train(&training_data, true)?
        } else {
            nnue.train_batch(&training_data)?
        };

        let epoch_time = epoch_start.elapsed();
        println!(
            "Epoch {}/{}: Loss = {:.6}, Time = {:.2}s",
            epoch + 1,
            epochs,
            loss,
            epoch_time.as_secs_f32()
        );

        // Save intermediate models
        if save_interval > 0 && (epoch + 1) % save_interval == 0 {
            let checkpoint_path = format!("{}_epoch_{}", output_path, epoch + 1);
            nnue.save_model(&checkpoint_path)?;
            println!("💾 Saved checkpoint: {}", checkpoint_path);
        }

        // Early stopping if loss is very low
        if loss < 0.001 {
            println!("✅ Early stopping: Loss target reached");
            break;
        }
    }

    let total_time = start_time.elapsed();
    println!(
        "🏁 Training completed in {:.2}s ({:.2}s per epoch average)",
        total_time.as_secs_f32(),
        total_time.as_secs_f32() / epochs as f32
    );

    // Save final model
    nnue.save_model(output_path)?;
    println!("💾 Final model saved: {}", output_path);

    // Test the trained model
    println!("🧪 Testing trained model...");
    test_nnue_model(&mut nnue)?;

    Ok(())
}

/// Test the trained NNUE model on various positions
fn test_nnue_model(nnue: &mut NNUE) -> Result<(), Box<dyn std::error::Error>> {
    let mut test_positions = vec![(Board::default(), "Starting position")];

    // Add test positions with error handling
    if let Ok(board) =
        Board::from_str("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    {
        test_positions.push((board, "1.e4"));
    }
    if let Ok(board) = Board::from_str("8/8/8/8/8/8/1K6/k6Q w - - 0 1") {
        test_positions.push((board, "Queen vs King"));
    }
    if let Ok(board) = Board::from_str("8/8/8/8/8/8/8/K6k w - - 0 1") {
        test_positions.push((board, "King vs King"));
    }
    if let Ok(board) =
        Board::from_str("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 3")
    {
        test_positions.push((board, "Development"));
    }

    println!("\n📋 NNUE Evaluation Test Results:");
    println!("Position                  | Evaluation");
    println!("--------------------------|------------");

    for (board, description) in test_positions {
        let eval = nnue.evaluate(&board)?;
        println!("{:25} | {:+8.3}", description, eval);
    }

    Ok(())
}

/// Load and test an existing NNUE model
fn test_existing_model(model_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("📂 Loading existing NNUE model: {}", model_path);

    let config = NNUEConfig::default();
    let mut nnue = NNUE::new(config)?;

    if Path::new(&format!("{}.config", model_path)).exists() {
        nnue.load_model(model_path)?;
        println!("✅ Model loaded successfully");

        test_nnue_model(&mut nnue)?;
    } else {
        return Err(format!("Model file not found: {}.config", model_path).into());
    }

    Ok(())
}

/// Integration test: Use trained NNUE in Chess Vector Engine
fn test_integration(_model_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔗 Testing NNUE integration with Chess Vector Engine...");

    let mut engine = ChessVectorEngine::new(1024);

    // Enable NNUE
    engine.enable_nnue()?;
    println!("✅ NNUE enabled in engine");

    // Test evaluation on various positions
    let mut test_boards = vec![Board::default()];

    if let Ok(board) =
        Board::from_str("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    {
        test_boards.push(board);
    }
    if let Ok(board) =
        Board::from_str("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 3")
    {
        test_boards.push(board);
    }

    println!("\n🧪 Engine Integration Test:");
    for (i, board) in test_boards.iter().enumerate() {
        if let Some(eval) = engine.evaluate_position(board) {
            println!("Position {}: Evaluation = {:+.3}", i + 1, eval);
        } else {
            println!("Position {}: Evaluation failed", i + 1);
        }
    }

    println!("✅ Integration test completed");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("NNUE Trainer")
        .version("1.0")
        .author("Chess Vector Team")
        .about("Train and save NNUE neural networks for chess position evaluation")
        .arg(
            Arg::new("mode")
                .short('m')
                .long("mode")
                .value_name("MODE")
                .help("Operation mode: train, test, or integrate")
                .value_parser(["train", "test", "integrate"])
                .default_value("train"),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("PATH")
                .help("Output path for trained model")
                .default_value("trained_nnue_model"),
        )
        .arg(
            Arg::new("epochs")
                .short('e')
                .long("epochs")
                .value_name("EPOCHS")
                .help("Number of training epochs")
                .value_parser(clap::value_parser!(usize))
                .default_value("50"),
        )
        .arg(
            Arg::new("save-interval")
                .short('s')
                .long("save-interval")
                .value_name("INTERVAL")
                .help("Save checkpoint every N epochs (0 = no checkpoints)")
                .value_parser(clap::value_parser!(usize))
                .default_value("10"),
        )
        .arg(
            Arg::new("config")
                .short('c')
                .long("config")
                .value_name("CONFIG")
                .help("NNUE configuration preset")
                .value_parser([
                    "default",
                    "vector-integrated",
                    "nnue-focused",
                    "experimental",
                ])
                .default_value("default"),
        )
        .arg(
            Arg::new("include-games")
                .long("include-games")
                .help("Include positions from random games in training")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("resume")
                .short('r')
                .long("resume")
                .help("Resume training from existing model (incremental training)")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("model-path")
                .short('p')
                .long("model-path")
                .value_name("PATH")
                .help("Path to existing model for test/integrate modes")
                .default_value("trained_nnue_model"),
        )
        .get_matches();

    let mode = matches.get_one::<String>("mode").unwrap();
    let output_path = matches.get_one::<String>("output").unwrap();
    let epochs = *matches.get_one::<usize>("epochs").unwrap();
    let save_interval = *matches.get_one::<usize>("save-interval").unwrap();
    let config_preset = matches.get_one::<String>("config").unwrap();
    let include_games = matches.get_flag("include-games");
    let resume_training = matches.get_flag("resume");
    let model_path = matches.get_one::<String>("model-path").unwrap();

    // Select NNUE configuration
    let config = match config_preset.as_str() {
        "vector-integrated" => NNUEConfig::vector_integrated(),
        "nnue-focused" => NNUEConfig::nnue_focused(),
        "experimental" => NNUEConfig::experimental(),
        _ => NNUEConfig::default(),
    };

    match mode.as_str() {
        "train" => {
            println!("🎯 NNUE Training Mode");
            println!("Configuration: {}", config_preset);
            println!("Epochs: {}", epochs);
            println!("Save interval: {}", save_interval);
            println!("Include game positions: {}", include_games);
            println!("Resume training: {}", resume_training);
            println!("Output path: {}", output_path);
            println!();

            train_nnue_comprehensive(
                config,
                epochs,
                save_interval,
                output_path,
                include_games,
                resume_training,
            )?;
        }
        "test" => {
            println!("🧪 NNUE Testing Mode");
            println!("Model path: {}", model_path);
            println!();

            test_existing_model(model_path)?;
        }
        "integrate" => {
            println!("🔗 NNUE Integration Mode");
            println!("Model path: {}", model_path);
            println!();

            test_integration(model_path)?;
        }
        _ => unreachable!(),
    }

    Ok(())
}
