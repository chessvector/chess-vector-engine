use chess::{Board, MoveGen};
use chess_vector_engine::{ChessVectorEngine, manifold_learner::ManifoldLearner};
use ndarray::Array2;
use rand::Rng;
use std::time::Instant;

fn main() {
    println!("Chess Vector Engine - Manifold Learning Demo");
    println!("============================================");
    
    // Create chess vector engine
    let engine = ChessVectorEngine::new(1024);
    
    // Generate training data
    println!("Generating training data from chess positions...");
    let num_positions = 100;
    let mut training_vectors = Vec::new();
    
    let start = Instant::now();
    for i in 0..num_positions {
        let board = generate_random_position().expect("Valid position");
        let vector = engine.encode_position(&board);
        training_vectors.push(vector.to_vec());
        
        if (i + 1) % 25 == 0 {
            println!("Generated {} positions", i + 1);
        }
    }
    let generation_time = start.elapsed();
    println!("Generated {} positions in {:?}", num_positions, generation_time);
    
    // Convert to Array2 for training
    let data_flat: Vec<f32> = training_vectors.into_iter().flatten().collect();
    let training_data = Array2::from_shape_vec((num_positions, 1024), data_flat)
        .expect("Failed to create training array");
    
    println!("\nTraining Data Statistics:");
    println!("  Shape: {:?}", training_data.dim());
    println!("  Original dimension: 1024");
    
    // Test different compression ratios
    let compression_targets = vec![
        (128, "8:1 compression"),
        (64, "16:1 compression"), 
        (32, "32:1 compression"),
    ];
    
    for (manifold_dim, description) in compression_targets {
        println!("\n=== Testing {} ===", description);
        
        // Create and train manifold learner
        let mut learner = ManifoldLearner::new(1024, manifold_dim);
        
        println!("Training autoencoder...");
        let train_start = Instant::now();
        match learner.train(&training_data, 5) {
            Ok(_) => {
                let train_time = train_start.elapsed();
                println!("Training completed in {:?}", train_time);
                
                // Test compression and reconstruction
                test_compression(&learner, &training_data, 10, manifold_dim);
            }
            Err(e) => {
                println!("Training failed: {}", e);
            }
        }
    }
    
    println!("\n=== Compression Benefits ===");
    println!("1. Storage: Reduce memory usage by 8-32x");
    println!("2. Speed: Faster similarity search in lower dimensions");
    println!("3. Generalization: Learn chess position structure");
    println!("4. Interpolation: Navigate between similar positions");
}

fn generate_random_position() -> Result<Board, Box<dyn std::error::Error>> {
    let mut rng = rand::thread_rng();
    let mut board = Board::default();
    let moves_to_play = rng.gen_range(0..15);
    
    for _ in 0..moves_to_play {
        let legal_moves: Vec<_> = MoveGen::new_legal(&board).collect();
        if legal_moves.is_empty() {
            break;
        }
        
        let random_move = legal_moves[rng.gen_range(0..legal_moves.len())];
        board = board.make_move_new(random_move);
    }
    
    Ok(board)
}

fn test_compression(learner: &ManifoldLearner, data: &Array2<f32>, num_samples: usize, manifold_dim: usize) {
    println!("Testing compression quality...");
    
    let mut total_mse = 0.0;
    let mut total_compression_ratio = 0.0;
    
    for i in 0..num_samples.min(data.nrows()) {
        let original = data.row(i).to_owned();
        
        // Encode to manifold space
        let encoded = learner.encode(&original);
        
        // Decode back to original space  
        let reconstructed = learner.decode(&encoded);
        
        // Calculate reconstruction error (MSE)
        let mse: f32 = original.iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>() / original.len() as f32;
        
        total_mse += mse;
        total_compression_ratio += 1024.0 / manifold_dim as f32;
    }
    
    let avg_mse = total_mse / num_samples as f32;
    let avg_compression = total_compression_ratio / num_samples as f32;
    
    println!("Compression Results:");
    println!("  Compression ratio: {:.1}:1", avg_compression);
    println!("  Reconstruction MSE: {:.6}", avg_mse);
    println!("  Memory savings: {:.1}%", (1.0 - 1.0/avg_compression) * 100.0);
    
    // Quality assessment
    let quality = if avg_mse < 0.01 {
        "Excellent"
    } else if avg_mse < 0.1 {
        "Good"
    } else if avg_mse < 1.0 {
        "Fair"
    } else {
        "Poor"
    };
    
    println!("  Quality assessment: {}", quality);
}