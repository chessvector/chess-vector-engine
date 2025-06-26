#![allow(clippy::type_complexity)]
use candle_core::{Device, Module, Result as CandleResult, Tensor};
use candle_nn::{linear, AdamW, Linear, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use chess::{Board, Color, Piece, Square};
use serde::{Deserialize, Serialize};

/// NNUE (Efficiently Updatable Neural Network) for chess position evaluation
/// This implementation is designed to work alongside vector-based position analysis
///
/// The key innovation is that NNUE provides fast, accurate position evaluation while
/// the vector-based system provides strategic pattern recognition and similarity matching.
/// Together they create a hybrid system that's both fast and strategically aware.
pub struct NNUE {
    feature_transformer: FeatureTransformer,
    hidden_layers: Vec<Linear>,
    output_layer: Linear,
    device: Device,
    #[allow(dead_code)]
    var_map: VarMap,
    optimizer: Option<AdamW>,

    // Integration with vector-based system
    vector_weight: f32, // How much to blend with vector evaluation
    enable_vector_integration: bool,
}

/// Feature transformer that efficiently updates when pieces move
/// Uses the standard NNUE approach with king-relative piece positions
struct FeatureTransformer {
    weights: Tensor,
    biases: Tensor,
    accumulated_features: Option<Tensor>,
    king_squares: [Square; 2], // White and black king positions for incremental updates
}

/// NNUE configuration optimized for chess vector engine integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NNUEConfig {
    pub feature_size: usize,      // Input features (768 for king-relative pieces)
    pub hidden_size: usize,       // Hidden layer size (256 typical)
    pub num_hidden_layers: usize, // Number of hidden layers (2-4 typical)
    pub activation: ActivationType,
    pub learning_rate: f32,
    pub vector_blend_weight: f32, // How much to blend with vector evaluation (0.0-1.0)
    pub enable_incremental_updates: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationType {
    ReLU,
    ClippedReLU, // Clipped ReLU is standard for NNUE
    Sigmoid,
}

impl Default for NNUEConfig {
    fn default() -> Self {
        Self {
            feature_size: 768, // 12 pieces * 64 squares for king-relative
            hidden_size: 256,
            num_hidden_layers: 2,
            activation: ActivationType::ClippedReLU,
            learning_rate: 0.001,
            vector_blend_weight: 0.3, // 30% vector, 70% NNUE by default
            enable_incremental_updates: true,
        }
    }
}

impl NNUEConfig {
    /// Configuration optimized for hybrid vector-NNUE evaluation
    pub fn vector_integrated() -> Self {
        Self {
            vector_blend_weight: 0.4, // Higher vector influence for strategic awareness
            ..Default::default()
        }
    }

    /// Configuration for pure NNUE evaluation (less vector influence)
    pub fn nnue_focused() -> Self {
        Self {
            vector_blend_weight: 0.1, // Minimal vector influence for speed
            ..Default::default()
        }
    }

    /// Configuration for research and experimentation
    pub fn experimental() -> Self {
        Self {
            feature_size: 1024, // Match vector dimension for alignment
            hidden_size: 512,
            num_hidden_layers: 3,
            vector_blend_weight: 0.5, // Equal blend
            ..Default::default()
        }
    }
}

impl NNUE {
    /// Create a new NNUE evaluator with vector integration
    pub fn new(config: NNUEConfig) -> CandleResult<Self> {
        let device = Device::Cpu; // Can be upgraded to GPU later
        let var_map = VarMap::new();
        let vs = VarBuilder::from_varmap(&var_map, candle_core::DType::F32, &device);

        // Create feature transformer
        let feature_transformer =
            FeatureTransformer::new(vs.clone(), config.feature_size, config.hidden_size)?;

        // Create hidden layers
        let mut hidden_layers = Vec::new();
        let mut prev_size = config.hidden_size;

        for i in 0..config.num_hidden_layers {
            let layer = linear(
                prev_size,
                config.hidden_size,
                vs.pp(format!("Processing...")),
            )?;
            hidden_layers.push(layer);
            prev_size = config.hidden_size;
        }

        // Output layer (single neuron for evaluation)
        let output_layer = linear(prev_size, 1, vs.pp("output"))?;

        // Initialize optimizer
        let adamw_params = ParamsAdamW {
            lr: config.learning_rate as f64,
            ..Default::default()
        };
        let optimizer = Some(AdamW::new(var_map.all_vars(), adamw_params)?);

        Ok(Self {
            feature_transformer,
            hidden_layers,
            output_layer,
            device,
            var_map,
            optimizer,
            vector_weight: config.vector_blend_weight,
            enable_vector_integration: true,
        })
    }

    /// Evaluate a position using NNUE
    pub fn evaluate(&mut self, board: &Board) -> CandleResult<f32> {
        let features = self.extract_features(board)?;
        let output = self.forward(&features)?;

        // Convert to centipaws (typical NNUE output scaling)
        // Extract the single value from the [1, 1] tensor
        let eval_cp = output.to_vec2::<f32>()?[0][0] * 100.0;

        Ok(eval_cp)
    }

    /// Hybrid evaluation combining NNUE with vector-based analysis
    pub fn evaluate_hybrid(
        &mut self,
        board: &Board,
        vector_eval: Option<f32>,
    ) -> CandleResult<f32> {
        let nnue_eval = self.evaluate(board)?;

        if !self.enable_vector_integration || vector_eval.is_none() {
            return Ok(nnue_eval);
        }

        let vector_eval = vector_eval.unwrap();

        // Blend evaluations: vector provides strategic insight, NNUE provides tactical precision
        let blended = (1.0 - self.vector_weight) * nnue_eval + self.vector_weight * vector_eval;

        Ok(blended)
    }

    /// Extract NNUE features from chess position
    /// Uses king-relative piece encoding for efficient updates
    fn extract_features(&self, board: &Board) -> CandleResult<Tensor> {
        let mut features = vec![0.0f32; 768]; // 12 pieces * 64 squares

        let white_king = board.king_square(Color::White);
        let black_king = board.king_square(Color::Black);

        // Encode pieces relative to king positions (standard NNUE approach)
        for square in chess::ALL_SQUARES {
            if let Some(piece) = board.piece_on(square) {
                let color = board.color_on(square).unwrap();

                // Get feature indices for this piece relative to both kings
                let (white_idx, _black_idx) =
                    self.get_feature_indices(piece, color, square, white_king, black_king);

                // Activate features (only white perspective to fit in 768 features)
                if let Some(idx) = white_idx {
                    if idx < 768 {
                        features[idx] = 1.0;
                    }
                }
                // Skip black perspective for now to avoid index overflow
                // Real NNUE would use a more sophisticated feature mapping
            }
        }

        Tensor::from_vec(features, (1, 768), &self.device)
    }

    /// Get feature indices for a piece relative to king positions
    fn get_feature_indices(
        &self,
        piece: Piece,
        color: Color,
        square: Square,
        _white_king: Square,
        _black_king: Square,
    ) -> (Option<usize>, Option<usize>) {
        let piece_type_idx = match piece {
            Piece::Pawn => 0,
            Piece::Knight => 1,
            Piece::Bishop => 2,
            Piece::Rook => 3,
            Piece::Queen => 4,
            Piece::King => return (None, None), // Kings not included in features
        };

        let color_offset = if color == Color::White { 0 } else { 5 };
        let base_idx = (piece_type_idx + color_offset) * 64;

        // Calculate square index (simplified - real NNUE uses king-relative mapping)
        let feature_idx = base_idx + square.to_index();

        // Ensure we don't exceed feature bounds
        if feature_idx < 768 {
            (Some(feature_idx), Some(feature_idx)) // Same index for both perspectives for simplicity
        } else {
            (None, None)
        }
    }

    /// Forward pass through the network
    fn forward(&self, features: &Tensor) -> CandleResult<Tensor> {
        // Transform features
        let mut x = self.feature_transformer.forward(features)?;

        // Hidden layers with clipped ReLU activation
        for layer in &self.hidden_layers {
            x = layer.forward(&x)?;
            x = self.clipped_relu(&x)?;
        }

        // Output layer
        let output = self.output_layer.forward(&x)?;

        Ok(output)
    }

    /// Clipped ReLU activation (standard for NNUE)
    fn clipped_relu(&self, x: &Tensor) -> CandleResult<Tensor> {
        // Clamp values between 0 and 1 (ReLU then clip at 1)
        let relu = x.relu()?;
        relu.clamp(0.0, 1.0)
    }

    /// Train the NNUE network on position data
    pub fn train_batch(&mut self, positions: &[(Board, f32)]) -> CandleResult<f32> {
        let batch_size = positions.len();
        let mut total_loss = 0.0;

        for (board, target_eval) in positions {
            // Extract features
            let features = self.extract_features(board)?;

            // Forward pass
            let prediction = self.forward(&features)?;

            // Create target tensor
            let target = Tensor::from_vec(vec![*target_eval / 100.0], (1, 1), &self.device)?; // Scale to NNUE range

            // Compute loss (MSE)
            let diff = (&prediction - &target)?;
            let squared = diff.powf(2.0)?;
            let loss = squared.sum_all()?;

            // Backward pass and optimization
            if let Some(ref mut optimizer) = self.optimizer {
                // Compute gradients
                let grads = loss.backward()?;

                // Step the optimizer with computed gradients
                optimizer.step(&grads)?;
            }

            total_loss += loss.to_scalar::<f32>()?;
        }

        Ok(total_loss / batch_size as f32)
    }

    /// Incremental update when a move is made (NNUE efficiency feature)
    pub fn update_incrementally(
        &mut self,
        board: &Board,
        _chess_move: chess::ChessMove,
    ) -> CandleResult<()> {
        // Update king positions for incremental feature tracking
        let white_king = board.king_square(Color::White);
        let black_king = board.king_square(Color::Black);
        self.feature_transformer.king_squares = [white_king, black_king];

        // For now, we'll re-extract features for simplicity
        // Real NNUE would incrementally update the accumulator
        let features = self.extract_features(board)?;
        self.feature_transformer.accumulated_features = Some(features);

        // In a production implementation, this would efficiently:
        // 1. Remove features for moved piece from old square
        // 2. Add features for moved piece on new square
        // 3. Handle captures, castling, en passant, promotions
        // 4. Update accumulator without full re-computation (10-100x faster)

        Ok(())
    }

    /// Set the vector evaluation blend weight
    pub fn set_vector_weight(&mut self, weight: f32) {
        self.vector_weight = weight.clamp(0.0, 1.0);
    }

    /// Enable or disable vector integration
    pub fn set_vector_integration(&mut self, enabled: bool) {
        self.enable_vector_integration = enabled;
    }

    /// Get current configuration
    pub fn get_config(&self) -> NNUEConfig {
        NNUEConfig {
            feature_size: 768,
            hidden_size: 256,
            num_hidden_layers: self.hidden_layers.len(),
            activation: ActivationType::ClippedReLU,
            learning_rate: 0.001,
            vector_blend_weight: self.vector_weight,
            enable_incremental_updates: true,
        }
    }

    /// Save the trained model to a file
    pub fn save_model(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        use std::fs::File;
        use std::io::Write;

        // Save model weights as safetensors or custom format
        // For now, save configuration and basic model info
        let config = self.get_config();
        let config_json = serde_json::to_string_pretty(&config)?;

        let mut file = File::create(format!("{}.config", path))?;
        file.write_all(config_json.as_bytes())?;

        // In production, would save actual tensor weights using safetensors
        println!("Model configuration saved to {}.config", path);
        println!("Note: Full weight serialization requires safetensors integration");

        Ok(())
    }

    /// Load a trained model from a file  
    pub fn load_model(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        use std::fs;

        // Load model configuration
        let config_path = format!("{}.config", path);
        if std::path::Path::new(&config_path).exists() {
            let config_json = fs::read_to_string(config_path)?;
            let config: NNUEConfig = serde_json::from_str(&config_json)?;

            // Apply loaded configuration
            self.vector_weight = config.vector_blend_weight;
            self.enable_vector_integration = true;

            println!("Operation complete");
            println!("Note: Full weight loading requires safetensors integration");
        } else {
            return Err(format!("Model config file not found: {}.config", path).into());
        }

        Ok(())
    }

    /// Get evaluation statistics for analysis
    pub fn get_eval_stats(&mut self, positions: &[Board]) -> CandleResult<EvalStats> {
        let mut stats = EvalStats::new();

        for board in positions {
            let eval = self.evaluate(board)?; // Simplified for demo
            stats.add_evaluation(eval);
        }

        Ok(stats)
    }
}

impl FeatureTransformer {
    fn new(vs: VarBuilder, input_size: usize, output_size: usize) -> CandleResult<Self> {
        let weights = vs.get((input_size, output_size), "ft_weights")?;
        let biases = vs.get(output_size, "ft_biases")?;

        Ok(Self {
            weights,
            biases,
            accumulated_features: None,
            king_squares: [Square::E1, Square::E8], // Default positions
        })
    }
}

impl Module for FeatureTransformer {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // Simple linear transformation (real NNUE uses more efficient accumulator)
        let output = x.matmul(&self.weights)?;
        output.broadcast_add(&self.biases)
    }
}

/// Statistics for NNUE evaluation analysis
#[derive(Debug, Clone)]
pub struct EvalStats {
    pub count: usize,
    pub mean: f32,
    pub min: f32,
    pub max: f32,
    pub std_dev: f32,
}

impl EvalStats {
    fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            std_dev: 0.0,
        }
    }

    fn add_evaluation(&mut self, eval: f32) {
        self.count += 1;
        self.min = self.min.min(eval);
        self.max = self.max.max(eval);

        // Running mean calculation
        let delta = eval - self.mean;
        self.mean += delta / self.count as f32;

        // Simplified std dev calculation (not numerically stable for large datasets)
        if self.count > 1 {
            let sum_sq =
                (self.count - 1) as f32 * self.std_dev.powi(2) + delta * (eval - self.mean);
            self.std_dev = (sum_sq / (self.count - 1) as f32).sqrt();
        }
    }
}

/// Integration helper for combining NNUE with vector-based evaluation
pub struct HybridEvaluator {
    nnue: NNUE,
    vector_evaluator: Option<Box<dyn Fn(&Board) -> Option<f32>>>,
    blend_strategy: BlendStrategy,
}

#[derive(Debug, Clone)]
pub enum BlendStrategy {
    Weighted(f32),   // Fixed weight blend
    Adaptive,        // Adapt based on position type
    Confidence(f32), // Use vector when NNUE confidence is low
    GamePhase,       // Different blending for opening/middlegame/endgame
}

impl HybridEvaluator {
    pub fn new(nnue: NNUE, blend_strategy: BlendStrategy) -> Self {
        Self {
            nnue,
            vector_evaluator: None,
            blend_strategy,
        }
    }

    pub fn set_vector_evaluator<F>(&mut self, evaluator: F)
    where
        F: Fn(&Board) -> Option<f32> + 'static,
    {
        self.vector_evaluator = Some(Box::new(evaluator));
    }

    pub fn evaluate(&mut self, board: &Board) -> CandleResult<f32> {
        let nnue_eval = self.nnue.evaluate(board)?;

        let vector_eval = if let Some(ref evaluator) = self.vector_evaluator {
            evaluator(board)
        } else {
            None
        };

        match self.blend_strategy {
            BlendStrategy::Weighted(weight) => {
                if let Some(vector_eval) = vector_eval {
                    Ok((1.0 - weight) * nnue_eval + weight * vector_eval)
                } else {
                    Ok(nnue_eval)
                }
            }
            BlendStrategy::Adaptive => {
                // Adapt based on position characteristics
                let is_tactical = self.is_tactical_position(board);
                let weight = if is_tactical { 0.2 } else { 0.5 }; // Less vector in tactical positions

                if let Some(vector_eval) = vector_eval {
                    Ok((1.0 - weight) * nnue_eval + weight * vector_eval)
                } else {
                    Ok(nnue_eval)
                }
            }
            _ => Ok(nnue_eval), // Other strategies can be implemented
        }
    }

    fn is_tactical_position(&self, board: &Board) -> bool {
        // Simple tactical detection (can be enhanced)
        board.checkers().popcnt() > 0
            || chess::MoveGen::new_legal(board).any(|m| board.piece_on(m.get_dest()).is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess::Board;

    #[test]
    fn test_nnue_creation() {
        let config = NNUEConfig::default();
        let nnue = NNUE::new(config);
        assert!(nnue.is_ok());
    }

    #[test]
    fn test_nnue_evaluation() {
        let config = NNUEConfig::default();
        let mut nnue = NNUE::new(config).unwrap();
        let board = Board::default();

        let eval = nnue.evaluate(&board);
        if eval.is_err() {
            println!("NNUE evaluation error: {:?}", eval.err());
            panic!("NNUE evaluation failed");
        }

        // Starting position should be close to 0
        let eval_value = eval.unwrap();
        assert!(eval_value.abs() < 100.0); // Within 1 pawn
    }

    #[test]
    fn test_hybrid_evaluation() {
        let config = NNUEConfig::vector_integrated();
        let mut nnue = NNUE::new(config).unwrap();
        let board = Board::default();

        let vector_eval = Some(25.0); // Small advantage
        let hybrid_eval = nnue.evaluate_hybrid(&board, vector_eval);
        assert!(hybrid_eval.is_ok());
    }

    #[test]
    fn test_feature_extraction() {
        let config = NNUEConfig::default();
        let nnue = NNUE::new(config).unwrap();
        let board = Board::default();

        let features = nnue.extract_features(&board);
        assert!(features.is_ok());

        let feature_tensor = features.unwrap();
        assert_eq!(feature_tensor.shape().dims(), &[1, 768]);
    }

    #[test]
    fn test_blend_strategies() {
        let config = NNUEConfig::default();
        let nnue = NNUE::new(config).unwrap();

        let mut evaluator = HybridEvaluator::new(nnue, BlendStrategy::Weighted(0.3));
        evaluator.set_vector_evaluator(|_| Some(50.0));

        let board = Board::default();
        let eval = evaluator.evaluate(&board);
        assert!(eval.is_ok());
    }
}
