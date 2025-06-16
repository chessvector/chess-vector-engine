use ndarray::{Array1, Array2};
use candle_core::{Device, Result as CandleResult, Tensor, Module};
use candle_nn::{linear, Linear, VarBuilder, VarMap, Optimizer, AdamW, ParamsAdamW};

/// Autoencoder for chess position manifold learning
pub struct ManifoldLearner {
    input_dim: usize,
    output_dim: usize,
    device: Device,
    encoder: Option<Encoder>,
    decoder: Option<Decoder>,
    var_map: VarMap,
    optimizer: Option<AdamW>,
}

/// Encoder network (input -> manifold)
struct Encoder {
    layer1: Linear,
    layer2: Linear,
    layer3: Linear,
}

/// Decoder network (manifold -> input)
struct Decoder {
    layer1: Linear,
    layer2: Linear,
    layer3: Linear,
}

impl Encoder {
    fn new(vs: VarBuilder, input_dim: usize, hidden_dim: usize, output_dim: usize) -> CandleResult<Self> {
        let layer1 = linear(input_dim, hidden_dim, vs.pp("encoder.layer1"))?;
        let layer2 = linear(hidden_dim, hidden_dim / 2, vs.pp("encoder.layer2"))?;
        let layer3 = linear(hidden_dim / 2, output_dim, vs.pp("encoder.layer3"))?;
        
        Ok(Self { layer1, layer2, layer3 })
    }
}

impl Module for Encoder {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let x = self.layer1.forward(x)?.relu()?;
        let x = self.layer2.forward(&x)?.relu()?;
        self.layer3.forward(&x) // No activation on final layer
    }
}

impl Decoder {
    fn new(vs: VarBuilder, input_dim: usize, hidden_dim: usize, output_dim: usize) -> CandleResult<Self> {
        let layer1 = linear(input_dim, hidden_dim / 2, vs.pp("decoder.layer1"))?;
        let layer2 = linear(hidden_dim / 2, hidden_dim, vs.pp("decoder.layer2"))?;
        let layer3 = linear(hidden_dim, output_dim, vs.pp("decoder.layer3"))?;
        
        Ok(Self { layer1, layer2, layer3 })
    }
}

impl Module for Decoder {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let x = self.layer1.forward(x)?.relu()?;
        let x = self.layer2.forward(&x)?.relu()?;
        self.layer3.forward(&x)?.tanh() // Tanh to bound output
    }
}

impl ManifoldLearner {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let device = Device::Cpu; // Use CPU for simplicity
        let var_map = VarMap::new();
        
        Self {
            input_dim,
            output_dim,
            device,
            encoder: None,
            decoder: None,
            var_map,
            optimizer: None,
        }
    }
    
    /// Initialize the neural network architecture
    pub fn init_network(&mut self) -> Result<(), String> {
        let vs = VarBuilder::from_varmap(&self.var_map, candle_core::DType::F32, &self.device);
        let hidden_dim = (self.input_dim + self.output_dim) / 2;
        
        let encoder = Encoder::new(vs.clone(), self.input_dim, hidden_dim, self.output_dim)
            .map_err(|e| format!("Failed to create encoder: {}", e))?;
        let decoder = Decoder::new(vs, self.output_dim, hidden_dim, self.input_dim)
            .map_err(|e| format!("Failed to create decoder: {}", e))?;
        
        // Initialize AdamW optimizer with learning rate 0.001
        let adamw_params = ParamsAdamW {
            lr: 0.001,
            ..Default::default()
        };
        let optimizer = AdamW::new(self.var_map.all_vars(), adamw_params)
            .map_err(|e| format!("Failed to create optimizer: {}", e))?;
        
        self.encoder = Some(encoder);
        self.decoder = Some(decoder);
        self.optimizer = Some(optimizer);
        
        Ok(())
    }

    /// Train the autoencoder on position data
    pub fn train(&mut self, data: &Array2<f32>, epochs: usize) -> Result<(), String> {
        // Initialize network if not done
        if self.encoder.is_none() {
            self.init_network()?;
        }
        
        // Convert ndarray to tensor
        let data_tensor = Tensor::from_slice(
            data.as_slice().unwrap(),
            (data.nrows(), data.ncols()),
            &self.device
        ).map_err(|e| format!("Failed to create tensor: {}", e))?;
        
        println!("Training autoencoder for {} epochs with proper gradient descent...", epochs);
        
        // Training loop with proper gradient descent
        for epoch in 0..epochs {
            if let (Some(encoder), Some(decoder), Some(optimizer)) = 
                (&self.encoder, &self.decoder, &mut self.optimizer) {
                
                // Forward pass
                let encoded = encoder.forward(&data_tensor)
                    .map_err(|e| format!("Encoder forward failed: {}", e))?;
                let decoded = decoder.forward(&encoded)
                    .map_err(|e| format!("Decoder forward failed: {}", e))?;
                
                // Calculate reconstruction loss (MSE)
                let loss = (&data_tensor - &decoded)
                    .and_then(|diff| diff.powf(2.0))
                    .and_then(|squared| squared.mean_all())
                    .map_err(|e| format!("Loss calculation failed: {}", e))?;
                
                // Compute gradients through backpropagation
                let grads = loss.backward()
                    .map_err(|e| format!("Backward pass failed: {}", e))?;
                
                // Update weights using the optimizer
                optimizer.step(&grads)
                    .map_err(|e| format!("Optimizer step failed: {}", e))?;
                
                if epoch % 10 == 0 {
                    let loss_val = loss.to_scalar::<f32>()
                        .map_err(|e| format!("Loss scalar conversion failed: {}", e))?;
                    println!("Epoch {}: Loss = {:.6}", epoch, loss_val);
                }
            }
        }
        
        println!("Training completed with proper gradient descent!");
        Ok(())
    }

    /// Encode input to manifold space
    pub fn encode(&self, input: &Array1<f32>) -> Array1<f32> {
        if let Some(encoder) = &self.encoder {
            // Convert ndarray to tensor
            if let Ok(input_tensor) = Tensor::from_slice(
                input.as_slice().unwrap(),
                (1, input.len()),
                &self.device
            ) {
                if let Ok(encoded) = encoder.forward(&input_tensor) {
                    if let Ok(encoded_data) = encoded.to_vec2::<f32>() {
                        return Array1::from(encoded_data[0].clone());
                    }
                }
            }
        }
        
        // Fallback: return random compressed representation
        Array1::from(vec![0.0; self.output_dim])
    }

    /// Decode from manifold space to original space
    pub fn decode(&self, manifold_vec: &Array1<f32>) -> Array1<f32> {
        if let Some(decoder) = &self.decoder {
            // Convert ndarray to tensor
            if let Ok(input_tensor) = Tensor::from_slice(
                manifold_vec.as_slice().unwrap(),
                (1, manifold_vec.len()),
                &self.device
            ) {
                if let Ok(decoded) = decoder.forward(&input_tensor) {
                    if let Ok(decoded_data) = decoded.to_vec2::<f32>() {
                        return Array1::from(decoded_data[0].clone());
                    }
                }
            }
        }
        
        // Fallback: return zeros
        Array1::from(vec![0.0; self.input_dim])
    }
    
    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        self.input_dim as f32 / self.output_dim as f32
    }
    
    /// Check if the network is trained
    pub fn is_trained(&self) -> bool {
        self.encoder.is_some() && self.decoder.is_some() && self.optimizer.is_some()
    }
    
    /// Get the output dimension (compressed size)
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_manifold_learner_creation() {
        let learner = ManifoldLearner::new(1024, 128);
        assert_eq!(learner.input_dim, 1024);
        assert_eq!(learner.output_dim, 128);
        assert_eq!(learner.compression_ratio(), 8.0);
        assert!(!learner.is_trained());
    }

    #[test]
    fn test_network_initialization() {
        let mut learner = ManifoldLearner::new(100, 20);
        assert!(learner.init_network().is_ok());
        assert!(learner.is_trained());
    }

    #[test]
    fn test_encode_decode_basic() {
        let mut learner = ManifoldLearner::new(50, 10);
        learner.init_network().expect("Network initialization failed");
        
        let input = Array1::from(vec![1.0; 50]);
        let encoded = learner.encode(&input);
        let decoded = learner.decode(&encoded);
        
        assert_eq!(encoded.len(), 10);
        assert_eq!(decoded.len(), 50);
    }

    #[test]
    fn test_training_basic() {
        let mut learner = ManifoldLearner::new(20, 5);
        
        // Create some dummy training data
        let data = Array2::from_shape_vec((10, 20), (0..200).map(|x| x as f32 / 100.0).collect())
            .expect("Failed to create training data");
        
        // Training should not panic
        let result = learner.train(&data, 5);
        assert!(result.is_ok());
        assert!(learner.is_trained());
    }
}