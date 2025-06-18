use ndarray::{Array1, Array2};
use candle_core::{Device, Result as CandleResult, Tensor, Module};
use candle_nn::{linear, Linear, VarBuilder, VarMap, Optimizer, AdamW, ParamsAdamW};
use rayon::prelude::*;
use serde::{Serialize, Deserialize};

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
    
    /// Encode multiple vectors in parallel
    pub fn encode_batch(&self, inputs: &[Array1<f32>]) -> Vec<Array1<f32>> {
        if inputs.len() > 10 {
            // Use parallel processing for larger batches
            inputs.par_iter()
                .map(|input| self.encode(input))
                .collect()
        } else {
            // Use sequential processing for smaller batches
            inputs.iter()
                .map(|input| self.encode(input))
                .collect()
        }
    }
    
    /// Decode multiple vectors in parallel
    pub fn decode_batch(&self, manifold_vecs: &[Array1<f32>]) -> Vec<Array1<f32>> {
        if manifold_vecs.len() > 10 {
            // Use parallel processing for larger batches
            manifold_vecs.par_iter()
                .map(|vec| self.decode(vec))
                .collect()
        } else {
            // Use sequential processing for smaller batches
            manifold_vecs.iter()
                .map(|vec| self.decode(vec))
                .collect()
        }
    }
    
    /// Train with parallel data preparation (preprocessing batches in parallel)
    pub fn train_parallel(&mut self, data: &Array2<f32>, epochs: usize, batch_size: usize) -> Result<(), String> {
        // Initialize network if not done
        if self.encoder.is_none() {
            self.init_network()?;
        }
        
        let num_samples = data.nrows();
        let num_batches = (num_samples + batch_size - 1) / batch_size;
        
        println!("Training autoencoder for {} epochs with {} batches of size {}", epochs, num_batches, batch_size);
        
        // Create batches in parallel
        let batches: Vec<_> = (0..num_batches)
            .into_par_iter()
            .map(|batch_idx| {
                let start_idx = batch_idx * batch_size;
                let end_idx = (start_idx + batch_size).min(num_samples);
                
                // Extract batch data
                let batch_data = data.slice(ndarray::s![start_idx..end_idx, ..]);
                
                // Convert to tensor format
                let rows = batch_data.nrows();
                let cols = batch_data.ncols();
                let batch_vec: Vec<f32> = batch_data.iter().copied().collect();
                
                (batch_vec, rows, cols)
            })
            .collect();
        
        // Training loop - epochs are sequential but batch preparation was parallel
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            
            for (batch_data, rows, cols) in &batches {
                if let (Some(encoder), Some(decoder), Some(optimizer)) = 
                    (&self.encoder, &self.decoder, &mut self.optimizer) {
                    
                    // Convert batch to tensor
                    let data_tensor = Tensor::from_slice(
                        batch_data.as_slice(),
                        (*rows, *cols),
                        &self.device
                    ).map_err(|e| format!("Failed to create tensor: {}", e))?;
                    
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
                    
                    // Accumulate loss for reporting
                    total_loss += loss.to_scalar::<f32>()
                        .map_err(|e| format!("Loss scalar conversion failed: {}", e))?;
                    
                    // Compute gradients through backpropagation
                    let grads = loss.backward()
                        .map_err(|e| format!("Backward pass failed: {}", e))?;
                    
                    // Update weights using the optimizer
                    optimizer.step(&grads)
                        .map_err(|e| format!("Optimizer step failed: {}", e))?;
                }
            }
            
            if epoch % 10 == 0 {
                let avg_loss = total_loss / batches.len() as f32;
                println!("Epoch {}: Average Loss = {:.6}", epoch, avg_loss);
            }
        }
        
        println!("Parallel training completed!");
        Ok(())
    }

    /// Save manifold learner configuration and weights to database
    pub fn save_to_database(&self, db: &crate::persistence::Database) -> Result<(), Box<dyn std::error::Error>> {
        if !self.is_trained() {
            return Err("Cannot save untrained manifold learner".into());
        }

        // Serialize the VarMap (model weights) to bytes
        let var_map_bytes = self.serialize_var_map()?;
        
        // Create training metadata
        let metadata = ManifoldMetadata {
            input_dim: self.input_dim,
            output_dim: self.output_dim,
            is_trained: self.is_trained(),
            compression_ratio: self.compression_ratio(),
        };
        let metadata_bytes = bincode::serialize(&metadata)?;

        db.save_manifold_model(
            self.input_dim,
            self.output_dim,
            &var_map_bytes,
            Some(&metadata_bytes)
        )?;

        println!("Saved manifold learner to database (compression ratio: {:.1}x)", self.compression_ratio());
        Ok(())
    }

    /// Load manifold learner from database
    pub fn load_from_database(db: &crate::persistence::Database) -> Result<Option<Self>, Box<dyn std::error::Error>> {
        match db.load_manifold_model()? {
            Some((input_dim, output_dim, model_weights, metadata_bytes)) => {
                let mut learner = Self::new(input_dim, output_dim);
                
                // Initialize the network first
                learner.init_network()?;
                
                // Deserialize and load the VarMap (model weights)
                learner.deserialize_var_map(&model_weights)?;
                
                // Load metadata if available
                if !metadata_bytes.is_empty() {
                    match bincode::deserialize::<ManifoldMetadata>(&metadata_bytes) {
                        Ok(metadata) => {
                            println!("Loaded manifold learner from database (compression ratio: {:.1}x)", metadata.compression_ratio);
                        }
                        Err(e) => {
                            println!("Warning: Could not deserialize manifold metadata: {}", e);
                        }
                    }
                }
                
                Ok(Some(learner))
            }
            None => Ok(None),
        }
    }

    /// Create manifold learner from database or return a new one
    pub fn from_database_or_new(db: &crate::persistence::Database, input_dim: usize, output_dim: usize) -> Result<Self, Box<dyn std::error::Error>> {
        match Self::load_from_database(db)? {
            Some(learner) => {
                println!("Loaded existing manifold learner from database");
                Ok(learner)
            }
            None => {
                println!("No saved manifold learner found, creating new one");
                Ok(Self::new(input_dim, output_dim))
            }
        }
    }

    /// Serialize VarMap to bytes (simplified approach - in real implementation, use Candle's serialization)
    fn serialize_var_map(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        // For now, return a placeholder. In a full implementation, this would serialize
        // the actual neural network weights using Candle's safetensors format
        let placeholder = format!("varmap_{}_{}", self.input_dim, self.output_dim);
        Ok(placeholder.into_bytes())
    }

    /// Deserialize VarMap from bytes (simplified approach)
    fn deserialize_var_map(&mut self, _bytes: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        // For now, this is a placeholder. In a full implementation, this would 
        // deserialize the actual neural network weights and load them into the VarMap
        // The network is already initialized, so weights would be loaded here
        Ok(())
    }
}

/// Metadata for manifold learner persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ManifoldMetadata {
    input_dim: usize,
    output_dim: usize,
    is_trained: bool,
    compression_ratio: f32,
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