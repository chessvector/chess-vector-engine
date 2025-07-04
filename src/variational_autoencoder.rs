use candle_core::{Device, Module, Result as CandleResult, Tensor};
use candle_nn::{linear, AdamW, Linear, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use ndarray::Array2;
use std::collections::HashMap;

/// Variational Autoencoder for chess position manifold learning with uncertainty quantification
pub struct VariationalAutoencoder {
    input_dim: usize,
    latent_dim: usize,
    device: Device,
    encoder: Option<VariationalEncoder>,
    decoder: Option<VariationalDecoder>,
    var_map: VarMap,
    optimizer: Option<AdamW>,
    beta: f32, // KL divergence weight for β-VAE
}

/// Variational encoder with mean and log-variance outputs
struct VariationalEncoder {
    shared_layers: Vec<Linear>,
    mean_layer: Linear,
    logvar_layer: Linear,
}

/// Variational decoder
struct VariationalDecoder {
    layers: Vec<Linear>,
}

impl VariationalEncoder {
    fn new(
        vs: VarBuilder,
        input_dim: usize,
        hidden_dims: &[usize],
        latent_dim: usize,
    ) -> CandleResult<Self> {
        let mut shared_layers = Vec::new();
        let mut prev_dim = input_dim;

        // Create shared hidden layers
        for (i, &hidden_dim) in hidden_dims.iter().enumerate() {
            let layer = linear(prev_dim, hidden_dim, vs.pp(format!("encoder.layer{i}")))?;
            shared_layers.push(layer);
            prev_dim = hidden_dim;
        }

        // Mean and log-variance branches
        let mean_layer = linear(prev_dim, latent_dim, vs.pp("encoder.mean"))?;
        let logvar_layer = linear(prev_dim, latent_dim, vs.pp("encoder.logvar"))?;

        Ok(Self {
            shared_layers,
            mean_layer,
            logvar_layer,
        })
    }

    /// Forward pass returning mean and log-variance
    fn encode(&self, x: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        let mut h = x.clone();

        // Pass through shared layers with ReLU activation
        for layer in &self.shared_layers {
            h = layer.forward(&h)?.relu()?;
        }

        // Compute mean and log-variance
        let mean = self.mean_layer.forward(&h)?;
        let logvar = self.logvar_layer.forward(&h)?;

        Ok((mean, logvar))
    }

    /// Reparameterization trick: z = μ + σ * ε
    fn reparameterize(&self, mean: &Tensor, logvar: &Tensor) -> CandleResult<Tensor> {
        let std = (logvar * 0.5)?.exp()?;
        let eps = Tensor::randn_like(&std, 0.0, 1.0)?;
        let scaled_eps = (&std * &eps)?;
        mean + &scaled_eps
    }
}

impl VariationalDecoder {
    fn new(
        vs: VarBuilder,
        latent_dim: usize,
        hidden_dims: &[usize],
        output_dim: usize,
    ) -> CandleResult<Self> {
        let mut layers = Vec::new();
        let mut prev_dim = latent_dim;

        // Create hidden layers (reverse of encoder)
        for (i, &hidden_dim) in hidden_dims.iter().rev().enumerate() {
            let layer = linear(prev_dim, hidden_dim, vs.pp(format!("decoder.layer{i}")))?;
            layers.push(layer);
            prev_dim = hidden_dim;
        }

        // Output layer
        let output_layer = linear(prev_dim, output_dim, vs.pp("decoder.output"))?;
        layers.push(output_layer);

        Ok(Self { layers })
    }
}

impl Module for VariationalDecoder {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let mut h = x.clone();

        // All layers except last use ReLU
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h)?;
            if i < self.layers.len() - 1 {
                h = h.relu()?;
            } else {
                // Output layer uses tanh to bound values
                h = h.tanh()?;
            }
        }

        Ok(h)
    }
}

impl VariationalAutoencoder {
    pub fn new(input_dim: usize, latent_dim: usize, beta: f32) -> Self {
        let device = Device::Cpu; // Use CPU by default, can be upgraded to GPU later
        let var_map = VarMap::new();

        Self {
            input_dim,
            latent_dim,
            device,
            encoder: None,
            decoder: None,
            var_map,
            optimizer: None,
            beta,
        }
    }

    /// Initialize the VAE network with configurable architecture
    pub fn init_network(&mut self, hidden_dims: &[usize]) -> Result<(), String> {
        let vs = VarBuilder::from_varmap(&self.var_map, candle_core::DType::F32, &self.device);

        let encoder =
            VariationalEncoder::new(vs.clone(), self.input_dim, hidden_dims, self.latent_dim)
                .map_err(|_e| "Processing...".to_string())?;
        let decoder = VariationalDecoder::new(vs, self.latent_dim, hidden_dims, self.input_dim)
            .map_err(|_e| "Processing...".to_string())?;

        // Initialize AdamW optimizer with lower learning rate for VAE stability
        let adamw_params = ParamsAdamW {
            lr: 0.0005,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 1e-4,
        };
        let optimizer = AdamW::new(self.var_map.all_vars(), adamw_params)
            .map_err(|_e| "Processing...".to_string())?;

        self.encoder = Some(encoder);
        self.decoder = Some(decoder);
        self.optimizer = Some(optimizer);

        Ok(())
    }

    /// Forward pass through the VAE
    pub fn forward(&self, x: &Tensor) -> CandleResult<(Tensor, Tensor, Tensor)> {
        let encoder = self
            .encoder
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("Encoder not initialized".into()))?;
        let decoder = self
            .decoder
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("Decoder not initialized".into()))?;

        // Encode to get mean and log-variance
        let (mean, logvar) = encoder.encode(x)?;

        // Sample from latent distribution
        let z = encoder.reparameterize(&mean, &logvar)?;

        // Decode back to original space
        let reconstruction = decoder.forward(&z)?;

        Ok((reconstruction, mean, logvar))
    }

    /// Compute VAE loss (reconstruction + KL divergence)
    pub fn compute_loss(
        &self,
        x: &Tensor,
        reconstruction: &Tensor,
        mean: &Tensor,
        logvar: &Tensor,
    ) -> CandleResult<Tensor> {
        let batch_size = x.dims()[0] as f32;

        // Reconstruction loss (MSE)
        let diff = (x - reconstruction)?;
        let squared = diff.powf(2.0)?;
        let sum_tensor = squared.sum_all()?;
        let batch_tensor = Tensor::new(batch_size, &self.device)?;
        let recon_loss = (&sum_tensor / &batch_tensor)?;

        // KL divergence: -0.5 * sum(1 + log(σ²) - μ² - σ²)
        let kl_div = {
            let var = logvar.exp()?;
            let mean_sq = mean.powf(2.0)?;
            let one_tensor = Tensor::ones_like(logvar)?;
            let logvar_plus_one = (logvar + &one_tensor)?;
            let minus_mean_sq = (&logvar_plus_one - &mean_sq)?;
            let kl_per_dim = (&minus_mean_sq - &var)?;
            let kl_sum = kl_per_dim.sum_all()?;
            let neg_half = Tensor::new(-0.5f32, &self.device)?;
            let kl_scaled = (&kl_sum * &neg_half)?;
            let batch_tensor = Tensor::new(batch_size, &self.device)?;
            (&kl_scaled / &batch_tensor)?
        };

        // Total loss with β weighting
        let beta_tensor = Tensor::new(self.beta, &self.device)?;
        let weighted_kl = (&kl_div * &beta_tensor)?;
        let total_loss = (&recon_loss + &weighted_kl)?;

        Ok(total_loss)
    }

    /// Train the VAE on a batch of position vectors
    pub fn train_step(&mut self, vectors: &Array2<f32>) -> Result<f32, String> {
        // Convert to tensor
        let batch_tensor = self.array_to_tensor(vectors)?;

        // Forward pass
        let (reconstruction, mean, logvar) = self
            .forward(&batch_tensor)
            .map_err(|_e| "Processing...".to_string())?;

        // Compute loss
        let loss = self
            .compute_loss(&batch_tensor, &reconstruction, &mean, &logvar)
            .map_err(|_e| "Processing...".to_string())?;

        // Get loss value for return
        let loss_value = loss
            .to_scalar::<f32>()
            .map_err(|_e| "Processing...".to_string())?;

        // Backward pass
        let grads = loss.backward().map_err(|_e| "Processing...".to_string())?;

        // Now get optimizer and step
        let optimizer = self.optimizer.as_mut().ok_or("Optimizer not initialized")?;
        optimizer
            .step(&grads)
            .map_err(|_e| "Processing...".to_string())?;

        // Return loss value
        Ok(loss_value)
    }

    /// Encode positions to latent space with uncertainty
    pub fn encode(&self, vectors: &Array2<f32>) -> Result<(Array2<f32>, Array2<f32>), String> {
        let encoder = self.encoder.as_ref().ok_or("Encoder not initialized")?;

        let input_tensor = self.array_to_tensor(vectors)?;
        let (mean, logvar) = encoder
            .encode(&input_tensor)
            .map_err(|_e| "Processing...".to_string())?;

        let mean_array = self.tensor_to_array(&mean)?;
        let logvar_array = self.tensor_to_array(&logvar)?;

        Ok((mean_array, logvar_array))
    }

    /// Sample from the latent space
    pub fn sample_latent(
        &self,
        mean: &Array2<f32>,
        logvar: &Array2<f32>,
    ) -> Result<Array2<f32>, String> {
        let encoder = self.encoder.as_ref().ok_or("Encoder not initialized")?;

        let mean_tensor = self.array_to_tensor(mean)?;
        let logvar_tensor = self.array_to_tensor(logvar)?;

        let z = encoder
            .reparameterize(&mean_tensor, &logvar_tensor)
            .map_err(|_e| "Processing...".to_string())?;

        self.tensor_to_array(&z)
    }

    /// Decode from latent space
    pub fn decode(&self, latent_vectors: &Array2<f32>) -> Result<Array2<f32>, String> {
        let decoder = self.decoder.as_ref().ok_or("Decoder not initialized")?;

        let latent_tensor = self.array_to_tensor(latent_vectors)?;
        let output = decoder
            .forward(&latent_tensor)
            .map_err(|_e| "Processing...".to_string())?;

        self.tensor_to_array(&output)
    }

    /// Full encoding pipeline (encode then sample)
    pub fn encode_with_sampling(&self, vectors: &Array2<f32>) -> Result<Array2<f32>, String> {
        let (mean, logvar) = self.encode(vectors)?;
        self.sample_latent(&mean, &logvar)
    }

    /// Generate new samples from the learned manifold
    pub fn generate(&self, num_samples: usize) -> Result<Array2<f32>, String> {
        let _decoder = self.decoder.as_ref().ok_or("Decoder not initialized")?;

        // Sample from standard normal distribution
        let latent_samples = Array2::from_shape_fn((num_samples, self.latent_dim), |_| {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            rng.gen::<f32>() * 2.0 - 1.0 // Sample from [-1, 1]
        });

        self.decode(&latent_samples)
    }

    /// Get reconstruction quality metrics
    pub fn evaluate_reconstruction(
        &self,
        vectors: &Array2<f32>,
    ) -> Result<HashMap<String, f32>, String> {
        let input_tensor = self.array_to_tensor(vectors)?;
        let (reconstruction, _mean, _logvar) = self
            .forward(&input_tensor)
            .map_err(|_e| "Processing...".to_string())?;

        let reconstruction_array = self.tensor_to_array(&reconstruction)?;

        // Compute metrics
        let mut metrics = HashMap::new();

        // MSE
        let mse = vectors
            .iter()
            .zip(reconstruction_array.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / (vectors.len() as f32);
        metrics.insert("mse".to_string(), mse);

        // RMSE
        metrics.insert("rmse".to_string(), mse.sqrt());

        // Mean absolute error
        let mae = vectors
            .iter()
            .zip(reconstruction_array.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / (vectors.len() as f32);
        metrics.insert("mae".to_string(), mae);

        // Compression ratio
        let compression_ratio = self.input_dim as f32 / self.latent_dim as f32;
        metrics.insert("compression_ratio".to_string(), compression_ratio);

        Ok(metrics)
    }

    /// Get the latent dimensionality
    pub fn latent_dim(&self) -> usize {
        self.latent_dim
    }

    /// Check if the VAE is initialized
    pub fn is_initialized(&self) -> bool {
        self.encoder.is_some() && self.decoder.is_some() && self.optimizer.is_some()
    }

    // Helper methods for tensor conversions
    fn array_to_tensor(&self, array: &Array2<f32>) -> Result<Tensor, String> {
        let shape = array.shape();
        let data: Vec<f32> = array.iter().cloned().collect();
        Tensor::from_vec(data, (shape[0], shape[1]), &self.device)
            .map_err(|_e| "Processing...".to_string())
    }

    fn tensor_to_array(&self, tensor: &Tensor) -> Result<Array2<f32>, String> {
        let shape = tensor.shape();
        if shape.dims().len() != 2 {
            return Err("Expected 2D tensor".to_string());
        }

        let data = tensor
            .to_vec2::<f32>()
            .map_err(|_e| "Processing...".to_string())?;

        Array2::from_shape_vec((shape.dims()[0], shape.dims()[1]), data.concat())
            .map_err(|_e| "Processing...".to_string())
    }
}

/// Configuration for VAE training
#[derive(Debug, Clone)]
pub struct VAEConfig {
    pub hidden_dims: Vec<usize>,
    pub beta: f32,
    pub learning_rate: f32,
    pub batch_size: usize,
    pub epochs: usize,
}

impl Default for VAEConfig {
    fn default() -> Self {
        Self {
            hidden_dims: vec![512, 256, 128], // Deeper architecture
            beta: 1.0,                        // Standard VAE
            learning_rate: 0.0005,
            batch_size: 32,
            epochs: 100,
        }
    }
}

impl VAEConfig {
    /// Configuration for β-VAE with higher disentanglement
    pub fn beta_vae(beta: f32) -> Self {
        Self {
            beta,
            ..Default::default()
        }
    }

    /// Configuration for high compression ratio
    pub fn high_compression() -> Self {
        Self {
            hidden_dims: vec![512, 256, 128, 64], // More layers for better compression
            beta: 0.5,                            // Lower KL weight for better reconstruction
            ..Default::default()
        }
    }

    /// Configuration optimized for chess positions
    pub fn chess_optimized() -> Self {
        Self {
            hidden_dims: vec![512, 256, 128], // Balanced architecture
            beta: 0.8,                        // Slightly favor reconstruction
            learning_rate: 0.001,
            batch_size: 64,
            epochs: 150,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_vae_initialization() {
        let mut vae = VariationalAutoencoder::new(1024, 128, 1.0);
        let config = VAEConfig::default();

        assert!(vae.init_network(&config.hidden_dims).is_ok());
        assert!(vae.is_initialized());
        assert_eq!(vae.latent_dim(), 128);
    }

    #[test]
    fn test_vae_forward_pass() {
        let mut vae = VariationalAutoencoder::new(64, 16, 1.0);
        let config = VAEConfig::default();
        vae.init_network(&config.hidden_dims).unwrap();

        let test_data = Array2::from_shape_fn((4, 64), |_| 0.5);
        let result = vae.encode_with_sampling(&test_data);

        assert!(result.is_ok());
        let encoded = result.unwrap();
        assert_eq!(encoded.shape(), &[4, 16]);
    }

    #[test]
    fn test_vae_reconstruction() {
        let mut vae = VariationalAutoencoder::new(32, 8, 1.0);
        let config = VAEConfig::default();
        vae.init_network(&config.hidden_dims).unwrap();

        let test_data = Array2::from_shape_fn((2, 32), |_| 0.3);
        let encoded = vae.encode_with_sampling(&test_data).unwrap();
        let decoded = vae.decode(&encoded).unwrap();

        assert_eq!(decoded.shape(), test_data.shape());
    }

    #[test]
    fn test_vae_generation() {
        let mut vae = VariationalAutoencoder::new(16, 4, 1.0);
        let config = VAEConfig::default();
        vae.init_network(&config.hidden_dims).unwrap();

        let generated = vae.generate(3);
        assert!(generated.is_ok());

        let samples = generated.unwrap();
        assert_eq!(samples.shape(), &[3, 16]);
    }
}
