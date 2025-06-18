use candle_core::{Device, Result as CandleResult, Tensor};
use ndarray::{Array1, Array2};
use std::sync::OnceLock;

/// GPU acceleration backend with intelligent device detection and CPU fallback
#[derive(Debug, Clone)]
pub struct GPUAccelerator {
    device: Device,
    device_type: DeviceType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DeviceType {
    CPU,
    CUDA,
    Metal,
}

static GPU_ACCELERATOR: OnceLock<GPUAccelerator> = OnceLock::new();

impl GPUAccelerator {
    /// Get the global GPU accelerator instance (singleton pattern for efficiency)
    pub fn global() -> &'static GPUAccelerator {
        GPU_ACCELERATOR.get_or_init(|| {
            Self::new().unwrap_or_else(|_| {
                println!("Warning: GPU acceleration failed to initialize, using CPU fallback");
                GPUAccelerator {
                    device: Device::Cpu,
                    device_type: DeviceType::CPU,
                }
            })
        })
    }

    /// Create a new GPU accelerator with intelligent device detection
    pub fn new() -> CandleResult<Self> {
        // Try GPU devices in order of preference: CUDA > Metal > CPU
        
        #[cfg(feature = "cuda")]
        {
            match Self::try_cuda() {
                Ok(accelerator) => {
                    println!("GPU acceleration enabled: CUDA device detected");
                    return Ok(accelerator);
                }
                Err(e) => {
                    println!("CUDA initialization failed: {}, trying Metal...", e);
                }
            }
        }

        #[cfg(feature = "metal")]
        {
            match Self::try_metal() {
                Ok(accelerator) => {
                    println!("GPU acceleration enabled: Metal device detected");
                    return Ok(accelerator);
                }
                Err(e) => {
                    println!("Metal initialization failed: {}, falling back to CPU", e);
                }
            }
        }

        println!("GPU acceleration not available, using CPU");
        Ok(GPUAccelerator {
            device: Device::Cpu,
            device_type: DeviceType::CPU,
        })
    }

    #[cfg(feature = "cuda")]
    fn try_cuda() -> CandleResult<Self> {
        let device = Device::new_cuda(0)?; // Try first CUDA device
        Ok(GPUAccelerator {
            device,
            device_type: DeviceType::CUDA,
        })
    }

    #[cfg(not(feature = "cuda"))]
    fn try_cuda() -> CandleResult<Self> {
        Err(candle_core::Error::Msg("CUDA not compiled".into()))
    }

    #[cfg(feature = "metal")]
    fn try_metal() -> CandleResult<Self> {
        let device = Device::new_metal(0)?; // Try first Metal device
        Ok(GPUAccelerator {
            device,
            device_type: DeviceType::Metal,
        })
    }

    #[cfg(not(feature = "metal"))]
    fn try_metal() -> CandleResult<Self> {
        Err(candle_core::Error::Msg("Metal not compiled".into()))
    }

    /// Get the device type being used
    pub fn device_type(&self) -> &DeviceType {
        &self.device_type
    }

    /// Get the underlying Candle device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Check if GPU acceleration is available
    pub fn is_gpu_enabled(&self) -> bool {
        matches!(self.device_type, DeviceType::CUDA | DeviceType::Metal)
    }

    /// Convert ndarray to Candle tensor on the appropriate device
    pub fn array_to_tensor(&self, array: &Array1<f32>) -> CandleResult<Tensor> {
        let data = array.as_slice().expect("Array must be contiguous");
        Tensor::from_slice(data, array.len(), &self.device)
    }

    /// Convert 2D ndarray to Candle tensor on the appropriate device
    pub fn array2_to_tensor(&self, array: &Array2<f32>) -> CandleResult<Tensor> {
        let shape = array.shape();
        let data = array.as_slice().expect("Array must be contiguous");  
        Tensor::from_slice(data, (shape[0], shape[1]), &self.device)
    }

    /// Convert Candle tensor back to ndarray
    pub fn tensor_to_array(&self, tensor: &Tensor) -> CandleResult<Array1<f32>> {
        let data = tensor.to_vec1::<f32>()?;
        Ok(Array1::from_vec(data))
    }

    /// Convert 2D Candle tensor back to ndarray
    pub fn tensor_to_array2(&self, tensor: &Tensor) -> CandleResult<Array2<f32>> {
        let dims = tensor.dims();
        if dims.len() != 2 {
            return Err(candle_core::Error::Msg("Expected 2D tensor".into()));
        }
        let data = tensor.to_vec2::<f32>()?;
        let flat_data: Vec<f32> = data.into_iter().flatten().collect();
        Ok(Array2::from_shape_vec((dims[0], dims[1]), flat_data)
            .map_err(|e| candle_core::Error::Msg(format!("Array reshape error: {}", e)))?)
    }

    /// Accelerated cosine similarity computation
    pub fn cosine_similarity_batch(&self, query: &Array1<f32>, vectors: &Array2<f32>) -> CandleResult<Array1<f32>> {
        if !self.is_gpu_enabled() || vectors.nrows() < 100 {
            // Fall back to CPU for small batches or when GPU not available
            return Ok(self.cosine_similarity_cpu(query, vectors));
        }

        // GPU-accelerated computation
        let query_tensor = self.array_to_tensor(query)?;
        let vectors_tensor = self.array2_to_tensor(vectors)?;

        // Normalize query vector
        let query_norm = query_tensor.sqr()?.sum_keepdim(0)?.sqrt()?;
        let query_normalized = query_tensor.div(&query_norm)?;

        // Normalize all vectors
        let vectors_norm = vectors_tensor.sqr()?.sum_keepdim(1)?.sqrt()?;
        let vectors_normalized = vectors_tensor.div(&vectors_norm)?;

        // Compute dot products (cosine similarity)
        let similarities = vectors_normalized.matmul(&query_normalized.unsqueeze(1)?)?.squeeze(1)?;

        self.tensor_to_array(&similarities)
    }

    /// CPU fallback for cosine similarity
    fn cosine_similarity_cpu(&self, query: &Array1<f32>, vectors: &Array2<f32>) -> Array1<f32> {
        let query_norm = query.dot(query).sqrt();
        let mut similarities = Array1::zeros(vectors.nrows());
        
        for (i, vector) in vectors.outer_iter().enumerate() {
            let dot_product = query.dot(&vector);
            let vector_norm = vector.dot(&vector).sqrt();
            similarities[i] = if vector_norm > 0.0 && query_norm > 0.0 {
                dot_product / (query_norm * vector_norm)
            } else {
                0.0
            };
        }
        
        similarities
    }

    /// Accelerated matrix multiplication
    pub fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> CandleResult<Array2<f32>> {
        if !self.is_gpu_enabled() || a.nrows() < 64 || a.ncols() < 64 {
            // CPU fallback for small matrices
            return Ok(a.dot(b));
        }

        let a_tensor = self.array2_to_tensor(a)?;
        let b_tensor = self.array2_to_tensor(b)?;
        let result_tensor = a_tensor.matmul(&b_tensor)?;
        self.tensor_to_array2(&result_tensor)
    }

    /// Accelerated vector addition
    pub fn add_vectors(&self, vectors: &[Array1<f32>]) -> CandleResult<Array1<f32>> {
        if vectors.is_empty() {
            return Err(candle_core::Error::Msg("Cannot add empty vector list".into()));
        }

        if !self.is_gpu_enabled() || vectors.len() < 10 {
            // CPU fallback
            let mut result = vectors[0].clone();
            for vector in &vectors[1..] {
                result = &result + vector;
            }
            return Ok(result);
        }

        // GPU acceleration
        let mut result_tensor = self.array_to_tensor(&vectors[0])?;
        for vector in &vectors[1..] {
            let vector_tensor = self.array_to_tensor(vector)?;
            result_tensor = result_tensor.add(&vector_tensor)?;
        }
        
        self.tensor_to_array(&result_tensor)
    }

    /// Get memory usage information
    pub fn memory_info(&self) -> String {
        match self.device_type {
            DeviceType::CPU => "CPU memory (system RAM)".to_string(),
            DeviceType::CUDA => {
                #[cfg(feature = "cuda")]
                {
                    // Would need CUDA runtime API calls to get actual memory info
                    "CUDA GPU memory (use nvidia-smi for details)".to_string()
                }
                #[cfg(not(feature = "cuda"))]
                "CUDA not available".to_string()
            }
            DeviceType::Metal => {
                #[cfg(feature = "metal")]
                {
                    "Metal GPU memory (system shared)".to_string()
                }
                #[cfg(not(feature = "metal"))]
                "Metal not available".to_string()
            }
        }
    }

    /// Benchmark the device performance
    pub fn benchmark(&self) -> CandleResult<f64> {
        let size = 1000;
        let a = Array2::<f32>::ones((size, size));
        let b = Array2::<f32>::ones((size, size));
        
        let start = std::time::Instant::now();
        let _result = self.matmul(&a, &b)?;
        let duration = start.elapsed();
        
        let ops = (size * size * size) as f64; // Matrix multiplication operations
        let gflops = ops / duration.as_secs_f64() / 1e9;
        
        Ok(gflops)
    }
}

impl Default for GPUAccelerator {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| GPUAccelerator {
            device: Device::Cpu,
            device_type: DeviceType::CPU,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_gpu_accelerator_creation() {
        let accelerator = GPUAccelerator::new().unwrap();
        println!("Device type: {:?}", accelerator.device_type());
    }

    #[test]
    fn test_cosine_similarity() {
        let accelerator = GPUAccelerator::global();
        let query = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let vectors = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        
        let similarities = accelerator.cosine_similarity_batch(&query, &vectors).unwrap();
        assert_eq!(similarities.len(), 2);
        assert!(similarities[0] > 0.9); // Should be close to 1.0 for identical vectors
    }

    #[test]
    fn test_matrix_multiplication() {
        let accelerator = GPUAccelerator::global();
        let a = Array2::<f32>::ones((2, 3));
        let b = Array2::<f32>::ones((3, 2));
        
        let result = accelerator.matmul(&a, &b).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result[(0, 0)], 3.0); // Sum of ones
    }

    #[test]
    fn test_benchmark() {
        let accelerator = GPUAccelerator::global();
        let gflops = accelerator.benchmark().unwrap();
        println!("Benchmark: {:.2} GFLOPS on {:?}", gflops, accelerator.device_type());
        assert!(gflops > 0.0);
    }
}