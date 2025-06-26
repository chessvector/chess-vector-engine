use candle_core::{Device, Result as CandleResult, Tensor};
use ndarray::{Array1, Array2};
use std::sync::OnceLock;

/// GPU acceleration backend with intelligent device detection and CPU fallback
#[derive(Debug, Clone)]
pub struct GPUAccelerator {
    device: Device,
    device_type: DeviceType,
    /// Available GPU devices for multi-GPU operations
    available_devices: Vec<Device>,
    /// Current device index for multi-GPU operations
    current_device_index: usize,
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
                    available_devices: vec![Device::Cpu],
                    current_device_index: 0,
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
                    println!("CUDA initialization failed: {e}, trying Metal...");
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
                    println!("Metal initialization failed: {e}, falling back to CPU");
                }
            }
        }

        println!("GPU acceleration not available, using CPU");
        Ok(GPUAccelerator {
            device: Device::Cpu,
            device_type: DeviceType::CPU,
            available_devices: vec![Device::Cpu],
            current_device_index: 0,
        })
    }

    #[cfg(feature = "cuda")]
    fn try_cuda() -> CandleResult<Self> {
        // Try to detect multiple CUDA devices
        let mut available_devices = Vec::new();
        let mut device_count = 0;

        // Try to detect up to 8 CUDA devices
        for i in 0..8 {
            if let Ok(device) = Device::new_cuda(i) {
                available_devices.push(device);
                device_count += 1;
            } else {
                break;
            }
        }

        if available_devices.is_empty() {
            return Err(candle_core::Error::Msg("No CUDA devices available".into()));
        }

        println!("🚀 Detected {device_count} CUDA device(s)");

        Ok(GPUAccelerator {
            device: available_devices[0].clone(),
            device_type: DeviceType::CUDA,
            available_devices,
            current_device_index: 0,
        })
    }

    #[cfg(not(feature = "cuda"))]
    #[allow(dead_code)]
    fn try_cuda() -> CandleResult<Self> {
        Err(candle_core::Error::Msg("CUDA not compiled".into()))
    }

    #[cfg(feature = "metal")]
    fn try_metal() -> CandleResult<Self> {
        // Try to detect multiple Metal devices
        let mut available_devices = Vec::new();
        let mut device_count = 0;

        // Try to detect up to 4 Metal devices (typically fewer than CUDA)
        for i in 0..4 {
            if let Ok(device) = Device::new_metal(i) {
                available_devices.push(device);
                device_count += 1;
            } else {
                break;
            }
        }

        if available_devices.is_empty() {
            return Err(candle_core::Error::Msg("No Metal devices available".into()));
        }

        println!("🍎 Detected {device_count} Metal device(s)");

        Ok(GPUAccelerator {
            device: available_devices[0].clone(),
            device_type: DeviceType::Metal,
            available_devices,
            current_device_index: 0,
        })
    }

    #[cfg(not(feature = "metal"))]
    #[allow(dead_code)]
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

    /// Get number of available GPU devices
    pub fn device_count(&self) -> usize {
        self.available_devices.len()
    }

    /// Check if multiple GPU devices are available
    pub fn is_multi_gpu_available(&self) -> bool {
        self.is_gpu_enabled() && self.available_devices.len() > 1
    }

    /// Get all available devices for multi-GPU operations
    pub fn all_devices(&self) -> &[Device] {
        &self.available_devices
    }

    /// Switch to a specific device (for multi-GPU operations)
    pub fn switch_device(&mut self, device_index: usize) -> Result<(), String> {
        if device_index >= self.available_devices.len() {
            return Err(format!(
                "Device index {} out of range (have {} devices)",
                device_index,
                self.available_devices.len()
            ));
        }

        self.device = self.available_devices[device_index].clone();
        self.current_device_index = device_index;
        Ok(())
    }

    /// Get current device index
    pub fn current_device_index(&self) -> usize {
        self.current_device_index
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
        Array2::from_shape_vec((dims[0], dims[1]), flat_data)
            .map_err(|_e| candle_core::Error::Msg("Processing...".to_string()))
    }

    /// Accelerated cosine similarity computation
    pub fn cosine_similarity_batch(
        &self,
        query: &Array1<f32>,
        vectors: &Array2<f32>,
    ) -> CandleResult<Array1<f32>> {
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
        let similarities = vectors_normalized
            .matmul(&query_normalized.unsqueeze(1)?)?
            .squeeze(1)?;

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
            return Err(candle_core::Error::Msg(
                "Cannot add empty vector list".into(),
            ));
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

    /// Multi-GPU parallel similarity search (when multiple GPUs available)
    pub fn multi_gpu_similarity_search(
        &self,
        query: &Array1<f32>,
        vectors: &Array2<f32>,
    ) -> CandleResult<Array1<f32>> {
        if !self.is_multi_gpu_available() || vectors.nrows() < 1000 {
            // Fall back to single GPU/CPU
            return self.cosine_similarity_batch(query, vectors);
        }

        println!(
            "🚀 Using multi-GPU similarity search across {} devices",
            self.device_count()
        );

        let chunk_size = vectors.nrows().div_ceil(self.device_count());
        let mut results = Vec::new();

        // Process chunks in parallel across different GPUs
        for (device_idx, chunk) in vectors
            .axis_chunks_iter(ndarray::Axis(0), chunk_size)
            .enumerate()
        {
            if device_idx >= self.available_devices.len() {
                break;
            }

            // Create tensor on specific device
            let device = &self.available_devices[device_idx];
            let query_tensor = Tensor::from_slice(
                query.as_slice().expect("Array must be contiguous"),
                query.len(),
                device,
            )?;

            let chunk_data = chunk.as_slice().expect("Chunk must be contiguous");
            let chunk_tensor =
                Tensor::from_slice(chunk_data, (chunk.nrows(), chunk.ncols()), device)?;

            // Compute similarities on this GPU
            let similarities =
                self.compute_cosine_similarity_tensor(&query_tensor, &chunk_tensor)?;
            let similarities_array = self.tensor_to_array(&similarities)?;
            results.push(similarities_array);
        }

        // Concatenate results
        let total_len: usize = results.iter().map(|r| r.len()).sum();
        let mut combined = Vec::with_capacity(total_len);
        for result in results {
            combined.extend(result.iter());
        }

        Ok(Array1::from_vec(combined))
    }

    /// Helper method to compute cosine similarity on tensor
    fn compute_cosine_similarity_tensor(
        &self,
        query: &Tensor,
        vectors: &Tensor,
    ) -> CandleResult<Tensor> {
        // Normalize query
        let query_norm = query.sqr()?.sum_keepdim(0)?.sqrt()?;
        let query_normalized = query.broadcast_div(&query_norm)?;

        // Normalize vectors
        let vectors_norm = vectors.sqr()?.sum_keepdim(1)?.sqrt()?;
        let vectors_normalized = vectors.broadcast_div(&vectors_norm)?;

        // Compute dot product (cosine similarity)
        vectors_normalized
            .matmul(&query_normalized.unsqueeze(1)?)?
            .squeeze(1)
    }

    /// Multi-GPU batch processing for large operations
    pub fn multi_gpu_batch_process<T, F>(&self, data: &[T], process_fn: F) -> Result<Vec<T>, String>
    where
        T: Clone + Send + Sync,
        F: Fn(&[T], usize) -> Result<Vec<T>, String> + Send + Sync,
    {
        if !self.is_multi_gpu_available() || data.len() < 1000 {
            // Single device processing
            return process_fn(data, 0);
        }

        use rayon::prelude::*;

        let chunk_size = data.len().div_ceil(self.device_count());

        println!(
            "🚀 Multi-GPU batch processing: {} items across {} devices",
            data.len(),
            self.device_count()
        );

        let results: Result<Vec<Vec<T>>, String> = data
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(device_idx, chunk)| {
                let gpu_idx = device_idx % self.device_count();
                process_fn(chunk, gpu_idx)
            })
            .collect();

        match results {
            Ok(chunks) => Ok(chunks.into_iter().flatten().collect()),
            Err(e) => Err(e),
        }
    }
}

impl Default for GPUAccelerator {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| GPUAccelerator {
            device: Device::Cpu,
            device_type: DeviceType::CPU,
            available_devices: vec![Device::Cpu],
            current_device_index: 0,
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

        let similarities = accelerator
            .cosine_similarity_batch(&query, &vectors)
            .unwrap();
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
        println!(
            "Benchmark: {:.2} GFLOPS on {:?}",
            gflops,
            accelerator.device_type()
        );
        assert!(gflops > 0.0);
    }
}
