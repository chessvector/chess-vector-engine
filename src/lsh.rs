use ndarray::{Array1, Array2};
use rand::Rng;
use rayon::prelude::*;
use std::collections::HashMap;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Locality Sensitive Hashing for approximate nearest neighbor search
#[derive(Clone)]
pub struct LSH {
    /// Number of hash tables
    num_tables: usize,
    /// Number of hash functions per table
    hash_size: usize,
    /// Vector dimension
    #[allow(dead_code)]
    vector_dim: usize,
    /// Random hyperplanes for each hash table
    hyperplanes: Vec<Array2<f32>>,
    /// Hash tables storing (hash, vector_index) pairs
    hash_tables: Vec<HashMap<Vec<bool>, Vec<usize>>>,
    /// Stored vectors for retrieval
    stored_vectors: Vec<Array1<f32>>,
    /// Associated data (evaluations)
    stored_data: Vec<f32>,
}

impl LSH {
    /// Create a new LSH index with dynamic sizing
    pub fn new(vector_dim: usize, num_tables: usize, hash_size: usize) -> Self {
        Self::with_expected_size(vector_dim, num_tables, hash_size, 1000)
    }

    /// Create a new LSH index with expected dataset size for optimal memory allocation
    pub fn with_expected_size(
        vector_dim: usize,
        num_tables: usize,
        hash_size: usize,
        expected_size: usize,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let mut hyperplanes = Vec::new();

        // Generate random hyperplanes for each table
        for _ in 0..num_tables {
            let mut table_hyperplanes = Array2::zeros((hash_size, vector_dim));
            for i in 0..hash_size {
                for j in 0..vector_dim {
                    table_hyperplanes[[i, j]] = rng.gen_range(-1.0..1.0);
                }
            }
            hyperplanes.push(table_hyperplanes);
        }

        // Calculate optimal hash table capacity based on expected size
        // Assume roughly 20% occupancy for good performance
        let bucket_capacity = (expected_size as f32 / 5.0).ceil() as usize;
        let optimal_capacity = bucket_capacity.max(100);

        let hash_tables = vec![HashMap::with_capacity(optimal_capacity); num_tables];

        Self {
            num_tables,
            hash_size,
            vector_dim,
            hyperplanes,
            hash_tables,
            stored_vectors: Vec::with_capacity(expected_size),
            stored_data: Vec::with_capacity(expected_size),
        }
    }

    /// Add a vector to the LSH index with dynamic resizing
    pub fn add_vector(&mut self, vector: Array1<f32>, data: f32) {
        let index = self.stored_vectors.len();

        // Check if we need to resize hash tables (when load factor > 0.75)
        let current_load =
            self.stored_vectors.len() as f32 / (self.hash_tables[0].capacity() as f32 * 0.2);
        if current_load > 0.75 {
            self.resize_hash_tables();
        }

        // Hash the vector in each table before storing (to avoid clone)
        let mut hashes = Vec::with_capacity(self.num_tables);
        for table_idx in 0..self.num_tables {
            hashes.push(self.hash_vector(&vector, table_idx));
        }

        // Now store the vector and data
        self.stored_vectors.push(vector);
        self.stored_data.push(data);

        // Insert into hash tables using pre-computed hashes
        for (table_idx, hash) in hashes.into_iter().enumerate() {
            self.hash_tables[table_idx]
                .entry(hash)
                .or_insert_with(|| Vec::with_capacity(8)) // Increased pre-allocation
                .push(index);
        }
    }

    /// Resize hash tables when they become too full
    fn resize_hash_tables(&mut self) {
        let new_capacity = (self.hash_tables[0].capacity() * 2).max(self.stored_vectors.len());

        for table in &mut self.hash_tables {
            // Reserve additional capacity to avoid frequent rehashing
            table.reserve(new_capacity - table.capacity());
        }
    }

    /// Find approximate nearest neighbors
    pub fn query(&self, query_vector: &Array1<f32>, k: usize) -> Vec<(Array1<f32>, f32, f32)> {
        let mut candidates = std::collections::HashSet::new();
        let max_candidates = (self.stored_vectors.len() / 4)
            .max(k * 10)
            .min(self.stored_vectors.len());

        // Parallelize hash table queries when we have many tables
        if self.num_tables > 4 {
            // Collect candidates from hash tables in parallel
            let candidate_sets: Vec<Vec<usize>> = (0..self.num_tables)
                .into_par_iter()
                .map(|table_idx| {
                    let hash = self.hash_vector(query_vector, table_idx);
                    if let Some(bucket) = self.hash_tables[table_idx].get(&hash) {
                        bucket.clone()
                    } else {
                        Vec::new()
                    }
                })
                .collect();

            // Merge candidate sets sequentially (avoiding race conditions on HashSet)
            for candidate_set in candidate_sets {
                for idx in candidate_set {
                    candidates.insert(idx);
                    if candidates.len() >= max_candidates {
                        break;
                    }
                }
                if candidates.len() >= max_candidates {
                    break;
                }
            }
        } else {
            // Sequential collection for smaller numbers of tables
            for table_idx in 0..self.num_tables {
                if candidates.len() >= max_candidates {
                    break;
                }

                let hash = self.hash_vector(query_vector, table_idx);
                if let Some(bucket) = self.hash_tables[table_idx].get(&hash) {
                    for &idx in bucket {
                        candidates.insert(idx);
                        if candidates.len() >= max_candidates {
                            break;
                        }
                    }
                }
            }
        }

        // If we have too few candidates, use a more efficient approach
        if candidates.len() < k * 3 && self.stored_vectors.len() > k * 3 {
            // Instead of random sampling, just take the first few indices
            let needed = k * 5;
            for idx in 0..needed.min(self.stored_vectors.len()) {
                candidates.insert(idx);
                if candidates.len() >= needed {
                    break;
                }
            }
        }

        // Calculate similarities for candidates in parallel for large candidate sets
        let mut results = if candidates.len() > 50 {
            candidates
                .par_iter()
                .map(|&idx| {
                    let stored_vector = &self.stored_vectors[idx];
                    let similarity = cosine_similarity(query_vector, stored_vector);
                    (stored_vector.clone(), self.stored_data[idx], similarity)
                })
                .collect()
        } else {
            // Sequential for smaller candidate sets
            let mut results = Vec::with_capacity(candidates.len());
            for &idx in &candidates {
                let stored_vector = &self.stored_vectors[idx];
                let similarity = cosine_similarity(query_vector, stored_vector);
                results.push((stored_vector.clone(), self.stored_data[idx], similarity));
            }
            results
        };

        // Sort by similarity (descending) and return top k
        if results.len() > 100 {
            results.par_sort_unstable_by(|a, b| {
                b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            results.sort_unstable_by(|a, b| {
                b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        results.truncate(k);

        results
    }

    /// Hash a vector using hyperplanes for a specific table
    fn hash_vector(&self, vector: &Array1<f32>, table_idx: usize) -> Vec<bool> {
        let hyperplanes = &self.hyperplanes[table_idx];

        // Use ndarray's optimized matrix-vector multiplication
        let dot_products = hyperplanes.dot(vector);

        // Convert to hash bits in one pass
        dot_products.iter().map(|&x| x >= 0.0).collect()
    }

    /// Get statistics about the index
    pub fn stats(&self) -> LSHStats {
        let mut bucket_sizes = Vec::new();
        let mut total_buckets = 0;
        let mut non_empty_buckets = 0;

        for table in &self.hash_tables {
            // Use checked_shl to avoid overflow - if hash_size is too large, use max value
            let buckets_per_table = 1_usize
                .checked_shl(self.hash_size as u32)
                .unwrap_or(usize::MAX);
            total_buckets += buckets_per_table;
            non_empty_buckets += table.len();

            for bucket in table.values() {
                bucket_sizes.push(bucket.len());
            }
        }

        bucket_sizes.sort();
        let median_bucket_size = if bucket_sizes.is_empty() {
            0.0
        } else {
            bucket_sizes[bucket_sizes.len() / 2] as f32
        };

        let avg_bucket_size = if bucket_sizes.is_empty() {
            0.0
        } else {
            bucket_sizes.iter().sum::<usize>() as f32 / bucket_sizes.len() as f32
        };

        LSHStats {
            num_vectors: self.stored_vectors.len(),
            num_tables: self.num_tables,
            hash_size: self.hash_size,
            total_buckets,
            non_empty_buckets,
            avg_bucket_size,
            median_bucket_size,
            max_bucket_size: bucket_sizes.last().copied().unwrap_or(0),
        }
    }

    /// Save LSH configuration and hash functions to database
    pub fn save_to_database(
        &self,
        db: &crate::persistence::Database,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use crate::persistence::{LSHHashFunction, LSHTableData};

        // Convert hyperplanes to serializable format
        let mut hash_functions = Vec::new();
        for hyperplane_matrix in &self.hyperplanes {
            for row in hyperplane_matrix.rows() {
                hash_functions.push(LSHHashFunction {
                    random_vector: row.to_vec().iter().map(|&x| x as f64).collect(),
                    threshold: 0.0, // We use zero threshold for hyperplane hashing
                });
            }
        }

        let config = LSHTableData {
            num_tables: self.num_tables,
            num_hash_functions: self.hash_size,
            vector_dim: self.vector_dim,
            hash_functions,
        };

        db.save_lsh_config(&config)?;

        // Clear existing bucket data and save new buckets
        db.clear_lsh_buckets()?;

        // Save hash bucket assignments (this maps positions to buckets)
        for (table_idx, table) in self.hash_tables.iter().enumerate() {
            for (hash_bits, indices) in table {
                let hash_string = hash_bits
                    .iter()
                    .map(|&b| if b { '1' } else { '0' })
                    .collect::<String>();

                for &position_idx in indices {
                    db.save_lsh_bucket(table_idx, &hash_string, position_idx as i64)?;
                }
            }
        }

        Ok(())
    }

    /// Load LSH configuration from database and rebuild hash tables
    pub fn load_from_database(
        db: &crate::persistence::Database,
        positions: &[(Array1<f32>, f32)],
    ) -> Result<Option<Self>, Box<dyn std::error::Error>> {
        let config = match db.load_lsh_config()? {
            Some(config) => config,
            None => return Ok(None),
        };

        // Reconstruct hyperplanes from saved hash functions
        let mut hyperplanes = Vec::new();
        let functions_per_table = config.num_hash_functions;

        for table_idx in 0..config.num_tables {
            let start_idx = table_idx * functions_per_table;
            let end_idx = start_idx + functions_per_table;

            if end_idx <= config.hash_functions.len() {
                let mut table_hyperplanes = Array2::zeros((functions_per_table, config.vector_dim));

                for (func_idx, hash_func) in
                    config.hash_functions[start_idx..end_idx].iter().enumerate()
                {
                    for (dim_idx, &value) in hash_func.random_vector.iter().enumerate() {
                        if dim_idx < config.vector_dim {
                            table_hyperplanes[[func_idx, dim_idx]] = value as f32;
                        }
                    }
                }

                hyperplanes.push(table_hyperplanes);
            }
        }

        // Create LSH with loaded configuration
        let mut lsh = Self {
            num_tables: config.num_tables,
            hash_size: config.num_hash_functions,
            vector_dim: config.vector_dim,
            hyperplanes,
            hash_tables: vec![HashMap::with_capacity(positions.len().max(100)); config.num_tables],
            stored_vectors: Vec::new(),
            stored_data: Vec::new(),
        };

        // Rebuild the index with provided positions
        for (vector, evaluation) in positions {
            lsh.add_vector(vector.clone(), *evaluation);
        }

        Ok(Some(lsh))
    }

    /// Create LSH from database or return None if no saved configuration exists
    pub fn from_database_or_new(
        db: &crate::persistence::Database,
        positions: &[(Array1<f32>, f32)],
        vector_dim: usize,
        num_tables: usize,
        hash_size: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        match Self::load_from_database(db, positions)? {
            Some(lsh) => {
                println!(
                    "Loaded LSH configuration from database with {} vectors",
                    lsh.stored_vectors.len()
                );
                Ok(lsh)
            }
            None => {
                println!("No saved LSH configuration found, creating new LSH index");
                let mut lsh =
                    Self::with_expected_size(vector_dim, num_tables, hash_size, positions.len());
                for (vector, evaluation) in positions {
                    lsh.add_vector(vector.clone(), *evaluation);
                }
                Ok(lsh)
            }
        }
    }
}

/// LSH performance statistics
#[derive(Debug)]
pub struct LSHStats {
    pub num_vectors: usize,
    pub num_tables: usize,
    pub hash_size: usize,
    pub total_buckets: usize,
    pub non_empty_buckets: usize,
    pub avg_bucket_size: f32,
    pub median_bucket_size: f32,
    pub max_bucket_size: usize,
}

/// Calculate cosine similarity between two vectors (SIMD optimized)
fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let a_slice = a.as_slice().unwrap();
    let b_slice = b.as_slice().unwrap();

    let dot_product = simd_dot_product(a_slice, b_slice);
    let norm_a_sq = simd_dot_product(a_slice, a_slice);
    let norm_b_sq = simd_dot_product(b_slice, b_slice);

    if norm_a_sq == 0.0 || norm_b_sq == 0.0 {
        0.0
    } else {
        dot_product / (norm_a_sq * norm_b_sq).sqrt()
    }
}

/// SIMD-optimized dot product calculation
#[inline]
fn simd_dot_product(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2_dot_product(a, b) };
        } else if is_x86_feature_detected!("sse4.1") {
            return unsafe { sse_dot_product(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { neon_dot_product(a, b) };
        }
    }

    // Fallback to scalar implementation
    scalar_dot_product(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_dot_product(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let vmul = _mm256_mul_ps(va, vb);
        sum = _mm256_add_ps(sum, vmul);
        i += 8;
    }

    let mut result = [0.0f32; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), sum);
    let mut final_sum = result.iter().sum::<f32>();

    while i < len {
        final_sum += a[i] * b[i];
        i += 1;
    }

    final_sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn sse_dot_product(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut sum = _mm_setzero_ps();
    let mut i = 0;

    while i + 4 <= len {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let vb = _mm_loadu_ps(b.as_ptr().add(i));
        let vmul = _mm_mul_ps(va, vb);
        sum = _mm_add_ps(sum, vmul);
        i += 4;
    }

    let mut result = [0.0f32; 4];
    _mm_storeu_ps(result.as_mut_ptr(), sum);
    let mut final_sum = result.iter().sum::<f32>();

    while i < len {
        final_sum += a[i] * b[i];
        i += 1;
    }

    final_sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_dot_product(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut sum = vdupq_n_f32(0.0);
    let mut i = 0;

    while i + 4 <= len {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        let vmul = vmulq_f32(va, vb);
        sum = vaddq_f32(sum, vmul);
        i += 4;
    }

    let mut result = [0.0f32; 4];
    vst1q_f32(result.as_mut_ptr(), sum);
    let mut final_sum = result.iter().sum::<f32>();

    while i < len {
        final_sum += a[i] * b[i];
        i += 1;
    }

    final_sum
}

#[inline]
fn scalar_dot_product(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut sum = 0.0f32;
    let mut i = 0;

    while i + 4 <= len {
        sum += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
        i += 4;
    }

    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_lsh_creation() {
        let lsh = LSH::new(128, 4, 8);
        assert_eq!(lsh.num_tables, 4);
        assert_eq!(lsh.hash_size, 8);
        assert_eq!(lsh.vector_dim, 128);
    }

    #[test]
    fn test_lsh_add_and_query() {
        let mut lsh = LSH::new(4, 2, 4);

        // Add some test vectors
        let vec1 = Array1::from(vec![1.0, 0.0, 0.0, 0.0]);
        let vec2 = Array1::from(vec![0.0, 1.0, 0.0, 0.0]);
        let vec3 = Array1::from(vec![1.0, 0.1, 0.0, 0.0]); // Similar to vec1

        lsh.add_vector(vec1.clone(), 1.0);
        lsh.add_vector(vec2, 2.0);
        lsh.add_vector(vec3, 1.1);

        // Query with vec1 should find similar vectors
        let results = lsh.query(&vec1, 2);
        assert!(!results.is_empty());

        // The most similar should be vec1 itself or vec3
        assert!(results[0].2 > 0.8); // High similarity
    }

    #[test]
    fn test_cosine_similarity() {
        let a = Array1::from(vec![1.0, 0.0, 0.0]);
        let b = Array1::from(vec![1.0, 0.0, 0.0]);
        let c = Array1::from(vec![0.0, 1.0, 0.0]);

        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 1e-6);
    }
}
