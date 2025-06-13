use ndarray::{Array1, Array2};
use rand::Rng;
use std::collections::HashMap;

/// Locality Sensitive Hashing for approximate nearest neighbor search
pub struct LSH {
    /// Number of hash tables
    num_tables: usize,
    /// Number of hash functions per table
    hash_size: usize,
    /// Vector dimension
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
    /// Create a new LSH index
    pub fn new(vector_dim: usize, num_tables: usize, hash_size: usize) -> Self {
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
        
        let hash_tables = vec![HashMap::with_capacity(100); num_tables];
        
        Self {
            num_tables,
            hash_size,
            vector_dim,
            hyperplanes,
            hash_tables,
            stored_vectors: Vec::with_capacity(1000), // Pre-allocate for better performance
            stored_data: Vec::with_capacity(1000),
        }
    }
    
    /// Add a vector to the LSH index
    pub fn add_vector(&mut self, vector: Array1<f32>, data: f32) {
        let index = self.stored_vectors.len();
        
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
                .or_insert_with(|| Vec::with_capacity(4)) // Pre-allocate capacity
                .push(index);
        }
    }
    
    /// Find approximate nearest neighbors
    pub fn query(&self, query_vector: &Array1<f32>, k: usize) -> Vec<(Array1<f32>, f32, f32)> {
        let mut candidates = std::collections::HashSet::new();
        let max_candidates = (self.stored_vectors.len() / 4).max(k * 10).min(self.stored_vectors.len());
        
        // Collect candidates from hash tables, but limit the total
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
        
        // Calculate similarities for candidates more efficiently
        let mut results = Vec::with_capacity(candidates.len());
        for &idx in &candidates {
            let stored_vector = &self.stored_vectors[idx];
            let similarity = cosine_similarity(query_vector, stored_vector);
            results.push((stored_vector.clone(), self.stored_data[idx], similarity));
        }
        
        // Sort by similarity (descending) and return top k
        results.sort_unstable_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
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
            total_buckets += 1 << self.hash_size; // 2^hash_size possible buckets
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

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    // Use ndarray's optimized dot product
    let dot_product = a.dot(b);
    
    // Calculate norms efficiently
    let norm_a_sq: f32 = a.dot(a);
    let norm_b_sq: f32 = b.dot(b);
    
    if norm_a_sq == 0.0 || norm_b_sq == 0.0 {
        0.0
    } else {
        dot_product / (norm_a_sq * norm_b_sq).sqrt()
    }
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