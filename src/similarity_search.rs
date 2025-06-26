#![allow(clippy::type_complexity)]
use crate::gpu_acceleration::GPUAccelerator;
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Entry in the similarity search index
#[derive(Debug, Clone)]
pub struct PositionEntry {
    pub vector: Array1<f32>,
    pub evaluation: f32,
    pub norm_squared: f32,
}

/// Result from similarity search (reference-based)
#[derive(Debug)]
pub struct SearchResultRef<'a> {
    pub similarity: f32,
    pub evaluation: f32,
    pub vector: &'a Array1<f32>,
}

/// Result from similarity search (owned)
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub similarity: f32,
    pub evaluation: f32,
    pub vector: Array1<f32>,
}

impl PartialEq for SearchResult {
    fn eq(&self, other: &Self) -> bool {
        self.similarity == other.similarity
    }
}

impl Eq for SearchResult {}

impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for max-heap behavior in BinaryHeap
        other
            .similarity
            .partial_cmp(&self.similarity)
            .unwrap_or(Ordering::Equal)
    }
}

/// Similarity search engine for chess positions
#[derive(Clone)]
pub struct SimilaritySearch {
    /// All stored positions
    positions: Vec<PositionEntry>,
    /// Dimension of vectors
    vector_size: usize,
}

impl SimilaritySearch {
    /// Create a new similarity search engine
    pub fn new(vector_size: usize) -> Self {
        Self {
            positions: Vec::new(),
            vector_size,
        }
    }

    /// Add a position to the search index
    pub fn add_position(&mut self, vector: Array1<f32>, evaluation: f32) {
        assert_eq!(vector.len(), self.vector_size, "Vector size mismatch");

        let norm_squared =
            self.simd_dot_product(vector.as_slice().unwrap(), vector.as_slice().unwrap());

        self.positions.push(PositionEntry {
            vector,
            evaluation,
            norm_squared,
        });
    }

    /// Search for k most similar positions with references (memory efficient)
    pub fn search_ref(&self, query: &Array1<f32>, k: usize) -> Vec<(&Array1<f32>, f32, f32)> {
        // Note: GPU search not supported for reference version due to lifetime constraints
        // Fall back to CPU-based search methods

        // Use parallel CPU search for medium datasets
        if self.positions.len() > 100 {
            self.parallel_search_ref(query, k)
        } else {
            self.sequential_search_ref(query, k)
        }
    }

    /// Search for k most similar positions (automatically chooses best method: GPU > parallel > sequential)
    pub fn search(&self, query: &Array1<f32>, k: usize) -> Vec<(Array1<f32>, f32, f32)> {
        let gpu_accelerator = GPUAccelerator::global();

        // Use GPU acceleration for large datasets if available
        if gpu_accelerator.is_gpu_enabled() && self.positions.len() > 500 {
            match self.gpu_accelerated_search(query, k) {
                Ok(results) => return results,
                Err(e) => {
                    println!("GPU search failed ({e}), falling back to CPU");
                }
            }
        }

        // Fall back to parallel CPU search for medium datasets
        if self.positions.len() > 100 {
            self.parallel_search(query, k)
        } else {
            self.sequential_search(query, k)
        }
    }

    /// GPU-accelerated similarity search for large datasets
    pub fn gpu_accelerated_search(
        &self,
        query: &Array1<f32>,
        k: usize,
    ) -> Result<Vec<(Array1<f32>, f32, f32)>, Box<dyn std::error::Error>> {
        assert_eq!(query.len(), self.vector_size, "Query vector size mismatch");

        if self.positions.is_empty() {
            return Ok(Vec::new());
        }

        let gpu_accelerator = GPUAccelerator::global();

        // Prepare vectors matrix for GPU computation
        let mut vectors_data = Vec::with_capacity(self.positions.len() * self.vector_size);
        for entry in &self.positions {
            vectors_data.extend_from_slice(entry.vector.as_slice().unwrap());
        }

        let vectors_matrix =
            Array2::from_shape_vec((self.positions.len(), self.vector_size), vectors_data)?;

        // Compute similarities on GPU
        let similarities = gpu_accelerator.cosine_similarity_batch(query, &vectors_matrix)?;

        // Find top-k results
        let mut indexed_similarities: Vec<(usize, f32)> = similarities
            .iter()
            .enumerate()
            .map(|(i, &sim)| (i, sim))
            .collect();

        // Sort by similarity (descending)
        indexed_similarities
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top-k and prepare results
        let mut results = Vec::new();
        for (idx, similarity) in indexed_similarities.into_iter().take(k) {
            let entry = &self.positions[idx];
            results.push((entry.vector.clone(), entry.evaluation, similarity));
        }

        Ok(results)
    }

    /// Sequential search implementation with references (memory efficient)
    pub fn sequential_search_ref(
        &self,
        query: &Array1<f32>,
        k: usize,
    ) -> Vec<(&Array1<f32>, f32, f32)> {
        assert_eq!(query.len(), self.vector_size, "Query vector size mismatch");

        if self.positions.is_empty() {
            return Vec::new();
        }

        let query_norm_squared = query.dot(query);

        // Collect all similarities with indices
        let mut indexed_similarities: Vec<(usize, f32)> = self
            .positions
            .iter()
            .enumerate()
            .map(|(idx, entry)| {
                let similarity = self.cosine_similarity_fast(query, query_norm_squared, entry);
                (idx, similarity)
            })
            .collect();

        // Sort by similarity (descending)
        indexed_similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        // Take top k and return references
        indexed_similarities
            .into_iter()
            .take(k)
            .map(|(idx, similarity)| {
                let entry = &self.positions[idx];
                (&entry.vector, entry.evaluation, similarity)
            })
            .collect()
    }

    /// Sequential search implementation (for small datasets)
    pub fn sequential_search(&self, query: &Array1<f32>, k: usize) -> Vec<(Array1<f32>, f32, f32)> {
        assert_eq!(query.len(), self.vector_size, "Query vector size mismatch");

        if self.positions.is_empty() {
            return Vec::new();
        }

        let query_norm_squared = query.dot(query);

        // Use a min-heap to keep track of top-k results
        let mut heap = BinaryHeap::new();

        for entry in &self.positions {
            let similarity = self.cosine_similarity_fast(query, query_norm_squared, entry);

            let result = SearchResult {
                similarity,
                evaluation: entry.evaluation,
                vector: entry.vector.clone(),
            };

            if heap.len() < k {
                heap.push(result);
            } else if similarity > heap.peek().unwrap().similarity {
                heap.pop();
                heap.push(result);
            }
        }

        // Convert heap to sorted vector (highest similarity first)
        let mut results = Vec::new();
        while let Some(result) = heap.pop() {
            results.push((result.vector, result.evaluation, result.similarity));
        }

        // Reverse to get highest similarity first
        results.reverse();
        results
    }

    /// Parallel search implementation with references (memory efficient)
    pub fn parallel_search_ref(
        &self,
        query: &Array1<f32>,
        k: usize,
    ) -> Vec<(&Array1<f32>, f32, f32)> {
        assert_eq!(query.len(), self.vector_size, "Query vector size mismatch");

        if self.positions.is_empty() {
            return Vec::new();
        }

        let query_norm_squared = query.dot(query);

        // Calculate similarities in parallel with indices
        let mut indexed_similarities: Vec<(usize, f32)> = self
            .positions
            .par_iter()
            .enumerate()
            .map(|(idx, entry)| {
                let similarity = self.cosine_similarity_fast(query, query_norm_squared, entry);
                (idx, similarity)
            })
            .collect();

        // Sort by similarity (descending) and take top k
        indexed_similarities
            .par_sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        indexed_similarities.truncate(k);

        // Return references instead of clones
        indexed_similarities
            .into_iter()
            .map(|(idx, similarity)| {
                let entry = &self.positions[idx];
                (&entry.vector, entry.evaluation, similarity)
            })
            .collect()
    }

    /// Parallel search implementation (for larger datasets)
    pub fn parallel_search(&self, query: &Array1<f32>, k: usize) -> Vec<(Array1<f32>, f32, f32)> {
        assert_eq!(query.len(), self.vector_size, "Query vector size mismatch");

        if self.positions.is_empty() {
            return Vec::new();
        }

        let query_norm_squared = query.dot(query);

        // Calculate similarities in parallel
        let mut results: Vec<_> = self
            .positions
            .par_iter()
            .map(|entry| {
                let similarity = self.cosine_similarity_fast(query, query_norm_squared, entry);
                (entry.vector.clone(), entry.evaluation, similarity)
            })
            .collect();

        // Sort by similarity (descending) and take top k
        results.par_sort_unstable_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));
        results.truncate(k);

        results
    }

    /// Brute force search (for small datasets or comparison)
    pub fn brute_force_search(
        &self,
        query: &Array1<f32>,
        k: usize,
    ) -> Vec<(Array1<f32>, f32, f32)> {
        let mut results: Vec<_> = if self.positions.len() > 100 {
            // Use parallel processing for larger datasets
            self.positions
                .par_iter()
                .map(|entry| {
                    let similarity = self.cosine_similarity(query, &entry.vector);
                    (entry.vector.clone(), entry.evaluation, similarity)
                })
                .collect()
        } else {
            // Use sequential processing for smaller datasets
            self.positions
                .iter()
                .map(|entry| {
                    let similarity = self.cosine_similarity(query, &entry.vector);
                    (entry.vector.clone(), entry.evaluation, similarity)
                })
                .collect()
        };

        // Sort by similarity (descending)
        if results.len() > 1000 {
            results.par_sort_unstable_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));
        } else {
            results.sort_unstable_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));
        }

        // Take top k
        results.truncate(k);
        results
    }

    /// Calculate cosine similarity between query vector and a position entry (SIMD optimized)
    fn cosine_similarity_fast(
        &self,
        query: &Array1<f32>,
        query_norm_squared: f32,
        entry: &PositionEntry,
    ) -> f32 {
        let dot_product =
            self.simd_dot_product(query.as_slice().unwrap(), entry.vector.as_slice().unwrap());

        if query_norm_squared == 0.0 || entry.norm_squared == 0.0 {
            0.0
        } else {
            dot_product / (query_norm_squared.sqrt() * entry.norm_squared.sqrt())
        }
    }

    /// SIMD-optimized dot product calculation
    #[inline]
    fn simd_dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { self.avx2_dot_product(a, b) };
            } else if is_x86_feature_detected!("sse4.1") {
                return unsafe { self.sse_dot_product(a, b) };
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return unsafe { self.neon_dot_product(a, b) };
            }
        }

        // Fallback to scalar implementation
        self.scalar_dot_product(a, b)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let mut sum = _mm256_setzero_ps();
        let mut i = 0;

        // Process 8 floats at a time with AVX2
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let vmul = _mm256_mul_ps(va, vb);
            sum = _mm256_add_ps(sum, vmul);
            i += 8;
        }

        // Horizontal sum of the AVX2 register
        let mut result = [0.0f32; 8];
        _mm256_storeu_ps(result.as_mut_ptr(), sum);
        let mut final_sum = result.iter().sum::<f32>();

        // Handle remaining elements
        while i < len {
            final_sum += a[i] * b[i];
            i += 1;
        }

        final_sum
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse4.1")]
    unsafe fn sse_dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let mut sum = _mm_setzero_ps();
        let mut i = 0;

        // Process 4 floats at a time with SSE
        while i + 4 <= len {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            let vb = _mm_loadu_ps(b.as_ptr().add(i));
            let vmul = _mm_mul_ps(va, vb);
            sum = _mm_add_ps(sum, vmul);
            i += 4;
        }

        // Horizontal sum of the SSE register
        let mut result = [0.0f32; 4];
        _mm_storeu_ps(result.as_mut_ptr(), sum);
        let mut final_sum = result.iter().sum::<f32>();

        // Handle remaining elements
        while i < len {
            final_sum += a[i] * b[i];
            i += 1;
        }

        final_sum
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn neon_dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let mut sum = vdupq_n_f32(0.0);
        let mut i = 0;

        // Process 4 floats at a time with NEON
        while i + 4 <= len {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            let vmul = vmulq_f32(va, vb);
            sum = vaddq_f32(sum, vmul);
            i += 4;
        }

        // Horizontal sum of the NEON register
        let mut result = [0.0f32; 4];
        vst1q_f32(result.as_mut_ptr(), sum);
        let mut final_sum = result.iter().sum::<f32>();

        // Handle remaining elements
        while i < len {
            final_sum += a[i] * b[i];
            i += 1;
        }

        final_sum
    }

    #[inline]
    fn scalar_dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let mut sum = 0.0f32;

        // Unroll loop for better performance
        let mut i = 0;
        while i + 4 <= len {
            sum += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
            i += 4;
        }

        // Handle remaining elements
        while i < len {
            sum += a[i] * b[i];
            i += 1;
        }

        sum
    }

    /// Calculate cosine similarity between two vectors (fallback method)
    fn cosine_similarity(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        let dot_product = a.dot(b);
        let norm_a = a.dot(a).sqrt();
        let norm_b = b.dot(b).sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }

    /// Calculate Euclidean distance between two vectors
    fn euclidean_distance(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        (a - b).mapv(|x| x * x).sum().sqrt()
    }

    /// Search using Euclidean distance (alternative to cosine similarity)
    pub fn search_by_distance(
        &self,
        query: &Array1<f32>,
        k: usize,
    ) -> Vec<(Array1<f32>, f32, f32)> {
        let mut results: Vec<_> = self
            .positions
            .iter()
            .map(|entry| {
                let distance = self.euclidean_distance(query, &entry.vector);
                (entry.vector.clone(), entry.evaluation, distance)
            })
            .collect();

        // Sort by distance (ascending - smaller distance = more similar)
        results.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));

        // Take top k
        results.truncate(k);
        results
    }

    /// Get number of positions in the index
    pub fn size(&self) -> usize {
        self.positions.len()
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Clear all positions
    pub fn clear(&mut self) {
        self.positions.clear();
    }

    /// Get statistics about the stored vectors
    pub fn statistics(&self) -> SimilaritySearchStats {
        if self.positions.is_empty() {
            return SimilaritySearchStats {
                count: 0,
                avg_evaluation: 0.0,
                min_evaluation: 0.0,
                max_evaluation: 0.0,
            };
        }

        let evaluations: Vec<f32> = self.positions.iter().map(|p| p.evaluation).collect();
        let sum: f32 = evaluations.iter().sum();
        let avg = sum / evaluations.len() as f32;
        let min = evaluations.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = evaluations.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        SimilaritySearchStats {
            count: self.positions.len(),
            avg_evaluation: avg,
            min_evaluation: min,
            max_evaluation: max,
        }
    }

    /// Get all stored positions (for LSH indexing)
    pub fn get_all_positions(&self) -> Vec<(Array1<f32>, f32)> {
        self.positions
            .iter()
            .map(|entry| (entry.vector.clone(), entry.evaluation))
            .collect()
    }

    /// Get position vector by reference to avoid cloning
    pub fn get_position_ref(&self, index: usize) -> Option<(&Array1<f32>, f32)> {
        self.positions
            .get(index)
            .map(|entry| (&entry.vector, entry.evaluation))
    }

    /// Get all positions as references (memory efficient iterator)
    pub fn iter_positions(&self) -> impl Iterator<Item = (&Array1<f32>, f32)> {
        self.positions
            .iter()
            .map(|entry| (&entry.vector, entry.evaluation))
    }
}

/// Statistics about the similarity search index
#[derive(Debug, Clone)]
pub struct SimilaritySearchStats {
    pub count: usize,
    pub avg_evaluation: f32,
    pub min_evaluation: f32,
    pub max_evaluation: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_similarity_search_creation() {
        let search = SimilaritySearch::new(100);
        assert_eq!(search.size(), 0);
        assert!(search.is_empty());
    }

    #[test]
    fn test_add_and_search() {
        let mut search = SimilaritySearch::new(3);

        // Add some test vectors
        let vec1 = Array1::from(vec![1.0, 0.0, 0.0]);
        let vec2 = Array1::from(vec![0.0, 1.0, 0.0]);
        let vec3 = Array1::from(vec![0.0, 0.0, 1.0]);

        search.add_position(vec1.clone(), 1.0);
        search.add_position(vec2, 0.5);
        search.add_position(vec3, 0.0);

        assert_eq!(search.size(), 3);

        // Search for similar to vec1
        let results = search.search(&vec1, 2);
        assert_eq!(results.len(), 2);

        // First result should be identical (similarity = 1.0)
        assert!((results[0].2 - 1.0).abs() < 1e-6);
        assert!((results[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity() {
        let search = SimilaritySearch::new(2);

        let vec1 = Array1::from(vec![1.0, 0.0]);
        let vec2 = Array1::from(vec![1.0, 0.0]);
        let vec3 = Array1::from(vec![0.0, 1.0]);

        // Identical vectors
        assert!((search.cosine_similarity(&vec1, &vec2) - 1.0).abs() < 1e-6);

        // Orthogonal vectors
        assert!((search.cosine_similarity(&vec1, &vec3) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_statistics() {
        let mut search = SimilaritySearch::new(2);

        let vec = Array1::from(vec![1.0, 0.0]);
        search.add_position(vec.clone(), 1.0);
        search.add_position(vec.clone(), 2.0);
        search.add_position(vec, 3.0);

        let stats = search.statistics();
        assert_eq!(stats.count, 3);
        assert!((stats.avg_evaluation - 2.0).abs() < 1e-6);
        assert!((stats.min_evaluation - 1.0).abs() < 1e-6);
        assert!((stats.max_evaluation - 3.0).abs() < 1e-6);
    }
}
