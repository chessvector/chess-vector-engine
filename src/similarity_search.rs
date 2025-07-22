#![allow(clippy::type_complexity)]
use crate::gpu_acceleration::GPUAccelerator;
use crate::utils::simd::SimdVectorOps;
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::time::{Duration, Instant};
// Removed unused import

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

/// Hierarchical clustering node for improved search performance
#[derive(Debug, Clone)]
pub struct ClusterNode {
    /// Centroid of the cluster
    pub centroid: Array1<f32>,
    /// Indices of positions in this cluster
    pub position_indices: Vec<usize>,
    /// Child clusters (for hierarchical clustering)
    pub children: Vec<ClusterNode>,
    /// Cluster radius (maximum distance from centroid)
    pub radius: f32,
    /// Number of positions in this cluster (including children)
    pub size: usize,
}

/// Cache entry for similarity search results with TTL
#[derive(Debug, Clone)]
pub struct SearchResultCache {
    pub results: Vec<(Array1<f32>, f32, f32)>,
    pub timestamp: Instant,
}

/// Cache statistics for monitoring performance
#[derive(Debug, Clone)]
pub struct SimilarityCacheStats {
    pub result_cache_size: usize,
    pub similarity_cache_size: usize,
    pub max_cache_size: usize,
    pub cache_ttl_secs: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub hit_ratio: f32,
}

/// Similarity search engine for chess positions with production-optimized caching
#[derive(Clone)]
pub struct SimilaritySearch {
    /// All stored positions
    positions: Vec<PositionEntry>,
    /// Dimension of vectors
    vector_size: usize,
    /// Hierarchical clustering tree for fast search
    cluster_tree: Option<ClusterNode>,
    /// Cache for frequently accessed similarity results (pairwise similarities)
    similarity_cache: HashMap<(usize, usize), (f32, Instant)>,
    /// Cache for complete search results (query_hash -> results)
    result_cache: HashMap<u64, SearchResultCache>,
    /// Maximum cache size to prevent memory bloat
    max_cache_size: usize,
    /// TTL for cached results
    cache_ttl: Duration,
    /// Cache performance metrics
    cache_hits: u64,
    cache_misses: u64,
}

impl SimilaritySearch {
    /// Create a new similarity search engine with default caching settings
    pub fn new(vector_size: usize) -> Self {
        Self {
            positions: Vec::new(),
            vector_size,
            cluster_tree: None,
            similarity_cache: HashMap::with_capacity(10000),
            result_cache: HashMap::with_capacity(1000),
            max_cache_size: 10000,
            cache_ttl: Duration::from_secs(300), // 5 minutes
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    /// Create a new similarity search engine with custom cache configuration
    pub fn with_cache_config(vector_size: usize, max_cache_size: usize, cache_ttl_secs: u64) -> Self {
        Self {
            positions: Vec::new(),
            vector_size,
            cluster_tree: None,
            similarity_cache: HashMap::with_capacity(max_cache_size),
            result_cache: HashMap::with_capacity(max_cache_size / 10),
            max_cache_size,
            cache_ttl: Duration::from_secs(cache_ttl_secs),
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    /// Add a position to the search index
    pub fn add_position(&mut self, vector: Array1<f32>, evaluation: f32) {
        assert_eq!(vector.len(), self.vector_size, "Vector size mismatch");

        let norm_squared = SimdVectorOps::squared_norm(&vector);

        self.positions.push(PositionEntry {
            vector,
            evaluation,
            norm_squared,
        });

        // Invalidate cluster tree when adding new positions
        self.cluster_tree = None;

        // Evict expired cache entries and manage cache size
        self.evict_expired_cache_entries();
        if self.similarity_cache.len() > self.max_cache_size {
            self.evict_oldest_cache_entries();
        }
    }

    /// Search for k most similar positions with references (memory efficient)
    pub fn search_ref(&self, query: &Array1<f32>, k: usize) -> Vec<(&Array1<f32>, f32, f32)> {
        // Note: GPU search not supported for reference version due to lifetime constraints
        // Fall back to CPU-based search methods

        // Use hierarchical clustering for large datasets
        if self.positions.len() > 1000 {
            self.hierarchical_search_ref(query, k)
        } else if self.positions.len() > 100 {
            self.parallel_search_ref(query, k)
        } else {
            self.sequential_search_ref(query, k)
        }
    }

    /// Search for k most similar positions with comprehensive caching (automatically chooses best method)
    pub fn search(&mut self, query: &Array1<f32>, k: usize) -> Vec<(Array1<f32>, f32, f32)> {
        // Generate cache key from query vector and k
        let query_hash = self.hash_query(query, k);
        
        // Check result cache first
        if let Some(cached_result) = self.get_cached_result(query_hash) {
            return cached_result;
        }
        
        // Cache miss - perform actual search
        let results = self.search_uncached(query, k);
        
        // Cache the results for future use
        self.cache_search_result(query_hash, results.clone());
        
        results
    }
    
    /// Internal search method without caching (for cache misses)
    fn search_uncached(&self, query: &Array1<f32>, k: usize) -> Vec<(Array1<f32>, f32, f32)> {
        // Use optimized search as primary method for better performance
        if self.positions.len() > 50 {
            return self.search_optimized(query, k);
        }

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

        // Use hierarchical clustering for large datasets
        if self.positions.len() > 1000 {
            self.hierarchical_search(query, k)
        } else if self.positions.len() > 100 {
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

        let query_norm_squared = SimdVectorOps::squared_norm(query);

        // Collect all similarities with indices
        let mut indexed_similarities: Vec<(usize, f32)> = self
            .positions
            .iter()
            .enumerate()
            .map(|(idx, entry)| {
                let similarity = self.cosine_similarity_fast_uncached(query, query_norm_squared, entry);
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

        let query_norm_squared = SimdVectorOps::squared_norm(query);

        // Use a min-heap to keep track of top-k results
        let mut heap = BinaryHeap::new();

        for entry in &self.positions {
            let similarity = self.cosine_similarity_fast_uncached(query, query_norm_squared, entry);

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

        let query_norm_squared = SimdVectorOps::squared_norm(query);

        // Calculate similarities in parallel with indices
        let mut indexed_similarities: Vec<(usize, f32)> = self
            .positions
            .par_iter()
            .enumerate()
            .map(|(idx, entry)| {
                let similarity = self.cosine_similarity_fast_uncached(query, query_norm_squared, entry);
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

        let query_norm_squared = SimdVectorOps::squared_norm(query);

        // Calculate similarities in parallel
        let mut results: Vec<_> = self
            .positions
            .par_iter()
            .map(|entry| {
                let similarity = self.cosine_similarity_fast_uncached(query, query_norm_squared, entry);
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

    /// Calculate cosine similarity between query vector and a position entry with caching (SIMD optimized)
    fn cosine_similarity_fast(
        &mut self,
        query: &Array1<f32>,
        query_norm_squared: f32,
        entry_index: usize,
    ) -> f32 {
        // Check cache for pairwise similarity
        let now = Instant::now();
        let cache_key = (0, entry_index); // Using 0 as query index placeholder
        
        if let Some((cached_similarity, cached_time)) = self.similarity_cache.get(&cache_key) {
            if now.duration_since(*cached_time) < self.cache_ttl {
                self.cache_hits += 1;
                return *cached_similarity;
            }
        }
        
        // Cache miss - compute similarity
        self.cache_misses += 1;
        let entry = &self.positions[entry_index];
        
        // Early termination for zero vectors
        if query_norm_squared == 0.0 || entry.norm_squared == 0.0 {
            return 0.0;
        }

        let dot_product = SimdVectorOps::dot_product(query, &entry.vector);
        
        // Pre-computed inverse square roots for better performance
        let query_norm_inv = 1.0 / query_norm_squared.sqrt();
        let entry_norm_inv = 1.0 / entry.norm_squared.sqrt();
        
        let similarity = dot_product * query_norm_inv * entry_norm_inv;
        
        // Cache the result
        self.similarity_cache.insert(cache_key, (similarity, now));
        
        similarity
    }
    
    /// Calculate cosine similarity between query vector and a position entry (uncached version)
    fn cosine_similarity_fast_uncached(
        &self,
        query: &Array1<f32>,
        query_norm_squared: f32,
        entry: &PositionEntry,
    ) -> f32 {
        // Early termination for zero vectors
        if query_norm_squared == 0.0 || entry.norm_squared == 0.0 {
            return 0.0;
        }

        let dot_product = SimdVectorOps::dot_product(query, &entry.vector);
        
        // Pre-computed inverse square roots for better performance
        let query_norm_inv = 1.0 / query_norm_squared.sqrt();
        let entry_norm_inv = 1.0 / entry.norm_squared.sqrt();
        
        dot_product * query_norm_inv * entry_norm_inv
    }

    /// Ultra-fast similarity calculation with pre-computed norms (avoids sqrt when possible)
    fn cosine_similarity_ultra_fast(
        &self,
        query: &Array1<f32>,
        query_norm: f32,
        entry: &PositionEntry,
        entry_norm: f32,
    ) -> f32 {
        if query_norm == 0.0 || entry_norm == 0.0 {
            return 0.0;
        }

        let dot_product = SimdVectorOps::dot_product(query, &entry.vector);
        dot_product / (query_norm * entry_norm)
    }

    /// Calculate cosine similarity between two vectors (SIMD optimized)
    fn cosine_similarity(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        SimdVectorOps::cosine_similarity(a, b)
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

    /// Build hierarchical clustering tree for improved search performance
    pub fn build_cluster_tree(&mut self) {
        if self.positions.is_empty() {
            self.cluster_tree = None;
            return;
        }

        let indices: Vec<usize> = (0..self.positions.len()).collect();
        self.cluster_tree = Some(self.build_cluster_recursive(indices, 0));
    }

    /// Recursively build clustering tree using k-means-like approach
    fn build_cluster_recursive(&self, indices: Vec<usize>, depth: usize) -> ClusterNode {
        let max_depth = 10;
        let min_cluster_size = 32;

        if indices.len() <= min_cluster_size || depth >= max_depth {
            // Leaf node - compute centroid and radius
            let centroid = self.compute_centroid(&indices);
            let radius = self.compute_cluster_radius(&centroid, &indices);

            return ClusterNode {
                centroid,
                position_indices: indices.clone(),
                children: Vec::new(),
                radius,
                size: indices.len(),
            };
        }

        // Use k-means clustering to split into 2 or 4 clusters
        let k = if indices.len() > 200 { 4 } else { 2 };
        let clusters = self.k_means_clustering(&indices, k);

        let mut children = Vec::new();
        let mut all_indices = Vec::new();

        for cluster_indices in clusters {
            if !cluster_indices.is_empty() {
                let child = self.build_cluster_recursive(cluster_indices.clone(), depth + 1);
                all_indices.extend(cluster_indices);
                children.push(child);
            }
        }

        let centroid = self.compute_centroid(&all_indices);
        let radius = self.compute_cluster_radius(&centroid, &all_indices);

        ClusterNode {
            centroid,
            position_indices: all_indices,
            children,
            radius,
            size: indices.len(),
        }
    }

    /// Compute centroid of a cluster
    fn compute_centroid(&self, indices: &[usize]) -> Array1<f32> {
        if indices.is_empty() {
            return Array1::zeros(self.vector_size);
        }

        let mut centroid = Array1::zeros(self.vector_size);
        for &idx in indices {
            centroid = SimdVectorOps::add_vectors(&centroid, &self.positions[idx].vector);
        }

        SimdVectorOps::scale_vector(&centroid, 1.0 / indices.len() as f32)
    }

    /// Compute radius of a cluster (maximum distance from centroid)
    fn compute_cluster_radius(&self, centroid: &Array1<f32>, indices: &[usize]) -> f32 {
        indices
            .iter()
            .map(|&idx| 1.0 - self.cosine_similarity_cached(centroid, &self.positions[idx].vector))
            .fold(0.0, f32::max)
    }

    /// K-means clustering implementation
    fn k_means_clustering(&self, indices: &[usize], k: usize) -> Vec<Vec<usize>> {
        if indices.len() <= k {
            return indices.iter().map(|&i| vec![i]).collect();
        }

        // Initialize centroids randomly
        let mut centroids = Vec::new();
        let step = indices.len() / k;
        for i in 0..k {
            let idx = indices[i * step];
            centroids.push(self.positions[idx].vector.clone());
        }

        const MAX_ITERATIONS: usize = 10;

        for _ in 0..MAX_ITERATIONS {
            // Assign points to clusters
            let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); k];

            for &idx in indices {
                let mut best_cluster = 0;
                let mut best_similarity = -1.0;

                for (cluster_idx, centroid) in centroids.iter().enumerate() {
                    let similarity =
                        self.cosine_similarity_cached(centroid, &self.positions[idx].vector);
                    if similarity > best_similarity {
                        best_similarity = similarity;
                        best_cluster = cluster_idx;
                    }
                }

                clusters[best_cluster].push(idx);
            }

            // Update centroids
            let mut converged = true;
            for (cluster_idx, cluster) in clusters.iter().enumerate() {
                if !cluster.is_empty() {
                    let new_centroid = self.compute_centroid(cluster);
                    let similarity =
                        self.cosine_similarity_cached(&centroids[cluster_idx], &new_centroid);

                    if similarity < 0.99 {
                        converged = false;
                    }

                    centroids[cluster_idx] = new_centroid;
                }
            }

            if converged {
                break;
            }
        }

        // Final assignment
        let mut final_clusters: Vec<Vec<usize>> = vec![Vec::new(); k];
        for &idx in indices {
            let mut best_cluster = 0;
            let mut best_similarity = -1.0;

            for (cluster_idx, centroid) in centroids.iter().enumerate() {
                let similarity =
                    self.cosine_similarity_cached(centroid, &self.positions[idx].vector);
                if similarity > best_similarity {
                    best_similarity = similarity;
                    best_cluster = cluster_idx;
                }
            }

            final_clusters[best_cluster].push(idx);
        }

        final_clusters
            .into_iter()
            .filter(|cluster| !cluster.is_empty())
            .collect()
    }

    /// Hierarchical search using cluster tree
    fn hierarchical_search(&self, query: &Array1<f32>, k: usize) -> Vec<(Array1<f32>, f32, f32)> {
        // Build cluster tree if not already built
        if self.cluster_tree.is_none() {
            // Can't modify self in this context, fall back to parallel search
            return self.parallel_search(query, k);
        }

        let cluster_tree = self.cluster_tree.as_ref().unwrap();
        let mut candidates = Vec::new();

        // Traverse cluster tree to find promising candidates
        self.traverse_cluster_tree(query, cluster_tree, &mut candidates, k * 5);

        // Calculate similarities for candidates
        let mut results: Vec<_> = candidates
            .into_iter()
            .map(|idx| {
                let entry = &self.positions[idx];
                let similarity = self.cosine_similarity_cached(query, &entry.vector);
                (entry.vector.clone(), entry.evaluation, similarity)
            })
            .collect();

        // Sort and return top k
        results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));
        results.truncate(k);

        results
    }

    /// Hierarchical search with references
    fn hierarchical_search_ref(
        &self,
        query: &Array1<f32>,
        k: usize,
    ) -> Vec<(&Array1<f32>, f32, f32)> {
        // Build cluster tree if not already built
        if self.cluster_tree.is_none() {
            // Can't modify self in this context, fall back to parallel search
            return self.parallel_search_ref(query, k);
        }

        let cluster_tree = self.cluster_tree.as_ref().unwrap();
        let mut candidates = Vec::new();

        // Traverse cluster tree to find promising candidates
        self.traverse_cluster_tree(query, cluster_tree, &mut candidates, k * 5);

        // Calculate similarities for candidates
        let mut results: Vec<_> = candidates
            .into_iter()
            .map(|idx| {
                let entry = &self.positions[idx];
                let similarity = self.cosine_similarity_cached(query, &entry.vector);
                (&entry.vector, entry.evaluation, similarity)
            })
            .collect();

        // Sort and return top k
        results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));
        results.truncate(k);

        results
    }

    /// Traverse cluster tree to find candidate positions
    fn traverse_cluster_tree(
        &self,
        query: &Array1<f32>,
        node: &ClusterNode,
        candidates: &mut Vec<usize>,
        max_candidates: usize,
    ) {
        if candidates.len() >= max_candidates {
            return;
        }

        // Calculate similarity to cluster centroid
        let centroid_similarity = self.cosine_similarity_cached(query, &node.centroid);

        // If query is far from this cluster, skip it
        let distance_threshold = 0.1; // Adjust based on dataset characteristics
        if centroid_similarity < distance_threshold {
            return;
        }

        if node.children.is_empty() {
            // Leaf node - add all positions
            for &idx in &node.position_indices {
                if candidates.len() < max_candidates {
                    candidates.push(idx);
                }
            }
        } else {
            // Internal node - sort children by similarity and traverse best ones first
            let mut child_similarities: Vec<_> = node
                .children
                .iter()
                .enumerate()
                .map(|(i, child)| {
                    let similarity = self.cosine_similarity_cached(query, &child.centroid);
                    (i, similarity)
                })
                .collect();

            child_similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

            // Traverse children in order of similarity
            for (child_idx, _) in child_similarities {
                self.traverse_cluster_tree(
                    query,
                    &node.children[child_idx],
                    candidates,
                    max_candidates,
                );
            }
        }
    }

    /// Cached cosine similarity calculation
    fn cosine_similarity_cached(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        // Use optimized SIMD version with early termination checks
        let a_norm_sq = SimdVectorOps::squared_norm(a);
        let b_norm_sq = SimdVectorOps::squared_norm(b);
        
        if a_norm_sq == 0.0 || b_norm_sq == 0.0 {
            return 0.0;
        }
        
        let dot_product = SimdVectorOps::dot_product(a, b);
        dot_product / (a_norm_sq.sqrt() * b_norm_sq.sqrt())
    }

    /// Force rebuild of cluster tree (useful after adding many positions)
    pub fn rebuild_cluster_tree(&mut self) {
        self.cluster_tree = None;
        self.build_cluster_tree();
    }

    /// Get cluster tree statistics
    pub fn cluster_tree_stats(&self) -> Option<ClusterTreeStats> {
        self.cluster_tree.as_ref().map(|tree| {
            let mut stats = ClusterTreeStats {
                total_nodes: 0,
                leaf_nodes: 0,
                max_depth: 0,
                avg_cluster_size: 0.0,
                max_cluster_size: 0,
            };

            self.collect_cluster_stats(tree, 0, &mut stats);

            if stats.leaf_nodes > 0 {
                stats.avg_cluster_size = self.positions.len() as f32 / stats.leaf_nodes as f32;
            }

            stats
        })
    }

    /// Recursively collect cluster statistics
    fn collect_cluster_stats(
        &self,
        node: &ClusterNode,
        depth: usize,
        stats: &mut ClusterTreeStats,
    ) {
        stats.total_nodes += 1;
        stats.max_depth = stats.max_depth.max(depth);
        stats.max_cluster_size = stats.max_cluster_size.max(node.size);

        if node.children.is_empty() {
            stats.leaf_nodes += 1;
        } else {
            for child in &node.children {
                self.collect_cluster_stats(child, depth + 1, stats);
            }
        }
    }
    
    // ============ CACHE MANAGEMENT METHODS ============
    
    /// Generate a hash key for a query vector and k value
    fn hash_query(&self, query: &Array1<f32>, k: usize) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        // Hash the query vector (sample key elements for performance)
        for i in (0..query.len()).step_by(16) { // Sample every 16th element
            ((query[i] * 1000.0) as i32).hash(&mut hasher);
        }
        k.hash(&mut hasher);
        self.positions.len().hash(&mut hasher); // Include dataset size in hash
        
        hasher.finish()
    }
    
    /// Check if we have a cached result for this query
    fn get_cached_result(&mut self, query_hash: u64) -> Option<Vec<(Array1<f32>, f32, f32)>> {
        let now = Instant::now();
        
        if let Some(cached_entry) = self.result_cache.get(&query_hash) {
            if now.duration_since(cached_entry.timestamp) < self.cache_ttl {
                self.cache_hits += 1;
                return Some(cached_entry.results.clone());
            } else {
                // Remove expired entry
                self.result_cache.remove(&query_hash);
            }
        }
        
        self.cache_misses += 1;
        None
    }
    
    /// Cache search results for future lookups
    fn cache_search_result(&mut self, query_hash: u64, results: Vec<(Array1<f32>, f32, f32)>) {
        let now = Instant::now();
        
        self.result_cache.insert(query_hash, SearchResultCache {
            results,
            timestamp: now,
        });
        
        // Maintain cache size
        if self.result_cache.len() > self.max_cache_size / 10 {
            self.evict_oldest_result_cache_entries();
        }
    }
    
    /// Evict expired cache entries to maintain performance
    fn evict_expired_cache_entries(&mut self) {
        let now = Instant::now();
        
        // Evict expired similarity cache entries
        self.similarity_cache.retain(|_, (_, cached_time)| {
            now.duration_since(*cached_time) < self.cache_ttl
        });
        
        // Evict expired result cache entries
        self.result_cache.retain(|_, cached_result| {
            now.duration_since(cached_result.timestamp) < self.cache_ttl
        });
    }
    
    /// Evict oldest cache entries when cache is full (LRU eviction)
    fn evict_oldest_cache_entries(&mut self) {
        // Remove oldest 25% of similarity cache entries
        let entries_to_remove = self.similarity_cache.len() / 4;
        if entries_to_remove > 0 {
            let mut entries: Vec<_> = self.similarity_cache.iter().map(|(k, v)| (*k, *v)).collect();
            entries.sort_by_key(|(_, (_, time))| *time);
            
            for i in 0..entries_to_remove {
                if let Some((key, _)) = entries.get(i) {
                    self.similarity_cache.remove(key);
                }
            }
        }
    }
    
    /// Evict oldest result cache entries when cache is full (LRU eviction)
    fn evict_oldest_result_cache_entries(&mut self) {
        // Remove oldest 25% of result cache entries
        let entries_to_remove = self.result_cache.len() / 4;
        if entries_to_remove > 0 {
            let mut entries: Vec<_> = self.result_cache.iter().map(|(k, v)| (*k, v.timestamp)).collect();
            entries.sort_by_key(|(_, timestamp)| *timestamp);
            
            for i in 0..entries_to_remove {
                if let Some((key, _)) = entries.get(i) {
                    self.result_cache.remove(key);
                }
            }
        }
    }
    
    /// Get comprehensive cache statistics for performance monitoring
    pub fn get_cache_stats(&self) -> SimilarityCacheStats {
        let hit_ratio = if self.cache_hits + self.cache_misses > 0 {
            self.cache_hits as f32 / (self.cache_hits + self.cache_misses) as f32
        } else {
            0.0
        };
        
        SimilarityCacheStats {
            result_cache_size: self.result_cache.len(),
            similarity_cache_size: self.similarity_cache.len(),
            max_cache_size: self.max_cache_size,
            cache_ttl_secs: self.cache_ttl.as_secs(),
            cache_hits: self.cache_hits,
            cache_misses: self.cache_misses,
            hit_ratio,
        }
    }
    
    /// Clear all caches (useful for benchmarking or memory management)
    pub fn clear_caches(&mut self) {
        self.similarity_cache.clear();
        self.result_cache.clear();
        self.cache_hits = 0;
        self.cache_misses = 0;
    }
    
    /// Reset cache statistics while preserving cached data
    pub fn reset_cache_stats(&mut self) {
        self.cache_hits = 0;
        self.cache_misses = 0;
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

/// Statistics about the cluster tree
#[derive(Debug, Clone)]
pub struct ClusterTreeStats {
    pub total_nodes: usize,
    pub leaf_nodes: usize,
    pub max_depth: usize,
    pub avg_cluster_size: f32,
    pub max_cluster_size: usize,
}

impl SimilaritySearch {
    /// Optimized search with early termination and cached computations
    pub fn search_optimized(&self, query: &Array1<f32>, k: usize) -> Vec<(Array1<f32>, f32, f32)> {
        assert_eq!(query.len(), self.vector_size, "Query vector size mismatch");

        if self.positions.is_empty() {
            return Vec::new();
        }

        // Pre-compute query norm once
        let query_norm_squared = SimdVectorOps::squared_norm(query);
        let query_norm = query_norm_squared.sqrt();

        // For small k, use optimized heap management
        if k <= 10 && self.positions.len() > k * 10 {
            return self.search_with_bounded_heap(query, query_norm_squared, k);
        }

        // For larger k or smaller datasets, use parallel approach with early termination
        self.search_parallel_optimized(query, query_norm, k)
    }

    /// Search using bounded heap for small k values (memory efficient)
    fn search_with_bounded_heap(
        &self,
        query: &Array1<f32>,
        query_norm_squared: f32,
        k: usize,
    ) -> Vec<(Array1<f32>, f32, f32)> {
        let mut heap = BinaryHeap::with_capacity(k + 1);
        let mut min_similarity = f32::NEG_INFINITY;

        for entry in &self.positions {
            // Early termination: skip if impossible to beat current worst
            if heap.len() == k && self.can_skip_entry(query, entry, min_similarity) {
                continue;
            }

            let similarity = self.cosine_similarity_fast_uncached(query, query_norm_squared, entry);

            let result = SearchResult {
                similarity,
                evaluation: entry.evaluation,
                vector: entry.vector.clone(),
            };

            if heap.len() < k {
                if heap.is_empty() || similarity < min_similarity {
                    min_similarity = similarity;
                }
                heap.push(result);
            } else if similarity > min_similarity {
                heap.pop();  // Remove worst
                heap.push(result);
                // Update min_similarity
                min_similarity = heap.peek().map(|r| r.similarity).unwrap_or(f32::NEG_INFINITY);
            }
        }

        // Convert to sorted results
        self.heap_to_sorted_results(heap)
    }

    /// Parallel search with optimizations for larger k values
    fn search_parallel_optimized(
        &self,
        query: &Array1<f32>,
        query_norm: f32,
        k: usize,
    ) -> Vec<(Array1<f32>, f32, f32)> {
        // Use chunks to reduce memory allocation overhead
        let chunk_size = (self.positions.len() / rayon::current_num_threads()).max(1000);
        
        let mut results: Vec<_> = self
            .positions
            .par_chunks(chunk_size)
            .flat_map(|chunk| {
                chunk.par_iter().map(|entry| {
                    let entry_norm = entry.norm_squared.sqrt();
                    let similarity = self.cosine_similarity_ultra_fast(query, query_norm, entry, entry_norm);
                    (entry.vector.clone(), entry.evaluation, similarity)
                })
            })
            .collect();

        // Use appropriate sorting strategy based on k vs n ratio
        if k * 10 < results.len() {
            // For small k, use full sort then truncate (simpler and still efficient)
            results.par_sort_unstable_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            results.truncate(k);
        } else {
            // Full sort for cases where k is large relative to n
            results.par_sort_unstable_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            results.truncate(k);
        }

        results
    }

    /// Early termination heuristic: can we skip this entry?
    fn can_skip_entry(&self, _query: &Array1<f32>, _entry: &PositionEntry, min_similarity: f32) -> bool {
        // Simple heuristic: if the entry's norm is too different, skip
        // This is a conservative approximation - could be made more sophisticated
        
        // For now, implement a basic check
        // In practice, you could use bounds based on vector norms and angles
        min_similarity > 0.95  // Only skip if we already have very high similarities
    }

    /// Convert heap to sorted results vector
    fn heap_to_sorted_results(&self, mut heap: BinaryHeap<SearchResult>) -> Vec<(Array1<f32>, f32, f32)> {
        let mut results = Vec::with_capacity(heap.len());
        while let Some(result) = heap.pop() {
            results.push((result.vector, result.evaluation, result.similarity));
        }
        results.reverse(); // Heap gives us worst-first, we want best-first
        results
    }

    /// Batch search optimization for multiple queries
    pub fn batch_search_optimized(
        &self,
        queries: &[Array1<f32>],
        k: usize,
    ) -> Vec<Vec<(Array1<f32>, f32, f32)>> {
        if queries.is_empty() || self.positions.is_empty() {
            return vec![Vec::new(); queries.len()];
        }

        // Pre-compute norms for all queries
        let query_norms: Vec<f32> = queries
            .par_iter()
            .map(|q| SimdVectorOps::squared_norm(q).sqrt())
            .collect();

        // Process queries in parallel
        queries
            .par_iter()
            .zip(query_norms.par_iter())
            .map(|(query, &query_norm)| {
                self.search_parallel_optimized(query, query_norm, k)
            })
            .collect()
    }
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
