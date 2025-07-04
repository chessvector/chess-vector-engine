use ndarray::Array1;
use std::cmp::Ordering;

/// Approximate Nearest Neighbor search using multiple strategies
pub struct ANNIndex {
    /// All stored vectors
    vectors: Vec<Array1<f32>>,
    /// Associated data (evaluations)
    data: Vec<f32>,
    /// LSH index for fast approximate search
    lsh: Option<crate::lsh::LSH>,
    /// Use random projections for dimensionality reduction
    use_random_projections: bool,
    /// Random projection matrix (if enabled)
    projection_matrix: Option<Array2<f32>>,
    /// Projected dimension
    projected_dim: usize,
    /// Original vector dimension
    vector_dim: usize,
}

/// Search result with similarity score
#[derive(Debug, Clone)]
pub struct ANNResult {
    pub vector: Array1<f32>,
    pub data: f32,
    pub similarity: f32,
}

impl PartialEq for ANNResult {
    fn eq(&self, other: &Self) -> bool {
        self.similarity == other.similarity
    }
}

impl Eq for ANNResult {}

impl PartialOrd for ANNResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ANNResult {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for max-heap
        other
            .similarity
            .partial_cmp(&self.similarity)
            .unwrap_or(Ordering::Equal)
    }
}

impl ANNIndex {
    /// Create a new ANN index
    pub fn new(vector_dim: usize) -> Self {
        Self {
            vectors: Vec::new(),
            data: Vec::new(),
            lsh: None,
            use_random_projections: false,
            projection_matrix: None,
            projected_dim: vector_dim / 4, // Default to 1/4 of original dimension
            vector_dim,
        }
    }

    /// Enable LSH indexing
    pub fn with_lsh(mut self, num_tables: usize, hash_size: usize) -> Self {
        self.lsh = Some(crate::lsh::LSH::new(self.vector_dim, num_tables, hash_size));
        self
    }

    /// Enable random projections for dimensionality reduction
    pub fn with_random_projections(mut self, projected_dim: usize) -> Self {
        self.use_random_projections = true;
        self.projected_dim = projected_dim;
        self
    }

    /// Add a vector to the index
    pub fn add_vector(&mut self, vector: Array1<f32>, data: f32) {
        // Initialize random projection matrix if needed
        if self.use_random_projections && self.projection_matrix.is_none() {
            self.init_random_projections(vector.len());
        }

        self.vectors.push(vector.clone());
        self.data.push(data);

        // Add to LSH if enabled
        if let Some(ref mut lsh) = self.lsh {
            lsh.add_vector(vector, data);
        }
    }

    /// Search for approximate nearest neighbors
    pub fn search(
        &self,
        query: &Array1<f32>,
        k: usize,
        strategy: SearchStrategy,
    ) -> Vec<ANNResult> {
        match strategy {
            SearchStrategy::LSH => self.search_lsh(query, k),
            SearchStrategy::RandomProjection => self.search_random_projection(query, k),
            SearchStrategy::Hybrid => self.search_hybrid(query, k),
            SearchStrategy::Exact => self.search_exact(query, k),
        }
    }

    /// LSH-based search
    fn search_lsh(&self, query: &Array1<f32>, k: usize) -> Vec<ANNResult> {
        if let Some(ref lsh) = self.lsh {
            lsh.query(query, k)
                .into_iter()
                .map(|(vec, data, sim)| ANNResult {
                    vector: vec,
                    data,
                    similarity: sim,
                })
                .collect()
        } else {
            self.search_exact(query, k)
        }
    }

    /// Random projection-based search
    fn search_random_projection(&self, query: &Array1<f32>, k: usize) -> Vec<ANNResult> {
        if !self.use_random_projections || self.projection_matrix.is_none() {
            return self.search_exact(query, k);
        }

        let proj_matrix = self.projection_matrix.as_ref().unwrap();
        let proj_query = self.project_vector(query, proj_matrix);

        let mut results: Vec<_> = self
            .vectors
            .iter()
            .zip(self.data.iter())
            .map(|(vec, &data)| {
                let proj_vec = self.project_vector(vec, proj_matrix);
                let similarity = cosine_similarity(&proj_query, &proj_vec);
                ANNResult {
                    vector: vec.clone(),
                    data,
                    similarity,
                }
            })
            .collect();

        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        results.truncate(k);
        results
    }

    /// Hybrid search combining multiple strategies
    fn search_hybrid(&self, query: &Array1<f32>, k: usize) -> Vec<ANNResult> {
        let mut candidate_indices = std::collections::HashSet::new();
        let mut results = Vec::new();

        // Get candidates from LSH
        if let Some(ref lsh) = self.lsh {
            let lsh_results = lsh.query(query, k * 2);
            for (vec, _data, _) in lsh_results {
                // Find the index of this vector in our stored vectors
                for (idx, stored_vec) in self.vectors.iter().enumerate() {
                    if vectors_approximately_equal(&vec, stored_vec) {
                        candidate_indices.insert(idx);
                        break;
                    }
                }
            }
        }

        // Get candidates from random projection
        if self.use_random_projections {
            let rp_results = self.search_random_projection(query, k * 2);
            for result in rp_results {
                // Find the index of this vector
                for (idx, stored_vec) in self.vectors.iter().enumerate() {
                    if vectors_approximately_equal(&result.vector, stored_vec) {
                        candidate_indices.insert(idx);
                        break;
                    }
                }
            }
        }

        // If we don't have enough candidates, add some random ones
        if candidate_indices.len() < k * 3 {
            for idx in 0..(k * 3).min(self.vectors.len()) {
                candidate_indices.insert(idx);
            }
        }

        // Re-rank candidates using exact similarity
        for &idx in &candidate_indices {
            let vec = &self.vectors[idx];
            let data = self.data[idx];
            let similarity = cosine_similarity(query, vec);
            results.push(ANNResult {
                vector: vec.clone(),
                data,
                similarity,
            });
        }

        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        results.truncate(k);
        results
    }

    /// Exact search (brute force)
    fn search_exact(&self, query: &Array1<f32>, k: usize) -> Vec<ANNResult> {
        let mut results: Vec<_> = self
            .vectors
            .iter()
            .zip(self.data.iter())
            .map(|(vec, &data)| {
                let similarity = cosine_similarity(query, vec);
                ANNResult {
                    vector: vec.clone(),
                    data,
                    similarity,
                }
            })
            .collect();

        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        results.truncate(k);
        results
    }

    /// Initialize random projection matrix
    fn init_random_projections(&mut self, input_dim: usize) {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Use the provided input_dim (should match self.vector_dim)
        assert_eq!(
            input_dim, self.vector_dim,
            "Input dimension should match vector dimension"
        );

        let mut matrix_data = Vec::with_capacity(self.projected_dim * input_dim);
        for _ in 0..(self.projected_dim * input_dim) {
            matrix_data.push(rng.gen_range(-1.0..1.0));
        }

        self.projection_matrix = Some(
            Array2::from_shape_vec((self.projected_dim, input_dim), matrix_data)
                .expect("Failed to create projection matrix"),
        );
    }

    /// Project a vector to lower dimension
    fn project_vector(&self, vector: &Array1<f32>, proj_matrix: &Array2<f32>) -> Array1<f32> {
        let mut result = Array1::zeros(self.projected_dim);
        for i in 0..self.projected_dim {
            let dot_product: f32 = vector
                .iter()
                .zip(proj_matrix.row(i).iter())
                .map(|(v, p)| v * p)
                .sum();
            result[i] = dot_product;
        }
        result
    }

    /// Get statistics about the index
    pub fn stats(&self) -> ANNStats {
        ANNStats {
            num_vectors: self.vectors.len(),
            vector_dim: if self.vectors.is_empty() {
                0
            } else {
                self.vectors[0].len()
            },
            has_lsh: self.lsh.is_some(),
            has_random_projections: self.use_random_projections,
            projected_dim: if self.use_random_projections {
                Some(self.projected_dim)
            } else {
                None
            },
        }
    }
}

/// Search strategies for ANN
#[derive(Debug, Clone, Copy)]
pub enum SearchStrategy {
    /// Use LSH for approximate search
    LSH,
    /// Use random projections
    RandomProjection,
    /// Combine multiple strategies
    Hybrid,
    /// Exact search (for comparison)
    Exact,
}

/// Statistics about the ANN index
#[derive(Debug)]
pub struct ANNStats {
    pub num_vectors: usize,
    pub vector_dim: usize,
    pub has_lsh: bool,
    pub has_random_projections: bool,
    pub projected_dim: Option<usize>,
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

/// Check if two vectors are approximately equal
fn vectors_approximately_equal(a: &Array1<f32>, b: &Array1<f32>) -> bool {
    if a.len() != b.len() {
        return false;
    }

    let threshold = 1e-6;
    for (x, y) in a.iter().zip(b.iter()) {
        if (x - y).abs() > threshold {
            return false;
        }
    }
    true
}

use ndarray::Array2;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_ann_index_creation() {
        let index = ANNIndex::new(128);
        assert_eq!(index.vectors.len(), 0);
        assert!(!index.use_random_projections);
        assert!(index.lsh.is_none());
    }

    #[test]
    fn test_ann_with_lsh() {
        let index = ANNIndex::new(128).with_lsh(4, 8);
        assert!(index.lsh.is_some());
    }

    #[test]
    fn test_ann_with_random_projections() {
        let index = ANNIndex::new(128).with_random_projections(32);
        assert!(index.use_random_projections);
        assert_eq!(index.projected_dim, 32);
    }

    #[test]
    fn test_add_and_search() {
        let mut index = ANNIndex::new(4);

        let vec1 = Array1::from(vec![1.0, 0.0, 0.0, 0.0]);
        let vec2 = Array1::from(vec![0.0, 1.0, 0.0, 0.0]);
        let vec3 = Array1::from(vec![1.0, 0.1, 0.0, 0.0]);

        index.add_vector(vec1.clone(), 1.0);
        index.add_vector(vec2, 2.0);
        index.add_vector(vec3, 1.1);

        let results = index.search(&vec1, 2, SearchStrategy::Exact);
        assert_eq!(results.len(), 2);
        assert!(results[0].similarity > 0.9); // Should find itself first
    }

    #[test]
    fn test_search_strategies() {
        let mut index = ANNIndex::new(4).with_lsh(2, 4).with_random_projections(2);

        let vec1 = Array1::from(vec![1.0, 0.0, 0.0, 0.0]);
        index.add_vector(vec1.clone(), 1.0);

        // Test all search strategies
        let exact = index.search(&vec1, 1, SearchStrategy::Exact);
        let lsh = index.search(&vec1, 1, SearchStrategy::LSH);
        let rp = index.search(&vec1, 1, SearchStrategy::RandomProjection);
        let hybrid = index.search(&vec1, 1, SearchStrategy::Hybrid);

        assert!(!exact.is_empty());
        assert!(!lsh.is_empty());
        assert!(!rp.is_empty());
        assert!(!hybrid.is_empty());
    }
}
