use ndarray::Array1;
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use rayon::prelude::*;

/// Entry in the similarity search index
#[derive(Debug, Clone)]
pub struct PositionEntry {
    pub vector: Array1<f32>,
    pub evaluation: f32,
}

/// Result from similarity search
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
        // Reverse ordering for max-heap behavior in BinaryHeap
        other.similarity.partial_cmp(&self.similarity)
    }
}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Similarity search engine for chess positions
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
        
        self.positions.push(PositionEntry {
            vector,
            evaluation,
        });
    }

    /// Search for k most similar positions (automatically chooses parallel or sequential)
    pub fn search(&self, query: &Array1<f32>, k: usize) -> Vec<(Array1<f32>, f32, f32)> {
        // Use parallel search for larger datasets
        if self.positions.len() > 100 {
            self.parallel_search(query, k)
        } else {
            self.sequential_search(query, k)
        }
    }
    
    /// Sequential search implementation (for small datasets)
    pub fn sequential_search(&self, query: &Array1<f32>, k: usize) -> Vec<(Array1<f32>, f32, f32)> {
        assert_eq!(query.len(), self.vector_size, "Query vector size mismatch");
        
        if self.positions.is_empty() {
            return Vec::new();
        }

        // Use a min-heap to keep track of top-k results
        let mut heap = BinaryHeap::new();
        
        for entry in &self.positions {
            let similarity = self.cosine_similarity(query, &entry.vector);
            
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
    
    /// Parallel search implementation (for larger datasets)
    pub fn parallel_search(&self, query: &Array1<f32>, k: usize) -> Vec<(Array1<f32>, f32, f32)> {
        assert_eq!(query.len(), self.vector_size, "Query vector size mismatch");
        
        if self.positions.is_empty() {
            return Vec::new();
        }

        // Calculate similarities in parallel
        let mut results: Vec<_> = self.positions
            .par_iter()
            .map(|entry| {
                let similarity = self.cosine_similarity(query, &entry.vector);
                (entry.vector.clone(), entry.evaluation, similarity)
            })
            .collect();

        // Sort by similarity (descending) and take top k
        results.par_sort_unstable_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));
        results.truncate(k);
        
        results
    }

    /// Brute force search (for small datasets or comparison)
    pub fn brute_force_search(&self, query: &Array1<f32>, k: usize) -> Vec<(Array1<f32>, f32, f32)> {
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

    /// Calculate cosine similarity between two vectors
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
    pub fn search_by_distance(&self, query: &Array1<f32>, k: usize) -> Vec<(Array1<f32>, f32, f32)> {
        let mut results: Vec<_> = self.positions
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