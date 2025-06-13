pub mod position_encoder;
pub mod similarity_search;
pub mod manifold_learner;
pub mod lsh;
pub mod ann;

pub use position_encoder::PositionEncoder;
pub use similarity_search::SimilaritySearch;
pub use lsh::LSH;
pub use manifold_learner::ManifoldLearner;

use chess::Board;
use ndarray::Array1;

/// Main chess vector engine
pub struct ChessVectorEngine {
    encoder: PositionEncoder,
    similarity_search: SimilaritySearch,
    lsh_index: Option<LSH>,
    use_lsh: bool,
}

impl ChessVectorEngine {
    /// Create a new chess vector engine
    pub fn new(vector_size: usize) -> Self {
        Self {
            encoder: PositionEncoder::new(vector_size),
            similarity_search: SimilaritySearch::new(vector_size),
            lsh_index: None,
            use_lsh: false,
        }
    }
    
    /// Create a new chess vector engine with LSH enabled
    pub fn new_with_lsh(vector_size: usize, num_tables: usize, hash_size: usize) -> Self {
        Self {
            encoder: PositionEncoder::new(vector_size),
            similarity_search: SimilaritySearch::new(vector_size),
            lsh_index: Some(LSH::new(vector_size, num_tables, hash_size)),
            use_lsh: true,
        }
    }
    
    /// Enable LSH indexing
    pub fn enable_lsh(&mut self, num_tables: usize, hash_size: usize) {
        self.lsh_index = Some(LSH::new(self.encoder.vector_size(), num_tables, hash_size));
        self.use_lsh = true;
        
        // Rebuild LSH index with existing positions
        if let Some(ref mut lsh) = self.lsh_index {
            for (vector, evaluation) in self.similarity_search.get_all_positions() {
                lsh.add_vector(vector, evaluation);
            }
        }
    }

    /// Add a position with its evaluation to the knowledge base
    pub fn add_position(&mut self, board: &Board, evaluation: f32) {
        let vector = self.encoder.encode(board);
        self.similarity_search.add_position(vector.clone(), evaluation);
        
        // Also add to LSH index if enabled
        if let Some(ref mut lsh) = self.lsh_index {
            lsh.add_vector(vector, evaluation);
        }
    }

    /// Find similar positions to the given board
    pub fn find_similar_positions(&self, board: &Board, k: usize) -> Vec<(Array1<f32>, f32, f32)> {
        let query_vector = self.encoder.encode(board);
        
        if self.use_lsh && self.lsh_index.is_some() {
            self.lsh_index.as_ref().unwrap().query(&query_vector, k)
        } else {
            self.similarity_search.search(&query_vector, k)
        }
    }

    /// Get evaluation for a position based on similar positions
    pub fn evaluate_position(&self, board: &Board) -> Option<f32> {
        let similar_positions = self.find_similar_positions(board, 5);
        
        if similar_positions.is_empty() {
            return None;
        }

        // Weighted average based on similarity
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        for (_, evaluation, similarity) in similar_positions {
            // Use similarity as weight (closer positions have more influence)
            let weight = similarity;
            weighted_sum += evaluation * weight;
            weight_sum += weight;
        }

        if weight_sum > 0.0 {
            Some(weighted_sum / weight_sum)
        } else {
            None
        }
    }

    /// Encode a position to vector (public interface)
    pub fn encode_position(&self, board: &Board) -> Array1<f32> {
        self.encoder.encode(board)
    }

    /// Calculate similarity between two boards
    pub fn calculate_similarity(&self, board1: &Board, board2: &Board) -> f32 {
        let vec1 = self.encoder.encode(board1);
        let vec2 = self.encoder.encode(board2);
        self.encoder.similarity(&vec1, &vec2)
    }

    /// Get the size of the knowledge base
    pub fn knowledge_base_size(&self) -> usize {
        self.similarity_search.size()
    }
    
    /// Check if LSH is enabled
    pub fn is_lsh_enabled(&self) -> bool {
        self.use_lsh
    }
    
    /// Get LSH statistics if enabled
    pub fn lsh_stats(&self) -> Option<crate::lsh::LSHStats> {
        self.lsh_index.as_ref().map(|lsh| lsh.stats())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess::Board;

    #[test]
    fn test_engine_creation() {
        let engine = ChessVectorEngine::new(1024);
        assert_eq!(engine.knowledge_base_size(), 0);
    }

    #[test]
    fn test_add_and_search() {
        let mut engine = ChessVectorEngine::new(1024);
        let board = Board::default();
        
        engine.add_position(&board, 0.0);
        assert_eq!(engine.knowledge_base_size(), 1);
        
        let similar = engine.find_similar_positions(&board, 1);
        assert_eq!(similar.len(), 1);
    }

    #[test]
    fn test_evaluation() {
        let mut engine = ChessVectorEngine::new(1024);
        let board = Board::default();
        
        // Add some positions with evaluations
        engine.add_position(&board, 0.5);
        
        let evaluation = engine.evaluate_position(&board);
        assert!(evaluation.is_some());
        assert!((evaluation.unwrap() - 0.5).abs() < 1e-6);
    }
}