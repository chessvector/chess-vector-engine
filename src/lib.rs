pub mod position_encoder;
pub mod similarity_search;
pub mod manifold_learner;
pub mod lsh;
pub mod ann;
pub mod training;

pub use position_encoder::PositionEncoder;
pub use similarity_search::SimilaritySearch;
pub use lsh::LSH;
pub use manifold_learner::ManifoldLearner;
pub use training::*;

use chess::{Board, ChessMove};
use ndarray::Array1;
use std::collections::HashMap;

/// Move recommendation data
#[derive(Debug, Clone)]
pub struct MoveRecommendation {
    pub chess_move: ChessMove,
    pub confidence: f32,
    pub from_similar_position_count: usize,
    pub average_outcome: f32,
}

/// Main chess vector engine
pub struct ChessVectorEngine {
    encoder: PositionEncoder,
    similarity_search: SimilaritySearch,
    lsh_index: Option<LSH>,
    use_lsh: bool,
    /// Map from position index to moves played and their outcomes
    position_moves: HashMap<usize, Vec<(ChessMove, f32)>>,
}

impl ChessVectorEngine {
    /// Create a new chess vector engine
    pub fn new(vector_size: usize) -> Self {
        Self {
            encoder: PositionEncoder::new(vector_size),
            similarity_search: SimilaritySearch::new(vector_size),
            lsh_index: None,
            use_lsh: false,
            position_moves: HashMap::new(),
        }
    }
    
    /// Create a new chess vector engine with LSH enabled
    pub fn new_with_lsh(vector_size: usize, num_tables: usize, hash_size: usize) -> Self {
        Self {
            encoder: PositionEncoder::new(vector_size),
            similarity_search: SimilaritySearch::new(vector_size),
            lsh_index: Some(LSH::new(vector_size, num_tables, hash_size)),
            use_lsh: true,
            position_moves: HashMap::new(),
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
    
    /// Add a move played from a position with its outcome
    pub fn add_position_with_move(&mut self, board: &Board, evaluation: f32, chess_move: Option<ChessMove>, move_outcome: Option<f32>) {
        let position_index = self.knowledge_base_size();
        
        // Add the position first
        self.add_position(board, evaluation);
        
        // If a move and outcome are provided, store the move information
        if let (Some(mov), Some(outcome)) = (chess_move, move_outcome) {
            self.position_moves.entry(position_index)
                .or_insert_with(Vec::new)
                .push((mov, outcome));
        }
    }
    
    /// Get move recommendations based on similar positions
    pub fn recommend_moves(&self, board: &Board, num_recommendations: usize) -> Vec<MoveRecommendation> {
        // Find similar positions
        let similar_positions = self.find_similar_positions(board, 10);
        
        if similar_positions.is_empty() {
            return Vec::new();
        }
        
        // Collect moves from similar positions
        let mut move_data: HashMap<ChessMove, Vec<(f32, f32)>> = HashMap::new(); // move -> (similarity, outcome)
        
        // We need to map back from vector similarity results to position indices
        // For now, we'll need to modify the approach since we can't easily map back to indices
        // Let's use a different strategy - we'll need to enhance our similarity search to return indices
        
        // For this initial implementation, let's use a simpler approach where we check all positions
        // and use similarity to weight the recommendations
        for (_position_index, moves) in &self.position_moves {
            // We'd need to get the board for this position to calculate similarity
            // For now, let's use a placeholder approach
            for &(chess_move, outcome) in moves {
                // Use a default similarity - in a full implementation we'd calculate actual similarity
                let similarity = 0.5; // Placeholder
                move_data.entry(chess_move)
                    .or_insert_with(Vec::new)
                    .push((similarity, outcome));
            }
        }
        
        // Calculate move recommendations
        let mut recommendations = Vec::new();
        
        for (chess_move, outcomes) in move_data {
            if outcomes.is_empty() { continue; }
            
            // Calculate weighted average outcome based on similarity
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;
            
            for &(similarity, outcome) in &outcomes {
                weighted_sum += similarity * outcome;
                weight_sum += similarity;
            }
            
            let average_outcome = if weight_sum > 0.0 {
                weighted_sum / weight_sum
            } else {
                0.0
            };
            
            // Confidence based on number of similar positions and average similarity
            let avg_similarity = outcomes.iter().map(|(s, _)| s).sum::<f32>() / outcomes.len() as f32;
            let confidence = avg_similarity * (outcomes.len() as f32).ln() / 10.0; // Logarithmic scaling
            
            recommendations.push(MoveRecommendation {
                chess_move,
                confidence: confidence.min(1.0), // Cap at 1.0
                from_similar_position_count: outcomes.len(),
                average_outcome,
            });
        }
        
        // Sort by confidence (descending)
        recommendations.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        
        // Return top recommendations
        recommendations.truncate(num_recommendations);
        recommendations
    }
    
    /// Generate legal move recommendations (filters recommendations by legal moves)
    pub fn recommend_legal_moves(&self, board: &Board, num_recommendations: usize) -> Vec<MoveRecommendation> {
        use chess::MoveGen;
        
        // Get all legal moves
        let legal_moves: std::collections::HashSet<ChessMove> = MoveGen::new_legal(board).collect();
        
        // Get recommendations and filter by legal moves
        let all_recommendations = self.recommend_moves(board, num_recommendations * 2); // Get more to account for filtering
        
        all_recommendations.into_iter()
            .filter(|rec| legal_moves.contains(&rec.chess_move))
            .take(num_recommendations)
            .collect()
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