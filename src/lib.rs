pub mod position_encoder;
pub mod similarity_search;
pub mod manifold_learner;
pub mod lsh;
pub mod ann;
pub mod training;
pub mod opening_book;

pub use position_encoder::PositionEncoder;
pub use similarity_search::SimilaritySearch;
pub use lsh::LSH;
pub use manifold_learner::ManifoldLearner;
pub use opening_book::{OpeningBook, OpeningEntry, OpeningBookStats};
pub use training::{TrainingData, TrainingDataset, GameExtractor, EngineEvaluator, TacticalPuzzle, TacticalTrainingData, TacticalPuzzleParser};

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
    manifold_learner: Option<ManifoldLearner>,
    use_lsh: bool,
    use_manifold: bool,
    /// Map from position index to moves played and their outcomes
    position_moves: HashMap<usize, Vec<(ChessMove, f32)>>,
    /// Compressed similarity search for manifold vectors
    manifold_similarity_search: Option<SimilaritySearch>,
    /// LSH index for compressed vectors
    manifold_lsh_index: Option<LSH>,
    /// Store position vectors for reverse lookup
    position_vectors: Vec<Array1<f32>>,
    /// Store boards for move generation
    position_boards: Vec<Board>,
    /// Store evaluations for each position
    position_evaluations: Vec<f32>,
    /// Opening book for position evaluation and move suggestions
    opening_book: Option<OpeningBook>,
}

impl ChessVectorEngine {
    /// Create a new chess vector engine
    pub fn new(vector_size: usize) -> Self {
        Self {
            encoder: PositionEncoder::new(vector_size),
            similarity_search: SimilaritySearch::new(vector_size),
            lsh_index: None,
            manifold_learner: None,
            use_lsh: false,
            use_manifold: false,
            position_moves: HashMap::new(),
            manifold_similarity_search: None,
            manifold_lsh_index: None,
            position_vectors: Vec::new(),
            position_boards: Vec::new(),
            position_evaluations: Vec::new(),
            opening_book: None,
        }
    }

    /// Create a new chess vector engine with intelligent architecture selection
    /// based on expected dataset size and use case
    pub fn new_adaptive(vector_size: usize, expected_positions: usize, use_case: &str) -> Self {
        match use_case {
            "training" => {
                if expected_positions > 10000 {
                    // Large training datasets benefit from LSH for loading speed
                    Self::new_with_lsh(vector_size, 12, 20)
                } else {
                    Self::new(vector_size)
                }
            }
            "gameplay" => {
                if expected_positions > 15000 {
                    // Gameplay needs balance of speed and accuracy
                    Self::new_with_lsh(vector_size, 10, 18)
                } else {
                    Self::new(vector_size)
                }
            }
            "analysis" => {
                if expected_positions > 10000 {
                    // Analysis prioritizes recall over speed
                    Self::new_with_lsh(vector_size, 14, 22)
                } else {
                    Self::new(vector_size)
                }
            }
            _ => Self::new(vector_size) // Default to linear search
        }
    }
    
    /// Create a new chess vector engine with LSH enabled
    pub fn new_with_lsh(vector_size: usize, num_tables: usize, hash_size: usize) -> Self {
        Self {
            encoder: PositionEncoder::new(vector_size),
            similarity_search: SimilaritySearch::new(vector_size),
            lsh_index: Some(LSH::new(vector_size, num_tables, hash_size)),
            manifold_learner: None,
            use_lsh: true,
            use_manifold: false,
            position_moves: HashMap::new(),
            manifold_similarity_search: None,
            manifold_lsh_index: None,
            position_vectors: Vec::new(),
            position_boards: Vec::new(),
            position_evaluations: Vec::new(),
            opening_book: None,
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
        
        // Store vector, board, and evaluation for reverse lookup
        self.position_vectors.push(vector.clone());
        self.position_boards.push(*board);
        self.position_evaluations.push(evaluation);
        
        // Also add to LSH index if enabled
        if let Some(ref mut lsh) = self.lsh_index {
            lsh.add_vector(vector.clone(), evaluation);
        }
        
        // Add to manifold indices if trained
        if self.use_manifold {
            if let Some(ref learner) = self.manifold_learner {
                let compressed = learner.encode(&vector);
                
                if let Some(ref mut search) = self.manifold_similarity_search {
                    search.add_position(compressed.clone(), evaluation);
                }
                
                if let Some(ref mut lsh) = self.manifold_lsh_index {
                    lsh.add_vector(compressed, evaluation);
                }
            }
        }
    }

    /// Find similar positions to the given board
    pub fn find_similar_positions(&self, board: &Board, k: usize) -> Vec<(Array1<f32>, f32, f32)> {
        let query_vector = self.encoder.encode(board);
        
        // Use manifold space if available and trained
        if self.use_manifold && self.manifold_learner.is_some() {
            let compressed_query = self.manifold_learner.as_ref().unwrap().encode(&query_vector);
            
            // Use LSH in manifold space if available
            if let Some(ref lsh) = self.manifold_lsh_index {
                return lsh.query(&compressed_query, k);
            }
            
            // Fall back to linear search in manifold space
            if let Some(ref search) = self.manifold_similarity_search {
                return search.search(&compressed_query, k);
            }
        }
        
        // Use original space with LSH if enabled
        if self.use_lsh && self.lsh_index.is_some() {
            self.lsh_index.as_ref().unwrap().query(&query_vector, k)
        } else {
            self.similarity_search.search(&query_vector, k)
        }
    }
    
    /// Find similar positions with indices for move recommendation
    pub fn find_similar_positions_with_indices(&self, board: &Board, k: usize) -> Vec<(usize, f32, f32)> {
        let query_vector = self.encoder.encode(board);
        
        // For now, use linear search to get accurate position indices
        // In the future, we could enhance LSH to return indices
        let mut results = Vec::new();
        
        for (i, stored_vector) in self.position_vectors.iter().enumerate() {
            let similarity = self.encoder.similarity(&query_vector, stored_vector);
            let eval = self.position_evaluations.get(i).copied().unwrap_or(0.0);
            results.push((i, eval, similarity));
        }
        
        // Sort by similarity (descending)
        results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        
        results
    }

    /// Get evaluation for a position based on similar positions and opening book
    pub fn evaluate_position(&self, board: &Board) -> Option<f32> {
        // First check opening book
        if let Some(entry) = self.get_opening_entry(board) {
            return Some(entry.evaluation);
        }
        
        // Fall back to similarity search
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
    
    /// Enable manifold learning with specified compression ratio
    pub fn enable_manifold_learning(&mut self, compression_ratio: f32) -> Result<(), String> {
        let input_dim = self.encoder.vector_size();
        let output_dim = ((input_dim as f32) / compression_ratio) as usize;
        
        if output_dim == 0 {
            return Err("Compression ratio too high, output dimension would be 0".to_string());
        }
        
        let mut learner = ManifoldLearner::new(input_dim, output_dim);
        learner.init_network()?;
        
        self.manifold_learner = Some(learner);
        self.manifold_similarity_search = Some(SimilaritySearch::new(output_dim));
        self.use_manifold = false; // Don't use until trained
        
        Ok(())
    }
    
    /// Train manifold learning on existing positions
    pub fn train_manifold_learning(&mut self, epochs: usize) -> Result<(), String> {
        if self.manifold_learner.is_none() {
            return Err("Manifold learning not enabled. Call enable_manifold_learning first.".to_string());
        }
        
        if self.similarity_search.size() == 0 {
            return Err("No positions in knowledge base to train on.".to_string());
        }
        
        // Collect all position vectors
        let all_positions = self.similarity_search.get_all_positions();
        let mut training_data = Vec::new();
        
        for (vector, _eval) in all_positions {
            training_data.extend_from_slice(vector.as_slice().unwrap());
        }
        
        let rows = self.similarity_search.size();
        let cols = self.encoder.vector_size();
        let training_matrix = ndarray::Array2::from_shape_vec((rows, cols), training_data)
            .map_err(|e| format!("Failed to create training matrix: {}", e))?;
        
        // Train the manifold learner
        if let Some(ref mut learner) = self.manifold_learner {
            learner.train(&training_matrix, epochs)?;
            let compression_ratio = learner.compression_ratio();
            
            // Release the mutable borrow before calling rebuild_manifold_indices
            let _ = learner;
            
            // Rebuild compressed indices
            self.rebuild_manifold_indices()?;
            self.use_manifold = true;
            
            println!("Manifold learning training completed. Compression ratio: {:.1}x", 
                     compression_ratio);
        }
        
        Ok(())
    }
    
    /// Rebuild manifold-based indices after training
    fn rebuild_manifold_indices(&mut self) -> Result<(), String> {
        if let Some(ref learner) = self.manifold_learner {
            let all_positions = self.similarity_search.get_all_positions();
            
            // Clear existing manifold indices
            let output_dim = learner.output_dim();
            if let Some(ref mut search) = self.manifold_similarity_search {
                *search = SimilaritySearch::new(output_dim);
            }
            if let Some(ref mut lsh) = self.manifold_lsh_index {
                *lsh = LSH::new(output_dim, 8, 16); // Default LSH params for compressed space
            }
            
            // Add compressed vectors to indices
            for (vector, eval) in all_positions {
                let compressed = learner.encode(&vector);
                
                if let Some(ref mut search) = self.manifold_similarity_search {
                    search.add_position(compressed.clone(), eval);
                }
                
                if let Some(ref mut lsh) = self.manifold_lsh_index {
                    lsh.add_vector(compressed, eval);
                }
            }
        }
        
        Ok(())
    }
    
    /// Enable LSH for manifold space
    pub fn enable_manifold_lsh(&mut self, num_tables: usize, hash_size: usize) -> Result<(), String> {
        if self.manifold_learner.is_none() {
            return Err("Manifold learning not enabled".to_string());
        }
        
        let output_dim = self.manifold_learner.as_ref().unwrap().output_dim();
        self.manifold_lsh_index = Some(LSH::new(output_dim, num_tables, hash_size));
        
        // Rebuild index if we have trained data
        if self.use_manifold {
            self.rebuild_manifold_indices()?;
        }
        
        Ok(())
    }
    
    /// Check if manifold learning is enabled and trained
    pub fn is_manifold_enabled(&self) -> bool {
        self.use_manifold && self.manifold_learner.is_some()
    }
    
    /// Get manifold learning compression ratio
    pub fn manifold_compression_ratio(&self) -> Option<f32> {
        self.manifold_learner.as_ref().map(|l| l.compression_ratio())
    }
    
    /// Enable opening book with standard openings
    pub fn enable_opening_book(&mut self) {
        self.opening_book = Some(OpeningBook::with_standard_openings());
    }
    
    /// Set custom opening book
    pub fn set_opening_book(&mut self, book: OpeningBook) {
        self.opening_book = Some(book);
    }
    
    /// Check if position is in opening book
    pub fn is_opening_position(&self, board: &Board) -> bool {
        self.opening_book.as_ref()
            .map(|book| book.contains(board))
            .unwrap_or(false)
    }
    
    /// Get opening book entry for position
    pub fn get_opening_entry(&self, board: &Board) -> Option<&OpeningEntry> {
        self.opening_book.as_ref()?.lookup(board)
    }
    
    /// Get opening book statistics
    pub fn opening_book_stats(&self) -> Option<OpeningBookStats> {
        self.opening_book.as_ref().map(|book| book.stats())
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
    
    /// Get move recommendations based on similar positions and opening book
    pub fn recommend_moves(&self, board: &Board, num_recommendations: usize) -> Vec<MoveRecommendation> {
        // First check opening book
        if let Some(entry) = self.get_opening_entry(board) {
            let mut recommendations = Vec::new();
            
            for (chess_move, strength) in &entry.best_moves {
                recommendations.push(MoveRecommendation {
                    chess_move: *chess_move,
                    confidence: strength * 0.9, // High confidence for opening book moves
                    from_similar_position_count: 1,
                    average_outcome: entry.evaluation,
                });
            }
            
            // Sort by confidence and limit results
            recommendations.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
            recommendations.truncate(num_recommendations);
            return recommendations;
        }
        
        // Fall back to similarity search
        let similar_positions = self.find_similar_positions_with_indices(board, 20);
        
        if similar_positions.is_empty() {
            return Vec::new();
        }
        
        // Collect moves from similar positions
        let mut move_data: HashMap<ChessMove, Vec<(f32, f32)>> = HashMap::new(); // move -> (similarity, outcome)
        
        // Use actual position indices to get moves and outcomes
        for (position_index, _eval, similarity) in similar_positions {
            if let Some(moves) = self.position_moves.get(&position_index) {
                for &(chess_move, outcome) in moves {
                    move_data.entry(chess_move)
                        .or_insert_with(Vec::new)
                        .push((similarity, outcome));
                }
            }
        }
        
        // If no moves found from stored data, generate legal moves and give them neutral recommendations
        if move_data.is_empty() {
            use chess::MoveGen;
            let legal_moves: Vec<ChessMove> = MoveGen::new_legal(board).collect();
            
            for chess_move in legal_moves.into_iter().take(num_recommendations) {
                move_data.insert(chess_move, vec![(0.5, 0.0)]); // Neutral similarity and outcome
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
            let confidence = avg_similarity * (outcomes.len() as f32).ln().max(1.0) / 10.0; // Logarithmic scaling
            
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