use chess::{Board, Color, Piece, Square};
use ndarray::Array1;

/// Basic position encoder that converts chess positions to vectors
pub struct PositionEncoder {
    /// Dimension of the output vector
    vector_size: usize,
}

impl PositionEncoder {
    pub fn new(vector_size: usize) -> Self {
        Self { vector_size }
    }
    
    /// Get the vector size
    pub fn vector_size(&self) -> usize {
        self.vector_size
    }

    /// Encode a chess position into a vector
    pub fn encode(&self, board: &Board) -> Array1<f32> {
        let mut features = Vec::with_capacity(self.vector_size);
        
        // Basic encoding strategy:
        // 1. Piece positions (64 squares * 12 piece types = 768 features)
        // 2. Game state features (castling, en passant, etc.)
        // 3. Material balance
        // 4. Positional features
        
        // 1. Piece position encoding
        self.encode_piece_positions(board, &mut features);
        
        // 2. Game state
        self.encode_game_state(board, &mut features);
        
        // 3. Material balance
        self.encode_material_balance(board, &mut features);
        
        // 4. Basic positional features
        self.encode_positional_features(board, &mut features);
        
        // Pad or truncate to desired size
        features.resize(self.vector_size, 0.0);
        
        Array1::from(features)
    }

    /// Encode piece positions on the board using dense representation
    fn encode_piece_positions(&self, board: &Board, features: &mut Vec<f32>) {
        // Dense encoding: 64 squares * 1 value each = 64 features
        // Values: -6 to -1 for black pieces, 0 for empty, +1 to +6 for white pieces
        for square in chess::ALL_SQUARES {
            let value = match board.piece_on(square) {
                Some(piece) => {
                    let color = board.color_on(square).unwrap();
                    let piece_value = match piece {
                        chess::Piece::Pawn => 1.0,
                        chess::Piece::Knight => 2.0,
                        chess::Piece::Bishop => 3.0,
                        chess::Piece::Rook => 4.0,
                        chess::Piece::Queen => 5.0,
                        chess::Piece::King => 6.0,
                    };
                    if color == chess::Color::White { piece_value } else { -piece_value }
                }
                None => 0.0,
            };
            features.push(value);
        }
        
        // Add piece interaction features - attacks/defends relationships
        self.encode_piece_interactions(board, features);
    }

    /// Encode piece interactions (attacks, defends)
    fn encode_piece_interactions(&self, board: &Board, features: &mut Vec<f32>) {
        // Count attacks by piece type for each color
        let mut white_attacks = vec![0.0; 6]; // pawn, knight, bishop, rook, queen, king
        let mut black_attacks = vec![0.0; 6];
        
        // Simplified attack counting - in practice would use chess engine's attack detection
        for square in chess::ALL_SQUARES {
            if let Some(piece) = board.piece_on(square) {
                let color = board.color_on(square).unwrap();
                let piece_idx = match piece {
                    Piece::Pawn => 0,
                    Piece::Knight => 1,
                    Piece::Bishop => 2,
                    Piece::Rook => 3,
                    Piece::Queen => 4,
                    Piece::King => 5,
                };
                
                // Simple attack count based on piece mobility
                let attack_value = match piece {
                    Piece::Pawn => 1.0,
                    Piece::Knight => 3.0,
                    Piece::Bishop => 3.0,
                    Piece::Rook => 5.0,
                    Piece::Queen => 9.0,
                    Piece::King => 1.0,
                };
                
                if color == Color::White {
                    white_attacks[piece_idx] += attack_value;
                } else {
                    black_attacks[piece_idx] += attack_value;
                }
            }
        }
        
        // Add attack features (12 more features)
        features.extend(white_attacks);
        features.extend(black_attacks);
    }

    /// Encode game state (castling rights, en passant, etc.)
    fn encode_game_state(&self, board: &Board, features: &mut Vec<f32>) {
        // Castling rights (4 features)
        features.push(if board.castle_rights(Color::White).has_kingside() { 1.0 } else { 0.0 });
        features.push(if board.castle_rights(Color::White).has_queenside() { 1.0 } else { 0.0 });
        features.push(if board.castle_rights(Color::Black).has_kingside() { 1.0 } else { 0.0 });
        features.push(if board.castle_rights(Color::Black).has_queenside() { 1.0 } else { 0.0 });
        
        // En passant
        features.push(if board.en_passant().is_some() { 1.0 } else { 0.0 });
        
        // Side to move
        features.push(if board.side_to_move() == Color::White { 1.0 } else { 0.0 });
        
        // Halfmove clock (simplified - just use 0 for now)
        features.push(0.0);
    }

    /// Encode material balance
    fn encode_material_balance(&self, board: &Board, features: &mut Vec<f32>) {
        let piece_values = [
            (Piece::Pawn, 1),
            (Piece::Knight, 3),
            (Piece::Bishop, 3),
            (Piece::Rook, 5),
            (Piece::Queen, 9),
            (Piece::King, 0),
        ];

        for (piece, _value) in piece_values {
            let white_count = board.pieces(piece) & board.color_combined(Color::White);
            let black_count = board.pieces(piece) & board.color_combined(Color::Black);
            
            features.push(white_count.popcnt() as f32);
            features.push(black_count.popcnt() as f32);
            features.push((white_count.popcnt() as i32 - black_count.popcnt() as i32) as f32);
        }
    }

    /// Encode basic positional features
    fn encode_positional_features(&self, board: &Board, features: &mut Vec<f32>) {
        // King safety (distance to center, surrounded pieces)
        for color in [Color::White, Color::Black] {
            let king_square = board.king_square(color);
            // Distance from center
            let center_distance = self.distance_to_center(king_square);
            features.push(center_distance);
            
            // Number of pieces around king (3x3 area)
            let surrounding_pieces = self.count_surrounding_pieces(board, king_square);
            features.push(surrounding_pieces as f32);
        }
        
        // Piece mobility (simplified)
        for color in [Color::White, Color::Black] {
            let mobility = self.calculate_mobility(board, color);
            features.push(mobility as f32);
        }
    }

    /// Calculate distance from square to center of board
    fn distance_to_center(&self, square: Square) -> f32 {
        let file = square.get_file().to_index() as f32;
        let rank = square.get_rank().to_index() as f32;
        let center_file = 3.5;
        let center_rank = 3.5;
        
        ((file - center_file).powi(2) + (rank - center_rank).powi(2)).sqrt()
    }

    /// Count pieces in 3x3 area around a square
    fn count_surrounding_pieces(&self, board: &Board, center: Square) -> u32 {
        let mut count = 0;
        let center_file = center.get_file().to_index() as i32;
        let center_rank = center.get_rank().to_index() as i32;
        
        for file_offset in -1..=1 {
            for rank_offset in -1..=1 {
                if file_offset == 0 && rank_offset == 0 { continue; }
                
                let new_file = center_file + file_offset;
                let new_rank = center_rank + rank_offset;
                
                if new_file >= 0 && new_file < 8 && new_rank >= 0 && new_rank < 8 {
                    let square = Square::make_square(
                        chess::Rank::from_index(new_rank as usize),
                        chess::File::from_index(new_file as usize)
                    );
                    if board.piece_on(square).is_some() {
                        count += 1;
                    }
                }
            }
        }
        count
    }

    /// Calculate basic mobility for a color
    fn calculate_mobility(&self, board: &Board, color: Color) -> u32 {
        // Simplified: count number of pieces that can move
        let pieces = board.color_combined(color);
        let mut mobility = 0;
        
        for _square in *pieces {
            // This is a simplified mobility calculation
            // In a real implementation, you'd generate all legal moves
            mobility += 1;
        }
        
        mobility
    }

    /// Calculate similarity between two position vectors
    pub fn similarity(&self, vec1: &Array1<f32>, vec2: &Array1<f32>) -> f32 {
        // Cosine similarity
        let dot_product = vec1.dot(vec2);
        let norm1 = vec1.dot(vec1).sqrt();
        let norm2 = vec2.dot(vec2).sqrt();
        
        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot_product / (norm1 * norm2)
        }
    }

    /// Calculate Euclidean distance between two vectors
    pub fn distance(&self, vec1: &Array1<f32>, vec2: &Array1<f32>) -> f32 {
        (vec1 - vec2).mapv(|x| x * x).sum().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess::Board;
    use std::str::FromStr;

    #[test]
    fn test_encode_starting_position() {
        let encoder = PositionEncoder::new(1024);
        let board = Board::default();
        let vector = encoder.encode(&board);
        
        assert_eq!(vector.len(), 1024);
        
        // Starting position should have all pieces
        assert!(vector.iter().any(|&x| x > 0.0));
    }

    #[test]
    fn test_similarity_identical_positions() {
        let encoder = PositionEncoder::new(1024);
        let board = Board::default();
        let vec1 = encoder.encode(&board);
        let vec2 = encoder.encode(&board);
        
        let similarity = encoder.similarity(&vec1, &vec2);
        assert!((similarity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_similarity_different_positions() {
        let encoder = PositionEncoder::new(1024);
        let board1 = Board::default();
        let board2 = Board::from_str("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1").unwrap();
        
        let vec1 = encoder.encode(&board1);
        let vec2 = encoder.encode(&board2);
        
        let similarity = encoder.similarity(&vec1, &vec2);
        assert!(similarity < 1.0);
        assert!(similarity > 0.8); // Should still be quite similar (only one move difference)
    }
}