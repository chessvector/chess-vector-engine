use chess::{Board, Color, Piece, Square};
use ndarray::Array1;
use rayon::prelude::*;

/// Basic position encoder that converts chess positions to vectors
#[derive(Clone)]
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

        // 5. Tactical pattern features
        self.encode_tactical_patterns(board, &mut features);

        // Pad or truncate to desired size
        features.resize(self.vector_size, 0.0);

        Array1::from(features)
    }

    /// Encode piece positions on the board using dense representation
    fn encode_piece_positions(&self, board: &Board, features: &mut Vec<f32>) {
        // Enhanced encoding: 64 squares * 12 piece types (6 pieces * 2 colors) = 768 features
        // This creates more distinctive representations
        for square in chess::ALL_SQUARES {
            // One-hot encoding for each piece type and color
            let mut square_features = vec![0.0; 12]; // 6 pieces * 2 colors

            if let Some(piece) = board.piece_on(square) {
                let color = board.color_on(square).unwrap();
                let piece_idx = match piece {
                    chess::Piece::Pawn => 0,
                    chess::Piece::Knight => 1,
                    chess::Piece::Bishop => 2,
                    chess::Piece::Rook => 3,
                    chess::Piece::Queen => 4,
                    chess::Piece::King => 5,
                };

                let color_offset = if color == chess::Color::White { 0 } else { 6 };
                square_features[piece_idx + color_offset] = 1.0;
            }

            features.extend(square_features);
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
        features.push(if board.castle_rights(Color::White).has_kingside() {
            1.0
        } else {
            0.0
        });
        features.push(if board.castle_rights(Color::White).has_queenside() {
            1.0
        } else {
            0.0
        });
        features.push(if board.castle_rights(Color::Black).has_kingside() {
            1.0
        } else {
            0.0
        });
        features.push(if board.castle_rights(Color::Black).has_queenside() {
            1.0
        } else {
            0.0
        });

        // En passant
        features.push(if board.en_passant().is_some() {
            1.0
        } else {
            0.0
        });

        // Side to move
        features.push(if board.side_to_move() == Color::White {
            1.0
        } else {
            0.0
        });

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

        // Add pawn structure features
        self.encode_pawn_structure(board, features);

        // Add tactical patterns
        self.encode_tactical_patterns(board, features);

        // Add center control
        self.encode_center_control(board, features);

        // Add piece coordination patterns
        self.encode_piece_coordination(board, features);
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
                if file_offset == 0 && rank_offset == 0 {
                    continue;
                }

                let new_file = center_file + file_offset;
                let new_rank = center_rank + rank_offset;

                if (0..8).contains(&new_file) && (0..8).contains(&new_rank) {
                    let square = Square::make_square(
                        chess::Rank::from_index(new_rank as usize),
                        chess::File::from_index(new_file as usize),
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

    /// Encode pawn structure features
    fn encode_pawn_structure(&self, board: &Board, features: &mut Vec<f32>) {
        for color in [Color::White, Color::Black] {
            let pawns = board.pieces(Piece::Pawn) & board.color_combined(color);

            // Count doubled pawns (simplified)
            let mut doubled_pawns = 0;
            for file in 0..8 {
                let mut file_pawn_count = 0;
                for rank in 0..8 {
                    let square = chess::Square::make_square(
                        chess::Rank::from_index(rank),
                        chess::File::from_index(file),
                    );
                    if (pawns & chess::BitBoard::from_square(square)).popcnt() > 0 {
                        file_pawn_count += 1;
                    }
                }
                if file_pawn_count > 1 {
                    doubled_pawns += file_pawn_count - 1;
                }
            }
            features.push(doubled_pawns as f32);

            // Count isolated pawns (simplified)
            let mut isolated_pawns = 0;
            for file in 0..8 {
                let mut file_has_pawn = false;
                for rank in 0..8 {
                    let square = chess::Square::make_square(
                        chess::Rank::from_index(rank),
                        chess::File::from_index(file),
                    );
                    if (pawns & chess::BitBoard::from_square(square)).popcnt() > 0 {
                        file_has_pawn = true;
                        break;
                    }
                }

                if file_has_pawn {
                    // Check adjacent files
                    let mut has_adjacent = false;
                    for adj_file in [file.saturating_sub(1), file + 1] {
                        if adj_file < 8 && adj_file != file {
                            for rank in 0..8 {
                                let adj_square = chess::Square::make_square(
                                    chess::Rank::from_index(rank),
                                    chess::File::from_index(adj_file),
                                );
                                if (pawns & chess::BitBoard::from_square(adj_square)).popcnt() > 0 {
                                    has_adjacent = true;
                                    break;
                                }
                            }
                        }
                        if has_adjacent {
                            break;
                        }
                    }

                    if !has_adjacent {
                        isolated_pawns += 1;
                    }
                }
            }
            features.push(isolated_pawns as f32);
        }
    }

    /// Encode tactical patterns
    fn encode_tactical_patterns(&self, board: &Board, features: &mut Vec<f32>) {
        // Count pins, forks, and other tactical motifs (simplified)
        for color in [Color::White, Color::Black] {
            let opponent_color = if color == Color::White {
                Color::Black
            } else {
                Color::White
            };

            // Count potential pins by looking at pieces on same lines as enemy king
            let enemy_king_square = board.king_square(opponent_color);
            let mut potential_pins = 0;

            // Check for pieces that could pin along ranks/files
            let rooks_queens = (board.pieces(Piece::Rook) | board.pieces(Piece::Queen))
                & board.color_combined(color);
            for square in chess::ALL_SQUARES {
                if (rooks_queens & chess::BitBoard::from_square(square)).popcnt() > 0
                    && (square.get_rank() == enemy_king_square.get_rank()
                        || square.get_file() == enemy_king_square.get_file())
                {
                    potential_pins += 1;
                }
            }

            // Check for pieces that could pin along diagonals
            let bishops_queens = (board.pieces(Piece::Bishop) | board.pieces(Piece::Queen))
                & board.color_combined(color);
            for square in chess::ALL_SQUARES {
                if (bishops_queens & chess::BitBoard::from_square(square)).popcnt() > 0 {
                    let rank_diff = (square.get_rank().to_index() as i32
                        - enemy_king_square.get_rank().to_index() as i32)
                        .abs();
                    let file_diff = (square.get_file().to_index() as i32
                        - enemy_king_square.get_file().to_index() as i32)
                        .abs();
                    if rank_diff == file_diff && rank_diff > 0 {
                        potential_pins += 1;
                    }
                }
            }

            features.push(potential_pins as f32);
        }

        // Add center control and piece coordination features
        self.encode_center_control(board, features);
        self.encode_piece_coordination(board, features);
    }

    /// Encode center control
    fn encode_center_control(&self, board: &Board, features: &mut Vec<f32>) {
        // Check control of central squares (d4, d5, e4, e5)
        let center_squares = [
            chess::Square::D4,
            chess::Square::D5,
            chess::Square::E4,
            chess::Square::E5,
        ];

        for color in [Color::White, Color::Black] {
            let mut center_control = 0.0;

            for &square in &center_squares {
                // Check if we have a piece on this square
                if let Some(_piece) = board.piece_on(square) {
                    if board.color_on(square) == Some(color) {
                        center_control += 2.0; // Extra weight for occupying center
                    }
                }

                // Count pieces that could attack this square (simplified)
                let pieces = board.color_combined(color);
                for piece_square in chess::ALL_SQUARES {
                    if (pieces & chess::BitBoard::from_square(piece_square)).popcnt() > 0 {
                        if let Some(piece) = board.piece_on(piece_square) {
                            let can_attack = match piece {
                                Piece::Pawn => {
                                    let rank_diff = (square.get_rank().to_index() as i32
                                        - piece_square.get_rank().to_index() as i32)
                                        .abs();
                                    let file_diff = (square.get_file().to_index() as i32
                                        - piece_square.get_file().to_index() as i32)
                                        .abs();
                                    rank_diff == 1 && file_diff == 1
                                }
                                Piece::Knight => {
                                    let rank_diff = (square.get_rank().to_index() as i32
                                        - piece_square.get_rank().to_index() as i32)
                                        .abs();
                                    let file_diff = (square.get_file().to_index() as i32
                                        - piece_square.get_file().to_index() as i32)
                                        .abs();
                                    (rank_diff == 2 && file_diff == 1)
                                        || (rank_diff == 1 && file_diff == 2)
                                }
                                _ => false, // Simplified - would need more complex logic for sliding pieces
                            };

                            if can_attack {
                                center_control += 0.5;
                            }
                        }
                    }
                }
            }

            features.push(center_control);
        }
    }

    /// Encode piece coordination patterns
    fn encode_piece_coordination(&self, board: &Board, features: &mut Vec<f32>) {
        for color in [Color::White, Color::Black] {
            let mut coordination_score = 0.0;

            // Count pieces defending each other
            let pieces = board.color_combined(color);
            for square1 in chess::ALL_SQUARES {
                if (pieces & chess::BitBoard::from_square(square1)).popcnt() > 0 {
                    for square2 in chess::ALL_SQUARES {
                        if (pieces & chess::BitBoard::from_square(square2)).popcnt() > 0
                            && square1 != square2
                        {
                            // Simplified check for mutual protection
                            let rank_diff = (square1.get_rank().to_index() as i32
                                - square2.get_rank().to_index() as i32)
                                .abs();
                            let file_diff = (square1.get_file().to_index() as i32
                                - square2.get_file().to_index() as i32)
                                .abs();

                            if rank_diff <= 2 && file_diff <= 2 {
                                coordination_score += 0.1;
                            }
                        }
                    }
                }
            }

            features.push(coordination_score);
        }
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

    /// Encode multiple positions in parallel
    pub fn encode_batch(&self, boards: &[Board]) -> Vec<Array1<f32>> {
        if boards.len() > 10 {
            // Use parallel processing for larger batches
            boards.par_iter().map(|board| self.encode(board)).collect()
        } else {
            // Use sequential processing for smaller batches
            boards.iter().map(|board| self.encode(board)).collect()
        }
    }

    /// Calculate similarities between a query vector and multiple position vectors in parallel
    pub fn batch_similarity(&self, query: &Array1<f32>, vectors: &[Array1<f32>]) -> Vec<f32> {
        if vectors.len() > 50 {
            // Use parallel processing for larger batches
            vectors
                .par_iter()
                .map(|vec| self.similarity(query, vec))
                .collect()
        } else {
            // Use sequential processing for smaller batches
            vectors
                .iter()
                .map(|vec| self.similarity(query, vec))
                .collect()
        }
    }

    /// Calculate pairwise similarities between all vectors in parallel
    pub fn pairwise_similarity_matrix(&self, vectors: &[Array1<f32>]) -> Vec<Vec<f32>> {
        if vectors.len() > 20 {
            // Use parallel processing for larger matrices
            vectors
                .par_iter()
                .enumerate()
                .map(|(i, vec1)| {
                    vectors
                        .iter()
                        .enumerate()
                        .map(|(j, vec2)| {
                            if i == j {
                                1.0 // Self-similarity
                            } else {
                                self.similarity(vec1, vec2)
                            }
                        })
                        .collect()
                })
                .collect()
        } else {
            // Use sequential processing for smaller matrices
            vectors
                .iter()
                .enumerate()
                .map(|(i, vec1)| {
                    vectors
                        .iter()
                        .enumerate()
                        .map(|(j, vec2)| {
                            if i == j {
                                1.0 // Self-similarity
                            } else {
                                self.similarity(vec1, vec2)
                            }
                        })
                        .collect()
                })
                .collect()
        }
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
        let board2 =
            Board::from_str("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1").unwrap();

        let vec1 = encoder.encode(&board1);
        let vec2 = encoder.encode(&board2);

        let similarity = encoder.similarity(&vec1, &vec2);
        assert!(similarity < 1.0);
        assert!(similarity > 0.8); // Should still be quite similar (only one move difference)
    }
}
