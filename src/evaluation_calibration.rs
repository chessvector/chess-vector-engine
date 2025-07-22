/// Evaluation Calibration System
/// 
/// Implements SOLID principles for clean, extensible chess evaluation calibration.
/// Maps internal evaluations to standard centipawn scale.

use chess::{Board, Color, Piece, Square};
use std::collections::HashMap;

/// Standard centipawn values for chess pieces
#[derive(Debug, Clone, Copy)]
pub struct PieceValues {
    pub pawn: i32,
    pub knight: i32, 
    pub bishop: i32,
    pub rook: i32,
    pub queen: i32,
    pub king: i32, // Usually 0, but included for completeness
}

impl Default for PieceValues {
    fn default() -> Self {
        Self {
            pawn: 100,
            knight: 300,
            bishop: 300,
            rook: 500,
            queen: 900,
            king: 0,
        }
    }
}

/// Interface for evaluation components (Interface Segregation Principle)
pub trait EvaluationComponent {
    fn evaluate(&self, board: &Board) -> i32; // Returns centipawns
    fn component_name(&self) -> &'static str;
}

/// Material evaluation component (Single Responsibility Principle)
#[derive(Debug)]
pub struct MaterialEvaluator {
    piece_values: PieceValues,
}

impl MaterialEvaluator {
    pub fn new(piece_values: PieceValues) -> Self {
        Self { piece_values }
    }

    fn count_material(&self, board: &Board, color: Color) -> i32 {
        let mut material = 0;
        
        for square in chess::ALL_SQUARES {
            if let Some(piece) = board.piece_on(square) {
                if board.color_on(square) == Some(color) {
                    material += self.get_piece_value(piece);
                }
            }
        }
        
        material
    }

    fn get_piece_value(&self, piece: Piece) -> i32 {
        match piece {
            Piece::Pawn => self.piece_values.pawn,
            Piece::Knight => self.piece_values.knight,
            Piece::Bishop => self.piece_values.bishop,
            Piece::Rook => self.piece_values.rook,
            Piece::Queen => self.piece_values.queen,
            Piece::King => self.piece_values.king,
        }
    }
}

impl EvaluationComponent for MaterialEvaluator {
    fn evaluate(&self, board: &Board) -> i32 {
        let white_material = self.count_material(board, Color::White);
        let black_material = self.count_material(board, Color::Black);
        
        let material_diff = white_material - black_material;
        
        // Return from white's perspective
        if board.side_to_move() == Color::White {
            material_diff
        } else {
            -material_diff
        }
    }

    fn component_name(&self) -> &'static str {
        "Material"
    }
}

/// Position evaluation component (Single Responsibility Principle)
#[derive(Debug)]
pub struct PositionalEvaluator {
    center_control_weight: i32,
    development_weight: i32,
    king_safety_weight: i32,
}

impl PositionalEvaluator {
    pub fn new() -> Self {
        Self {
            center_control_weight: 10, // Centipawns per center square
            development_weight: 15,    // Centipawns per developed piece
            king_safety_weight: 20,    // Centipawns per safety factor
        }
    }

    fn evaluate_center_control(&self, board: &Board, color: Color) -> i32 {
        let center_squares = [
            Square::D4, Square::D5, Square::E4, Square::E5
        ];
        
        let mut control_count = 0;
        
        for &square in &center_squares {
            if let Some(piece_color) = board.color_on(square) {
                if piece_color == color {
                    control_count += 1;
                }
            }
        }
        
        control_count * self.center_control_weight
    }

    fn evaluate_development(&self, board: &Board, color: Color) -> i32 {
        let mut developed_pieces = 0;
        
        // Count developed knights and bishops
        for square in chess::ALL_SQUARES {
            if let Some(piece) = board.piece_on(square) {
                if board.color_on(square) == Some(color) {
                    match piece {
                        Piece::Knight | Piece::Bishop => {
                            // Simple heuristic: not on back rank = developed
                            let rank = square.get_rank();
                            let back_rank = if color == Color::White { 
                                chess::Rank::First 
                            } else { 
                                chess::Rank::Eighth 
                            };
                            
                            if rank != back_rank {
                                developed_pieces += 1;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        
        developed_pieces * self.development_weight
    }

    fn evaluate_king_safety(&self, board: &Board, color: Color) -> i32 {
        // Simple king safety: has king castled?
        let castling_rights = board.castle_rights(color);
        
        if castling_rights.has_kingside() || castling_rights.has_queenside() {
            0 // Can still castle
        } else {
            // Check if king has moved (simplified)
            let king_square = board.king_square(color);
            let expected_king_square = if color == Color::White {
                Square::E1
            } else {
                Square::E8
            };
            
            if king_square == expected_king_square {
                self.king_safety_weight // King still on starting square but can't castle
            } else {
                0 // King has moved (could be castled or unsafe)
            }
        }
    }
}

impl EvaluationComponent for PositionalEvaluator {
    fn evaluate(&self, board: &Board) -> i32 {
        let white_positional = self.evaluate_center_control(board, Color::White)
            + self.evaluate_development(board, Color::White)
            + self.evaluate_king_safety(board, Color::White);
            
        let black_positional = self.evaluate_center_control(board, Color::Black)
            + self.evaluate_development(board, Color::Black)
            + self.evaluate_king_safety(board, Color::Black);

        let positional_diff = white_positional - black_positional;
        
        // Return from current player's perspective
        if board.side_to_move() == Color::White {
            positional_diff
        } else {
            -positional_diff
        }
    }

    fn component_name(&self) -> &'static str {
        "Positional"
    }
}

/// Configuration for evaluation calibration
#[derive(Debug, Clone)]
pub struct CalibrationConfig {
    pub material_weight: f32,
    pub positional_weight: f32,
    pub pattern_weight: f32,
    pub scale_factor: f32, // Converts to final centipawn range
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            material_weight: 1.0,   // Material is most important
            positional_weight: 0.3, // Positional features are secondary
            pattern_weight: 0.2,    // Pattern recognition provides insight
            scale_factor: 1.0,      // Direct centipawn mapping
        }
    }
}

/// Main calibrated evaluator (Dependency Inversion Principle)
pub struct CalibratedEvaluator {
    components: Vec<Box<dyn EvaluationComponent>>,
    config: CalibrationConfig,
}

impl CalibratedEvaluator {
    pub fn new(config: CalibrationConfig) -> Self {
        let mut components: Vec<Box<dyn EvaluationComponent>> = Vec::new();
        
        // Add standard evaluation components
        components.push(Box::new(MaterialEvaluator::new(PieceValues::default())));
        components.push(Box::new(PositionalEvaluator::new()));
        
        Self { components, config }
    }

    /// Add custom evaluation component (Open/Closed Principle)
    pub fn add_component(&mut self, component: Box<dyn EvaluationComponent>) {
        self.components.push(component);
    }

    /// Evaluate position in centipawns
    pub fn evaluate_centipawns(&self, board: &Board) -> i32 {
        let mut total_evaluation = 0;
        
        for component in &self.components {
            let component_eval = component.evaluate(board);
            
            // Apply weights based on component type
            let weighted_eval = match component.component_name() {
                "Material" => (component_eval as f32 * self.config.material_weight) as i32,
                "Positional" => (component_eval as f32 * self.config.positional_weight) as i32,
                _ => component_eval, // Default weight for custom components
            };
            
            total_evaluation += weighted_eval;
        }

        // Apply final scale factor
        (total_evaluation as f32 * self.config.scale_factor) as i32
    }

    /// Get detailed evaluation breakdown
    pub fn evaluate_detailed(&self, board: &Board) -> EvaluationBreakdown {
        let mut breakdown = EvaluationBreakdown::new();
        
        for component in &self.components {
            let component_eval = component.evaluate(board);
            breakdown.add_component(component.component_name(), component_eval);
        }
        
        breakdown.total = self.evaluate_centipawns(board);
        breakdown
    }

    /// Convert raw pattern evaluation to calibrated centipawns
    pub fn calibrate_pattern_evaluation(&self, pattern_eval: f32) -> i32 {
        // Map pattern evaluation (-1.0 to 1.0) to reasonable centipawn range
        let centipawn_range = 200; // ±200 centipawns for pattern evaluation
        let calibrated = pattern_eval * centipawn_range as f32 * self.config.pattern_weight;
        calibrated as i32
    }
}

/// Detailed evaluation breakdown for analysis
#[derive(Debug)]
pub struct EvaluationBreakdown {
    pub components: HashMap<String, i32>,
    pub total: i32,
}

impl EvaluationBreakdown {
    fn new() -> Self {
        Self {
            components: HashMap::new(),
            total: 0,
        }
    }

    fn add_component(&mut self, name: &str, value: i32) {
        self.components.insert(name.to_string(), value);
    }

    pub fn display(&self) -> String {
        let mut result = String::new();
        result.push_str(&format!("Total: {} cp\n", self.total));
        result.push_str("Breakdown:\n");
        
        for (component, value) in &self.components {
            result.push_str(&format!("  {}: {} cp\n", component, value));
        }
        
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_material_evaluator() {
        let evaluator = MaterialEvaluator::new(PieceValues::default());
        
        // Test starting position (equal material)
        let board = Board::default();
        assert_eq!(evaluator.evaluate(&board), 0);
        
        // Test position with extra knight for white
        let board = Board::from_str("rnbqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        assert_eq!(evaluator.evaluate(&board), 300); // Extra knight = 300 cp
    }

    #[test]
    fn test_positional_evaluator() {
        let evaluator = PositionalEvaluator::new();
        
        // Test starting position
        let board = Board::default();
        let eval = evaluator.evaluate(&board);
        assert_eq!(eval, 0); // Equal position
    }

    #[test]
    fn test_calibrated_evaluator() {
        let evaluator = CalibratedEvaluator::new(CalibrationConfig::default());
        
        // Test starting position
        let board = Board::default();
        let eval = evaluator.evaluate_centipawns(&board);
        assert!(eval.abs() <= 50); // Should be close to equal
        
        // Test detailed breakdown
        let breakdown = evaluator.evaluate_detailed(&board);
        assert!(breakdown.components.contains_key("Material"));
        assert!(breakdown.components.contains_key("Positional"));
    }

    #[test]
    fn test_pattern_calibration() {
        let evaluator = CalibratedEvaluator::new(CalibrationConfig::default());
        
        // Test pattern evaluation calibration
        let pattern_eval = 0.5; // Slight advantage
        let calibrated = evaluator.calibrate_pattern_evaluation(pattern_eval);
        assert!(calibrated > 0 && calibrated <= 50); // Should be reasonable
    }
}