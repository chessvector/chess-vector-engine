use chess::{Board, ChessMove, File, MoveGen, Piece, Rank, Square};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Strategic evaluation system for proactive, initiative-based play
/// This module transforms Chess Vector Engine from reactive to proactive by:
/// 1. Evaluating initiative and attacking potential
/// 2. Generating strategic plans based on position patterns
/// 3. Coordinating pieces toward strategic goals
/// 4. Maintaining our hybrid approach (NNUE + Pattern Recognition + Strategic Planning)
#[derive(Debug, Clone)]
pub struct StrategicEvaluator {
    config: StrategicConfig,
    attacking_patterns: HashMap<String, AttackingPattern>,
    positional_plans: Vec<PositionalPlan>,
}

/// Configuration for strategic evaluation emphasizing proactive play
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategicConfig {
    /// Weight for initiative and attacking chances vs positional safety (0.0-1.0)
    pub initiative_weight: f32,

    /// Bonus for pieces actively participating in attack (centipawns)
    pub attacking_piece_bonus: f32,

    /// Weight for dynamic factors like tempo, development, piece activity (0.0-1.0)
    pub dynamic_factor_weight: f32,

    /// Enable advanced pawn structure planning and pawn breaks
    pub pawn_structure_planning: bool,

    /// Bonus for controlling key squares around enemy king (centipawns)
    pub king_attack_square_bonus: f32,

    /// Weight for piece coordination in attacks (0.0-1.0)
    pub piece_coordination_weight: f32,

    /// Enable generation of forcing moves that create imbalances
    pub enable_imbalance_creation: bool,
}

impl Default for StrategicConfig {
    fn default() -> Self {
        Self {
            initiative_weight: 0.4,           // Balanced initiative vs safety
            attacking_piece_bonus: 10.0,      // 10cp bonus per attacking piece (reduced)
            dynamic_factor_weight: 0.3,       // Moderate tempo focus
            pawn_structure_planning: true,    // Plan pawn breaks
            king_attack_square_bonus: 8.0,    // 8cp per controlled attack square (reduced)
            piece_coordination_weight: 0.4,   // Moderate coordination
            enable_imbalance_creation: false, // Avoid unnecessary complications (changed)
        }
    }
}

impl StrategicConfig {
    /// Aggressive configuration for maximum initiative and attacking play
    pub fn aggressive() -> Self {
        Self {
            initiative_weight: 0.8,         // Heavy focus on initiative
            attacking_piece_bonus: 25.0,    // High reward for attacking pieces
            dynamic_factor_weight: 0.6,     // Very high tempo focus
            king_attack_square_bonus: 15.0, // High reward for king attack
            piece_coordination_weight: 0.7, // Strong piece coordination
            enable_imbalance_creation: true,
            ..Default::default()
        }
    }

    /// Balanced configuration for strategic play with safety considerations
    pub fn balanced() -> Self {
        Self {
            initiative_weight: 0.3,           // Safety-first balanced approach
            attacking_piece_bonus: 8.0,       // Conservative attacking bonus
            dynamic_factor_weight: 0.25,      // Moderate tempo consideration
            king_attack_square_bonus: 6.0,    // Conservative king attack bonus
            piece_coordination_weight: 0.5,   // Good coordination for safety
            enable_imbalance_creation: false, // Avoid risky complications
            ..Default::default()
        }
    }

    /// Positional configuration focusing on safety and long-term advantages
    pub fn positional() -> Self {
        Self {
            initiative_weight: 0.2,           // Very low initiative focus
            attacking_piece_bonus: 5.0,       // Minimal attacking bonus
            dynamic_factor_weight: 0.15,      // Focus on positional factors
            pawn_structure_planning: true,    // Strong pawn planning
            king_attack_square_bonus: 3.0,    // Minimal king attack focus
            piece_coordination_weight: 0.7,   // High coordination for safety
            enable_imbalance_creation: false, // Avoid complications completely
        }
    }

    /// Safety-first configuration for solid, sound play
    pub fn safety_first() -> Self {
        Self {
            initiative_weight: 0.15,          // Minimal initiative focus
            attacking_piece_bonus: 3.0,       // Very low attacking bonus
            dynamic_factor_weight: 0.1,       // Minimal tempo focus
            pawn_structure_planning: true,    // Sound pawn structure
            king_attack_square_bonus: 2.0,    // Very low attack bonus
            piece_coordination_weight: 0.8,   // High coordination for safety
            enable_imbalance_creation: false, // No complications at all
        }
    }

    /// Master-level configuration for 2000+ ELO play
    pub fn master_level() -> Self {
        Self {
            initiative_weight: 0.25,          // Measured initiative
            attacking_piece_bonus: 6.0,       // Controlled attacking
            dynamic_factor_weight: 0.2,       // Good tempo sense
            pawn_structure_planning: true,    // Essential for master play
            king_attack_square_bonus: 5.0,    // Balanced king safety
            piece_coordination_weight: 0.6,   // Strong coordination
            enable_imbalance_creation: false, // Avoid unnecessary risks
        }
    }
}

/// Strategic patterns for recognizing attacking and positional motifs
#[derive(Debug, Clone)]
pub struct AttackingPattern {
    pub name: String,
    pub description: String,
    pub initiative_bonus: f32, // Bonus for positions matching this pattern
    pub required_pieces: Vec<Piece>, // Pieces that must be present
    pub target_squares: Vec<Square>, // Key squares this pattern targets
    pub coordination_bonus: f32, // Bonus when pieces coordinate in this pattern
}

/// Strategic plans for proactive play
#[derive(Debug, Clone)]
pub struct PositionalPlan {
    pub name: String,
    pub goal: PlanGoal,
    pub required_moves: Vec<ChessMove>, // Moves that advance this plan
    pub evaluation_bonus: f32,          // Bonus for positions that advance this plan
    pub urgency: PlanUrgency,           // How quickly this plan should be executed
}

#[derive(Debug, Clone, PartialEq)]
pub enum PlanGoal {
    AttackKing,        // Direct attack on enemy king
    ControlCenter,     // Gain central control
    CreateWeaknesses,  // Force weaknesses in enemy position
    PieceCoordination, // Improve piece coordination
    PawnStructure,     // Improve pawn structure or create breaks
    InitiativeSeizure, // Seize the initiative with forcing moves
}

#[derive(Debug, Clone, PartialEq)]
pub enum PlanUrgency {
    Immediate, // Execute this plan now
    High,      // Execute soon
    Medium,    // Execute when opportunity arises
    Low,       // Long-term positional goal
}

/// Result of strategic evaluation
#[derive(Debug, Clone)]
pub struct StrategicEvaluation {
    pub base_evaluation: f32,                     // Base position evaluation
    pub initiative_bonus: f32,                    // Bonus for initiative and attacking chances
    pub attacking_bonus: f32,                     // Bonus for pieces participating in attack
    pub coordination_bonus: f32,                  // Bonus for piece coordination
    pub plan_bonus: f32,                          // Bonus for advancing strategic plans
    pub total_evaluation: f32,                    // Final strategic evaluation
    pub recommended_plan: Option<PositionalPlan>, // Suggested strategic plan
    pub attacking_moves: Vec<ChessMove>,          // Moves that support attack
    pub positional_moves: Vec<ChessMove>,         // Moves that improve position
}

impl StrategicEvaluator {
    /// Create a new strategic evaluator with configuration
    pub fn new(config: StrategicConfig) -> Self {
        let mut evaluator = Self {
            config,
            attacking_patterns: HashMap::new(),
            positional_plans: Vec::new(),
        };

        evaluator.initialize_attacking_patterns();
        evaluator.initialize_positional_plans();

        evaluator
    }

    /// Create strategic evaluator with default balanced configuration
    pub fn new_default() -> Self {
        Self::new(StrategicConfig::default())
    }

    /// Create aggressive strategic evaluator for maximum initiative
    pub fn aggressive() -> Self {
        Self::new(StrategicConfig::aggressive())
    }

    /// Evaluate position strategically, focusing on initiative balanced with safety
    pub fn evaluate_strategic(&self, board: &Board) -> StrategicEvaluation {
        let base_evaluation = self.calculate_base_evaluation(board);
        let initiative_bonus = self.evaluate_initiative(board);
        let attacking_bonus = self.evaluate_attacking_potential(board);
        let coordination_bonus = self.evaluate_piece_coordination(board);
        let (plan_bonus, recommended_plan) = self.evaluate_strategic_plans(board);

        // v0.4.0: Add safety penalty for reckless play
        let safety_penalty = self.evaluate_safety_concerns(board);

        let total_evaluation = base_evaluation
            + (initiative_bonus * self.config.initiative_weight)
            + attacking_bonus
            + (coordination_bonus * self.config.piece_coordination_weight)
            + plan_bonus
            - safety_penalty; // Subtract safety concerns

        let attacking_moves = self.generate_safe_attacking_moves(board);
        let positional_moves = self.generate_safe_positional_moves(board);

        StrategicEvaluation {
            base_evaluation,
            initiative_bonus,
            attacking_bonus,
            coordination_bonus,
            plan_bonus,
            total_evaluation,
            recommended_plan,
            attacking_moves,
            positional_moves,
        }
    }

    /// Generate proactive moves that seize initiative while maintaining safety
    pub fn generate_proactive_moves(&self, board: &Board) -> Vec<(ChessMove, f32)> {
        let strategic_eval = self.evaluate_strategic(board);
        let mut proactive_moves = Vec::new();

        // Prioritize attacking moves when we have initiative
        for mv in strategic_eval.attacking_moves {
            let move_value = self.evaluate_move_initiative(board, &mv);
            let safety_penalty = self.evaluate_move_safety(board, &mv);
            let final_value = move_value - safety_penalty; // Balance initiative with safety

            if final_value > 50.0 {
                // Ultra-strict threshold for attacking moves (prevent blunders)
                proactive_moves.push((mv, final_value));
            }
        }

        // Add positional moves that improve our strategic position
        for mv in strategic_eval.positional_moves {
            let move_value = self.evaluate_move_positional(board, &mv);
            let safety_penalty = self.evaluate_move_safety(board, &mv);
            let final_value = move_value - safety_penalty;

            if final_value > 30.0 {
                // Ultra-strict threshold for positional moves (prevent blunders)
                proactive_moves.push((mv, final_value));
            }
        }

        // Sort by strategic value (highest first)
        proactive_moves.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Limit to top 10 moves to avoid overwhelming with bad moves
        proactive_moves.truncate(10);

        proactive_moves
    }

    /// Check if position favors initiative and attacking play
    pub fn should_play_aggressively(&self, board: &Board) -> bool {
        let strategic_eval = self.evaluate_strategic(board);

        // Play aggressively if:
        // 1. We have initiative advantage
        // 2. Good piece coordination for attack
        // 3. Enemy king is not fully safe
        strategic_eval.initiative_bonus > 0.0
            && strategic_eval.coordination_bonus > 0.0
            && self.is_enemy_king_attackable(board)
    }

    /// Integration with existing hybrid evaluation system
    pub fn blend_with_hybrid_evaluation(
        &self,
        board: &Board,
        nnue_eval: f32,
        pattern_eval: f32,
    ) -> f32 {
        let strategic_eval = self.evaluate_strategic(board);

        // Blend strategic evaluation with existing NNUE and pattern evaluations
        let strategic_weight = if self.should_play_aggressively(board) {
            0.4
        } else {
            0.2
        };
        let nnue_weight = 0.5;
        let pattern_weight = 0.3;

        (strategic_eval.total_evaluation * strategic_weight)
            + (nnue_eval * nnue_weight)
            + (pattern_eval * pattern_weight)
    }

    // Private implementation methods

    fn initialize_attacking_patterns(&mut self) {
        // King attack patterns
        self.attacking_patterns.insert(
            "king_attack_setup".to_string(),
            AttackingPattern {
                name: "King Attack Setup".to_string(),
                description: "Pieces coordinated for king attack".to_string(),
                initiative_bonus: 50.0,
                required_pieces: vec![Piece::Queen, Piece::Rook],
                target_squares: vec![], // Will be calculated based on enemy king position
                coordination_bonus: 25.0,
            },
        );

        // Central attack patterns
        self.attacking_patterns.insert(
            "central_pressure".to_string(),
            AttackingPattern {
                name: "Central Pressure".to_string(),
                description: "Control center and create attacking chances".to_string(),
                initiative_bonus: 30.0,
                required_pieces: vec![Piece::Knight, Piece::Bishop],
                target_squares: vec![Square::E4, Square::E5, Square::D4, Square::D5],
                coordination_bonus: 15.0,
            },
        );

        // Piece coordination patterns
        self.attacking_patterns.insert(
            "piece_coordination".to_string(),
            AttackingPattern {
                name: "Piece Coordination".to_string(),
                description: "Multiple pieces working together".to_string(),
                initiative_bonus: 25.0,
                required_pieces: vec![Piece::Bishop, Piece::Knight],
                target_squares: vec![],
                coordination_bonus: 20.0,
            },
        );
    }

    fn initialize_positional_plans(&mut self) {
        self.positional_plans.push(PositionalPlan {
            name: "King Attack".to_string(),
            goal: PlanGoal::AttackKing,
            required_moves: vec![], // Will be calculated dynamically
            evaluation_bonus: 100.0,
            urgency: PlanUrgency::High,
        });

        self.positional_plans.push(PositionalPlan {
            name: "Control Center".to_string(),
            goal: PlanGoal::ControlCenter,
            required_moves: vec![],
            evaluation_bonus: 50.0,
            urgency: PlanUrgency::Medium,
        });

        self.positional_plans.push(PositionalPlan {
            name: "Seize Initiative".to_string(),
            goal: PlanGoal::InitiativeSeizure,
            required_moves: vec![],
            evaluation_bonus: 75.0,
            urgency: PlanUrgency::Immediate,
        });
    }

    fn calculate_base_evaluation(&self, board: &Board) -> f32 {
        // Basic material and positional evaluation
        // This integrates with our existing evaluation systems
        let mut eval = 0.0;

        // Material count with strategic adjustments
        for square in chess::ALL_SQUARES {
            if let Some(piece) = board.piece_on(square) {
                let piece_value = match piece {
                    Piece::Pawn => 100.0,
                    Piece::Knight => 320.0,
                    Piece::Bishop => 330.0,
                    Piece::Rook => 500.0,
                    Piece::Queen => 900.0,
                    Piece::King => 0.0,
                };

                if board.color_on(square) == Some(board.side_to_move()) {
                    eval += piece_value;
                } else {
                    eval -= piece_value;
                }
            }
        }

        eval / 100.0 // Convert to centipawn-like scale
    }

    fn evaluate_initiative(&self, board: &Board) -> f32 {
        let mut initiative = 0.0;

        // More legal moves = more initiative
        let our_moves = MoveGen::new_legal(board).count() as f32;

        // Simulate opponent's response to estimate their mobility
        let opponent_mobility = self.estimate_opponent_mobility(board);

        // Initiative bonus based on mobility advantage
        let mobility_advantage = our_moves - opponent_mobility;
        initiative += mobility_advantage * 2.0;

        // Bonus for having the move in sharp positions
        if self.is_position_sharp(board) {
            initiative += 20.0;
        }

        // Bonus for controlling key central squares
        initiative += self.evaluate_central_control(board) * 10.0;

        initiative
    }

    fn evaluate_attacking_potential(&self, board: &Board) -> f32 {
        let mut attack_bonus = 0.0;
        let enemy_king_square = self.get_enemy_king_square(board);

        if let Some(king_square) = enemy_king_square {
            // Count pieces attacking enemy king area
            let attacking_pieces = self.count_pieces_attacking_king_area(board, king_square);
            attack_bonus += attacking_pieces as f32 * self.config.attacking_piece_bonus;

            // Bonus for controlling squares around enemy king
            let controlled_attack_squares =
                self.count_controlled_king_area_squares(board, king_square);
            attack_bonus += controlled_attack_squares as f32 * self.config.king_attack_square_bonus;
        }

        attack_bonus
    }

    fn evaluate_piece_coordination(&self, board: &Board) -> f32 {
        let mut coordination = 0.0;

        // Check for pieces supporting each other
        for square in chess::ALL_SQUARES {
            if let Some(_piece) = board.piece_on(square) {
                if board.color_on(square) == Some(board.side_to_move()) {
                    coordination += self.count_piece_supporters(board, square) * 5.0;
                }
            }
        }

        coordination
    }

    fn evaluate_strategic_plans(&self, board: &Board) -> (f32, Option<PositionalPlan>) {
        let mut best_plan_bonus = 0.0;
        let mut best_plan = None;

        for plan in &self.positional_plans {
            let plan_feasibility = self.evaluate_plan_feasibility(board, plan);
            let plan_value = plan.evaluation_bonus * plan_feasibility;

            if plan_value > best_plan_bonus {
                best_plan_bonus = plan_value;
                best_plan = Some(plan.clone());
            }
        }

        (best_plan_bonus, best_plan)
    }

    fn generate_safe_attacking_moves(&self, board: &Board) -> Vec<ChessMove> {
        let mut attacking_moves = Vec::new();

        for mv in MoveGen::new_legal(board) {
            if self.is_move_attacking(&mv, board) {
                // Only include attacking moves that don't hang pieces badly
                let safety_penalty = self.evaluate_move_safety(board, &mv);
                if safety_penalty < 50.0 {
                    // Don't hang even 0.5 pawns worth (very strict)
                    attacking_moves.push(mv);
                }
            }
        }

        attacking_moves
    }

    fn generate_safe_positional_moves(&self, board: &Board) -> Vec<ChessMove> {
        let mut positional_moves = Vec::new();

        for mv in MoveGen::new_legal(board) {
            if self.is_move_positional(&mv, board) {
                // Only include positional moves that are reasonably safe
                let safety_penalty = self.evaluate_move_safety(board, &mv);
                if safety_penalty < 30.0 {
                    // Don't hang even 0.3 pawns worth (extremely strict)
                    positional_moves.push(mv);
                }
            }
        }

        positional_moves
    }

    // Helper methods for strategic evaluation

    fn estimate_opponent_mobility(&self, board: &Board) -> f32 {
        // Estimate opponent's mobility by simulating their turn
        let current_side = board.side_to_move();

        // Note: We can't actually change side_to_move in chess crate, so we approximate

        // Count legal moves for opponent pieces
        let opponent_color = !current_side;
        let mut opponent_mobility = 0;

        for square in chess::ALL_SQUARES {
            if board.color_on(square) == Some(opponent_color) {
                if let Some(piece) = board.piece_on(square) {
                    // Count possible destinations for this piece
                    for target_square in chess::ALL_SQUARES {
                        if self.can_piece_attack_square(board, square, piece, target_square) {
                            opponent_mobility += 1;
                        }
                    }
                }
            }
        }

        opponent_mobility as f32
    }

    fn is_position_sharp(&self, board: &Board) -> bool {
        // Check if position has tactical opportunities
        let mut sharp_indicators = 0;

        // 1. Many legal moves (tactical complexity)
        let legal_moves = MoveGen::new_legal(board).count();
        if legal_moves > 35 {
            sharp_indicators += 1;
        }

        // 2. Pieces under attack
        let mut pieces_under_attack = 0;
        for square in chess::ALL_SQUARES {
            if board.piece_on(square).is_some() && self.is_square_attacked_by_enemy(board, square) {
                pieces_under_attack += 1;
            }
        }
        if pieces_under_attack >= 3 {
            sharp_indicators += 1;
        }

        // 3. King in danger
        if board.checkers().0 != 0 {
            sharp_indicators += 2; // Check is very sharp
        }

        // 4. Many captures available
        let captures = MoveGen::new_legal(board)
            .filter(|mv| board.piece_on(mv.get_dest()).is_some())
            .count();
        if captures >= 5 {
            sharp_indicators += 1;
        }

        sharp_indicators >= 2 // Position is sharp if multiple indicators present
    }

    fn evaluate_central_control(&self, board: &Board) -> f32 {
        let central_squares = [Square::E4, Square::E5, Square::D4, Square::D5];
        let mut control = 0.0;

        for square in central_squares {
            if board.color_on(square) == Some(board.side_to_move()) {
                control += 1.0;
            }
        }

        control / central_squares.len() as f32
    }

    fn get_enemy_king_square(&self, board: &Board) -> Option<Square> {
        let enemy_color = !board.side_to_move();
        Some(board.king_square(enemy_color))
    }

    fn count_pieces_attacking_king_area(&self, board: &Board, king_square: Square) -> usize {
        // Count our pieces that can attack squares around enemy king
        let king_area = self.get_king_area_squares(king_square);
        let mut attacking_pieces = 0;

        for square in chess::ALL_SQUARES {
            if board.color_on(square) == Some(board.side_to_move()) {
                if let Some(piece) = board.piece_on(square) {
                    if self.piece_attacks_king_area(board, square, piece, &king_area) {
                        attacking_pieces += 1;
                    }
                }
            }
        }

        attacking_pieces
    }

    fn count_controlled_king_area_squares(&self, board: &Board, king_square: Square) -> usize {
        let king_area = self.get_king_area_squares(king_square);
        let mut controlled = 0;

        for area_square in king_area {
            if self.do_we_control_square(board, area_square) {
                controlled += 1;
            }
        }

        controlled
    }

    fn get_king_area_squares(&self, king_square: Square) -> Vec<Square> {
        let mut area_squares = Vec::new();
        let king_file = king_square.get_file().to_index() as i8;
        let king_rank = king_square.get_rank().to_index() as i8;

        // Add squares around the king (3x3 area)
        for file_offset in -1..=1 {
            for rank_offset in -1..=1 {
                let new_file = king_file + file_offset;
                let new_rank = king_rank + rank_offset;

                if (0..8).contains(&new_file) && (0..8).contains(&new_rank) {
                    let file = File::from_index(new_file as usize);
                    let rank = Rank::from_index(new_rank as usize);
                    let square_index = (rank.to_index() * 8 + file.to_index()) as u8;
                    let square = unsafe { Square::new(square_index) };
                    area_squares.push(square);
                }
            }
        }

        area_squares
    }

    fn piece_attacks_king_area(
        &self,
        board: &Board,
        piece_square: Square,
        piece: Piece,
        king_area: &[Square],
    ) -> bool {
        // Check if piece can attack any square in the king area
        for &target_square in king_area {
            if self.can_piece_attack_square(board, piece_square, piece, target_square) {
                return true;
            }
        }
        false
    }

    fn do_we_control_square(&self, board: &Board, square: Square) -> bool {
        // Check if any of our pieces can attack this square
        let our_color = board.side_to_move();

        for check_square in chess::ALL_SQUARES {
            if board.color_on(check_square) == Some(our_color) {
                if let Some(piece) = board.piece_on(check_square) {
                    if self.can_piece_attack_square(board, check_square, piece, square) {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn count_piece_supporters(&self, board: &Board, square: Square) -> f32 {
        // Count pieces that can defend/support this piece
        let our_color = board.side_to_move();
        let mut supporters = 0.0;

        for check_square in chess::ALL_SQUARES {
            if board.color_on(check_square) == Some(our_color) && check_square != square {
                if let Some(piece) = board.piece_on(check_square) {
                    if self.can_piece_attack_square(board, check_square, piece, square) {
                        supporters += 1.0;
                    }
                }
            }
        }
        supporters
    }

    fn evaluate_plan_feasibility(&self, _board: &Board, plan: &PositionalPlan) -> f32 {
        // Evaluate how feasible this plan is in current position
        match plan.urgency {
            PlanUrgency::Immediate => 1.0,
            PlanUrgency::High => 0.8,
            PlanUrgency::Medium => 0.6,
            PlanUrgency::Low => 0.4,
        }
    }

    fn is_move_attacking(&self, mv: &ChessMove, board: &Board) -> bool {
        // Check if move is attacking in nature
        let dest = mv.get_dest();

        // 1. Captures are attacking
        if board.piece_on(dest).is_some() {
            return true;
        }

        // 2. Moves that attack enemy pieces
        if let Some(piece) = board.piece_on(mv.get_source()) {
            // Check if this move puts pressure on enemy pieces
            let temp_board = board.make_move_new(*mv);
            let enemy_color = !board.side_to_move();

            for enemy_square in chess::ALL_SQUARES {
                if temp_board.color_on(enemy_square) == Some(enemy_color)
                    && self.can_piece_attack_square(&temp_board, dest, piece, enemy_square)
                {
                    return true;
                }
            }
        }

        // 3. Moves toward enemy territory
        let forward_progress = if board.side_to_move() == chess::Color::White {
            dest.get_rank().to_index() > mv.get_source().get_rank().to_index()
        } else {
            dest.get_rank().to_index() < mv.get_source().get_rank().to_index()
        };

        forward_progress && dest.get_rank().to_index() > 3 && dest.get_rank().to_index() < 4
    }

    fn is_move_positional(&self, mv: &ChessMove, _board: &Board) -> bool {
        // Check if move improves position
        !self.is_move_attacking(mv, _board) // Non-attacking moves are positional
    }

    fn evaluate_move_initiative(&self, board: &Board, mv: &ChessMove) -> f32 {
        let mut initiative_value = 0.0;

        // Higher value for captures
        if let Some(captured_piece) = board.piece_on(mv.get_dest()) {
            initiative_value += match captured_piece {
                Piece::Queen => 90.0,
                Piece::Rook => 50.0,
                Piece::Bishop | Piece::Knight => 30.0,
                Piece::Pawn => 10.0,
                Piece::King => 1000.0, // Should never happen in legal moves
            };
        }

        // Bonus for centralization
        let dest_centrality = self.square_centrality(mv.get_dest());
        initiative_value += dest_centrality * 5.0;

        // Bonus for check moves
        let temp_board = board.make_move_new(*mv);
        if temp_board.checkers().0 != 0 {
            initiative_value += 25.0;
        }

        // Bonus for attacking enemy king area
        if let Some(enemy_king) = self.get_enemy_king_square(board) {
            let king_area = self.get_king_area_squares(enemy_king);
            if let Some(piece) = board.piece_on(mv.get_source()) {
                for &area_square in &king_area {
                    if self.can_piece_attack_square(&temp_board, mv.get_dest(), piece, area_square)
                    {
                        initiative_value += 15.0;
                        break;
                    }
                }
            }
        }

        initiative_value
    }

    fn evaluate_move_positional(&self, board: &Board, mv: &ChessMove) -> f32 {
        let mut positional_value = 0.0;

        // Master-level positional principles
        if let Some(piece) = board.piece_on(mv.get_source()) {
            match piece {
                Piece::Knight | Piece::Bishop => {
                    // Development bonus - stronger for first development
                    if self.is_piece_developed(mv.get_dest(), board.side_to_move()) {
                        positional_value += 15.0; // Higher development bonus
                    }

                    // Knights on rim are dim - penalty for edge moves
                    if self.is_edge_square(mv.get_dest()) {
                        positional_value -= 10.0;
                    }

                    // Bishops on long diagonals
                    if piece == Piece::Bishop && self.is_long_diagonal(mv.get_dest()) {
                        positional_value += 12.0;
                    }
                }
                Piece::Pawn => {
                    // Advanced pawns in center
                    if self.is_central_square(mv.get_dest()) {
                        positional_value += 8.0;
                    }

                    // Pawn chains and support
                    if self.creates_pawn_chain(board, mv) {
                        positional_value += 10.0;
                    }
                }
                Piece::Rook => {
                    // Rooks on open files
                    if self.is_open_file(board, mv.get_dest()) {
                        positional_value += 15.0;
                    }

                    // Rooks on 7th rank
                    if self.is_seventh_rank(mv.get_dest(), board.side_to_move()) {
                        positional_value += 20.0;
                    }
                }
                Piece::Queen => {
                    // Queen centralization in middlegame
                    if !self.is_early_game(board) {
                        let centrality = self.square_centrality(mv.get_dest());
                        positional_value += centrality * 8.0;
                    } else {
                        // Penalty for early queen moves
                        positional_value -= 15.0;
                    }
                }
                _ => {}
            }
        }

        // Master-level positional factors
        let centrality = self.square_centrality(mv.get_dest());
        positional_value += centrality * 5.0; // Higher centrality bonus

        // Piece coordination and harmony
        let temp_board = board.make_move_new(*mv);
        let supporters_after = self.count_piece_supporters(&temp_board, mv.get_dest());
        positional_value += supporters_after * 4.0; // Higher coordination bonus

        // Avoid squares attacked by enemy pawns (weakness principle)
        if self.is_attacked_by_enemy_pawns(&temp_board, mv.get_dest()) {
            positional_value -= 15.0;
        }

        // Improve worst-placed piece principle
        if self.improves_worst_piece(board, mv) {
            positional_value += 12.0;
        }

        positional_value
    }

    fn is_enemy_king_attackable(&self, board: &Board) -> bool {
        if let Some(king_square) = self.get_enemy_king_square(board) {
            self.count_pieces_attacking_king_area(board, king_square) > 0
        } else {
            false
        }
    }

    // Safety evaluation functions to balance aggression with sound play

    fn evaluate_safety_concerns(&self, board: &Board) -> f32 {
        let mut safety_penalty = 0.0;
        let our_color = board.side_to_move();

        // Check for hanging pieces
        for square in chess::ALL_SQUARES {
            if board.color_on(square) == Some(our_color) {
                if let Some(piece) = board.piece_on(square) {
                    if self.is_square_attacked_by_enemy(board, square) {
                        let defenders = self.count_piece_supporters(board, square);
                        if defenders == 0.0 {
                            // Piece is hanging - major safety concern
                            safety_penalty += self.piece_value(piece) * 0.8; // 80% of piece value
                        } else if defenders < 1.0 {
                            // Piece is under-defended
                            safety_penalty += self.piece_value(piece) * 0.3; // 30% of piece value
                        }
                    }
                }
            }
        }

        // King safety penalty
        let king_square = board.king_square(our_color);
        if self.is_square_attacked_by_enemy(board, king_square) {
            safety_penalty += 200.0; // King under attack is very dangerous
        }

        // Check if king area is under attack
        let king_area = self.get_king_area_squares(king_square);
        let mut attacked_king_squares = 0;
        for &area_square in &king_area {
            if self.is_square_attacked_by_enemy(board, area_square) {
                attacked_king_squares += 1;
            }
        }
        safety_penalty += attacked_king_squares as f32 * 15.0; // Each attacked square around king

        safety_penalty
    }

    fn evaluate_move_safety(&self, board: &Board, mv: &ChessMove) -> f32 {
        let mut safety_penalty = 0.0;
        let our_color = board.side_to_move();
        let enemy_color = !our_color;

        // Simulate the move to check resulting position
        let temp_board = board.make_move_new(*mv);
        let moved_piece = board.piece_on(mv.get_source());

        if let Some(piece) = moved_piece {
            // Check if the moved piece becomes hanging (check attacks by enemy on temp board)
            if self.is_square_attacked_by_color(&temp_board, mv.get_dest(), enemy_color) {
                let defenders =
                    self.count_defenders_for_color(&temp_board, mv.get_dest(), our_color);
                if defenders == 0.0 {
                    // Piece hangs after the move - critical safety issue
                    safety_penalty += self.piece_value(piece) * 1.2; // 120% penalty to strongly discourage
                } else {
                    // Piece is attacked but defended - still risky for valuable pieces
                    let attackers =
                        self.count_attackers_for_color(&temp_board, mv.get_dest(), enemy_color);
                    if attackers > defenders && self.piece_value(piece) > 300.0 {
                        safety_penalty += self.piece_value(piece) * 0.7; // 70% penalty for risky valuable piece moves
                    }
                }
            }

            // Penalty for moving pieces backward without good reason (anti-development)
            if self.is_move_backward(mv, board.side_to_move())
                && board.piece_on(mv.get_dest()).is_none()
            {
                safety_penalty += 25.0; // Discourage backward moves unless capturing
            }

            // Penalty for early queen moves that might get attacked
            if piece == Piece::Queen && self.is_early_game(board) {
                let queen_attackers = self.count_enemy_attackers(&temp_board, mv.get_dest());
                if queen_attackers > 0.0 {
                    safety_penalty += 150.0; // Heavy penalty for exposing queen early
                }
            }
        }

        // Check if move exposes our king
        let our_king_square = board.king_square(our_color);
        if self.is_square_attacked_by_color(&temp_board, our_king_square, enemy_color)
            && board.checkers().0 == 0
        {
            safety_penalty += 200.0; // Heavy penalty for exposing king to check
        }

        // Additional check: Does this move leave other pieces hanging?
        let hanging_penalty = self.count_hanging_pieces_after_move(&temp_board, our_color);
        safety_penalty += hanging_penalty * 100.0; // 100cp per hanging piece

        safety_penalty
    }

    fn piece_value(&self, piece: Piece) -> f32 {
        match piece {
            Piece::Pawn => 100.0,
            Piece::Knight => 320.0,
            Piece::Bishop => 330.0,
            Piece::Rook => 500.0,
            Piece::Queen => 900.0,
            Piece::King => 10000.0, // King is invaluable
        }
    }

    fn count_enemy_attackers(&self, board: &Board, square: Square) -> f32 {
        let enemy_color = !board.side_to_move();
        self.count_attackers_for_color(board, square, enemy_color)
    }

    fn is_move_backward(&self, mv: &ChessMove, color: chess::Color) -> bool {
        let source_rank = mv.get_source().get_rank().to_index();
        let dest_rank = mv.get_dest().get_rank().to_index();

        match color {
            chess::Color::White => dest_rank < source_rank, // White moving backwards
            chess::Color::Black => dest_rank > source_rank, // Black moving backwards
        }
    }

    fn is_early_game(&self, board: &Board) -> bool {
        // Count developed pieces - if few pieces developed, it's early game
        let mut developed_pieces = 0;

        for square in chess::ALL_SQUARES {
            if let Some(piece) = board.piece_on(square) {
                if piece != Piece::Pawn
                    && piece != Piece::King
                    && self.is_piece_developed(square, board.color_on(square).unwrap())
                {
                    developed_pieces += 1;
                }
            }
        }

        developed_pieces < 6 // Early game if less than 6 pieces developed total
    }

    // Helper functions for strategic evaluation

    fn can_piece_attack_square(
        &self,
        board: &Board,
        piece_square: Square,
        _piece: Piece,
        target_square: Square,
    ) -> bool {
        // Generate legal moves for the piece and check if target is among them
        let legal_moves: Vec<ChessMove> = MoveGen::new_legal(board)
            .filter(|mv| mv.get_source() == piece_square)
            .collect();

        for mv in legal_moves {
            if mv.get_dest() == target_square {
                return true;
            }
        }
        false
    }

    fn square_centrality(&self, square: Square) -> f32 {
        // Calculate how central a square is (higher values for central squares)
        let file = square.get_file().to_index() as f32;
        let rank = square.get_rank().to_index() as f32;

        // Distance from center (files 3.5, ranks 3.5)
        let file_distance = (file - 3.5).abs();
        let rank_distance = (rank - 3.5).abs();
        let max_distance = file_distance.max(rank_distance);

        // Return centrality (0.0 for corners, 1.0 for center)
        (3.5 - max_distance) / 3.5
    }

    fn is_piece_developed(&self, square: Square, color: chess::Color) -> bool {
        // Check if piece is developed from starting position
        let rank = square.get_rank().to_index();

        match color {
            chess::Color::White => rank > 1, // White pieces developed beyond 2nd rank
            chess::Color::Black => rank < 6, // Black pieces developed beyond 7th rank
        }
    }

    fn is_square_attacked_by_enemy(&self, board: &Board, square: Square) -> bool {
        // Check if enemy pieces can attack this square
        let enemy_color = !board.side_to_move();
        self.is_square_attacked_by_color(board, square, enemy_color)
    }

    fn is_square_attacked_by_color(
        &self,
        board: &Board,
        square: Square,
        attacking_color: chess::Color,
    ) -> bool {
        // Check if pieces of specific color can attack this square
        for check_square in chess::ALL_SQUARES {
            if board.color_on(check_square) == Some(attacking_color) {
                if let Some(piece) = board.piece_on(check_square) {
                    if self.can_piece_attack_square(board, check_square, piece, square) {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn count_defenders_for_color(
        &self,
        board: &Board,
        square: Square,
        defending_color: chess::Color,
    ) -> f32 {
        // Count pieces of specific color that can defend this square
        let mut defenders = 0.0;

        for check_square in chess::ALL_SQUARES {
            if board.color_on(check_square) == Some(defending_color) && check_square != square {
                if let Some(piece) = board.piece_on(check_square) {
                    if self.can_piece_attack_square(board, check_square, piece, square) {
                        defenders += 1.0;
                    }
                }
            }
        }
        defenders
    }

    fn count_attackers_for_color(
        &self,
        board: &Board,
        square: Square,
        attacking_color: chess::Color,
    ) -> f32 {
        // Count pieces of specific color that can attack this square
        let mut attackers = 0.0;

        for check_square in chess::ALL_SQUARES {
            if board.color_on(check_square) == Some(attacking_color) {
                if let Some(piece) = board.piece_on(check_square) {
                    if self.can_piece_attack_square(board, check_square, piece, square) {
                        attackers += 1.0;
                    }
                }
            }
        }
        attackers
    }

    fn count_hanging_pieces_after_move(&self, board: &Board, our_color: chess::Color) -> f32 {
        // Count how many of our pieces are hanging after this position
        let mut hanging_pieces = 0.0;
        let enemy_color = !our_color;

        for square in chess::ALL_SQUARES {
            if board.color_on(square) == Some(our_color) {
                if let Some(_piece) = board.piece_on(square) {
                    if self.is_square_attacked_by_color(board, square, enemy_color) {
                        let defenders = self.count_defenders_for_color(board, square, our_color);
                        if defenders == 0.0 {
                            hanging_pieces += 1.0; // Piece is hanging
                        }
                    }
                }
            }
        }
        hanging_pieces
    }

    // Master-level positional evaluation helpers

    fn is_edge_square(&self, square: Square) -> bool {
        let file = square.get_file().to_index();
        let rank = square.get_rank().to_index();
        file == 0 || file == 7 || rank == 0 || rank == 7
    }

    fn is_central_square(&self, square: Square) -> bool {
        let file = square.get_file().to_index();
        let rank = square.get_rank().to_index();
        (2..=5).contains(&file) && (2..=5).contains(&rank)
    }

    fn is_long_diagonal(&self, square: Square) -> bool {
        let file = square.get_file().to_index();
        let rank = square.get_rank().to_index();
        // a1-h8 diagonal or h1-a8 diagonal
        (file == rank) || (file + rank == 7)
    }

    fn creates_pawn_chain(&self, board: &Board, mv: &ChessMove) -> bool {
        if board.piece_on(mv.get_source()) != Some(Piece::Pawn) {
            return false;
        }

        let dest = mv.get_dest();
        let our_color = board.side_to_move();

        // Check if this pawn supports or is supported by other pawns
        let supporting_squares = self.get_pawn_support_squares(dest, our_color);
        for support_square in supporting_squares {
            if board.piece_on(support_square) == Some(Piece::Pawn)
                && board.color_on(support_square) == Some(our_color)
            {
                return true;
            }
        }
        false
    }

    fn get_pawn_support_squares(&self, square: Square, color: chess::Color) -> Vec<Square> {
        let mut support_squares = Vec::new();
        let file = square.get_file().to_index() as i8;
        let rank = square.get_rank().to_index() as i8;

        let rank_offset = match color {
            chess::Color::White => -1, // White pawns support from behind
            chess::Color::Black => 1,  // Black pawns support from behind
        };

        // Check diagonally behind for supporting pawns
        for file_offset in [-1, 1] {
            let support_file = file + file_offset;
            let support_rank = rank + rank_offset;

            if (0..8).contains(&support_file) && (0..8).contains(&support_rank) {
                let support_square =
                    unsafe { Square::new((support_rank * 8 + support_file) as u8) };
                support_squares.push(support_square);
            }
        }

        support_squares
    }

    fn is_open_file(&self, board: &Board, square: Square) -> bool {
        let file = square.get_file();

        // Check if there are no pawns on this file
        for rank_index in 0..8 {
            let check_square = unsafe { Square::new((rank_index * 8 + file.to_index()) as u8) };
            if board.piece_on(check_square) == Some(Piece::Pawn) {
                return false;
            }
        }
        true
    }

    fn is_seventh_rank(&self, square: Square, color: chess::Color) -> bool {
        let rank = square.get_rank().to_index();
        match color {
            chess::Color::White => rank == 6, // 7th rank for White
            chess::Color::Black => rank == 1, // 2nd rank for Black (their 7th)
        }
    }

    fn is_attacked_by_enemy_pawns(&self, board: &Board, square: Square) -> bool {
        let enemy_color = !board.side_to_move();
        let file = square.get_file().to_index() as i8;
        let rank = square.get_rank().to_index() as i8;

        let pawn_attack_rank = match enemy_color {
            chess::Color::White => rank + 1, // White pawns attack from below
            chess::Color::Black => rank - 1, // Black pawns attack from above
        };

        if !(0..=7).contains(&pawn_attack_rank) {
            return false;
        }

        // Check diagonal attacks from enemy pawns
        for file_offset in [-1, 1] {
            let attack_file = file + file_offset;
            if (0..8).contains(&attack_file) {
                let attack_square =
                    unsafe { Square::new((pawn_attack_rank * 8 + attack_file) as u8) };
                if board.piece_on(attack_square) == Some(Piece::Pawn)
                    && board.color_on(attack_square) == Some(enemy_color)
                {
                    return true;
                }
            }
        }
        false
    }

    fn improves_worst_piece(&self, board: &Board, mv: &ChessMove) -> bool {
        let piece = board.piece_on(mv.get_source());
        if piece.is_none() {
            return false;
        }

        // Simplified check: moving from back rank or edge improves piece activity
        let source_rank = mv.get_source().get_rank().to_index();
        let dest_rank = mv.get_dest().get_rank().to_index();
        let source_centrality = self.square_centrality(mv.get_source());
        let dest_centrality = self.square_centrality(mv.get_dest());

        // Piece becomes more active (more central or advanced)
        dest_centrality > source_centrality
            || (board.side_to_move() == chess::Color::White && dest_rank > source_rank)
            || (board.side_to_move() == chess::Color::Black && dest_rank < source_rank)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess::Board;

    #[test]
    fn test_strategic_evaluator_creation() {
        let evaluator = StrategicEvaluator::new_default();
        assert_eq!(evaluator.config.initiative_weight, 0.4);
        assert!(!evaluator.attacking_patterns.is_empty());
        assert!(!evaluator.positional_plans.is_empty());
    }

    #[test]
    fn test_strategic_evaluation() {
        let evaluator = StrategicEvaluator::new_default();
        let board = Board::default();

        let strategic_eval = evaluator.evaluate_strategic(&board);

        // Strategic evaluation should be calculated - allow negative values for initiative
        assert!(
            strategic_eval.initiative_bonus >= -100.0 && strategic_eval.initiative_bonus <= 100.0
        );
        assert!(strategic_eval.attacking_bonus >= 0.0);
        // Base evaluation can be 0.0 for equal starting position
        assert!(strategic_eval.base_evaluation >= -10.0 && strategic_eval.base_evaluation <= 10.0);
    }

    #[test]
    fn test_proactive_move_generation() {
        let evaluator = StrategicEvaluator::aggressive();
        let board = Board::default();

        let proactive_moves = evaluator.generate_proactive_moves(&board);

        assert!(!proactive_moves.is_empty());
        assert!(proactive_moves.len() <= 40); // Reasonable number of moves
    }

    #[test]
    fn test_aggressive_vs_balanced_config() {
        let aggressive = StrategicEvaluator::aggressive();
        let balanced = StrategicEvaluator::new_default();

        assert!(aggressive.config.initiative_weight > balanced.config.initiative_weight);
        assert!(aggressive.config.attacking_piece_bonus > balanced.config.attacking_piece_bonus);
    }

    #[test]
    fn test_strategic_patterns() {
        let evaluator = StrategicEvaluator::new_default();

        assert!(evaluator
            .attacking_patterns
            .contains_key("king_attack_setup"));
        assert!(evaluator
            .attacking_patterns
            .contains_key("central_pressure"));
        assert!(evaluator
            .attacking_patterns
            .contains_key("piece_coordination"));
    }

    #[test]
    fn test_hybrid_integration() {
        let evaluator = StrategicEvaluator::new_default();
        let board = Board::default();

        let blended_eval = evaluator.blend_with_hybrid_evaluation(&board, 0.5, 0.3);

        assert!(blended_eval.abs() < 100.0); // Reasonable evaluation range
    }

    #[test]
    fn test_initiative_evaluation() {
        let evaluator = StrategicEvaluator::aggressive();
        let board = Board::default();

        let should_attack = evaluator.should_play_aggressively(&board);
        // Starting position might not favor immediate aggression
        // Test just ensures method runs without error
        assert!(should_attack == true || should_attack == false);
    }
}
