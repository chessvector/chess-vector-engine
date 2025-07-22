use crate::errors::Result;
use crate::hybrid_evaluation::{EvaluationComponent, StrategicEvaluator};
use crate::utils::cache::EvaluationCache;
use chess::{Board, ChessMove, Color, Piece, Rank, Square};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Strategic initiative evaluation system that analyzes long-term positional factors
pub struct StrategicInitiativeEvaluator {
    /// Initiative analyzer
    initiative_analyzer: InitiativeAnalyzer,
    /// Positional pressure evaluator
    pressure_evaluator: PositionalPressureEvaluator,
    /// Strategic plan generator
    plan_generator: StrategicPlanGenerator,
    /// Time pressure analyzer
    time_analyzer: TimePressureAnalyzer,
    /// Initiative cache
    initiative_cache: Arc<EvaluationCache>,
}

impl StrategicInitiativeEvaluator {
    /// Create a new strategic initiative evaluator
    pub fn new() -> Self {
        Self {
            initiative_analyzer: InitiativeAnalyzer::new(),
            pressure_evaluator: PositionalPressureEvaluator::new(),
            plan_generator: StrategicPlanGenerator::new(),
            time_analyzer: TimePressureAnalyzer::new(),
            initiative_cache: Arc::new(EvaluationCache::new(
                5000,
                std::time::Duration::from_secs(600),
            )),
        }
    }

    /// Evaluate strategic initiative for a position
    pub fn evaluate_strategic_initiative(
        &self,
        board: &Board,
    ) -> Result<StrategicInitiativeResult> {
        let fen = board.to_string();

        // Check cache first
        if let Some(cached_eval) = self.initiative_cache.get_evaluation(&fen) {
            return Ok(StrategicInitiativeResult {
                initiative_score: cached_eval,
                white_initiative: 0.0,
                black_initiative: 0.0,
                positional_pressure: PositionalPressure::default(),
                strategic_plans: Vec::new(),
                time_pressure: TimePressure::default(),
                initiative_factors: InitiativeFactors::default(),
                from_cache: true,
            });
        }

        let start_time = std::time::Instant::now();

        // Analyze initiative for both sides
        let white_initiative = self
            .initiative_analyzer
            .analyze_initiative(board, Color::White)?;
        let black_initiative = self
            .initiative_analyzer
            .analyze_initiative(board, Color::Black)?;

        // Evaluate positional pressure
        let positional_pressure = self.pressure_evaluator.evaluate_pressure(board)?;

        // Generate strategic plans
        let strategic_plans = self.plan_generator.generate_plans(board, 3)?;

        // Analyze time pressure factors
        let time_pressure = self.time_analyzer.analyze_time_factors(board)?;

        // Compute overall initiative score (positive = white advantage)
        let initiative_score = white_initiative.total_score - black_initiative.total_score;

        let result = StrategicInitiativeResult {
            initiative_score,
            white_initiative: white_initiative.total_score,
            black_initiative: black_initiative.total_score,
            positional_pressure,
            strategic_plans,
            time_pressure,
            initiative_factors: InitiativeFactors {
                white: white_initiative,
                black: black_initiative,
            },
            from_cache: false,
        };

        // Cache the result
        self.initiative_cache
            .store_evaluation(&fen, initiative_score);

        Ok(result)
    }

    /// Get strategic initiative statistics
    pub fn get_statistics(&self) -> StrategicInitiativeStats {
        let cache_stats = self.initiative_cache.stats();

        StrategicInitiativeStats {
            evaluations_performed: 0, // Would need to track this
            cache_hit_ratio: cache_stats.hit_ratio,
            average_evaluation_time_ms: 0.0, // Would need to compute this
            plans_generated: 0,              // Would need to track this
        }
    }
}

impl Default for StrategicInitiativeEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategicEvaluator for StrategicInitiativeEvaluator {
    fn evaluate_position(&self, board: &Board) -> Result<EvaluationComponent> {
        let start_time = std::time::Instant::now();
        let initiative_result = self.evaluate_strategic_initiative(board)?;
        let computation_time = start_time.elapsed().as_millis() as u64;

        let mut additional_info = HashMap::new();
        additional_info.insert(
            "white_initiative".to_string(),
            initiative_result.white_initiative,
        );
        additional_info.insert(
            "black_initiative".to_string(),
            initiative_result.black_initiative,
        );
        additional_info.insert(
            "space_advantage".to_string(),
            initiative_result.positional_pressure.space_advantage,
        );
        additional_info.insert(
            "development_advantage".to_string(),
            initiative_result.positional_pressure.development_advantage,
        );
        additional_info.insert(
            "attacking_potential".to_string(),
            initiative_result.positional_pressure.attacking_potential,
        );
        additional_info.insert(
            "strategic_plans_count".to_string(),
            initiative_result.strategic_plans.len() as f32,
        );

        // Compute confidence based on initiative clarity
        let confidence = self.compute_initiative_confidence(&initiative_result);

        Ok(EvaluationComponent {
            evaluation: initiative_result.initiative_score,
            confidence,
            computation_time_ms: computation_time,
            additional_info,
        })
    }
}

impl StrategicInitiativeEvaluator {
    fn compute_initiative_confidence(&self, result: &StrategicInitiativeResult) -> f32 {
        let mut confidence_factors = Vec::new();

        // Factor 1: Initiative clarity (larger difference = higher confidence)
        let initiative_clarity = result.initiative_score.abs().min(1.0);
        confidence_factors.push(initiative_clarity);

        // Factor 2: Number of strategic plans found
        let plan_confidence = (result.strategic_plans.len() as f32 / 5.0).min(1.0);
        confidence_factors.push(plan_confidence);

        // Factor 3: Positional pressure consistency
        let pressure_consistency = (result.positional_pressure.space_advantage.abs()
            + result.positional_pressure.development_advantage.abs()
            + result.positional_pressure.attacking_potential.abs())
            / 3.0;
        confidence_factors.push(pressure_consistency.min(1.0));

        // Factor 4: Time pressure clarity
        let time_clarity = if result.time_pressure.critical_moves.is_empty() {
            0.7
        } else {
            0.9
        };
        confidence_factors.push(time_clarity);

        // Average confidence
        if confidence_factors.is_empty() {
            0.5
        } else {
            confidence_factors.iter().sum::<f32>() / confidence_factors.len() as f32
        }
    }
}

/// Initiative analyzer for individual colors
pub struct InitiativeAnalyzer {
    cached_analyses: RwLock<HashMap<String, ColorInitiativeAnalysis>>,
}

impl InitiativeAnalyzer {
    pub fn new() -> Self {
        Self {
            cached_analyses: RwLock::new(HashMap::new()),
        }
    }

    /// Analyze initiative for a specific color
    pub fn analyze_initiative(
        &self,
        board: &Board,
        color: Color,
    ) -> Result<ColorInitiativeAnalysis> {
        let cache_key = format!("{}_{:?}", board.to_string(), color);

        // Check cache
        if let Ok(cache) = self.cached_analyses.read() {
            if let Some(analysis) = cache.get(&cache_key) {
                return Ok(analysis.clone());
            }
        }

        let mut analysis = ColorInitiativeAnalysis::default();

        // Analyze development initiative
        analysis.development_initiative = self.analyze_development_initiative(board, color);

        // Analyze attacking initiative
        analysis.attacking_initiative = self.analyze_attacking_initiative(board, color);

        // Analyze space initiative
        analysis.space_initiative = self.analyze_space_initiative(board, color);

        // Analyze tempo initiative
        analysis.tempo_initiative = self.analyze_tempo_initiative(board, color);

        // Analyze pawn initiative
        analysis.pawn_initiative = self.analyze_pawn_initiative(board, color);

        // Analyze coordination initiative
        analysis.coordination_initiative = self.analyze_coordination_initiative(board, color);

        // Compute total initiative score
        analysis.total_score = analysis.development_initiative * 0.2
            + analysis.attacking_initiative * 0.25
            + analysis.space_initiative * 0.15
            + analysis.tempo_initiative * 0.1
            + analysis.pawn_initiative * 0.15
            + analysis.coordination_initiative * 0.15;

        // Cache the result with LRU-style management
        if let Ok(mut cache) = self.cached_analyses.write() {
            if cache.len() > 1000 {
                // Remove oldest 50% of entries to maintain performance
                let keys_to_remove: Vec<_> = cache.keys().take(500).cloned().collect();
                for key in keys_to_remove {
                    cache.remove(&key);
                }
            }
            cache.insert(cache_key, analysis.clone());
        }

        Ok(analysis)
    }

    fn analyze_development_initiative(&self, board: &Board, color: Color) -> f32 {
        let pieces = board.color_combined(color);
        let mut development_score = 0.0;

        // Count developed pieces
        let developed_knights = (board.pieces(Piece::Knight) & pieces)
            .into_iter()
            .filter(|&square| self.is_piece_developed(square, Piece::Knight, color))
            .count();

        let developed_bishops = (board.pieces(Piece::Bishop) & pieces)
            .into_iter()
            .filter(|&square| self.is_piece_developed(square, Piece::Bishop, color))
            .count();

        development_score += developed_knights as f32 * 0.3;
        development_score += developed_bishops as f32 * 0.3;

        // Castling bonus
        if board.castle_rights(color).has_kingside() || board.castle_rights(color).has_queenside() {
            development_score += 0.2;
        }

        // Central control bonus
        let central_squares = [Square::E4, Square::E5, Square::D4, Square::D5];
        let central_control = central_squares
            .iter()
            .filter(|&&square| self.attacks_square(board, color, square))
            .count();

        development_score += central_control as f32 * 0.1;

        development_score.min(1.0)
    }

    fn analyze_attacking_initiative(&self, board: &Board, color: Color) -> f32 {
        let opponent_color = !color;
        let opponent_king_square = board.king_square(opponent_color);
        let mut attacking_score = 0.0;

        // King safety pressure
        let king_attackers = self.count_attackers_near_king(board, color, opponent_king_square);
        attacking_score += king_attackers as f32 * 0.2;

        // Piece activity in opponent territory
        let active_pieces = self.count_active_pieces_in_enemy_territory(board, color);
        attacking_score += active_pieces as f32 * 0.15;

        // Tactical threats
        let threats = self.count_tactical_threats(board, color);
        attacking_score += threats as f32 * 0.25;

        // Weak square control
        let weak_squares_controlled = self.count_weak_squares_controlled(board, color);
        attacking_score += weak_squares_controlled as f32 * 0.1;

        attacking_score.min(1.0)
    }

    fn analyze_space_initiative(&self, board: &Board, color: Color) -> f32 {
        let mut space_score = 0.0;

        // Count squares controlled
        let squares_controlled = self.count_controlled_squares(board, color);
        space_score += (squares_controlled as f32 / 64.0) * 0.5;

        // Advanced pawn structure
        let advanced_pawns = self.count_advanced_pawns(board, color);
        space_score += advanced_pawns as f32 * 0.1;

        // Outposts
        let outposts = self.count_outposts(board, color);
        space_score += outposts as f32 * 0.2;

        space_score.min(1.0)
    }

    fn analyze_tempo_initiative(&self, board: &Board, color: Color) -> f32 {
        let mut tempo_score = 0.0;

        // Check if it's this color's turn
        if board.side_to_move() == color {
            tempo_score += 0.1;
        }

        // Count forcing moves available
        let forcing_moves = self.count_forcing_moves(board, color);
        tempo_score += forcing_moves as f32 * 0.05;

        // Check for opponent pieces under attack
        let pieces_under_attack = self.count_opponent_pieces_under_attack(board, color);
        tempo_score += pieces_under_attack as f32 * 0.1;

        tempo_score.min(1.0)
    }

    fn analyze_pawn_initiative(&self, board: &Board, color: Color) -> f32 {
        let mut pawn_score = 0.0;

        // Passed pawns
        let passed_pawns = self.count_passed_pawns(board, color);
        pawn_score += passed_pawns as f32 * 0.2;

        // Pawn chains
        let pawn_chains = self.count_pawn_chains(board, color);
        pawn_score += pawn_chains as f32 * 0.1;

        // Pawn storms
        let pawn_storms = self.evaluate_pawn_storms(board, color);
        pawn_score += pawn_storms * 0.3;

        pawn_score.min(1.0)
    }

    fn analyze_coordination_initiative(&self, board: &Board, color: Color) -> f32 {
        let mut coordination_score = 0.0;

        // Piece coordination
        let coordinated_pieces = self.count_coordinated_pieces(board, color);
        coordination_score += coordinated_pieces as f32 * 0.1;

        // Battery formations (rook+queen, bishop+queen)
        let batteries = self.count_batteries(board, color);
        coordination_score += batteries as f32 * 0.2;

        // Piece harmony (pieces working together)
        let harmony_score = self.evaluate_piece_harmony(board, color);
        coordination_score += harmony_score * 0.3;

        coordination_score.min(1.0)
    }

    // Helper methods (simplified implementations for now)
    fn is_piece_developed(&self, _square: Square, _piece: Piece, _color: Color) -> bool {
        // Simplified - in practice would check if piece is on starting square
        true
    }

    fn attacks_square(&self, _board: &Board, _color: Color, _square: Square) -> bool {
        // Simplified implementation
        false
    }

    fn count_attackers_near_king(&self, _board: &Board, _color: Color, _king_square: Square) -> u8 {
        // Simplified implementation
        0
    }

    fn count_active_pieces_in_enemy_territory(&self, _board: &Board, _color: Color) -> u8 {
        // Simplified implementation
        0
    }

    fn count_tactical_threats(&self, _board: &Board, _color: Color) -> u8 {
        // Simplified implementation
        0
    }

    fn count_weak_squares_controlled(&self, _board: &Board, _color: Color) -> u8 {
        // Simplified implementation
        0
    }

    fn count_controlled_squares(&self, _board: &Board, _color: Color) -> u8 {
        // Simplified implementation
        20
    }

    fn count_advanced_pawns(&self, board: &Board, color: Color) -> u8 {
        let pawns = board.pieces(Piece::Pawn) & board.color_combined(color);
        let mut count = 0;

        for square in pawns {
            let rank = square.get_rank();
            let advanced = match color {
                Color::White => rank >= Rank::Fifth,
                Color::Black => rank <= Rank::Fourth,
            };

            if advanced {
                count += 1;
            }
        }

        count
    }

    fn count_outposts(&self, _board: &Board, _color: Color) -> u8 {
        // Simplified implementation
        0
    }

    fn count_forcing_moves(&self, _board: &Board, _color: Color) -> u8 {
        // Simplified implementation
        0
    }

    fn count_opponent_pieces_under_attack(&self, _board: &Board, _color: Color) -> u8 {
        // Simplified implementation
        0
    }

    fn count_passed_pawns(&self, _board: &Board, _color: Color) -> u8 {
        // Simplified implementation
        0
    }

    fn count_pawn_chains(&self, _board: &Board, _color: Color) -> u8 {
        // Simplified implementation
        0
    }

    fn evaluate_pawn_storms(&self, _board: &Board, _color: Color) -> f32 {
        // Simplified implementation
        0.0
    }

    fn count_coordinated_pieces(&self, _board: &Board, _color: Color) -> u8 {
        // Simplified implementation
        0
    }

    fn count_batteries(&self, _board: &Board, _color: Color) -> u8 {
        // Simplified implementation
        0
    }

    fn evaluate_piece_harmony(&self, _board: &Board, _color: Color) -> f32 {
        // Simplified implementation
        0.0
    }
}

/// Positional pressure evaluator
pub struct PositionalPressureEvaluator;

impl PositionalPressureEvaluator {
    pub fn new() -> Self {
        Self
    }

    pub fn evaluate_pressure(&self, board: &Board) -> Result<PositionalPressure> {
        let mut pressure = PositionalPressure::default();

        // Evaluate space advantage
        pressure.space_advantage = self.evaluate_space_advantage(board);

        // Evaluate development advantage
        pressure.development_advantage = self.evaluate_development_advantage(board);

        // Evaluate attacking potential
        pressure.attacking_potential = self.evaluate_attacking_potential(board);

        // Evaluate positional weaknesses
        pressure.positional_weaknesses = self.evaluate_positional_weaknesses(board);

        Ok(pressure)
    }

    fn evaluate_space_advantage(&self, _board: &Board) -> f32 {
        // Simplified implementation
        0.0
    }

    fn evaluate_development_advantage(&self, _board: &Board) -> f32 {
        // Simplified implementation
        0.0
    }

    fn evaluate_attacking_potential(&self, _board: &Board) -> f32 {
        // Simplified implementation
        0.0
    }

    fn evaluate_positional_weaknesses(&self, _board: &Board) -> f32 {
        // Simplified implementation
        0.0
    }
}

/// Strategic plan generator
pub struct StrategicPlanGenerator;

impl StrategicPlanGenerator {
    pub fn new() -> Self {
        Self
    }

    pub fn generate_plans(&self, board: &Board, max_plans: usize) -> Result<Vec<StrategicPlan>> {
        let mut plans = Vec::new();

        // Generate different types of strategic plans
        plans.extend(self.generate_attacking_plans(board)?);
        plans.extend(self.generate_positional_plans(board)?);
        plans.extend(self.generate_endgame_plans(board)?);

        // Sort by priority and take top plans
        plans.sort_by(|a, b| {
            b.priority
                .partial_cmp(&a.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        plans.truncate(max_plans);

        Ok(plans)
    }

    fn generate_attacking_plans(&self, _board: &Board) -> Result<Vec<StrategicPlan>> {
        // Simplified implementation
        Ok(vec![StrategicPlan {
            plan_type: StrategicPlanType::KingsideAttack,
            priority: 0.8,
            key_moves: Vec::new(),
            target_squares: Vec::new(),
            expected_outcome: PlanOutcome::MaterialGain,
        }])
    }

    fn generate_positional_plans(&self, _board: &Board) -> Result<Vec<StrategicPlan>> {
        // Simplified implementation
        Ok(vec![StrategicPlan {
            plan_type: StrategicPlanType::CentralControl,
            priority: 0.6,
            key_moves: Vec::new(),
            target_squares: Vec::new(),
            expected_outcome: PlanOutcome::PositionalAdvantage,
        }])
    }

    fn generate_endgame_plans(&self, _board: &Board) -> Result<Vec<StrategicPlan>> {
        // Simplified implementation
        Ok(Vec::new())
    }
}

/// Time pressure analyzer
pub struct TimePressureAnalyzer;

impl TimePressureAnalyzer {
    pub fn new() -> Self {
        Self
    }

    pub fn analyze_time_factors(&self, board: &Board) -> Result<TimePressure> {
        let mut time_pressure = TimePressure::default();

        // Identify critical moves that must be played soon
        time_pressure.critical_moves = self.identify_critical_moves(board)?;

        // Evaluate zugzwang potential
        time_pressure.zugzwang_potential = self.evaluate_zugzwang_potential(board);

        // Evaluate tempo importance
        time_pressure.tempo_importance = self.evaluate_tempo_importance(board);

        Ok(time_pressure)
    }

    fn identify_critical_moves(&self, _board: &Board) -> Result<Vec<ChessMove>> {
        // Simplified implementation
        Ok(Vec::new())
    }

    fn evaluate_zugzwang_potential(&self, _board: &Board) -> f32 {
        // Simplified implementation
        0.0
    }

    fn evaluate_tempo_importance(&self, _board: &Board) -> f32 {
        // Simplified implementation
        0.5
    }
}

/// Strategic initiative evaluation result
#[derive(Debug, Clone)]
pub struct StrategicInitiativeResult {
    /// Overall initiative score (positive = white advantage)
    pub initiative_score: f32,
    /// White's initiative score
    pub white_initiative: f32,
    /// Black's initiative score
    pub black_initiative: f32,
    /// Positional pressure analysis
    pub positional_pressure: PositionalPressure,
    /// Generated strategic plans
    pub strategic_plans: Vec<StrategicPlan>,
    /// Time pressure factors
    pub time_pressure: TimePressure,
    /// Detailed initiative factors
    pub initiative_factors: InitiativeFactors,
    /// Whether result came from cache
    pub from_cache: bool,
}

/// Initiative analysis for a single color
#[derive(Debug, Clone, Default)]
pub struct ColorInitiativeAnalysis {
    pub development_initiative: f32,
    pub attacking_initiative: f32,
    pub space_initiative: f32,
    pub tempo_initiative: f32,
    pub pawn_initiative: f32,
    pub coordination_initiative: f32,
    pub total_score: f32,
}

/// Initiative factors for both colors
#[derive(Debug, Clone, Default)]
pub struct InitiativeFactors {
    pub white: ColorInitiativeAnalysis,
    pub black: ColorInitiativeAnalysis,
}

/// Positional pressure analysis
#[derive(Debug, Clone, Default)]
pub struct PositionalPressure {
    pub space_advantage: f32,
    pub development_advantage: f32,
    pub attacking_potential: f32,
    pub positional_weaknesses: f32,
}

/// Strategic plan
#[derive(Debug, Clone)]
pub struct StrategicPlan {
    pub plan_type: StrategicPlanType,
    pub priority: f32,
    pub key_moves: Vec<ChessMove>,
    pub target_squares: Vec<Square>,
    pub expected_outcome: PlanOutcome,
}

/// Types of strategic plans
#[derive(Debug, Clone)]
pub enum StrategicPlanType {
    KingsideAttack,
    QueensideAttack,
    CentralControl,
    PawnStorm,
    PieceManeuver,
    EndgameTransition,
}

/// Expected outcome of a strategic plan
#[derive(Debug, Clone)]
pub enum PlanOutcome {
    MaterialGain,
    PositionalAdvantage,
    KingAttack,
    Promotion,
    Draw,
}

/// Time pressure factors
#[derive(Debug, Clone, Default)]
pub struct TimePressure {
    pub critical_moves: Vec<ChessMove>,
    pub zugzwang_potential: f32,
    pub tempo_importance: f32,
}

/// Strategic initiative statistics
#[derive(Debug, Clone)]
pub struct StrategicInitiativeStats {
    pub evaluations_performed: u64,
    pub cache_hit_ratio: f64,
    pub average_evaluation_time_ms: f64,
    pub plans_generated: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess::Board;
    use std::str::FromStr;

    #[test]
    fn test_strategic_initiative_evaluator() {
        let evaluator = StrategicInitiativeEvaluator::new();
        let board =
            Board::from_str("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();

        let result = evaluator.evaluate_strategic_initiative(&board).unwrap();
        assert!(result.initiative_score.is_finite());
        assert!(result.white_initiative >= 0.0);
        assert!(result.black_initiative >= 0.0);
        assert!(!result.from_cache);
    }

    #[test]
    fn test_initiative_analyzer() {
        let analyzer = InitiativeAnalyzer::new();
        let board =
            Board::from_str("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();

        let white_analysis = analyzer.analyze_initiative(&board, Color::White).unwrap();
        let black_analysis = analyzer.analyze_initiative(&board, Color::Black).unwrap();

        assert!(white_analysis.total_score.is_finite());
        assert!(black_analysis.total_score.is_finite());
        assert!(white_analysis.development_initiative >= 0.0);
        assert!(black_analysis.development_initiative >= 0.0);
    }

    #[test]
    fn test_strategic_evaluator_trait() {
        let evaluator = StrategicInitiativeEvaluator::new();
        let board =
            Board::from_str("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();

        let evaluation = evaluator.evaluate_position(&board).unwrap();
        assert!(evaluation.evaluation.is_finite());
        assert!(evaluation.confidence >= 0.0 && evaluation.confidence <= 1.0);
        assert!(evaluation.computation_time_ms > 0);
        assert!(!evaluation.additional_info.is_empty());
    }

    #[test]
    fn test_strategic_plan_generator() {
        let generator = StrategicPlanGenerator::new();
        let board =
            Board::from_str("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();

        let plans = generator.generate_plans(&board, 3).unwrap();
        assert!(plans.len() <= 3);

        for plan in &plans {
            assert!(plan.priority >= 0.0 && plan.priority <= 1.0);
        }
    }

    #[test]
    fn test_positional_pressure_evaluator() {
        let evaluator = PositionalPressureEvaluator::new();
        let board =
            Board::from_str("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();

        let pressure = evaluator.evaluate_pressure(&board).unwrap();
        assert!(pressure.space_advantage.is_finite());
        assert!(pressure.development_advantage.is_finite());
        assert!(pressure.attacking_potential.is_finite());
        assert!(pressure.positional_weaknesses.is_finite());
    }

    #[test]
    fn test_time_pressure_analyzer() {
        let analyzer = TimePressureAnalyzer::new();
        let board =
            Board::from_str("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();

        let time_pressure = analyzer.analyze_time_factors(&board).unwrap();
        assert!(time_pressure.zugzwang_potential >= 0.0);
        assert!(time_pressure.tempo_importance >= 0.0);
    }
}
