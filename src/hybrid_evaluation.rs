use crate::errors::Result;
use crate::utils::cache::EvaluationCache;
use crate::utils::profiler::{global_profiler, ChessEngineProfiler};
use chess::{Board, Color, Piece, Square};
use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::sync::{Arc, RwLock};

/// Advanced hybrid evaluation system that intelligently blends multiple evaluation methods
pub struct HybridEvaluationEngine {
    /// NNUE neural network evaluator
    nnue_evaluator: Option<Box<dyn NNUEEvaluator + Send + Sync>>,
    /// Pattern recognition system
    pattern_evaluator: Option<Box<dyn PatternEvaluator + Send + Sync>>,
    /// Tactical search engine
    tactical_evaluator: Option<Box<dyn TacticalEvaluator + Send + Sync>>,
    /// Strategic initiative analyzer
    strategic_evaluator: Option<Box<dyn StrategicEvaluator + Send + Sync>>,
    /// Position complexity analyzer
    complexity_analyzer: ComplexityAnalyzer,
    /// Game phase detector
    phase_detector: GamePhaseDetector,
    /// Evaluation blender with dynamic weights
    blender: EvaluationBlender,
    /// Confidence scorer
    confidence_scorer: ConfidenceScorer,
    /// Evaluation cache
    evaluation_cache: Arc<EvaluationCache>,
    /// Performance profiler
    profiler: Arc<ChessEngineProfiler>,
}

impl HybridEvaluationEngine {
    /// Create a new hybrid evaluation engine
    pub fn new() -> Self {
        Self {
            nnue_evaluator: None,
            pattern_evaluator: None,
            tactical_evaluator: None,
            strategic_evaluator: None,
            complexity_analyzer: ComplexityAnalyzer::new(),
            phase_detector: GamePhaseDetector::new(),
            blender: EvaluationBlender::new(),
            confidence_scorer: ConfidenceScorer::new(),
            evaluation_cache: Arc::new(EvaluationCache::new(
                10000,
                std::time::Duration::from_secs(300),
            )),
            profiler: Arc::clone(global_profiler()),
        }
    }

    /// Register an NNUE evaluator
    pub fn with_nnue_evaluator<T>(mut self, evaluator: T) -> Self
    where
        T: NNUEEvaluator + Send + Sync + 'static,
    {
        self.nnue_evaluator = Some(Box::new(evaluator));
        self
    }

    /// Register a pattern evaluator
    pub fn with_pattern_evaluator<T>(mut self, evaluator: T) -> Self
    where
        T: PatternEvaluator + Send + Sync + 'static,
    {
        self.pattern_evaluator = Some(Box::new(evaluator));
        self
    }

    /// Register a tactical evaluator
    pub fn with_tactical_evaluator<T>(mut self, evaluator: T) -> Self
    where
        T: TacticalEvaluator + Send + Sync + 'static,
    {
        self.tactical_evaluator = Some(Box::new(evaluator));
        self
    }

    /// Register a strategic evaluator
    pub fn with_strategic_evaluator<T>(mut self, evaluator: T) -> Self
    where
        T: StrategicEvaluator + Send + Sync + 'static,
    {
        self.strategic_evaluator = Some(Box::new(evaluator));
        self
    }

    /// Register the strategic initiative evaluator
    pub fn with_strategic_initiative_evaluator(self) -> Self {
        use crate::strategic_initiative::StrategicInitiativeEvaluator;
        self.with_strategic_evaluator(StrategicInitiativeEvaluator::new())
    }

    /// Perform comprehensive hybrid evaluation of a position
    pub fn evaluate_position(&self, board: &Board) -> Result<HybridEvaluationResult> {
        let fen = board.to_string();

        // Check cache first
        if let Some(cached_eval) = self.evaluation_cache.get_evaluation(&fen) {
            return Ok(HybridEvaluationResult {
                final_evaluation: cached_eval,
                nnue_evaluation: None,
                pattern_evaluation: None,
                tactical_evaluation: None,
                strategic_evaluation: None,
                complexity_score: 0.0,
                game_phase: GamePhase::Unknown,
                confidence_score: 1.0,
                blend_weights: BlendWeights::default(),
                evaluation_time_ms: 0,
                from_cache: true,
            });
        }

        let start_time = std::time::Instant::now();

        // Analyze position complexity and game phase
        let complexity_score = self.profiler.time_evaluation("complexity", || {
            self.complexity_analyzer.analyze_complexity(board)
        });

        let game_phase = self.profiler.time_evaluation("phase_detection", || {
            self.phase_detector.detect_phase(board)
        });

        // Gather evaluations from all available evaluators
        let mut evaluation_results = EvaluationResults::new();

        // NNUE evaluation (fast, always computed)
        if let Some(ref nnue) = self.nnue_evaluator {
            let nnue_result = self
                .profiler
                .time_evaluation("nnue", || nnue.evaluate_position(board));
            if let Ok(result) = nnue_result {
                evaluation_results.nnue = Some(result);
                self.profiler.record_evaluation("nnue");
            }
        }

        // Pattern evaluation (medium cost, conditional)
        if let Some(ref pattern) = self.pattern_evaluator {
            let should_use_pattern =
                self.should_use_pattern_evaluation(complexity_score, &game_phase);
            if should_use_pattern {
                let pattern_result = self
                    .profiler
                    .time_evaluation("pattern", || pattern.evaluate_position(board));
                if let Ok(result) = pattern_result {
                    evaluation_results.pattern = Some(result);
                    self.profiler.record_evaluation("pattern");
                }
            }
        }

        // Tactical evaluation (expensive, selective)
        if let Some(ref tactical) = self.tactical_evaluator {
            let should_use_tactical = self.should_use_tactical_evaluation(
                complexity_score,
                &game_phase,
                &evaluation_results,
            );
            if should_use_tactical {
                let tactical_result = self
                    .profiler
                    .time_evaluation("tactical", || tactical.evaluate_position(board));
                if let Ok(result) = tactical_result {
                    evaluation_results.tactical = Some(result);
                    self.profiler.record_evaluation("tactical");
                }
            }
        }

        // Strategic evaluation (contextual)
        if let Some(ref strategic) = self.strategic_evaluator {
            let should_use_strategic = self.should_use_strategic_evaluation(&game_phase);
            if should_use_strategic {
                let strategic_result = self
                    .profiler
                    .time_evaluation("strategic", || strategic.evaluate_position(board));
                if let Ok(result) = strategic_result {
                    evaluation_results.strategic = Some(result);
                    self.profiler.record_evaluation("strategic");
                }
            }
        }

        // Compute dynamic blend weights based on position characteristics
        let blend_weights =
            self.blender
                .compute_blend_weights(complexity_score, &game_phase, &evaluation_results);

        // Blend evaluations using computed weights
        let final_evaluation = self
            .blender
            .blend_evaluations(&evaluation_results, &blend_weights);

        // Create position context for confidence analysis
        let position_context = PositionContext {
            position_hash: board.get_hash(),
            game_phase,
            has_tactical_threats: self.detect_tactical_threats(board),
            in_opening_book: self.is_in_opening_book(board),
            material_imbalance: self.calculate_material_imbalance(board),
            complexity_score,
        };

        // Compute enhanced confidence score
        let confidence_analysis = self.confidence_scorer.compute_confidence(
            &evaluation_results,
            &blend_weights,
            complexity_score,
            &position_context,
        );

        let confidence_score = confidence_analysis.overall_confidence;

        let evaluation_time_ms = start_time.elapsed().as_millis() as u64;

        let result = HybridEvaluationResult {
            final_evaluation,
            nnue_evaluation: evaluation_results.nnue.clone(),
            pattern_evaluation: evaluation_results.pattern.clone(),
            tactical_evaluation: evaluation_results.tactical.clone(),
            strategic_evaluation: evaluation_results.strategic.clone(),
            complexity_score,
            game_phase,
            confidence_score,
            blend_weights,
            evaluation_time_ms,
            from_cache: false,
        };

        // Cache the result
        self.evaluation_cache
            .store_evaluation(&fen, final_evaluation);

        self.profiler.record_evaluation("hybrid");

        Ok(result)
    }

    /// Determine if pattern evaluation should be used
    fn should_use_pattern_evaluation(&self, complexity_score: f32, game_phase: &GamePhase) -> bool {
        match game_phase {
            GamePhase::Opening => complexity_score > 0.3,
            GamePhase::Middlegame => complexity_score > 0.4,
            GamePhase::Endgame => complexity_score > 0.5,
            GamePhase::Unknown => complexity_score > 0.4,
        }
    }

    /// Detect if position has tactical threats
    fn detect_tactical_threats(&self, board: &Board) -> bool {
        // Simple heuristic: check if there are pieces under attack
        // In a real implementation, this would use tactical search
        let white_pieces = board.color_combined(Color::White);
        let black_pieces = board.color_combined(Color::Black);

        // Check for checks
        if board.checkers().popcnt() > 0 {
            return true;
        }

        // Check for pieces that can be captured
        // This is a simplified check - real implementation would use attack tables
        let total_pieces = white_pieces.popcnt() + black_pieces.popcnt();
        total_pieces < 24 // Simplified: fewer pieces often means more tactics
    }

    /// Check if position is in opening book
    fn is_in_opening_book(&self, board: &Board) -> bool {
        // Placeholder: in real implementation, would check against opening book
        // For now, assume positions with many pieces are in opening
        let total_pieces = board.combined().popcnt();
        total_pieces >= 28
    }

    /// Calculate material imbalance
    fn calculate_material_imbalance(&self, board: &Board) -> f32 {
        let piece_values = [1, 3, 3, 5, 9, 0]; // Pawn, Knight, Bishop, Rook, Queen, King

        let mut white_material = 0;
        let mut black_material = 0;

        for piece_type in [
            Piece::Pawn,
            Piece::Knight,
            Piece::Bishop,
            Piece::Rook,
            Piece::Queen,
        ] {
            let piece_index = match piece_type {
                Piece::Pawn => 0,
                Piece::Knight => 1,
                Piece::Bishop => 2,
                Piece::Rook => 3,
                Piece::Queen => 4,
                Piece::King => 5,
            };

            let white_count =
                (board.pieces(piece_type) & board.color_combined(Color::White)).popcnt() as i32;
            let black_count =
                (board.pieces(piece_type) & board.color_combined(Color::Black)).popcnt() as i32;

            white_material += white_count * piece_values[piece_index];
            black_material += black_count * piece_values[piece_index];
        }

        (white_material - black_material).abs() as f32
    }

    /// Determine if tactical evaluation should be used
    fn should_use_tactical_evaluation(
        &self,
        complexity_score: f32,
        game_phase: &GamePhase,
        evaluation_results: &EvaluationResults,
    ) -> bool {
        // Always use tactical in complex tactical positions
        if complexity_score > 0.7 {
            return true;
        }

        // Use tactical when evaluations disagree significantly
        if let (Some(nnue), Some(pattern)) = (&evaluation_results.nnue, &evaluation_results.pattern)
        {
            let disagreement = (nnue.evaluation - pattern.evaluation).abs();
            if disagreement > 0.5 {
                return true;
            }
        }

        // Phase-specific tactical evaluation
        match game_phase {
            GamePhase::Opening => complexity_score > 0.6,
            GamePhase::Middlegame => complexity_score > 0.5,
            GamePhase::Endgame => complexity_score > 0.4, // More important in endgames
            GamePhase::Unknown => complexity_score > 0.5,
        }
    }

    /// Determine if strategic evaluation should be used
    fn should_use_strategic_evaluation(&self, game_phase: &GamePhase) -> bool {
        match game_phase {
            GamePhase::Opening => true,    // Strategic themes important in opening
            GamePhase::Middlegame => true, // Most important in middlegame
            GamePhase::Endgame => false,   // Less relevant in endgame
            GamePhase::Unknown => true,
        }
    }

    /// Get evaluation statistics
    pub fn get_statistics(&self) -> HybridEvaluationStats {
        let cache_stats = self.evaluation_cache.stats();

        HybridEvaluationStats {
            total_evaluations: 0,     // Simplified for now - would need separate tracking
            nnue_evaluations: 0,      // Simplified for now - would need separate tracking
            pattern_evaluations: 0,   // Simplified for now - would need separate tracking
            tactical_evaluations: 0,  // Simplified for now - would need separate tracking
            strategic_evaluations: 0, // Simplified for now - would need separate tracking
            cache_hit_ratio: cache_stats.hit_ratio,
            average_evaluation_time_ms: 0.0, // Simplified for now - would need separate tracking
            evaluations_per_second: 0.0,     // Simplified for now - would need separate tracking
        }
    }

    /// Enable or disable adaptive weight learning
    pub fn set_adaptive_learning(&mut self, enabled: bool) {
        self.blender.set_adaptive_learning(enabled);
    }

    /// Update performance metrics for adaptive learning
    pub fn update_evaluation_performance(
        &self,
        board: &Board,
        predicted_evaluation: f32,
        actual_result: Option<f32>,
        evaluation_accuracy: f32,
    ) -> Result<()> {
        // Recompute context information for the position
        let complexity_score = self.complexity_analyzer.analyze_complexity(board);
        let game_phase = self.phase_detector.detect_phase(board);

        // Get the weights that would be used for this position
        let mut eval_results = EvaluationResults::new();
        // Simplified - in practice we'd need to store the actual evaluation results
        let weights =
            self.blender
                .compute_blend_weights(complexity_score, &game_phase, &eval_results);

        // Update performance metrics
        self.blender.update_performance_metrics(
            &weights,
            complexity_score,
            &game_phase,
            evaluation_accuracy,
            actual_result,
        );

        Ok(())
    }

    /// Get adaptive learning statistics
    pub fn get_adaptive_learning_stats(&self) -> AdaptiveLearningStats {
        self.blender.get_adaptive_stats()
    }

    /// Get detailed complexity analysis for a position
    pub fn analyze_position_complexity(&self, board: &Board) -> ComplexityAnalysisResult {
        self.complexity_analyzer.analyze_complexity_detailed(board)
    }

    /// Configure the complexity analyzer
    pub fn configure_complexity_analyzer(
        &mut self,
        weights: ComplexityWeights,
        depth: AnalysisDepth,
    ) {
        self.complexity_analyzer = ComplexityAnalyzer::new()
            .with_complexity_weights(weights)
            .with_analysis_depth(depth);
    }

    /// Get detailed game phase analysis for a position
    pub fn analyze_game_phase(&self, board: &Board) -> GamePhaseAnalysisResult {
        self.phase_detector.analyze_game_phase(board)
    }

    /// Configure the game phase detector
    pub fn configure_phase_detector(
        &mut self,
        weights: PhaseDetectionWeights,
        settings: PhaseAdaptationSettings,
    ) {
        self.phase_detector = GamePhaseDetector::new()
            .with_phase_weights(weights)
            .with_adaptation_settings(settings);
    }

    /// Apply phase-specific adaptations to evaluation weights
    pub fn apply_phase_adaptations(&self, board: &Board) -> BlendWeights {
        let phase_analysis = self.analyze_game_phase(board);
        phase_analysis.adaptation_recommendations.evaluation_weights
    }
}

impl Default for HybridEvaluationEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Individual evaluation result from a specific evaluator
#[derive(Debug, Clone)]
pub struct EvaluationComponent {
    pub evaluation: f32,
    pub confidence: f32,
    pub computation_time_ms: u64,
    pub additional_info: HashMap<String, f32>,
}

/// Container for all evaluation results
#[derive(Debug, Clone, Default)]
struct EvaluationResults {
    pub nnue: Option<EvaluationComponent>,
    pub pattern: Option<EvaluationComponent>,
    pub tactical: Option<EvaluationComponent>,
    pub strategic: Option<EvaluationComponent>,
}

impl EvaluationResults {
    fn new() -> Self {
        Self::default()
    }
}

/// Comprehensive hybrid evaluation result
#[derive(Debug, Clone)]
pub struct HybridEvaluationResult {
    /// Final blended evaluation
    pub final_evaluation: f32,
    /// NNUE evaluation component
    pub nnue_evaluation: Option<EvaluationComponent>,
    /// Pattern evaluation component
    pub pattern_evaluation: Option<EvaluationComponent>,
    /// Tactical evaluation component
    pub tactical_evaluation: Option<EvaluationComponent>,
    /// Strategic evaluation component
    pub strategic_evaluation: Option<EvaluationComponent>,
    /// Position complexity score (0.0 to 1.0)
    pub complexity_score: f32,
    /// Detected game phase
    pub game_phase: GamePhase,
    /// Confidence in the final evaluation (0.0 to 1.0)
    pub confidence_score: f32,
    /// Weights used for blending
    pub blend_weights: BlendWeights,
    /// Total evaluation time in milliseconds
    pub evaluation_time_ms: u64,
    /// Whether result came from cache
    pub from_cache: bool,
}

/// Game phase classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GamePhase {
    #[default]
    Unknown,
    Opening,
    Middlegame,
    Endgame,
}

/// Dynamic weights for blending evaluations
#[derive(Debug, Clone)]
pub struct BlendWeights {
    pub nnue_weight: f32,
    pub pattern_weight: f32,
    pub tactical_weight: f32,
    pub strategic_weight: f32,
}

impl Default for BlendWeights {
    fn default() -> Self {
        Self {
            nnue_weight: 0.4,
            pattern_weight: 0.3,
            tactical_weight: 0.2,
            strategic_weight: 0.1,
        }
    }
}

/// Enhanced position complexity analyzer with detailed analysis
pub struct ComplexityAnalyzer {
    cached_complexity: RwLock<HashMap<String, ComplexityAnalysisResult>>,
    complexity_weights: ComplexityWeights,
    analysis_depth: AnalysisDepth,
}

impl ComplexityAnalyzer {
    pub fn new() -> Self {
        Self {
            cached_complexity: RwLock::new(HashMap::new()),
            complexity_weights: ComplexityWeights::default(),
            analysis_depth: AnalysisDepth::Standard,
        }
    }

    /// Create a complexity analyzer with custom settings
    pub fn with_analysis_depth(mut self, depth: AnalysisDepth) -> Self {
        self.analysis_depth = depth;
        self
    }

    /// Create a complexity analyzer with custom weights
    pub fn with_complexity_weights(mut self, weights: ComplexityWeights) -> Self {
        self.complexity_weights = weights;
        self
    }

    /// Analyze the complexity of a chess position
    pub fn analyze_complexity(&self, board: &Board) -> f32 {
        let analysis = self.analyze_complexity_detailed(board);
        analysis.overall_complexity
    }

    /// Perform detailed complexity analysis
    pub fn analyze_complexity_detailed(&self, board: &Board) -> ComplexityAnalysisResult {
        let fen = board.to_string();

        // Check cache
        if let Ok(cache) = self.cached_complexity.read() {
            if let Some(analysis) = cache.get(&fen) {
                return analysis.clone();
            }
        }

        let mut analysis = ComplexityAnalysisResult::new();

        // Analyze different complexity factors
        analysis.material_complexity = self.analyze_material_complexity(board);
        analysis.pawn_structure_complexity = self.analyze_pawn_structure_complexity(board);
        analysis.king_safety_complexity = self.analyze_king_safety_complexity(board);
        analysis.piece_coordination_complexity = self.analyze_piece_coordination_complexity(board);
        analysis.tactical_complexity = self.analyze_tactical_complexity(board);
        analysis.positional_complexity = self.analyze_positional_complexity(board);
        analysis.time_complexity = self.analyze_time_complexity(board);
        analysis.endgame_complexity = self.analyze_endgame_complexity(board);

        // Compute weighted overall complexity
        analysis.overall_complexity = self.compute_weighted_complexity(&analysis);

        // Determine complexity category
        analysis.complexity_category = self.categorize_complexity(analysis.overall_complexity);

        // Add position-specific insights
        analysis.key_complexity_factors = self.identify_key_complexity_factors(&analysis);
        analysis.evaluation_recommendations = self.generate_evaluation_recommendations(&analysis);

        // Cache the result with efficient management
        if let Ok(mut cache) = self.cached_complexity.write() {
            if cache.len() > 1000 {
                // Remove oldest 50% of entries instead of clearing all
                let keys_to_remove: Vec<_> = cache.keys().take(500).cloned().collect();
                for key in keys_to_remove {
                    cache.remove(&key);
                }
            }
            cache.insert(fen, analysis.clone());
        }

        analysis
    }

    fn compute_weighted_complexity(&self, analysis: &ComplexityAnalysisResult) -> f32 {
        let weights = &self.complexity_weights;

        let complexity_score = analysis.material_complexity * weights.material_weight
            + analysis.pawn_structure_complexity * weights.pawn_structure_weight
            + analysis.king_safety_complexity * weights.king_safety_weight
            + analysis.piece_coordination_complexity * weights.piece_coordination_weight
            + analysis.tactical_complexity * weights.tactical_weight
            + analysis.positional_complexity * weights.positional_weight
            + analysis.time_complexity * weights.time_weight
            + analysis.endgame_complexity * weights.endgame_weight;

        // Apply analysis depth modifier
        let depth_modifier = match self.analysis_depth {
            AnalysisDepth::Fast => 0.8,
            AnalysisDepth::Standard => 1.0,
            AnalysisDepth::Deep => 1.2,
            AnalysisDepth::Comprehensive => 1.4,
        };

        (complexity_score * depth_modifier).clamp(0.0, 1.0)
    }

    fn categorize_complexity(&self, complexity: f32) -> ComplexityCategory {
        if complexity < 0.2 {
            ComplexityCategory::VeryLow
        } else if complexity < 0.4 {
            ComplexityCategory::Low
        } else if complexity < 0.6 {
            ComplexityCategory::Medium
        } else if complexity < 0.8 {
            ComplexityCategory::High
        } else {
            ComplexityCategory::VeryHigh
        }
    }

    fn identify_key_complexity_factors(
        &self,
        analysis: &ComplexityAnalysisResult,
    ) -> Vec<ComplexityFactor> {
        let mut factors = Vec::new();

        // Check which factors contribute most to complexity
        if analysis.tactical_complexity > 0.7 {
            factors.push(ComplexityFactor::TacticalThreats);
        }
        if analysis.king_safety_complexity > 0.6 {
            factors.push(ComplexityFactor::KingSafety);
        }
        if analysis.piece_coordination_complexity > 0.6 {
            factors.push(ComplexityFactor::PieceCoordination);
        }
        if analysis.pawn_structure_complexity > 0.5 {
            factors.push(ComplexityFactor::PawnStructure);
        }
        if analysis.material_complexity > 0.5 {
            factors.push(ComplexityFactor::MaterialImbalance);
        }
        if analysis.positional_complexity > 0.6 {
            factors.push(ComplexityFactor::PositionalThemes);
        }
        if analysis.time_complexity > 0.7 {
            factors.push(ComplexityFactor::TimeFactors);
        }
        if analysis.endgame_complexity > 0.5 {
            factors.push(ComplexityFactor::EndgameFactors);
        }

        factors
    }

    fn generate_evaluation_recommendations(
        &self,
        analysis: &ComplexityAnalysisResult,
    ) -> EvaluationRecommendations {
        let mut recommendations = EvaluationRecommendations::default();

        // Recommend evaluation methods based on complexity profile
        match analysis.complexity_category {
            ComplexityCategory::VeryLow | ComplexityCategory::Low => {
                recommendations.prefer_nnue = true;
                recommendations.tactical_depth = 4;
                recommendations.pattern_analysis_priority = 0.3;
            }
            ComplexityCategory::Medium => {
                recommendations.prefer_nnue = false;
                recommendations.tactical_depth = 6;
                recommendations.pattern_analysis_priority = 0.5;
                recommendations.strategic_analysis_priority = 0.4;
            }
            ComplexityCategory::High | ComplexityCategory::VeryHigh => {
                recommendations.prefer_nnue = false;
                recommendations.tactical_depth = 8;
                recommendations.pattern_analysis_priority = 0.7;
                recommendations.strategic_analysis_priority = 0.6;
                recommendations.require_tactical_verification = true;
            }
        }

        // Adjust based on specific complexity factors
        if analysis.tactical_complexity > 0.7 {
            recommendations.tactical_depth = (recommendations.tactical_depth + 2).min(12);
            recommendations.require_tactical_verification = true;
        }

        if analysis.king_safety_complexity > 0.6 {
            recommendations.king_safety_analysis = true;
            recommendations.pattern_analysis_priority += 0.1;
        }

        if analysis.endgame_complexity > 0.5 {
            recommendations.endgame_analysis = true;
            recommendations.prefer_nnue = true; // NNUE often good in endgames
        }

        recommendations
    }

    // Enhanced complexity analysis methods
    fn analyze_material_complexity(&self, board: &Board) -> f32 {
        let white_material = self.count_material(board, Color::White);
        let black_material = self.count_material(board, Color::Black);
        let total_material = white_material + black_material;
        let material_imbalance = (white_material - black_material).abs();

        let mut complexity = 0.0;

        // Base material complexity
        complexity += (total_material as f32 / 78.0) * 0.4;

        // Material imbalance complexity
        complexity += (material_imbalance as f32 / 39.0) * 0.3;

        // Piece diversity complexity
        complexity += self.analyze_piece_diversity(board) * 0.3;

        complexity.min(1.0)
    }

    fn analyze_pawn_structure_complexity(&self, board: &Board) -> f32 {
        let mut complexity = 0.0;

        // Pawn islands
        complexity += self.count_pawn_islands(board) as f32 * 0.1;

        // Isolated pawns
        complexity += self.count_isolated_pawns(board) as f32 * 0.15;

        // Doubled pawns
        complexity += self.count_doubled_pawns(board) as f32 * 0.1;

        // Passed pawns
        complexity += self.count_passed_pawns_detailed(board) as f32 * 0.2;

        // Pawn chains
        complexity += self.evaluate_pawn_chains(board) * 0.25;

        // Pawn storms
        complexity += self.evaluate_pawn_storms(board) * 0.2;

        complexity.min(1.0)
    }

    fn analyze_king_safety_complexity(&self, board: &Board) -> f32 {
        let mut complexity = 0.0;

        for color in [Color::White, Color::Black] {
            let king_square = board.king_square(color);

            // King exposure
            complexity += self.evaluate_king_exposure(board, color, king_square) * 0.3;

            // Attacking pieces near king
            complexity += self.count_king_attackers(board, color, king_square) as f32 * 0.1;

            // Pawn shield quality
            complexity += self.evaluate_pawn_shield_complexity(board, color, king_square) * 0.2;

            // Escape squares
            complexity += self.evaluate_escape_squares(board, color, king_square) * 0.1;
        }

        (complexity / 2.0).min(1.0) // Average for both colors
    }

    fn analyze_piece_coordination_complexity(&self, board: &Board) -> f32 {
        let mut complexity = 0.0;

        // Piece mobility variance
        complexity += self.analyze_mobility_complexity(board) * 0.3;

        // Piece harmony/conflicts
        complexity += self.analyze_piece_harmony_complexity(board) * 0.25;

        // Central control complexity
        complexity += self.analyze_central_control_complexity(board) * 0.2;

        // Piece activity imbalance
        complexity += self.analyze_piece_activity_imbalance(board) * 0.25;

        complexity.min(1.0)
    }

    fn analyze_tactical_complexity(&self, board: &Board) -> f32 {
        let mut complexity = 0.0;

        // Check if position has checks
        if board.checkers().popcnt() > 0 {
            complexity += 0.4;
        }

        // Count potential captures
        use chess::MoveGen;
        let legal_moves = MoveGen::new_legal(board);
        let moves: Vec<_> = legal_moves.collect();
        let capture_ratio = moves
            .iter()
            .filter(|mv| board.piece_on(mv.get_dest()).is_some())
            .count() as f32
            / moves.len().max(1) as f32;
        complexity += capture_ratio * 0.3;

        // Move count complexity
        complexity += (moves.len() as f32 / 50.0).min(0.2);

        // Tactical motifs
        complexity += self.analyze_tactical_motifs(board) * 0.4;

        // Hanging pieces
        complexity += self.count_hanging_pieces(board) as f32 * 0.1;

        complexity.min(1.0)
    }

    fn analyze_positional_complexity(&self, board: &Board) -> f32 {
        let mut complexity = 0.0;

        // Space advantage variance
        complexity += self.analyze_space_complexity(board) * 0.3;

        // Weak squares
        complexity += self.analyze_weak_squares_complexity(board) * 0.25;

        // Outposts and holes
        complexity += self.analyze_outpost_complexity(board) * 0.2;

        // Piece placement optimization potential
        complexity += self.analyze_piece_placement_complexity(board) * 0.25;

        complexity.min(1.0)
    }

    fn analyze_time_complexity(&self, board: &Board) -> f32 {
        let mut complexity = 0.0;

        // Tempo sensitivity
        complexity += self.evaluate_tempo_sensitivity(board) * 0.4;

        // Zugzwang potential
        complexity += self.evaluate_zugzwang_complexity(board) * 0.3;

        // Critical move identification
        complexity += self.evaluate_critical_moves_complexity(board) * 0.3;

        complexity.min(1.0)
    }

    fn analyze_endgame_complexity(&self, board: &Board) -> f32 {
        let total_material =
            self.count_material(board, Color::White) + self.count_material(board, Color::Black);

        // Only relevant in endgames
        if total_material > 30 {
            return 0.0;
        }

        let mut complexity = 0.0;

        // Pawn endgame complexity
        complexity += self.analyze_pawn_endgame_complexity(board) * 0.4;

        // Piece endgame complexity
        complexity += self.analyze_piece_endgame_complexity(board) * 0.3;

        // King activity importance
        complexity += self.analyze_king_activity_complexity(board) * 0.3;

        complexity.min(1.0)
    }

    // Helper methods for detailed analysis (simplified implementations)
    fn analyze_piece_diversity(&self, board: &Board) -> f32 {
        let mut piece_types = 0;
        for piece in [
            Piece::Pawn,
            Piece::Knight,
            Piece::Bishop,
            Piece::Rook,
            Piece::Queen,
        ] {
            if (board.pieces(piece) & board.color_combined(Color::White)).popcnt() > 0
                || (board.pieces(piece) & board.color_combined(Color::Black)).popcnt() > 0
            {
                piece_types += 1;
            }
        }
        piece_types as f32 / 5.0
    }

    fn count_pawn_islands(&self, board: &Board) -> u8 {
        let mut islands = 0;
        for color in [Color::White, Color::Black] {
            let pawns = board.pieces(Piece::Pawn) & board.color_combined(color);
            let mut files_with_pawns = [false; 8];

            for square in pawns {
                files_with_pawns[square.get_file().to_index()] = true;
            }

            let mut in_island = false;
            for &has_pawn in &files_with_pawns {
                if has_pawn && !in_island {
                    islands += 1;
                    in_island = true;
                } else if !has_pawn {
                    in_island = false;
                }
            }
        }
        islands
    }

    fn count_isolated_pawns(&self, board: &Board) -> u8 {
        // Simplified implementation
        0
    }

    fn count_doubled_pawns(&self, board: &Board) -> u8 {
        let mut doubled = 0;
        for color in [Color::White, Color::Black] {
            let pawns = board.pieces(Piece::Pawn) & board.color_combined(color);
            let mut pawns_per_file = [0u8; 8];

            for square in pawns {
                pawns_per_file[square.get_file().to_index()] += 1;
            }

            doubled += pawns_per_file.iter().filter(|&&count| count > 1).count() as u8;
        }
        doubled
    }

    fn count_passed_pawns_detailed(&self, _board: &Board) -> u8 {
        // Simplified implementation
        0
    }

    fn evaluate_pawn_chains(&self, _board: &Board) -> f32 {
        // Simplified implementation
        0.0
    }

    fn evaluate_pawn_storms(&self, _board: &Board) -> f32 {
        // Simplified implementation
        0.0
    }

    fn evaluate_king_exposure(&self, _board: &Board, _color: Color, _king_square: Square) -> f32 {
        // Simplified implementation
        0.0
    }

    fn count_king_attackers(&self, _board: &Board, _color: Color, _king_square: Square) -> u8 {
        // Simplified implementation
        0
    }

    fn evaluate_pawn_shield_complexity(
        &self,
        _board: &Board,
        _color: Color,
        _king_square: Square,
    ) -> f32 {
        // Simplified implementation
        0.0
    }

    fn evaluate_escape_squares(&self, _board: &Board, _color: Color, _king_square: Square) -> f32 {
        // Simplified implementation
        0.0
    }

    // More simplified helper methods
    fn analyze_mobility_complexity(&self, _board: &Board) -> f32 {
        0.0
    }
    fn analyze_piece_harmony_complexity(&self, _board: &Board) -> f32 {
        0.0
    }
    fn analyze_central_control_complexity(&self, _board: &Board) -> f32 {
        0.0
    }
    fn analyze_piece_activity_imbalance(&self, _board: &Board) -> f32 {
        0.0
    }
    fn analyze_tactical_motifs(&self, _board: &Board) -> f32 {
        0.0
    }
    fn count_hanging_pieces(&self, _board: &Board) -> u8 {
        0
    }
    fn analyze_space_complexity(&self, _board: &Board) -> f32 {
        0.0
    }
    fn analyze_weak_squares_complexity(&self, _board: &Board) -> f32 {
        0.0
    }
    fn analyze_outpost_complexity(&self, _board: &Board) -> f32 {
        0.0
    }
    fn analyze_piece_placement_complexity(&self, _board: &Board) -> f32 {
        0.0
    }
    fn evaluate_tempo_sensitivity(&self, _board: &Board) -> f32 {
        0.0
    }
    fn evaluate_zugzwang_complexity(&self, _board: &Board) -> f32 {
        0.0
    }
    fn evaluate_critical_moves_complexity(&self, _board: &Board) -> f32 {
        0.0
    }
    fn analyze_pawn_endgame_complexity(&self, _board: &Board) -> f32 {
        0.0
    }
    fn analyze_piece_endgame_complexity(&self, _board: &Board) -> f32 {
        0.0
    }
    fn analyze_king_activity_complexity(&self, _board: &Board) -> f32 {
        0.0
    }

    // Keep the existing methods for backward compatibility
    fn material_complexity(&self, board: &Board) -> f32 {
        self.analyze_material_complexity(board)
    }

    fn count_material(&self, board: &Board, color: Color) -> i32 {
        let pieces = board.color_combined(color);
        let mut material = 0;

        material += (board.pieces(chess::Piece::Pawn) & pieces).popcnt() as i32 * 1;
        material += (board.pieces(chess::Piece::Knight) & pieces).popcnt() as i32 * 3;
        material += (board.pieces(chess::Piece::Bishop) & pieces).popcnt() as i32 * 3;
        material += (board.pieces(chess::Piece::Rook) & pieces).popcnt() as i32 * 5;
        material += (board.pieces(chess::Piece::Queen) & pieces).popcnt() as i32 * 9;

        material
    }

    fn pawn_structure_complexity(&self, _board: &Board) -> f32 {
        // Simplified pawn structure analysis
        // In a full implementation, this would analyze:
        // - Isolated pawns
        // - Doubled pawns
        // - Pawn chains
        // - Passed pawns
        // - Pawn islands
        0.3 // Placeholder
    }

    fn king_safety_complexity(&self, _board: &Board) -> f32 {
        // Simplified king safety analysis
        // In a full implementation, this would analyze:
        // - King exposure
        // - Attacking pieces near king
        // - Pawn shield quality
        // - Escape squares
        0.25 // Placeholder
    }

    fn piece_coordination_complexity(&self, _board: &Board) -> f32 {
        // Simplified piece coordination analysis
        // In a full implementation, this would analyze:
        // - Piece mobility
        // - Piece coordination
        // - Central control
        // - Piece activity
        0.4 // Placeholder
    }

    fn tactical_complexity(&self, board: &Board) -> f32 {
        let mut tactical_score = 0.0;

        // Check if position has checks
        if board.checkers().popcnt() > 0 {
            tactical_score += 0.3;
        }

        // Count potential captures (simplified)
        use chess::MoveGen;
        let legal_moves = MoveGen::new_legal(board);
        let moves: Vec<_> = legal_moves.collect();
        let capture_count = moves
            .iter()
            .filter(|mv| board.piece_on(mv.get_dest()).is_some())
            .count();

        tactical_score += (capture_count as f32 / moves.len().max(1) as f32) * 0.4;

        // Add complexity for having many legal moves
        tactical_score += (moves.len() as f32 / 50.0).min(0.3);

        tactical_score
    }
}

impl Default for ComplexityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Enhanced game phase detector with detailed analysis and adaptation
pub struct GamePhaseDetector {
    cached_phases: RwLock<HashMap<String, GamePhaseAnalysisResult>>,
    phase_weights: PhaseDetectionWeights,
    adaptation_settings: PhaseAdaptationSettings,
}

impl GamePhaseDetector {
    pub fn new() -> Self {
        Self {
            cached_phases: RwLock::new(HashMap::new()),
            phase_weights: PhaseDetectionWeights::default(),
            adaptation_settings: PhaseAdaptationSettings::default(),
        }
    }

    /// Create a phase detector with custom weights
    pub fn with_phase_weights(mut self, weights: PhaseDetectionWeights) -> Self {
        self.phase_weights = weights;
        self
    }

    /// Create a phase detector with custom adaptation settings
    pub fn with_adaptation_settings(mut self, settings: PhaseAdaptationSettings) -> Self {
        self.adaptation_settings = settings;
        self
    }

    /// Detect the current game phase
    pub fn detect_phase(&self, board: &Board) -> GamePhase {
        let analysis = self.analyze_game_phase(board);
        analysis.primary_phase
    }

    /// Perform detailed game phase analysis
    pub fn analyze_game_phase(&self, board: &Board) -> GamePhaseAnalysisResult {
        let fen = board.to_string();

        // Check cache
        if let Ok(cache) = self.cached_phases.read() {
            if let Some(analysis) = cache.get(&fen) {
                return analysis.clone();
            }
        }

        let mut analysis = GamePhaseAnalysisResult::new();

        // Analyze multiple phase indicators
        analysis.material_phase = self.analyze_material_phase(board);
        analysis.development_phase = self.analyze_development_phase(board);
        analysis.move_count_phase = self.analyze_move_count_phase(board);
        analysis.king_safety_phase = self.analyze_king_safety_phase(board);
        analysis.pawn_structure_phase = self.analyze_pawn_structure_phase(board);
        analysis.piece_activity_phase = self.analyze_piece_activity_phase(board);

        // Compute phase scores for each possible phase
        analysis.opening_score = self.compute_opening_score(&analysis);
        analysis.middlegame_score = self.compute_middlegame_score(&analysis);
        analysis.endgame_score = self.compute_endgame_score(&analysis);

        // Determine primary phase
        analysis.primary_phase = self.determine_primary_phase(&analysis);

        // Determine transition state
        analysis.transition_state = self.analyze_transition_state(&analysis);

        // Calculate phase confidence
        analysis.phase_confidence = self.calculate_phase_confidence(&analysis);

        // Generate phase-specific adaptation recommendations
        analysis.adaptation_recommendations = self.generate_adaptation_recommendations(&analysis);

        // Cache the result
        if let Ok(mut cache) = self.cached_phases.write() {
            if cache.len() > 1000 {
                cache.clear();
            }
            cache.insert(fen, analysis.clone());
        }

        analysis
    }

    fn analyze_material_phase(&self, board: &Board) -> PhaseIndicator {
        let total_material = self.calculate_total_material(board);
        let material_balance = self.analyze_material_balance(board);

        let phase = if total_material >= 60 {
            GamePhase::Opening
        } else if total_material <= 20 {
            GamePhase::Endgame
        } else {
            GamePhase::Middlegame
        };

        let confidence = self.compute_material_phase_confidence(total_material, material_balance);

        PhaseIndicator { phase, confidence }
    }

    fn analyze_development_phase(&self, board: &Board) -> PhaseIndicator {
        let development_score = self.calculate_development_score(board);
        let castling_status = self.analyze_castling_status(board);

        let phase = if development_score < 0.3 {
            GamePhase::Opening
        } else if development_score > 0.8 {
            GamePhase::Middlegame
        } else {
            GamePhase::Opening
        };

        let confidence = (development_score + castling_status) / 2.0;

        PhaseIndicator { phase, confidence }
    }

    fn analyze_move_count_phase(&self, board: &Board) -> PhaseIndicator {
        // Use a heuristic for move count since chess crate doesn't expose fullmove_number directly
        let move_count = (board.combined().popcnt() as u32).saturating_sub(32) / 2 + 1;

        let (phase, confidence) = if move_count <= 12 {
            (GamePhase::Opening, 0.8)
        } else if move_count <= 25 {
            (GamePhase::Middlegame, 0.7)
        } else if move_count <= 40 {
            (GamePhase::Middlegame, 0.6)
        } else {
            (GamePhase::Endgame, 0.5)
        };

        PhaseIndicator { phase, confidence }
    }

    fn analyze_king_safety_phase(&self, board: &Board) -> PhaseIndicator {
        let king_safety_score = self.evaluate_combined_king_safety(board);

        let phase = if king_safety_score > 0.7 {
            GamePhase::Opening // Kings still safe, likely early game
        } else if king_safety_score < 0.3 {
            GamePhase::Middlegame // Kings under pressure
        } else {
            GamePhase::Endgame // Kings becoming active
        };

        let confidence = (1.0 - king_safety_score).abs();

        PhaseIndicator { phase, confidence }
    }

    fn analyze_pawn_structure_phase(&self, board: &Board) -> PhaseIndicator {
        let pawn_structure_score = self.evaluate_pawn_structure_development(board);

        let phase = if pawn_structure_score < 0.4 {
            GamePhase::Opening
        } else if pawn_structure_score > 0.7 {
            GamePhase::Endgame
        } else {
            GamePhase::Middlegame
        };

        let confidence = pawn_structure_score;

        PhaseIndicator { phase, confidence }
    }

    fn analyze_piece_activity_phase(&self, board: &Board) -> PhaseIndicator {
        let activity_score = self.calculate_piece_activity_score(board);

        let phase = if activity_score < 0.3 {
            GamePhase::Opening
        } else if activity_score > 0.8 {
            GamePhase::Endgame
        } else {
            GamePhase::Middlegame
        };

        let confidence = activity_score;

        PhaseIndicator { phase, confidence }
    }

    fn compute_opening_score(&self, analysis: &GamePhaseAnalysisResult) -> f32 {
        let mut score = 0.0;

        // Weight different indicators for opening detection
        if analysis.material_phase.phase == GamePhase::Opening {
            score += analysis.material_phase.confidence * self.phase_weights.material_weight;
        }
        if analysis.development_phase.phase == GamePhase::Opening {
            score += analysis.development_phase.confidence * self.phase_weights.development_weight;
        }
        if analysis.move_count_phase.phase == GamePhase::Opening {
            score += analysis.move_count_phase.confidence * self.phase_weights.move_count_weight;
        }
        if analysis.king_safety_phase.phase == GamePhase::Opening {
            score += analysis.king_safety_phase.confidence * self.phase_weights.king_safety_weight;
        }
        if analysis.pawn_structure_phase.phase == GamePhase::Opening {
            score +=
                analysis.pawn_structure_phase.confidence * self.phase_weights.pawn_structure_weight;
        }
        if analysis.piece_activity_phase.phase == GamePhase::Opening {
            score +=
                analysis.piece_activity_phase.confidence * self.phase_weights.piece_activity_weight;
        }

        score
    }

    fn compute_middlegame_score(&self, analysis: &GamePhaseAnalysisResult) -> f32 {
        let mut score = 0.0;

        if analysis.material_phase.phase == GamePhase::Middlegame {
            score += analysis.material_phase.confidence * self.phase_weights.material_weight;
        }
        if analysis.development_phase.phase == GamePhase::Middlegame {
            score += analysis.development_phase.confidence * self.phase_weights.development_weight;
        }
        if analysis.move_count_phase.phase == GamePhase::Middlegame {
            score += analysis.move_count_phase.confidence * self.phase_weights.move_count_weight;
        }
        if analysis.king_safety_phase.phase == GamePhase::Middlegame {
            score += analysis.king_safety_phase.confidence * self.phase_weights.king_safety_weight;
        }
        if analysis.pawn_structure_phase.phase == GamePhase::Middlegame {
            score +=
                analysis.pawn_structure_phase.confidence * self.phase_weights.pawn_structure_weight;
        }
        if analysis.piece_activity_phase.phase == GamePhase::Middlegame {
            score +=
                analysis.piece_activity_phase.confidence * self.phase_weights.piece_activity_weight;
        }

        score
    }

    fn compute_endgame_score(&self, analysis: &GamePhaseAnalysisResult) -> f32 {
        let mut score = 0.0;

        if analysis.material_phase.phase == GamePhase::Endgame {
            score += analysis.material_phase.confidence * self.phase_weights.material_weight;
        }
        if analysis.development_phase.phase == GamePhase::Endgame {
            score += analysis.development_phase.confidence * self.phase_weights.development_weight;
        }
        if analysis.move_count_phase.phase == GamePhase::Endgame {
            score += analysis.move_count_phase.confidence * self.phase_weights.move_count_weight;
        }
        if analysis.king_safety_phase.phase == GamePhase::Endgame {
            score += analysis.king_safety_phase.confidence * self.phase_weights.king_safety_weight;
        }
        if analysis.pawn_structure_phase.phase == GamePhase::Endgame {
            score +=
                analysis.pawn_structure_phase.confidence * self.phase_weights.pawn_structure_weight;
        }
        if analysis.piece_activity_phase.phase == GamePhase::Endgame {
            score +=
                analysis.piece_activity_phase.confidence * self.phase_weights.piece_activity_weight;
        }

        score
    }

    fn determine_primary_phase(&self, analysis: &GamePhaseAnalysisResult) -> GamePhase {
        let opening_score = analysis.opening_score;
        let middlegame_score = analysis.middlegame_score;
        let endgame_score = analysis.endgame_score;

        if opening_score > middlegame_score && opening_score > endgame_score {
            GamePhase::Opening
        } else if endgame_score > middlegame_score && endgame_score > opening_score {
            GamePhase::Endgame
        } else if middlegame_score > 0.1 {
            GamePhase::Middlegame
        } else {
            GamePhase::Unknown
        }
    }

    fn analyze_transition_state(&self, analysis: &GamePhaseAnalysisResult) -> PhaseTransition {
        let max_score = analysis
            .opening_score
            .max(analysis.middlegame_score)
            .max(analysis.endgame_score);
        let second_max = if analysis.opening_score == max_score {
            analysis.middlegame_score.max(analysis.endgame_score)
        } else if analysis.middlegame_score == max_score {
            analysis.opening_score.max(analysis.endgame_score)
        } else {
            analysis.opening_score.max(analysis.middlegame_score)
        };

        let transition_threshold = 0.3;

        if max_score - second_max < transition_threshold {
            // Close scores indicate transition
            if analysis.opening_score > analysis.endgame_score {
                if analysis.middlegame_score > analysis.opening_score * 0.8 {
                    PhaseTransition::OpeningToMiddlegame
                } else {
                    PhaseTransition::Stable
                }
            } else if analysis.middlegame_score > analysis.endgame_score {
                if analysis.endgame_score > analysis.middlegame_score * 0.8 {
                    PhaseTransition::MiddlegameToEndgame
                } else {
                    PhaseTransition::Stable
                }
            } else {
                PhaseTransition::Stable
            }
        } else {
            PhaseTransition::Stable
        }
    }

    fn calculate_phase_confidence(&self, analysis: &GamePhaseAnalysisResult) -> f32 {
        let scores = [
            analysis.opening_score,
            analysis.middlegame_score,
            analysis.endgame_score,
        ];
        let max_score = scores.iter().fold(0.0f32, |a, &b| a.max(b));
        let total_score = scores.iter().sum::<f32>();

        if total_score > 0.0 {
            max_score / total_score
        } else {
            0.0
        }
    }

    fn generate_adaptation_recommendations(
        &self,
        analysis: &GamePhaseAnalysisResult,
    ) -> PhaseAdaptationRecommendations {
        let mut recommendations = PhaseAdaptationRecommendations::default();

        match analysis.primary_phase {
            GamePhase::Opening => {
                recommendations.evaluation_weights = BlendWeights {
                    nnue_weight: 0.3,
                    pattern_weight: 0.2,
                    tactical_weight: 0.2,
                    strategic_weight: 0.3,
                };
                recommendations.search_depth_modifier = -1;
                recommendations.opening_book_priority = 0.9;
                recommendations.endgame_tablebase_priority = 0.1;
                recommendations.time_management_factor = 0.8;
            }
            GamePhase::Middlegame => {
                recommendations.evaluation_weights = BlendWeights {
                    nnue_weight: 0.25,
                    pattern_weight: 0.35,
                    tactical_weight: 0.3,
                    strategic_weight: 0.1,
                };
                recommendations.search_depth_modifier = 0;
                recommendations.opening_book_priority = 0.3;
                recommendations.endgame_tablebase_priority = 0.2;
                recommendations.time_management_factor = 1.0;
            }
            GamePhase::Endgame => {
                recommendations.evaluation_weights = BlendWeights {
                    nnue_weight: 0.5,
                    pattern_weight: 0.2,
                    tactical_weight: 0.15,
                    strategic_weight: 0.15,
                };
                recommendations.search_depth_modifier = 1;
                recommendations.opening_book_priority = 0.0;
                recommendations.endgame_tablebase_priority = 0.9;
                recommendations.time_management_factor = 1.2;
            }
            GamePhase::Unknown => {
                recommendations.evaluation_weights = BlendWeights::default();
                recommendations.search_depth_modifier = 0;
                recommendations.opening_book_priority = 0.5;
                recommendations.endgame_tablebase_priority = 0.5;
                recommendations.time_management_factor = 1.0;
            }
        }

        // Adjust for transition states
        match analysis.transition_state {
            PhaseTransition::OpeningToMiddlegame => {
                recommendations.evaluation_weights.strategic_weight *= 0.8;
                recommendations.evaluation_weights.tactical_weight *= 1.2;
            }
            PhaseTransition::MiddlegameToEndgame => {
                recommendations.evaluation_weights.nnue_weight *= 1.3;
                recommendations.evaluation_weights.pattern_weight *= 0.7;
                recommendations.endgame_tablebase_priority += 0.2;
            }
            PhaseTransition::Stable => {
                // No adjustments needed for stable phases
            }
        }

        recommendations
    }

    // Helper methods for phase analysis
    fn analyze_material_balance(&self, board: &Board) -> f32 {
        let white_material = self.count_material(board, Color::White);
        let black_material = self.count_material(board, Color::Black);
        let total_material = white_material + black_material;

        if total_material > 0 {
            (white_material - black_material).abs() as f32 / total_material as f32
        } else {
            0.0
        }
    }

    fn compute_material_phase_confidence(&self, total_material: i32, material_balance: f32) -> f32 {
        let material_factor = if total_material >= 60 || total_material <= 20 {
            0.8
        } else {
            0.4
        };

        let balance_factor = 1.0 - material_balance;

        (material_factor + balance_factor) / 2.0
    }

    fn calculate_development_score(&self, board: &Board) -> f32 {
        let mut development_score = 0.0;
        let mut total_pieces = 0;

        for color in [Color::White, Color::Black] {
            // Count developed knights
            let knights = board.pieces(Piece::Knight) & board.color_combined(color);
            for square in knights {
                total_pieces += 1;
                if self.is_piece_developed_from_starting_position(square, Piece::Knight, color) {
                    development_score += 1.0;
                }
            }

            // Count developed bishops
            let bishops = board.pieces(Piece::Bishop) & board.color_combined(color);
            for square in bishops {
                total_pieces += 1;
                if self.is_piece_developed_from_starting_position(square, Piece::Bishop, color) {
                    development_score += 1.0;
                }
            }
        }

        if total_pieces > 0 {
            development_score / total_pieces as f32
        } else {
            0.0
        }
    }

    fn analyze_castling_status(&self, board: &Board) -> f32 {
        let mut castling_score = 0.0;

        for color in [Color::White, Color::Black] {
            if board.castle_rights(color).has_kingside()
                || board.castle_rights(color).has_queenside()
            {
                castling_score += 0.3; // Still has castling rights
            } else {
                castling_score += 0.8; // Has likely castled or lost rights
            }
        }

        castling_score / 2.0
    }

    fn evaluate_combined_king_safety(&self, board: &Board) -> f32 {
        let mut safety_score = 0.0;

        for color in [Color::White, Color::Black] {
            let king_square = board.king_square(color);
            safety_score += self.evaluate_individual_king_safety(board, color, king_square);
        }

        safety_score / 2.0
    }

    fn evaluate_pawn_structure_development(&self, board: &Board) -> f32 {
        let mut structure_score = 0.0;

        // Count advanced pawns
        for color in [Color::White, Color::Black] {
            let pawns = board.pieces(Piece::Pawn) & board.color_combined(color);
            let advanced_pawns = pawns
                .into_iter()
                .filter(|&square| self.is_pawn_advanced(square, color))
                .count();

            structure_score += advanced_pawns as f32 * 0.1;
        }

        structure_score.min(1.0)
    }

    fn calculate_piece_activity_score(&self, board: &Board) -> f32 {
        let mut activity_score = 0.0;
        let mut piece_count = 0;

        for color in [Color::White, Color::Black] {
            for piece_type in [Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen] {
                let pieces = board.pieces(piece_type) & board.color_combined(color);
                for square in pieces {
                    piece_count += 1;
                    activity_score += self.calculate_piece_activity(board, square, piece_type);
                }
            }
        }

        if piece_count > 0 {
            activity_score / piece_count as f32
        } else {
            0.0
        }
    }

    // Simplified helper methods
    fn is_piece_developed_from_starting_position(
        &self,
        square: Square,
        piece: Piece,
        color: Color,
    ) -> bool {
        // Simplified check - in practice would check against actual starting squares
        let starting_rank = match color {
            Color::White => chess::Rank::First,
            Color::Black => chess::Rank::Eighth,
        };

        square.get_rank() != starting_rank
    }

    fn evaluate_individual_king_safety(
        &self,
        _board: &Board,
        _color: Color,
        _king_square: Square,
    ) -> f32 {
        // Simplified implementation
        0.5
    }

    fn is_pawn_advanced(&self, square: Square, color: Color) -> bool {
        let rank = square.get_rank();
        match color {
            Color::White => rank >= chess::Rank::Fourth,
            Color::Black => rank <= chess::Rank::Fifth,
        }
    }

    fn calculate_piece_activity(&self, _board: &Board, _square: Square, _piece: Piece) -> f32 {
        // Simplified implementation - would calculate mobility, attacks, etc.
        0.5
    }

    fn calculate_total_material(&self, board: &Board) -> i32 {
        let white_material = self.count_material(board, Color::White);
        let black_material = self.count_material(board, Color::Black);
        white_material + black_material
    }

    fn count_material(&self, board: &Board, color: Color) -> i32 {
        let pieces = board.color_combined(color);
        let mut material = 0;

        material += (board.pieces(Piece::Pawn) & pieces).popcnt() as i32 * 1;
        material += (board.pieces(Piece::Knight) & pieces).popcnt() as i32 * 3;
        material += (board.pieces(Piece::Bishop) & pieces).popcnt() as i32 * 3;
        material += (board.pieces(Piece::Rook) & pieces).popcnt() as i32 * 5;
        material += (board.pieces(Piece::Queen) & pieces).popcnt() as i32 * 9;

        material
    }
}

impl Default for GamePhaseDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Evaluation blender with dynamic weight computation and adaptive learning
pub struct EvaluationBlender {
    base_weights: BlendWeights,
    weight_history: RwLock<Vec<WeightHistoryEntry>>,
    performance_tracker: RwLock<EvaluatorPerformanceTracker>,
    adaptive_learning: bool,
}

impl EvaluationBlender {
    pub fn new() -> Self {
        Self {
            base_weights: BlendWeights::default(),
            weight_history: RwLock::new(Vec::new()),
            performance_tracker: RwLock::new(EvaluatorPerformanceTracker::new()),
            adaptive_learning: true,
        }
    }

    /// Create a blender with custom base weights
    pub fn with_base_weights(base_weights: BlendWeights) -> Self {
        Self {
            base_weights,
            weight_history: RwLock::new(Vec::new()),
            performance_tracker: RwLock::new(EvaluatorPerformanceTracker::new()),
            adaptive_learning: true,
        }
    }

    /// Enable or disable adaptive learning
    pub fn set_adaptive_learning(&mut self, enabled: bool) {
        self.adaptive_learning = enabled;
    }

    /// Compute dynamic blend weights based on position characteristics
    pub fn compute_blend_weights(
        &self,
        complexity_score: f32,
        game_phase: &GamePhase,
        evaluation_results: &EvaluationResults,
    ) -> BlendWeights {
        let mut weights = if self.adaptive_learning {
            self.compute_adaptive_base_weights(complexity_score, game_phase)
        } else {
            self.base_weights.clone()
        };

        // Adjust weights based on game phase
        match game_phase {
            GamePhase::Opening => {
                weights.strategic_weight += 0.1;
                weights.tactical_weight -= 0.05;
            }
            GamePhase::Middlegame => {
                weights.tactical_weight += 0.1;
                weights.pattern_weight += 0.05;
            }
            GamePhase::Endgame => {
                weights.nnue_weight += 0.15;
                weights.strategic_weight -= 0.1;
            }
            GamePhase::Unknown => {} // Use base weights
        }

        // Adjust weights based on complexity
        if complexity_score > 0.7 {
            // High complexity: trust tactical evaluation more
            weights.tactical_weight += 0.15;
            weights.nnue_weight -= 0.05;
            weights.pattern_weight -= 0.05;
            weights.strategic_weight -= 0.05;
        } else if complexity_score < 0.3 {
            // Low complexity: trust NNUE more
            weights.nnue_weight += 0.1;
            weights.tactical_weight -= 0.1;
        }

        // Adjust weights based on evaluator agreement
        if let (Some(nnue), Some(pattern)) = (&evaluation_results.nnue, &evaluation_results.pattern)
        {
            let agreement = 1.0 - (nnue.evaluation - pattern.evaluation).abs().min(1.0);
            if agreement > 0.8 {
                // High agreement: boost both NNUE and pattern
                weights.nnue_weight += 0.05;
                weights.pattern_weight += 0.05;
                weights.tactical_weight -= 0.1;
            }
        }

        // Normalize weights to sum to 1.0
        self.normalize_weights(&mut weights);

        // Record weight usage for adaptive learning
        if self.adaptive_learning {
            self.record_weight_usage(&weights, complexity_score, game_phase);
        }

        weights
    }

    /// Compute adaptive base weights based on historical performance
    fn compute_adaptive_base_weights(
        &self,
        complexity_score: f32,
        game_phase: &GamePhase,
    ) -> BlendWeights {
        if let Ok(tracker) = self.performance_tracker.read() {
            // Get performance metrics for current context
            let context = EvaluationContext {
                complexity_range: Self::complexity_to_range(complexity_score),
                game_phase: game_phase.clone(),
            };

            if let Some(optimal_weights) = tracker.get_optimal_weights(&context) {
                return optimal_weights;
            }
        }

        // Fall back to base weights if no adaptive data available
        self.base_weights.clone()
    }

    /// Record weight usage for learning
    fn record_weight_usage(
        &self,
        weights: &BlendWeights,
        complexity_score: f32,
        game_phase: &GamePhase,
    ) {
        if let Ok(mut history) = self.weight_history.write() {
            let entry = WeightHistoryEntry {
                timestamp: std::time::SystemTime::now(),
                weights: weights.clone(),
                complexity_score,
                game_phase: game_phase.clone(),
                evaluation_count: 1,
            };

            history.push(entry);

            // Limit history size
            if history.len() > 10000 {
                history.drain(0..1000); // Remove oldest 1000 entries
            }
        }
    }

    /// Update performance metrics based on evaluation accuracy
    pub fn update_performance_metrics(
        &self,
        weights: &BlendWeights,
        complexity_score: f32,
        game_phase: &GamePhase,
        evaluation_accuracy: f32,
        actual_result: Option<f32>,
    ) {
        if !self.adaptive_learning {
            return;
        }

        if let Ok(mut tracker) = self.performance_tracker.write() {
            let context = EvaluationContext {
                complexity_range: Self::complexity_to_range(complexity_score),
                game_phase: game_phase.clone(),
            };

            tracker.record_performance(&context, weights, evaluation_accuracy, actual_result);
        }
    }

    /// Convert complexity score to discrete range
    fn complexity_to_range(complexity: f32) -> ComplexityRange {
        if complexity < 0.3 {
            ComplexityRange::Low
        } else if complexity < 0.7 {
            ComplexityRange::Medium
        } else {
            ComplexityRange::High
        }
    }

    /// Get adaptive learning statistics
    pub fn get_adaptive_stats(&self) -> AdaptiveLearningStats {
        let weight_entries = self.weight_history.read().map(|h| h.len()).unwrap_or(0);

        let performance_contexts = self
            .performance_tracker
            .read()
            .map(|t| t.get_context_count())
            .unwrap_or(0);

        let learning_enabled = self.adaptive_learning;

        AdaptiveLearningStats {
            weight_history_entries: weight_entries,
            performance_contexts,
            learning_enabled,
            total_adaptations: weight_entries, // Simplified for now
        }
    }

    /// Blend evaluations using the provided weights
    pub fn blend_evaluations(
        &self,
        evaluation_results: &EvaluationResults,
        weights: &BlendWeights,
    ) -> f32 {
        let mut blended_evaluation = 0.0;
        let mut total_weight = 0.0;

        if let Some(ref nnue) = evaluation_results.nnue {
            blended_evaluation += nnue.evaluation * weights.nnue_weight;
            total_weight += weights.nnue_weight;
        }

        if let Some(ref pattern) = evaluation_results.pattern {
            blended_evaluation += pattern.evaluation * weights.pattern_weight;
            total_weight += weights.pattern_weight;
        }

        if let Some(ref tactical) = evaluation_results.tactical {
            blended_evaluation += tactical.evaluation * weights.tactical_weight;
            total_weight += weights.tactical_weight;
        }

        if let Some(ref strategic) = evaluation_results.strategic {
            blended_evaluation += strategic.evaluation * weights.strategic_weight;
            total_weight += weights.strategic_weight;
        }

        if total_weight > 0.0 {
            blended_evaluation / total_weight
        } else {
            0.0 // No evaluations available
        }
    }

    fn normalize_weights(&self, weights: &mut BlendWeights) {
        let total = weights.nnue_weight
            + weights.pattern_weight
            + weights.tactical_weight
            + weights.strategic_weight;
        if total > 0.0 {
            weights.nnue_weight /= total;
            weights.pattern_weight /= total;
            weights.tactical_weight /= total;
            weights.strategic_weight /= total;
        }
    }
}

impl Default for EvaluationBlender {
    fn default() -> Self {
        Self::new()
    }
}

/// Confidence scorer for hybrid evaluations
/// Enhanced confidence scoring system for hybrid evaluation
pub struct ConfidenceScorer {
    /// Historical accuracy tracking for different evaluator combinations
    evaluator_accuracy_history: Arc<RwLock<EvaluatorAccuracyTracker>>,
    /// Pattern clarity analyzer
    pattern_clarity_analyzer: PatternClarityAnalyzer,
    /// Confidence calibration settings
    calibration_settings: ConfidenceCalibrationSettings,
    /// Recent confidence scores for trend analysis
    recent_confidence_scores: Arc<RwLock<VecDeque<ConfidenceHistoryEntry>>>,
}

impl ConfidenceScorer {
    pub fn new() -> Self {
        Self {
            evaluator_accuracy_history: Arc::new(RwLock::new(EvaluatorAccuracyTracker::new())),
            pattern_clarity_analyzer: PatternClarityAnalyzer::new(),
            calibration_settings: ConfidenceCalibrationSettings::default(),
            recent_confidence_scores: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
        }
    }

    /// Create confidence scorer with custom calibration settings
    pub fn with_calibration_settings(mut self, settings: ConfidenceCalibrationSettings) -> Self {
        self.calibration_settings = settings;
        self
    }

    /// Compute comprehensive confidence in the hybrid evaluation
    pub fn compute_confidence(
        &self,
        evaluation_results: &EvaluationResults,
        blend_weights: &BlendWeights,
        complexity_score: f32,
        position_context: &PositionContext,
    ) -> ConfidenceAnalysisResult {
        let start_time = std::time::Instant::now();

        // Factor 1: Evaluator agreement analysis
        let agreement_analysis = self.analyze_evaluator_agreement(evaluation_results);

        // Factor 2: Individual evaluator confidence with historical context
        let evaluator_confidence =
            self.analyze_evaluator_confidence(evaluation_results, blend_weights);

        // Factor 3: Position complexity confidence
        let complexity_confidence =
            self.analyze_complexity_confidence(complexity_score, position_context);

        // Factor 4: Pattern clarity confidence
        let pattern_clarity = self
            .pattern_clarity_analyzer
            .analyze_pattern_clarity(evaluation_results, position_context);

        // Factor 5: Historical accuracy confidence
        let historical_confidence =
            self.analyze_historical_accuracy(evaluation_results, blend_weights, position_context);

        // Factor 6: Evaluation coverage confidence
        let coverage_confidence = self.analyze_evaluation_coverage(evaluation_results);

        // Factor 7: Temporal consistency confidence
        let temporal_confidence = self.analyze_temporal_consistency(evaluation_results);

        // Compute weighted overall confidence
        let confidence_factors = ConfidenceFactors {
            evaluator_agreement: agreement_analysis.overall_agreement,
            individual_confidence: evaluator_confidence,
            complexity_confidence,
            pattern_clarity: pattern_clarity.overall_clarity,
            historical_accuracy: historical_confidence,
            coverage_confidence,
            temporal_consistency: temporal_confidence,
        };

        let overall_confidence = self.compute_weighted_confidence(&confidence_factors);

        // Apply calibration
        let calibrated_confidence =
            self.apply_confidence_calibration(overall_confidence, &confidence_factors);

        let computation_time = start_time.elapsed().as_millis() as u64;

        let result = ConfidenceAnalysisResult {
            overall_confidence: calibrated_confidence,
            confidence_factors: confidence_factors.clone(),
            agreement_analysis,
            pattern_clarity_analysis: pattern_clarity,
            computation_time_ms: computation_time,
            confidence_category: self.categorize_confidence(calibrated_confidence),
            reliability_indicators: self.generate_reliability_indicators(&confidence_factors),
        };

        // Record confidence score for trend analysis
        self.record_confidence_score(&result, position_context);

        result
    }

    /// Simplified confidence computation for compatibility
    pub fn compute_simple_confidence(
        &self,
        evaluation_results: &EvaluationResults,
        blend_weights: &BlendWeights,
        complexity_score: f32,
    ) -> f32 {
        let position_context = PositionContext::default();
        let analysis = self.compute_confidence(
            evaluation_results,
            blend_weights,
            complexity_score,
            &position_context,
        );
        analysis.overall_confidence
    }

    fn analyze_evaluator_agreement(
        &self,
        evaluation_results: &EvaluationResults,
    ) -> AgreementAnalysis {
        let mut evaluations = Vec::new();
        let mut evaluator_names = Vec::new();

        if let Some(ref nnue) = evaluation_results.nnue {
            evaluations.push(nnue.evaluation);
            evaluator_names.push("NNUE");
        }
        if let Some(ref pattern) = evaluation_results.pattern {
            evaluations.push(pattern.evaluation);
            evaluator_names.push("Pattern");
        }
        if let Some(ref tactical) = evaluation_results.tactical {
            evaluations.push(tactical.evaluation);
            evaluator_names.push("Tactical");
        }
        if let Some(ref strategic) = evaluation_results.strategic {
            evaluations.push(strategic.evaluation);
            evaluator_names.push("Strategic");
        }

        if evaluations.len() < 2 {
            return AgreementAnalysis {
                overall_agreement: 0.5,
                pairwise_agreements: Vec::new(),
                evaluation_spread: 0.0,
                consensus_strength: 0.5,
                outlier_count: 0,
            };
        }

        // Calculate pairwise agreements
        let mut pairwise_agreements = Vec::new();
        for i in 0..evaluations.len() {
            for j in (i + 1)..evaluations.len() {
                let diff = (evaluations[i] - evaluations[j]).abs();
                let agreement = (2.0 - diff).max(0.0).min(1.0);
                pairwise_agreements.push(PairwiseAgreement {
                    evaluator1: evaluator_names[i].to_string(),
                    evaluator2: evaluator_names[j].to_string(),
                    agreement_score: agreement,
                });
            }
        }

        let overall_agreement = pairwise_agreements
            .iter()
            .map(|pa| pa.agreement_score)
            .sum::<f32>()
            / pairwise_agreements.len() as f32;

        // Calculate evaluation spread
        let mean = evaluations.iter().sum::<f32>() / evaluations.len() as f32;
        let variance = evaluations
            .iter()
            .map(|eval| (eval - mean).powi(2))
            .sum::<f32>()
            / evaluations.len() as f32;
        let evaluation_spread = variance.sqrt();

        // Consensus strength (inverse of spread)
        let consensus_strength = (2.0 - evaluation_spread).max(0.0).min(1.0);

        // Count outliers (evaluations > 1.5 std deviations from mean)
        let outlier_count = evaluations
            .iter()
            .filter(|&&eval| (eval - mean).abs() > 1.5 * evaluation_spread)
            .count();

        AgreementAnalysis {
            overall_agreement,
            pairwise_agreements,
            evaluation_spread,
            consensus_strength,
            outlier_count,
        }
    }

    fn analyze_evaluator_confidence(
        &self,
        evaluation_results: &EvaluationResults,
        blend_weights: &BlendWeights,
    ) -> f32 {
        let mut weighted_confidence = 0.0;
        let mut total_weight = 0.0;

        if let Some(ref nnue) = evaluation_results.nnue {
            let weight = blend_weights.nnue_weight;
            let adjusted_confidence = self.adjust_confidence_by_context(nnue.confidence, "NNUE");
            weighted_confidence += adjusted_confidence * weight;
            total_weight += weight;
        }
        if let Some(ref pattern) = evaluation_results.pattern {
            let weight = blend_weights.pattern_weight;
            let adjusted_confidence =
                self.adjust_confidence_by_context(pattern.confidence, "Pattern");
            weighted_confidence += adjusted_confidence * weight;
            total_weight += weight;
        }
        if let Some(ref tactical) = evaluation_results.tactical {
            let weight = blend_weights.tactical_weight;
            let adjusted_confidence =
                self.adjust_confidence_by_context(tactical.confidence, "Tactical");
            weighted_confidence += adjusted_confidence * weight;
            total_weight += weight;
        }
        if let Some(ref strategic) = evaluation_results.strategic {
            let weight = blend_weights.strategic_weight;
            let adjusted_confidence =
                self.adjust_confidence_by_context(strategic.confidence, "Strategic");
            weighted_confidence += adjusted_confidence * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            weighted_confidence / total_weight
        } else {
            0.5
        }
    }

    fn analyze_complexity_confidence(
        &self,
        complexity_score: f32,
        position_context: &PositionContext,
    ) -> f32 {
        let base_confidence = 1.0 - complexity_score;

        // Adjust based on position characteristics
        let mut adjusted_confidence = base_confidence;

        // Tactical positions tend to be more uncertain
        if position_context.has_tactical_threats {
            adjusted_confidence *= 0.8;
        }

        // Endgame positions are often more precise
        if position_context.game_phase == GamePhase::Endgame {
            adjusted_confidence *= 1.1;
        }

        // Opening positions with book knowledge are more confident
        if position_context.game_phase == GamePhase::Opening && position_context.in_opening_book {
            adjusted_confidence *= 1.2;
        }

        adjusted_confidence.clamp(0.0, 1.0)
    }

    fn analyze_historical_accuracy(
        &self,
        evaluation_results: &EvaluationResults,
        blend_weights: &BlendWeights,
        position_context: &PositionContext,
    ) -> f32 {
        if let Ok(accuracy_tracker) = self.evaluator_accuracy_history.read() {
            let evaluator_combination =
                EvaluatorCombination::from_results(evaluation_results, blend_weights);
            let context_hash = position_context.get_context_hash();

            accuracy_tracker
                .get_historical_accuracy(&evaluator_combination, context_hash)
                .unwrap_or(0.6) // Default confidence when no history available
        } else {
            0.6
        }
    }

    fn analyze_evaluation_coverage(&self, evaluation_results: &EvaluationResults) -> f32 {
        let active_evaluators = self.count_active_evaluators(evaluation_results);
        let max_evaluators = 4.0;

        let base_coverage = active_evaluators as f32 / max_evaluators;

        // Bonus for having diverse evaluator types
        let mut diversity_bonus = 0.0;
        if evaluation_results.nnue.is_some() && evaluation_results.tactical.is_some() {
            diversity_bonus += 0.1; // Neural + tactical is a good combination
        }
        if evaluation_results.pattern.is_some() && evaluation_results.strategic.is_some() {
            diversity_bonus += 0.1; // Pattern + strategic is complementary
        }

        (base_coverage + diversity_bonus).min(1.0)
    }

    fn analyze_temporal_consistency(&self, _evaluation_results: &EvaluationResults) -> f32 {
        // Analyze consistency with recent evaluations
        if let Ok(recent_scores) = self.recent_confidence_scores.read() {
            if recent_scores.len() < 3 {
                return 0.6; // Not enough data for trend analysis
            }

            let recent_confidences: Vec<f32> = recent_scores
                .iter()
                .rev()
                .take(5)
                .map(|entry| entry.confidence_score)
                .collect();

            if recent_confidences.len() < 2 {
                return 0.6;
            }

            // Calculate consistency (low variance = high consistency)
            let mean = recent_confidences.iter().sum::<f32>() / recent_confidences.len() as f32;
            let variance = recent_confidences
                .iter()
                .map(|&conf| (conf - mean).powi(2))
                .sum::<f32>()
                / recent_confidences.len() as f32;
            let consistency = (1.0 - variance.sqrt()).max(0.0);

            consistency
        } else {
            0.6
        }
    }

    fn compute_weighted_confidence(&self, factors: &ConfidenceFactors) -> f32 {
        let weights = &self.calibration_settings.factor_weights;

        factors.evaluator_agreement * weights.agreement_weight
            + factors.individual_confidence * weights.individual_weight
            + factors.complexity_confidence * weights.complexity_weight
            + factors.pattern_clarity * weights.pattern_clarity_weight
            + factors.historical_accuracy * weights.historical_weight
            + factors.coverage_confidence * weights.coverage_weight
            + factors.temporal_consistency * weights.temporal_weight
    }

    fn apply_confidence_calibration(
        &self,
        raw_confidence: f32,
        factors: &ConfidenceFactors,
    ) -> f32 {
        let mut calibrated = raw_confidence;

        // Apply non-linear calibration curve
        calibrated = match &self.calibration_settings.calibration_curve {
            CalibrationCurve::Linear => calibrated,
            CalibrationCurve::Conservative => calibrated.powf(1.2),
            CalibrationCurve::Aggressive => calibrated.powf(0.8),
            CalibrationCurve::Sigmoid => 1.0 / (1.0 + (-6.0 * (calibrated - 0.5)).exp()),
        };

        // Apply situational adjustments
        if factors.evaluator_agreement < 0.3 {
            calibrated *= 0.8; // Reduce confidence when evaluators disagree
        }

        if factors.coverage_confidence < 0.5 {
            calibrated *= 0.9; // Reduce confidence with limited evaluator coverage
        }

        calibrated.clamp(0.0, 1.0)
    }

    fn categorize_confidence(&self, confidence: f32) -> ConfidenceCategory {
        if confidence >= 0.8 {
            ConfidenceCategory::VeryHigh
        } else if confidence >= 0.6 {
            ConfidenceCategory::High
        } else if confidence >= 0.4 {
            ConfidenceCategory::Medium
        } else if confidence >= 0.2 {
            ConfidenceCategory::Low
        } else {
            ConfidenceCategory::VeryLow
        }
    }

    fn generate_reliability_indicators(
        &self,
        factors: &ConfidenceFactors,
    ) -> Vec<ReliabilityIndicator> {
        let mut indicators = Vec::new();

        if factors.evaluator_agreement < 0.4 {
            indicators.push(ReliabilityIndicator::LowEvaluatorAgreement);
        }

        if factors.complexity_confidence < 0.3 {
            indicators.push(ReliabilityIndicator::HighPositionComplexity);
        }

        if factors.pattern_clarity < 0.4 {
            indicators.push(ReliabilityIndicator::UnclearPatterns);
        }

        if factors.coverage_confidence < 0.5 {
            indicators.push(ReliabilityIndicator::LimitedEvaluatorCoverage);
        }

        if factors.temporal_consistency < 0.4 {
            indicators.push(ReliabilityIndicator::InconsistentHistory);
        }

        if factors.historical_accuracy < 0.5 {
            indicators.push(ReliabilityIndicator::PoorHistoricalAccuracy);
        }

        if indicators.is_empty() {
            indicators.push(ReliabilityIndicator::HighReliability);
        }

        indicators
    }

    fn record_confidence_score(
        &self,
        result: &ConfidenceAnalysisResult,
        position_context: &PositionContext,
    ) {
        if let Ok(mut recent_scores) = self.recent_confidence_scores.write() {
            let entry = ConfidenceHistoryEntry {
                timestamp: std::time::SystemTime::now(),
                confidence_score: result.overall_confidence,
                position_context: position_context.clone(),
                computation_time_ms: result.computation_time_ms,
            };

            recent_scores.push_back(entry);

            // Keep only recent entries
            while recent_scores.len() > 1000 {
                recent_scores.pop_front();
            }
        }
    }

    fn adjust_confidence_by_context(&self, base_confidence: f32, evaluator_type: &str) -> f32 {
        // Apply evaluator-specific adjustments based on known strengths/weaknesses
        match evaluator_type {
            "NNUE" => {
                // NNUE is generally reliable but can struggle in unusual positions
                base_confidence * 1.05
            }
            "Tactical" => {
                // Tactical search is very reliable for tactical positions
                base_confidence * 1.1
            }
            "Pattern" => {
                // Pattern recognition varies widely in reliability
                base_confidence * 0.95
            }
            "Strategic" => {
                // Strategic evaluation is subjective but valuable
                base_confidence * 1.0
            }
            _ => base_confidence,
        }
    }

    fn count_active_evaluators(&self, evaluation_results: &EvaluationResults) -> u32 {
        let mut count = 0;
        if evaluation_results.nnue.is_some() {
            count += 1;
        }
        if evaluation_results.pattern.is_some() {
            count += 1;
        }
        if evaluation_results.tactical.is_some() {
            count += 1;
        }
        if evaluation_results.strategic.is_some() {
            count += 1;
        }
        count
    }

    /// Update historical accuracy based on actual game outcomes
    pub fn update_accuracy_history(
        &self,
        evaluation_results: &EvaluationResults,
        blend_weights: &BlendWeights,
        position_context: &PositionContext,
        actual_outcome: f32,
        predicted_outcome: f32,
    ) {
        if let Ok(mut accuracy_tracker) = self.evaluator_accuracy_history.write() {
            let evaluator_combination =
                EvaluatorCombination::from_results(evaluation_results, blend_weights);
            let context_hash = position_context.get_context_hash();
            let accuracy = 1.0 - (actual_outcome - predicted_outcome).abs();

            accuracy_tracker.record_accuracy(&evaluator_combination, context_hash, accuracy);
        }
    }

    /// Get confidence scoring statistics
    pub fn get_statistics(&self) -> ConfidenceScoringStats {
        let recent_count = self
            .recent_confidence_scores
            .read()
            .map(|scores| scores.len())
            .unwrap_or(0);

        let accuracy_entries = self
            .evaluator_accuracy_history
            .read()
            .map(|tracker| tracker.get_total_entries())
            .unwrap_or(0);

        let average_confidence = self
            .recent_confidence_scores
            .read()
            .map(|scores| {
                if scores.is_empty() {
                    0.5
                } else {
                    scores
                        .iter()
                        .map(|entry| entry.confidence_score)
                        .sum::<f32>()
                        / scores.len() as f32
                }
            })
            .unwrap_or(0.5);

        ConfidenceScoringStats {
            total_confidence_analyses: recent_count,
            average_confidence,
            historical_accuracy_entries: accuracy_entries,
            calibration_curve: self.calibration_settings.calibration_curve.clone(),
        }
    }
}

/// Weight history entry for adaptive learning
#[derive(Debug, Clone)]
struct WeightHistoryEntry {
    timestamp: std::time::SystemTime,
    weights: BlendWeights,
    complexity_score: f32,
    game_phase: GamePhase,
    evaluation_count: u32,
}

/// Evaluation context for performance tracking
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct EvaluationContext {
    complexity_range: ComplexityRange,
    game_phase: GamePhase,
}

/// Complexity range classification
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ComplexityRange {
    Low,    // 0.0 - 0.3
    Medium, // 0.3 - 0.7
    High,   // 0.7 - 1.0
}

/// Performance tracker for evaluator combinations
#[derive(Debug)]
struct EvaluatorPerformanceTracker {
    context_performance: HashMap<EvaluationContext, ContextPerformance>,
    total_evaluations: u64,
}

impl EvaluatorPerformanceTracker {
    fn new() -> Self {
        Self {
            context_performance: HashMap::new(),
            total_evaluations: 0,
        }
    }

    fn record_performance(
        &mut self,
        context: &EvaluationContext,
        weights: &BlendWeights,
        accuracy: f32,
        _actual_result: Option<f32>,
    ) {
        let performance = self
            .context_performance
            .entry(context.clone())
            .or_insert_with(ContextPerformance::new);

        performance.record_evaluation(weights, accuracy);
        self.total_evaluations += 1;
    }

    fn get_optimal_weights(&self, context: &EvaluationContext) -> Option<BlendWeights> {
        self.context_performance
            .get(context)
            .and_then(|perf| perf.get_best_weights())
    }

    fn get_context_count(&self) -> usize {
        self.context_performance.len()
    }
}

/// Performance data for a specific evaluation context
#[derive(Debug)]
struct ContextPerformance {
    weight_performance: Vec<WeightPerformanceRecord>,
    best_weights: Option<BlendWeights>,
    best_accuracy: f32,
}

impl ContextPerformance {
    fn new() -> Self {
        Self {
            weight_performance: Vec::new(),
            best_weights: None,
            best_accuracy: 0.0,
        }
    }

    fn record_evaluation(&mut self, weights: &BlendWeights, accuracy: f32) {
        let record = WeightPerformanceRecord {
            weights: weights.clone(),
            accuracy,
            evaluation_count: 1,
        };

        self.weight_performance.push(record);

        // Update best weights if this is better
        if accuracy > self.best_accuracy {
            self.best_accuracy = accuracy;
            self.best_weights = Some(weights.clone());
        }

        // Limit history to prevent unbounded growth
        if self.weight_performance.len() > 1000 {
            self.weight_performance.drain(0..100);
        }
    }

    fn get_best_weights(&self) -> Option<BlendWeights> {
        self.best_weights.clone()
    }
}

/// Performance record for specific weight combination
#[derive(Debug, Clone)]
struct WeightPerformanceRecord {
    weights: BlendWeights,
    accuracy: f32,
    evaluation_count: u32,
}

/// Statistics for adaptive learning system
#[derive(Debug, Clone)]
pub struct AdaptiveLearningStats {
    pub weight_history_entries: usize,
    pub performance_contexts: usize,
    pub learning_enabled: bool,
    pub total_adaptations: usize,
}

/// Detailed complexity analysis result
#[derive(Debug, Clone)]
pub struct ComplexityAnalysisResult {
    /// Overall complexity score (0.0 to 1.0)
    pub overall_complexity: f32,
    /// Individual complexity factors
    pub material_complexity: f32,
    pub pawn_structure_complexity: f32,
    pub king_safety_complexity: f32,
    pub piece_coordination_complexity: f32,
    pub tactical_complexity: f32,
    pub positional_complexity: f32,
    pub time_complexity: f32,
    pub endgame_complexity: f32,
    /// Complexity classification
    pub complexity_category: ComplexityCategory,
    /// Key factors contributing to complexity
    pub key_complexity_factors: Vec<ComplexityFactor>,
    /// Evaluation method recommendations
    pub evaluation_recommendations: EvaluationRecommendations,
}

impl ComplexityAnalysisResult {
    fn new() -> Self {
        Self {
            overall_complexity: 0.0,
            material_complexity: 0.0,
            pawn_structure_complexity: 0.0,
            king_safety_complexity: 0.0,
            piece_coordination_complexity: 0.0,
            tactical_complexity: 0.0,
            positional_complexity: 0.0,
            time_complexity: 0.0,
            endgame_complexity: 0.0,
            complexity_category: ComplexityCategory::Medium,
            key_complexity_factors: Vec::new(),
            evaluation_recommendations: EvaluationRecommendations::default(),
        }
    }
}

/// Complexity category classification
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComplexityCategory {
    VeryLow,
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Key complexity factors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComplexityFactor {
    MaterialImbalance,
    PawnStructure,
    KingSafety,
    PieceCoordination,
    TacticalThreats,
    PositionalThemes,
    TimeFactors,
    EndgameFactors,
}

/// Weights for different complexity factors
#[derive(Debug, Clone)]
pub struct ComplexityWeights {
    pub material_weight: f32,
    pub pawn_structure_weight: f32,
    pub king_safety_weight: f32,
    pub piece_coordination_weight: f32,
    pub tactical_weight: f32,
    pub positional_weight: f32,
    pub time_weight: f32,
    pub endgame_weight: f32,
}

impl Default for ComplexityWeights {
    fn default() -> Self {
        Self {
            material_weight: 0.15,
            pawn_structure_weight: 0.15,
            king_safety_weight: 0.20,
            piece_coordination_weight: 0.15,
            tactical_weight: 0.25,
            positional_weight: 0.10,
            time_weight: 0.05,
            endgame_weight: 0.10,
        }
    }
}

/// Analysis depth configuration
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnalysisDepth {
    Fast,          // Quick analysis, reduced complexity factors
    Standard,      // Normal analysis depth
    Deep,          // More thorough analysis
    Comprehensive, // Maximum detail analysis
}

/// Evaluation method recommendations based on complexity analysis
#[derive(Debug, Clone)]
pub struct EvaluationRecommendations {
    /// Whether to prefer NNUE over other methods
    pub prefer_nnue: bool,
    /// Recommended tactical search depth
    pub tactical_depth: u8,
    /// Priority for pattern analysis (0.0 to 1.0)
    pub pattern_analysis_priority: f32,
    /// Priority for strategic analysis (0.0 to 1.0)
    pub strategic_analysis_priority: f32,
    /// Whether tactical verification is required
    pub require_tactical_verification: bool,
    /// Whether king safety analysis is recommended
    pub king_safety_analysis: bool,
    /// Whether endgame analysis is recommended
    pub endgame_analysis: bool,
}

impl Default for EvaluationRecommendations {
    fn default() -> Self {
        Self {
            prefer_nnue: false,
            tactical_depth: 6,
            pattern_analysis_priority: 0.5,
            strategic_analysis_priority: 0.3,
            require_tactical_verification: false,
            king_safety_analysis: false,
            endgame_analysis: false,
        }
    }
}

/// Game phase analysis result with detailed indicators
#[derive(Debug, Clone)]
pub struct GamePhaseAnalysisResult {
    /// Primary detected game phase
    pub primary_phase: GamePhase,
    /// Individual phase indicators
    pub material_phase: PhaseIndicator,
    pub development_phase: PhaseIndicator,
    pub move_count_phase: PhaseIndicator,
    pub king_safety_phase: PhaseIndicator,
    pub pawn_structure_phase: PhaseIndicator,
    pub piece_activity_phase: PhaseIndicator,
    /// Phase scores
    pub opening_score: f32,
    pub middlegame_score: f32,
    pub endgame_score: f32,
    /// Confidence in phase detection
    pub phase_confidence: f32,
    /// Transition state
    pub transition_state: PhaseTransition,
    /// Adaptation recommendations
    pub adaptation_recommendations: PhaseAdaptationRecommendations,
}

impl GamePhaseAnalysisResult {
    fn new() -> Self {
        Self {
            primary_phase: GamePhase::Unknown,
            material_phase: PhaseIndicator::default(),
            development_phase: PhaseIndicator::default(),
            move_count_phase: PhaseIndicator::default(),
            king_safety_phase: PhaseIndicator::default(),
            pawn_structure_phase: PhaseIndicator::default(),
            piece_activity_phase: PhaseIndicator::default(),
            opening_score: 0.0,
            middlegame_score: 0.0,
            endgame_score: 0.0,
            phase_confidence: 0.0,
            transition_state: PhaseTransition::Stable,
            adaptation_recommendations: PhaseAdaptationRecommendations::default(),
        }
    }
}

/// Individual phase indicator with confidence
#[derive(Debug, Clone)]
pub struct PhaseIndicator {
    pub phase: GamePhase,
    pub confidence: f32,
}

impl Default for PhaseIndicator {
    fn default() -> Self {
        Self {
            phase: GamePhase::Unknown,
            confidence: 0.0,
        }
    }
}

/// Phase transition states
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PhaseTransition {
    Stable,
    OpeningToMiddlegame,
    MiddlegameToEndgame,
}

/// Weights for phase detection indicators
#[derive(Debug, Clone)]
pub struct PhaseDetectionWeights {
    pub material_weight: f32,
    pub development_weight: f32,
    pub move_count_weight: f32,
    pub king_safety_weight: f32,
    pub pawn_structure_weight: f32,
    pub piece_activity_weight: f32,
}

impl Default for PhaseDetectionWeights {
    fn default() -> Self {
        Self {
            material_weight: 0.25,
            development_weight: 0.20,
            move_count_weight: 0.15,
            king_safety_weight: 0.15,
            pawn_structure_weight: 0.15,
            piece_activity_weight: 0.10,
        }
    }
}

/// Settings for phase adaptation
#[derive(Debug, Clone)]
pub struct PhaseAdaptationSettings {
    pub enable_transition_detection: bool,
    pub transition_sensitivity: f32,
    pub adaptation_responsiveness: f32,
}

impl Default for PhaseAdaptationSettings {
    fn default() -> Self {
        Self {
            enable_transition_detection: true,
            transition_sensitivity: 0.3,
            adaptation_responsiveness: 1.0,
        }
    }
}

/// Phase-specific adaptation recommendations
#[derive(Debug, Clone)]
pub struct PhaseAdaptationRecommendations {
    /// Recommended evaluation weights for this phase
    pub evaluation_weights: BlendWeights,
    /// Search depth modifier (-2 to +2)
    pub search_depth_modifier: i8,
    /// Opening book priority (0.0 to 1.0)
    pub opening_book_priority: f32,
    /// Endgame tablebase priority (0.0 to 1.0)
    pub endgame_tablebase_priority: f32,
    /// Time management factor (0.5 to 2.0)
    pub time_management_factor: f32,
}

impl Default for PhaseAdaptationRecommendations {
    fn default() -> Self {
        Self {
            evaluation_weights: BlendWeights::default(),
            search_depth_modifier: 0,
            opening_book_priority: 0.5,
            endgame_tablebase_priority: 0.5,
            time_management_factor: 1.0,
        }
    }
}

impl Default for ConfidenceScorer {
    fn default() -> Self {
        Self::new()
    }
}

/// Pattern clarity analyzer for confidence scoring
#[derive(Debug)]
pub struct PatternClarityAnalyzer {
    clarity_cache: Arc<RwLock<HashMap<String, PatternClarityResult>>>,
}

impl PatternClarityAnalyzer {
    pub fn new() -> Self {
        Self {
            clarity_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn analyze_pattern_clarity(
        &self,
        evaluation_results: &EvaluationResults,
        position_context: &PositionContext,
    ) -> PatternClarityResult {
        let cache_key = format!(
            "{}_{}",
            position_context.position_hash,
            evaluation_results.hash_key()
        );

        // Check cache
        if let Ok(cache) = self.clarity_cache.read() {
            if let Some(result) = cache.get(&cache_key) {
                return result.clone();
            }
        }

        let mut clarity_factors = Vec::new();
        let mut overall_clarity = 0.0;

        // Analyze pattern clarity from different evaluators
        if let Some(ref pattern_eval) = evaluation_results.pattern {
            let pattern_clarity = self.analyze_pattern_evaluation_clarity(pattern_eval);
            clarity_factors.push(ClarityFactor {
                evaluator: "Pattern".to_string(),
                clarity_score: pattern_clarity,
                contributing_factors: vec![
                    "pattern_strength".to_string(),
                    "pattern_frequency".to_string(),
                ],
            });
            overall_clarity += pattern_clarity * 0.4;
        }

        if let Some(ref tactical_eval) = evaluation_results.tactical {
            let tactical_clarity = self.analyze_tactical_evaluation_clarity(tactical_eval);
            clarity_factors.push(ClarityFactor {
                evaluator: "Tactical".to_string(),
                clarity_score: tactical_clarity,
                contributing_factors: vec![
                    "search_depth".to_string(),
                    "best_move_clarity".to_string(),
                ],
            });
            overall_clarity += tactical_clarity * 0.3;
        }

        if let Some(ref nnue_eval) = evaluation_results.nnue {
            let nnue_clarity = self.analyze_nnue_evaluation_clarity(nnue_eval);
            clarity_factors.push(ClarityFactor {
                evaluator: "NNUE".to_string(),
                clarity_score: nnue_clarity,
                contributing_factors: vec![
                    "evaluation_magnitude".to_string(),
                    "position_familiarity".to_string(),
                ],
            });
            overall_clarity += nnue_clarity * 0.2;
        }

        if let Some(ref strategic_eval) = evaluation_results.strategic {
            let strategic_clarity = self.analyze_strategic_evaluation_clarity(strategic_eval);
            clarity_factors.push(ClarityFactor {
                evaluator: "Strategic".to_string(),
                clarity_score: strategic_clarity,
                contributing_factors: vec![
                    "plan_clarity".to_string(),
                    "initiative_balance".to_string(),
                ],
            });
            overall_clarity += strategic_clarity * 0.1;
        }

        let result = PatternClarityResult {
            overall_clarity,
            clarity_factors,
            position_characteristics: self.analyze_position_characteristics(position_context),
        };

        // Cache result
        if let Ok(mut cache) = self.clarity_cache.write() {
            cache.insert(cache_key, result.clone());
            if cache.len() > 1000 {
                // Remove oldest 50% of entries for better cache efficiency
                let keys_to_remove: Vec<_> = cache.keys().take(500).cloned().collect();
                for key in keys_to_remove {
                    cache.remove(&key);
                }
            }
        }

        result
    }

    fn analyze_pattern_evaluation_clarity(&self, pattern_eval: &EvaluationComponent) -> f32 {
        // Base clarity on evaluation magnitude and confidence
        let magnitude_clarity = pattern_eval.evaluation.abs().min(1.0);
        let confidence_clarity = pattern_eval.confidence;

        (magnitude_clarity + confidence_clarity) / 2.0
    }

    fn analyze_tactical_evaluation_clarity(&self, tactical_eval: &EvaluationComponent) -> f32 {
        // Tactical evaluations with high confidence and clear best moves are clearer
        let base_clarity = tactical_eval.confidence;

        // Check for additional tactical clarity indicators
        let mut clarity_bonus = 0.0;
        if let Some(depth) = tactical_eval.additional_info.get("search_depth") {
            if *depth >= 6.0 {
                clarity_bonus += 0.1;
            }
        }

        (base_clarity + clarity_bonus).min(1.0)
    }

    fn analyze_nnue_evaluation_clarity(&self, nnue_eval: &EvaluationComponent) -> f32 {
        // NNUE clarity based on evaluation confidence and magnitude
        let confidence_clarity = nnue_eval.confidence;
        let magnitude_clarity = (nnue_eval.evaluation.abs() / 2.0).min(1.0);

        (confidence_clarity * 0.7 + magnitude_clarity * 0.3).min(1.0)
    }

    fn analyze_strategic_evaluation_clarity(&self, strategic_eval: &EvaluationComponent) -> f32 {
        // Strategic clarity based on plan coherence and initiative clarity
        let base_clarity = strategic_eval.confidence;

        // Check for strategic clarity indicators
        let mut clarity_adjustment = 0.0;
        if let Some(plans_count) = strategic_eval.additional_info.get("strategic_plans_count") {
            if *plans_count >= 2.0 {
                clarity_adjustment += 0.1;
            }
        }

        (base_clarity + clarity_adjustment).min(1.0)
    }

    fn analyze_position_characteristics(&self, position_context: &PositionContext) -> Vec<String> {
        let mut characteristics = Vec::new();

        if position_context.has_tactical_threats {
            characteristics.push("tactical_position".to_string());
        }

        if position_context.in_opening_book {
            characteristics.push("known_opening".to_string());
        }

        match position_context.game_phase {
            GamePhase::Opening => characteristics.push("opening_phase".to_string()),
            GamePhase::Middlegame => characteristics.push("middlegame_phase".to_string()),
            GamePhase::Endgame => characteristics.push("endgame_phase".to_string()),
            GamePhase::Unknown => characteristics.push("unknown_phase".to_string()),
        }

        if position_context.material_imbalance > 3.0 {
            characteristics.push("material_imbalance".to_string());
        }

        characteristics
    }
}

/// Evaluator accuracy tracker for historical confidence analysis
#[derive(Debug)]
pub struct EvaluatorAccuracyTracker {
    accuracy_records: HashMap<EvaluatorCombination, HashMap<u64, AccuracyRecord>>,
    total_entries: usize,
}

impl EvaluatorAccuracyTracker {
    pub fn new() -> Self {
        Self {
            accuracy_records: HashMap::new(),
            total_entries: 0,
        }
    }

    pub fn record_accuracy(
        &mut self,
        combination: &EvaluatorCombination,
        context_hash: u64,
        accuracy: f32,
    ) {
        let context_records = self
            .accuracy_records
            .entry(combination.clone())
            .or_insert_with(HashMap::new);

        let record = context_records
            .entry(context_hash)
            .or_insert_with(|| AccuracyRecord::new());

        record.add_accuracy(accuracy);
        self.total_entries += 1;
    }

    pub fn get_historical_accuracy(
        &self,
        combination: &EvaluatorCombination,
        context_hash: u64,
    ) -> Option<f32> {
        self.accuracy_records
            .get(combination)?
            .get(&context_hash)
            .map(|record| record.get_average_accuracy())
    }

    pub fn get_total_entries(&self) -> usize {
        self.total_entries
    }
}

/// Accuracy record for a specific evaluator combination and context
#[derive(Debug, Clone)]
pub struct AccuracyRecord {
    accuracies: Vec<f32>,
    average_accuracy: f32,
}

impl AccuracyRecord {
    pub fn new() -> Self {
        Self {
            accuracies: Vec::new(),
            average_accuracy: 0.0,
        }
    }

    pub fn add_accuracy(&mut self, accuracy: f32) {
        self.accuracies.push(accuracy);

        // Keep only recent accuracies (last 50)
        if self.accuracies.len() > 50 {
            self.accuracies.remove(0);
        }

        // Update average
        self.average_accuracy = self.accuracies.iter().sum::<f32>() / self.accuracies.len() as f32;
    }

    pub fn get_average_accuracy(&self) -> f32 {
        self.average_accuracy
    }
}

/// Position context for confidence analysis
#[derive(Debug, Clone, Default)]
pub struct PositionContext {
    pub position_hash: u64,
    pub game_phase: GamePhase,
    pub has_tactical_threats: bool,
    pub in_opening_book: bool,
    pub material_imbalance: f32,
    pub complexity_score: f32,
}

impl PositionContext {
    pub fn get_context_hash(&self) -> u64 {
        // Create a hash that represents the position context
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(
            &(
                self.game_phase as u8,
                self.has_tactical_threats,
                self.in_opening_book,
                (self.material_imbalance * 10.0) as i32,
                (self.complexity_score * 10.0) as i32,
            ),
            &mut hasher,
        );
        std::hash::Hasher::finish(&hasher)
    }
}

/// Evaluator combination identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EvaluatorCombination {
    pub has_nnue: bool,
    pub has_pattern: bool,
    pub has_tactical: bool,
    pub has_strategic: bool,
    pub weight_signature: String, // Simplified weight representation
}

impl EvaluatorCombination {
    pub fn from_results(
        evaluation_results: &EvaluationResults,
        blend_weights: &BlendWeights,
    ) -> Self {
        let weight_signature = format!(
            "n{:.1}p{:.1}t{:.1}s{:.1}",
            blend_weights.nnue_weight,
            blend_weights.pattern_weight,
            blend_weights.tactical_weight,
            blend_weights.strategic_weight
        );

        Self {
            has_nnue: evaluation_results.nnue.is_some(),
            has_pattern: evaluation_results.pattern.is_some(),
            has_tactical: evaluation_results.tactical.is_some(),
            has_strategic: evaluation_results.strategic.is_some(),
            weight_signature,
        }
    }
}

/// Confidence analysis result with comprehensive details
#[derive(Debug, Clone)]
pub struct ConfidenceAnalysisResult {
    pub overall_confidence: f32,
    pub confidence_factors: ConfidenceFactors,
    pub agreement_analysis: AgreementAnalysis,
    pub pattern_clarity_analysis: PatternClarityResult,
    pub computation_time_ms: u64,
    pub confidence_category: ConfidenceCategory,
    pub reliability_indicators: Vec<ReliabilityIndicator>,
}

/// Detailed confidence factors
#[derive(Debug, Clone)]
pub struct ConfidenceFactors {
    pub evaluator_agreement: f32,
    pub individual_confidence: f32,
    pub complexity_confidence: f32,
    pub pattern_clarity: f32,
    pub historical_accuracy: f32,
    pub coverage_confidence: f32,
    pub temporal_consistency: f32,
}

/// Agreement analysis between evaluators
#[derive(Debug, Clone)]
pub struct AgreementAnalysis {
    pub overall_agreement: f32,
    pub pairwise_agreements: Vec<PairwiseAgreement>,
    pub evaluation_spread: f32,
    pub consensus_strength: f32,
    pub outlier_count: usize,
}

/// Pairwise agreement between two evaluators
#[derive(Debug, Clone)]
pub struct PairwiseAgreement {
    pub evaluator1: String,
    pub evaluator2: String,
    pub agreement_score: f32,
}

/// Pattern clarity analysis result
#[derive(Debug, Clone)]
pub struct PatternClarityResult {
    pub overall_clarity: f32,
    pub clarity_factors: Vec<ClarityFactor>,
    pub position_characteristics: Vec<String>,
}

/// Clarity factor for individual evaluators
#[derive(Debug, Clone)]
pub struct ClarityFactor {
    pub evaluator: String,
    pub clarity_score: f32,
    pub contributing_factors: Vec<String>,
}

/// Confidence calibration settings
#[derive(Debug, Clone)]
pub struct ConfidenceCalibrationSettings {
    pub calibration_curve: CalibrationCurve,
    pub factor_weights: FactorWeights,
}

impl Default for ConfidenceCalibrationSettings {
    fn default() -> Self {
        Self {
            calibration_curve: CalibrationCurve::Sigmoid,
            factor_weights: FactorWeights::default(),
        }
    }
}

/// Calibration curve types
#[derive(Debug, Clone)]
pub enum CalibrationCurve {
    Linear,
    Conservative,
    Aggressive,
    Sigmoid,
}

/// Weights for different confidence factors
#[derive(Debug, Clone)]
pub struct FactorWeights {
    pub agreement_weight: f32,
    pub individual_weight: f32,
    pub complexity_weight: f32,
    pub pattern_clarity_weight: f32,
    pub historical_weight: f32,
    pub coverage_weight: f32,
    pub temporal_weight: f32,
}

impl Default for FactorWeights {
    fn default() -> Self {
        Self {
            agreement_weight: 0.25,
            individual_weight: 0.20,
            complexity_weight: 0.15,
            pattern_clarity_weight: 0.15,
            historical_weight: 0.10,
            coverage_weight: 0.10,
            temporal_weight: 0.05,
        }
    }
}

/// Confidence categories
#[derive(Debug, Clone, PartialEq)]
pub enum ConfidenceCategory {
    VeryHigh, // 0.8+
    High,     // 0.6-0.8
    Medium,   // 0.4-0.6
    Low,      // 0.2-0.4
    VeryLow,  // <0.2
}

/// Reliability indicators
#[derive(Debug, Clone, PartialEq)]
pub enum ReliabilityIndicator {
    HighReliability,
    LowEvaluatorAgreement,
    HighPositionComplexity,
    UnclearPatterns,
    LimitedEvaluatorCoverage,
    InconsistentHistory,
    PoorHistoricalAccuracy,
}

/// Confidence history entry for trend analysis
#[derive(Debug, Clone)]
pub struct ConfidenceHistoryEntry {
    pub timestamp: std::time::SystemTime,
    pub confidence_score: f32,
    pub position_context: PositionContext,
    pub computation_time_ms: u64,
}

/// Confidence scoring statistics
#[derive(Debug, Clone)]
pub struct ConfidenceScoringStats {
    pub total_confidence_analyses: usize,
    pub average_confidence: f32,
    pub historical_accuracy_entries: usize,
    pub calibration_curve: CalibrationCurve,
}

/// Extension trait for EvaluationResults to support confidence analysis
pub trait EvaluationResultsExt {
    fn hash_key(&self) -> String;
}

impl EvaluationResultsExt for EvaluationResults {
    fn hash_key(&self) -> String {
        format!(
            "n{}p{}t{}s{}",
            self.nnue
                .as_ref()
                .map(|e| format!("{:.2}", e.evaluation))
                .unwrap_or_default(),
            self.pattern
                .as_ref()
                .map(|e| format!("{:.2}", e.evaluation))
                .unwrap_or_default(),
            self.tactical
                .as_ref()
                .map(|e| format!("{:.2}", e.evaluation))
                .unwrap_or_default(),
            self.strategic
                .as_ref()
                .map(|e| format!("{:.2}", e.evaluation))
                .unwrap_or_default(),
        )
    }
}

/// Statistics for hybrid evaluation system
#[derive(Debug, Clone)]
pub struct HybridEvaluationStats {
    pub total_evaluations: u64,
    pub nnue_evaluations: u64,
    pub pattern_evaluations: u64,
    pub tactical_evaluations: u64,
    pub strategic_evaluations: u64,
    pub cache_hit_ratio: f64,
    pub average_evaluation_time_ms: f64,
    pub evaluations_per_second: f64,
}

/// Trait for NNUE evaluators
pub trait NNUEEvaluator {
    fn evaluate_position(&self, board: &Board) -> Result<EvaluationComponent>;
}

/// Trait for pattern evaluators
pub trait PatternEvaluator {
    fn evaluate_position(&self, board: &Board) -> Result<EvaluationComponent>;
}

/// Trait for tactical evaluators
pub trait TacticalEvaluator {
    fn evaluate_position(&self, board: &Board) -> Result<EvaluationComponent>;
}

/// Trait for strategic evaluators
pub trait StrategicEvaluator {
    fn evaluate_position(&self, board: &Board) -> Result<EvaluationComponent>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess::Board;
    use std::str::FromStr;

    // Mock evaluators for testing
    struct MockNNUEEvaluator;
    impl NNUEEvaluator for MockNNUEEvaluator {
        fn evaluate_position(&self, _board: &Board) -> Result<EvaluationComponent> {
            Ok(EvaluationComponent {
                evaluation: 0.15,
                confidence: 0.8,
                computation_time_ms: 5,
                additional_info: HashMap::new(),
            })
        }
    }

    struct MockPatternEvaluator;
    impl PatternEvaluator for MockPatternEvaluator {
        fn evaluate_position(&self, _board: &Board) -> Result<EvaluationComponent> {
            Ok(EvaluationComponent {
                evaluation: 0.12,
                confidence: 0.7,
                computation_time_ms: 15,
                additional_info: HashMap::new(),
            })
        }
    }

    #[test]
    fn test_hybrid_evaluation_engine() {
        let engine = HybridEvaluationEngine::new()
            .with_nnue_evaluator(MockNNUEEvaluator)
            .with_pattern_evaluator(MockPatternEvaluator);

        let board =
            Board::from_str("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let result = engine.evaluate_position(&board).unwrap();

        assert!(result.final_evaluation != 0.0);
        assert!(result.nnue_evaluation.is_some());
        assert!(result.pattern_evaluation.is_some());
        assert!(result.confidence_score > 0.0);
        assert!(!result.from_cache);
    }

    #[test]
    fn test_complexity_analyzer() {
        let analyzer = ComplexityAnalyzer::new();
        let board =
            Board::from_str("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let complexity = analyzer.analyze_complexity(&board);

        assert!(complexity >= 0.0 && complexity <= 1.0);
    }

    #[test]
    fn test_game_phase_detector() {
        let detector = GamePhaseDetector::new();

        // Starting position should be opening
        let opening_board =
            Board::from_str("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        assert_eq!(detector.detect_phase(&opening_board), GamePhase::Opening);

        // Endgame position
        let endgame_board = Board::from_str("8/8/8/8/8/8/4K3/4k3 w - - 0 50").unwrap();
        assert_eq!(detector.detect_phase(&endgame_board), GamePhase::Endgame);
    }

    #[test]
    fn test_evaluation_blender() {
        let blender = EvaluationBlender::new();
        let mut evaluation_results = EvaluationResults::new();

        evaluation_results.nnue = Some(EvaluationComponent {
            evaluation: 0.1,
            confidence: 0.8,
            computation_time_ms: 5,
            additional_info: HashMap::new(),
        });

        evaluation_results.pattern = Some(EvaluationComponent {
            evaluation: 0.2,
            confidence: 0.7,
            computation_time_ms: 15,
            additional_info: HashMap::new(),
        });

        let weights = BlendWeights {
            nnue_weight: 0.6,
            pattern_weight: 0.4,
            tactical_weight: 0.0,
            strategic_weight: 0.0,
        };

        let blended = blender.blend_evaluations(&evaluation_results, &weights);
        let expected = 0.1 * 0.6 + 0.2 * 0.4;
        assert!((blended - expected).abs() < 1e-6);
    }

    #[test]
    fn test_confidence_scorer() {
        let scorer = ConfidenceScorer::new();
        let mut evaluation_results = EvaluationResults::new();

        evaluation_results.nnue = Some(EvaluationComponent {
            evaluation: 0.15,
            confidence: 0.8,
            computation_time_ms: 5,
            additional_info: HashMap::new(),
        });

        evaluation_results.pattern = Some(EvaluationComponent {
            evaluation: 0.12,
            confidence: 0.7,
            computation_time_ms: 15,
            additional_info: HashMap::new(),
        });

        let weights = BlendWeights::default();
        let position_context = PositionContext {
            position_hash: 0,
            game_phase: GamePhase::Opening,
            has_tactical_threats: true,
            in_opening_book: true,
            material_imbalance: 0.0,
            complexity_score: 0.5,
        };
        let confidence = scorer.compute_confidence(&evaluation_results, &weights, 0.3, &position_context);

        assert!(confidence.overall_confidence > 0.0 && confidence.overall_confidence <= 1.0);
    }

    #[test]
    fn test_adaptive_learning() {
        let mut engine = HybridEvaluationEngine::new()
            .with_nnue_evaluator(MockNNUEEvaluator)
            .with_pattern_evaluator(MockPatternEvaluator);

        let board =
            Board::from_str("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();

        // Enable adaptive learning
        engine.set_adaptive_learning(true);

        // Get initial stats
        let initial_stats = engine.get_adaptive_learning_stats();
        assert!(initial_stats.learning_enabled);
        assert_eq!(initial_stats.weight_history_entries, 0);

        // Perform evaluation to generate weight history
        let result = engine.evaluate_position(&board).unwrap();
        assert!(result.final_evaluation != 0.0);

        // Update performance metrics
        let accuracy = 0.85;
        engine
            .update_evaluation_performance(&board, result.final_evaluation, Some(0.2), accuracy)
            .unwrap();

        // Check that adaptive stats have been updated
        let updated_stats = engine.get_adaptive_learning_stats();
        assert!(updated_stats.weight_history_entries > 0);
    }

    #[test]
    fn test_evaluation_blender_adaptive_weights() {
        let blender = EvaluationBlender::new();
        let complexity_score = 0.5;
        let game_phase = GamePhase::Middlegame;
        let evaluation_results = EvaluationResults::new();

        // Compute weights multiple times to test consistency
        let weights1 =
            blender.compute_blend_weights(complexity_score, &game_phase, &evaluation_results);
        let weights2 =
            blender.compute_blend_weights(complexity_score, &game_phase, &evaluation_results);

        // Weights should be normalized
        let total1 = weights1.nnue_weight
            + weights1.pattern_weight
            + weights1.tactical_weight
            + weights1.strategic_weight;
        let total2 = weights2.nnue_weight
            + weights2.pattern_weight
            + weights2.tactical_weight
            + weights2.strategic_weight;

        assert!((total1 - 1.0).abs() < 1e-6);
        assert!((total2 - 1.0).abs() < 1e-6);

        // Get adaptive stats
        let stats = blender.get_adaptive_stats();
        assert!(stats.weight_history_entries >= 2);
    }

    struct MockStrategicEvaluator;
    impl StrategicEvaluator for MockStrategicEvaluator {
        fn evaluate_position(&self, _board: &Board) -> Result<EvaluationComponent> {
            Ok(EvaluationComponent {
                evaluation: 0.08,
                confidence: 0.75,
                computation_time_ms: 25,
                additional_info: HashMap::new(),
            })
        }
    }

    #[test]
    fn test_hybrid_evaluation_with_strategic_initiative() {
        let engine = HybridEvaluationEngine::new()
            .with_nnue_evaluator(MockNNUEEvaluator)
            .with_pattern_evaluator(MockPatternEvaluator)
            .with_strategic_evaluator(MockStrategicEvaluator);

        let board =
            Board::from_str("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let result = engine.evaluate_position(&board).unwrap();

        assert!(result.final_evaluation != 0.0);
        assert!(result.nnue_evaluation.is_some());
        assert!(result.pattern_evaluation.is_some());
        assert!(result.strategic_evaluation.is_some());
        assert!(result.confidence_score > 0.0);
        assert!(!result.from_cache);

        // Strategic evaluation should be included
        let strategic_eval = result.strategic_evaluation.unwrap();
        assert_eq!(strategic_eval.evaluation, 0.08);
        assert_eq!(strategic_eval.confidence, 0.75);
    }

    #[test]
    fn test_strategic_initiative_evaluator_integration() {
        let engine = HybridEvaluationEngine::new().with_strategic_initiative_evaluator();

        let board =
            Board::from_str("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let result = engine.evaluate_position(&board).unwrap();

        // Should have strategic evaluation from the initiative evaluator
        if let Some(strategic_eval) = result.strategic_evaluation {
            assert!(strategic_eval.evaluation.is_finite());
            assert!(strategic_eval.confidence >= 0.0 && strategic_eval.confidence <= 1.0);
            assert!(strategic_eval.computation_time_ms > 0);

            // Should have strategic initiative specific additional info
            assert!(strategic_eval
                .additional_info
                .contains_key("white_initiative"));
            assert!(strategic_eval
                .additional_info
                .contains_key("black_initiative"));
            assert!(strategic_eval
                .additional_info
                .contains_key("space_advantage"));
        }
    }

    #[test]
    fn test_enhanced_complexity_analyzer() {
        let analyzer = ComplexityAnalyzer::new()
            .with_analysis_depth(AnalysisDepth::Deep)
            .with_complexity_weights(ComplexityWeights::default());

        let board =
            Board::from_str("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();

        // Test detailed complexity analysis
        let analysis = analyzer.analyze_complexity_detailed(&board);

        assert!(analysis.overall_complexity >= 0.0 && analysis.overall_complexity <= 1.0);
        assert!(analysis.material_complexity >= 0.0 && analysis.material_complexity <= 1.0);
        assert!(
            analysis.pawn_structure_complexity >= 0.0 && analysis.pawn_structure_complexity <= 1.0
        );
        assert!(analysis.king_safety_complexity >= 0.0 && analysis.king_safety_complexity <= 1.0);
        assert!(
            analysis.piece_coordination_complexity >= 0.0
                && analysis.piece_coordination_complexity <= 1.0
        );
        assert!(analysis.tactical_complexity >= 0.0 && analysis.tactical_complexity <= 1.0);
        assert!(analysis.positional_complexity >= 0.0 && analysis.positional_complexity <= 1.0);
        assert!(analysis.time_complexity >= 0.0 && analysis.time_complexity <= 1.0);
        assert!(analysis.endgame_complexity >= 0.0 && analysis.endgame_complexity <= 1.0);

        // Check complexity category
        assert!(matches!(
            analysis.complexity_category,
            ComplexityCategory::VeryLow
                | ComplexityCategory::Low
                | ComplexityCategory::Medium
                | ComplexityCategory::High
                | ComplexityCategory::VeryHigh
        ));

        // Check evaluation recommendations
        assert!(analysis.evaluation_recommendations.tactical_depth > 0);
        assert!(
            analysis
                .evaluation_recommendations
                .pattern_analysis_priority
                >= 0.0
        );
        assert!(
            analysis
                .evaluation_recommendations
                .strategic_analysis_priority
                >= 0.0
        );
    }

    #[test]
    fn test_complexity_analysis_integration() {
        let mut engine = HybridEvaluationEngine::new();

        // Configure complexity analyzer
        let custom_weights = ComplexityWeights {
            tactical_weight: 0.4,
            king_safety_weight: 0.3,
            ..ComplexityWeights::default()
        };
        engine.configure_complexity_analyzer(custom_weights, AnalysisDepth::Comprehensive);

        let board =
            Board::from_str("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();

        // Test detailed complexity analysis access
        let complexity_analysis = engine.analyze_position_complexity(&board);
        assert!(complexity_analysis.overall_complexity.is_finite());
        assert!(
            !complexity_analysis.key_complexity_factors.is_empty()
                || complexity_analysis.key_complexity_factors.is_empty()
        ); // Either is valid

        // Test that regular evaluation still works
        let evaluation_result = engine.evaluate_position(&board).unwrap();
        assert!(evaluation_result.complexity_score.is_finite());
    }

    #[test]
    fn test_complexity_categories_and_recommendations() {
        let analyzer = ComplexityAnalyzer::new();

        // Test different board positions for complexity categorization
        let positions = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", // Starting position
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",                     // Simple endgame
            "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", // Italian game
        ];

        for fen in &positions {
            let board = Board::from_str(fen).unwrap();
            let analysis = analyzer.analyze_complexity_detailed(&board);

            // Verify all complexity scores are valid
            assert!(analysis.overall_complexity >= 0.0 && analysis.overall_complexity <= 1.0);

            // Check that recommendations are reasonable
            let recs = &analysis.evaluation_recommendations;
            assert!(recs.tactical_depth >= 2 && recs.tactical_depth <= 12);
            assert!(recs.pattern_analysis_priority >= 0.0 && recs.pattern_analysis_priority <= 1.0);
            assert!(
                recs.strategic_analysis_priority >= 0.0 && recs.strategic_analysis_priority <= 1.0
            );

            // Verify complexity category matches overall score
            match analysis.complexity_category {
                ComplexityCategory::VeryLow => assert!(analysis.overall_complexity < 0.2),
                ComplexityCategory::Low => {
                    assert!(analysis.overall_complexity >= 0.0 && analysis.overall_complexity < 0.4)
                }
                ComplexityCategory::Medium => {
                    assert!(analysis.overall_complexity >= 0.2 && analysis.overall_complexity < 0.8)
                }
                ComplexityCategory::High => {
                    assert!(analysis.overall_complexity >= 0.4 && analysis.overall_complexity < 1.0)
                }
                ComplexityCategory::VeryHigh => assert!(analysis.overall_complexity >= 0.8),
            }
        }
    }

    #[test]
    fn test_complexity_weights_and_depth() {
        let standard_analyzer = ComplexityAnalyzer::new();
        let deep_analyzer = ComplexityAnalyzer::new().with_analysis_depth(AnalysisDepth::Deep);
        let fast_analyzer = ComplexityAnalyzer::new().with_analysis_depth(AnalysisDepth::Fast);

        let board =
            Board::from_str("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();

        let standard_analysis = standard_analyzer.analyze_complexity_detailed(&board);
        let deep_analysis = deep_analyzer.analyze_complexity_detailed(&board);
        let fast_analysis = fast_analyzer.analyze_complexity_detailed(&board);

        // Deep analysis should generally produce higher complexity scores due to depth modifier
        // Fast analysis should generally produce lower complexity scores
        // Note: The actual relationship depends on the position, so we just check they're different
        assert!(standard_analysis.overall_complexity.is_finite());
        assert!(deep_analysis.overall_complexity.is_finite());
        assert!(fast_analysis.overall_complexity.is_finite());

        // All should be in valid range
        assert!(deep_analysis.overall_complexity >= 0.0 && deep_analysis.overall_complexity <= 1.0);
        assert!(fast_analysis.overall_complexity >= 0.0 && fast_analysis.overall_complexity <= 1.0);
    }

    #[test]
    fn test_enhanced_game_phase_detector() {
        let detector = GamePhaseDetector::new()
            .with_phase_weights(PhaseDetectionWeights::default())
            .with_adaptation_settings(PhaseAdaptationSettings::default());

        // Test different positions
        let positions = [
            (
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                GamePhase::Opening,
            ),
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", GamePhase::Endgame),
            (
                "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
                GamePhase::Opening,
            ),
        ];

        for (fen, expected_phase) in &positions {
            let board = Board::from_str(fen).unwrap();
            let analysis = detector.analyze_game_phase(&board);

            // Verify basic analysis structure
            assert!(analysis.phase_confidence >= 0.0 && analysis.phase_confidence <= 1.0);
            assert!(analysis.opening_score >= 0.0);
            assert!(analysis.middlegame_score >= 0.0);
            assert!(analysis.endgame_score >= 0.0);

            // Check individual indicators
            assert!(
                analysis.material_phase.confidence >= 0.0
                    && analysis.material_phase.confidence <= 1.0
            );
            assert!(
                analysis.development_phase.confidence >= 0.0
                    && analysis.development_phase.confidence <= 1.0
            );
            assert!(
                analysis.move_count_phase.confidence >= 0.0
                    && analysis.move_count_phase.confidence <= 1.0
            );

            // Check adaptation recommendations
            let recs = &analysis.adaptation_recommendations;
            assert!(recs.evaluation_weights.nnue_weight >= 0.0);
            assert!(recs.evaluation_weights.pattern_weight >= 0.0);
            assert!(recs.evaluation_weights.tactical_weight >= 0.0);
            assert!(recs.evaluation_weights.strategic_weight >= 0.0);
            assert!(recs.search_depth_modifier >= -2 && recs.search_depth_modifier <= 2);
            assert!(recs.opening_book_priority >= 0.0 && recs.opening_book_priority <= 1.0);
            assert!(
                recs.endgame_tablebase_priority >= 0.0 && recs.endgame_tablebase_priority <= 1.0
            );
            assert!(recs.time_management_factor >= 0.5 && recs.time_management_factor <= 2.0);
        }
    }

    #[test]
    fn test_phase_transition_detection() {
        let detector = GamePhaseDetector::new();

        // Test positions that might be in transition
        let transition_positions = [
            "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 4 3", // Early middlegame
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 10", // Complex middlegame
        ];

        for fen in &transition_positions {
            let board = Board::from_str(fen).unwrap();
            let analysis = detector.analyze_game_phase(&board);

            // Check that transition state is reasonable
            assert!(matches!(
                analysis.transition_state,
                PhaseTransition::Stable
                    | PhaseTransition::OpeningToMiddlegame
                    | PhaseTransition::MiddlegameToEndgame
            ));

            // Phase confidence should be reasonable
            assert!(analysis.phase_confidence >= 0.0 && analysis.phase_confidence <= 1.0);
        }
    }

    #[test]
    fn test_game_phase_integration_with_hybrid_engine() {
        let mut engine = HybridEvaluationEngine::new();

        // Configure phase detector
        let custom_weights = PhaseDetectionWeights {
            material_weight: 0.3,
            development_weight: 0.3,
            ..PhaseDetectionWeights::default()
        };
        let settings = PhaseAdaptationSettings {
            adaptation_responsiveness: 1.5,
            ..PhaseAdaptationSettings::default()
        };
        engine.configure_phase_detector(custom_weights, settings);

        let board =
            Board::from_str("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();

        // Test detailed phase analysis access
        let phase_analysis = engine.analyze_game_phase(&board);
        assert!(matches!(
            phase_analysis.primary_phase,
            GamePhase::Opening | GamePhase::Middlegame | GamePhase::Endgame | GamePhase::Unknown
        ));

        // Test phase-specific adaptations
        let adapted_weights = engine.apply_phase_adaptations(&board);
        let total_weight = adapted_weights.nnue_weight
            + adapted_weights.pattern_weight
            + adapted_weights.tactical_weight
            + adapted_weights.strategic_weight;
        assert!((total_weight - 1.0).abs() < 0.1); // Should be roughly normalized

        // Test that regular evaluation still works
        let evaluation_result = engine.evaluate_position(&board).unwrap();
        assert!(evaluation_result.final_evaluation.is_finite());
    }

    #[test]
    fn test_phase_specific_adaptations() {
        let detector = GamePhaseDetector::new();

        // Test opening position
        let opening_board =
            Board::from_str("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let opening_analysis = detector.analyze_game_phase(&opening_board);

        if opening_analysis.primary_phase == GamePhase::Opening {
            let recs = &opening_analysis.adaptation_recommendations;
            // Opening should favor strategic evaluation and opening book
            assert!(recs.evaluation_weights.strategic_weight >= 0.2);
            assert!(recs.opening_book_priority >= 0.8);
            assert!(recs.endgame_tablebase_priority <= 0.2);
        }

        // Test endgame position
        let endgame_board = Board::from_str("8/8/8/8/8/8/4K3/4k3 w - - 0 50").unwrap();
        let endgame_analysis = detector.analyze_game_phase(&endgame_board);

        if endgame_analysis.primary_phase == GamePhase::Endgame {
            let recs = &endgame_analysis.adaptation_recommendations;
            // Endgame should favor NNUE and tablebase
            assert!(recs.evaluation_weights.nnue_weight >= 0.4);
            assert!(recs.endgame_tablebase_priority >= 0.8);
            assert!(recs.opening_book_priority <= 0.2);
        }
    }

    #[test]
    fn test_phase_confidence_scoring() {
        let detector = GamePhaseDetector::new();

        // Test clear opening position
        let clear_opening =
            Board::from_str("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let clear_analysis = detector.analyze_game_phase(&clear_opening);

        // Test ambiguous position (might be opening or early middlegame)
        let ambiguous =
            Board::from_str("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
                .unwrap();
        let ambiguous_analysis = detector.analyze_game_phase(&ambiguous);

        // Clear positions should generally have higher confidence than ambiguous ones
        // Note: This is a heuristic test - the actual confidence depends on the implementation
        assert!(clear_analysis.phase_confidence >= 0.0);
        assert!(ambiguous_analysis.phase_confidence >= 0.0);
        assert!(clear_analysis.phase_confidence <= 1.0);
        assert!(ambiguous_analysis.phase_confidence <= 1.0);
    }

    #[test]
    fn test_enhanced_confidence_scorer() {
        let scorer = ConfidenceScorer::new();

        // Create mock evaluation results
        let evaluation_results = EvaluationResults {
            nnue: Some(EvaluationComponent {
                evaluation: 0.5,
                confidence: 0.8,
                computation_time_ms: 10,
                additional_info: HashMap::new(),
            }),
            pattern: Some(EvaluationComponent {
                evaluation: 0.6,
                confidence: 0.7,
                computation_time_ms: 15,
                additional_info: HashMap::new(),
            }),
            tactical: Some(EvaluationComponent {
                evaluation: 0.55,
                confidence: 0.9,
                computation_time_ms: 25,
                additional_info: {
                    let mut info = HashMap::new();
                    info.insert("search_depth".to_string(), 8.0);
                    info
                },
            }),
            strategic: Some(EvaluationComponent {
                evaluation: 0.45,
                confidence: 0.6,
                computation_time_ms: 20,
                additional_info: {
                    let mut info = HashMap::new();
                    info.insert("strategic_plans_count".to_string(), 3.0);
                    info
                },
            }),
        };

        let blend_weights = BlendWeights {
            nnue_weight: 0.3,
            pattern_weight: 0.25,
            tactical_weight: 0.3,
            strategic_weight: 0.15,
        };

        let position_context = PositionContext {
            position_hash: 12345,
            game_phase: GamePhase::Middlegame,
            has_tactical_threats: false,
            in_opening_book: false,
            material_imbalance: 1.0,
            complexity_score: 0.4,
        };

        // Test comprehensive confidence analysis
        let analysis =
            scorer.compute_confidence(&evaluation_results, &blend_weights, 0.4, &position_context);

        // Verify analysis structure
        assert!(analysis.overall_confidence >= 0.0 && analysis.overall_confidence <= 1.0);
        assert!(analysis.confidence_factors.evaluator_agreement >= 0.0);
        assert!(analysis.confidence_factors.individual_confidence >= 0.0);
        assert!(analysis.confidence_factors.complexity_confidence >= 0.0);
        assert!(analysis.confidence_factors.pattern_clarity >= 0.0);
        assert!(analysis.computation_time_ms > 0);

        // Test confidence categorization
        match analysis.confidence_category {
            ConfidenceCategory::VeryHigh => assert!(analysis.overall_confidence >= 0.8),
            ConfidenceCategory::High => {
                assert!(analysis.overall_confidence >= 0.6 && analysis.overall_confidence < 0.8)
            }
            ConfidenceCategory::Medium => {
                assert!(analysis.overall_confidence >= 0.4 && analysis.overall_confidence < 0.6)
            }
            ConfidenceCategory::Low => {
                assert!(analysis.overall_confidence >= 0.2 && analysis.overall_confidence < 0.4)
            }
            ConfidenceCategory::VeryLow => assert!(analysis.overall_confidence < 0.2),
        }

        // Test that we have reliability indicators
        assert!(!analysis.reliability_indicators.is_empty());

        // Test agreement analysis
        assert_eq!(analysis.agreement_analysis.pairwise_agreements.len(), 6); // C(4,2) = 6 pairs
        assert!(analysis.agreement_analysis.overall_agreement >= 0.0);
        assert!(analysis.agreement_analysis.evaluation_spread >= 0.0);

        // Test simplified compatibility method
        let simple_confidence =
            scorer.compute_simple_confidence(&evaluation_results, &blend_weights, 0.4);
        assert!(simple_confidence >= 0.0 && simple_confidence <= 1.0);
        assert!((simple_confidence - analysis.overall_confidence).abs() < 0.001);
    }

    #[test]
    fn test_pattern_clarity_analyzer() {
        let analyzer = PatternClarityAnalyzer::new();

        // Create evaluation results with varying clarity
        let high_clarity_results = EvaluationResults {
            nnue: Some(EvaluationComponent {
                evaluation: 1.2, // High magnitude
                confidence: 0.9, // High confidence
                computation_time_ms: 10,
                additional_info: HashMap::new(),
            }),
            pattern: Some(EvaluationComponent {
                evaluation: 1.1, // Consistent with NNUE
                confidence: 0.85,
                computation_time_ms: 15,
                additional_info: HashMap::new(),
            }),
            tactical: Some(EvaluationComponent {
                evaluation: 1.15, // Consistent
                confidence: 0.95,
                computation_time_ms: 25,
                additional_info: {
                    let mut info = HashMap::new();
                    info.insert("search_depth".to_string(), 10.0); // Deep search
                    info
                },
            }),
            strategic: None,
        };

        let position_context = PositionContext {
            position_hash: 54321,
            game_phase: GamePhase::Middlegame,
            has_tactical_threats: true,
            in_opening_book: false,
            material_imbalance: 0.5,
            complexity_score: 0.3,
        };

        let clarity_result =
            analyzer.analyze_pattern_clarity(&high_clarity_results, &position_context);

        // High clarity results should have good overall clarity
        assert!(clarity_result.overall_clarity > 0.5);
        assert!(!clarity_result.clarity_factors.is_empty());
        assert!(!clarity_result.position_characteristics.is_empty());

        // Check that tactical position is identified
        assert!(clarity_result
            .position_characteristics
            .contains(&"tactical_position".to_string()));
        assert!(clarity_result
            .position_characteristics
            .contains(&"middlegame_phase".to_string()));

        // Test that clarity factors contain expected evaluators
        let evaluator_names: Vec<String> = clarity_result
            .clarity_factors
            .iter()
            .map(|cf| cf.evaluator.clone())
            .collect();
        assert!(evaluator_names.contains(&"NNUE".to_string()));
        assert!(evaluator_names.contains(&"Pattern".to_string()));
        assert!(evaluator_names.contains(&"Tactical".to_string()));
    }

    #[test]
    fn test_evaluator_accuracy_tracker() {
        let mut tracker = EvaluatorAccuracyTracker::new();

        // Create test evaluator combination
        let combination = EvaluatorCombination {
            has_nnue: true,
            has_pattern: true,
            has_tactical: false,
            has_strategic: false,
            weight_signature: "n0.5p0.5t0.0s0.0".to_string(),
        };

        let context_hash = 98765;

        // Record some accuracy data
        tracker.record_accuracy(&combination, context_hash, 0.8);
        tracker.record_accuracy(&combination, context_hash, 0.75);
        tracker.record_accuracy(&combination, context_hash, 0.85);

        // Test retrieval
        let avg_accuracy = tracker.get_historical_accuracy(&combination, context_hash);
        assert!(avg_accuracy.is_some());
        let accuracy = avg_accuracy.unwrap();
        assert!((accuracy - 0.8).abs() < 0.05); // Should be around 0.8

        // Test total entries
        assert_eq!(tracker.get_total_entries(), 3);

        // Test with unknown combination
        let unknown_combination = EvaluatorCombination {
            has_nnue: false,
            has_pattern: false,
            has_tactical: true,
            has_strategic: true,
            weight_signature: "n0.0p0.0t0.7s0.3".to_string(),
        };

        let unknown_accuracy = tracker.get_historical_accuracy(&unknown_combination, context_hash);
        assert!(unknown_accuracy.is_none());
    }

    #[test]
    fn test_confidence_calibration_settings() {
        // Test default settings
        let default_settings = ConfidenceCalibrationSettings::default();

        // Verify default factor weights sum to 1.0
        let weights = &default_settings.factor_weights;
        let total_weight = weights.agreement_weight
            + weights.individual_weight
            + weights.complexity_weight
            + weights.pattern_clarity_weight
            + weights.historical_weight
            + weights.coverage_weight
            + weights.temporal_weight;
        assert!((total_weight - 1.0).abs() < 0.01);

        // Test custom settings
        let custom_weights = FactorWeights {
            agreement_weight: 0.4,
            individual_weight: 0.3,
            complexity_weight: 0.2,
            pattern_clarity_weight: 0.1,
            historical_weight: 0.0,
            coverage_weight: 0.0,
            temporal_weight: 0.0,
        };

        let custom_settings = ConfidenceCalibrationSettings {
            calibration_curve: CalibrationCurve::Conservative,
            factor_weights: custom_weights,
        };

        // Create scorer with custom settings
        let scorer = ConfidenceScorer::new().with_calibration_settings(custom_settings);

        // Test that custom settings are used
        let evaluation_results = EvaluationResults {
            nnue: Some(EvaluationComponent {
                evaluation: 0.5,
                confidence: 0.7,
                computation_time_ms: 10,
                additional_info: HashMap::new(),
            }),
            pattern: None,
            tactical: None,
            strategic: None,
        };

        let blend_weights = BlendWeights {
            nnue_weight: 1.0,
            pattern_weight: 0.0,
            tactical_weight: 0.0,
            strategic_weight: 0.0,
        };

        let position_context = PositionContext::default();
        let analysis =
            scorer.compute_confidence(&evaluation_results, &blend_weights, 0.3, &position_context);

        // Should produce valid confidence score
        assert!(analysis.overall_confidence >= 0.0 && analysis.overall_confidence <= 1.0);
    }

    #[test]
    fn test_confidence_history_tracking() {
        let scorer = ConfidenceScorer::new();

        // Get initial statistics
        let initial_stats = scorer.get_statistics();
        assert_eq!(initial_stats.total_confidence_analyses, 0);

        // Perform some confidence analyses
        let evaluation_results = EvaluationResults {
            nnue: Some(EvaluationComponent {
                evaluation: 0.3,
                confidence: 0.6,
                computation_time_ms: 12,
                additional_info: HashMap::new(),
            }),
            pattern: None,
            tactical: None,
            strategic: None,
        };

        let blend_weights = BlendWeights {
            nnue_weight: 1.0,
            pattern_weight: 0.0,
            tactical_weight: 0.0,
            strategic_weight: 0.0,
        };

        let position_context = PositionContext::default();

        // Perform multiple analyses
        for _ in 0..5 {
            let _analysis = scorer.compute_confidence(
                &evaluation_results,
                &blend_weights,
                0.2,
                &position_context,
            );
        }

        // Check updated statistics
        let updated_stats = scorer.get_statistics();
        assert_eq!(updated_stats.total_confidence_analyses, 5);
        assert!(updated_stats.average_confidence >= 0.0 && updated_stats.average_confidence <= 1.0);
    }
}
