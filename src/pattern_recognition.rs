use crate::errors::Result;
use crate::hybrid_evaluation::{EvaluationComponent, PatternEvaluator};
use crate::utils::cache::PatternCache;
use chess::{Board, Color, Piece};
use ndarray::{s, Array1};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Advanced pattern recognition system for chess positions
pub struct AdvancedPatternRecognizer {
    /// Individual pattern recognizers
    pawn_structure_recognizer: PawnStructureRecognizer,
    piece_coordination_recognizer: PieceCoordinationRecognizer,
    king_safety_recognizer: KingSafetyRecognizer,
    tactical_pattern_recognizer: TacticalPatternRecognizer,
    endgame_pattern_recognizer: EndgamePatternRecognizer,

    /// Pattern weights for different game phases
    pattern_weights: PatternWeights,

    /// Pattern cache
    pattern_cache: Arc<PatternCache<String, PatternAnalysisResult>>,

    /// Learned pattern database
    learned_patterns: Arc<RwLock<LearnedPatternDatabase>>,
}

impl AdvancedPatternRecognizer {
    /// Create a new pattern recognizer
    pub fn new() -> Self {
        Self {
            pawn_structure_recognizer: PawnStructureRecognizer::new(),
            piece_coordination_recognizer: PieceCoordinationRecognizer::new(),
            king_safety_recognizer: KingSafetyRecognizer::new(),
            tactical_pattern_recognizer: TacticalPatternRecognizer::new(),
            endgame_pattern_recognizer: EndgamePatternRecognizer::new(),
            pattern_weights: PatternWeights::default(),
            pattern_cache: Arc::new(PatternCache::new(10000)),
            learned_patterns: Arc::new(RwLock::new(LearnedPatternDatabase::new())),
        }
    }

    /// Analyze all patterns in a position
    pub fn analyze_patterns(&self, board: &Board) -> Result<PatternAnalysisResult> {
        let fen = board.to_string();

        // Check cache first
        if let Some(cached_result) = self.pattern_cache.get(&fen) {
            return Ok(cached_result);
        }

        let mut result = PatternAnalysisResult::new();

        // Pawn structure analysis
        result.pawn_structure = self.pawn_structure_recognizer.analyze(board)?;

        // Piece coordination analysis
        result.piece_coordination = self.piece_coordination_recognizer.analyze(board)?;

        // King safety analysis
        result.king_safety = self.king_safety_recognizer.analyze(board)?;

        // Tactical pattern analysis
        result.tactical_patterns = self.tactical_pattern_recognizer.analyze(board)?;

        // Endgame pattern analysis
        result.endgame_patterns = self.endgame_pattern_recognizer.analyze(board)?;

        // Compute overall pattern score
        result.overall_score = self.compute_overall_score(&result);

        // Check for learned patterns
        result.learned_patterns = self.check_learned_patterns(board)?;

        // Cache the result
        self.pattern_cache.insert(fen, result.clone());

        Ok(result)
    }

    /// Compute vector representation of patterns
    pub fn encode_patterns(&self, board: &Board) -> Result<Array1<f32>> {
        let analysis = self.analyze_patterns(board)?;

        // Create a 256-dimensional pattern vector
        let mut pattern_vector = Array1::zeros(256);

        // Encode pawn structure features (64 dimensions)
        self.encode_pawn_structure(
            &analysis.pawn_structure,
            pattern_vector.slice_mut(s![0..64]),
        );

        // Encode piece coordination features (64 dimensions)
        self.encode_piece_coordination(
            &analysis.piece_coordination,
            pattern_vector.slice_mut(s![64..128]),
        );

        // Encode king safety features (32 dimensions)
        self.encode_king_safety(
            &analysis.king_safety,
            pattern_vector.slice_mut(s![128..160]),
        );

        // Encode tactical patterns (32 dimensions)
        self.encode_tactical_patterns(
            &analysis.tactical_patterns,
            pattern_vector.slice_mut(s![160..192]),
        );

        // Encode endgame patterns (32 dimensions)
        self.encode_endgame_patterns(
            &analysis.endgame_patterns,
            pattern_vector.slice_mut(s![192..224]),
        );

        // Encode learned patterns (32 dimensions)
        self.encode_learned_patterns(
            &analysis.learned_patterns,
            pattern_vector.slice_mut(s![224..256]),
        );

        Ok(pattern_vector)
    }

    fn compute_overall_score(&self, analysis: &PatternAnalysisResult) -> f32 {
        let mut score = 0.0;

        score += analysis.pawn_structure.score * self.pattern_weights.pawn_structure;
        score += analysis.piece_coordination.score * self.pattern_weights.piece_coordination;
        score += analysis.king_safety.score * self.pattern_weights.king_safety;
        score += analysis.tactical_patterns.score * self.pattern_weights.tactical_patterns;
        score += analysis.endgame_patterns.score * self.pattern_weights.endgame_patterns;

        // Add learned pattern contributions
        for pattern_match in &analysis.learned_patterns {
            score += pattern_match.strength * pattern_match.confidence * 0.1;
        }

        score
    }

    fn check_learned_patterns(&self, board: &Board) -> Result<Vec<LearnedPatternMatch>> {
        if let Ok(database) = self.learned_patterns.read() {
            database.find_matching_patterns(board)
        } else {
            Ok(Vec::new())
        }
    }

    fn encode_pawn_structure(
        &self,
        pawn_analysis: &PawnStructureAnalysis,
        mut slice: ndarray::ArrayViewMut1<f32>,
    ) {
        slice[0] = pawn_analysis.isolated_pawns as f32 / 8.0;
        slice[1] = pawn_analysis.doubled_pawns as f32 / 8.0;
        slice[2] = pawn_analysis.passed_pawns as f32 / 8.0;
        slice[3] = pawn_analysis.backward_pawns as f32 / 8.0;
        slice[4] = pawn_analysis.pawn_islands as f32 / 8.0;
        slice[5] = pawn_analysis.pawn_chains as f32 / 8.0;
        slice[6] = pawn_analysis.pawn_storm_potential;
        slice[7] = pawn_analysis.pawn_shield_quality;

        // Encode pawn structure patterns (remaining 56 dimensions)
        for i in 8..64 {
            slice[i] = 0.0; // Placeholder for additional pawn features
        }
    }

    fn encode_piece_coordination(
        &self,
        coord_analysis: &PieceCoordinationAnalysis,
        mut slice: ndarray::ArrayViewMut1<f32>,
    ) {
        slice[0] = coord_analysis.piece_activity;
        slice[1] = coord_analysis.piece_harmony;
        slice[2] = coord_analysis.central_control;
        slice[3] = coord_analysis.outpost_strength;
        slice[4] = coord_analysis.piece_mobility;
        slice[5] = coord_analysis.piece_safety;

        // Encode coordination patterns (remaining 58 dimensions)
        for i in 6..64 {
            slice[i] = 0.0; // Placeholder for additional coordination features
        }
    }

    fn encode_king_safety(
        &self,
        safety_analysis: &KingSafetyAnalysis,
        mut slice: ndarray::ArrayViewMut1<f32>,
    ) {
        slice[0] = safety_analysis.king_exposure;
        slice[1] = safety_analysis.attacking_pieces;
        slice[2] = safety_analysis.escape_squares;
        slice[3] = safety_analysis.pawn_shield;
        slice[4] = safety_analysis.king_tropism;

        // Encode additional king safety features (remaining 27 dimensions)
        for i in 5..32 {
            slice[i] = 0.0; // Placeholder
        }
    }

    fn encode_tactical_patterns(
        &self,
        tactical_analysis: &TacticalPatternAnalysis,
        mut slice: ndarray::ArrayViewMut1<f32>,
    ) {
        slice[0] = if tactical_analysis.pins > 0 { 1.0 } else { 0.0 };
        slice[1] = if tactical_analysis.forks > 0 {
            1.0
        } else {
            0.0
        };
        slice[2] = if tactical_analysis.skewers > 0 {
            1.0
        } else {
            0.0
        };
        slice[3] = if tactical_analysis.discovered_attacks > 0 {
            1.0
        } else {
            0.0
        };
        slice[4] = tactical_analysis.hanging_pieces as f32 / 16.0;
        slice[5] = tactical_analysis.undefended_pieces as f32 / 16.0;

        // Encode additional tactical features (remaining 26 dimensions)
        for i in 6..32 {
            slice[i] = 0.0; // Placeholder
        }
    }

    fn encode_endgame_patterns(
        &self,
        endgame_analysis: &EndgamePatternAnalysis,
        mut slice: ndarray::ArrayViewMut1<f32>,
    ) {
        slice[0] = if endgame_analysis.opposition {
            1.0
        } else {
            0.0
        };
        slice[1] = if endgame_analysis.zugzwang_potential {
            1.0
        } else {
            0.0
        };
        slice[2] = endgame_analysis.king_activity;
        slice[3] = endgame_analysis.pawn_majority_value;
        slice[4] = endgame_analysis.piece_vs_pawns_evaluation;

        // Encode additional endgame features (remaining 27 dimensions)
        for i in 5..32 {
            slice[i] = 0.0; // Placeholder
        }
    }

    fn encode_learned_patterns(
        &self,
        learned_matches: &[LearnedPatternMatch],
        mut slice: ndarray::ArrayViewMut1<f32>,
    ) {
        // Encode up to 16 learned pattern matches (2 dimensions each)
        for (i, pattern_match) in learned_matches.iter().take(16).enumerate() {
            slice[i * 2] = pattern_match.strength;
            slice[i * 2 + 1] = pattern_match.confidence;
        }

        // Fill remaining dimensions with zeros
        for i in (learned_matches.len().min(16) * 2)..32 {
            slice[i] = 0.0;
        }
    }

    /// Train the pattern recognizer with position-evaluation pairs
    pub fn train_patterns(&self, training_data: &[(Board, f32)]) -> Result<()> {
        if let Ok(mut database) = self.learned_patterns.write() {
            database.learn_from_data(training_data)?;
        }
        Ok(())
    }

    /// Get pattern recognition statistics
    pub fn get_statistics(&self) -> PatternRecognitionStats {
        let cache_stats = self.pattern_cache.stats();
        let learned_count = self
            .learned_patterns
            .read()
            .map(|db| db.pattern_count())
            .unwrap_or(0);

        PatternRecognitionStats {
            cached_analyses: cache_stats.cache_size,
            learned_patterns: learned_count,
            cache_hit_ratio: cache_stats.cache_hit_ratio,
        }
    }
}

impl Default for AdvancedPatternRecognizer {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternEvaluator for AdvancedPatternRecognizer {
    fn evaluate_position(&self, board: &Board) -> Result<EvaluationComponent> {
        let start_time = std::time::Instant::now();
        let analysis = self.analyze_patterns(board)?;
        let computation_time = start_time.elapsed().as_millis() as u64;

        let mut additional_info = HashMap::new();
        additional_info.insert(
            "pawn_structure_score".to_string(),
            analysis.pawn_structure.score,
        );
        additional_info.insert(
            "piece_coordination_score".to_string(),
            analysis.piece_coordination.score,
        );
        additional_info.insert("king_safety_score".to_string(), analysis.king_safety.score);
        additional_info.insert(
            "tactical_score".to_string(),
            analysis.tactical_patterns.score,
        );
        additional_info.insert("endgame_score".to_string(), analysis.endgame_patterns.score);
        additional_info.insert(
            "learned_patterns_count".to_string(),
            analysis.learned_patterns.len() as f32,
        );

        // Compute confidence based on pattern clarity and agreement
        let confidence = self.compute_pattern_confidence(&analysis);

        Ok(EvaluationComponent {
            evaluation: analysis.overall_score,
            confidence,
            computation_time_ms: computation_time,
            additional_info,
        })
    }
}

impl AdvancedPatternRecognizer {
    fn compute_pattern_confidence(&self, analysis: &PatternAnalysisResult) -> f32 {
        let mut confidence_factors = Vec::new();

        // Confidence based on pattern strength
        confidence_factors.push(analysis.pawn_structure.score.abs().min(1.0));
        confidence_factors.push(analysis.piece_coordination.score.abs().min(1.0));
        confidence_factors.push(analysis.king_safety.score.abs().min(1.0));

        // Confidence based on tactical clarity
        if analysis.tactical_patterns.pins > 0 || analysis.tactical_patterns.forks > 0 {
            confidence_factors.push(0.8);
        }

        // Confidence based on learned patterns
        let learned_confidence = analysis
            .learned_patterns
            .iter()
            .map(|p| p.confidence)
            .fold(0.0, f32::max);
        confidence_factors.push(learned_confidence);

        // Average confidence
        if confidence_factors.is_empty() {
            0.5
        } else {
            confidence_factors.iter().sum::<f32>() / confidence_factors.len() as f32
        }
    }
}

/// Comprehensive pattern analysis result
#[derive(Debug, Clone)]
pub struct PatternAnalysisResult {
    pub pawn_structure: PawnStructureAnalysis,
    pub piece_coordination: PieceCoordinationAnalysis,
    pub king_safety: KingSafetyAnalysis,
    pub tactical_patterns: TacticalPatternAnalysis,
    pub endgame_patterns: EndgamePatternAnalysis,
    pub learned_patterns: Vec<LearnedPatternMatch>,
    pub overall_score: f32,
}

impl PatternAnalysisResult {
    fn new() -> Self {
        Self {
            pawn_structure: PawnStructureAnalysis::default(),
            piece_coordination: PieceCoordinationAnalysis::default(),
            king_safety: KingSafetyAnalysis::default(),
            tactical_patterns: TacticalPatternAnalysis::default(),
            endgame_patterns: EndgamePatternAnalysis::default(),
            learned_patterns: Vec::new(),
            overall_score: 0.0,
        }
    }
}

/// Pawn structure recognizer
pub struct PawnStructureRecognizer;

impl PawnStructureRecognizer {
    pub fn new() -> Self {
        Self
    }

    pub fn analyze(&self, board: &Board) -> Result<PawnStructureAnalysis> {
        let mut analysis = PawnStructureAnalysis::default();

        // Analyze for both colors
        let white_analysis = self.analyze_color(board, Color::White);
        let black_analysis = self.analyze_color(board, Color::Black);

        // Compute relative scores (positive = good for white)
        analysis.isolated_pawns =
            (black_analysis.isolated_pawns - white_analysis.isolated_pawns) as i8;
        analysis.doubled_pawns =
            (black_analysis.doubled_pawns - white_analysis.doubled_pawns) as i8;
        analysis.passed_pawns = (white_analysis.passed_pawns - black_analysis.passed_pawns) as i8;
        analysis.backward_pawns =
            (black_analysis.backward_pawns - white_analysis.backward_pawns) as i8;
        analysis.pawn_islands = (black_analysis.pawn_islands - white_analysis.pawn_islands) as i8;
        analysis.pawn_chains = (white_analysis.pawn_chains - black_analysis.pawn_chains) as i8;

        // Compute overall pawn structure score
        analysis.score = self.compute_pawn_score(&analysis);

        // Additional pawn structure evaluations
        analysis.pawn_storm_potential = self.evaluate_pawn_storm(board);
        analysis.pawn_shield_quality = self.evaluate_pawn_shield(board);

        Ok(analysis)
    }

    fn analyze_color(&self, board: &Board, color: Color) -> PawnColorAnalysis {
        let pawns = board.pieces(Piece::Pawn) & board.color_combined(color);
        let mut analysis = PawnColorAnalysis::default();

        // Count pawns on each file
        let mut pawns_per_file = [0u8; 8];
        for square in pawns {
            pawns_per_file[square.get_file().to_index()] += 1;
        }

        // Count doubled pawns
        analysis.doubled_pawns = pawns_per_file.iter().filter(|&&count| count > 1).count();

        // Count isolated pawns
        for (file_idx, &pawn_count) in pawns_per_file.iter().enumerate() {
            if pawn_count > 0 {
                let has_neighbor = (file_idx > 0 && pawns_per_file[file_idx - 1] > 0)
                    || (file_idx < 7 && pawns_per_file[file_idx + 1] > 0);
                if !has_neighbor {
                    analysis.isolated_pawns += pawn_count as usize;
                }
            }
        }

        // Count pawn islands
        let mut in_island = false;
        for &pawn_count in &pawns_per_file {
            if pawn_count > 0 {
                if !in_island {
                    analysis.pawn_islands += 1;
                    in_island = true;
                }
            } else {
                in_island = false;
            }
        }

        // Count passed pawns (simplified)
        analysis.passed_pawns = self.count_passed_pawns(board, color);

        // Count backward pawns (simplified)
        analysis.backward_pawns = self.count_backward_pawns(board, color);

        // Count pawn chains (simplified)
        analysis.pawn_chains = self.count_pawn_chains(board, color);

        analysis
    }

    fn count_passed_pawns(&self, _board: &Board, _color: Color) -> usize {
        // Simplified implementation
        0
    }

    fn count_backward_pawns(&self, _board: &Board, _color: Color) -> usize {
        // Simplified implementation
        0
    }

    fn count_pawn_chains(&self, _board: &Board, _color: Color) -> usize {
        // Simplified implementation
        0
    }

    fn compute_pawn_score(&self, analysis: &PawnStructureAnalysis) -> f32 {
        let mut score = 0.0;

        // Isolated pawns are bad
        score -= analysis.isolated_pawns as f32 * 0.2;

        // Doubled pawns are bad
        score -= analysis.doubled_pawns as f32 * 0.15;

        // Passed pawns are good
        score += analysis.passed_pawns as f32 * 0.5;

        // Backward pawns are bad
        score -= analysis.backward_pawns as f32 * 0.1;

        // Too many pawn islands are bad
        score -= analysis.pawn_islands as f32 * 0.1;

        // Pawn chains are good
        score += analysis.pawn_chains as f32 * 0.15;

        score
    }

    fn evaluate_pawn_storm(&self, _board: &Board) -> f32 {
        // Simplified implementation
        0.0
    }

    fn evaluate_pawn_shield(&self, _board: &Board) -> f32 {
        // Simplified implementation
        0.0
    }
}

impl Default for PawnStructureRecognizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Other recognizers (simplified implementations for now)
pub struct PieceCoordinationRecognizer;
pub struct KingSafetyRecognizer;
pub struct TacticalPatternRecognizer;
pub struct EndgamePatternRecognizer;

impl PieceCoordinationRecognizer {
    pub fn new() -> Self {
        Self
    }
    pub fn analyze(&self, _board: &Board) -> Result<PieceCoordinationAnalysis> {
        Ok(PieceCoordinationAnalysis::default())
    }
}

impl KingSafetyRecognizer {
    pub fn new() -> Self {
        Self
    }
    pub fn analyze(&self, _board: &Board) -> Result<KingSafetyAnalysis> {
        Ok(KingSafetyAnalysis::default())
    }
}

impl TacticalPatternRecognizer {
    pub fn new() -> Self {
        Self
    }
    pub fn analyze(&self, _board: &Board) -> Result<TacticalPatternAnalysis> {
        Ok(TacticalPatternAnalysis::default())
    }
}

impl EndgamePatternRecognizer {
    pub fn new() -> Self {
        Self
    }
    pub fn analyze(&self, _board: &Board) -> Result<EndgamePatternAnalysis> {
        Ok(EndgamePatternAnalysis::default())
    }
}

/// Pattern analysis structures
#[derive(Debug, Clone, Default)]
pub struct PawnStructureAnalysis {
    pub isolated_pawns: i8,
    pub doubled_pawns: i8,
    pub passed_pawns: i8,
    pub backward_pawns: i8,
    pub pawn_islands: i8,
    pub pawn_chains: i8,
    pub pawn_storm_potential: f32,
    pub pawn_shield_quality: f32,
    pub score: f32,
}

#[derive(Debug, Clone, Default)]
struct PawnColorAnalysis {
    isolated_pawns: usize,
    doubled_pawns: usize,
    passed_pawns: usize,
    backward_pawns: usize,
    pawn_islands: usize,
    pawn_chains: usize,
}

#[derive(Debug, Clone, Default)]
pub struct PieceCoordinationAnalysis {
    pub piece_activity: f32,
    pub piece_harmony: f32,
    pub central_control: f32,
    pub outpost_strength: f32,
    pub piece_mobility: f32,
    pub piece_safety: f32,
    pub score: f32,
}

#[derive(Debug, Clone, Default)]
pub struct KingSafetyAnalysis {
    pub king_exposure: f32,
    pub attacking_pieces: f32,
    pub escape_squares: f32,
    pub pawn_shield: f32,
    pub king_tropism: f32,
    pub score: f32,
}

#[derive(Debug, Clone, Default)]
pub struct TacticalPatternAnalysis {
    pub pins: u8,
    pub forks: u8,
    pub skewers: u8,
    pub discovered_attacks: u8,
    pub hanging_pieces: u8,
    pub undefended_pieces: u8,
    pub score: f32,
}

#[derive(Debug, Clone, Default)]
pub struct EndgamePatternAnalysis {
    pub opposition: bool,
    pub zugzwang_potential: bool,
    pub king_activity: f32,
    pub pawn_majority_value: f32,
    pub piece_vs_pawns_evaluation: f32,
    pub score: f32,
}

/// Learned pattern database
pub struct LearnedPatternDatabase {
    patterns: Vec<LearnedPattern>,
    pattern_cache: HashMap<u64, Vec<usize>>, // Position hash -> pattern indices
}

impl LearnedPatternDatabase {
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
            pattern_cache: HashMap::new(),
        }
    }

    pub fn learn_from_data(&mut self, training_data: &[(Board, f32)]) -> Result<()> {
        // Simplified pattern learning
        for (board, evaluation) in training_data {
            if evaluation.abs() > 0.5 {
                // Only learn from decisive positions
                let pattern = LearnedPattern {
                    position_hash: board.get_hash(),
                    evaluation: *evaluation,
                    frequency: 1,
                    strength: evaluation.abs(),
                };
                self.patterns.push(pattern);
            }
        }
        Ok(())
    }

    pub fn find_matching_patterns(&self, board: &Board) -> Result<Vec<LearnedPatternMatch>> {
        let position_hash = board.get_hash();
        let mut matches = Vec::new();

        // Sophisticated pattern matching using both exact hash and similarity
        for (pattern_id, pattern) in self.patterns.iter().enumerate() {
            let mut match_confidence = 0.0;
            
            // Exact hash match gets full confidence
            if pattern.position_hash == position_hash {
                match_confidence = 1.0;
            } else {
                // Check for similar positions using Hamming distance
                let hash_diff = (pattern.position_hash ^ position_hash).count_ones();
                if hash_diff <= 3 {  // Allow small differences (piece moves, captures)
                    match_confidence = 1.0 - (hash_diff as f32 / 64.0);
                }
            }
            
            if match_confidence > 0.1 {  // Only include patterns with reasonable confidence
                matches.push(LearnedPatternMatch {
                    pattern_id: pattern_id as u32,
                    strength: pattern.strength * match_confidence,  // Scale strength by confidence
                    confidence: (pattern.frequency as f32 / 100.0).min(1.0) * match_confidence,
                });
            }
        }

        Ok(matches)
    }

    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }
}

#[derive(Debug, Clone)]
pub struct LearnedPattern {
    pub position_hash: u64,
    pub evaluation: f32,
    pub frequency: u32,
    pub strength: f32,
}

#[derive(Debug, Clone)]
pub struct LearnedPatternMatch {
    pub pattern_id: u32,
    pub strength: f32,
    pub confidence: f32,
}

/// Pattern weights for different aspects
#[derive(Debug, Clone)]
pub struct PatternWeights {
    pub pawn_structure: f32,
    pub piece_coordination: f32,
    pub king_safety: f32,
    pub tactical_patterns: f32,
    pub endgame_patterns: f32,
}

impl Default for PatternWeights {
    fn default() -> Self {
        Self {
            pawn_structure: 0.25,
            piece_coordination: 0.25,
            king_safety: 0.20,
            tactical_patterns: 0.20,
            endgame_patterns: 0.10,
        }
    }
}

/// Pattern recognition statistics
#[derive(Debug, Clone)]
pub struct PatternRecognitionStats {
    pub cached_analyses: usize,
    pub learned_patterns: usize,
    pub cache_hit_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess::Board;
    use std::str::FromStr;

    #[test]
    fn test_pattern_recognizer() {
        let recognizer = AdvancedPatternRecognizer::new();
        let board =
            Board::from_str("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();

        let analysis = recognizer.analyze_patterns(&board).unwrap();
        assert!(analysis.overall_score.is_finite());
    }

    #[test]
    fn test_pattern_encoding() {
        let recognizer = AdvancedPatternRecognizer::new();
        let board =
            Board::from_str("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();

        let pattern_vector = recognizer.encode_patterns(&board).unwrap();
        assert_eq!(pattern_vector.len(), 256);
        assert!(pattern_vector.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_pawn_structure_recognizer() {
        let recognizer = PawnStructureRecognizer::new();
        let board =
            Board::from_str("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();

        let analysis = recognizer.analyze(&board).unwrap();
        assert!(analysis.score.is_finite());
    }

    #[test]
    fn test_pattern_evaluator_trait() {
        let recognizer = AdvancedPatternRecognizer::new();
        let board =
            Board::from_str("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();

        let evaluation = recognizer.evaluate_position(&board).unwrap();
        assert!(evaluation.evaluation.is_finite());
        assert!(evaluation.confidence >= 0.0 && evaluation.confidence <= 1.0);
        assert!(evaluation.computation_time_ms > 0);
        assert!(!evaluation.additional_info.is_empty());
    }
}
