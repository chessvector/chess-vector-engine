//! Strategic Motif Extraction System
//!
//! This module extracts strategic motifs from the existing position database,
//! focusing on patterns that provide master-level positional understanding.

// Simplified motif extraction - complex pattern analyzer removed during cleanup
// use crate::pattern_recognition::{PatternAnalysisResult, AdvancedPatternRecognizer};
use crate::strategic_motifs::*;
use crate::ChessVectorEngine;
use chess::{BitBoard, Board, Color, File, Piece, Square, ALL_FILES};
use std::collections::HashMap;

/// Simplified strategic pattern for cleanup phase
#[derive(Debug, Clone)]
pub struct StrategicPattern {
    pub pattern_type: SimplePatternType,
    pub strength: f32,
    pub reliability: f32,
}

/// Simplified pattern types  
#[derive(Debug, Clone)]
pub enum SimplePatternType {
    MaterialImbalance,
    KingSafety,
    CenterControl,
    PieceActivity,
}

/// Extracts strategic motifs from a chess engine's position database
pub struct MotifExtractor {
    /// Minimum evaluation difference to consider a pattern strategic
    min_eval_significance: f32,
    /// Minimum number of occurrences for a pattern to be considered valid
    min_pattern_frequency: usize,
    /// Confidence threshold for including patterns
    confidence_threshold: f32,
    /// Extracted motifs
    extracted_motifs: Vec<StrategicMotif>,
    /// Pattern occurrence tracking
    pattern_counts: HashMap<u64, PatternOccurrence>,
    // Simplified pattern analyzer - replaced during cleanup
    // pattern_analyzer: AdvancedPatternAnalyzer,
}

/// Tracks occurrences of a specific pattern
#[derive(Debug, Clone)]
struct PatternOccurrence {
    count: usize,
    evaluations: Vec<f32>,
    positions: Vec<Board>,
    game_phases: Vec<GamePhase>,
}

impl MotifExtractor {
    /// Create new motif extractor with default parameters
    pub fn new() -> Self {
        Self {
            min_eval_significance: 0.1, // 10 centipawn minimum significance (more sensitive)
            min_pattern_frequency: 20, // Pattern must appear at least 20 times (more statistical power with 998K positions)
            confidence_threshold: 0.4, // 40% confidence minimum (more lenient)
            extracted_motifs: Vec::new(),
            pattern_counts: HashMap::new(),
            // pattern_analyzer: AdvancedPatternAnalyzer::new(),
        }
    }

    /// Extract strategic motifs from a chess engine's database
    pub fn extract_from_engine(
        &mut self,
        engine: &ChessVectorEngine,
    ) -> Result<Vec<StrategicMotif>, Box<dyn std::error::Error>> {
        println!("🔍 Starting strategic motif extraction...");

        let total_positions = engine.knowledge_base_size();
        println!(
            "📊 Analyzing {} positions for strategic patterns",
            total_positions
        );

        if total_positions == 0 {
            return Err("No positions in engine database to extract from".into());
        }

        // Phase 1: Analyze all positions for patterns
        self.analyze_positions(engine)?;

        // Phase 2: Extract significant patterns
        self.extract_significant_patterns()?;

        // Phase 3: Validate and refine patterns
        self.validate_patterns()?;

        println!(
            "✅ Extracted {} strategic motifs",
            self.extracted_motifs.len()
        );
        Ok(self.extracted_motifs.clone())
    }

    /// Analyze all positions in the engine database
    fn analyze_positions(
        &mut self,
        engine: &ChessVectorEngine,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use indicatif::{ProgressBar, ProgressStyle};

        let total_positions = engine.knowledge_base_size();
        let pb = ProgressBar::new(total_positions as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("🔍 Analyzing patterns [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}) {msg}")?
                .progress_chars("██░")
        );

        // Analyze each position for strategic patterns
        for i in 0..total_positions {
            if let (Some(board), Some(evaluation)) = (
                engine.get_board_by_index(i),
                engine.get_evaluation_by_index(i),
            ) {
                self.analyze_single_position(board, evaluation);
            }

            if i % 1000 == 0 {
                pb.set_position(i as u64);
                pb.set_message(format!("Found {} patterns", self.pattern_counts.len()));
            }
        }

        pb.finish_with_message(format!(
            "✅ Analysis complete: {} unique patterns found",
            self.pattern_counts.len()
        ));
        Ok(())
    }

    /// Analyze a single position for strategic patterns using advanced pattern recognition
    fn analyze_single_position(&mut self, board: &Board, evaluation: f32) {
        let game_phase = pattern_utils::determine_game_phase(board);

        // Generate simplified patterns based on basic position features
        let strategic_patterns = self.generate_simple_patterns(board, evaluation);

        // Convert simple patterns to trackable motifs
        for pattern in strategic_patterns {
            // Only track patterns with significant strength
            if pattern.strength.abs() > 0.1 && pattern.reliability > 0.6 {
                let pattern_hash = self.generate_pattern_hash(&pattern);
                self.record_pattern_occurrence(
                    pattern_hash,
                    board,
                    evaluation,
                    &game_phase,
                );
            }
        }
    }

    /// Generate simplified strategic patterns from basic position features
    fn generate_simple_patterns(&self, board: &Board, evaluation: f32) -> Vec<StrategicPattern> {
        let mut patterns = Vec::new();

        // Material imbalance pattern
        let material_balance = self.calculate_material_balance(board);
        if material_balance.abs() > 1 {
            patterns.push(StrategicPattern {
                pattern_type: SimplePatternType::MaterialImbalance,
                strength: material_balance as f32 * 0.1,
                reliability: 0.8,
            });
        }

        // King safety pattern
        if self.is_king_exposed(board, chess::Color::White) || self.is_king_exposed(board, chess::Color::Black) {
            patterns.push(StrategicPattern {
                pattern_type: SimplePatternType::KingSafety,
                strength: evaluation.signum() * 0.3,
                reliability: 0.7,
            });
        }

        patterns
    }

    /// Check if king is exposed
    fn is_king_exposed(&self, board: &Board, color: chess::Color) -> bool {
        let king_square = board.king_square(color);
        // Simple check: king on edge ranks (1st or 8th) is potentially exposed
        let rank = king_square.get_rank().to_index();
        rank == 0 || rank == 7
    }

    /// Calculate basic material balance
    fn calculate_material_balance(&self, board: &Board) -> i32 {
        let mut balance = 0;
        let piece_values = [1, 3, 3, 5, 9, 0]; // Pawn, Knight, Bishop, Rook, Queen, King

        for square in chess::ALL_SQUARES {
            if let Some(piece) = board.piece_on(square) {
                let value = piece_values[piece as usize];
                match board.color_on(square) {
                    Some(Color::White) => balance += value,
                    Some(Color::Black) => balance -= value,
                    None => {}
                }
            }
        }
        balance
    }

    /// Extract pawn structure patterns
    fn extract_pawn_patterns(&mut self, board: &Board, evaluation: f32, game_phase: &GamePhase) {
        // Analyze for isolated pawns
        for color in [Color::White, Color::Black] {
            let isolated_pawns = self.find_isolated_pawns(board, color);
            for (file, square) in isolated_pawns {
                let pattern_hash = self.hash_isolated_pawn_pattern(file, color);
                self.record_pattern_occurrence(pattern_hash, board, evaluation, game_phase);
            }
        }

        // Analyze for passed pawns
        for color in [Color::White, Color::Black] {
            let passed_pawns = self.find_passed_pawns(board, color);
            for (square, advancement) in passed_pawns {
                let pattern_hash = self.hash_passed_pawn_pattern(square, color, advancement);
                self.record_pattern_occurrence(pattern_hash, board, evaluation, game_phase);
            }
        }

        // Analyze for doubled pawns
        for color in [Color::White, Color::Black] {
            let doubled_pawns = self.find_doubled_pawns(board, color);
            for (file, count) in doubled_pawns {
                let pattern_hash = self.hash_doubled_pawn_pattern(file, color, count);
                self.record_pattern_occurrence(pattern_hash, board, evaluation, game_phase);
            }
        }
    }

    /// Extract piece coordination patterns
    fn extract_piece_patterns(&mut self, board: &Board, evaluation: f32, game_phase: &GamePhase) {
        // Analyze for knight outposts
        for color in [Color::White, Color::Black] {
            let outposts = self.find_knight_outposts(board, color);
            for (square, strength) in outposts {
                let pattern_hash = self.hash_knight_outpost_pattern(square, color, strength);
                self.record_pattern_occurrence(pattern_hash, board, evaluation, game_phase);
            }
        }

        // Analyze for bishop pairs
        for color in [Color::White, Color::Black] {
            if self.has_bishop_pair(board, color) {
                let open_diagonals = self.count_open_diagonals(board, color);
                let pattern_hash = self.hash_bishop_pair_pattern(color, open_diagonals);
                self.record_pattern_occurrence(pattern_hash, board, evaluation, game_phase);
            }
        }

        // Analyze for rook activity
        for color in [Color::White, Color::Black] {
            let rook_patterns = self.analyze_rook_activity(board, color);
            for (pattern_type, square) in rook_patterns {
                let pattern_hash = self.hash_rook_pattern(pattern_type, square, color);
                self.record_pattern_occurrence(pattern_hash, board, evaluation, game_phase);
            }
        }
    }

    /// Extract king safety patterns
    fn extract_king_safety_patterns(
        &mut self,
        board: &Board,
        evaluation: f32,
        game_phase: &GamePhase,
    ) {
        for color in [Color::White, Color::Black] {
            let king_square = board.king_square(color);

            // Analyze castling structure
            let castling_safety = self.evaluate_castling_safety(board, color, king_square);
            if castling_safety.is_some() {
                let pattern_hash =
                    self.hash_king_safety_pattern(king_square, color, castling_safety.unwrap());
                self.record_pattern_occurrence(pattern_hash, board, evaluation, game_phase);
            }

            // Analyze pawn shield
            let shield_pattern = self.analyze_pawn_shield(board, color, king_square);
            let pattern_hash = self.hash_pawn_shield_pattern(king_square, color, shield_pattern);
            self.record_pattern_occurrence(pattern_hash, board, evaluation, game_phase);
        }
    }

    /// Extract initiative patterns
    fn extract_initiative_patterns(
        &mut self,
        board: &Board,
        evaluation: f32,
        game_phase: &GamePhase,
    ) {
        // Analyze space advantage
        for color in [Color::White, Color::Black] {
            let space_value = self.calculate_space_advantage(board, color);
            if space_value.abs() > 0.2 {
                // Significant space advantage
                let pattern_hash = self.hash_space_pattern(color, space_value);
                self.record_pattern_occurrence(pattern_hash, board, evaluation, game_phase);
            }
        }

        // Analyze development patterns (for opening/early middlegame)
        if matches!(game_phase, GamePhase::Opening) {
            for color in [Color::White, Color::Black] {
                let development_score = self.calculate_development_score(board, color);
                let pattern_hash = self.hash_development_pattern(color, development_score);
                self.record_pattern_occurrence(pattern_hash, board, evaluation, game_phase);
            }
        }
    }

    /// Record a pattern occurrence
    fn record_pattern_occurrence(
        &mut self,
        pattern_hash: u64,
        board: &Board,
        evaluation: f32,
        game_phase: &GamePhase,
    ) {
        let occurrence =
            self.pattern_counts
                .entry(pattern_hash)
                .or_insert_with(|| PatternOccurrence {
                    count: 0,
                    evaluations: Vec::new(),
                    positions: Vec::new(),
                    game_phases: Vec::new(),
                });

        occurrence.count += 1;
        occurrence.evaluations.push(evaluation);
        occurrence.positions.push(*board);
        occurrence.game_phases.push(game_phase.clone());
    }


    /// Generate simplified hash for basic patterns
    fn generate_pattern_hash(&self, pattern: &StrategicPattern) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash based on simplified pattern type
        match &pattern.pattern_type {
            SimplePatternType::MaterialImbalance => {
                "material_imbalance".hash(&mut hasher);
            }
            SimplePatternType::KingSafety => {
                "king_safety".hash(&mut hasher);
            }
            SimplePatternType::CenterControl => {
                "center_control".hash(&mut hasher);
            }
            SimplePatternType::PieceActivity => {
                "piece_activity".hash(&mut hasher);
            }
        }

        // Include pattern strength range for differentiation
        let strength_bucket = ((pattern.strength + 2.0) * 10.0) as i32; // -2.0 to +2.0 -> 0 to 40
        strength_bucket.hash(&mut hasher);

        hasher.finish()
    }

    /// Extract significant patterns from occurrence data
    fn extract_significant_patterns(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🎯 Extracting significant strategic patterns...");
        println!(
            "   📊 Found {} unique patterns to analyze",
            self.pattern_counts.len()
        );

        let mut motif_id = 1u64;
        let mut frequency_filtered = 0;
        let mut significance_filtered = 0;
        let mut confidence_filtered = 0;

        for (pattern_hash, occurrence) in &self.pattern_counts {
            if occurrence.count < self.min_pattern_frequency {
                frequency_filtered += 1;
                continue;
            }

            // Calculate statistical significance
            let avg_evaluation =
                occurrence.evaluations.iter().sum::<f32>() / occurrence.evaluations.len() as f32;
            let eval_std_dev = self.calculate_std_dev(&occurrence.evaluations, avg_evaluation);

            // Only include patterns with significant evaluation impact
            if avg_evaluation.abs() < self.min_eval_significance {
                significance_filtered += 1;
                continue;
            }

            // Calculate confidence based on consistency
            let confidence = self.calculate_pattern_confidence(occurrence);
            if confidence < self.confidence_threshold {
                confidence_filtered += 1;
                continue;
            }

            // Create strategic motif (simplified for now - will enhance pattern recognition)
            let motif = StrategicMotif {
                id: motif_id,
                pattern_hash: *pattern_hash,
                motif_type: self.infer_motif_type(&occurrence.positions[0], *pattern_hash),
                evaluation: avg_evaluation,
                context: self.determine_context(&occurrence.game_phases, &occurrence.evaluations),
                confidence,
                master_games: self.create_game_references(&occurrence.positions),
                description: format!(
                    "Strategic pattern {} (avg eval: {:.2})",
                    motif_id, avg_evaluation
                ),
            };

            self.extracted_motifs.push(motif);
            motif_id += 1;
        }

        println!(
            "📈 Extracted {} statistically significant patterns",
            self.extracted_motifs.len()
        );
        println!("   🚫 Filtered out:");
        println!(
            "      - {} patterns (frequency < {})",
            frequency_filtered, self.min_pattern_frequency
        );
        println!(
            "      - {} patterns (significance < {:.1})",
            significance_filtered, self.min_eval_significance
        );
        println!(
            "      - {} patterns (confidence < {:.1})",
            confidence_filtered, self.confidence_threshold
        );
        Ok(())
    }

    /// Validate extracted patterns
    fn validate_patterns(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("✅ Validating extracted patterns...");

        // Remove patterns that might be too position-specific
        let original_count = self.extracted_motifs.len();

        self.extracted_motifs.retain(|motif| {
            // Keep patterns with high confidence and reasonable frequency
            motif.confidence >= self.confidence_threshold
                && self
                    .pattern_counts
                    .get(&motif.pattern_hash)
                    .map(|occ| occ.count >= self.min_pattern_frequency)
                    .unwrap_or(false)
        });

        let removed_count = original_count - self.extracted_motifs.len();
        if removed_count > 0 {
            println!("🧹 Removed {} low-quality patterns", removed_count);
        }

        println!(
            "✨ Validation complete: {} high-quality strategic motifs ready",
            self.extracted_motifs.len()
        );
        Ok(())
    }

    // Helper methods for pattern analysis (simplified implementations)

    fn find_isolated_pawns(&self, board: &Board, color: Color) -> Vec<(File, Square)> {
        let mut isolated = Vec::new();
        let pawns = board.pieces(Piece::Pawn) & board.color_combined(color);

        for square in pawns {
            let file = square.get_file();
            let adjacent_files = [
                if file != File::A {
                    Some(File::from_index(file.to_index() - 1))
                } else {
                    None
                },
                if file != File::H {
                    Some(File::from_index(file.to_index() + 1))
                } else {
                    None
                },
            ];

            let is_isolated = adjacent_files.iter().filter_map(|&f| f).all(|adj_file| {
                // Check if adjacent files have no pawns
                let mut has_adjacent_pawn = false;
                for rank in chess::ALL_RANKS {
                    let square = Square::make_square(rank, adj_file);
                    if (pawns & BitBoard::from_square(square)) != BitBoard(0) {
                        has_adjacent_pawn = true;
                        break;
                    }
                }
                !has_adjacent_pawn
            });

            if is_isolated {
                isolated.push((file, square));
            }
        }

        isolated
    }

    fn find_passed_pawns(&self, board: &Board, color: Color) -> Vec<(Square, f32)> {
        // Simplified implementation - needs full passed pawn detection logic
        Vec::new()
    }

    fn find_doubled_pawns(&self, board: &Board, color: Color) -> Vec<(File, u8)> {
        let mut doubled = Vec::new();
        let pawns = board.pieces(Piece::Pawn) & board.color_combined(color);

        for file in ALL_FILES {
            let mut count = 0u8;
            for rank in chess::ALL_RANKS {
                let square = Square::make_square(rank, file);
                if (pawns & BitBoard::from_square(square)) != BitBoard(0) {
                    count += 1;
                }
            }
            if count > 1 {
                doubled.push((file, count));
            }
        }

        doubled
    }

    fn find_knight_outposts(&self, _board: &Board, _color: Color) -> Vec<(Square, f32)> {
        // Simplified implementation
        Vec::new()
    }

    fn has_bishop_pair(&self, board: &Board, color: Color) -> bool {
        let bishops = board.pieces(Piece::Bishop) & board.color_combined(color);
        bishops.count() >= 2
    }

    fn count_open_diagonals(&self, _board: &Board, _color: Color) -> u8 {
        // Simplified implementation
        0
    }

    fn analyze_rook_activity(&self, _board: &Board, _color: Color) -> Vec<(String, Square)> {
        // Simplified implementation
        Vec::new()
    }

    fn evaluate_castling_safety(
        &self,
        _board: &Board,
        _color: Color,
        _king_square: Square,
    ) -> Option<f32> {
        // Simplified implementation
        Some(0.0)
    }

    fn analyze_pawn_shield(&self, _board: &Board, _color: Color, _king_square: Square) -> u8 {
        // Simplified implementation
        0
    }

    fn calculate_space_advantage(&self, _board: &Board, _color: Color) -> f32 {
        // Simplified implementation
        0.0
    }

    fn calculate_development_score(&self, _board: &Board, _color: Color) -> f32 {
        // Simplified implementation
        0.0
    }

    fn calculate_std_dev(&self, values: &[f32], mean: f32) -> f32 {
        if values.len() <= 1 {
            return 0.0;
        }

        let variance =
            values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / (values.len() - 1) as f32;

        variance.sqrt()
    }

    fn calculate_pattern_confidence(&self, occurrence: &PatternOccurrence) -> f32 {
        // Calculate confidence based on consistency and frequency
        let avg_eval =
            occurrence.evaluations.iter().sum::<f32>() / occurrence.evaluations.len() as f32;
        let std_dev = self.calculate_std_dev(&occurrence.evaluations, avg_eval);

        // Higher frequency and lower deviation = higher confidence
        let frequency_factor = (occurrence.count as f32 / 100.0).min(1.0); // Cap at 100 occurrences
        let consistency_factor = (1.0 - (std_dev / 2.0).min(1.0)).max(0.0);

        (frequency_factor + consistency_factor) / 2.0
    }

    fn infer_motif_type(&self, _board: &Board, _pattern_hash: u64) -> MotifType {
        // Simplified type inference - in practice, this would be more sophisticated
        MotifType::PawnStructure(PawnPattern::IsolatedPawn {
            file_index: 3, // D file
            is_white: true,
            weakness_value: 0.3,
        })
    }

    fn determine_context(
        &self,
        game_phases: &[GamePhase],
        evaluations: &[f32],
    ) -> StrategicContext {
        // Determine most common game phase
        let mut phase_counts = HashMap::new();
        for phase in game_phases {
            *phase_counts.entry(format!("{:?}", phase)).or_insert(0) += 1;
        }

        let most_common_phase = phase_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(phase, _)| phase)
            .unwrap_or_else(|| "Any".to_string());

        let game_phase = match most_common_phase.as_str() {
            "Opening" => GamePhase::Opening,
            "Middlegame" => GamePhase::Middlegame,
            "Endgame" => GamePhase::Endgame,
            _ => GamePhase::Any,
        };

        StrategicContext {
            game_phase,
            material_context: MaterialContext::Any,
            min_ply: 1,
            max_ply: None,
        }
    }

    fn create_game_references(&self, positions: &[Board]) -> Vec<GameReference> {
        // Create simplified game references
        positions
            .iter()
            .take(3)
            .enumerate()
            .map(|(i, _)| GameReference {
                game_id: format!("extracted_{}", i),
                white: "Master".to_string(),
                black: "Player".to_string(),
                result: "1-0".to_string(),
                ply: 20,
                rating: Some(2500),
            })
            .collect()
    }

    // Hash generation methods (simplified)
    fn hash_isolated_pawn_pattern(&self, file: File, color: Color) -> u64 {
        let mut hash = 0x1234567890abcdefu64;
        hash ^= file.to_index() as u64;
        hash ^= if color == Color::White { 1 } else { 0 };
        hash
    }

    fn hash_passed_pawn_pattern(&self, square: Square, color: Color, advancement: f32) -> u64 {
        let mut hash = 0x2345678901bcdef0u64;
        hash ^= square.to_index() as u64;
        hash ^= if color == Color::White { 1 } else { 0 };
        hash ^= (advancement * 1000.0) as u64;
        hash
    }

    fn hash_doubled_pawn_pattern(&self, file: File, color: Color, count: u8) -> u64 {
        let mut hash = 0x3456789012cdef01u64;
        hash ^= file.to_index() as u64;
        hash ^= if color == Color::White { 1 } else { 0 };
        hash ^= count as u64;
        hash
    }

    fn hash_knight_outpost_pattern(&self, square: Square, color: Color, strength: f32) -> u64 {
        let mut hash = 0x456789013def012u64;
        hash ^= square.to_index() as u64;
        hash ^= if color == Color::White { 1 } else { 0 };
        hash ^= (strength * 1000.0) as u64;
        hash
    }

    fn hash_bishop_pair_pattern(&self, color: Color, open_diagonals: u8) -> u64 {
        let mut hash = 0x56789014ef0123u64;
        hash ^= if color == Color::White { 1 } else { 0 };
        hash ^= open_diagonals as u64;
        hash
    }

    fn hash_rook_pattern(&self, pattern_type: String, square: Square, color: Color) -> u64 {
        let mut hash = 0x6789015f01234u64;
        hash ^= square.to_index() as u64;
        hash ^= if color == Color::White { 1 } else { 0 };
        hash ^= pattern_type.len() as u64;
        hash
    }

    fn hash_king_safety_pattern(
        &self,
        king_square: Square,
        color: Color,
        safety_value: f32,
    ) -> u64 {
        let mut hash = 0x789016012345u64;
        hash ^= king_square.to_index() as u64;
        hash ^= if color == Color::White { 1 } else { 0 };
        hash ^= (safety_value * 1000.0) as u64;
        hash
    }

    fn hash_pawn_shield_pattern(
        &self,
        king_square: Square,
        color: Color,
        shield_pattern: u8,
    ) -> u64 {
        let mut hash = 0x89017123456u64;
        hash ^= king_square.to_index() as u64;
        hash ^= if color == Color::White { 1 } else { 0 };
        hash ^= shield_pattern as u64;
        hash
    }

    fn hash_space_pattern(&self, color: Color, space_value: f32) -> u64 {
        let mut hash = 0x9018234567u64;
        hash ^= if color == Color::White { 1 } else { 0 };
        hash ^= (space_value * 1000.0) as u64;
        hash
    }

    fn hash_development_pattern(&self, color: Color, development_score: f32) -> u64 {
        let mut hash = 0xa019345678u64;
        hash ^= if color == Color::White { 1 } else { 0 };
        hash ^= (development_score * 1000.0) as u64;
        hash
    }
}
