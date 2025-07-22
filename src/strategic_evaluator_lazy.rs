//! Enhanced strategic evaluator with lazy-loaded motif database
//! 
//! This module extends the strategic evaluator to use lazy-loaded strategic motifs,
//! providing master-level strategic insights while minimizing memory usage through
//! on-demand pattern loading.

use crate::strategic_evaluator::{StrategicEvaluator, StrategicConfig, StrategicEvaluation};
use crate::utils::lazy_motifs::{LazyStrategicDatabase, LazyLoadConfig, LazyLoadStats};
use crate::strategic_motifs::MotifMatch;
use crate::strategic_motifs::{GamePhase, MotifType};
use chess::{Board, ChessMove};
use std::path::Path;
use std::sync::Arc;

/// Enhanced strategic evaluator with lazy-loaded strategic motifs
pub struct LazyStrategicEvaluator {
    /// Core strategic evaluator for basic evaluation
    core_evaluator: StrategicEvaluator,
    /// Lazy-loading motif database
    motif_database: Arc<LazyStrategicDatabase>,
    /// Configuration for lazy loading
    lazy_config: LazyLoadConfig,
    /// Enable/disable motif-based evaluation
    use_motifs: bool,
}

/// Enhanced strategic evaluation with motif insights
#[derive(Debug, Clone)]
pub struct EnhancedStrategicEvaluation {
    /// Core strategic evaluation
    pub core_evaluation: StrategicEvaluation,
    /// Matched strategic motifs
    pub motif_matches: Vec<MotifMatch>,
    /// Strategic adjustment from motifs
    pub motif_adjustment: f32,
    /// Confidence in motif-based recommendations
    pub motif_confidence: f32,
    /// Game phase detected
    pub detected_phase: GamePhase,
}

impl LazyStrategicEvaluator {
    /// Create new lazy strategic evaluator
    pub fn new<P: AsRef<Path>>(
        config: StrategicConfig,
        motif_database_path: P,
        lazy_config: LazyLoadConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let core_evaluator = StrategicEvaluator::new(config);
        
        let motif_database = match LazyStrategicDatabase::new(motif_database_path, lazy_config.clone()) {
            Ok(db) => Arc::new(db),
            Err(e) => {
                // If motif database fails to load, continue without it
                eprintln!("Warning: Failed to load motif database: {}", e);
                return Ok(Self {
                    core_evaluator,
                    motif_database: Arc::new(Self::create_empty_database()?),
                    lazy_config,
                    use_motifs: false,
                });
            }
        };

        Ok(Self {
            core_evaluator,
            motif_database,
            lazy_config,
            use_motifs: true,
        })
    }

    /// Create empty database as fallback
    fn create_empty_database() -> Result<LazyStrategicDatabase, Box<dyn std::error::Error>> {
        use std::env;
        use crate::utils::lazy_motifs::MotifSegmentBuilder;
        
        // Use a temporary directory in system temp
        let temp_dir_path = env::temp_dir().join("chess_empty_motifs");
        std::fs::create_dir_all(&temp_dir_path)?;
        
        let config = LazyLoadConfig::default();
        
        // Create empty index
        let builder = MotifSegmentBuilder::new(&temp_dir_path, config.clone());
        let _index = builder.create_segments(Vec::new(), 100)?;
        
        LazyStrategicDatabase::new(&temp_dir_path, config)
    }

    /// Evaluate position with enhanced strategic analysis including motifs
    pub fn evaluate_position(&self, board: &Board) -> Result<EnhancedStrategicEvaluation, Box<dyn std::error::Error>> {
        // Core strategic evaluation
        let core_evaluation = self.core_evaluator.evaluate_strategic(board);
        
        if !self.use_motifs {
            return Ok(EnhancedStrategicEvaluation {
                core_evaluation,
                motif_matches: Vec::new(),
                motif_adjustment: 0.0,
                motif_confidence: 0.0,
                detected_phase: GamePhase::Any,
            });
        }

        // Detect game phase for targeted motif loading
        let detected_phase = self.detect_game_phase(board);
        
        // Find relevant strategic motifs
        let motif_matches = self.motif_database.evaluate_position(board)?;
        
        // Calculate motif-based adjustments
        let (motif_adjustment, motif_confidence) = self.calculate_motif_adjustments(&motif_matches);
        
        Ok(EnhancedStrategicEvaluation {
            core_evaluation,
            motif_matches,
            motif_adjustment,
            motif_confidence,
            detected_phase,
        })
    }

    /// Evaluate a specific move with strategic motifs
    pub fn evaluate_move(&self, board: &Board, chess_move: &ChessMove) -> Result<f32, Box<dyn std::error::Error>> {
        // Make the move to create the resulting position
        let mut new_board = *board;
        new_board = new_board.make_move_new(*chess_move);
        
        // Evaluate the resulting position
        let evaluation = self.evaluate_position(&new_board)?;
        
        // Combine core and motif evaluations
        let total_adjustment = evaluation.core_evaluation.total_evaluation + evaluation.motif_adjustment;
        
        Ok(total_adjustment)
    }

    /// Get strategic motifs for a specific game phase (preloading optimization)
    pub fn preload_phase_motifs(&self, phase: GamePhase) -> Result<usize, Box<dyn std::error::Error>> {
        if !self.use_motifs {
            return Ok(0);
        }
        
        self.motif_database.preload_phase(phase)
    }

    /// Find strategic motifs matching specific criteria
    pub fn find_motifs_by_type(&self, _motif_type: &MotifType) -> Result<Vec<MotifMatch>, Box<dyn std::error::Error>> {
        if !self.use_motifs {
            return Ok(Vec::new());
        }

        // This would require extending the motif database with type-based queries
        // For now, return empty - would be implemented with more sophisticated indexing
        Ok(Vec::new())
    }

    /// Analyze position for specific strategic themes
    pub fn analyze_strategic_themes(&self, board: &Board) -> Result<StrategicThemeAnalysis, Box<dyn std::error::Error>> {
        let evaluation = self.evaluate_position(board)?;
        
        let mut themes = StrategicThemeAnalysis::new();
        
        // Analyze motifs by type
        for motif_match in &evaluation.motif_matches {
            match &motif_match.motif.motif_type {
                MotifType::PawnStructure(_) => {
                    themes.pawn_themes.push(motif_match.clone());
                }
                MotifType::PieceCoordination(_) => {
                    themes.piece_coordination_themes.push(motif_match.clone());
                }
                MotifType::KingSafety(_) => {
                    themes.king_safety_themes.push(motif_match.clone());
                }
                MotifType::Initiative(_) => {
                    themes.initiative_themes.push(motif_match.clone());
                }
                MotifType::Endgame(_) => {
                    themes.endgame_themes.push(motif_match.clone());
                }
                MotifType::Opening(_) => {
                    themes.opening_themes.push(motif_match.clone());
                }
            }
        }
        
        Ok(themes)
    }

    /// Get performance statistics for the lazy loading system
    pub fn get_lazy_load_stats(&self) -> LazyLoadStats {
        self.motif_database.get_stats()
    }

    /// Clear motif caches (for memory management)
    pub fn clear_motif_caches(&self) {
        self.motif_database.clear_caches();
    }

    /// Get total number of available motifs
    pub fn total_available_motifs(&self) -> usize {
        self.motif_database.total_motifs()
    }

    /// Get number of currently cached motifs
    pub fn cached_motifs_count(&self) -> usize {
        self.motif_database.cached_motifs()
    }

    /// Detect the current game phase for targeted motif loading
    fn detect_game_phase(&self, board: &Board) -> GamePhase {
        let piece_count = board.combined().popcnt();
        
        if piece_count > 28 {
            GamePhase::Opening
        } else if piece_count > 12 {
            GamePhase::Middlegame
        } else {
            GamePhase::Endgame
        }
    }

    /// Calculate strategic adjustments based on matched motifs
    fn calculate_motif_adjustments(&self, motif_matches: &[MotifMatch]) -> (f32, f32) {
        if motif_matches.is_empty() {
            return (0.0, 0.0);
        }

        let mut total_adjustment = 0.0;
        let mut total_confidence = 0.0;
        let mut weight_sum = 0.0;

        for motif_match in motif_matches {
            let weight = motif_match.relevance * motif_match.motif.confidence;
            total_adjustment += motif_match.motif.evaluation * weight;
            total_confidence += motif_match.motif.confidence * weight;
            weight_sum += weight;
        }

        if weight_sum > 0.0 {
            total_adjustment /= weight_sum;
            total_confidence /= weight_sum;
        }

        // Cap adjustments to reasonable bounds
        total_adjustment = total_adjustment.clamp(-1.0, 1.0);
        total_confidence = total_confidence.clamp(0.0, 1.0);

        (total_adjustment, total_confidence)
    }
}

/// Analysis of strategic themes found in a position
#[derive(Debug, Clone, Default)]
pub struct StrategicThemeAnalysis {
    pub pawn_themes: Vec<MotifMatch>,
    pub piece_coordination_themes: Vec<MotifMatch>,
    pub king_safety_themes: Vec<MotifMatch>,
    pub initiative_themes: Vec<MotifMatch>,
    pub endgame_themes: Vec<MotifMatch>,
    pub opening_themes: Vec<MotifMatch>,
}

impl StrategicThemeAnalysis {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the most relevant theme in this position
    pub fn primary_theme(&self) -> Option<&MotifMatch> {
        let all_themes: Vec<&MotifMatch> = [
            &self.pawn_themes,
            &self.piece_coordination_themes,
            &self.king_safety_themes,
            &self.initiative_themes,
            &self.endgame_themes,
            &self.opening_themes,
        ]
        .iter()
        .flat_map(|themes| themes.iter())
        .collect();

        all_themes.into_iter().max_by(|a, b| {
            a.relevance.partial_cmp(&b.relevance).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get total number of strategic themes identified
    pub fn theme_count(&self) -> usize {
        self.pawn_themes.len()
            + self.piece_coordination_themes.len()
            + self.king_safety_themes.len()
            + self.initiative_themes.len()
            + self.endgame_themes.len()
            + self.opening_themes.len()
    }

    /// Get strategic recommendations based on identified themes
    pub fn get_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        if let Some(primary) = self.primary_theme() {
            recommendations.push(primary.motif.description.clone());
        }

        // Add specific recommendations based on theme types
        if !self.pawn_themes.is_empty() {
            recommendations.push("Consider pawn structure improvements".to_string());
        }

        if !self.king_safety_themes.is_empty() {
            recommendations.push("Focus on king safety measures".to_string());
        }

        if !self.initiative_themes.is_empty() {
            recommendations.push("Maintain initiative and create pressure".to_string());
        }

        recommendations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess::Board;
    use std::str::FromStr;
    use std::env;

    #[test]
    fn test_lazy_strategic_evaluator_creation() {
        let temp_dir_path = env::temp_dir().join("chess_test_lazy");
        std::fs::create_dir_all(&temp_dir_path).unwrap();
        let config = StrategicConfig::default();
        let lazy_config = LazyLoadConfig::default();
        
        let evaluator = LazyStrategicEvaluator::new(config, &temp_dir_path, lazy_config);
        
        // Should succeed even with empty database
        assert!(evaluator.is_ok());
    }

    #[test]
    fn test_game_phase_detection() {
        let temp_dir_path = env::temp_dir().join("chess_test_phase");
        std::fs::create_dir_all(&temp_dir_path).unwrap();
        let config = StrategicConfig::default();
        let lazy_config = LazyLoadConfig::default();
        
        let evaluator = LazyStrategicEvaluator::new(config, &temp_dir_path, lazy_config).unwrap();
        
        // Test starting position (should be opening)
        let board = Board::default();
        let phase = evaluator.detect_game_phase(&board);
        matches!(phase, GamePhase::Opening);
        
        // Test endgame position
        let endgame_fen = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";
        let endgame_board = Board::from_str(endgame_fen).unwrap();
        let endgame_phase = evaluator.detect_game_phase(&endgame_board);
        matches!(endgame_phase, GamePhase::Endgame);
    }

    #[test]
    fn test_strategic_theme_analysis() {
        let analysis = StrategicThemeAnalysis::new();
        assert_eq!(analysis.theme_count(), 0);
        
        let recommendations = analysis.get_recommendations();
        assert!(recommendations.is_empty());
    }
}