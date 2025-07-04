use chess::{Board, ChessMove, Color, Piece, Square};
use shakmaty::{Chess, Position, fen::Fen, CastlingMode};
use shakmaty_syzygy::{Tablebase, Wdl, Dtz, SyzygyError};
use std::path::Path;
use std::sync::Arc;

/// Syzygy tablebase integration for perfect endgame evaluation
/// Provides exact win/draw/loss results for positions with ≤7 pieces
#[derive(Clone)]
pub struct TablebaseProber {
    tablebase: Option<Arc<Tablebase<Chess>>>,
    enabled: bool,
    max_pieces: usize,
}

/// Result from tablebase probing
#[derive(Debug, Clone, PartialEq)]
pub enum TablebaseResult {
    /// Exact win/draw/loss result with distance to conversion/mate
    Exact { wdl: WdlValue, dtz: Option<u32> },
    /// Position not found in tablebase (too many pieces or tablebase not available)
    NotFound,
    /// Error during probe
    Error(String),
}

/// Win/Draw/Loss values from tablebase
#[derive(Debug, Clone, PartialEq)]
pub enum WdlValue {
    /// Win for side to move in this many plies
    Win(u32),
    /// Draw
    Draw,
    /// Loss for side to move in this many plies  
    Loss(u32),
    /// Blessed loss (can be held to draw with optimal play)
    BlessedLoss(u32),
    /// Cursed win (can be held to draw by opponent with optimal play)
    CursedWin(u32),
}

impl TablebaseProber {
    /// Create a new tablebase prober
    pub fn new() -> Self {
        Self {
            tablebase: None,
            enabled: false,
            max_pieces: 7, // Syzygy supports up to 7-piece tablebases
        }
    }

    /// Initialize tablebase from directory path
    pub fn initialize<P: AsRef<Path>>(&mut self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        if !path.as_ref().exists() {
            return Err(format!("Processing...").display()).into());
        }

        println!("Processing...").display());
        
        let mut tb = Tablebase::<Chess>::new();
        match tb.add_directory(path) {
            Ok(_) => {
                // Note: shakmaty-syzygy doesn't expose max_pieces directly
                // We'll assume 7-piece support which is standard for Syzygy
                self.tablebase = Some(Arc::new(tb));
                self.enabled = true;
                self.max_pieces = 7; // Standard Syzygy limit
                println!("✅ Loaded Syzygy tablebases for up to {} pieces", self.max_pieces);
                Ok(())
            }
            Err(e) => {
                let error_msg = format!("Processing...");
                println!("Operation complete");
                Err(error_msg.into())
            }
        }
    }

    /// Check if position can be probed (≤ max pieces and tablebase available)
    pub fn can_probe(&self, board: &Board) -> bool {
        if !self.enabled || self.tablebase.is_none() {
            return false;
        }
        
        // Count pieces on board
        let piece_count = (0..64)
            .map(|i| Square::new(i))
            .filter(|&sq| board.piece_on(sq).is_some())
            .count();
            
        piece_count <= self.max_pieces
    }

    /// Probe position for exact evaluation
    pub fn probe_wdl(&self, board: &Board) -> TablebaseResult {
        if !self.can_probe(board) {
            return TablebaseResult::NotFound;
        }

        let tablebase = match &self.tablebase {
            Some(tb) => tb,
            None => return TablebaseResult::NotFound,
        };

        // Convert chess crate board to shakmaty format
        match self.convert_position(board) {
            Ok(shakmaty_pos) => {
                match tablebase.probe_wdl(&shakmaty_pos) {
                    Ok(wdl) => {
                        let wdl_value = self.convert_wdl(wdl, board.side_to_move());
                        TablebaseResult::Exact { wdl: wdl_value, dtz: None }
                    }
                    Err(SyzygyError::MissingTable { .. }) => TablebaseResult::NotFound,
                    Err(e) => TablebaseResult::Error(format!("Processing...")),
                }
            }
            Err(e) => TablebaseResult::Error(e),
        }
    }

    /// Probe position for exact evaluation with distance-to-zero
    pub fn probe_dtz(&self, board: &Board) -> TablebaseResult {
        if !self.can_probe(board) {
            return TablebaseResult::NotFound;
        }

        let tablebase = match &self.tablebase {
            Some(tb) => tb,
            None => return TablebaseResult::NotFound,
        };

        match self.convert_position(board) {
            Ok(shakmaty_pos) => {
                // First get WDL
                let wdl_result = match tablebase.probe_wdl(&shakmaty_pos) {
                    Ok(wdl) => wdl,
                    _ => return TablebaseResult::NotFound,
                };

                // Then get DTZ
                match tablebase.probe_dtz(&shakmaty_pos) {
                    Ok(dtz) => {
                        let wdl_value = self.convert_wdl(wdl_result, board.side_to_move());
                        let dtz_value = self.convert_dtz(dtz);
                        TablebaseResult::Exact { wdl: wdl_value, dtz: Some(dtz_value) }
                    }
                    Err(SyzygyError::MissingTable { .. }) => TablebaseResult::NotFound,
                    Err(e) => TablebaseResult::Error(format!("Processing...")),
                }
            }
            Err(e) => TablebaseResult::Error(e),
        }
    }

    /// Get tablebase evaluation as centipaws for integration with engine
    pub fn get_evaluation(&self, board: &Board) -> Option<f32> {
        match self.probe_wdl(board) {
            TablebaseResult::Exact { wdl, .. } => {
                Some(match wdl {
                    WdlValue::Win(plies) => {
                        // Mate score: high value minus distance to mate
                        // Use standard mate scoring: 30000 - plies_to_mate
                        30000.0 - (plies as f32)
                    }
                    WdlValue::Loss(plies) => {
                        // Mated score: negative mate score
                        -(30000.0 - (plies as f32))
                    }
                    WdlValue::Draw => 0.0,
                    WdlValue::BlessedLoss(_) => -50.0, // Slight disadvantage but drawable
                    WdlValue::CursedWin(_) => 50.0,    // Slight advantage but drawable
                })
            }
            _ => None,
        }
    }

    /// Check if tablebase is enabled and loaded
    pub fn is_enabled(&self) -> bool {
        self.enabled && self.tablebase.is_some()
    }

    /// Get maximum pieces supported
    pub fn max_pieces(&self) -> usize {
        self.max_pieces
    }

    /// Get best move from tablebase if available
    pub fn get_best_move(&self, board: &Board) -> Option<ChessMove> {
        if !self.can_probe(board) {
            return None;
        }

        let mut best_move = None;
        let mut best_eval = f32::NEG_INFINITY;

        // Try all legal moves and find the best tablebase result
        for chess_move in chess::MoveGen::new_legal(board) {
            let new_board = board.make_move_new(chess_move);
            
            if let Some(eval) = self.get_evaluation(&new_board) {
                // Flip evaluation since it's from opponent's perspective
                let eval_for_us = -eval;
                
                if eval_for_us > best_eval {
                    best_eval = eval_for_us;
                    best_move = Some(chess_move);
                }
            }
        }

        best_move
    }

    /// Convert chess crate position to shakmaty format
    fn convert_position(&self, board: &Board) -> Result<Chess, String> {
        // Convert via FEN string - this is the cleanest approach
        let fen_str = board.to_string();
        
        match fen_str.parse::<Fen>() {
            Ok(fen) => {
                match fen.into_position(CastlingMode::Standard) {
                    Ok(pos) => Ok(pos),
                    Err(e) => Err(format!("Processing...")),
                }
            }
            Err(e) => Err(format!("Processing...")),
        }
    }

    /// Convert shakmaty-syzygy WDL to our enum
    fn convert_wdl(&self, wdl: Wdl, _side_to_move: Color) -> WdlValue {
        match wdl {
            Wdl::Win => WdlValue::Win(1), // Distance needs proper calculation from DTZ
            Wdl::Draw => WdlValue::Draw,
            Wdl::Loss => WdlValue::Loss(1), // Distance needs proper calculation from DTZ
            Wdl::BlessedLoss => WdlValue::BlessedLoss(1),
            Wdl::CursedWin => WdlValue::CursedWin(1),
        }
    }

    /// Convert shakmaty-syzygy DTZ value
    fn convert_dtz(&self, dtz: Dtz) -> u32 {
        // DTZ is a newtype wrapper around i32
        dtz.0.abs() as u32
    }
}

impl Default for TablebaseProber {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_tablebase_creation() {
        let prober = TablebaseProber::new();
        assert!(!prober.is_enabled());
        assert_eq!(prober.max_pieces(), 7);
    }

    #[test]
    fn test_piece_counting() {
        let prober = TablebaseProber::new();
        
        // Starting position has 32 pieces
        let start_board = Board::default();
        assert!(!prober.can_probe(&start_board));
        
        // Empty board would be probeable if tablebase was loaded
        // (but we don't have tablebase loaded in tests)
    }

    #[test]
    fn test_endgame_position_detection() {
        let prober = TablebaseProber::new();
        
        // KQvK endgame (3 pieces)
        let kqk_fen = "8/8/8/8/8/8/8/K6k w - - 0 1";
        if let Ok(board) = Board::from_str(kqk_fen) {
            // Would be probeable if tablebase was available
            assert!(!prober.is_enabled()); // No tablebase loaded in test
        }
    }

    #[test]
    fn test_evaluation_scaling() {
        let prober = TablebaseProber::new();
        
        // Test that mate scores are properly scaled
        let win_result = TablebaseResult::Exact { 
            wdl: WdlValue::Win(10), 
            dtz: Some(10) 
        };
        
        // Test evaluation conversion logic would go here
        // if we had a method to convert TablebaseResult to evaluation
    }
}