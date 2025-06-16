use chess::{Board, ChessMove};
use std::collections::HashMap;
use std::str::FromStr;

/// Opening book entry containing position evaluation and recommended moves
#[derive(Debug, Clone)]
pub struct OpeningEntry {
    pub evaluation: f32,
    pub best_moves: Vec<(ChessMove, f32)>, // (move, relative_strength)
    pub name: String,
    pub eco_code: Option<String>, // ECO (Encyclopedia of Chess Openings) code
}

/// Opening book for chess positions
pub struct OpeningBook {
    /// Map from FEN string to opening entry
    entries: HashMap<String, OpeningEntry>,
}

impl OpeningBook {
    /// Create a new opening book
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }
    
    /// Create a basic opening book with common openings
    pub fn with_standard_openings() -> Self {
        let mut book = Self::new();
        book.add_standard_openings();
        book
    }
    
    /// Add an opening entry
    pub fn add_opening(
        &mut self,
        fen: &str,
        evaluation: f32,
        best_moves: Vec<(ChessMove, f32)>,
        name: String,
        eco_code: Option<String>,
    ) -> Result<(), String> {
        // Validate FEN by parsing it
        Board::from_str(fen).map_err(|e| format!("Invalid FEN: {}", e))?;
        
        let entry = OpeningEntry {
            evaluation,
            best_moves,
            name,
            eco_code,
        };
        
        self.entries.insert(fen.to_string(), entry);
        Ok(())
    }
    
    /// Look up position in opening book
    pub fn lookup(&self, board: &Board) -> Option<&OpeningEntry> {
        let fen = board.to_string();
        self.entries.get(&fen)
    }
    
    /// Check if position is in opening book
    pub fn contains(&self, board: &Board) -> bool {
        let fen = board.to_string();
        self.entries.contains_key(&fen)
    }
    
    /// Get all known openings
    pub fn get_all_openings(&self) -> &HashMap<String, OpeningEntry> {
        &self.entries
    }
    
    /// Add standard chess openings
    fn add_standard_openings(&mut self) {
        // Starting position
        if let Ok(board) = Board::from_str("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1") {
            let moves = vec![
                (ChessMove::from_str("e2e4").unwrap(), 1.0),   // King's Pawn
                (ChessMove::from_str("d2d4").unwrap(), 0.9),   // Queen's Pawn  
                (ChessMove::from_str("g1f3").unwrap(), 0.8),   // King's Knight
                (ChessMove::from_str("c2c4").unwrap(), 0.7),   // English Opening
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.0,
                moves,
                "Starting Position".to_string(),
                None,
            );
        }
        
        // King's Pawn Game: 1.e4
        if let Ok(board) = Board::from_str("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1") {
            let moves = vec![
                (ChessMove::from_str("e7e5").unwrap(), 1.0),   // King's Pawn response
                (ChessMove::from_str("c7c5").unwrap(), 0.9),   // Sicilian Defense
                (ChessMove::from_str("e7e6").unwrap(), 0.7),   // French Defense
                (ChessMove::from_str("c7c6").unwrap(), 0.6),   // Caro-Kann Defense
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.25,
                moves,
                "King's Pawn Game".to_string(),
                Some("B00".to_string()),
            );
        }
        
        // King's Pawn Game: 1.e4 e5
        if let Ok(board) = Board::from_str("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2") {
            let moves = vec![
                (ChessMove::from_str("g1f3").unwrap(), 1.0),   // King's Knight Attack
                (ChessMove::from_str("f2f4").unwrap(), 0.6),   // King's Gambit
                (ChessMove::from_str("b1c3").unwrap(), 0.5),   // Vienna Game
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.15,
                moves,
                "Open Game".to_string(),
                Some("C20".to_string()),
            );
        }
        
        // Sicilian Defense: 1.e4 c5
        if let Ok(board) = Board::from_str("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2") {
            let moves = vec![
                (ChessMove::from_str("g1f3").unwrap(), 1.0),   // Open Sicilian
                (ChessMove::from_str("b1c3").unwrap(), 0.7),   // Closed Sicilian
                (ChessMove::from_str("f2f4").unwrap(), 0.5),   // Grand Prix Attack
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.3,
                moves,
                "Sicilian Defense".to_string(),
                Some("B20".to_string()),
            );
        }
        
        // Queen's Pawn Game: 1.d4
        if let Ok(board) = Board::from_str("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1") {
            let moves = vec![
                (ChessMove::from_str("d7d5").unwrap(), 1.0),   // Queen's Gambit
                (ChessMove::from_str("g8f6").unwrap(), 0.9),   // Indian Defenses
                (ChessMove::from_str("f7f5").unwrap(), 0.4),   // Dutch Defense
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.2,
                moves,
                "Queen's Pawn Game".to_string(),
                Some("D00".to_string()),
            );
        }
        
        // Queen's Gambit: 1.d4 d5 2.c4
        if let Ok(board) = Board::from_str("rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2") {
            let moves = vec![
                (ChessMove::from_str("d5c4").unwrap(), 0.7),   // Queen's Gambit Accepted
                (ChessMove::from_str("e7e6").unwrap(), 1.0),   // Queen's Gambit Declined
                (ChessMove::from_str("c7c6").unwrap(), 0.8),   // Slav Defense
                (ChessMove::from_str("g8f6").unwrap(), 0.8),   // Various Defenses
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.3,
                moves,
                "Queen's Gambit".to_string(),
                Some("D06".to_string()),
            );
        }
        
        // Italian Game: 1.e4 e5 2.Nf3 Nc6 3.Bc4
        if let Ok(board) = Board::from_str("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3") {
            let moves = vec![
                (ChessMove::from_str("g8f6").unwrap(), 1.0),   // Italian Game main line
                (ChessMove::from_str("f7f5").unwrap(), 0.6),   // Rousseau Gambit
                (ChessMove::from_str("f8e7").unwrap(), 0.7),   // Hungarian Defense
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.4,
                moves,
                "Italian Game".to_string(),
                Some("C50".to_string()),
            );
        }
        
        // Ruy Lopez: 1.e4 e5 2.Nf3 Nc6 3.Bb5
        if let Ok(board) = Board::from_str("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3") {
            let moves = vec![
                (ChessMove::from_str("a7a6").unwrap(), 1.0),   // Morphy Defense
                (ChessMove::from_str("g8f6").unwrap(), 0.8),   // Berlin Defense
                (ChessMove::from_str("f7f5").unwrap(), 0.4),   // Schliemann Defense
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.5,
                moves,
                "Ruy Lopez".to_string(),
                Some("C60".to_string()),
            );
        }
    }
    
    /// Get statistics about the opening book
    pub fn stats(&self) -> OpeningBookStats {
        let total_positions = self.entries.len();
        let eco_codes: std::collections::HashSet<_> = self.entries.values()
            .filter_map(|entry| entry.eco_code.as_ref())
            .collect();
        let eco_classifications = eco_codes.len();
        
        let avg_moves_per_position = if total_positions > 0 {
            self.entries.values()
                .map(|entry| entry.best_moves.len())
                .sum::<usize>() as f32 / total_positions as f32
        } else {
            0.0
        };
        
        OpeningBookStats {
            total_positions,
            eco_classifications,
            avg_moves_per_position,
        }
    }
}

/// Opening book statistics
#[derive(Debug)]
pub struct OpeningBookStats {
    pub total_positions: usize,
    pub eco_classifications: usize,
    pub avg_moves_per_position: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_opening_book_creation() {
        let book = OpeningBook::new();
        assert_eq!(book.entries.len(), 0);
    }
    
    #[test]
    fn test_standard_openings() {
        let book = OpeningBook::with_standard_openings();
        assert!(book.entries.len() > 0);
        
        // Test starting position lookup
        let board = Board::default();
        let entry = book.lookup(&board);
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().name, "Starting Position");
    }
    
    #[test]
    fn test_opening_lookup() {
        let mut book = OpeningBook::new();
        let board = Board::default();
        
        // Should not be found initially
        assert!(!book.contains(&board));
        
        // Add entry
        let moves = vec![(ChessMove::from_str("e2e4").unwrap(), 1.0)];
        book.add_opening(
            &board.to_string(),
            0.0,
            moves,
            "Test Opening".to_string(),
            None,
        ).unwrap();
        
        // Should be found now
        assert!(book.contains(&board));
        let entry = book.lookup(&board).unwrap();
        assert_eq!(entry.name, "Test Opening");
        assert_eq!(entry.best_moves.len(), 1);
    }
    
    #[test]
    fn test_stats() {
        let book = OpeningBook::with_standard_openings();
        let stats = book.stats();
        
        assert!(stats.total_positions > 0);
        assert!(stats.avg_moves_per_position > 0.0);
    }
}