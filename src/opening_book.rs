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
#[derive(Clone)]
pub struct OpeningBook {
    /// Map from FEN string to opening entry
    entries: HashMap<String, OpeningEntry>,
}

impl Default for OpeningBook {
    fn default() -> Self {
        Self::new()
    }
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
        Board::from_str(fen).map_err(|_e| "Invalid FEN".to_string())?;

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

    /// Get a random opening move from the starting position
    pub fn get_random_opening(&self) -> Option<ChessMove> {
        use rand::seq::SliceRandom;
        let board = Board::default();
        if let Some(entry) = self.lookup(&board) {
            let moves: Vec<ChessMove> = entry.best_moves.iter().map(|(mv, _)| *mv).collect();
            moves.choose(&mut rand::thread_rng()).copied()
        } else {
            None
        }
    }

    /// Add standard chess openings
    fn add_standard_openings(&mut self) {
        // Starting position
        if let Ok(board) =
            Board::from_str("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        {
            let moves = vec![
                (ChessMove::from_str("e2e4").unwrap(), 1.0), // King's Pawn
                (ChessMove::from_str("d2d4").unwrap(), 0.9), // Queen's Pawn
                (ChessMove::from_str("g1f3").unwrap(), 0.8), // King's Knight
                (ChessMove::from_str("c2c4").unwrap(), 0.7), // English Opening
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
        if let Ok(board) =
            Board::from_str("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
        {
            let moves = vec![
                (ChessMove::from_str("e7e5").unwrap(), 1.0), // King's Pawn response
                (ChessMove::from_str("c7c5").unwrap(), 0.9), // Sicilian Defense
                (ChessMove::from_str("e7e6").unwrap(), 0.7), // French Defense
                (ChessMove::from_str("c7c6").unwrap(), 0.6), // Caro-Kann Defense
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
        if let Ok(board) =
            Board::from_str("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
        {
            let moves = vec![
                (ChessMove::from_str("g1f3").unwrap(), 1.0), // King's Knight Attack
                (ChessMove::from_str("f2f4").unwrap(), 0.6), // King's Gambit
                (ChessMove::from_str("b1c3").unwrap(), 0.5), // Vienna Game
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.15,
                moves,
                "Open Game".to_string(),
                Some("C20".to_string()),
            );
        }

        // Italian Game: 1.e4 e5 2.Nf3 Nc6 3.Bc4
        if let Ok(board) =
            Board::from_str("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
        {
            let moves = vec![
                (ChessMove::from_str("g8f6").unwrap(), 1.0), // Italian Game main line
                (ChessMove::from_str("f7f5").unwrap(), 0.6), // Rousseau Gambit
                (ChessMove::from_str("f8e7").unwrap(), 0.4), // Hungarian Defense
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.25,
                moves,
                "Italian Game".to_string(),
                Some("C50".to_string()),
            );
        }

        // Ruy Lopez: 1.e4 e5 2.Nf3 Nc6 3.Bb5
        if let Ok(board) =
            Board::from_str("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
        {
            let moves = vec![
                (ChessMove::from_str("a7a6").unwrap(), 1.0), // Morphy Defense
                (ChessMove::from_str("g8f6").unwrap(), 0.9), // Berlin Defense
                (ChessMove::from_str("f7f5").unwrap(), 0.4), // Schliemann Defense
                (ChessMove::from_str("b8d4").unwrap(), 0.3), // Bird Defense
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.3,
                moves,
                "Ruy Lopez".to_string(),
                Some("C60".to_string()),
            );
        }

        // Sicilian Defense: 1.e4 c5
        if let Ok(board) =
            Board::from_str("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
        {
            let moves = vec![
                (ChessMove::from_str("g1f3").unwrap(), 1.0), // Open Sicilian
                (ChessMove::from_str("b1c3").unwrap(), 0.7), // Closed Sicilian
                (ChessMove::from_str("f2f4").unwrap(), 0.5), // Grand Prix Attack
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.3,
                moves,
                "Sicilian Defense".to_string(),
                Some("B20".to_string()),
            );
        }

        // French Defense: 1.e4 e6
        if let Ok(board) =
            Board::from_str("rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
        {
            let moves = vec![
                (ChessMove::from_str("d2d4").unwrap(), 1.0), // French Defense main line
                (ChessMove::from_str("d2d3").unwrap(), 0.5), // King's Indian Attack
                (ChessMove::from_str("g1f3").unwrap(), 0.6), // Two Knights Defense
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.35,
                moves,
                "French Defense".to_string(),
                Some("C00".to_string()),
            );
        }

        // Caro-Kann Defense: 1.e4 c6
        if let Ok(board) =
            Board::from_str("rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
        {
            let moves = vec![
                (ChessMove::from_str("d2d4").unwrap(), 1.0), // Caro-Kann main line
                (ChessMove::from_str("b1c3").unwrap(), 0.7), // Two Knights Attack
                (ChessMove::from_str("f2f4").unwrap(), 0.4), // Hillbilly Attack
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.3,
                moves,
                "Caro-Kann Defense".to_string(),
                Some("B10".to_string()),
            );
        }

        // Queen's Pawn Game: 1.d4
        if let Ok(board) =
            Board::from_str("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1")
        {
            let moves = vec![
                (ChessMove::from_str("d7d5").unwrap(), 1.0), // Queen's Gambit
                (ChessMove::from_str("g8f6").unwrap(), 0.9), // Indian Defenses
                (ChessMove::from_str("f7f5").unwrap(), 0.4), // Dutch Defense
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
        if let Ok(board) =
            Board::from_str("rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2")
        {
            let moves = vec![
                (ChessMove::from_str("d5c4").unwrap(), 0.7), // Queen's Gambit Accepted
                (ChessMove::from_str("e7e6").unwrap(), 1.0), // Queen's Gambit Declined
                (ChessMove::from_str("c7c6").unwrap(), 0.8), // Slav Defense
                (ChessMove::from_str("d5d4").unwrap(), 0.5), // Marshall Defense
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.25,
                moves,
                "Queen's Gambit".to_string(),
                Some("D06".to_string()),
            );
        }

        // English Opening: 1.c4
        if let Ok(board) =
            Board::from_str("rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq - 0 1")
        {
            let moves = vec![
                (ChessMove::from_str("g8f6").unwrap(), 1.0), // Anglo-Indian
                (ChessMove::from_str("e7e5").unwrap(), 0.9), // Reversed Sicilian
                (ChessMove::from_str("c7c5").unwrap(), 0.8), // Symmetrical English
                (ChessMove::from_str("e7e6").unwrap(), 0.7), // Anglo-French
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.15,
                moves,
                "English Opening".to_string(),
                Some("A10".to_string()),
            );
        }

        // Nimzo-Indian Defense: 1.d4 Nf6 2.c4 e6 3.Nc3 Bb4
        if let Ok(board) =
            Board::from_str("rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4")
        {
            let moves = vec![
                (ChessMove::from_str("e2e3").unwrap(), 0.9), // Rubinstein System
                (ChessMove::from_str("f2f3").unwrap(), 0.7), // Kmoch Variation
                (ChessMove::from_str("a2a3").unwrap(), 0.8), // Saemisch Variation
                (ChessMove::from_str("d1c2").unwrap(), 1.0), // Classical Variation
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.1,
                moves,
                "Nimzo-Indian Defense".to_string(),
                Some("E20".to_string()),
            );
        }

        // King's Indian Defense setup: 1.d4 Nf6 2.c4 g6
        if let Ok(board) =
            Board::from_str("rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3")
        {
            let moves = vec![
                (ChessMove::from_str("b1c3").unwrap(), 1.0), // Classical setup
                (ChessMove::from_str("g1f3").unwrap(), 0.9), // Fianchetto setup
                (ChessMove::from_str("f2f3").unwrap(), 0.6), // Saemisch Variation
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.2,
                moves,
                "King's Indian Defense".to_string(),
                Some("E60".to_string()),
            );
        }

        // Sicilian Najdorf: 1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 a6
        if let Ok(board) =
            Board::from_str("rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6")
        {
            let moves = vec![
                (ChessMove::from_str("c1e3").unwrap(), 1.0), // English Attack
                (ChessMove::from_str("f2f3").unwrap(), 0.9), // Be3 system
                (ChessMove::from_str("h2h3").unwrap(), 0.7), // Positional setup
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.2,
                moves,
                "Sicilian Najdorf".to_string(),
                Some("B90".to_string()),
            );
        }
    }

    /// Get opening book statistics for evaluation
    pub fn get_statistics(&self) -> OpeningBookStats {
        let total_openings = self.entries.len();
        let eco_coverage = self
            .entries
            .values()
            .filter(|entry| entry.eco_code.is_some())
            .count();

        OpeningBookStats {
            total_openings,
            eco_coverage,
            avg_moves_per_opening: if total_openings > 0 {
                self.entries
                    .values()
                    .map(|entry| entry.best_moves.len())
                    .sum::<usize>() as f32
                    / total_openings as f32
            } else {
                0.0
            },
        }
    }
}

/// Statistics about the opening book coverage
#[derive(Debug, Clone)]
pub struct OpeningBookStats {
    pub total_openings: usize,
    pub eco_coverage: usize,
    pub avg_moves_per_opening: f32,
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
        assert!(!book.entries.is_empty());

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
        )
        .unwrap();

        // Should be found now
        assert!(book.contains(&board));
        let entry = book.lookup(&board).unwrap();
        assert_eq!(entry.name, "Test Opening");
        assert_eq!(entry.best_moves.len(), 1);
    }

    #[test]
    fn test_comprehensive_opening_coverage() {
        let book = OpeningBook::with_standard_openings();
        let stats = book.get_statistics();

        // Should have substantial opening coverage
        assert!(
            stats.total_openings >= 12,
            "Expected at least 12 openings, got {}",
            stats.total_openings
        );
        assert!(
            stats.eco_coverage >= 8,
            "Expected at least 8 ECO codes, got {}",
            stats.eco_coverage
        );
        assert!(
            stats.avg_moves_per_opening >= 2.0,
            "Expected average 2+ moves per opening"
        );

        // Test that major openings are covered
        let opening_names: Vec<_> = book.entries.values().map(|entry| &entry.name).collect();

        let has_sicilian = opening_names.iter().any(|name| name.contains("Sicilian"));
        let has_italian = opening_names.iter().any(|name| name.contains("Italian"));
        let has_ruy_lopez = opening_names.iter().any(|name| name.contains("Ruy Lopez"));
        let has_french = opening_names.iter().any(|name| name.contains("French"));

        assert!(has_sicilian, "Should have Sicilian Defense");
        assert!(has_italian, "Should have Italian Game");
        assert!(has_ruy_lopez, "Should have Ruy Lopez");
        assert!(has_french, "Should have French Defense");
    }
}
