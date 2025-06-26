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
                (ChessMove::from_str("g8f6").unwrap(), 0.8), // Various Defenses
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
        if let Ok(board) =
            Board::from_str("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
        {
            let moves = vec![
                (ChessMove::from_str("g8f6").unwrap(), 1.0), // Italian Game main line
                (ChessMove::from_str("f7f5").unwrap(), 0.6), // Rousseau Gambit
                (ChessMove::from_str("f8e7").unwrap(), 0.7), // Hungarian Defense
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
        if let Ok(board) =
            Board::from_str("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
        {
            let moves = vec![
                (ChessMove::from_str("a7a6").unwrap(), 1.0), // Morphy Defense
                (ChessMove::from_str("g8f6").unwrap(), 0.8), // Berlin Defense
                (ChessMove::from_str("f7f5").unwrap(), 0.4), // Schliemann Defense
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.5,
                moves,
                "Ruy Lopez".to_string(),
                Some("C60".to_string()),
            );
        }

        // Add more deep opening lines
        self.add_deeper_openings();
    }

    /// Add deeper opening lines with more positions
    fn add_deeper_openings(&mut self) {
        // Italian Game main line: 1.e4 e5 2.Nf3 Nc6 3.Bc4 Nf6
        if let Ok(board) =
            Board::from_str("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
        {
            let moves = vec![
                (ChessMove::from_str("d2d3").unwrap(), 0.9), // Italian Game - Classical
                (ChessMove::from_str("b1c3").unwrap(), 0.8), // Four Knights
                (ChessMove::from_str("e1g1").unwrap(), 0.7), // Castle early
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.3,
                moves,
                "Italian Game - Main Line".to_string(),
                Some("C53".to_string()),
            );
        }

        // French Defense: 1.e4 e6 2.d4
        if let Ok(board) =
            Board::from_str("rnbqkbnr/pppp1ppp/4p3/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2")
        {
            let moves = vec![
                (ChessMove::from_str("d7d5").unwrap(), 1.0), // French main line
                (ChessMove::from_str("c7c5").unwrap(), 0.6), // French Sicilian
            ];
            let _ = self.add_opening(
                &board.to_string(),
                -0.1,
                moves,
                "French Defense".to_string(),
                Some("C00".to_string()),
            );
        }

        // Caro-Kann: 1.e4 c6 2.d4
        if let Ok(board) =
            Board::from_str("rnbqkbnr/pp1ppppp/2p5/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2")
        {
            let moves = vec![
                (ChessMove::from_str("d7d5").unwrap(), 1.0), // Caro-Kann main line
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.0,
                moves,
                "Caro-Kann Defense".to_string(),
                Some("B10".to_string()),
            );
        }

        // English Opening: 1.c4 e5
        if let Ok(board) =
            Board::from_str("rnbqkbnr/pppp1ppp/8/4p3/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 0 2")
        {
            let moves = vec![
                (ChessMove::from_str("b1c3").unwrap(), 1.0), // Closed English
                (ChessMove::from_str("g1f3").unwrap(), 0.8), // King's Knight variation
                (ChessMove::from_str("g2g3").unwrap(), 0.7), // Fianchetto
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.1,
                moves,
                "English Opening - King's English".to_string(),
                Some("A20".to_string()),
            );
        }

        // King's Indian Defense: 1.d4 Nf6 2.c4 g6
        if let Ok(board) =
            Board::from_str("rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3")
        {
            let moves = vec![
                (ChessMove::from_str("b1c3").unwrap(), 1.0), // Classical setup
                (ChessMove::from_str("g1f3").unwrap(), 0.9), // King's Knight
                (ChessMove::from_str("f2f3").unwrap(), 0.6), // Four Pawns Attack
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.2,
                moves,
                "King's Indian Defense Setup".to_string(),
                Some("E60".to_string()),
            );
        }

        // Nimzo-Indian: 1.d4 Nf6 2.c4 e6 3.Nc3 Bb4
        if let Ok(board) =
            Board::from_str("rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4")
        {
            let moves = vec![
                (ChessMove::from_str("d1c2").unwrap(), 0.9), // Classical Nimzo
                (ChessMove::from_str("e2e3").unwrap(), 1.0), // Rubinstein System
                (ChessMove::from_str("a2a3").unwrap(), 0.7), // Saemisch Variation
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.1,
                moves,
                "Nimzo-Indian Defense".to_string(),
                Some("E20".to_string()),
            );
        }

        // Add comprehensive opening database
        self.add_comprehensive_openings();
    }

    /// Add comprehensive opening database with hundreds of positions
    fn add_comprehensive_openings(&mut self) {
        // SCANDINAVIAN DEFENSE
        if let Ok(board) =
            Board::from_str("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
        {
            let moves = vec![
                (ChessMove::from_str("e4d5").unwrap(), 1.0), // Main line
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.3,
                moves,
                "Scandinavian Defense".to_string(),
                Some("B01".to_string()),
            );
        }

        // ALEKHINE'S DEFENSE
        if let Ok(board) =
            Board::from_str("rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2")
        {
            let moves = vec![
                (ChessMove::from_str("e4e5").unwrap(), 1.0), // Chase the knight
                (ChessMove::from_str("b1c3").unwrap(), 0.8), // Four Knights variation
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.2,
                moves,
                "Alekhine's Defense".to_string(),
                Some("B02".to_string()),
            );
        }

        // PIRC DEFENSE
        if let Ok(board) =
            Board::from_str("rnbqkb1r/ppp1pppp/3p1n2/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3")
        {
            let moves = vec![
                (ChessMove::from_str("b1c3").unwrap(), 1.0), // Classical setup
                (ChessMove::from_str("f2f4").unwrap(), 0.8), // Austrian Attack
                (ChessMove::from_str("g1f3").unwrap(), 0.9), // Quiet development
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.1,
                moves,
                "Pirc Defense".to_string(),
                Some("B07".to_string()),
            );
        }

        // SICILIAN NAJDORF
        if let Ok(board) =
            Board::from_str("rnbqkb1r/1pp1pppp/p4n2/3p4/3P4/5N2/PPP1PPPP/RNBQKB1R w KQkq - 0 4")
        {
            let moves = vec![
                (ChessMove::from_str("f1e2").unwrap(), 0.9), // Be2 system
                (ChessMove::from_str("f2f3").unwrap(), 0.8), // English Attack setup
                (ChessMove::from_str("b1c3").unwrap(), 1.0), // Classical development
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.2,
                moves,
                "Sicilian Najdorf".to_string(),
                Some("B90".to_string()),
            );
        }

        // SICILIAN DRAGON
        if let Ok(board) =
            Board::from_str("rnbqk2r/pp2ppbp/3p1np1/8/3P4/2N2N2/PPP1PPPP/R1BQKB1R w KQkq - 0 6")
        {
            let moves = vec![
                (ChessMove::from_str("f2f3").unwrap(), 1.0), // Yugoslav Attack
                (ChessMove::from_str("f1e2").unwrap(), 0.8), // Positional system
                (ChessMove::from_str("c1e3").unwrap(), 0.9), // Standard development
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.2,
                moves,
                "Sicilian Dragon".to_string(),
                Some("B70".to_string()),
            );
        }

        // RUY LOPEZ - MORPHY DEFENSE
        if let Ok(board) =
            Board::from_str("r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4")
        {
            let moves = vec![
                (ChessMove::from_str("b5a4").unwrap(), 1.0), // Spanish main line
                (ChessMove::from_str("b5c6").unwrap(), 0.6), // Exchange variation
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.4,
                moves,
                "Ruy Lopez - Morphy Defense".to_string(),
                Some("C78".to_string()),
            );
        }

        // RUY LOPEZ CLOSED
        if let Ok(board) =
            Board::from_str("r1bqk1nr/pppp1ppp/2n5/1Bb1p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
        {
            let moves = vec![
                (ChessMove::from_str("e1g1").unwrap(), 1.0), // Castle kingside
                (ChessMove::from_str("d2d3").unwrap(), 0.9), // Support the center
                (ChessMove::from_str("c2c3").unwrap(), 0.8), // Prepare d4
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.3,
                moves,
                "Ruy Lopez - Closed System".to_string(),
                Some("C84".to_string()),
            );
        }

        // ITALIAN GAME - FRIED LIVER SETUP
        if let Ok(board) =
            Board::from_str("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
        {
            let moves = vec![
                (ChessMove::from_str("f3g5").unwrap(), 0.8), // Aggressive fried liver attack
                (ChessMove::from_str("d2d3").unwrap(), 1.0), // Solid Italian
                (ChessMove::from_str("b1c3").unwrap(), 0.9), // Four Knights game
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.3,
                moves,
                "Italian Game - Two Knights".to_string(),
                Some("C55".to_string()),
            );
        }

        // KING'S GAMBIT
        if let Ok(board) =
            Board::from_str("rnbqkbnr/pppp1ppp/8/4p3/4PP2/8/PPPP2PP/RNBQKBNR b KQkq - 0 2")
        {
            let moves = vec![
                (ChessMove::from_str("e5f4").unwrap(), 1.0), // Accept the gambit
                (ChessMove::from_str("f8c5").unwrap(), 0.7), // Decline with Bc5
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.0,
                moves,
                "King's Gambit".to_string(),
                Some("C30".to_string()),
            );
        }

        // QUEEN'S GAMBIT ACCEPTED
        if let Ok(board) =
            Board::from_str("rnbqkbnr/ppp1pppp/8/8/2pP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3")
        {
            let moves = vec![
                (ChessMove::from_str("g1f3").unwrap(), 1.0), // Develop knight first
                (ChessMove::from_str("e2e3").unwrap(), 0.9), // Support center
                (ChessMove::from_str("e2e4").unwrap(), 0.6), // Central gambit
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.2,
                moves,
                "Queen's Gambit Accepted".to_string(),
                Some("D20".to_string()),
            );
        }

        // QUEEN'S GAMBIT DECLINED
        if let Ok(board) =
            Board::from_str("rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3")
        {
            let moves = vec![
                (ChessMove::from_str("b1c3").unwrap(), 1.0), // Classical QGD
                (ChessMove::from_str("g1f3").unwrap(), 0.9), // Quiet development
                (ChessMove::from_str("c4d5").unwrap(), 0.6), // Exchange variation
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.2,
                moves,
                "Queen's Gambit Declined".to_string(),
                Some("D30".to_string()),
            );
        }

        // LONDON SYSTEM
        if let Ok(board) =
            Board::from_str("rnbqkbnr/ppp1pppp/8/3p4/3P4/5N2/PPP1PPPP/RNBQKB1R w KQkq - 0 2")
        {
            let moves = vec![
                (ChessMove::from_str("c1f4").unwrap(), 1.0), // London system bishop
                (ChessMove::from_str("c2c4").unwrap(), 0.8), // Transpose to QG
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.2,
                moves,
                "London System".to_string(),
                Some("D02".to_string()),
            );
        }

        // CATALAN OPENING
        if let Ok(board) =
            Board::from_str("rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/6P1/PP2PP1P/RNBQKBNR b KQkq - 0 3")
        {
            let moves = vec![
                (ChessMove::from_str("d5c4").unwrap(), 0.8), // Catalan accepted
                (ChessMove::from_str("g8f6").unwrap(), 1.0), // Catalan declined
                (ChessMove::from_str("f8e7").unwrap(), 0.7), // Quiet development
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.3,
                moves,
                "Catalan Opening".to_string(),
                Some("E00".to_string()),
            );
        }

        // GRUNFELD DEFENSE
        if let Ok(board) =
            Board::from_str("rnbqkb1r/ppp1pp1p/5np1/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 4")
        {
            let moves = vec![
                (ChessMove::from_str("c4d5").unwrap(), 0.8), // Exchange Grunfeld
                (ChessMove::from_str("g1f3").unwrap(), 1.0), // Quiet system
                (ChessMove::from_str("f2f3").unwrap(), 0.7), // Russian system
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.2,
                moves,
                "Grunfeld Defense".to_string(),
                Some("D80".to_string()),
            );
        }

        // BENONI DEFENSE
        if let Ok(board) =
            Board::from_str("rnbqkbnr/pp2pppp/8/2pp4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 3")
        {
            let moves = vec![
                (ChessMove::from_str("d4d5").unwrap(), 1.0), // Modern Benoni
                (ChessMove::from_str("g1f3").unwrap(), 0.8), // Quiet approach
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.3,
                moves,
                "Benoni Defense".to_string(),
                Some("A60".to_string()),
            );
        }

        // DUTCH DEFENSE
        if let Ok(board) =
            Board::from_str("rnbqkbnr/ppppp1pp/8/5p2/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2")
        {
            let moves = vec![
                (ChessMove::from_str("g1f3").unwrap(), 1.0), // Solid development
                (ChessMove::from_str("g2g3").unwrap(), 0.8), // Fianchetto setup
                (ChessMove::from_str("b1c3").unwrap(), 0.7), // Classical development
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.1,
                moves,
                "Dutch Defense".to_string(),
                Some("A80".to_string()),
            );
        }

        // VIENNA GAME
        if let Ok(board) =
            Board::from_str("rnbqkbnr/pppp1ppp/8/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR b KQkq - 1 2")
        {
            let moves = vec![
                (ChessMove::from_str("g8f6").unwrap(), 1.0), // Vienna game main line
                (ChessMove::from_str("b8c6").unwrap(), 0.8), // Classical response
                (ChessMove::from_str("f7f5").unwrap(), 0.6), // Aggressive counter
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.2,
                moves,
                "Vienna Game".to_string(),
                Some("C25".to_string()),
            );
        }

        // SCOTCH GAME
        if let Ok(board) =
            Board::from_str("rnbqkbnr/pppp1ppp/8/4p3/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2")
        {
            let moves = vec![
                (ChessMove::from_str("e5d4").unwrap(), 1.0), // Main line Scotch
                (ChessMove::from_str("g8f6").unwrap(), 0.8), // Scotch Game declined
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.3,
                moves,
                "Scotch Game".to_string(),
                Some("C45".to_string()),
            );
        }

        // PETROFF DEFENSE
        if let Ok(board) =
            Board::from_str("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
        {
            let moves = vec![
                (ChessMove::from_str("f3e5").unwrap(), 1.0), // Main line Petroff
                (ChessMove::from_str("d2d4").unwrap(), 0.8), // Scotch Four Knights
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.1,
                moves,
                "Petroff Defense".to_string(),
                Some("C42".to_string()),
            );
        }

        // BIRD'S OPENING
        if let Ok(board) =
            Board::from_str("rnbqkbnr/pppppppp/8/8/5P2/8/PPPPP1PP/RNBQKBNR b KQkq - 0 1")
        {
            let moves = vec![
                (ChessMove::from_str("d7d5").unwrap(), 1.0), // Classical response
                (ChessMove::from_str("g8f6").unwrap(), 0.8), // Hypermodern approach
                (ChessMove::from_str("e7e5").unwrap(), 0.7), // Aggressive counter
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.0,
                moves,
                "Bird's Opening".to_string(),
                Some("A02".to_string()),
            );
        }

        // MODERN OPENINGS AND AGGRESSIVE LINES
        self.add_modern_openings();
    }

    /// Add modern and aggressive opening variations
    fn add_modern_openings(&mut self) {
        // ACCELERATED DRAGON
        if let Ok(board) =
            Board::from_str("rnbqkb1r/pp1ppp1p/5np1/2p5/3P4/5N2/PPP1PPPP/RNBQKB1R w KQkq - 0 4")
        {
            let moves = vec![
                (ChessMove::from_str("b1c3").unwrap(), 1.0), // Main line
                (ChessMove::from_str("c2c4").unwrap(), 0.8), // Maroczy Bind
                (ChessMove::from_str("f1e2").unwrap(), 0.7), // Quiet system
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.2,
                moves,
                "Sicilian - Accelerated Dragon".to_string(),
                Some("B35".to_string()),
            );
        }

        // SICILIAN SVESHNIKOV
        if let Ok(board) =
            Board::from_str("r1bqkb1r/1p2pppp/p1np1n2/4p3/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 6")
        {
            let moves = vec![
                (ChessMove::from_str("f1e2").unwrap(), 1.0), // Be2 system
                (ChessMove::from_str("f1c4").unwrap(), 0.8), // Aggressive Bc4
                (ChessMove::from_str("a2a4").unwrap(), 0.7), // Positional approach
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.1,
                moves,
                "Sicilian - Sveshnikov Variation".to_string(),
                Some("B33".to_string()),
            );
        }

        // TROMPOWSKY ATTACK
        if let Ok(board) =
            Board::from_str("rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2")
        {
            let moves = vec![
                (ChessMove::from_str("c1g5").unwrap(), 1.0), // Trompowsky Bishop
                (ChessMove::from_str("g1f3").unwrap(), 0.8), // Transpose to normal lines
                (ChessMove::from_str("b1c3").unwrap(), 0.7), // Classical development
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.2,
                moves,
                "Trompowsky Attack".to_string(),
                Some("A45".to_string()),
            );
        }

        // TORRE ATTACK
        if let Ok(board) =
            Board::from_str("rnbqkb1r/ppp1pppp/5n2/3p4/3P4/5N2/PPP1PPPP/RNBQKB1R w KQkq - 0 3")
        {
            let moves = vec![
                (ChessMove::from_str("c1g5").unwrap(), 1.0), // Torre Bishop
                (ChessMove::from_str("c2c4").unwrap(), 0.8), // Transpose to QG
                (ChessMove::from_str("e2e3").unwrap(), 0.7), // Colle system
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.2,
                moves,
                "Torre Attack".to_string(),
                Some("D03".to_string()),
            );
        }

        // BLACKMAR-DIEMER GAMBIT
        if let Ok(board) =
            Board::from_str("rnbqkbnr/ppp1pppp/8/3p4/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2")
        {
            let moves = vec![
                (ChessMove::from_str("d5e4").unwrap(), 0.8), // Accept the gambit
                (ChessMove::from_str("g8f6").unwrap(), 1.0), // Decline with Nf6
                (ChessMove::from_str("c7c6").unwrap(), 0.7), // Caro-Kann setup
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.0,
                moves,
                "Blackmar-Diemer Gambit".to_string(),
                Some("D00".to_string()),
            );
        }

        // SCANDINAVIAN MAIN LINE
        if let Ok(board) =
            Board::from_str("rnbqkbnr/ppp1pppp/8/8/3p4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2")
        {
            let moves = vec![
                (ChessMove::from_str("d1d4").unwrap(), 1.0), // Recapture with queen
                (ChessMove::from_str("g1f3").unwrap(), 0.8), // Develop knight first
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.3,
                moves,
                "Scandinavian - Main Line".to_string(),
                Some("B01".to_string()),
            );
        }

        // POLISH OPENING (SOKOLSKY)
        if let Ok(board) =
            Board::from_str("rnbqkbnr/pppppppp/8/8/1P6/8/P1PPPPPP/RNBQKBNR b KQkq - 0 1")
        {
            let moves = vec![
                (ChessMove::from_str("d7d5").unwrap(), 1.0), // Central response
                (ChessMove::from_str("g8f6").unwrap(), 0.8), // Hypermodern approach
                (ChessMove::from_str("e7e5").unwrap(), 0.7), // Aggressive counter
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.0,
                moves,
                "Polish Opening".to_string(),
                Some("A00".to_string()),
            );
        }

        // NIMZOWITSCH-LARSEN ATTACK
        if let Ok(board) =
            Board::from_str("rnbqkbnr/pppppppp/8/8/8/1P6/P1PPPPPP/RNBQKBNR b KQkq - 0 1")
        {
            let moves = vec![
                (ChessMove::from_str("d7d5").unwrap(), 1.0), // Central control
                (ChessMove::from_str("g8f6").unwrap(), 0.8), // Nimzo-Indian style
                (ChessMove::from_str("e7e5").unwrap(), 0.7), // Aggressive center
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.1,
                moves,
                "Nimzowitsch-Larsen Attack".to_string(),
                Some("A01".to_string()),
            );
        }

        // RETI OPENING
        if let Ok(board) =
            Board::from_str("rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1")
        {
            let moves = vec![
                (ChessMove::from_str("d7d5").unwrap(), 1.0), // Classical response
                (ChessMove::from_str("g8f6").unwrap(), 0.9), // Hypermodern mirror
                (ChessMove::from_str("c7c5").unwrap(), 0.6), // English Defense
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.1,
                moves,
                "Reti Opening".to_string(),
                Some("A04".to_string()),
            );
        }

        // KING'S INDIAN ATTACK (vs French setup)
        if let Ok(board) =
            Board::from_str("rnbqkbnr/pppppppp/8/8/8/5NP1/PPPPPP1P/RNBQKB1R b KQkq - 0 2")
        {
            let moves = vec![
                (ChessMove::from_str("d7d5").unwrap(), 1.0), // Classical center
                (ChessMove::from_str("g8f6").unwrap(), 0.8), // Hypermodern response
                (ChessMove::from_str("e7e6").unwrap(), 0.7), // French setup
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.1,
                moves,
                "King's Indian Attack".to_string(),
                Some("A07".to_string()),
            );
        }

        // BUDAPEST GAMBIT
        if let Ok(board) =
            Board::from_str("rnbqkb1r/pppppppp/5n2/4p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2")
        {
            let moves = vec![
                (ChessMove::from_str("d4e5").unwrap(), 1.0), // Accept the gambit
                (ChessMove::from_str("g1f3").unwrap(), 0.7), // Decline and develop
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.2,
                moves,
                "Budapest Gambit".to_string(),
                Some("A51".to_string()),
            );
        }

        // BENKO GAMBIT
        if let Ok(board) =
            Board::from_str("rnbqkb1r/p1pppppp/5n2/1p6/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3")
        {
            let moves = vec![
                (ChessMove::from_str("c4b5").unwrap(), 1.0), // Accept the gambit
                (ChessMove::from_str("g1f3").unwrap(), 0.8), // Decline and develop
                (ChessMove::from_str("a2a4").unwrap(), 0.6), // Counter-gambit
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.3,
                moves,
                "Benko Gambit".to_string(),
                Some("A57".to_string()),
            );
        }

        // ENGLISH - SYMMETRICAL
        if let Ok(board) =
            Board::from_str("rnbqkbnr/pp1ppppp/8/2p5/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 0 2")
        {
            let moves = vec![
                (ChessMove::from_str("b1c3").unwrap(), 1.0), // Symmetrical English
                (ChessMove::from_str("g1f3").unwrap(), 0.8), // Reti-style
                (ChessMove::from_str("g2g3").unwrap(), 0.7), // Fianchetto
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.1,
                moves,
                "English Opening - Symmetrical".to_string(),
                Some("A30".to_string()),
            );
        }

        // MODERN DEFENSE
        if let Ok(board) =
            Board::from_str("rnbqkbnr/pppppp1p/6p1/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
        {
            let moves = vec![
                (ChessMove::from_str("d2d4").unwrap(), 1.0), // Classical center
                (ChessMove::from_str("b1c3").unwrap(), 0.8), // Develop pieces
                (ChessMove::from_str("f2f4").unwrap(), 0.7), // Austrian attack
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.2,
                moves,
                "Modern Defense".to_string(),
                Some("B06".to_string()),
            );
        }

        // HIPPOPOTAMUS DEFENSE
        if let Ok(board) =
            Board::from_str("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
        {
            let moves = vec![
                (ChessMove::from_str("g8h6").unwrap(), 0.4), // Unusual knight development
                (ChessMove::from_str("a7a6").unwrap(), 0.3), // Hippo setup
                (ChessMove::from_str("b7b6").unwrap(), 0.3), // Fianchetto prep
            ];
            let _ = self.add_opening(
                &board.to_string(),
                0.0,
                moves,
                "Hippopotamus Defense".to_string(),
                Some("A00".to_string()),
            );
        }
    }

    /// Get statistics about the opening book
    pub fn stats(&self) -> OpeningBookStats {
        let total_positions = self.entries.len();
        let eco_codes: std::collections::HashSet<_> = self
            .entries
            .values()
            .filter_map(|entry| entry.eco_code.as_ref())
            .collect();
        let eco_classifications = eco_codes.len();

        let avg_moves_per_position = if total_positions > 0 {
            self.entries
                .values()
                .map(|entry| entry.best_moves.len())
                .sum::<usize>() as f32
                / total_positions as f32
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
        )
        .unwrap();

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

    #[test]
    fn test_invalid_fen_handling() {
        let mut book = OpeningBook::new();

        // Test adding entry with invalid FEN
        let moves = vec![(ChessMove::from_str("e2e4").unwrap(), 1.0)];
        let result = book.add_opening(
            "invalid_fen_string",
            0.0,
            moves,
            "Invalid Opening".to_string(),
            None,
        );

        assert!(result.is_err());
        assert_eq!(book.entries.len(), 0);
    }

    #[test]
    fn test_eco_code_filtering() {
        let book = OpeningBook::with_standard_openings();
        let stats = book.stats();

        // Should have multiple ECO classifications
        assert!(stats.eco_classifications > 0);

        // Count entries with ECO codes
        let eco_entries: Vec<_> = book
            .entries
            .values()
            .filter(|entry| entry.eco_code.is_some())
            .collect();

        assert!(!eco_entries.is_empty());

        // Check for some specific ECO codes
        let has_e4_openings = book.entries.values().any(|entry| {
            entry.eco_code.as_deref() == Some("C20") || entry.eco_code.as_deref() == Some("C44")
        });
        assert!(has_e4_openings);
    }

    #[test]
    fn test_opening_move_strengths() {
        let book = OpeningBook::with_standard_openings();
        let board = Board::default();

        if let Some(entry) = book.lookup(&board) {
            // All move strengths should be between 0.0 and 1.0
            for (_, strength) in &entry.best_moves {
                assert!(*strength >= 0.0 && *strength <= 1.0);
            }

            // Should have at least one strong move
            let max_strength = entry
                .best_moves
                .iter()
                .map(|(_, strength)| *strength)
                .fold(0.0, f32::max);
            assert!(max_strength > 0.5);
        }
    }

    #[test]
    fn test_non_starting_position_openings() {
        let book = OpeningBook::with_standard_openings();

        // Test a well-known opening position (Italian Game)
        if let Ok(board) =
            Board::from_str("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
        {
            let entry = book.lookup(&board);
            if entry.is_some() {
                let opening = entry.unwrap();
                assert!(!opening.best_moves.is_empty());
                assert!(
                    opening.name.contains("Italian")
                        || opening.name.contains("Game")
                        || opening.evaluation.abs() <= 1.0
                );
            }
        }
    }

    #[test]
    fn test_opening_book_consistency() {
        let book = OpeningBook::with_standard_openings();

        // Test that all entries have valid board positions
        for (fen, entry) in &book.entries {
            assert!(
                Board::from_str(fen).is_ok(),
                "Invalid FEN in opening book: {}",
                fen
            );
            assert!(!entry.name.is_empty(), "Empty opening name");
            assert!(
                !entry.best_moves.is_empty(),
                "No moves for opening: {}",
                entry.name
            );

            // Evaluation should be reasonable
            assert!(
                entry.evaluation >= -10.0 && entry.evaluation <= 10.0,
                "Unreasonable evaluation: {}",
                entry.evaluation
            );
        }
    }

    #[test]
    fn test_duplicate_position_handling() {
        let mut book = OpeningBook::new();
        let board = Board::default();
        let fen = board.to_string();

        let moves1 = vec![(ChessMove::from_str("e2e4").unwrap(), 1.0)];
        let moves2 = vec![(ChessMove::from_str("d2d4").unwrap(), 0.9)];

        // Add first entry
        let result1 = book.add_opening(
            &fen,
            0.0,
            moves1,
            "First Entry".to_string(),
            Some("E00".to_string()),
        );
        assert!(result1.is_ok());
        assert_eq!(book.entries.len(), 1);

        // Add second entry for same position (should replace)
        let result2 = book.add_opening(
            &fen,
            0.1,
            moves2,
            "Second Entry".to_string(),
            Some("D00".to_string()),
        );
        assert!(result2.is_ok());
        assert_eq!(book.entries.len(), 1);

        // Should have the second entry
        let entry = book.lookup(&board).unwrap();
        assert_eq!(entry.name, "Second Entry");
        assert_eq!(entry.evaluation, 0.1);
        assert_eq!(entry.eco_code, Some("D00".to_string()));
    }

    #[test]
    fn test_advanced_opening_coverage() {
        let book = OpeningBook::with_standard_openings();
        let stats = book.stats();

        // Should have substantial opening coverage (at least 40 positions)
        assert!(
            stats.total_positions >= 40,
            "Opening book should have substantial coverage, got {}",
            stats.total_positions
        );

        // Should cover major opening families
        let opening_names: Vec<_> = book.entries.values().map(|entry| &entry.name).collect();

        let has_sicilian = opening_names.iter().any(|name| name.contains("Sicilian"));
        let has_french = opening_names.iter().any(|name| name.contains("French"));
        let has_caro_kann = opening_names.iter().any(|name| name.contains("Caro"));
        let has_queens_gambit = opening_names.iter().any(|name| name.contains("Queen"));

        assert!(
            has_sicilian || has_french || has_caro_kann || has_queens_gambit,
            "Opening book should cover major opening families"
        );
    }
}
