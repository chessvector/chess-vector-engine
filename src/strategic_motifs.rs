//! Strategic Motif Recognition System
//!
//! This module implements a curated database of strategic chess patterns that provide
//! master-level positional understanding. Instead of storing millions of positions,
//! we focus on ~10K strategic motifs that capture the essence of positional play.

use chess::{Board, Piece, Square};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A strategic chess motif representing a meaningful positional pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategicMotif {
    /// Unique identifier for this motif
    pub id: u64,
    /// Pattern hash for fast matching
    pub pattern_hash: u64,
    /// Type of strategic pattern
    pub motif_type: MotifType,
    /// Strategic evaluation adjustment (-2.0 to +2.0)
    pub evaluation: f32,
    /// When this pattern is most relevant
    pub context: StrategicContext,
    /// Confidence in this pattern (0.0 to 1.0)
    pub confidence: f32,
    /// Source game references where this pattern was successful
    pub master_games: Vec<GameReference>,
    /// Human-readable description
    pub description: String,
}

/// Types of strategic motifs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MotifType {
    /// Pawn structure patterns
    PawnStructure(PawnPattern),
    /// Piece coordination and placement
    PieceCoordination(CoordinationPattern),
    /// King safety configurations
    KingSafety(SafetyPattern),
    /// Initiative and tempo patterns
    Initiative(InitiativePattern),
    /// Endgame-specific patterns
    Endgame(EndgamePattern),
    /// Opening-specific strategic ideas
    Opening(OpeningPattern),
}

/// Pawn structure patterns that affect strategic evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PawnPattern {
    /// Isolated pawn weaknesses
    IsolatedPawn {
        file_index: u8,
        is_white: bool,
        weakness_value: f32,
    },
    /// Doubled pawn formations
    DoubledPawns {
        file_index: u8,
        is_white: bool,
        count: u8,
    },
    /// Passed pawn advantages
    PassedPawn {
        square_index: u8,
        is_white: bool,
        advancement: f32,
    },
    /// Pawn chains and support structures
    PawnChain {
        base_square_index: u8,
        length: u8,
        is_white: bool,
    },
    /// Hanging pawns (abreast, unsupported)
    HangingPawns {
        file1_index: u8,
        file2_index: u8,
        is_white: bool,
    },
    /// Backward pawn weaknesses
    BackwardPawn { square_index: u8, is_white: bool },
    /// Pawn majority patterns
    PawnMajority {
        kingside: bool,
        is_white: bool,
        advantage: f32,
    },
}

/// Piece coordination patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationPattern {
    /// Knight outpost on strong squares
    KnightOutpost {
        square_index: u8,
        is_white: bool,
        strength: f32,
    },
    /// Bishop pair advantages in open positions
    BishopPair { is_white: bool, open_diagonals: u8 },
    /// Rook lift patterns for attack
    RookLift {
        from_rank_index: u8,
        to_rank_index: u8,
        is_white: bool,
    },
    /// Queen and knight coordination
    QueenKnightAttack {
        target_area: KingArea,
        is_white: bool,
    },
    /// Piece sacrifices for positional advantage
    PositionalSacrifice { piece_type: u8, compensation: f32 }, // 0=Pawn, 1=Knight, 2=Bishop, 3=Rook, 4=Queen, 5=King
    /// Piece activity vs opponent passivity
    ActivityAdvantage {
        is_white: bool,
        activity_differential: f32,
    },
}

/// King safety patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SafetyPattern {
    /// Castling structure integrity
    CastlingStructure {
        side: CastlingSide,
        is_white: bool,
        safety_value: f32,
    },
    /// Pawn shield configurations
    PawnShield {
        king_square_index: u8,
        shield_pattern: u8,
    },
    /// King exposure levels
    KingExposure {
        square_index: u8,
        is_white: bool,
        danger_level: f32,
    },
    /// Opposite-side castling attack patterns
    OppositeCastling {
        attacker_is_white: bool,
        attack_potential: f32,
    },
}

/// Initiative and tempo patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InitiativePattern {
    /// Space advantage in center or wings
    SpaceAdvantage {
        area: BoardArea,
        is_white: bool,
        space_value: f32,
    },
    /// Tempo advantages in development
    DevelopmentLead { is_white: bool, tempo_count: u8 },
    /// Pressure point creation
    PressurePoints {
        square_indices: Vec<u8>,
        is_white: bool,
    },
    /// Strategic pawn breaks
    PawnBreak {
        break_square_index: u8,
        is_white: bool,
        timing: f32,
    },
}

/// Endgame-specific patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EndgamePattern {
    /// Good vs bad bishop evaluations
    BishopEndgame { is_white: bool, bishop_quality: f32 },
    /// Rook endgame principles
    RookEndgame {
        pattern: RookPattern,
        advantage: f32,
    },
    /// King activity in endgames
    KingActivity {
        square_index: u8,
        is_white: bool,
        activity: f32,
    },
    /// Pawn endgame races
    PawnRace { is_white: bool, race_advantage: f32 },
}

/// Opening-specific strategic patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OpeningPattern {
    /// Central control strategies
    CentralControl {
        square_indices: Vec<u8>,
        is_white: bool,
        control_value: f32,
    },
    /// Development principles
    DevelopmentPrinciple {
        piece_type: u8,
        target_square_index: u8,
        value: f32,
    },
    /// Opening pawn breaks
    OpeningBreak {
        break_move: String,
        is_white: bool,
        strategic_value: f32,
    },
}

/// Context for when a motif is most relevant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategicContext {
    /// Game phase where this motif matters most
    pub game_phase: GamePhase,
    /// Material balance context
    pub material_context: MaterialContext,
    /// Minimum ply for relevance
    pub min_ply: u16,
    /// Maximum ply for relevance (None = always relevant)
    pub max_ply: Option<u16>,
}

/// Game phases for contextual relevance
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum GamePhase {
    Opening,
    Middlegame,
    Endgame,
    Any,
}

/// Material context for pattern relevance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MaterialContext {
    /// Equal material
    Equal,
    /// Material advantage for white (true) or black (false)
    Advantage(bool),
    /// Material disadvantage requires compensation for white (true) or black (false)
    Compensation(bool),
    /// Any material situation
    Any,
}

/// Supporting types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CastlingSide {
    Kingside,
    Queenside,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BoardArea {
    Center,
    Kingside,
    Queenside,
    Rank(u8), // rank index 0-7
    File(u8), // file index 0-7
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KingArea {
    Kingside,
    Queenside,
    Center,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RookPattern {
    ActiveRook,
    PassiveRook,
    RookBehindPasser,
    SeventhRank,
}

/// Reference to a master game where this motif appeared
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameReference {
    /// Game identifier (e.g., Lichess game ID)
    pub game_id: String,
    /// Player names
    pub white: String,
    pub black: String,
    /// Game result
    pub result: String,
    /// Ply where this motif was relevant
    pub ply: u16,
    /// Rating of players (for confidence weighting)
    pub rating: Option<u16>,
}

/// Match result when finding motifs in a position
#[derive(Debug, Clone)]
pub struct MotifMatch {
    /// The matched motif
    pub motif: StrategicMotif,
    /// Relevance score (0.0 to 1.0)
    pub relevance: f32,
    /// Specific squares involved in this match
    pub matching_squares: Vec<Square>,
}

/// Fast strategic database for real-time pattern matching
pub struct StrategicDatabase {
    /// All strategic motifs indexed by pattern hash
    motifs: HashMap<u64, StrategicMotif>,
    /// Pattern matchers for different motif types
    pawn_matcher: PawnPatternMatcher,
    piece_matcher: PiecePatternMatcher,
    king_matcher: KingPatternMatcher,
    /// Cache for recent evaluations
    evaluation_cache: lru::LruCache<u64, f32>,
    /// Statistics
    total_motifs: usize,
}

impl StrategicDatabase {
    /// Create new strategic database
    pub fn new() -> Self {
        Self {
            motifs: HashMap::new(),
            pawn_matcher: PawnPatternMatcher::new(),
            piece_matcher: PiecePatternMatcher::new(),
            king_matcher: KingPatternMatcher::new(),
            evaluation_cache: lru::LruCache::new(std::num::NonZeroUsize::new(10000).unwrap()),
            total_motifs: 0,
        }
    }

    /// Load strategic database from binary file (ultra-fast)
    pub fn load_from_binary<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        use std::fs::File;
        use std::io::BufReader;

        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let motifs: Vec<StrategicMotif> = bincode::deserialize_from(reader)?;

        let mut database = Self::new();
        for motif in motifs {
            database.add_motif(motif);
        }

        println!("📚 Loaded {} strategic motifs", database.total_motifs);
        Ok(database)
    }

    /// Save strategic database to binary file
    pub fn save_to_binary<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use std::fs::File;
        use std::io::BufWriter;

        let motifs: Vec<&StrategicMotif> = self.motifs.values().collect();
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, &motifs)?;

        Ok(())
    }

    /// Add a motif to the database
    pub fn add_motif(&mut self, motif: StrategicMotif) {
        // Index the motif for fast retrieval
        self.motifs.insert(motif.pattern_hash, motif.clone());

        // Add to specialized pattern matchers
        match &motif.motif_type {
            MotifType::PawnStructure(_) => self.pawn_matcher.add_pattern(&motif),
            MotifType::PieceCoordination(_) => self.piece_matcher.add_pattern(&motif),
            MotifType::KingSafety(_) => self.king_matcher.add_pattern(&motif),
            _ => {} // Other types use general pattern matching
        }

        self.total_motifs += 1;
    }

    /// Find strategic motifs in a position
    pub fn find_motifs(&mut self, board: &Board) -> Vec<MotifMatch> {
        let board_hash = board.get_hash();

        // Check cache first
        if let Some(&_cached_eval) = self.evaluation_cache.get(&board_hash) {
            // Return empty matches but signal cached evaluation available
            return vec![];
        }

        let mut matches = Vec::new();

        // Find pawn structure motifs
        matches.extend(self.pawn_matcher.find_matches(board));

        // Find piece coordination motifs
        matches.extend(self.piece_matcher.find_matches(board));

        // Find king safety motifs
        matches.extend(self.king_matcher.find_matches(board));

        // Cache the result
        let total_eval = self.evaluate_motifs(&matches);
        self.evaluation_cache.put(board_hash, total_eval);

        matches
    }

    /// Evaluate a collection of motif matches
    pub fn evaluate_motifs(&self, matches: &[MotifMatch]) -> f32 {
        if matches.is_empty() {
            return 0.0;
        }

        let weighted_sum: f32 = matches
            .iter()
            .map(|m| m.motif.evaluation * m.relevance * m.motif.confidence)
            .sum();

        let total_weight: f32 = matches
            .iter()
            .map(|m| m.relevance * m.motif.confidence)
            .sum();

        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        }
    }

    /// Get strategic evaluation for a position
    pub fn get_strategic_evaluation(&mut self, board: &Board) -> f32 {
        let matches = self.find_motifs(board);
        self.evaluate_motifs(&matches)
    }

    /// Get database statistics
    pub fn stats(&self) -> StrategicDatabaseStats {
        StrategicDatabaseStats {
            total_motifs: self.total_motifs,
            cache_size: self.evaluation_cache.len(),
            cache_hit_rate: 0.0, // TODO: implement cache hit tracking
        }
    }
}

/// Statistics for the strategic database
#[derive(Debug)]
pub struct StrategicDatabaseStats {
    pub total_motifs: usize,
    pub cache_size: usize,
    pub cache_hit_rate: f32,
}

/// Pattern matcher for pawn structures
pub struct PawnPatternMatcher {
    patterns: Vec<StrategicMotif>,
}

impl PawnPatternMatcher {
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
        }
    }

    pub fn add_pattern(&mut self, motif: &StrategicMotif) {
        self.patterns.push(motif.clone());
    }

    pub fn find_matches(&self, board: &Board) -> Vec<MotifMatch> {
        let mut matches = Vec::new();

        // Analyze pawn structure for each pattern
        for pattern in &self.patterns {
            if let Some(motif_match) = self.match_pawn_pattern(board, pattern) {
                matches.push(motif_match);
            }
        }

        matches
    }

    fn match_pawn_pattern(&self, _board: &Board, _motif: &StrategicMotif) -> Option<MotifMatch> {
        // Implementation will analyze specific pawn patterns
        // For now, return None (to be implemented)
        None
    }
}

/// Pattern matcher for piece coordination
pub struct PiecePatternMatcher {
    patterns: Vec<StrategicMotif>,
}

impl PiecePatternMatcher {
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
        }
    }

    pub fn add_pattern(&mut self, motif: &StrategicMotif) {
        self.patterns.push(motif.clone());
    }

    pub fn find_matches(&self, _board: &Board) -> Vec<MotifMatch> {
        // Implementation for piece pattern matching
        Vec::new()
    }
}

/// Pattern matcher for king safety
pub struct KingPatternMatcher {
    patterns: Vec<StrategicMotif>,
}

impl KingPatternMatcher {
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
        }
    }

    pub fn add_pattern(&mut self, motif: &StrategicMotif) {
        self.patterns.push(motif.clone());
    }

    pub fn find_matches(&self, _board: &Board) -> Vec<MotifMatch> {
        // Implementation for king safety pattern matching
        Vec::new()
    }
}

/// Utility functions for pattern analysis
pub mod pattern_utils {
    use super::*;

    /// Generate pattern hash for a chess position focusing on strategic elements
    pub fn generate_pattern_hash(board: &Board) -> u64 {
        // Use a hash that focuses on positional features rather than exact piece placement
        // This allows similar positions to match even with minor piece differences

        let mut hash = 0u64;

        // Hash pawn structure (most important for strategic patterns)
        hash ^= hash_pawn_structure(board);

        // Hash piece placement patterns
        hash ^= hash_piece_patterns(board);

        // Hash king positions
        hash ^= hash_king_positions(board);

        hash
    }

    fn hash_pawn_structure(_board: &Board) -> u64 {
        // Implementation for hashing pawn structure
        0
    }

    fn hash_piece_patterns(_board: &Board) -> u64 {
        // Implementation for hashing piece patterns
        0
    }

    fn hash_king_positions(_board: &Board) -> u64 {
        // Implementation for hashing king positions
        0
    }

    /// Determine game phase based on material
    pub fn determine_game_phase(board: &Board) -> GamePhase {
        let material_count = count_material(board);

        if material_count > 60 {
            GamePhase::Opening
        } else if material_count > 20 {
            GamePhase::Middlegame
        } else {
            GamePhase::Endgame
        }
    }

    fn count_material(board: &Board) -> u32 {
        // Simple material counting (can be improved)
        let mut total = 0u32;

        // Count all pieces except kings and pawns for phase determination
        for square in chess::ALL_SQUARES {
            if let Some(piece) = board.piece_on(square) {
                match piece {
                    Piece::Queen => total += 9,
                    Piece::Rook => total += 5,
                    Piece::Bishop => total += 3,
                    Piece::Knight => total += 3,
                    _ => {}
                }
            }
        }

        total
    }
}
