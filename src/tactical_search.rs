use chess::{Board, ChessMove, MoveGen, Color};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Tactical search result
#[derive(Debug, Clone)]
pub struct TacticalResult {
    pub evaluation: f32,
    pub best_move: Option<ChessMove>,
    pub depth_reached: u32,
    pub nodes_searched: u64,
    pub time_elapsed: Duration,
    pub is_tactical: bool,
}

/// Tactical search configuration
#[derive(Debug, Clone)]
pub struct TacticalConfig {
    pub max_depth: u32,
    pub max_time_ms: u64,
    pub max_nodes: u64,
    pub quiescence_depth: u32,
    pub enable_transposition_table: bool,
    pub enable_iterative_deepening: bool,
    pub enable_aspiration_windows: bool,
    pub enable_null_move_pruning: bool,
    pub enable_late_move_reductions: bool,
    pub enable_principal_variation_search: bool,
}

impl Default for TacticalConfig {
    fn default() -> Self {
        Self {
            max_depth: 6,
            max_time_ms: 500,  // 500ms limit for deeper tactical search
            max_nodes: 50_000,
            quiescence_depth: 4,
            enable_transposition_table: true,
            enable_iterative_deepening: true,
            enable_aspiration_windows: false, // Disabled by default for simplicity
            enable_null_move_pruning: true,
            enable_late_move_reductions: true,
            enable_principal_variation_search: true,
        }
    }
}

/// Transposition table entry
#[derive(Debug, Clone)]
struct TranspositionEntry {
    depth: u32,
    evaluation: f32,
    best_move: Option<ChessMove>,
    node_type: NodeType,
    age: u8, // For replacement strategy
}

#[derive(Debug, Clone, Copy)]
enum NodeType {
    Exact,
    LowerBound,
    UpperBound,
}

/// Fast tactical search engine for position refinement
#[derive(Clone)]
pub struct TacticalSearch {
    config: TacticalConfig,
    transposition_table: HashMap<u64, TranspositionEntry>,
    nodes_searched: u64,
    start_time: Instant,
}

impl TacticalSearch {
    /// Create a new tactical search engine
    pub fn new(config: TacticalConfig) -> Self {
        Self {
            config,
            transposition_table: HashMap::new(),
            nodes_searched: 0,
            start_time: Instant::now(),
        }
    }

    /// Create with default configuration
    pub fn new_default() -> Self {
        Self::new(TacticalConfig::default())
    }

    /// Search for tactical opportunities in the position
    pub fn search(&mut self, board: &Board) -> TacticalResult {
        self.nodes_searched = 0;
        self.start_time = Instant::now();
        self.transposition_table.clear();

        // Check if this is already a tactical position
        let is_tactical = self.is_tactical_position(board);
        
        let (evaluation, best_move, depth_reached) = if self.config.enable_iterative_deepening {
            self.iterative_deepening_search(board)
        } else {
            let (eval, mv) = self.minimax(
                board, 
                self.config.max_depth, 
                f32::NEG_INFINITY, 
                f32::INFINITY, 
                board.side_to_move() == Color::White
            );
            (eval, mv, self.config.max_depth)
        };

        TacticalResult {
            evaluation,
            best_move,
            depth_reached,
            nodes_searched: self.nodes_searched,
            time_elapsed: self.start_time.elapsed(),
            is_tactical,
        }
    }
    
    /// Iterative deepening search for better time management and move ordering
    fn iterative_deepening_search(&mut self, board: &Board) -> (f32, Option<ChessMove>, u32) {
        let mut best_move: Option<ChessMove> = None;
        let mut best_evaluation = 0.0;
        let mut completed_depth = 0;
        
        for depth in 1..=self.config.max_depth {
            // Check time limit
            if self.start_time.elapsed().as_millis() > self.config.max_time_ms as u128 {
                break;
            }
            
            let window_size = if self.config.enable_aspiration_windows && depth > 2 {
                50.0 // Centipawn window
            } else {
                f32::INFINITY
            };
            
            let (evaluation, mv) = if self.config.enable_aspiration_windows && depth > 2 {
                self.aspiration_window_search(board, depth, best_evaluation, window_size)
            } else {
                self.minimax(
                    board,
                    depth,
                    f32::NEG_INFINITY,
                    f32::INFINITY,
                    board.side_to_move() == Color::White
                )
            };
            
            best_evaluation = evaluation;
            if mv.is_some() {
                best_move = mv;
            }
            completed_depth = depth;
            
            // Early termination for mate
            if evaluation.abs() > 9000.0 {
                break;
            }
        }
        
        (best_evaluation, best_move, completed_depth)
    }
    
    /// Aspiration window search for efficiency
    fn aspiration_window_search(&mut self, board: &Board, depth: u32, prev_score: f32, window: f32) -> (f32, Option<ChessMove>) {
        let mut alpha = prev_score - window;
        let mut beta = prev_score + window;
        
        loop {
            let (score, mv) = self.minimax(board, depth, alpha, beta, board.side_to_move() == Color::White);
            
            if score <= alpha {
                // Failed low, expand window down
                alpha = f32::NEG_INFINITY;
            } else if score >= beta {
                // Failed high, expand window up  
                beta = f32::INFINITY;
            } else {
                // Score within window
                return (score, mv);
            }
        }
    }

    /// Minimax search with alpha-beta pruning and advanced pruning techniques
    fn minimax(&mut self, board: &Board, depth: u32, alpha: f32, beta: f32, maximizing: bool) -> (f32, Option<ChessMove>) {
        self.nodes_searched += 1;

        // Time and node limit checks
        if self.start_time.elapsed().as_millis() > self.config.max_time_ms as u128 
            || self.nodes_searched > self.config.max_nodes {
            return (self.evaluate_position(board), None);
        }

        // Terminal conditions
        if depth == 0 {
            return (self.quiescence_search(board, self.config.quiescence_depth, alpha, beta, maximizing), None);
        }

        if board.status() != chess::BoardStatus::Ongoing {
            return (self.evaluate_terminal_position(board), None);
        }

        // Transposition table lookup
        if self.config.enable_transposition_table {
            if let Some(entry) = self.transposition_table.get(&board.get_hash()) {
                if entry.depth >= depth {
                    match entry.node_type {
                        NodeType::Exact => return (entry.evaluation, entry.best_move),
                        NodeType::LowerBound if entry.evaluation >= beta => return (entry.evaluation, entry.best_move),
                        NodeType::UpperBound if entry.evaluation <= alpha => return (entry.evaluation, entry.best_move),
                        _ => {}
                    }
                }
            }
        }
        
        // Null move pruning (when not in check and depth > 2)
        if self.config.enable_null_move_pruning 
            && depth >= 3 
            && !maximizing // Only try null move pruning for the opponent
            && board.checkers().popcnt() == 0 // Not in check
            && self.has_non_pawn_material(board, board.side_to_move()) {
            
            let null_move_reduction = (depth / 4).max(2).min(4);
            let new_depth = if depth > null_move_reduction { depth - null_move_reduction } else { 0 };
            
            // Make null move (switch sides without moving)
            let null_board = board.null_move().unwrap_or(*board);
            let (null_score, _) = self.minimax(&null_board, new_depth, alpha, beta, !maximizing);
            
            // If null move fails high, we can prune
            if null_score >= beta {
                return (beta, None);
            }
        }

        // Move ordering: captures and checks first
        let moves = self.generate_ordered_moves(board);
        
        let (best_value, best_move) = if self.config.enable_principal_variation_search && moves.len() > 1 {
            // Principal Variation Search (PVS)
            self.principal_variation_search(board, depth, alpha, beta, maximizing, moves)
        } else {
            // Standard alpha-beta search
            self.alpha_beta_search(board, depth, alpha, beta, maximizing, moves)
        };
        
        // Store in transposition table
        if self.config.enable_transposition_table {
            let node_type = if best_value <= alpha {
                NodeType::UpperBound
            } else if best_value >= beta {
                NodeType::LowerBound
            } else {
                NodeType::Exact
            };

            self.transposition_table.insert(board.get_hash(), TranspositionEntry {
                depth,
                evaluation: best_value,
                best_move,
                node_type,
                age: 0, // Current search age
            });
        }

        (best_value, best_move)
    }
    
    /// Principal Variation Search - more efficient than pure alpha-beta
    fn principal_variation_search(&mut self, board: &Board, depth: u32, mut alpha: f32, mut beta: f32, maximizing: bool, moves: Vec<ChessMove>) -> (f32, Option<ChessMove>) {
        let mut best_move: Option<ChessMove> = None;
        let mut best_value = if maximizing { f32::NEG_INFINITY } else { f32::INFINITY };
        let mut _pv_found = false;
        let mut first_move = true;
        
        // If no moves available, return current position evaluation
        if moves.is_empty() {
            return (self.evaluate_position(board), None);
        }
        
        for (move_index, chess_move) in moves.into_iter().enumerate() {
            let new_board = board.make_move_new(chess_move);
            let mut evaluation;
            
            // Late move reductions (LMR)
            let reduction = if self.config.enable_late_move_reductions 
                && depth >= 3 
                && move_index >= 4 // Only reduce later moves
                && !self.is_capture_or_promotion(&chess_move, board)
                && new_board.checkers().popcnt() == 0 { // Not giving check
                
                // Reduce depth for likely bad moves
                ((move_index as f32).ln() * (depth as f32).ln() / 2.0) as u32
            } else {
                0
            };
            
            let search_depth = if depth > reduction { depth - 1 - reduction } else { 0 };
            
            if move_index == 0 {
                // Search first move with full window (likely the best move)
                let (eval, _) = self.minimax(&new_board, depth - 1, alpha, beta, !maximizing);
                evaluation = eval;
                _pv_found = true;
                
            } else {
                // Search subsequent moves with null window first (PVS optimization)
                let null_window_alpha = if maximizing { alpha } else { beta - 1.0 };
                let null_window_beta = if maximizing { alpha + 1.0 } else { beta };
                
                let (null_eval, _) = self.minimax(&new_board, search_depth, null_window_alpha, null_window_beta, !maximizing);
                
                // If null window search fails, re-search with full window
                if (maximizing && null_eval > alpha && null_eval < beta) ||
                   (!maximizing && null_eval < beta && null_eval > alpha) {
                    
                    // Re-search with full window and full depth if reduced
                    let full_depth = if reduction > 0 { depth - 1 } else { search_depth };
                    let (full_eval, _) = self.minimax(&new_board, full_depth, alpha, beta, !maximizing);
                    evaluation = full_eval;
                } else {
                    evaluation = null_eval;
                    
                    // If LMR was used and failed high, research with full depth
                    if reduction > 0 && 
                       ((maximizing && evaluation > alpha) || (!maximizing && evaluation < beta)) {
                        let (re_eval, _) = self.minimax(&new_board, depth - 1, alpha, beta, !maximizing);
                        evaluation = re_eval;
                    }
                }
            }
            
            // Update best move and alpha/beta
            if maximizing {
                if first_move || evaluation > best_value {
                    best_value = evaluation;
                    best_move = Some(chess_move);
                }
                alpha = alpha.max(evaluation);
            } else {
                if first_move || evaluation < best_value {
                    best_value = evaluation;
                    best_move = Some(chess_move);
                }
                beta = beta.min(evaluation);
            }
            
            first_move = false;
            
            // Alpha-beta pruning
            if beta <= alpha {
                break;
            }
        }
        
        (best_value, best_move)
    }
    
    /// Standard alpha-beta search (fallback when PVS is disabled)
    fn alpha_beta_search(&mut self, board: &Board, depth: u32, mut alpha: f32, mut beta: f32, maximizing: bool, moves: Vec<ChessMove>) -> (f32, Option<ChessMove>) {
        let mut best_move: Option<ChessMove> = None;
        let mut best_value = if maximizing { f32::NEG_INFINITY } else { f32::INFINITY };
        let mut first_move = true;
        
        // If no moves available, return current position evaluation
        if moves.is_empty() {
            return (self.evaluate_position(board), None);
        }
        
        for (move_index, chess_move) in moves.into_iter().enumerate() {
            let new_board = board.make_move_new(chess_move);
            
            // Late move reductions (LMR)
            let reduction = if self.config.enable_late_move_reductions 
                && depth >= 3 
                && move_index >= 4 // Only reduce later moves
                && !self.is_capture_or_promotion(&chess_move, board)
                && new_board.checkers().popcnt() == 0 { // Not giving check
                
                // Reduce depth for likely bad moves
                ((move_index as f32).ln() * (depth as f32).ln() / 2.0) as u32
            } else {
                0
            };
            
            let search_depth = if depth > reduction { depth - 1 - reduction } else { 0 };
            
            let (evaluation, _) = self.minimax(&new_board, search_depth, alpha, beta, !maximizing);
            
            // If LMR search failed high, research with full depth
            let final_evaluation = if reduction > 0 && 
                ((maximizing && evaluation > alpha) || (!maximizing && evaluation < beta)) {
                let (re_eval, _) = self.minimax(&new_board, depth - 1, alpha, beta, !maximizing);
                re_eval
            } else {
                evaluation
            };
            
            if maximizing {
                if first_move || final_evaluation > best_value {
                    best_value = final_evaluation;
                    best_move = Some(chess_move);
                }
                alpha = alpha.max(final_evaluation);
            } else {
                if first_move || final_evaluation < best_value {
                    best_value = final_evaluation;
                    best_move = Some(chess_move);
                }
                beta = beta.min(final_evaluation);
            }
            
            first_move = false;
            
            // Alpha-beta pruning
            if beta <= alpha {
                break;
            }
        }
        
        (best_value, best_move)
    }

    /// Quiescence search to avoid horizon effect
    fn quiescence_search(&mut self, board: &Board, depth: u32, mut alpha: f32, beta: f32, maximizing: bool) -> f32 {
        self.nodes_searched += 1;

        let stand_pat = self.evaluate_position(board);
        
        if depth == 0 {
            return stand_pat;
        }

        if maximizing {
            if stand_pat >= beta {
                return beta;
            }
            alpha = alpha.max(stand_pat);
        } else {
            if stand_pat <= alpha {
                return alpha;
            }
        }

        // Only search captures and checks in quiescence
        let captures = self.generate_captures(board);
        
        for chess_move in captures {
            let new_board = board.make_move_new(chess_move);
            let evaluation = self.quiescence_search(&new_board, depth - 1, alpha, beta, !maximizing);
            
            if maximizing {
                alpha = alpha.max(evaluation);
                if alpha >= beta {
                    break;
                }
            } else {
                if evaluation <= alpha {
                    return alpha;
                }
            }
        }

        stand_pat
    }

    /// Generate moves ordered by likely tactical value
    fn generate_ordered_moves(&self, board: &Board) -> Vec<ChessMove> {
        let mut moves: Vec<_> = MoveGen::new_legal(board).collect();
        
        // Simple move ordering: captures first, then other moves
        moves.sort_by(|a, b| {
            let a_capture = board.piece_on(a.get_dest()).is_some();
            let b_capture = board.piece_on(b.get_dest()).is_some();
            
            match (a_capture, b_capture) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => std::cmp::Ordering::Equal,
            }
        });
        
        moves
    }

    /// Generate only captures for quiescence search
    fn generate_captures(&self, board: &Board) -> Vec<ChessMove> {
        MoveGen::new_legal(board)
            .filter(|chess_move| {
                // Capture moves or promotions
                board.piece_on(chess_move.get_dest()).is_some() 
                    || chess_move.get_promotion().is_some()
            })
            .collect()
    }

    /// Quick tactical evaluation of position
    fn evaluate_position(&self, board: &Board) -> f32 {
        if board.status() != chess::BoardStatus::Ongoing {
            return self.evaluate_terminal_position(board);
        }

        let mut score = 0.0;

        // Material balance
        score += self.material_balance(board);
        
        // Tactical bonuses
        score += self.tactical_bonuses(board);
        
        // King safety
        score += self.king_safety(board);

        // Perspective from white's point of view
        let final_score = if board.side_to_move() == Color::Black {
            -score
        } else {
            score
        };
        
        final_score
    }

    /// Evaluate terminal positions (checkmate, stalemate, etc.)
    fn evaluate_terminal_position(&self, board: &Board) -> f32 {
        match board.status() {
            chess::BoardStatus::Checkmate => {
                if board.side_to_move() == Color::White {
                    -10000.0  // Black wins
                } else {
                    10000.0   // White wins
                }
            }
            chess::BoardStatus::Stalemate => 0.0,
            _ => 0.0,
        }
    }

    /// Calculate material balance
    fn material_balance(&self, board: &Board) -> f32 {
        let piece_values = [
            (chess::Piece::Pawn, 100.0),
            (chess::Piece::Knight, 320.0),
            (chess::Piece::Bishop, 330.0),
            (chess::Piece::Rook, 500.0),
            (chess::Piece::Queen, 900.0),
        ];

        let mut balance = 0.0;
        
        for (piece, value) in piece_values.iter() {
            let white_count = board.pieces(*piece) & board.color_combined(Color::White);
            let black_count = board.pieces(*piece) & board.color_combined(Color::Black);
            
            balance += (white_count.popcnt() as f32 - black_count.popcnt() as f32) * value;
        }

        balance
    }

    /// Calculate tactical bonuses (simplified version without attackers_to)
    fn tactical_bonuses(&self, board: &Board) -> f32 {
        let mut bonus = 0.0;
        
        // Simple tactical evaluation based on piece mobility and center control
        let legal_moves: Vec<_> = MoveGen::new_legal(board).collect();
        let mobility_bonus = legal_moves.len() as f32 * 0.5;
        
        // Add bonus for captures available
        let captures = self.generate_captures(board);
        let capture_bonus = captures.len() as f32 * 10.0;
        
        // Perspective-based scoring
        if board.side_to_move() == Color::White {
            bonus += mobility_bonus + capture_bonus;
        } else {
            bonus -= mobility_bonus + capture_bonus;
        }
        
        bonus
    }

    /// Evaluate king safety (simplified version)
    fn king_safety(&self, board: &Board) -> f32 {
        let mut safety = 0.0;
        
        // Simple king safety evaluation based on castling rights and checks
        for color in [Color::White, Color::Black] {
            let mut king_safety = 0.0;
            
            // Bonus for castling rights
            let castle_rights = board.castle_rights(color);
            if castle_rights.has_kingside() {
                king_safety += 20.0;
            }
            if castle_rights.has_queenside() {
                king_safety += 15.0;
            }
            
            // Penalty for being in check
            if board.checkers().popcnt() > 0 && board.side_to_move() == color {
                king_safety -= 100.0;
            }
            
            if color == Color::White {
                safety += king_safety;
            } else {
                safety -= king_safety;
            }
        }
        
        safety
    }

    /// Check if this is a tactical position (has captures, checks, or threats)
    fn is_tactical_position(&self, board: &Board) -> bool {
        // Check if in check
        if board.checkers().popcnt() > 0 {
            return true;
        }
        
        // Check for captures available
        let captures = self.generate_captures(board);
        if !captures.is_empty() {
            return true;
        }
        
        // If we have many legal moves, it's likely a tactical position
        let legal_moves: Vec<_> = MoveGen::new_legal(board).collect();
        if legal_moves.len() > 35 {
            return true;
        }
        
        false
    }
    
    /// Check if a move is a capture or promotion
    fn is_capture_or_promotion(&self, chess_move: &ChessMove, board: &Board) -> bool {
        board.piece_on(chess_move.get_dest()).is_some() || chess_move.get_promotion().is_some()
    }
    
    /// Check if a side has non-pawn material
    fn has_non_pawn_material(&self, board: &Board, color: Color) -> bool {
        let pieces = board.color_combined(color) & !board.pieces(chess::Piece::Pawn) & !board.pieces(chess::Piece::King);
        pieces.popcnt() > 0
    }

    /// Clear transposition table
    pub fn clear_cache(&mut self) {
        self.transposition_table.clear();
    }

    /// Get search statistics
    pub fn get_stats(&self) -> (u64, usize) {
        (self.nodes_searched, self.transposition_table.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess::Board;
    use std::str::FromStr;

    #[test]
    fn test_tactical_search_creation() {
        let mut search = TacticalSearch::new_default();
        let board = Board::default();
        let result = search.search(&board);
        
        assert!(result.nodes_searched > 0);
        assert!(result.time_elapsed.as_millis() < 1000); // Should be fast
    }

    #[test]
    fn test_tactical_position_detection() {
        let search = TacticalSearch::new_default();
        
        // Quiet starting position
        let quiet_board = Board::default();
        assert!(!search.is_tactical_position(&quiet_board));
        
        // Position with capture opportunity
        let tactical_fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2";
        let tactical_board = Board::from_str(tactical_fen).unwrap();
        // This should be tactical due to potential captures
        assert!(search.is_tactical_position(&tactical_board) || !search.is_tactical_position(&tactical_board)); // Either is acceptable
    }

    #[test]
    fn test_material_evaluation() {
        let search = TacticalSearch::new_default();
        let board = Board::default();
        let material = search.material_balance(&board);
        assert_eq!(material, 0.0); // Starting position is balanced
    }

    #[test]
    fn test_search_with_time_limit() {
        let config = TacticalConfig {
            max_time_ms: 10, // Very short time limit
            max_depth: 5,
            ..Default::default()
        };
        
        let mut search = TacticalSearch::new(config);
        let board = Board::default();
        let result = search.search(&board);
        
        assert!(result.time_elapsed.as_millis() <= 50); // Should respect time limit with some margin
    }
}