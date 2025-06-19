use chess::{Board, ChessMove, Color};
use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::str::FromStr;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};

use crate::{ChessVectorEngine, TacticalConfig, HybridConfig};

/// UCI (Universal Chess Interface) protocol implementation for the chess vector engine
pub struct UCIEngine {
    engine: ChessVectorEngine,
    board: Board,
    debug: bool,
    engine_name: String,
    engine_author: String,
    options: HashMap<String, UCIOption>,
    thinking: bool,
    stop_search: bool,
}

/// UCI option types
#[derive(Debug, Clone)]
pub enum UCIOption {
    Check { default: bool, value: bool },
    Spin { default: i32, min: i32, max: i32, value: i32 },
    Combo { default: String, options: Vec<String>, value: String },
    Button,
    String { default: String, value: String },
}

/// Search information for UCI info command
#[derive(Debug, Clone)]
pub struct SearchInfo {
    depth: u32,
    seldepth: Option<u32>,
    time: u64, // milliseconds
    nodes: u64,
    nps: u64, // nodes per second
    score: SearchScore,
    pv: Vec<ChessMove>, // principal variation
    currmove: Option<ChessMove>,
    currmovenumber: Option<u32>,
}

#[derive(Debug, Clone)]
pub enum SearchScore {
    Centipawns(i32),
    Mate(i32), // moves to mate (positive if we're winning)
}

impl UCIEngine {
    pub fn new() -> Self {
        let mut engine = ChessVectorEngine::new(1024);
        
        // Enable all advanced features for UCI
        engine.enable_opening_book();
        engine.enable_tactical_search_default();
        engine.configure_hybrid_evaluation(HybridConfig::default());
        
        let mut options = HashMap::new();
        
        // Standard UCI options
        options.insert("Hash".to_string(), UCIOption::Spin {
            default: 128,
            min: 1,
            max: 2048,
            value: 128,
        });
        
        options.insert("Threads".to_string(), UCIOption::Spin {
            default: 1,
            min: 1,
            max: 64,
            value: 1,
        });
        
        options.insert("MultiPV".to_string(), UCIOption::Spin {
            default: 1,
            min: 1,
            max: 10,
            value: 1,
        });
        
        // Chess Vector Engine specific options
        options.insert("Pattern_Weight".to_string(), UCIOption::Spin {
            default: 60,
            min: 0,
            max: 100,
            value: 60,
        });
        
        options.insert("Tactical_Depth".to_string(), UCIOption::Spin {
            default: 3,
            min: 1,
            max: 10,
            value: 3,
        });
        
        options.insert("Pattern_Confidence_Threshold".to_string(), UCIOption::Spin {
            default: 75,
            min: 0,
            max: 100,
            value: 75,
        });
        
        options.insert("Enable_LSH".to_string(), UCIOption::Check {
            default: true,
            value: true,
        });
        
        options.insert("Enable_GPU".to_string(), UCIOption::Check {
            default: true,
            value: true,
        });
        
        Self {
            engine,
            board: Board::default(),
            debug: false,
            engine_name: "Chess Vector Engine".to_string(),
            engine_author: "Chess Vector Engine Team".to_string(),
            options,
            thinking: false,
            stop_search: false,
        }
    }
    
    /// Main UCI loop
    pub fn run(&mut self) {
        let stdin = io::stdin();
        let mut stdout = io::stdout();
        
        for line in stdin.lock().lines() {
            match line {
                Ok(command) => {
                    let response = self.process_command(&command.trim());
                    if !response.is_empty() {
                        writeln!(stdout, "{}", response).unwrap();
                        stdout.flush().unwrap();
                    }
                    
                    if command.trim() == "quit" {
                        break;
                    }
                }
                Err(e) => {
                    if self.debug {
                        writeln!(stdout, "info string Error reading input: {}", e).unwrap();
                    }
                    break;
                }
            }
        }
    }
    
    /// Process UCI command and return response
    fn process_command(&mut self, command: &str) -> String {
        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts.is_empty() {
            return String::new();
        }
        
        match parts[0] {
            "uci" => self.handle_uci(),
            "debug" => self.handle_debug(&parts),
            "isready" => self.handle_isready(),
            "setoption" => self.handle_setoption(&parts),
            "register" => String::new(), // Not implemented
            "ucinewgame" => self.handle_ucinewgame(),
            "position" => self.handle_position(&parts),
            "go" => self.handle_go(&parts),
            "stop" => self.handle_stop(),
            "ponderhit" => String::new(), // Not implemented
            "quit" => String::new(),
            _ => {
                if self.debug {
                    format!("info string Unknown command: {}", command)
                } else {
                    String::new()
                }
            }
        }
    }
    
    fn handle_uci(&self) -> String {
        let mut response = String::new();
        response.push_str(&format!("id name {}\n", self.engine_name));
        response.push_str(&format!("id author {}\n", self.engine_author));
        
        // Send options
        for (name, option) in &self.options {
            match option {
                UCIOption::Check { default, .. } => {
                    response.push_str(&format!("option name {} type check default {}\n", name, default));
                }
                UCIOption::Spin { default, min, max, .. } => {
                    response.push_str(&format!("option name {} type spin default {} min {} max {}\n", name, default, min, max));
                }
                UCIOption::Combo { default, options, .. } => {
                    let combo_options = options.join(" var ");
                    response.push_str(&format!("option name {} type combo default {} var {}\n", name, default, combo_options));
                }
                UCIOption::Button => {
                    response.push_str(&format!("option name {} type button\n", name));
                }
                UCIOption::String { default, .. } => {
                    response.push_str(&format!("option name {} type string default {}\n", name, default));
                }
            }
        }
        
        response.push_str("uciok");
        response
    }
    
    fn handle_debug(&mut self, parts: &[&str]) -> String {
        if parts.len() >= 2 {
            match parts[1] {
                "on" => self.debug = true,
                "off" => self.debug = false,
                _ => {}
            }
        }
        String::new()
    }
    
    fn handle_isready(&self) -> String {
        "readyok".to_string()
    }
    
    fn handle_setoption(&mut self, parts: &[&str]) -> String {
        // Parse: setoption name <name> value <value>
        if parts.len() >= 4 && parts[1] == "name" {
            let mut name_parts = Vec::new();
            let mut value_parts = Vec::new();
            let mut in_value = false;
            
            for &part in &parts[2..] {
                if part == "value" {
                    in_value = true;
                } else if in_value {
                    value_parts.push(part);
                } else {
                    name_parts.push(part);
                }
            }
            
            let name = name_parts.join(" ");
            let value = value_parts.join(" ");
            
            self.set_option(&name, &value);
        }
        
        String::new()
    }
    
    fn set_option(&mut self, name: &str, value: &str) {
        if let Some(option) = self.options.get_mut(name) {
            match option {
                UCIOption::Check { value: ref mut val, .. } => {
                    *val = value == "true";
                }
                UCIOption::Spin { value: ref mut val, min, max, .. } => {
                    if let Ok(new_val) = value.parse::<i32>() {
                        if new_val >= *min && new_val <= *max {
                            *val = new_val;
                        }
                    }
                }
                UCIOption::Combo { value: ref mut val, options, .. } => {
                    if options.contains(&value.to_string()) {
                        *val = value.to_string();
                    }
                }
                UCIOption::String { value: ref mut val, .. } => {
                    *val = value.to_string();
                }
                UCIOption::Button => {
                    // Handle button press
                }
            }
        }
        
        // Apply engine-specific options
        self.apply_options();
    }
    
    fn apply_options(&mut self) {
        if let Some(UCIOption::Spin { value: pattern_weight, .. }) = self.options.get("Pattern_Weight") {
            let weight = (*pattern_weight as f32) / 100.0;
            let mut config = HybridConfig::default();
            config.pattern_weight = weight;
            self.engine.configure_hybrid_evaluation(config);
        }
        
        if let Some(UCIOption::Spin { value: depth, .. }) = self.options.get("Tactical_Depth") {
            let config = TacticalConfig {
                max_depth: *depth as u32,
                ..TacticalConfig::default()
            };
            self.engine.enable_tactical_search(config);
        }
        
        if let Some(UCIOption::Spin { value: threshold, .. }) = self.options.get("Pattern_Confidence_Threshold") {
            let mut config = HybridConfig::default();
            config.pattern_confidence_threshold = (*threshold as f32) / 100.0;
            self.engine.configure_hybrid_evaluation(config);
        }
        
        if let Some(UCIOption::Check { value: enable_lsh, .. }) = self.options.get("Enable_LSH") {
            if *enable_lsh && !self.engine.is_lsh_enabled() {
                self.engine.enable_lsh(8, 16);
            }
        }
    }
    
    fn handle_ucinewgame(&mut self) -> String {
        self.board = Board::default();
        // Could reset engine state here if needed
        String::new()
    }
    
    fn handle_position(&mut self, parts: &[&str]) -> String {
        if parts.len() < 2 {
            return String::new();
        }
        
        let mut board = if parts[1] == "startpos" {
            Board::default()
        } else if parts[1] == "fen" && parts.len() >= 8 {
            let fen = parts[2..8].join(" ");
            match Board::from_str(&fen) {
                Ok(board) => board,
                Err(_) => {
                    if self.debug {
                        return "info string Invalid FEN".to_string();
                    }
                    return String::new();
                }
            }
        } else {
            return String::new();
        };
        
        // Apply moves if present
        if let Some(moves_idx) = parts.iter().position(|&x| x == "moves") {
            for move_str in &parts[moves_idx + 1..] {
                if let Ok(chess_move) = ChessMove::from_str(move_str) {
                    if board.legal(chess_move) {
                        board = board.make_move_new(chess_move);
                    } else if self.debug {
                        return format!("info string Illegal move: {}", move_str);
                    }
                }
            }
        }
        
        self.board = board;
        String::new()
    }
    
    fn handle_go(&mut self, parts: &[&str]) -> String {
        if self.thinking {
            return String::new();
        }
        
        // Parse go command parameters
        let mut wtime = None;
        let mut btime = None;
        let mut winc = None;
        let mut binc = None;
        let mut movestogo = None;
        let mut depth = None;
        let mut nodes = None;
        let mut movetime = None;
        let mut infinite = false;
        
        let mut i = 1;
        while i < parts.len() {
            match parts[i] {
                "wtime" if i + 1 < parts.len() => {
                    wtime = parts[i + 1].parse().ok();
                    i += 2;
                }
                "btime" if i + 1 < parts.len() => {
                    btime = parts[i + 1].parse().ok();
                    i += 2;
                }
                "winc" if i + 1 < parts.len() => {
                    winc = parts[i + 1].parse().ok();
                    i += 2;
                }
                "binc" if i + 1 < parts.len() => {
                    binc = parts[i + 1].parse().ok();
                    i += 2;
                }
                "movestogo" if i + 1 < parts.len() => {
                    movestogo = parts[i + 1].parse().ok();
                    i += 2;
                }
                "depth" if i + 1 < parts.len() => {
                    depth = parts[i + 1].parse().ok();
                    i += 2;
                }
                "nodes" if i + 1 < parts.len() => {
                    nodes = parts[i + 1].parse().ok();
                    i += 2;
                }
                "movetime" if i + 1 < parts.len() => {
                    movetime = parts[i + 1].parse().ok();
                    i += 2;
                }
                "infinite" => {
                    infinite = true;
                    i += 1;
                }
                _ => i += 1,
            }
        }
        
        // Calculate search time
        let search_time = if let Some(mt) = movetime {
            Duration::from_millis(mt)
        } else if infinite {
            Duration::from_secs(3600) // 1 hour max
        } else {
            // Simple time management
            let our_time = if self.board.side_to_move() == Color::White {
                wtime.unwrap_or(30000)
            } else {
                btime.unwrap_or(30000)
            };
            
            let moves_left = movestogo.unwrap_or(30);
            let time_per_move = our_time / moves_left.max(1);
            Duration::from_millis(time_per_move.min(our_time / 2))
        };
        
        // Start search in separate thread
        self.start_search(search_time, depth);
        
        String::new()
    }
    
    fn start_search(&mut self, max_time: Duration, max_depth: Option<u32>) {
        self.thinking = true;
        self.stop_search = false;
        
        let board = self.board;
        let mut engine = self.engine.clone();
        let start_time = Instant::now();
        
        // Spawn search thread
        thread::spawn(move || {
            let mut search_info = SearchInfo {
                depth: 1,
                seldepth: None,
                time: 0,
                nodes: 0,
                nps: 0,
                score: SearchScore::Centipawns(0),
                pv: Vec::new(),
                currmove: None,
                currmovenumber: None,
            };
            
            // Get move recommendations from the engine
            let recommendations = engine.recommend_legal_moves(&board, 5);
            
            if let Some(best_move) = recommendations.first() {
                // Convert confidence to centipawns (rough approximation)
                let score_cp = ((best_move.confidence - 0.5) * 200.0) as i32;
                
                search_info.score = SearchScore::Centipawns(score_cp);
                search_info.pv = vec![best_move.chess_move];
                search_info.time = start_time.elapsed().as_millis() as u64;
                search_info.nodes = 1; // Simplified
                search_info.nps = if search_info.time > 0 {
                    (search_info.nodes * 1000) / search_info.time
                } else {
                    0
                };
                
                // Send info
                println!("info depth {} score cp {} time {} nodes {} nps {} pv {}", 
                    search_info.depth,
                    score_cp,
                    search_info.time,
                    search_info.nodes,
                    search_info.nps,
                    best_move.chess_move
                );
                
                // Send best move
                println!("bestmove {}", best_move.chess_move);
            } else {
                // No moves found, resign
                println!("bestmove 0000");
            }
        });
    }
    
    fn handle_stop(&mut self) -> String {
        self.stop_search = true;
        self.thinking = false;
        String::new()
    }
}

impl Default for UCIEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for the UCI engine
#[derive(Debug, Clone)]
pub struct UCIConfig {
    pub engine_name: String,
    pub engine_author: String,
    pub enable_debug: bool,
    pub default_hash_size: i32,
    pub default_threads: i32,
}

impl Default for UCIConfig {
    fn default() -> Self {
        Self {
            engine_name: "Chess Vector Engine".to_string(),
            engine_author: "Chess Vector Engine Team".to_string(),
            enable_debug: false,
            default_hash_size: 128,
            default_threads: 1,
        }
    }
}

/// Run the UCI engine with default configuration
pub fn run_uci_engine() {
    let mut engine = UCIEngine::new();
    engine.run();
}

/// Run the UCI engine with custom configuration
pub fn run_uci_engine_with_config(config: UCIConfig) {
    let mut engine = UCIEngine::new();
    engine.engine_name = config.engine_name;
    engine.engine_author = config.engine_author;
    engine.debug = config.enable_debug;
    
    // Apply configuration to options
    if let Some(UCIOption::Spin { value, .. }) = engine.options.get_mut("Hash") {
        *value = config.default_hash_size;
    }
    if let Some(UCIOption::Spin { value, .. }) = engine.options.get_mut("Threads") {
        *value = config.default_threads;
    }
    
    engine.run();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;
    
    #[test]
    fn test_uci_initialization() {
        let engine = UCIEngine::new();
        assert_eq!(engine.board, Board::default());
        assert!(!engine.debug);
        assert!(!engine.thinking);
    }
    
    #[test]
    fn test_uci_command() {
        let engine = UCIEngine::new();
        let response = engine.handle_uci();
        assert!(response.contains("id name"));
        assert!(response.contains("id author"));
        assert!(response.contains("uciok"));
    }
    
    #[test]
    fn test_isready_command() {
        let engine = UCIEngine::new();
        let response = engine.handle_isready();
        assert_eq!(response, "readyok");
    }
    
    #[test]
    fn test_position_startpos() {
        let mut engine = UCIEngine::new();
        let parts = vec!["position", "startpos"];
        engine.handle_position(&parts);
        assert_eq!(engine.board, Board::default());
    }
    
    #[test]
    fn test_position_with_moves() {
        let mut engine = UCIEngine::new();
        let parts = vec!["position", "startpos", "moves", "e2e4", "e7e5"];
        engine.handle_position(&parts);
        
        let expected_board = Board::default()
            .make_move_new(ChessMove::from_str("e2e4").unwrap())
            .make_move_new(ChessMove::from_str("e7e5").unwrap());
        
        assert_eq!(engine.board, expected_board);
    }
    
    #[test]
    fn test_option_setting() {
        let mut engine = UCIEngine::new();
        engine.set_option("Pattern_Weight", "80");
        
        if let Some(UCIOption::Spin { value, .. }) = engine.options.get("Pattern_Weight") {
            assert_eq!(*value, 80);
        } else {
            panic!("Option not found or wrong type");
        }
    }
    
    #[test]
    fn test_debug_toggle() {
        let mut engine = UCIEngine::new();
        
        engine.handle_debug(&["debug", "on"]);
        assert!(engine.debug);
        
        engine.handle_debug(&["debug", "off"]);
        assert!(!engine.debug);
    }
}