/// Controlled Stockfish Testing Framework
/// 
/// SOLID, clean implementation for testing against Stockfish with controlled conditions.
/// Tests our calibrated engine against Stockfish at various reduced strengths.

use chess::{Board, ChessMove, Color};
use std::process::{Command, Stdio};
use std::io::{BufRead, BufReader, Write};
use std::str::FromStr;
use std::time::{Duration, Instant};
use crate::{ChessVectorEngine, CalibratedEvaluator, CalibrationConfig};

/// Configuration for Stockfish testing (Single Responsibility Principle)
#[derive(Debug, Clone)]
pub struct StockfishTestConfig {
    pub stockfish_path: String,
    pub depth_limit: Option<u8>,        // Limit Stockfish search depth
    pub time_limit_ms: Option<u64>,     // Limit Stockfish thinking time
    pub skill_level: Option<u8>,        // Stockfish skill level (0-20, lower = weaker)
    pub num_threads: Option<u8>,        // Limit Stockfish threads
    pub hash_size_mb: Option<u32>,      // Limit Stockfish hash table
}

impl Default for StockfishTestConfig {
    fn default() -> Self {
        Self {
            stockfish_path: "stockfish".to_string(), // Assume stockfish in PATH
            depth_limit: Some(6),     // Match our engine's typical depth
            time_limit_ms: Some(1000), // 1 second per move
            skill_level: Some(10),    // Mid-level skill (0-20 range)
            num_threads: Some(1),     // Single threaded for fairness
            hash_size_mb: Some(64),   // Limited hash
        }
    }
}

/// Result of a single position evaluation comparison
#[derive(Debug, Clone)]
pub struct EvaluationComparison {
    pub fen: String,
    pub our_evaluation_cp: i32,
    pub stockfish_evaluation_cp: i32,
    pub our_best_move: Option<ChessMove>,
    pub stockfish_best_move: Option<ChessMove>,
    pub evaluation_diff_cp: i32,
    pub move_agreement: bool,
    pub evaluation_category: EvaluationCategory,
}

/// Categories for evaluation comparison analysis
#[derive(Debug, Clone, PartialEq)]
pub enum EvaluationCategory {
    ExactMatch,      // Within ±25cp
    CloseMatch,      // Within ±50cp  
    ReasonableMatch, // Within ±100cp
    SignMatch,       // Same sign (both positive or both negative)
    Mismatch,        // Different signs or >100cp difference
}

impl EvaluationCategory {
    fn from_difference(diff_cp: i32) -> Self {
        match diff_cp.abs() {
            0..=25 => Self::ExactMatch,
            26..=50 => Self::CloseMatch,
            51..=100 => Self::ReasonableMatch,
            _ => {
                // For large differences, we need evaluation context to check signs
                // This will be determined by the caller with actual evaluation values
                Self::SignMatch // Default classification for large differences
            }
        }
    }
}

/// UCI communication interface (Interface Segregation Principle)
trait UCIEngine {
    fn send_command(&mut self, command: &str) -> Result<(), StockfishTestError>;
    fn read_response(&mut self) -> Result<String, StockfishTestError>;
    fn evaluate_position(&mut self, fen: &str) -> Result<(i32, Option<ChessMove>), StockfishTestError>;
    fn close(&mut self) -> Result<(), StockfishTestError>;
}

/// Stockfish UCI engine wrapper (Single Responsibility Principle)
pub struct StockfishEngine {
    process: std::process::Child,
    stdin: std::process::ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
    config: StockfishTestConfig,
}

impl StockfishEngine {
    pub fn new(config: StockfishTestConfig) -> Result<Self, StockfishTestError> {
        let mut process = Command::new(&config.stockfish_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| StockfishTestError::LaunchError(format!("Failed to start Stockfish: {}", e)))?;

        let stdin = process.stdin.take()
            .ok_or_else(|| StockfishTestError::LaunchError("Failed to get stdin".to_string()))?;
        
        let stdout = BufReader::new(process.stdout.take()
            .ok_or_else(|| StockfishTestError::LaunchError("Failed to get stdout".to_string()))?);

        let mut engine = Self {
            process,
            stdin,
            stdout,
            config,
        };

        // Initialize Stockfish with UCI protocol
        engine.initialize()?;

        Ok(engine)
    }

    fn initialize(&mut self) -> Result<(), StockfishTestError> {
        // Send UCI command
        self.send_command("uci")?;
        
        // Wait for uciok
        loop {
            let response = self.read_response()?;
            if response.contains("uciok") {
                break;
            }
        }

        // Configure Stockfish based on test config
        if let Some(skill_level) = self.config.skill_level {
            self.send_command(&format!("setoption name Skill Level value {}", skill_level))?;
        }

        if let Some(threads) = self.config.num_threads {
            self.send_command(&format!("setoption name Threads value {}", threads))?;
        }

        if let Some(hash_size) = self.config.hash_size_mb {
            self.send_command(&format!("setoption name Hash value {}", hash_size))?;
        }

        // Disable pondering for deterministic behavior
        self.send_command("setoption name Ponder value false")?;

        self.send_command("isready")?;
        
        // Wait for readyok
        loop {
            let response = self.read_response()?;
            if response.contains("readyok") {
                break;
            }
        }

        Ok(())
    }
}

impl UCIEngine for StockfishEngine {
    fn send_command(&mut self, command: &str) -> Result<(), StockfishTestError> {
        writeln!(self.stdin, "{}", command)
            .map_err(|e| StockfishTestError::CommunicationError(format!("Send failed: {}", e)))?;
        self.stdin.flush()
            .map_err(|e| StockfishTestError::CommunicationError(format!("Flush failed: {}", e)))?;
        Ok(())
    }

    fn read_response(&mut self) -> Result<String, StockfishTestError> {
        let mut line = String::new();
        self.stdout.read_line(&mut line)
            .map_err(|e| StockfishTestError::CommunicationError(format!("Read failed: {}", e)))?;
        Ok(line.trim().to_string())
    }

    fn evaluate_position(&mut self, fen: &str) -> Result<(i32, Option<ChessMove>), StockfishTestError> {
        // Set position
        self.send_command(&format!("position fen {}", fen))?;

        // Start search with configured limits
        let mut go_command = "go".to_string();
        
        if let Some(depth) = self.config.depth_limit {
            go_command.push_str(&format!(" depth {}", depth));
        }
        
        if let Some(time_ms) = self.config.time_limit_ms {
            go_command.push_str(&format!(" movetime {}", time_ms));
        }

        self.send_command(&go_command)?;

        let mut best_move = None;
        let mut evaluation_cp = 0;

        // Read search results
        loop {
            let response = self.read_response()?;
            
            if response.starts_with("info") {
                // Parse evaluation from info line
                if let Some(cp_pos) = response.find(" cp ") {
                    if let Ok(cp) = response[cp_pos + 4..].split_whitespace().next()
                        .unwrap_or("0").parse::<i32>() {
                        evaluation_cp = cp;
                    }
                }
            } else if response.starts_with("bestmove") {
                // Parse best move
                let parts: Vec<&str> = response.split_whitespace().collect();
                if parts.len() >= 2 && parts[1] != "(none)" {
                    // For now, we'll parse the move string manually since ChessMove::from_str 
                    // may not handle all UCI move formats. This is a TODO for full implementation.
                    if parts[1].len() >= 4 {
                        // Basic move validation - just check it's a reasonable move string
                        let move_str = parts[1];
                        if move_str.chars().all(|c| c.is_ascii_alphanumeric()) {
                            // TODO: Implement proper ChessMove parsing from UCI format
                            // For now, we'll skip move comparison and focus on evaluation
                            // best_move = Some(parsed_move); // TODO: Complete move parsing
                        }
                    }
                }
                break;
            }
        }

        Ok((evaluation_cp, best_move))
    }

    fn close(&mut self) -> Result<(), StockfishTestError> {
        self.send_command("quit")?;
        self.process.wait()
            .map_err(|e| StockfishTestError::LaunchError(format!("Failed to close: {}", e)))?;
        Ok(())
    }
}

/// Error types for Stockfish testing (Single Responsibility Principle)
#[derive(Debug)]
pub enum StockfishTestError {
    LaunchError(String),
    CommunicationError(String),
    ParseError(String),
    TimeoutError(String),
}

impl std::fmt::Display for StockfishTestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LaunchError(msg) => write!(f, "Launch error: {}", msg),
            Self::CommunicationError(msg) => write!(f, "Communication error: {}", msg),
            Self::ParseError(msg) => write!(f, "Parse error: {}", msg),
            Self::TimeoutError(msg) => write!(f, "Timeout error: {}", msg),
        }
    }
}

impl std::error::Error for StockfishTestError {}

/// Main testing framework (Open/Closed Principle - extensible for different test types)
pub struct StockfishTester {
    our_engine: ChessVectorEngine,
    calibrated_evaluator: CalibratedEvaluator,
    stockfish_config: StockfishTestConfig,
}

impl StockfishTester {
    pub fn new(stockfish_config: StockfishTestConfig) -> Self {
        let mut our_engine = ChessVectorEngine::new(1024);
        our_engine.enable_opening_book();
        
        let calibration_config = CalibrationConfig::default();
        let calibrated_evaluator = CalibratedEvaluator::new(calibration_config);
        
        Self {
            our_engine,
            calibrated_evaluator,
            stockfish_config,
        }
    }

    /// Classify evaluation agreement based on actual values
    fn classify_evaluation_match(our_eval_cp: i32, stockfish_eval_cp: i32) -> EvaluationCategory {
        let diff = (our_eval_cp - stockfish_eval_cp).abs();
        
        match diff {
            0..=25 => EvaluationCategory::ExactMatch,
            26..=50 => EvaluationCategory::CloseMatch, 
            51..=100 => EvaluationCategory::ReasonableMatch,
            _ => {
                // Check if signs match for large differences
                let our_sign = our_eval_cp > 0;
                let stockfish_sign = stockfish_eval_cp > 0;
                
                if our_sign == stockfish_sign || (our_eval_cp.abs() <= 50 && stockfish_eval_cp.abs() <= 50) {
                    EvaluationCategory::SignMatch
                } else {
                    EvaluationCategory::Mismatch
                }
            }
        }
    }

    /// Test a single position against Stockfish
    pub fn test_position(&mut self, fen: &str) -> Result<EvaluationComparison, StockfishTestError> {
        let board = Board::from_str(fen)
            .map_err(|e| StockfishTestError::ParseError(format!("Invalid FEN: {}", e)))?;

        // Get our evaluation
        let our_evaluation_cp = self.calibrated_evaluator.evaluate_centipawns(&board);

        // Get Stockfish evaluation
        let mut stockfish = StockfishEngine::new(self.stockfish_config.clone())?;
        let (stockfish_evaluation_cp, stockfish_best_move) = stockfish.evaluate_position(fen)?;
        stockfish.close()?;

        // Calculate differences
        let evaluation_diff_cp = our_evaluation_cp - stockfish_evaluation_cp;
        let evaluation_category = Self::classify_evaluation_match(our_evaluation_cp, stockfish_evaluation_cp);
        
        // Move agreement requires full move generation - will be implemented in next phase
        let move_agreement = false; // TODO: Implement move generation and comparison
        let our_best_move = None; // TODO: Add move generation to our engine

        Ok(EvaluationComparison {
            fen: fen.to_string(),
            our_evaluation_cp,
            stockfish_evaluation_cp,
            our_best_move,
            stockfish_best_move,
            evaluation_diff_cp,
            move_agreement,
            evaluation_category,
        })
    }

    /// Test multiple positions and provide statistical analysis
    pub fn test_positions(&mut self, positions: &[&str]) -> Result<TestSuiteResults, StockfishTestError> {
        let mut results = Vec::new();
        let start_time = Instant::now();

        for (i, fen) in positions.iter().enumerate() {
            println!("Testing position {}/{}: {}", i + 1, positions.len(), fen);
            
            match self.test_position(fen) {
                Ok(comparison) => {
                    println!("  Our: {}cp, Stockfish: {}cp, Diff: {}cp, Category: {:?}",
                        comparison.our_evaluation_cp,
                        comparison.stockfish_evaluation_cp,
                        comparison.evaluation_diff_cp,
                        comparison.evaluation_category);
                    results.push(comparison);
                }
                Err(e) => {
                    println!("  Error: {}", e);
                    return Err(e);
                }
            }
        }

        let total_time = start_time.elapsed();
        
        Ok(TestSuiteResults::new(results, total_time))
    }
}

/// Statistical analysis of test results (Single Responsibility Principle)
#[derive(Debug)]
pub struct TestSuiteResults {
    pub comparisons: Vec<EvaluationComparison>,
    pub total_time: Duration,
    pub statistics: TestStatistics,
}

#[derive(Debug)]
pub struct TestStatistics {
    pub total_positions: usize,
    pub exact_matches: usize,
    pub close_matches: usize,
    pub reasonable_matches: usize,
    pub sign_matches: usize,
    pub mismatches: usize,
    pub avg_evaluation_diff_cp: f32,
    pub rms_evaluation_diff_cp: f32,
    pub success_rate: f32,
}

impl TestSuiteResults {
    fn new(comparisons: Vec<EvaluationComparison>, total_time: Duration) -> Self {
        let statistics = Self::calculate_statistics(&comparisons);
        Self {
            comparisons,
            total_time,
            statistics,
        }
    }

    fn calculate_statistics(comparisons: &[EvaluationComparison]) -> TestStatistics {
        let total_positions = comparisons.len();
        
        let mut exact_matches = 0;
        let mut close_matches = 0;
        let mut reasonable_matches = 0;
        let mut sign_matches = 0;
        let mut mismatches = 0;
        
        let mut sum_diff = 0.0;
        let mut sum_squared_diff = 0.0;

        for comparison in comparisons {
            match comparison.evaluation_category {
                EvaluationCategory::ExactMatch => exact_matches += 1,
                EvaluationCategory::CloseMatch => close_matches += 1,
                EvaluationCategory::ReasonableMatch => reasonable_matches += 1,
                EvaluationCategory::SignMatch => sign_matches += 1,
                EvaluationCategory::Mismatch => mismatches += 1,
            }

            let diff = comparison.evaluation_diff_cp as f32;
            sum_diff += diff;
            sum_squared_diff += diff * diff;
        }

        let avg_evaluation_diff_cp = if total_positions > 0 {
            sum_diff / total_positions as f32
        } else {
            0.0
        };

        let rms_evaluation_diff_cp = if total_positions > 0 {
            (sum_squared_diff / total_positions as f32).sqrt()
        } else {
            0.0
        };

        // Success rate = exact + close + reasonable matches
        let successful_matches = exact_matches + close_matches + reasonable_matches;
        let success_rate = if total_positions > 0 {
            successful_matches as f32 / total_positions as f32
        } else {
            0.0
        };

        TestStatistics {
            total_positions,
            exact_matches,
            close_matches,
            reasonable_matches,
            sign_matches,
            mismatches,
            avg_evaluation_diff_cp,
            rms_evaluation_diff_cp,
            success_rate,
        }
    }

    pub fn display_summary(&self) -> String {
        let stats = &self.statistics;
        
        format!(
            "Stockfish Comparison Results\n\
            =============================\n\
            Total positions: {}\n\
            Test duration: {:.2}s\n\n\
            Evaluation Agreement:\n\
            - Exact matches (±25cp): {} ({:.1}%)\n\
            - Close matches (±50cp): {} ({:.1}%)\n\
            - Reasonable matches (±100cp): {} ({:.1}%)\n\
            - Sign matches: {} ({:.1}%)\n\
            - Mismatches: {} ({:.1}%)\n\n\
            Statistical Analysis:\n\
            - Success rate: {:.1}%\n\
            - Average difference: {:.1}cp\n\
            - RMS difference: {:.1}cp\n\n\
            Assessment: {}",
            stats.total_positions,
            self.total_time.as_secs_f32(),
            stats.exact_matches, Self::percentage(stats.exact_matches, stats.total_positions),
            stats.close_matches, Self::percentage(stats.close_matches, stats.total_positions),
            stats.reasonable_matches, Self::percentage(stats.reasonable_matches, stats.total_positions),
            stats.sign_matches, Self::percentage(stats.sign_matches, stats.total_positions),
            stats.mismatches, Self::percentage(stats.mismatches, stats.total_positions),
            stats.success_rate * 100.0,
            stats.avg_evaluation_diff_cp,
            stats.rms_evaluation_diff_cp,
            Self::assessment(stats.success_rate)
        )
    }

    fn percentage(count: usize, total: usize) -> f32 {
        if total > 0 {
            count as f32 / total as f32 * 100.0
        } else {
            0.0
        }
    }

    fn assessment(success_rate: f32) -> &'static str {
        match (success_rate * 100.0) as u32 {
            80..=100 => "✅ Excellent agreement with Stockfish",
            60..=79 => "✅ Good agreement with Stockfish", 
            40..=59 => "⚠️ Moderate agreement with Stockfish",
            20..=39 => "⚠️ Poor agreement with Stockfish",
            _ => "❌ Very poor agreement with Stockfish",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stockfish_config_default() {
        let config = StockfishTestConfig::default();
        assert_eq!(config.depth_limit, Some(6));
        assert_eq!(config.skill_level, Some(10));
    }

    #[test]
    fn test_evaluation_category_classification() {
        assert_eq!(EvaluationCategory::from_difference(10), EvaluationCategory::ExactMatch);
        assert_eq!(EvaluationCategory::from_difference(40), EvaluationCategory::CloseMatch);
        assert_eq!(EvaluationCategory::from_difference(80), EvaluationCategory::ReasonableMatch);
    }
}