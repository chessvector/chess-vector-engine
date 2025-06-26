#![allow(clippy::type_complexity)]
use crate::TrainingData;
use chess::{Board, ChessMove, Color};
use rayon::prelude::*;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Lichess puzzle entry from CSV
#[derive(Debug, Deserialize)]
struct LichessPuzzle {
    #[serde(rename = "PuzzleId")]
    #[allow(dead_code)]
    puzzle_id: String,
    #[serde(rename = "FEN")]
    #[allow(dead_code)]
    fen: String,
    #[serde(rename = "Moves")]
    #[allow(dead_code)]
    moves: String,
    #[serde(rename = "Rating")]
    #[allow(dead_code)]
    rating: u32,
    #[serde(rename = "RatingDeviation")]
    #[allow(dead_code)]
    rating_deviation: u32,
    #[serde(rename = "Popularity")]
    #[allow(dead_code)]
    popularity: i32,
    #[serde(rename = "NbPlays")]
    #[allow(dead_code)]
    nb_plays: u32,
    #[serde(rename = "Themes")]
    #[allow(dead_code)]
    themes: String,
    #[serde(rename = "GameUrl")]
    #[allow(dead_code)]
    game_url: String,
}

/// High-performance Lichess puzzle database loader
pub struct LichessLoader {
    /// Minimum puzzle rating to include
    min_rating: u32,
    /// Maximum puzzle rating to include  
    max_rating: u32,
    /// Batch size for parallel processing
    batch_size: usize,
    /// Number of worker threads
    num_threads: usize,
    /// Filter by themes (e.g., "checkmate", "fork", "pin")
    theme_filter: Option<Vec<String>>,
}

impl LichessLoader {
    /// Create a new Lichess loader with default settings
    pub fn new() -> Self {
        Self {
            min_rating: 800,                      // Exclude beginner puzzles
            max_rating: 2800,                     // Include all reasonable puzzles
            batch_size: 10_000,                   // Process 10k puzzles per batch
            num_threads: num_cpus::get().min(16), // Use available cores (max 16)
            theme_filter: None,
        }
    }

    /// Create a premium loader with optimized settings
    pub fn new_premium() -> Self {
        Self {
            min_rating: 1000,                     // Focus on intermediate+ puzzles
            max_rating: 2500,                     // Exclude super-GM puzzles
            batch_size: 50_000,                   // Larger batches for premium performance
            num_threads: num_cpus::get().min(32), // Use more cores for premium
            theme_filter: Some(vec![
                "checkmate".to_string(),
                "mateIn2".to_string(),
                "mateIn3".to_string(),
                "fork".to_string(),
                "pin".to_string(),
                "skewer".to_string(),
                "discovery".to_string(),
                "sacrifice".to_string(),
                "deflection".to_string(),
                "attraction".to_string(),
            ]),
        }
    }

    /// Set rating range filter
    pub fn with_rating_range(mut self, min: u32, max: u32) -> Self {
        self.min_rating = min;
        self.max_rating = max;
        self
    }

    /// Set theme filter
    pub fn with_themes(mut self, themes: Vec<String>) -> Self {
        self.theme_filter = Some(themes);
        self
    }

    /// Set batch size for memory control
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Load training data from Lichess puzzle CSV with lightning speed
    pub fn load_parallel<P: AsRef<Path>>(
        &self,
        csv_path: P,
    ) -> Result<Vec<TrainingData>, Box<dyn std::error::Error>> {
        let start_time = Instant::now();
        let path = csv_path.as_ref();

        println!("🔥 Lightning-fast Lichess puzzle loader starting...");
        println!("Loading from file: {}", path.display());
        println!("⚡ Parallel processing with {} threads", self.num_threads);
        println!("Processing data...");

        // Configure rayon thread pool
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.num_threads)
            .build()?;

        let results = Arc::new(Mutex::new(Vec::new()));
        let total_processed = Arc::new(Mutex::new(0usize));
        let valid_puzzles = Arc::new(Mutex::new(0usize));

        // Read file in streaming chunks to control memory
        let file = File::open(path)?;
        let reader = BufReader::with_capacity(1024 * 1024, file); // 1MB buffer

        // Skip header line
        let mut lines = reader.lines();
        lines.next(); // Skip CSV header

        // Process in batches
        let mut batch = Vec::with_capacity(self.batch_size);
        let mut batch_count = 0;

        for line in lines {
            let line = line?;
            batch.push(line);

            if batch.len() >= self.batch_size {
                batch_count += 1;
                let batch_data = std::mem::take(&mut batch);

                // Process batch in parallel
                let batch_results = self.process_batch_parallel(&pool, batch_data)?;

                // Accumulate results
                {
                    let mut results_guard = results.lock().unwrap();
                    let mut processed_guard = total_processed.lock().unwrap();
                    let mut valid_guard = valid_puzzles.lock().unwrap();

                    *processed_guard += self.batch_size;
                    *valid_guard += batch_results.len();
                    results_guard.extend(batch_results);

                    if batch_count % 10 == 0 {
                        println!(
                            "📈 Batch {}: Processed {}k puzzles, {} valid positions",
                            batch_count,
                            *processed_guard / 1000,
                            *valid_guard
                        );
                    }
                }

                batch = Vec::with_capacity(self.batch_size);
            }
        }

        // Process final partial batch
        if !batch.is_empty() {
            let batch_results = self.process_batch_parallel(&pool, batch)?;
            let mut results_guard = results.lock().unwrap();
            results_guard.extend(batch_results);
        }

        let final_results = Arc::try_unwrap(results).unwrap().into_inner().unwrap();
        let elapsed = start_time.elapsed();

        println!("🎉 Lightning loading complete!");
        println!("⏱️  Time: {:.2}s", elapsed.as_secs_f64());
        println!("📊 Loaded {} training positions", final_results.len());
        println!(
            "🚀 Speed: {:.0} puzzles/second",
            final_results.len() as f64 / elapsed.as_secs_f64()
        );

        Ok(final_results)
    }

    /// Load training data with moves from Lichess puzzle CSV for pattern recognition
    pub fn load_parallel_with_moves<P: AsRef<Path>>(
        &self,
        csv_path: P,
    ) -> Result<Vec<(Board, f32, ChessMove)>, Box<dyn std::error::Error>> {
        let start_time = Instant::now();
        let path = csv_path.as_ref();

        println!("🧠 Lightning-fast Lichess puzzle loader (with moves) starting...");
        println!("Loading from file: {}", path.display());
        println!("⚡ Parallel processing with {} threads", self.num_threads);
        println!("Processing data...");

        // Configure rayon thread pool
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.num_threads)
            .build()?;

        let results = Arc::new(Mutex::new(Vec::new()));
        let total_processed = Arc::new(Mutex::new(0usize));
        let valid_puzzles = Arc::new(Mutex::new(0usize));

        // Read file in streaming chunks to control memory
        let file = File::open(path)?;
        let reader = BufReader::with_capacity(1024 * 1024, file); // 1MB buffer

        // Skip header line
        let mut lines = reader.lines();
        lines.next(); // Skip CSV header

        // Process in batches
        let mut batch = Vec::with_capacity(self.batch_size);
        let mut batch_count = 0;

        for line in lines {
            let line = line?;
            batch.push(line);

            if batch.len() >= self.batch_size {
                batch_count += 1;
                let batch_data = std::mem::take(&mut batch);

                // Process batch in parallel
                let batch_results = self.process_batch_parallel_with_moves(&pool, batch_data)?;

                // Accumulate results
                {
                    let mut results_guard = results.lock().unwrap();
                    let mut processed_guard = total_processed.lock().unwrap();
                    let mut valid_guard = valid_puzzles.lock().unwrap();

                    *processed_guard += self.batch_size;
                    *valid_guard += batch_results.len();
                    results_guard.extend(batch_results);

                    if batch_count % 10 == 0 {
                        println!(
                            "📈 Batch {}: Processed {}k puzzles, {} valid moves",
                            batch_count,
                            *processed_guard / 1000,
                            *valid_guard
                        );
                    }
                }

                batch = Vec::with_capacity(self.batch_size);
            }
        }

        // Process final partial batch
        if !batch.is_empty() {
            let batch_results = self.process_batch_parallel_with_moves(&pool, batch)?;
            let mut results_guard = results.lock().unwrap();
            results_guard.extend(batch_results);
        }

        let final_results = Arc::try_unwrap(results).unwrap().into_inner().unwrap();
        let elapsed = start_time.elapsed();

        println!("🎉 Lightning loading with moves complete!");
        println!("⏱️  Time: {:.2}s", elapsed.as_secs_f64());
        println!("🧠 Loaded {} tactical moves", final_results.len());
        println!(
            "🚀 Speed: {:.0} puzzles/second",
            final_results.len() as f64 / elapsed.as_secs_f64()
        );

        Ok(final_results)
    }

    /// Process a batch of CSV lines in parallel
    fn process_batch_parallel(
        &self,
        pool: &rayon::ThreadPool,
        batch: Vec<String>,
    ) -> Result<Vec<TrainingData>, Box<dyn std::error::Error>> {
        let loader = self;
        let batch_results: Vec<_> = pool.install(|| {
            batch
                .par_iter()
                .filter_map(|line| loader.parse_puzzle_line(line).ok().flatten())
                .collect()
        });

        Ok(batch_results)
    }

    /// Process a batch of CSV lines in parallel with moves for pattern recognition
    fn process_batch_parallel_with_moves(
        &self,
        pool: &rayon::ThreadPool,
        batch: Vec<String>,
    ) -> Result<Vec<(Board, f32, ChessMove)>, Box<dyn std::error::Error>> {
        let loader = self;
        let batch_results: Vec<_> = pool.install(|| {
            batch
                .par_iter()
                .filter_map(|line| loader.parse_puzzle_line_with_move(line).ok().flatten())
                .collect()
        });

        Ok(batch_results)
    }

    /// Parse a single CSV line into training data
    fn parse_puzzle_line(
        &self,
        line: &str,
    ) -> Result<Option<TrainingData>, Box<dyn std::error::Error>> {
        // Wrap entire parsing in panic protection to catch any chess library panics
        match std::panic::catch_unwind(
            || -> Result<Option<TrainingData>, Box<dyn std::error::Error>> {
                // Use proper CSV parsing to handle quotes and commas correctly
                let mut reader = csv::ReaderBuilder::new()
                    .has_headers(false)
                    .from_reader(line.as_bytes());

                let record = match reader.records().next() {
                    Some(Ok(record)) => record,
                    _ => return Ok(None), // Skip malformed lines
                };

                if record.len() < 8 {
                    return Ok(None); // Skip malformed lines
                }

                // Extract key fields by position with proper CSV parsing
                let fen = record.get(1).unwrap_or("").trim();
                let moves = record.get(2).unwrap_or("").trim();
                let rating: u32 = record.get(3).unwrap_or("0").parse().unwrap_or(0);
                let themes = record.get(7).unwrap_or("").trim();

                // Apply filters for performance
                if rating < self.min_rating || rating > self.max_rating {
                    return Ok(None);
                }

                if let Some(ref theme_filter) = self.theme_filter {
                    let has_target_theme = theme_filter.iter().any(|theme| themes.contains(theme));
                    if !has_target_theme {
                        return Ok(None);
                    }
                }

                // Parse board position
                let board = match Board::from_str(fen) {
                    Ok(b) => b,
                    Err(_) => return Ok(None), // Skip invalid FEN
                };

                // Parse moves and create training data
                let move_sequence: Vec<&str> = moves.split_whitespace().collect();
                if move_sequence.is_empty() {
                    return Ok(None);
                }

                // Use the first move as the target move - validate it's legal and board is valid
                let _target_move = match ChessMove::from_str(move_sequence[0]) {
                    Ok(m) => {
                        // Verify the move is legal for this position
                        use chess::MoveGen;
                        let legal_moves: Vec<ChessMove> = MoveGen::new_legal(&board).collect();

                        // Skip positions with no legal moves (checkmate/stalemate)
                        if legal_moves.is_empty() {
                            return Ok(None);
                        }

                        if legal_moves.contains(&m) {
                            m
                        } else {
                            return Ok(None); // Skip illegal moves
                        }
                    }
                    Err(_) => return Ok(None), // Skip invalid moves
                };

                // Calculate evaluation based on puzzle rating and themes
                let evaluation = self.calculate_puzzle_evaluation(rating, themes, &board);

                Ok(Some(TrainingData {
                    board,
                    evaluation,
                    depth: 1,                 // Puzzle depth
                    game_id: rating as usize, // Use puzzle rating as game_id for uniqueness
                }))
            },
        ) {
            Ok(result) => result,
            Err(_) => Ok(None), // Skip any position that causes panic
        }
    }

    /// Calculate position evaluation based on puzzle characteristics
    fn calculate_puzzle_evaluation(&self, rating: u32, themes: &str, board: &Board) -> f32 {
        let mut eval = 0.0;

        // Base evaluation from puzzle difficulty (normalized to pawn units)
        eval += (rating as f32 - 1500.0) / 1000.0; // Much smaller scaling for reasonable range

        // Moderate tactical theme adjustments (in pawn units)
        if themes.contains("checkmate") || themes.contains("mateIn") {
            eval += if board.side_to_move() == Color::White {
                5.0 // Strong advantage (5 pawns)
            } else {
                -5.0 // Strong disadvantage (5 pawns)
            };
        } else if themes.contains("fork") || themes.contains("pin") {
            eval += if board.side_to_move() == Color::White {
                2.0 // Moderate advantage (2 pawns)
            } else {
                -2.0 // Moderate disadvantage (2 pawns)
            };
        } else if themes.contains("sacrifice") {
            eval += if board.side_to_move() == Color::White {
                1.5 // Small advantage (1.5 pawns)
            } else {
                -1.5 // Small disadvantage (1.5 pawns)
            };
        }

        // Clamp to reasonable evaluation range
        eval.clamp(-8.0, 8.0)
    }

    /// Parse a single CSV line into position, evaluation, and best move
    fn parse_puzzle_line_with_move(
        &self,
        line: &str,
    ) -> Result<Option<(Board, f32, ChessMove)>, Box<dyn std::error::Error>> {
        // Wrap entire parsing in panic protection to catch any chess library panics
        match std::panic::catch_unwind(
            || -> Result<Option<(Board, f32, ChessMove)>, Box<dyn std::error::Error>> {
                // Use proper CSV parsing to handle quotes and commas correctly
                let mut reader = csv::ReaderBuilder::new()
                    .has_headers(false)
                    .from_reader(line.as_bytes());

                let record = match reader.records().next() {
                    Some(Ok(record)) => record,
                    _ => return Ok(None), // Skip malformed lines
                };

                if record.len() < 8 {
                    return Ok(None); // Skip malformed lines
                }

                // Extract key fields by position with proper CSV parsing
                let fen = record.get(1).unwrap_or("").trim();
                let moves = record.get(2).unwrap_or("").trim();
                let rating: u32 = record.get(3).unwrap_or("0").parse().unwrap_or(0);
                let themes = record.get(7).unwrap_or("").trim();

                // Apply filters for performance
                if rating < self.min_rating || rating > self.max_rating {
                    return Ok(None);
                }

                if let Some(ref theme_filter) = self.theme_filter {
                    let has_target_theme = theme_filter.iter().any(|theme| themes.contains(theme));
                    if !has_target_theme {
                        return Ok(None);
                    }
                }

                // Parse board position
                let board = match Board::from_str(fen) {
                    Ok(b) => b,
                    Err(_) => return Ok(None), // Skip invalid FEN
                };

                // Parse moves and create training data
                let move_sequence: Vec<&str> = moves.split_whitespace().collect();
                if move_sequence.is_empty() {
                    return Ok(None);
                }

                // Use the first move as the target move - validate it's legal and board is valid
                let target_move = match ChessMove::from_str(move_sequence[0]) {
                    Ok(m) => {
                        // Verify the move is legal for this position
                        use chess::MoveGen;
                        let legal_moves: Vec<ChessMove> = MoveGen::new_legal(&board).collect();

                        // Skip positions with no legal moves (checkmate/stalemate)
                        if legal_moves.is_empty() {
                            return Ok(None);
                        }

                        if legal_moves.contains(&m) {
                            m
                        } else {
                            // For debugging: could add logging here to track rejection reasons
                            return Ok(None); // Skip illegal moves
                        }
                    }
                    Err(_) => return Ok(None), // Skip invalid moves
                };

                // Calculate evaluation based on puzzle rating and themes
                let evaluation = self.calculate_puzzle_evaluation(rating, themes, &board);

                Ok(Some((board, evaluation, target_move)))
            },
        ) {
            Ok(result) => result,
            Err(_) => Ok(None), // Skip any position that causes panic
        }
    }
}

impl Default for LichessLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Premium feature: Load Lichess puzzles with maximum performance
pub fn load_lichess_puzzles_premium<P: AsRef<Path>>(
    csv_path: P,
) -> Result<Vec<TrainingData>, Box<dyn std::error::Error>> {
    let loader = LichessLoader::new_premium()
        .with_rating_range(1200, 2400) // Focus on strong tactical puzzles
        .with_batch_size(100_000); // Large batches for premium speed

    loader.load_parallel(csv_path)
}

/// Open source feature: Load limited Lichess puzzles
pub fn load_lichess_puzzles_basic<P: AsRef<Path>>(
    csv_path: P,
    max_puzzles: usize,
) -> Result<Vec<TrainingData>, Box<dyn std::error::Error>> {
    let loader = LichessLoader::new()
        .with_rating_range(1000, 2000) // Basic tactical puzzles
        .with_batch_size(10_000); // Smaller batches for basic tier

    let mut results = loader.load_parallel(csv_path)?;
    results.truncate(max_puzzles); // Limit for open source
    Ok(results)
}

/// Premium feature: Load Lichess puzzles with moves for pattern recognition
pub fn load_lichess_puzzles_premium_with_moves<P: AsRef<Path>>(
    csv_path: P,
) -> Result<Vec<(Board, f32, ChessMove)>, Box<dyn std::error::Error>> {
    let loader = LichessLoader::new_premium()
        .with_rating_range(1200, 2400) // Focus on strong tactical puzzles
        .with_batch_size(100_000); // Large batches for premium speed

    loader.load_parallel_with_moves(csv_path)
}

/// Open source feature: Load limited Lichess puzzles with moves
pub fn load_lichess_puzzles_basic_with_moves<P: AsRef<Path>>(
    csv_path: P,
    max_puzzles: usize,
) -> Result<Vec<(Board, f32, ChessMove)>, Box<dyn std::error::Error>> {
    let loader = LichessLoader::new()
        .with_rating_range(1000, 2000) // Basic tactical puzzles
        .with_batch_size(10_000); // Smaller batches for basic tier

    let mut results = loader.load_parallel_with_moves(csv_path)?;
    results.truncate(max_puzzles); // Limit for open source
    Ok(results)
}
