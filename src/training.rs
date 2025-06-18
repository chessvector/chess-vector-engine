use chess::{Board, Game, MoveGen, ChessMove};
use pgn_reader::{RawHeader, SanPlus, Skip, Visitor, BufferedReader};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Command, Stdio};
use std::str::FromStr;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::ChessVectorEngine;

/// Training data point containing a position and its evaluation
#[derive(Debug, Clone)]
pub struct TrainingData {
    pub board: Board,
    pub evaluation: f32,
    pub depth: u8,
    pub game_id: usize,
}

/// Tactical puzzle data from Lichess puzzle database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TacticalPuzzle {
    #[serde(rename = "PuzzleId")]
    pub puzzle_id: String,
    #[serde(rename = "FEN")]
    pub fen: String,
    #[serde(rename = "Moves")]
    pub moves: String,           // Space-separated move sequence
    #[serde(rename = "Rating")]
    pub rating: u32,
    #[serde(rename = "RatingDeviation")]
    pub rating_deviation: u32,
    #[serde(rename = "Popularity")]
    pub popularity: i32,
    #[serde(rename = "NbPlays")]
    pub nb_plays: u32,
    #[serde(rename = "Themes")]
    pub themes: String,          // Space-separated themes
    #[serde(rename = "GameUrl")]
    pub game_url: Option<String>,
    #[serde(rename = "OpeningTags")]
    pub opening_tags: Option<String>,
}

/// Processed tactical training data
#[derive(Debug, Clone)]
pub struct TacticalTrainingData {
    pub position: Board,
    pub solution_move: ChessMove,
    pub move_theme: String,
    pub difficulty: f32,         // Rating as difficulty
    pub tactical_value: f32,     // High value for move outcome
}

/// PGN game visitor for extracting positions
pub struct GameExtractor {
    pub positions: Vec<TrainingData>,
    pub current_game: Game,
    pub move_count: usize,
    pub max_moves_per_game: usize,
    pub game_id: usize,
}

impl GameExtractor {
    pub fn new(max_moves_per_game: usize) -> Self {
        Self {
            positions: Vec::new(),
            current_game: Game::new(),
            move_count: 0,
            max_moves_per_game,
            game_id: 0,
        }
    }
}

impl Visitor for GameExtractor {
    type Result = ();

    fn begin_game(&mut self) {
        self.current_game = Game::new();
        self.move_count = 0;
        self.game_id += 1;
    }

    fn header(&mut self, _key: &[u8], _value: RawHeader<'_>) {}

    fn san(&mut self, san_plus: SanPlus) {
        if self.move_count >= self.max_moves_per_game {
            return;
        }

        let san_str = san_plus.san.to_string();
        
        // First validate that we have a legal position to work with
        let current_pos = self.current_game.current_position();
        
        // Try to parse and make the move
        match chess::ChessMove::from_san(&current_pos, &san_str) {
            Ok(chess_move) => {
                // Verify the move is legal before making it
                let legal_moves: Vec<chess::ChessMove> = MoveGen::new_legal(&current_pos).collect();
                if legal_moves.contains(&chess_move) {
                    if self.current_game.make_move(chess_move) {
                        self.move_count += 1;
                        
                        // Store position (we'll evaluate it later with Stockfish)
                        self.positions.push(TrainingData {
                            board: self.current_game.current_position().clone(),
                            evaluation: 0.0, // Will be filled by Stockfish
                            depth: 0,
                            game_id: self.game_id,
                        });
                    }
                } else {
                    // Move parsed but isn't legal - skip silently to avoid spam
                }
            }
            Err(_) => {
                // Failed to parse move - could be notation issues, corruption, etc.
                // Skip silently to avoid excessive error output
                // Only log if it's not a common problematic pattern
                if !san_str.contains("O-O") && !san_str.contains("=") && san_str.len() > 6 {
                    // Only log unusual failed moves to reduce noise
                }
            }
        }
    }

    fn begin_variation(&mut self) -> Skip {
        Skip(true) // Skip variations for now
    }

    fn end_game(&mut self) -> Self::Result {}
}

/// Stockfish engine wrapper for position evaluation
pub struct StockfishEvaluator {
    depth: u8,
}

impl StockfishEvaluator {
    pub fn new(depth: u8) -> Self {
        Self { depth }
    }

    /// Evaluate a single position using Stockfish
    pub fn evaluate_position(&self, board: &Board) -> Result<f32, Box<dyn std::error::Error>> {
        let mut child = Command::new("stockfish")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        let stdin = child.stdin.as_mut().unwrap();
        let fen = board.to_string();
        
        // Send UCI commands
        use std::io::Write;
        writeln!(stdin, "uci")?;
        writeln!(stdin, "isready")?;
        writeln!(stdin, "position fen {}", fen)?;
        writeln!(stdin, "go depth {}", self.depth)?;
        writeln!(stdin, "quit")?;
        
        let output = child.wait_with_output()?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        
        // Parse the evaluation from Stockfish output
        for line in stdout.lines() {
            if line.starts_with("info") && line.contains("score cp") {
                if let Some(cp_pos) = line.find("score cp ") {
                    let cp_str = &line[cp_pos + 9..];
                    if let Some(end) = cp_str.find(' ') {
                        let cp_value = cp_str[..end].parse::<i32>()?;
                        return Ok(cp_value as f32 / 100.0); // Convert centipawns to pawns
                    }
                }
            } else if line.starts_with("info") && line.contains("score mate") {
                // Handle mate scores
                if let Some(mate_pos) = line.find("score mate ") {
                    let mate_str = &line[mate_pos + 11..];
                    if let Some(end) = mate_str.find(' ') {
                        let mate_moves = mate_str[..end].parse::<i32>()?;
                        return Ok(if mate_moves > 0 { 100.0 } else { -100.0 });
                    }
                }
            }
        }
        
        Ok(0.0) // Default to 0 if no evaluation found
    }

    /// Batch evaluate multiple positions
    pub fn evaluate_batch(&self, positions: &mut [TrainingData]) -> Result<(), Box<dyn std::error::Error>> {
        let pb = ProgressBar::new(positions.len() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-"));

        for (i, data) in positions.iter_mut().enumerate() {
            match self.evaluate_position(&data.board) {
                Ok(eval) => {
                    data.evaluation = eval;
                    data.depth = self.depth;
                }
                Err(e) => {
                    eprintln!("Error evaluating position {}: {}", i, e);
                    data.evaluation = 0.0;
                }
            }
            pb.inc(1);
        }
        
        pb.finish_with_message("Evaluation complete");
        Ok(())
    }
    
    /// Evaluate multiple positions in parallel using concurrent Stockfish instances
    pub fn evaluate_batch_parallel(&self, positions: &mut [TrainingData], num_threads: usize) -> Result<(), Box<dyn std::error::Error>> {
        let pb = ProgressBar::new(positions.len() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} Parallel evaluation")
            .unwrap()
            .progress_chars("#>-"));

        // Set the thread pool size
        let pool = rayon::ThreadPoolBuilder::new().num_threads(num_threads).build()?;
        
        pool.install(|| {
            // Use parallel iterator to evaluate positions
            positions.par_iter_mut().for_each(|data| {
                match self.evaluate_position(&data.board) {
                    Ok(eval) => {
                        data.evaluation = eval;
                        data.depth = self.depth;
                    }
                    Err(_) => {
                        // Silently fail individual positions to avoid spamming output
                        data.evaluation = 0.0;
                    }
                }
                pb.inc(1);
            });
        });
        
        pb.finish_with_message("Parallel evaluation complete");
        Ok(())
    }
}

/// Training dataset manager
pub struct TrainingDataset {
    pub data: Vec<TrainingData>,
}

impl TrainingDataset {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Load positions from a PGN file
    pub fn load_from_pgn<P: AsRef<Path>>(
        &mut self,
        path: P,
        max_games: Option<usize>,
        max_moves_per_game: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        
        let mut extractor = GameExtractor::new(max_moves_per_game);
        let mut games_processed = 0;
        
        // Create a simple PGN parser
        let mut pgn_content = String::new();
        for line in reader.lines() {
            let line = line?;
            pgn_content.push_str(&line);
            pgn_content.push('\n');
            
            // Check if this is the end of a game
            if line.trim().ends_with("1-0") || line.trim().ends_with("0-1") || 
               line.trim().ends_with("1/2-1/2") || line.trim().ends_with("*") {
                
                // Parse this game
                let cursor = std::io::Cursor::new(&pgn_content);
                let mut reader = BufferedReader::new(cursor);
                if let Err(e) = reader.read_all(&mut extractor) {
                    eprintln!("Error parsing game {}: {}", games_processed + 1, e);
                }
                
                games_processed += 1;
                pgn_content.clear();
                
                if let Some(max) = max_games {
                    if games_processed >= max {
                        break;
                    }
                }
                
                if games_processed % 100 == 0 {
                    println!("Processed {} games, extracted {} positions", 
                            games_processed, extractor.positions.len());
                }
            }
        }
        
        self.data.extend(extractor.positions);
        println!("Loaded {} positions from {} games", self.data.len(), games_processed);
        Ok(())
    }

    /// Evaluate all positions using Stockfish
    pub fn evaluate_with_stockfish(&mut self, depth: u8) -> Result<(), Box<dyn std::error::Error>> {
        let evaluator = StockfishEvaluator::new(depth);
        evaluator.evaluate_batch(&mut self.data)
    }
    
    /// Evaluate all positions using Stockfish in parallel
    pub fn evaluate_with_stockfish_parallel(&mut self, depth: u8, num_threads: usize) -> Result<(), Box<dyn std::error::Error>> {
        let evaluator = StockfishEvaluator::new(depth);
        evaluator.evaluate_batch_parallel(&mut self.data, num_threads)
    }

    /// Train the vector engine with this dataset
    pub fn train_engine(&self, engine: &mut ChessVectorEngine) {
        let pb = ProgressBar::new(self.data.len() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} Training positions")
            .unwrap()
            .progress_chars("#>-"));

        for data in &self.data {
            engine.add_position(&data.board, data.evaluation);
            pb.inc(1);
        }
        
        pb.finish_with_message("Training complete");
        println!("Trained engine with {} positions", self.data.len());
    }

    /// Split dataset into train/test sets by games to prevent data leakage
    pub fn split(&self, train_ratio: f32) -> (TrainingDataset, TrainingDataset) {
        use std::collections::{HashMap, HashSet};
        use rand::seq::SliceRandom;
        use rand::thread_rng;
        
        // Group positions by game_id
        let mut games: HashMap<usize, Vec<&TrainingData>> = HashMap::new();
        for data in &self.data {
            games.entry(data.game_id).or_insert_with(Vec::new).push(data);
        }
        
        // Get unique game IDs and shuffle them
        let mut game_ids: Vec<usize> = games.keys().cloned().collect();
        game_ids.shuffle(&mut thread_rng());
        
        // Split games by ratio
        let split_point = (game_ids.len() as f32 * train_ratio) as usize;
        let train_game_ids: HashSet<usize> = game_ids[..split_point].iter().cloned().collect();
        
        // Separate positions based on game membership
        let mut train_data = Vec::new();
        let mut test_data = Vec::new();
        
        for data in &self.data {
            if train_game_ids.contains(&data.game_id) {
                train_data.push(data.clone());
            } else {
                test_data.push(data.clone());
            }
        }
        
        (
            TrainingDataset { data: train_data },
            TrainingDataset { data: test_data }
        )
    }

    /// Save dataset to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(&self.data)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load dataset from file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let data = serde_json::from_str(&content)?;
        Ok(Self { data })
    }

    /// Remove near-duplicate positions to reduce overfitting
    pub fn deduplicate(&mut self, similarity_threshold: f32) {
        use crate::PositionEncoder;
        use ndarray::Array1;
        
        if self.data.is_empty() {
            return;
        }
        
        let encoder = PositionEncoder::new(1024);
        let mut keep_indices: Vec<bool> = vec![true; self.data.len()];
        
        // Encode all positions in parallel
        let vectors: Vec<Array1<f32>> = if self.data.len() > 50 {
            self.data.par_iter()
                .map(|data| encoder.encode(&data.board))
                .collect()
        } else {
            self.data.iter()
                .map(|data| encoder.encode(&data.board))
                .collect()
        };
        
        // Compare each position with all previous ones
        for i in 1..self.data.len() {
            if !keep_indices[i] {
                continue;
            }
            
            for j in 0..i {
                if !keep_indices[j] {
                    continue;
                }
                
                let similarity = Self::cosine_similarity(&vectors[i], &vectors[j]);
                if similarity > similarity_threshold {
                    keep_indices[i] = false;
                    break;
                }
            }
        }
        
        // Filter data based on keep_indices
        let original_len = self.data.len();
        self.data = self.data.iter()
            .enumerate()
            .filter_map(|(i, data)| if keep_indices[i] { Some(data.clone()) } else { None })
            .collect();
        
        println!("Deduplicated: {} -> {} positions (removed {} duplicates)", 
                 original_len, self.data.len(), original_len - self.data.len());
    }
    
    /// Remove near-duplicate positions using parallel comparison (faster for large datasets)
    pub fn deduplicate_parallel(&mut self, similarity_threshold: f32, chunk_size: usize) {
        use crate::PositionEncoder;
        use ndarray::Array1;
        use std::sync::{Arc, Mutex};
        
        if self.data.is_empty() {
            return;
        }
        
        let encoder = PositionEncoder::new(1024);
        
        // Encode all positions in parallel
        let vectors: Vec<Array1<f32>> = self.data.par_iter()
            .map(|data| encoder.encode(&data.board))
            .collect();
        
        let keep_indices = Arc::new(Mutex::new(vec![true; self.data.len()]));
        
        // Process in chunks to balance parallelism and memory usage
        (1..self.data.len())
            .collect::<Vec<_>>()
            .par_chunks(chunk_size)
            .for_each(|chunk| {
                for &i in chunk {
                    // Check if this position is still being kept
                    {
                        let indices = keep_indices.lock().unwrap();
                        if !indices[i] {
                            continue;
                        }
                    }
                    
                    // Compare with all previous positions
                    for j in 0..i {
                        {
                            let indices = keep_indices.lock().unwrap();
                            if !indices[j] {
                                continue;
                            }
                        }
                        
                        let similarity = Self::cosine_similarity(&vectors[i], &vectors[j]);
                        if similarity > similarity_threshold {
                            let mut indices = keep_indices.lock().unwrap();
                            indices[i] = false;
                            break;
                        }
                    }
                }
            });
        
        // Filter data based on keep_indices
        let keep_indices = keep_indices.lock().unwrap();
        let original_len = self.data.len();
        self.data = self.data.iter()
            .enumerate()
            .filter_map(|(i, data)| if keep_indices[i] { Some(data.clone()) } else { None })
            .collect();
        
        println!("Parallel deduplicated: {} -> {} positions (removed {} duplicates)", 
                 original_len, self.data.len(), original_len - self.data.len());
    }

    /// Calculate cosine similarity between two vectors
    fn cosine_similarity(a: &ndarray::Array1<f32>, b: &ndarray::Array1<f32>) -> f32 {
        let dot_product = a.dot(b);
        let norm_a = a.dot(a).sqrt();
        let norm_b = b.dot(b).sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
}

/// Engine performance evaluator
pub struct EngineEvaluator {
    stockfish_depth: u8,
}

impl EngineEvaluator {
    pub fn new(stockfish_depth: u8) -> Self {
        Self { stockfish_depth }
    }

    /// Compare engine evaluations against Stockfish on test set
    pub fn evaluate_accuracy(
        &self,
        engine: &ChessVectorEngine,
        test_data: &TrainingDataset,
    ) -> Result<f32, Box<dyn std::error::Error>> {
        let mut total_error = 0.0;
        let mut valid_comparisons = 0;

        let pb = ProgressBar::new(test_data.data.len() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} Evaluating accuracy")
            .unwrap()
            .progress_chars("#>-"));

        for data in &test_data.data {
            if let Some(engine_eval) = engine.evaluate_position(&data.board) {
                let error = (engine_eval - data.evaluation).abs();
                total_error += error;
                valid_comparisons += 1;
            }
            pb.inc(1);
        }

        pb.finish_with_message("Accuracy evaluation complete");
        
        if valid_comparisons > 0 {
            let mean_absolute_error = total_error / valid_comparisons as f32;
            println!("Mean Absolute Error: {:.3} pawns", mean_absolute_error);
            println!("Evaluated {} positions", valid_comparisons);
            Ok(mean_absolute_error)
        } else {
            Ok(f32::INFINITY)
        }
    }
}

// Make TrainingData serializable
impl serde::Serialize for TrainingData {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("TrainingData", 4)?;
        state.serialize_field("fen", &self.board.to_string())?;
        state.serialize_field("evaluation", &self.evaluation)?;
        state.serialize_field("depth", &self.depth)?;
        state.serialize_field("game_id", &self.game_id)?;
        state.end()
    }
}

impl<'de> serde::Deserialize<'de> for TrainingData {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};
        use std::fmt;

        struct TrainingDataVisitor;

        impl<'de> Visitor<'de> for TrainingDataVisitor {
            type Value = TrainingData;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct TrainingData")
            }

            fn visit_map<V>(self, mut map: V) -> Result<TrainingData, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut fen = None;
                let mut evaluation = None;
                let mut depth = None;
                let mut game_id = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        "fen" => {
                            if fen.is_some() {
                                return Err(de::Error::duplicate_field("fen"));
                            }
                            fen = Some(map.next_value()?);
                        }
                        "evaluation" => {
                            if evaluation.is_some() {
                                return Err(de::Error::duplicate_field("evaluation"));
                            }
                            evaluation = Some(map.next_value()?);
                        }
                        "depth" => {
                            if depth.is_some() {
                                return Err(de::Error::duplicate_field("depth"));
                            }
                            depth = Some(map.next_value()?);
                        }
                        "game_id" => {
                            if game_id.is_some() {
                                return Err(de::Error::duplicate_field("game_id"));
                            }
                            game_id = Some(map.next_value()?);
                        }
                        _ => {
                            let _: serde_json::Value = map.next_value()?;
                        }
                    }
                }

                let fen: String = fen.ok_or_else(|| de::Error::missing_field("fen"))?;
                let evaluation = evaluation.ok_or_else(|| de::Error::missing_field("evaluation"))?;
                let depth = depth.ok_or_else(|| de::Error::missing_field("depth"))?;
                let game_id = game_id.unwrap_or(0); // Default to 0 for backward compatibility

                let board = Board::from_str(&fen)
                    .map_err(|e| de::Error::custom(format!("Invalid FEN: {}", e)))?;

                Ok(TrainingData {
                    board,
                    evaluation,
                    depth,
                    game_id,
                })
            }
        }

        const FIELDS: &'static [&'static str] = &["fen", "evaluation", "depth", "game_id"];
        deserializer.deserialize_struct("TrainingData", FIELDS, TrainingDataVisitor)
    }
}

/// Tactical puzzle parser for Lichess puzzle database
pub struct TacticalPuzzleParser;

impl TacticalPuzzleParser {
    /// Parse Lichess puzzle CSV file
    pub fn parse_csv<P: AsRef<Path>>(
        file_path: P,
        max_puzzles: Option<usize>,
        min_rating: Option<u32>,
        max_rating: Option<u32>,
    ) -> Result<Vec<TacticalTrainingData>, Box<dyn std::error::Error>> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        
        // Create CSV reader without headers since Lichess CSV has no header row
        // Set flexible field count to handle inconsistent CSV structure
        let mut csv_reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .flexible(true)  // Allow variable number of fields
            .from_reader(reader);
        
        let mut tactical_data = Vec::new();
        let mut processed = 0;
        let mut skipped = 0;
        
        let pb = ProgressBar::new_spinner();
        pb.set_style(ProgressStyle::default_spinner()
            .template("{spinner:.green} Parsing tactical puzzles: {pos} (skipped: {skipped})")
            .unwrap());
        
        for result in csv_reader.records() {
            let record = match result {
                Ok(r) => r,
                Err(e) => {
                    skipped += 1;
                    println!("CSV parse error: {}", e);
                    continue;
                }
            };
            
            // Parse record manually since we don't have headers
            // Need at least 8 fields (PuzzleId, FEN, Moves, Rating, RatingDeviation, Popularity, NbPlays, Themes)
            if record.len() < 8 {
                skipped += 1;
                continue;
            }
            
            let puzzle_id = record[0].to_string();
            let fen = record[1].to_string();
            let moves = record[2].to_string();
            let rating: u32 = match record[3].parse() {
                Ok(r) => r,
                Err(_) => {
                    skipped += 1;
                    continue;
                }
            };
            let rating_deviation: u32 = match record[4].parse() {
                Ok(r) => r,
                Err(_) => {
                    skipped += 1;
                    continue;
                }
            };
            let popularity: i32 = match record[5].parse() {
                Ok(p) => p,
                Err(_) => {
                    skipped += 1;
                    continue;
                }
            };
            let nb_plays: u32 = match record[6].parse() {
                Ok(n) => n,
                Err(_) => {
                    skipped += 1;
                    continue;
                }
            };
            let themes = record[7].to_string();
            let game_url = if record.len() > 8 { Some(record[8].to_string()) } else { None };
            let opening_tags = if record.len() > 9 { Some(record[9].to_string()) } else { None };
            
            let puzzle = TacticalPuzzle {
                puzzle_id,
                fen,
                moves,
                rating,
                rating_deviation,
                popularity,
                nb_plays,
                themes,
                game_url,
                opening_tags,
            };
            
            // Apply rating filters
            if let Some(min) = min_rating {
                if puzzle.rating < min { 
                    skipped += 1;
                    continue; 
                }
            }
            if let Some(max) = max_rating {
                if puzzle.rating > max { 
                    skipped += 1;
                    continue; 
                }
            }
            
            // Parse and process puzzle
            if let Some(training_data) = Self::process_puzzle(puzzle) {
                tactical_data.push(training_data);
                processed += 1;
                pb.set_position(processed as u64);
                
                // Check max limit
                if let Some(max) = max_puzzles {
                    if processed >= max { break; }
                }
            } else {
                skipped += 1;
            }
            
            // Update progress message with skip count
            if (processed + skipped) % 1000 == 0 {
                pb.set_message(format!("Parsing tactical puzzles: {} (skipped: {})", processed, skipped));
            }
        }
        
        pb.finish_with_message(format!("Parsed {} tactical puzzles (skipped {})", tactical_data.len(), skipped));
        Ok(tactical_data)
    }
    
    /// Process individual puzzle into training data
    fn process_puzzle(puzzle: TacticalPuzzle) -> Option<TacticalTrainingData> {
        // Parse FEN position
        let position = match Board::from_str(&puzzle.fen) {
            Ok(board) => board,
            Err(_) => return None,
        };
        
        // Parse move sequence - first move is the solution
        let moves: Vec<&str> = puzzle.moves.split_whitespace().collect();
        if moves.is_empty() {
            return None;
        }
        
        // Parse the solution move (first move in sequence)
        let solution_move = match ChessMove::from_str(moves[0]) {
            Ok(mv) => mv,
            Err(_) => {
                // Try parsing as SAN
                match ChessMove::from_san(&position, moves[0]) {
                    Ok(mv) => mv,
                    Err(_) => return None,
                }
            }
        };
        
        // Verify move is legal
        let legal_moves: Vec<ChessMove> = MoveGen::new_legal(&position).collect();
        if !legal_moves.contains(&solution_move) {
            return None;
        }
        
        // Extract primary theme
        let themes: Vec<&str> = puzzle.themes.split_whitespace().collect();
        let primary_theme = themes.first().unwrap_or(&"tactical").to_string();
        
        // Calculate tactical value based on rating and popularity
        let difficulty = puzzle.rating as f32 / 1000.0; // Normalize to 0.8-3.0 range
        let popularity_bonus = (puzzle.popularity as f32 / 100.0).min(2.0);
        let tactical_value = difficulty + popularity_bonus; // High value for move outcome
        
        Some(TacticalTrainingData {
            position,
            solution_move,
            move_theme: primary_theme,
            difficulty,
            tactical_value,
        })
    }
    
    /// Load tactical training data into chess engine
    pub fn load_into_engine(
        tactical_data: &[TacticalTrainingData],
        engine: &mut ChessVectorEngine,
    ) {
        let pb = ProgressBar::new(tactical_data.len() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} Loading tactical patterns")
            .unwrap()
            .progress_chars("#>-"));
        
        for data in tactical_data {
            // Add position with high-value tactical move
            engine.add_position_with_move(
                &data.position,
                0.0, // Position evaluation (neutral for puzzles)
                Some(data.solution_move),
                Some(data.tactical_value), // High tactical value
            );
            pb.inc(1);
        }
        
        pb.finish_with_message(format!("Loaded {} tactical patterns", tactical_data.len()));
    }
}