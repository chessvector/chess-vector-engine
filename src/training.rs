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

// Make TacticalTrainingData serializable
impl serde::Serialize for TacticalTrainingData {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("TacticalTrainingData", 5)?;
        state.serialize_field("fen", &self.position.to_string())?;
        state.serialize_field("solution_move", &self.solution_move.to_string())?;
        state.serialize_field("move_theme", &self.move_theme)?;
        state.serialize_field("difficulty", &self.difficulty)?;
        state.serialize_field("tactical_value", &self.tactical_value)?;
        state.end()
    }
}

impl<'de> serde::Deserialize<'de> for TacticalTrainingData {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};
        use std::fmt;

        struct TacticalTrainingDataVisitor;

        impl<'de> Visitor<'de> for TacticalTrainingDataVisitor {
            type Value = TacticalTrainingData;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct TacticalTrainingData")
            }

            fn visit_map<V>(self, mut map: V) -> Result<TacticalTrainingData, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut fen = None;
                let mut solution_move = None;
                let mut move_theme = None;
                let mut difficulty = None;
                let mut tactical_value = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        "fen" => {
                            if fen.is_some() {
                                return Err(de::Error::duplicate_field("fen"));
                            }
                            fen = Some(map.next_value()?);
                        }
                        "solution_move" => {
                            if solution_move.is_some() {
                                return Err(de::Error::duplicate_field("solution_move"));
                            }
                            solution_move = Some(map.next_value()?);
                        }
                        "move_theme" => {
                            if move_theme.is_some() {
                                return Err(de::Error::duplicate_field("move_theme"));
                            }
                            move_theme = Some(map.next_value()?);
                        }
                        "difficulty" => {
                            if difficulty.is_some() {
                                return Err(de::Error::duplicate_field("difficulty"));
                            }
                            difficulty = Some(map.next_value()?);
                        }
                        "tactical_value" => {
                            if tactical_value.is_some() {
                                return Err(de::Error::duplicate_field("tactical_value"));
                            }
                            tactical_value = Some(map.next_value()?);
                        }
                        _ => {
                            let _: serde_json::Value = map.next_value()?;
                        }
                    }
                }

                let fen: String = fen.ok_or_else(|| de::Error::missing_field("fen"))?;
                let solution_move_str: String = solution_move.ok_or_else(|| de::Error::missing_field("solution_move"))?;
                let move_theme = move_theme.ok_or_else(|| de::Error::missing_field("move_theme"))?;
                let difficulty = difficulty.ok_or_else(|| de::Error::missing_field("difficulty"))?;
                let tactical_value = tactical_value.ok_or_else(|| de::Error::missing_field("tactical_value"))?;

                let position = Board::from_str(&fen)
                    .map_err(|e| de::Error::custom(format!("Invalid FEN: {}", e)))?;
                
                let solution_move = ChessMove::from_str(&solution_move_str)
                    .map_err(|e| de::Error::custom(format!("Invalid move: {}", e)))?;

                Ok(TacticalTrainingData {
                    position,
                    solution_move,
                    move_theme,
                    difficulty,
                    tactical_value,
                })
            }
        }

        const FIELDS: &'static [&'static str] = &["fen", "solution_move", "move_theme", "difficulty", "tactical_value"];
        deserializer.deserialize_struct("TacticalTrainingData", FIELDS, TacticalTrainingDataVisitor)
    }
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

    /// Load and append data from file to existing dataset (incremental training)
    pub fn load_and_append<P: AsRef<Path>>(&mut self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let existing_len = self.data.len();
        let additional_data = Self::load(path)?;
        self.data.extend(additional_data.data);
        println!("Loaded {} additional positions (total: {})", 
                self.data.len() - existing_len, self.data.len());
        Ok(())
    }

    /// Merge another dataset into this one
    pub fn merge(&mut self, other: TrainingDataset) {
        let existing_len = self.data.len();
        self.data.extend(other.data);
        println!("Merged {} positions (total: {})", 
                self.data.len() - existing_len, self.data.len());
    }

    /// Save incrementally (append to existing file if it exists)
    pub fn save_incremental<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let path = path.as_ref();
        
        if path.exists() {
            // Load existing data, merge, and save
            let mut existing = Self::load(path)?;
            let original_len = existing.data.len();
            existing.data.extend(self.data.iter().cloned());
            
            // Deduplicate to avoid storing the same positions
            existing.deduplicate(0.999); // Very high threshold to only remove exact duplicates
            
            let json = serde_json::to_string_pretty(&existing.data)?;
            std::fs::write(path, json)?;
            
            println!("Incremental save: added {} new positions (total: {})", 
                    existing.data.len() - original_len, existing.data.len());
        } else {
            // File doesn't exist, just save normally
            self.save(path)?;
        }
        Ok(())
    }

    /// Add a single training data point
    pub fn add_position(&mut self, board: Board, evaluation: f32, depth: u8, game_id: usize) {
        self.data.push(TrainingData {
            board,
            evaluation,
            depth,
            game_id,
        });
    }

    /// Get the next available game ID for incremental training
    pub fn next_game_id(&self) -> usize {
        self.data.iter()
            .map(|data| data.game_id)
            .max()
            .unwrap_or(0) + 1
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
    #[allow(dead_code)]
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
    /// Parse Lichess puzzle CSV file with parallel processing
    pub fn parse_csv<P: AsRef<Path>>(
        file_path: P,
        max_puzzles: Option<usize>,
        min_rating: Option<u32>,
        max_rating: Option<u32>,
    ) -> Result<Vec<TacticalTrainingData>, Box<dyn std::error::Error>> {
        let file = File::open(&file_path)?;
        let file_size = file.metadata()?.len();
        
        // For large files (>100MB), use parallel processing
        if file_size > 100_000_000 {
            Self::parse_csv_parallel(file_path, max_puzzles, min_rating, max_rating)
        } else {
            Self::parse_csv_sequential(file_path, max_puzzles, min_rating, max_rating)
        }
    }
    
    /// Sequential CSV parsing for smaller files
    fn parse_csv_sequential<P: AsRef<Path>>(
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
            
            if let Some(puzzle_data) = Self::parse_csv_record(&record, min_rating, max_rating) {
                if let Some(tactical_data_item) = Self::convert_puzzle_to_training_data(&puzzle_data) {
                    tactical_data.push(tactical_data_item);
                    processed += 1;
                    
                    if let Some(max) = max_puzzles {
                        if processed >= max {
                            break;
                        }
                    }
                } else {
                    skipped += 1;
                }
            } else {
                skipped += 1;
            }
            
            pb.set_message(format!("Parsing tactical puzzles: {} (skipped: {})", processed, skipped));
        }
        
        pb.finish_with_message(format!("Parsed {} puzzles (skipped: {})", processed, skipped));
        
        Ok(tactical_data)
    }
    
    /// Parallel CSV parsing for large files
    fn parse_csv_parallel<P: AsRef<Path>>(
        file_path: P,
        max_puzzles: Option<usize>,
        min_rating: Option<u32>,
        max_rating: Option<u32>,
    ) -> Result<Vec<TacticalTrainingData>, Box<dyn std::error::Error>> {
        use std::io::Read;
        
        let mut file = File::open(&file_path)?;
        
        // Read entire file into memory for parallel processing
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        
        // Split into lines for parallel processing
        let lines: Vec<&str> = contents.lines().collect();
        
        let pb = ProgressBar::new(lines.len() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} Parallel CSV parsing")
            .unwrap()
            .progress_chars("#>-"));
        
        // Process lines in parallel
        let tactical_data: Vec<TacticalTrainingData> = lines
            .par_iter()
            .take(max_puzzles.unwrap_or(usize::MAX))
            .filter_map(|line| {
                // Parse CSV line manually
                let fields: Vec<&str> = line.split(',').collect();
                if fields.len() < 8 {
                    return None;
                }
                
                // Build puzzle from fields
                if let Some(puzzle_data) = Self::parse_csv_fields(&fields, min_rating, max_rating) {
                    Self::convert_puzzle_to_training_data(&puzzle_data)
                } else {
                    None
                }
            })
            .collect();
        
        pb.finish_with_message(format!("Parallel parsing complete: {} puzzles", tactical_data.len()));
        
        Ok(tactical_data)
    }
    
    /// Parse CSV record into TacticalPuzzle
    fn parse_csv_record(
        record: &csv::StringRecord,
        min_rating: Option<u32>,
        max_rating: Option<u32>,
    ) -> Option<TacticalPuzzle> {
        // Need at least 8 fields (PuzzleId, FEN, Moves, Rating, RatingDeviation, Popularity, NbPlays, Themes)
        if record.len() < 8 {
            return None;
        }
        
        let rating: u32 = record[3].parse().ok()?;
        let rating_deviation: u32 = record[4].parse().ok()?;
        let popularity: i32 = record[5].parse().ok()?;
        let nb_plays: u32 = record[6].parse().ok()?;
        
        // Apply rating filters
        if let Some(min) = min_rating {
            if rating < min { 
                return None;
            }
        }
        if let Some(max) = max_rating {
            if rating > max { 
                return None;
            }
        }
        
        Some(TacticalPuzzle {
            puzzle_id: record[0].to_string(),
            fen: record[1].to_string(),
            moves: record[2].to_string(),
            rating,
            rating_deviation,
            popularity,
            nb_plays,
            themes: record[7].to_string(),
            game_url: if record.len() > 8 { Some(record[8].to_string()) } else { None },
            opening_tags: if record.len() > 9 { Some(record[9].to_string()) } else { None },
        })
    }
    
    /// Parse CSV fields into TacticalPuzzle (for parallel processing)
    fn parse_csv_fields(
        fields: &[&str],
        min_rating: Option<u32>,
        max_rating: Option<u32>,
    ) -> Option<TacticalPuzzle> {
        if fields.len() < 8 {
            return None;
        }
        
        let rating: u32 = fields[3].parse().ok()?;
        let rating_deviation: u32 = fields[4].parse().ok()?;
        let popularity: i32 = fields[5].parse().ok()?;
        let nb_plays: u32 = fields[6].parse().ok()?;
        
        // Apply rating filters
        if let Some(min) = min_rating {
            if rating < min { 
                return None;
            }
        }
        if let Some(max) = max_rating {
            if rating > max { 
                return None;
            }
        }
        
        Some(TacticalPuzzle {
            puzzle_id: fields[0].to_string(),
            fen: fields[1].to_string(),
            moves: fields[2].to_string(),
            rating,
            rating_deviation,
            popularity,
            nb_plays,
            themes: fields[7].to_string(),
            game_url: if fields.len() > 8 { Some(fields[8].to_string()) } else { None },
            opening_tags: if fields.len() > 9 { Some(fields[9].to_string()) } else { None },
        })
    }
    
    /// Convert puzzle to training data
    fn convert_puzzle_to_training_data(puzzle: &TacticalPuzzle) -> Option<TacticalTrainingData> {
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

    /// Load tactical training data into chess engine incrementally (preserves existing data)
    pub fn load_into_engine_incremental(
        tactical_data: &[TacticalTrainingData],
        engine: &mut ChessVectorEngine,
    ) {
        let initial_size = engine.knowledge_base_size();
        let initial_moves = engine.position_moves.len();
        
        // For large datasets, use parallel batch processing
        if tactical_data.len() > 1000 {
            Self::load_into_engine_incremental_parallel(tactical_data, engine, initial_size, initial_moves);
        } else {
            Self::load_into_engine_incremental_sequential(tactical_data, engine, initial_size, initial_moves);
        }
    }
    
    /// Sequential loading for smaller datasets
    fn load_into_engine_incremental_sequential(
        tactical_data: &[TacticalTrainingData],
        engine: &mut ChessVectorEngine,
        initial_size: usize,
        initial_moves: usize,
    ) {
        let pb = ProgressBar::new(tactical_data.len() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} Loading tactical patterns (incremental)")
            .unwrap()
            .progress_chars("#>-"));
        
        let mut added = 0;
        let mut skipped = 0;
        
        for data in tactical_data {
            // Check if this position already exists to avoid duplicates
            if !engine.position_boards.contains(&data.position) {
                engine.add_position_with_move(
                    &data.position,
                    0.0, // Position evaluation (neutral for puzzles)
                    Some(data.solution_move),
                    Some(data.tactical_value), // High tactical value
                );
                added += 1;
            } else {
                skipped += 1;
            }
            pb.inc(1);
        }
        
        pb.finish_with_message(format!(
            "Loaded {} new tactical patterns (skipped {} duplicates, total: {})", 
            added, skipped, engine.knowledge_base_size()
        ));
        
        println!("Incremental tactical training:");
        println!("  - Positions: {} → {} (+{})", initial_size, engine.knowledge_base_size(), engine.knowledge_base_size() - initial_size);
        println!("  - Move entries: {} → {} (+{})", initial_moves, engine.position_moves.len(), engine.position_moves.len() - initial_moves);
    }
    
    /// Parallel batch loading for large datasets
    fn load_into_engine_incremental_parallel(
        tactical_data: &[TacticalTrainingData],
        engine: &mut ChessVectorEngine,
        initial_size: usize,
        initial_moves: usize,
    ) {
        let pb = ProgressBar::new(tactical_data.len() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} Optimized batch loading tactical patterns")
            .unwrap()
            .progress_chars("#>-"));
        
        // Parallel pre-filtering to avoid duplicates (this is thread-safe for read operations)
        let filtered_data: Vec<&TacticalTrainingData> = tactical_data
            .par_iter()
            .filter(|data| !engine.position_boards.contains(&data.position))
            .collect();
        
        let batch_size = 1000; // Larger batches for better performance
        let mut added = 0;
        
        println!("Pre-filtered: {} → {} positions (removed {} duplicates)", 
                 tactical_data.len(), filtered_data.len(), tactical_data.len() - filtered_data.len());
        
        // Process in sequential batches (engine operations aren't thread-safe)
        // But use optimized batch processing
        for batch in filtered_data.chunks(batch_size) {
            let batch_start = added;
            
            for data in batch {
                // Final duplicate check (should be minimal after pre-filtering)
                if !engine.position_boards.contains(&data.position) {
                    engine.add_position_with_move(
                        &data.position,
                        0.0, // Position evaluation (neutral for puzzles) 
                        Some(data.solution_move),
                        Some(data.tactical_value), // High tactical value
                    );
                    added += 1;
                }
                pb.inc(1);
            }
            
            // Update progress message every batch
            pb.set_message(format!("Loaded batch: {} positions", added - batch_start));
        }
        
        let skipped = tactical_data.len() - added;
        
        pb.finish_with_message(format!(
            "Optimized loaded {} new tactical patterns (skipped {} duplicates, total: {})", 
            added, skipped, engine.knowledge_base_size()
        ));
        
        println!("Incremental tactical training (optimized):");
        println!("  - Positions: {} → {} (+{})", initial_size, engine.knowledge_base_size(), engine.knowledge_base_size() - initial_size);
        println!("  - Move entries: {} → {} (+{})", initial_moves, engine.position_moves.len(), engine.position_moves.len() - initial_moves);
        println!("  - Batch size: {}, Pre-filtered efficiency: {:.1}%", 
                 batch_size, (filtered_data.len() as f32 / tactical_data.len() as f32) * 100.0);
    }

    /// Save tactical puzzles to file for incremental loading later
    pub fn save_tactical_puzzles<P: AsRef<std::path::Path>>(
        tactical_data: &[TacticalTrainingData],
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(tactical_data)?;
        std::fs::write(path, json)?;
        println!("Saved {} tactical puzzles", tactical_data.len());
        Ok(())
    }

    /// Load tactical puzzles from file
    pub fn load_tactical_puzzles<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Vec<TacticalTrainingData>, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let tactical_data: Vec<TacticalTrainingData> = serde_json::from_str(&content)?;
        println!("Loaded {} tactical puzzles from file", tactical_data.len());
        Ok(tactical_data)
    }

    /// Save tactical puzzles incrementally (appends to existing file)
    pub fn save_tactical_puzzles_incremental<P: AsRef<std::path::Path>>(
        tactical_data: &[TacticalTrainingData],
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let path = path.as_ref();
        
        if path.exists() {
            // Load existing puzzles
            let mut existing = Self::load_tactical_puzzles(path)?;
            let original_len = existing.len();
            
            // Add new puzzles, checking for duplicates by puzzle ID if available
            for new_puzzle in tactical_data {
                // Check if this puzzle already exists (by position)
                let exists = existing.iter().any(|existing_puzzle| {
                    existing_puzzle.position == new_puzzle.position &&
                    existing_puzzle.solution_move == new_puzzle.solution_move
                });
                
                if !exists {
                    existing.push(new_puzzle.clone());
                }
            }
            
            // Save merged data
            let json = serde_json::to_string_pretty(&existing)?;
            std::fs::write(path, json)?;
            
            println!("Incremental save: added {} new puzzles (total: {})", 
                    existing.len() - original_len, existing.len());
        } else {
            // File doesn't exist, just save normally
            Self::save_tactical_puzzles(tactical_data, path)?;
        }
        Ok(())
    }

    /// Parse Lichess puzzles incrementally (preserves existing engine state)
    pub fn parse_and_load_incremental<P: AsRef<std::path::Path>>(
        file_path: P,
        engine: &mut ChessVectorEngine,
        max_puzzles: Option<usize>,
        min_rating: Option<u32>,
        max_rating: Option<u32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("Parsing Lichess puzzles incrementally...");
        
        // Parse puzzles
        let tactical_data = Self::parse_csv(file_path, max_puzzles, min_rating, max_rating)?;
        
        // Load into engine incrementally
        Self::load_into_engine_incremental(&tactical_data, engine);
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess::Board;
    use std::str::FromStr;

    #[test]
    fn test_training_dataset_creation() {
        let dataset = TrainingDataset::new();
        assert_eq!(dataset.data.len(), 0);
    }

    #[test]
    fn test_add_training_data() {
        let mut dataset = TrainingDataset::new();
        let board = Board::default();
        
        let training_data = TrainingData {
            board,
            evaluation: 0.5,
            depth: 15,
            game_id: 1,
        };
        
        dataset.data.push(training_data);
        assert_eq!(dataset.data.len(), 1);
        assert_eq!(dataset.data[0].evaluation, 0.5);
    }

    #[test]
    fn test_chess_engine_integration() {
        let mut dataset = TrainingDataset::new();
        let board = Board::default();
        
        let training_data = TrainingData {
            board,
            evaluation: 0.3,
            depth: 15,
            game_id: 1,
        };
        
        dataset.data.push(training_data);
        
        let mut engine = ChessVectorEngine::new(1024);
        dataset.train_engine(&mut engine);
        
        assert_eq!(engine.knowledge_base_size(), 1);
        
        let eval = engine.evaluate_position(&board);
        assert!(eval.is_some());
        assert!((eval.unwrap() - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_deduplication() {
        let mut dataset = TrainingDataset::new();
        let board = Board::default();
        
        // Add duplicate positions
        for i in 0..5 {
            let training_data = TrainingData {
                board,
                evaluation: i as f32 * 0.1,
                depth: 15,
                game_id: i,
            };
            dataset.data.push(training_data);
        }
        
        assert_eq!(dataset.data.len(), 5);
        
        // Deduplicate with high threshold (should keep only 1)
        dataset.deduplicate(0.999);
        assert_eq!(dataset.data.len(), 1);
    }

    #[test]
    fn test_dataset_serialization() {
        let mut dataset = TrainingDataset::new();
        let board = Board::from_str("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1").unwrap();
        
        let training_data = TrainingData {
            board,
            evaluation: 0.2,
            depth: 10,
            game_id: 42,
        };
        
        dataset.data.push(training_data);
        
        // Test serialization/deserialization
        let json = serde_json::to_string(&dataset.data).unwrap();
        let loaded_data: Vec<TrainingData> = serde_json::from_str(&json).unwrap();
        let loaded_dataset = TrainingDataset { data: loaded_data };
        
        assert_eq!(loaded_dataset.data.len(), 1);
        assert_eq!(loaded_dataset.data[0].evaluation, 0.2);
        assert_eq!(loaded_dataset.data[0].depth, 10);
        assert_eq!(loaded_dataset.data[0].game_id, 42);
    }

    #[test]
    fn test_tactical_puzzle_processing() {
        let puzzle = TacticalPuzzle {
            puzzle_id: "test123".to_string(),
            fen: "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4".to_string(),
            moves: "Bxf7+ Ke7".to_string(),
            rating: 1500,
            rating_deviation: 100,
            popularity: 150,
            nb_plays: 1000,
            themes: "fork pin".to_string(),
            game_url: None,
            opening_tags: None,
        };
        
        let tactical_data = TacticalPuzzleParser::convert_puzzle_to_training_data(&puzzle);
        assert!(tactical_data.is_some());
        
        let data = tactical_data.unwrap();
        assert_eq!(data.move_theme, "fork");
        assert!(data.tactical_value > 1.0); // Should have high tactical value
        assert!(data.difficulty > 0.0);
    }

    #[test]
    fn test_tactical_puzzle_invalid_fen() {
        let puzzle = TacticalPuzzle {
            puzzle_id: "test123".to_string(),
            fen: "invalid_fen".to_string(),
            moves: "e2e4".to_string(),
            rating: 1500,
            rating_deviation: 100,
            popularity: 150,
            nb_plays: 1000,
            themes: "tactics".to_string(),
            game_url: None,
            opening_tags: None,
        };
        
        let tactical_data = TacticalPuzzleParser::convert_puzzle_to_training_data(&puzzle);
        assert!(tactical_data.is_none());
    }

    #[test]
    fn test_engine_evaluator() {
        let evaluator = EngineEvaluator::new(15);
        
        // Create test dataset
        let mut dataset = TrainingDataset::new();
        let board = Board::default();
        
        let training_data = TrainingData {
            board,
            evaluation: 0.0,
            depth: 15,
            game_id: 1,
        };
        
        dataset.data.push(training_data);
        
        // Create engine with some data
        let mut engine = ChessVectorEngine::new(1024);
        engine.add_position(&board, 0.1);
        
        // Test accuracy evaluation
        let accuracy = evaluator.evaluate_accuracy(&engine, &dataset);
        assert!(accuracy.is_ok());
        assert!(accuracy.unwrap() < 1.0); // Should have some accuracy
    }

    #[test] 
    fn test_tactical_training_integration() {
        let tactical_data = vec![
            TacticalTrainingData {
                position: Board::default(),
                solution_move: ChessMove::from_str("e2e4").unwrap(),
                move_theme: "opening".to_string(),
                difficulty: 1.2,
                tactical_value: 2.5,
            }
        ];
        
        let mut engine = ChessVectorEngine::new(1024);
        TacticalPuzzleParser::load_into_engine(&tactical_data, &mut engine);
        
        assert_eq!(engine.knowledge_base_size(), 1);
        assert_eq!(engine.position_moves.len(), 1);
        
        // Test that tactical move is available in recommendations
        let recommendations = engine.recommend_moves(&Board::default(), 5);
        assert!(!recommendations.is_empty());
    }

    #[test]
    fn test_multithreading_operations() {
        let mut dataset = TrainingDataset::new();
        let board = Board::default();
        
        // Add test data
        for i in 0..10 {
            let training_data = TrainingData {
                board,
                evaluation: i as f32 * 0.1,
                depth: 15,
                game_id: i,
            };
            dataset.data.push(training_data);
        }
        
        // Test parallel deduplication doesn't crash
        dataset.deduplicate_parallel(0.95, 5);
        assert!(dataset.data.len() <= 10);
    }

    #[test]
    fn test_incremental_dataset_operations() {
        let mut dataset1 = TrainingDataset::new();
        let board1 = Board::default();
        let board2 = Board::from_str("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1").unwrap();
        
        // Add initial data
        dataset1.add_position(board1, 0.0, 15, 1);
        dataset1.add_position(board2, 0.2, 15, 2);
        assert_eq!(dataset1.data.len(), 2);
        
        // Create second dataset
        let mut dataset2 = TrainingDataset::new();
        dataset2.add_position(
            Board::from_str("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2").unwrap(),
            0.3,
            15,
            3
        );
        
        // Merge datasets
        dataset1.merge(dataset2);
        assert_eq!(dataset1.data.len(), 3);
        
        // Test next_game_id
        let next_id = dataset1.next_game_id();
        assert_eq!(next_id, 4); // Should be max(1,2,3) + 1
    }

    #[test]
    fn test_save_load_incremental() {
        use tempfile::tempdir;
        
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("incremental_test.json");
        
        // Create and save first dataset
        let mut dataset1 = TrainingDataset::new();
        dataset1.add_position(Board::default(), 0.0, 15, 1);
        dataset1.save(&file_path).unwrap();
        
        // Create second dataset and save incrementally
        let mut dataset2 = TrainingDataset::new();
        dataset2.add_position(
            Board::from_str("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1").unwrap(),
            0.2,
            15,
            2
        );
        dataset2.save_incremental(&file_path).unwrap();
        
        // Load and verify merged data
        let loaded = TrainingDataset::load(&file_path).unwrap();
        assert_eq!(loaded.data.len(), 2);
        
        // Test load_and_append
        let mut dataset3 = TrainingDataset::new();
        dataset3.add_position(
            Board::from_str("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2").unwrap(),
            0.3,
            15,
            3
        );
        dataset3.load_and_append(&file_path).unwrap();
        assert_eq!(dataset3.data.len(), 3); // 1 original + 2 from file
    }

    #[test]
    fn test_add_position_method() {
        let mut dataset = TrainingDataset::new();
        let board = Board::default();
        
        // Test add_position method
        dataset.add_position(board, 0.5, 20, 42);
        assert_eq!(dataset.data.len(), 1);
        assert_eq!(dataset.data[0].evaluation, 0.5);
        assert_eq!(dataset.data[0].depth, 20);
        assert_eq!(dataset.data[0].game_id, 42);
    }

    #[test]
    fn test_incremental_save_deduplication() {
        use tempfile::tempdir;
        
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("dedup_test.json");
        
        // Create and save first dataset
        let mut dataset1 = TrainingDataset::new();
        dataset1.add_position(Board::default(), 0.0, 15, 1);
        dataset1.save(&file_path).unwrap();
        
        // Create second dataset with duplicate position
        let mut dataset2 = TrainingDataset::new();
        dataset2.add_position(Board::default(), 0.1, 15, 2); // Same position, different eval
        dataset2.save_incremental(&file_path).unwrap();
        
        // Should deduplicate and keep only one
        let loaded = TrainingDataset::load(&file_path).unwrap();
        assert_eq!(loaded.data.len(), 1);
    }

    #[test]
    fn test_tactical_puzzle_incremental_loading() {
        let tactical_data = vec![
            TacticalTrainingData {
                position: Board::default(),
                solution_move: ChessMove::from_str("e2e4").unwrap(),
                move_theme: "opening".to_string(),
                difficulty: 1.2,
                tactical_value: 2.5,
            },
            TacticalTrainingData {
                position: Board::from_str("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1").unwrap(),
                solution_move: ChessMove::from_str("e7e5").unwrap(),
                move_theme: "opening".to_string(),
                difficulty: 1.0,
                tactical_value: 2.0,
            }
        ];
        
        let mut engine = ChessVectorEngine::new(1024);
        
        // Add some existing data
        engine.add_position(&Board::default(), 0.1);
        assert_eq!(engine.knowledge_base_size(), 1);
        
        // Load tactical puzzles incrementally
        TacticalPuzzleParser::load_into_engine_incremental(&tactical_data, &mut engine);
        
        // Should have added the new position but skipped the duplicate
        assert_eq!(engine.knowledge_base_size(), 2);
        
        // Should have move data for both puzzles
        assert!(engine.training_stats().has_move_data);
        assert!(engine.training_stats().move_data_entries > 0);
    }

    #[test]
    fn test_tactical_puzzle_serialization() {
        use tempfile::tempdir;
        
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("tactical_test.json");
        
        let tactical_data = vec![
            TacticalTrainingData {
                position: Board::default(),
                solution_move: ChessMove::from_str("e2e4").unwrap(),
                move_theme: "fork".to_string(),
                difficulty: 1.5,
                tactical_value: 3.0,
            }
        ];
        
        // Save tactical puzzles
        TacticalPuzzleParser::save_tactical_puzzles(&tactical_data, &file_path).unwrap();
        
        // Load them back
        let loaded = TacticalPuzzleParser::load_tactical_puzzles(&file_path).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].move_theme, "fork");
        assert_eq!(loaded[0].difficulty, 1.5);
        assert_eq!(loaded[0].tactical_value, 3.0);
    }

    #[test]
    fn test_tactical_puzzle_incremental_save() {
        use tempfile::tempdir;
        
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("incremental_tactical.json");
        
        // Save first batch
        let batch1 = vec![
            TacticalTrainingData {
                position: Board::default(),
                solution_move: ChessMove::from_str("e2e4").unwrap(),
                move_theme: "opening".to_string(),
                difficulty: 1.0,
                tactical_value: 2.0,
            }
        ];
        TacticalPuzzleParser::save_tactical_puzzles(&batch1, &file_path).unwrap();
        
        // Save second batch incrementally
        let batch2 = vec![
            TacticalTrainingData {
                position: Board::from_str("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1").unwrap(),
                solution_move: ChessMove::from_str("e7e5").unwrap(),
                move_theme: "counter".to_string(),
                difficulty: 1.2,
                tactical_value: 2.2,
            }
        ];
        TacticalPuzzleParser::save_tactical_puzzles_incremental(&batch2, &file_path).unwrap();
        
        // Load and verify merged data
        let loaded = TacticalPuzzleParser::load_tactical_puzzles(&file_path).unwrap();
        assert_eq!(loaded.len(), 2);
    }

    #[test]
    fn test_tactical_puzzle_incremental_deduplication() {
        use tempfile::tempdir;
        
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("dedup_tactical.json");
        
        let tactical_data = TacticalTrainingData {
            position: Board::default(),
            solution_move: ChessMove::from_str("e2e4").unwrap(),
            move_theme: "opening".to_string(),
            difficulty: 1.0,
            tactical_value: 2.0,
        };
        
        // Save first time
        TacticalPuzzleParser::save_tactical_puzzles(&[tactical_data.clone()], &file_path).unwrap();
        
        // Try to save the same puzzle again
        TacticalPuzzleParser::save_tactical_puzzles_incremental(&[tactical_data], &file_path).unwrap();
        
        // Should still have only one puzzle (deduplicated)
        let loaded = TacticalPuzzleParser::load_tactical_puzzles(&file_path).unwrap();
        assert_eq!(loaded.len(), 1);
    }
}