use chess::Board;
use indicatif::{ProgressBar, ProgressStyle};
use serde_json;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Ultra-fast streaming loader for massive datasets
/// Optimized for loading 100k-1M+ positions efficiently
pub struct StreamingLoader {
    pub loaded_count: usize,
    pub duplicate_count: usize,
    pub total_processed: usize,
}

impl StreamingLoader {
    pub fn new() -> Self {
        Self {
            loaded_count: 0,
            duplicate_count: 0,
            total_processed: 0,
        }
    }

    /// Stream-load massive JSON files with minimal memory usage
    /// Uses streaming JSON parser and batched processing
    pub fn stream_load_json<P: AsRef<Path>>(
        &mut self,
        path: P,
        engine: &mut crate::ChessVectorEngine,
        batch_size: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let path_ref = path.as_ref();
        println!("Operation complete");

        let file = File::open(path_ref)?;
        let reader = BufReader::with_capacity(64 * 1024, file); // 64KB buffer

        // Estimate total lines for progress tracking
        let total_lines = self.estimate_line_count(path_ref)?;
        println!("📊 Estimated {total_lines} lines to process");

        let pb = ProgressBar::new(total_lines as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("⚡ Streaming [{elapsed_precise}] [{bar:40.green/blue}] {pos}/{len} ({percent}%) {msg}")?
                .progress_chars("██░")
        );

        // Use existing positions as a bloom filter approximation
        let existing_boards: HashSet<Board> = engine.position_boards.iter().cloned().collect();
        let initial_size = existing_boards.len();

        // Batch processing variables
        let mut batch_boards = Vec::with_capacity(batch_size);
        let mut batch_evaluations = Vec::with_capacity(batch_size);
        let mut line_count = 0;

        // Stream process each line
        for line_result in reader.lines() {
            let line = line_result?;
            line_count += 1;

            if line.trim().is_empty() {
                continue;
            }

            // Parse JSON line
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line) {
                if let Some((board, evaluation)) = self.extract_position_data(&json)? {
                    // Quick duplicate check (not perfect but fast)
                    if !existing_boards.contains(&board) {
                        batch_boards.push(board);
                        batch_evaluations.push(evaluation);

                        // Process batch when full
                        if batch_boards.len() >= batch_size {
                            self.process_batch(engine, &mut batch_boards, &mut batch_evaluations)?;

                            pb.set_message(format!(
                                "{loaded} loaded, {dupes} dupes",
                                loaded = self.loaded_count,
                                dupes = self.duplicate_count
                            ));
                        }
                    } else {
                        self.duplicate_count += 1;
                    }
                }
            }

            self.total_processed += 1;

            // Update progress every 1000 lines
            if line_count % 1000 == 0 {
                pb.set_position(line_count as u64);
            }
        }

        // Process remaining batch
        if !batch_boards.is_empty() {
            self.process_batch(engine, &mut batch_boards, &mut batch_evaluations)?;
        }

        pb.finish_with_message(format!(
            "✅ Complete: {} loaded, {} duplicates from {} lines",
            self.loaded_count, self.duplicate_count, line_count
        ));

        let new_positions = engine.position_boards.len() - initial_size;
        println!("🎯 Added {new_positions} new positions to engine");

        Ok(())
    }

    /// Ultra-fast binary format streaming loader
    /// For pre-processed binary training data
    pub fn stream_load_binary<P: AsRef<Path>>(
        &mut self,
        path: P,
        engine: &mut crate::ChessVectorEngine,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let path_ref = path.as_ref();
        println!("Operation complete");

        // Load binary data
        let data = std::fs::read(path_ref)?;
        println!("📦 Read {} bytes", data.len());

        // Try LZ4 decompression first
        let decompressed_data = if let Ok(decompressed) = lz4_flex::decompress_size_prepended(&data)
        {
            println!(
                "🗜️  LZ4 decompressed: {} → {} bytes",
                data.len(),
                decompressed.len()
            );
            decompressed
        } else {
            data
        };

        // Deserialize positions
        let positions: Vec<(String, f32)> = bincode::deserialize(&decompressed_data)?;
        let total_positions = positions.len();
        println!("📊 Loaded {total_positions} positions from binary");

        if total_positions == 0 {
            return Ok(());
        }

        let pb = ProgressBar::new(total_positions as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("⚡ Binary loading [{elapsed_precise}] [{bar:40.blue/green}] {pos}/{len} ({percent}%) {msg}")?
                .progress_chars("██░")
        );

        // Use existing positions for duplicate detection
        let existing_boards: HashSet<Board> = engine.position_boards.iter().cloned().collect();

        // Process in large batches for efficiency
        const BATCH_SIZE: usize = 10000;
        let mut processed = 0;

        for chunk in positions.chunks(BATCH_SIZE) {
            let mut batch_boards = Vec::with_capacity(BATCH_SIZE);
            let mut batch_evaluations = Vec::with_capacity(BATCH_SIZE);

            for (fen, evaluation) in chunk {
                if let Ok(board) = fen.parse::<Board>() {
                    if !existing_boards.contains(&board) {
                        batch_boards.push(board);
                        batch_evaluations.push(*evaluation);
                    } else {
                        self.duplicate_count += 1;
                    }
                }
                processed += 1;
            }

            // Add batch to engine
            if !batch_boards.is_empty() {
                self.process_batch(engine, &mut batch_boards, &mut batch_evaluations)?;
            }

            pb.set_position(processed as u64);
            pb.set_message(format!("{count} loaded", count = self.loaded_count));
        }

        pb.finish_with_message(format!(
            "✅ Loaded {count} positions",
            count = self.loaded_count
        ));

        Ok(())
    }

    /// Process a batch of positions efficiently
    fn process_batch(
        &mut self,
        engine: &mut crate::ChessVectorEngine,
        boards: &mut Vec<Board>,
        evaluations: &mut Vec<f32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Add all positions in batch
        for (board, evaluation) in boards.iter().zip(evaluations.iter()) {
            engine.add_position(board, *evaluation);
            self.loaded_count += 1;
        }

        // Clear for next batch
        boards.clear();
        evaluations.clear();

        Ok(())
    }

    /// Extract position data from JSON value
    fn extract_position_data(
        &self,
        json: &serde_json::Value,
    ) -> Result<Option<(Board, f32)>, Box<dyn std::error::Error>> {
        // Try different JSON schemas
        if let (Some(fen), Some(eval)) = (
            json.get("fen").and_then(|v| v.as_str()),
            json.get("evaluation").and_then(|v| v.as_f64()),
        ) {
            if let Ok(board) = fen.parse::<Board>() {
                return Ok(Some((board, eval as f32)));
            }
        }

        if let (Some(fen), Some(eval)) = (
            json.get("board").and_then(|v| v.as_str()),
            json.get("eval").and_then(|v| v.as_f64()),
        ) {
            if let Ok(board) = fen.parse::<Board>() {
                return Ok(Some((board, eval as f32)));
            }
        }

        if let (Some(fen), Some(eval)) = (
            json.get("position").and_then(|v| v.as_str()),
            json.get("score").and_then(|v| v.as_f64()),
        ) {
            if let Ok(board) = fen.parse::<Board>() {
                return Ok(Some((board, eval as f32)));
            }
        }

        Ok(None)
    }

    /// Estimate line count for progress tracking
    fn estimate_line_count<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        use std::io::Read;

        let path_ref = path.as_ref();
        let file = File::open(path_ref)?;
        let mut reader = BufReader::new(file);

        // Sample first 1MB to estimate
        let mut sample = vec![0u8; 1024 * 1024];
        let bytes_read = reader.read(&mut sample)?;

        if bytes_read == 0 {
            return Ok(0);
        }

        // Count newlines in sample
        let newlines_in_sample = sample[..bytes_read].iter().filter(|&&b| b == b'\n').count();

        // Get total file size by re-opening the file
        let total_size = std::fs::metadata(path_ref)?.len() as usize;

        if bytes_read >= total_size {
            // We read the whole file
            return Ok(newlines_in_sample);
        }

        // Estimate based on sample
        let estimated_lines = (newlines_in_sample * total_size) / bytes_read;
        Ok(estimated_lines)
    }
}

impl Default for StreamingLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_json_extraction() {
        let loader = StreamingLoader::new();

        let json = serde_json::json!({
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "evaluation": 0.25
        });

        let result = loader.extract_position_data(&json).unwrap();
        assert!(result.is_some());

        let (board, eval) = result.unwrap();
        assert_eq!(board, Board::default());
        assert_eq!(eval, 0.25);
    }

    #[test]
    fn test_line_estimation() {
        let loader = StreamingLoader::new();

        // Create a temporary file with known line count
        let mut temp_file = NamedTempFile::new().unwrap();
        for _i in 0..100 {
            writeln!(temp_file, "Loading complete").unwrap();
        }

        let estimated = loader.estimate_line_count(temp_file.path()).unwrap();
        // Should be approximately 100 (within reasonable range)
        assert!((80..=120).contains(&estimated));
    }
}
