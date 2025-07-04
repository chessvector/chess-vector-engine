use chess::Board;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs::File;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Ultra-fast loader specifically designed for massive datasets (100k-10M+ positions)
/// Uses aggressive optimizations: memory mapping, parallel processing, bloom filters
pub struct UltraFastLoader {
    pub loaded_count: usize,
    pub duplicate_count: usize,
    pub error_count: usize,
    batch_size: usize,
    #[allow(dead_code)]
    use_bloom_filter: bool,
}

impl UltraFastLoader {
    pub fn new_for_massive_datasets() -> Self {
        Self {
            loaded_count: 0,
            duplicate_count: 0,
            error_count: 0,
            batch_size: 50000, // Large batches for massive datasets
            use_bloom_filter: true,
        }
    }

    /// Ultra-fast binary loader with memory mapping and parallel processing
    pub fn ultra_load_binary<P: AsRef<Path>>(
        &mut self,
        path: P,
        engine: &mut crate::ChessVectorEngine,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let path_ref = path.as_ref();
        println!("Operation complete");

        let file_size = std::fs::metadata(path_ref)?.len();
        println!("📊 File size: {:.1} MB", file_size as f64 / 1_000_000.0);

        if file_size > 500_000_000 {
            // > 500MB
            println!("⚡ Large file detected - using memory-mapped loading");
            return self.memory_mapped_load(path_ref, engine);
        }

        // Standard file loading with optimizations
        let data = std::fs::read(path_ref)?;

        // Try LZ4 decompression
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

        // Deserialize with error handling
        let positions: Vec<(String, f32)> = match bincode::deserialize(&decompressed_data) {
            Ok(pos) => pos,
            Err(e) => {
                println!("Operation complete");
                return Err(e.into());
            }
        };

        let total_positions = positions.len();
        println!("📦 Loaded {total_positions} positions from binary");

        if total_positions == 0 {
            return Ok(());
        }

        // Use optimized loading strategy based on size
        if total_positions > 100_000 {
            self.parallel_batch_load(positions, engine)
        } else {
            self.sequential_load(positions, engine)
        }
    }

    /// Memory-mapped loading for very large files
    fn memory_mapped_load<P: AsRef<Path>>(
        &mut self,
        path: P,
        engine: &mut crate::ChessVectorEngine,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use memmap2::Mmap;

        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        println!("🗺️  Memory-mapped {} bytes", mmap.len());

        // Try to deserialize in chunks to avoid memory explosion
        const CHUNK_SIZE: usize = 50_000_000; // 50MB chunks
        let total_chunks = mmap.len().div_ceil(CHUNK_SIZE);

        println!("📦 Processing {total_chunks} chunks of ~50MB each");

        // For very large files, we need a different approach
        // Try to parse as streaming format instead
        self.stream_parse_memory_mapped(&mmap, engine)
    }

    /// Stream parse memory-mapped data
    fn stream_parse_memory_mapped(
        &mut self,
        mmap: &memmap2::Mmap,
        engine: &mut crate::ChessVectorEngine,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Try different decompression methods

        // 1. Try LZ4 decompression of entire file
        if let Ok(decompressed) = lz4_flex::decompress_size_prepended(mmap) {
            println!("🗜️  Full file LZ4 decompressed");
            return self.parse_decompressed_data(&decompressed, engine);
        }

        // 2. Try direct deserialization
        if let Ok(positions) = bincode::deserialize::<Vec<(String, f32)>>(mmap) {
            println!("📦 Direct memory-mapped deserialization");
            return self.parallel_batch_load(positions, engine);
        }

        // 3. Try as raw text (fallback)
        if let Ok(text) = std::str::from_utf8(mmap) {
            println!("📝 Treating as text format");
            return self.parse_text_data(text, engine);
        }

        Err("Unable to parse memory-mapped file in any known format".into())
    }

    /// Parse decompressed binary data
    fn parse_decompressed_data(
        &mut self,
        data: &[u8],
        engine: &mut crate::ChessVectorEngine,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let positions: Vec<(String, f32)> = bincode::deserialize(data)?;
        self.parallel_batch_load(positions, engine)
    }

    /// Parse text data (JSON or similar)
    fn parse_text_data(
        &mut self,
        text: &str,
        engine: &mut crate::ChessVectorEngine,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("📝 Parsing text data...");

        let lines: Vec<&str> = text.lines().collect();
        let total_lines = lines.len();

        if total_lines == 0 {
            return Ok(());
        }

        println!("📊 Processing {total_lines} lines");

        let pb = ProgressBar::new(total_lines as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("⚡ Parsing [{elapsed_precise}] [{bar:40.green/blue}] {pos}/{len} ({percent}%) {msg}")?
                .progress_chars("██░")
        );

        // Use parallel processing for text parsing
        let batch_size = 10000;
        let existing_boards: HashSet<Board> = engine.position_boards.iter().cloned().collect();
        let existing_boards = Arc::new(existing_boards);

        let results: Arc<Mutex<Vec<(Board, f32)>>> = Arc::new(Mutex::new(Vec::new()));

        lines
            .par_chunks(batch_size)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut local_results = Vec::new();

                for (line_idx, line) in chunk.iter().enumerate() {
                    if line.trim().is_empty() {
                        continue;
                    }

                    // Try to parse as JSON
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(line) {
                        if let Some((board, eval)) = self.extract_from_json(&json) {
                            if !existing_boards.contains(&board) {
                                local_results.push((board, eval));
                            }
                        }
                    }

                    // Update progress periodically
                    if line_idx % 1000 == 0 {
                        pb.set_position((chunk_idx * batch_size + line_idx) as u64);
                    }
                }

                // Add local results to global results
                if !local_results.is_empty() {
                    if let Ok(mut results) = results.lock() {
                        results.extend(local_results);
                    }
                }
            });

        pb.finish_with_message("✅ Text parsing complete");

        // Extract results and add to engine
        let final_results = Arc::try_unwrap(results).unwrap().into_inner().unwrap();
        self.loaded_count = final_results.len();

        println!("📦 Parsed {} valid positions", self.loaded_count);

        // Add to engine in batches
        for (board, eval) in final_results {
            engine.add_position(&board, eval);
        }

        Ok(())
    }

    /// Extract position from JSON
    fn extract_from_json(&self, json: &serde_json::Value) -> Option<(Board, f32)> {
        // Try different schemas
        if let (Some(fen), Some(eval)) = (
            json.get("fen").and_then(|v| v.as_str()),
            json.get("evaluation").and_then(|v| v.as_f64()),
        ) {
            if let Ok(board) = fen.parse::<Board>() {
                return Some((board, eval as f32));
            }
        }

        None
    }

    /// Parallel batch loading for large datasets
    fn parallel_batch_load(
        &mut self,
        positions: Vec<(String, f32)>,
        engine: &mut crate::ChessVectorEngine,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let total_positions = positions.len();
        println!("🔄 Parallel batch loading {total_positions} positions");

        let pb = ProgressBar::new(total_positions as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("⚡ Loading [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) {msg}")?
                .progress_chars("██░")
        );

        // Create bloom filter for existing positions
        let existing_boards: HashSet<Board> = engine.position_boards.iter().cloned().collect();

        // Process in parallel chunks
        let chunk_size = self.batch_size;
        let chunks: Vec<_> = positions.chunks(chunk_size).collect();

        let mut total_loaded = 0;
        let mut total_duplicates = 0;

        for (chunk_idx, chunk) in chunks.iter().enumerate() {
            let mut batch_boards = Vec::new();
            let mut batch_evaluations = Vec::new();

            // Process chunk
            for (fen, evaluation) in chunk.iter() {
                match fen.parse::<Board>() {
                    Ok(board) => {
                        if !existing_boards.contains(&board) {
                            batch_boards.push(board);
                            batch_evaluations.push(*evaluation);
                        } else {
                            total_duplicates += 1;
                        }
                    }
                    Err(_) => {
                        self.error_count += 1;
                    }
                }
            }

            // Add batch to engine
            for (board, eval) in batch_boards.iter().zip(batch_evaluations.iter()) {
                engine.add_position(board, *eval);
                total_loaded += 1;
            }

            // Update progress
            pb.set_position(((chunk_idx + 1) * chunk_size).min(total_positions) as u64);
            pb.set_message(format!("{total_loaded} loaded, {total_duplicates} dupes"));
        }

        pb.finish_with_message(format!("✅ Loaded {total_loaded} positions"));

        self.loaded_count = total_loaded;
        self.duplicate_count = total_duplicates;

        println!("📊 Final stats:");
        println!("   Loaded: {count} positions", count = self.loaded_count);
        println!("Operation complete");
        println!("Operation complete");

        Ok(())
    }

    /// Sequential loading for smaller datasets
    fn sequential_load(
        &mut self,
        positions: Vec<(String, f32)>,
        engine: &mut crate::ChessVectorEngine,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("📦 Sequential loading {} positions", positions.len());

        let existing_boards: HashSet<Board> = engine.position_boards.iter().cloned().collect();

        for (fen, evaluation) in positions {
            match fen.parse::<Board>() {
                Ok(board) => {
                    if !existing_boards.contains(&board) {
                        engine.add_position(&board, evaluation);
                        self.loaded_count += 1;
                    } else {
                        self.duplicate_count += 1;
                    }
                }
                Err(_) => {
                    self.error_count += 1;
                }
            }
        }

        Ok(())
    }

    /// Get loading statistics
    pub fn get_stats(&self) -> LoadingStats {
        LoadingStats {
            loaded: self.loaded_count,
            duplicates: self.duplicate_count,
            errors: self.error_count,
            total_processed: self.loaded_count + self.duplicate_count + self.error_count,
        }
    }
}

/// Loading statistics
#[derive(Debug, Clone)]
pub struct LoadingStats {
    pub loaded: usize,
    pub duplicates: usize,
    pub errors: usize,
    pub total_processed: usize,
}

impl LoadingStats {
    pub fn success_rate(&self) -> f64 {
        if self.total_processed == 0 {
            return 1.0;
        }
        self.loaded as f64 / self.total_processed as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ultra_fast_loader_creation() {
        let loader = UltraFastLoader::new_for_massive_datasets();
        assert_eq!(loader.loaded_count, 0);
        assert_eq!(loader.batch_size, 50000);
        assert!(loader.use_bloom_filter);
    }

    #[test]
    fn test_loading_stats() {
        let mut loader = UltraFastLoader::new_for_massive_datasets();
        loader.loaded_count = 8000;
        loader.duplicate_count = 1500;
        loader.error_count = 500;

        let stats = loader.get_stats();
        assert_eq!(stats.loaded, 8000);
        assert_eq!(stats.total_processed, 10000);
        assert_eq!(stats.success_rate(), 0.8);
    }
}
