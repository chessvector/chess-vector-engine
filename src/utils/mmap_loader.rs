use crate::errors::ChessEngineError;
use memmap2::Mmap;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Result};
use std::path::Path;

/// Memory-mapped file loader for ultra-fast loading of large datasets
pub struct MmapLoader {
    file: File,
    mmap: Mmap,
}

impl MmapLoader {
    /// Create a new memory-mapped loader for a file
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        Ok(Self { file, mmap })
    }

    /// Get the raw memory-mapped data
    pub fn data(&self) -> &[u8] {
        &self.mmap
    }

    /// Get the size of the mapped file
    pub fn size(&self) -> usize {
        self.mmap.len()
    }

    /// Load binary data from a specific offset
    pub fn load_at_offset<T>(&self, offset: usize) -> Result<&T>
    where
        T: Sized,
    {
        let size = std::mem::size_of::<T>();
        if offset + size > self.size() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Offset exceeds file size",
            ));
        }

        let ptr = unsafe { self.mmap.as_ptr().add(offset) as *const T };
        Ok(unsafe { &*ptr })
    }

    /// Load a slice of data from a specific offset
    pub fn load_slice_at_offset<T>(&self, offset: usize, count: usize) -> Result<&[T]>
    where
        T: Sized,
    {
        let size = std::mem::size_of::<T>() * count;
        if offset + size > self.size() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Offset and size exceed file size",
            ));
        }

        let ptr = unsafe { self.mmap.as_ptr().add(offset) as *const T };
        Ok(unsafe { std::slice::from_raw_parts(ptr, count) })
    }
}

/// Fast position loader using memory-mapped files
pub struct FastPositionLoader;

impl FastPositionLoader {
    /// Load positions from a binary file using memory mapping
    pub fn load_positions_mmap<P: AsRef<Path>>(
        path: P,
    ) -> crate::errors::Result<Vec<(Array1<f32>, f32)>> {
        let loader = MmapLoader::new(path)
            .map_err(|e| ChessEngineError::IoError(format!("Failed to memory-map file: {}", e)))?;

        let data = loader.data();
        if data.len() < 8 {
            return Err(ChessEngineError::IoError(
                "File too small to contain header".to_string(),
            ));
        }

        // Read header: [version: u32, count: u32]
        let version = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let count = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);

        if version != 1 {
            return Err(ChessEngineError::IoError(format!(
                "Unsupported version: {}",
                version
            )));
        }

        let mut positions = Vec::with_capacity(count as usize);
        let mut offset = 8;

        for _ in 0..count {
            // Read vector size
            if offset + 4 > data.len() {
                return Err(ChessEngineError::IoError(
                    "Unexpected end of file".to_string(),
                ));
            }
            let vector_size = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
            offset += 4;

            // Read vector data
            let vector_bytes = vector_size * 4; // 4 bytes per f32
            if offset + vector_bytes > data.len() {
                return Err(ChessEngineError::IoError(
                    "Unexpected end of file".to_string(),
                ));
            }

            let vector_data = loader
                .load_slice_at_offset::<f32>(offset, vector_size)
                .map_err(|e| {
                    ChessEngineError::IoError(format!("Failed to load vector data: {}", e))
                })?;

            let vector = Array1::from_vec(vector_data.to_vec());
            offset += vector_bytes;

            // Read evaluation
            if offset + 4 > data.len() {
                return Err(ChessEngineError::IoError(
                    "Unexpected end of file".to_string(),
                ));
            }
            let evaluation = f32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            offset += 4;

            positions.push((vector, evaluation));
        }

        Ok(positions)
    }

    /// Save positions to a binary file for fast loading
    pub fn save_positions_binary<P: AsRef<Path>>(
        path: P,
        positions: &[(Array1<f32>, f32)],
    ) -> crate::errors::Result<()> {
        use std::fs::OpenOptions;
        use std::io::Write;

        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)
            .map_err(|e| ChessEngineError::IoError(format!("Failed to create file: {}", e)))?;

        // Write header: [version: u32, count: u32]
        let version = 1u32;
        let count = positions.len() as u32;
        file.write_all(&version.to_le_bytes())
            .map_err(|e| ChessEngineError::IoError(format!("Failed to write header: {}", e)))?;
        file.write_all(&count.to_le_bytes())
            .map_err(|e| ChessEngineError::IoError(format!("Failed to write count: {}", e)))?;

        for (vector, evaluation) in positions {
            // Write vector size
            let vector_size = vector.len() as u32;
            file.write_all(&vector_size.to_le_bytes()).map_err(|e| {
                ChessEngineError::IoError(format!("Failed to write vector size: {}", e))
            })?;

            // Write vector data
            let vector_bytes = unsafe {
                std::slice::from_raw_parts(vector.as_ptr() as *const u8, vector.len() * 4)
            };
            file.write_all(vector_bytes).map_err(|e| {
                ChessEngineError::IoError(format!("Failed to write vector data: {}", e))
            })?;

            // Write evaluation
            file.write_all(&evaluation.to_le_bytes()).map_err(|e| {
                ChessEngineError::IoError(format!("Failed to write evaluation: {}", e))
            })?;
        }

        file.flush()
            .map_err(|e| ChessEngineError::IoError(format!("Failed to flush file: {}", e)))?;

        Ok(())
    }

    /// Load positions from JSON file with streaming for large files
    pub fn load_positions_json_streaming<P: AsRef<Path>>(
        path: P,
    ) -> crate::errors::Result<Vec<(Array1<f32>, f32)>> {
        let file = File::open(path)
            .map_err(|e| ChessEngineError::IoError(format!("Failed to open file: {}", e)))?;

        let reader = BufReader::new(file);
        let mut positions = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| {
                ChessEngineError::IoError(format!("Failed to read line {}: {}", line_num, e))
            })?;

            // Skip empty lines and comments
            if line.trim().is_empty() || line.trim().starts_with('#') {
                continue;
            }

            let position: JsonPosition = serde_json::from_str(&line).map_err(|e| {
                ChessEngineError::IoError(format!(
                    "Failed to parse JSON at line {}: {}",
                    line_num, e
                ))
            })?;

            let vector = Array1::from_vec(position.vector);
            positions.push((vector, position.evaluation));
        }

        Ok(positions)
    }
}

/// JSON position format for streaming
#[derive(Debug, Serialize, Deserialize)]
struct JsonPosition {
    vector: Vec<f32>,
    evaluation: f32,
}

/// Chunked file loader for processing large files in chunks
pub struct ChunkedLoader {
    chunk_size: usize,
}

impl ChunkedLoader {
    /// Create a new chunked loader with specified chunk size
    pub fn new(chunk_size: usize) -> Self {
        Self { chunk_size }
    }

    /// Process a large file in chunks to avoid memory issues
    pub fn process_in_chunks<P, F, R>(
        &self,
        path: P,
        mut processor: F,
    ) -> crate::errors::Result<Vec<R>>
    where
        P: AsRef<Path>,
        F: FnMut(&[(Array1<f32>, f32)]) -> crate::errors::Result<R>,
    {
        let loader = MmapLoader::new(path)
            .map_err(|e| ChessEngineError::IoError(format!("Failed to memory-map file: {}", e)))?;

        let data = loader.data();
        if data.len() < 8 {
            return Err(ChessEngineError::IoError(
                "File too small to contain header".to_string(),
            ));
        }

        // Read header
        let version = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let count = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);

        if version != 1 {
            return Err(ChessEngineError::IoError(format!(
                "Unsupported version: {}",
                version
            )));
        }

        let mut results = Vec::new();
        let mut offset = 8;
        let mut processed = 0;

        while processed < count {
            let chunk_end = ((processed + self.chunk_size as u32).min(count)) as usize;
            let chunk_count = chunk_end - processed as usize;

            let mut chunk = Vec::with_capacity(chunk_count);

            // Load chunk
            for _ in 0..chunk_count {
                // Read vector size
                if offset + 4 > data.len() {
                    return Err(ChessEngineError::IoError(
                        "Unexpected end of file".to_string(),
                    ));
                }
                let vector_size = u32::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ]) as usize;
                offset += 4;

                // Read vector data
                let vector_bytes = vector_size * 4;
                if offset + vector_bytes > data.len() {
                    return Err(ChessEngineError::IoError(
                        "Unexpected end of file".to_string(),
                    ));
                }

                let vector_data = loader
                    .load_slice_at_offset::<f32>(offset, vector_size)
                    .map_err(|e| {
                        ChessEngineError::IoError(format!("Failed to load vector data: {}", e))
                    })?;

                let vector = Array1::from_vec(vector_data.to_vec());
                offset += vector_bytes;

                // Read evaluation
                if offset + 4 > data.len() {
                    return Err(ChessEngineError::IoError(
                        "Unexpected end of file".to_string(),
                    ));
                }
                let evaluation = f32::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ]);
                offset += 4;

                chunk.push((vector, evaluation));
            }

            // Process chunk
            let result = processor(&chunk)?;
            results.push(result);

            processed += chunk_count as u32;
        }

        Ok(results)
    }
}

/// Compressed file loader using various compression formats
pub struct CompressedLoader;

impl CompressedLoader {
    /// Load from a compressed file (supports gzip, zstd, lz4)
    pub fn load_compressed<P: AsRef<Path>>(
        path: P,
    ) -> crate::errors::Result<Vec<(Array1<f32>, f32)>> {
        let path = path.as_ref();
        let extension = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");

        match extension {
            "gz" => Self::load_gzip(path),
            "zst" => Self::load_zstd(path),
            "lz4" => Self::load_lz4(path),
            _ => Err(ChessEngineError::IoError(format!(
                "Unsupported compression format: {}",
                extension
            ))),
        }
    }

    /// Load from gzip compressed file
    fn load_gzip<P: AsRef<Path>>(_path: P) -> crate::errors::Result<Vec<(Array1<f32>, f32)>> {
        // Note: flate2 not available in current dependencies
        // This is a placeholder for future implementation
        Err(ChessEngineError::IoError(
            "Gzip support not implemented".to_string(),
        ))
        /*
        use flate2::read::GzDecoder;

        let _file = File::open(path)
            .map_err(|e| ChessEngineError::IoError(format!("Failed to open gzip file: {}", e)))?;

        // Implementation would go here once flate2 is available
        return Err(ChessEngineError::IoError("Gzip support not implemented".to_string()));
        */
    }

    /// Load from zstd compressed file  
    fn load_zstd<P: AsRef<Path>>(_path: P) -> crate::errors::Result<Vec<(Array1<f32>, f32)>> {
        // TODO: Implement zstd loading when zstd crate is available
        Err(ChessEngineError::IoError(
            "Zstd support not implemented".to_string(),
        ))
    }

    /// Load from lz4 compressed file
    fn load_lz4<P: AsRef<Path>>(_path: P) -> crate::errors::Result<Vec<(Array1<f32>, f32)>> {
        // TODO: Implement lz4 loading when lz4 crate is available
        Err(ChessEngineError::IoError(
            "LZ4 support not implemented".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_binary_save_load() {
        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path();

        // Create test data
        let original_positions = vec![
            (Array1::from_vec(vec![1.0, 2.0, 3.0]), 0.5),
            (Array1::from_vec(vec![4.0, 5.0, 6.0]), -0.3),
            (Array1::from_vec(vec![7.0, 8.0, 9.0]), 0.1),
        ];

        // Save to binary file
        FastPositionLoader::save_positions_binary(temp_path, &original_positions).unwrap();

        // Load from binary file
        let loaded_positions = FastPositionLoader::load_positions_mmap(temp_path).unwrap();

        // Verify
        assert_eq!(loaded_positions.len(), original_positions.len());
        for (loaded, original) in loaded_positions.iter().zip(original_positions.iter()) {
            assert_eq!(loaded.0.len(), original.0.len());
            for (l, o) in loaded.0.iter().zip(original.0.iter()) {
                assert!((l - o).abs() < 1e-6);
            }
            assert!((loaded.1 - original.1).abs() < 1e-6);
        }
    }

    #[test]
    fn test_chunked_processing() {
        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path();

        // Create test data
        let original_positions = vec![
            (Array1::from_vec(vec![1.0, 2.0]), 0.5),
            (Array1::from_vec(vec![3.0, 4.0]), -0.3),
            (Array1::from_vec(vec![5.0, 6.0]), 0.1),
            (Array1::from_vec(vec![7.0, 8.0]), 0.7),
        ];

        // Save to binary file
        FastPositionLoader::save_positions_binary(temp_path, &original_positions).unwrap();

        // Process in chunks of 2
        let chunked_loader = ChunkedLoader::new(2);
        let results = chunked_loader
            .process_in_chunks(temp_path, |chunk| Ok(chunk.len()))
            .unwrap();

        // Should have 2 chunks of 2 items each
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], 2);
        assert_eq!(results[1], 2);
    }

    #[test]
    fn test_mmap_loader() {
        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path();

        // Write some test data
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .open(temp_path)
            .unwrap();

        let test_data = [1u32, 2u32, 3u32, 4u32];
        for value in &test_data {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
        file.flush().unwrap();
        drop(file);

        // Test memory mapping
        let loader = MmapLoader::new(temp_path).unwrap();
        assert_eq!(loader.size(), 16); // 4 * 4 bytes

        // Test loading values
        let value1 = loader.load_at_offset::<u32>(0).unwrap();
        assert_eq!(*value1, 1);

        let value2 = loader.load_at_offset::<u32>(4).unwrap();
        assert_eq!(*value2, 2);

        // Test loading slice
        let slice = loader.load_slice_at_offset::<u32>(0, 4).unwrap();
        assert_eq!(slice, &test_data);
    }
}
