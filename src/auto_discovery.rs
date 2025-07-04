use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// File format priority (lower = better)
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum FormatPriority {
    MemoryMapped = 1,
    MessagePack = 2,
    Binary = 3,
    Zstd = 4,
    Json = 5,
}

/// Training data file information
#[derive(Debug, Clone)]
pub struct TrainingFile {
    pub path: PathBuf,
    pub format: String,
    pub priority: FormatPriority,
    pub base_name: String,
    pub size_bytes: u64,
}

/// Auto-discovery and format consolidation engine
pub struct AutoDiscovery;

impl AutoDiscovery {
    /// Discover all training data files in a directory
    pub fn discover_training_files<P: AsRef<Path>>(
        base_path: P,
        recursive: bool,
    ) -> Result<Vec<TrainingFile>, Box<dyn std::error::Error>> {
        let mut discovered_files = Vec::new();
        let base_path = base_path.as_ref();

        println!(
            "🔍 Discovering training data files in {}...",
            base_path.display()
        );

        Self::scan_directory(base_path, recursive, &mut discovered_files)?;

        // Sort by base name, then by priority
        discovered_files.sort_by(|a, b| {
            a.base_name
                .cmp(&b.base_name)
                .then_with(|| a.priority.cmp(&b.priority))
        });

        println!(
            "📁 Discovered {} training data files",
            discovered_files.len()
        );
        for file in &discovered_files {
            println!(
                "   {} - {} ({})",
                file.format,
                file.path.display(),
                Self::format_bytes(file.size_bytes)
            );
        }

        Ok(discovered_files)
    }

    /// Group files by base name and select best format for each
    pub fn consolidate_by_base_name(files: Vec<TrainingFile>) -> HashMap<String, TrainingFile> {
        let mut groups: HashMap<String, Vec<TrainingFile>> = HashMap::new();

        // Group by base name
        for file in files {
            groups.entry(file.base_name.clone()).or_default().push(file);
        }

        let mut consolidated = HashMap::new();

        // Select best format for each group
        for (base_name, mut group) in groups {
            group.sort_by(|a, b| a.priority.cmp(&b.priority));

            if let Some(best_file) = group.into_iter().next() {
                consolidated.insert(base_name, best_file);
            }
        }

        consolidated
    }

    /// Get list of inferior formats that can be cleaned up
    pub fn get_cleanup_candidates(files: &[TrainingFile]) -> Vec<PathBuf> {
        let mut cleanup_files = Vec::new();
        let consolidated = Self::consolidate_by_base_name(files.to_vec());

        for file in files {
            if let Some(best_file) = consolidated.get(&file.base_name) {
                // If this file is not the best format for its base name, mark for cleanup
                if file.path != best_file.path && file.priority > best_file.priority {
                    cleanup_files.push(file.path.clone());
                }
            }
        }

        cleanup_files
    }

    /// Clean up old format files
    pub fn cleanup_old_formats(
        files_to_remove: &[PathBuf],
        dry_run: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if dry_run {
            println!(
                "🧹 DRY RUN - Would remove {} old format files:",
                files_to_remove.len()
            );
            for _path in files_to_remove {
                println!("Discovery complete");
            }
            return Ok(());
        }

        println!(
            "🧹 Cleaning up {} old format files...",
            files_to_remove.len()
        );

        for path in files_to_remove {
            match fs::remove_file(path) {
                Ok(()) => println!("Removed file: {}", path.display()),
                Err(e) => println!("Error removing file: {e}"),
            }
        }

        Ok(())
    }

    /// Scan directory recursively for training files
    fn scan_directory(
        dir: &Path,
        recursive: bool,
        files: &mut Vec<TrainingFile>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !dir.is_dir() {
            return Ok(());
        }

        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() && recursive {
                Self::scan_directory(&path, recursive, files)?;
            } else if path.is_file() {
                if let Some(training_file) = Self::analyze_file(&path)? {
                    files.push(training_file);
                }
            }
        }

        Ok(())
    }

    /// Analyze a file to determine if it's training data
    fn analyze_file(path: &Path) -> Result<Option<TrainingFile>, Box<dyn std::error::Error>> {
        let metadata = fs::metadata(path)?;
        let size_bytes = metadata.len();

        // Skip system files, hidden files, and very small files
        if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
            if file_name.starts_with('.')
                || file_name.starts_with("~")
                || file_name.contains("lock")
                || file_name.contains("tmp")
                || size_bytes < 10
            {
                return Ok(None);
            }
        }

        // Get base name (without extension)
        let base_name = Self::extract_base_name(path);

        // Detect format by extension and content
        if let Some((format, priority)) = Self::detect_format(path)? {
            return Ok(Some(TrainingFile {
                path: path.to_path_buf(),
                format,
                priority,
                base_name,
                size_bytes,
            }));
        }

        Ok(None)
    }

    /// Extract base name from path, removing format extensions
    fn extract_base_name(path: &Path) -> String {
        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        // Remove known extensions to get base name

        file_name
            .replace(".mmap", "")
            .replace(".msgpack", "")
            .replace(".bin", "")
            .replace(".zst", "")
            .replace(".json", "")
    }

    /// Detect file format and priority
    fn detect_format(
        path: &Path,
    ) -> Result<Option<(String, FormatPriority)>, Box<dyn std::error::Error>> {
        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

        // Check by extension
        if file_name.ends_with(".mmap") {
            return Ok(Some(("MMAP".to_string(), FormatPriority::MemoryMapped)));
        }

        if file_name.ends_with(".msgpack") {
            return Ok(Some(("MSGPACK".to_string(), FormatPriority::MessagePack)));
        }

        if file_name.ends_with(".bin") && Self::verify_binary_training_data(path)? {
            return Ok(Some(("BINARY".to_string(), FormatPriority::Binary)));
        }

        if file_name.ends_with(".zst") {
            return Ok(Some(("ZSTD".to_string(), FormatPriority::Zstd)));
        }

        if file_name.ends_with(".json") && Self::verify_json_training_data(path)? {
            return Ok(Some(("JSON".to_string(), FormatPriority::Json)));
        }

        // Check by content for files that might not have proper extensions
        if (file_name.contains("training")
            || file_name.contains("position")
            || file_name.contains("tactical"))
            && Self::verify_json_training_data(path)?
        {
            return Ok(Some(("JSON".to_string(), FormatPriority::Json)));
        }

        Ok(None)
    }

    /// Verify that a JSON file contains training data
    fn verify_json_training_data(path: &Path) -> Result<bool, Box<dyn std::error::Error>> {
        // Check file size first
        if let Ok(metadata) = std::fs::metadata(path) {
            let size = metadata.len();
            if size == 0 || size > 10_000_000 {
                // Skip empty files or files > 10MB
                return Ok(false);
            }
        }

        let file = std::fs::File::open(path)?;
        let reader = BufReader::new(file);

        // Check first few lines for training data structure
        for (i, line_result) in reader.lines().enumerate() {
            if i >= 10 {
                break;
            } // Only check first 10 lines

            // Handle UTF-8 errors gracefully
            let line = match line_result {
                Ok(line) => line,
                Err(_) => {
                    // If we can't read as UTF-8, it's not a JSON training file
                    return Ok(false);
                }
            };

            if line.trim().is_empty() {
                continue;
            }

            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line) {
                // Look for common training data fields
                if json.get("fen").is_some() && json.get("evaluation").is_some() {
                    return Ok(true);
                }
                if json.get("board").is_some() && json.get("eval").is_some() {
                    return Ok(true);
                }
                if json.get("position").is_some() && json.get("score").is_some() {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    /// Verify that a binary file contains training data
    fn verify_binary_training_data(path: &Path) -> Result<bool, Box<dyn std::error::Error>> {
        use std::io::Read;

        // Skip very small files and very large files for safety
        if let Ok(metadata) = std::fs::metadata(path) {
            let size = metadata.len();
            if !(100..=100_000_000).contains(&size) {
                // Skip files < 100B or > 100MB
                return Ok(false);
            }
        }

        let mut file = std::fs::File::open(path)?;
        let mut buffer = vec![0u8; 1024]; // Read first 1KB
        let bytes_read = file.read(&mut buffer)?;

        if bytes_read == 0 {
            return Ok(false);
        }

        buffer.truncate(bytes_read);

        // Try to deserialize as training data (catch all errors)
        if bincode::deserialize::<Vec<(String, f32)>>(&buffer).is_ok() {
            return Ok(true);
        }

        // Try LZ4 decompression first (catch all errors)
        if let Ok(decompressed) = lz4_flex::decompress_size_prepended(&buffer) {
            if bincode::deserialize::<Vec<(String, f32)>>(&decompressed).is_ok() {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Format bytes for display
    fn format_bytes(bytes: u64) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB"];
        let mut size = bytes as f64;
        let mut unit_index = 0;

        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }

        "Processing files...".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_name_extraction() {
        assert_eq!(
            AutoDiscovery::extract_base_name(Path::new("training_data.json")),
            "training_data"
        );
        assert_eq!(
            AutoDiscovery::extract_base_name(Path::new("training_data.mmap")),
            "training_data"
        );
        assert_eq!(
            AutoDiscovery::extract_base_name(Path::new("tactical_training_data.msgpack")),
            "tactical_training_data"
        );
    }

    #[test]
    fn test_format_priority() {
        assert!(FormatPriority::MemoryMapped < FormatPriority::MessagePack);
        assert!(FormatPriority::MessagePack < FormatPriority::Binary);
        assert!(FormatPriority::Binary < FormatPriority::Json);
    }

    #[test]
    fn test_consolidation() {
        let files = vec![
            TrainingFile {
                path: PathBuf::from("training_data.json"),
                format: "JSON".to_string(),
                priority: FormatPriority::Json,
                base_name: "training_data".to_string(),
                size_bytes: 1000,
            },
            TrainingFile {
                path: PathBuf::from("training_data.mmap"),
                format: "MMAP".to_string(),
                priority: FormatPriority::MemoryMapped,
                base_name: "training_data".to_string(),
                size_bytes: 800,
            },
        ];

        let consolidated = AutoDiscovery::consolidate_by_base_name(files);
        let best = consolidated.get("training_data").unwrap();

        assert_eq!(best.format, "MMAP");
        assert_eq!(best.priority, FormatPriority::MemoryMapped);
    }
}
