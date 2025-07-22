use crate::errors::{ChessEngineError, Result};
use ndarray::Array1;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Instant;

/// Lazy-loaded value that is computed on first access
pub struct Lazy<T> {
    data: RwLock<Option<T>>,
    initializer: Box<dyn Fn() -> Result<T> + Send + Sync>,
}

impl<T> Lazy<T> {
    /// Create a new lazy value
    pub fn new<F>(initializer: F) -> Self
    where
        F: Fn() -> Result<T> + Send + Sync + 'static,
    {
        Self {
            data: RwLock::new(None),
            initializer: Box::new(initializer),
        }
    }

    /// Get the value, computing it if necessary
    pub fn get(&self) -> Result<&T> {
        // First, try to read without blocking
        {
            let read_guard = self.data.read().unwrap();
            if read_guard.is_some() {
                // Safe because we know it's Some and we hold the read lock
                let ptr = read_guard.as_ref().unwrap() as *const T;
                unsafe {
                    return Ok(&*ptr);
                }
            }
        }

        // Need to initialize
        let mut write_guard = self.data.write().unwrap();

        // Double-check pattern
        if write_guard.is_none() {
            let value = (self.initializer)()?;
            *write_guard = Some(value);
        }

        // Safe because we know it's Some and we hold the write lock
        let ptr = write_guard.as_ref().unwrap() as *const T;
        unsafe { Ok(&*ptr) }
    }

    /// Check if the value has been initialized
    pub fn is_initialized(&self) -> bool {
        self.data.read().unwrap().is_some()
    }

    /// Force initialization without returning the value
    pub fn initialize(&self) -> Result<()> {
        self.get().map(|_| ())
    }

    /// Clear the lazy value, forcing re-initialization on next access
    pub fn clear(&self) {
        *self.data.write().unwrap() = None;
    }
}

/// Lazy collection that loads items on-demand
pub struct LazyCollection<K, V> {
    items: RwLock<HashMap<K, V>>,
    loader: Box<dyn Fn(&K) -> Result<V> + Send + Sync>,
    cache_size_limit: Option<usize>,
}

impl<K, V> LazyCollection<K, V>
where
    K: Clone + Eq + std::hash::Hash,
    V: Clone,
{
    /// Create a new lazy collection
    pub fn new<F>(loader: F) -> Self
    where
        F: Fn(&K) -> Result<V> + Send + Sync + 'static,
    {
        Self {
            items: RwLock::new(HashMap::new()),
            loader: Box::new(loader),
            cache_size_limit: None,
        }
    }

    /// Create a lazy collection with a cache size limit
    pub fn with_cache_limit<F>(loader: F, cache_limit: usize) -> Self
    where
        F: Fn(&K) -> Result<V> + Send + Sync + 'static,
    {
        Self {
            items: RwLock::new(HashMap::new()),
            loader: Box::new(loader),
            cache_size_limit: Some(cache_limit),
        }
    }

    /// Get an item, loading it if necessary
    pub fn get(&self, key: &K) -> Result<V> {
        // Try to read from cache first
        {
            let read_guard = self.items.read().unwrap();
            if let Some(value) = read_guard.get(key) {
                return Ok(value.clone());
            }
        }

        // Load the item
        let value = (self.loader)(key)?;

        // Cache the loaded value
        {
            let mut write_guard = self.items.write().unwrap();

            // Check cache size limit
            if let Some(limit) = self.cache_size_limit {
                if write_guard.len() >= limit {
                    // Remove a random item to make space
                    if let Some(key_to_remove) = write_guard.keys().next().cloned() {
                        write_guard.remove(&key_to_remove);
                    }
                }
            }

            write_guard.insert(key.clone(), value.clone());
        }

        Ok(value)
    }

    /// Check if an item is cached
    pub fn is_cached(&self, key: &K) -> bool {
        self.items.read().unwrap().contains_key(key)
    }

    /// Preload an item into the cache
    pub fn preload(&self, key: &K) -> Result<()> {
        self.get(key).map(|_| ())
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> LazyCollectionStats {
        let items = self.items.read().unwrap();
        LazyCollectionStats {
            cached_items: items.len(),
            cache_limit: self.cache_size_limit,
        }
    }

    /// Clear the cache
    pub fn clear_cache(&self) {
        self.items.write().unwrap().clear();
    }
}

/// Statistics for lazy collections
#[derive(Debug, Clone)]
pub struct LazyCollectionStats {
    pub cached_items: usize,
    pub cache_limit: Option<usize>,
}

/// Lazy file loader for large datasets
pub struct LazyFileLoader {
    base_path: PathBuf,
    loaded_files: RwLock<HashMap<String, Vec<u8>>>,
    file_metadata: RwLock<HashMap<String, FileMetadata>>,
    max_cache_size: usize,
    current_cache_size: RwLock<usize>,
}

#[derive(Debug, Clone)]
struct FileMetadata {
    size: usize,
    last_accessed: Instant,
    load_count: usize,
}

impl LazyFileLoader {
    /// Create a new lazy file loader
    pub fn new<P: AsRef<Path>>(base_path: P, max_cache_size: usize) -> Self {
        Self {
            base_path: base_path.as_ref().to_path_buf(),
            loaded_files: RwLock::new(HashMap::new()),
            file_metadata: RwLock::new(HashMap::new()),
            max_cache_size,
            current_cache_size: RwLock::new(0),
        }
    }

    /// Load a file, using cache if available
    pub fn load_file(&self, filename: &str) -> Result<Vec<u8>> {
        // Check cache first
        {
            let mut metadata = self.file_metadata.write().unwrap();
            if let Some(meta) = metadata.get_mut(filename) {
                meta.last_accessed = Instant::now();
                meta.load_count += 1;

                let files = self.loaded_files.read().unwrap();
                if let Some(data) = files.get(filename) {
                    return Ok(data.clone());
                }
            }
        }

        // Load from disk
        let file_path = self.base_path.join(filename);
        let data = std::fs::read(&file_path).map_err(|e| {
            ChessEngineError::IoError(format!("Failed to read file {}: {}", filename, e))
        })?;

        let file_size = data.len();

        // Check if we need to evict files to make space
        self.evict_if_necessary(file_size)?;

        // Cache the loaded file
        {
            let mut files = self.loaded_files.write().unwrap();
            let mut metadata = self.file_metadata.write().unwrap();
            let mut cache_size = self.current_cache_size.write().unwrap();

            files.insert(filename.to_string(), data.clone());
            metadata.insert(
                filename.to_string(),
                FileMetadata {
                    size: file_size,
                    last_accessed: Instant::now(),
                    load_count: 1,
                },
            );
            *cache_size += file_size;
        }

        Ok(data)
    }

    /// Evict files from cache if necessary
    fn evict_if_necessary(&self, needed_size: usize) -> Result<()> {
        let current_size = *self.current_cache_size.read().unwrap();

        if current_size + needed_size <= self.max_cache_size {
            return Ok(()); // No eviction needed
        }

        // Calculate how much space we need to free
        let space_to_free = (current_size + needed_size) - self.max_cache_size;

        // Get files sorted by access time (LRU)
        let files_to_evict = {
            let metadata = self.file_metadata.read().unwrap();
            let mut file_list: Vec<_> = metadata
                .iter()
                .map(|(name, meta)| (name.clone(), meta.last_accessed, meta.size))
                .collect();

            file_list.sort_by_key(|(_, access_time, _)| *access_time);

            let mut freed_space = 0;
            let mut to_evict = Vec::new();

            for (name, _, size) in file_list {
                to_evict.push(name);
                freed_space += size;
                if freed_space >= space_to_free {
                    break;
                }
            }

            to_evict
        };

        // Evict the selected files
        {
            let mut files = self.loaded_files.write().unwrap();
            let mut metadata = self.file_metadata.write().unwrap();
            let mut cache_size = self.current_cache_size.write().unwrap();

            for filename in files_to_evict {
                if let Some(meta) = metadata.remove(&filename) {
                    files.remove(&filename);
                    *cache_size = cache_size.saturating_sub(meta.size);
                }
            }
        }

        Ok(())
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> LazyFileLoaderStats {
        let files = self.loaded_files.read().unwrap();
        let current_size = *self.current_cache_size.read().unwrap();
        let metadata = self.file_metadata.read().unwrap();

        let total_loads = metadata.values().map(|m| m.load_count).sum();

        LazyFileLoaderStats {
            cached_files: files.len(),
            cache_size: current_size,
            max_cache_size: self.max_cache_size,
            total_file_loads: total_loads,
        }
    }

    /// Clear all cached files
    pub fn clear_cache(&self) {
        let mut files = self.loaded_files.write().unwrap();
        let mut metadata = self.file_metadata.write().unwrap();
        let mut cache_size = self.current_cache_size.write().unwrap();

        files.clear();
        metadata.clear();
        *cache_size = 0;
    }
}

/// Statistics for lazy file loader
#[derive(Debug, Clone)]
pub struct LazyFileLoaderStats {
    pub cached_files: usize,
    pub cache_size: usize,
    pub max_cache_size: usize,
    pub total_file_loads: usize,
}

/// Lazy position dataset for chess training data
pub struct LazyPositionDataset {
    file_loader: LazyFileLoader,
    position_cache: LazyCollection<String, Vec<(Array1<f32>, f32)>>,
    format_parsers:
        HashMap<String, Box<dyn Fn(&[u8]) -> Result<Vec<(Array1<f32>, f32)>> + Send + Sync>>,
}

impl LazyPositionDataset {
    /// Create a new lazy position dataset
    pub fn new<P: AsRef<Path>>(base_path: P, max_cache_size: usize) -> Self {
        let file_loader = LazyFileLoader::new(base_path, max_cache_size);

        let position_cache = LazyCollection::with_cache_limit(
            |_filename: &String| {
                // This will be properly implemented by calling the method directly
                Err(ChessEngineError::IoError("Not implemented".to_string()))
            },
            1000, // Cache up to 1000 position sets
        );

        let mut format_parsers: HashMap<
            String,
            Box<dyn Fn(&[u8]) -> Result<Vec<(Array1<f32>, f32)>> + Send + Sync>,
        > = HashMap::new();

        // Add JSON parser
        format_parsers.insert(
            "json".to_string(),
            Box::new(|data| Self::parse_json_positions(data)),
        );

        // Add binary parser
        format_parsers.insert(
            "bin".to_string(),
            Box::new(|data| Self::parse_binary_positions(data)),
        );

        Self {
            file_loader,
            position_cache,
            format_parsers,
        }
    }

    /// Load positions from a file
    pub fn load_positions(&self, filename: &str) -> Result<Vec<(Array1<f32>, f32)>> {
        self.position_cache.get(&filename.to_string())
    }

    /// Implementation of position loading
    fn load_positions_impl(&self, filename: &str) -> Result<Vec<(Array1<f32>, f32)>> {
        let data = self.file_loader.load_file(filename)?;

        // Determine file format from extension
        let extension = Path::new(filename)
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");

        if let Some(parser) = self.format_parsers.get(extension) {
            parser(&data)
        } else {
            Err(ChessEngineError::IoError(format!(
                "Unsupported file format: {}",
                extension
            )))
        }
    }

    /// Parse JSON positions
    fn parse_json_positions(data: &[u8]) -> Result<Vec<(Array1<f32>, f32)>> {
        let text = std::str::from_utf8(data)
            .map_err(|e| ChessEngineError::IoError(format!("Invalid UTF-8: {}", e)))?;

        let mut positions = Vec::new();

        for line in text.lines() {
            if line.trim().is_empty() || line.trim().starts_with('#') {
                continue;
            }

            let json_pos: serde_json::Value = serde_json::from_str(line)?;

            if let (Some(vector), Some(eval)) = (json_pos.get("vector"), json_pos.get("evaluation"))
            {
                if let (Some(vector_array), Some(eval_number)) = (vector.as_array(), eval.as_f64())
                {
                    let vector_data: std::result::Result<Vec<f32>, ChessEngineError> = vector_array
                        .iter()
                        .map(|v| {
                            v.as_f64().ok_or_else(|| {
                                ChessEngineError::IoError("Invalid vector element".to_string())
                            })
                        })
                        .map(|r| r.map(|v| v as f32))
                        .collect();

                    if let Ok(vector_data) = vector_data {
                        let array = Array1::from_vec(vector_data);
                        positions.push((array, eval_number as f32));
                    }
                }
            }
        }

        Ok(positions)
    }

    /// Parse binary positions
    fn parse_binary_positions(data: &[u8]) -> Result<Vec<(Array1<f32>, f32)>> {
        if data.len() < 8 {
            return Err(ChessEngineError::IoError("File too small".to_string()));
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
            if offset + 4 > data.len() {
                break;
            }

            // Read vector size
            let vector_size = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
            offset += 4;

            // Read vector data
            let vector_bytes = vector_size * 4;
            if offset + vector_bytes + 4 > data.len() {
                break;
            }

            let vector_data = unsafe {
                std::slice::from_raw_parts(data[offset..].as_ptr() as *const f32, vector_size)
            };
            let vector = Array1::from_vec(vector_data.to_vec());
            offset += vector_bytes;

            // Read evaluation
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

    /// Preload a set of files
    pub fn preload_files(&self, filenames: &[&str]) -> Result<()> {
        for filename in filenames {
            self.position_cache.preload(&filename.to_string())?;
        }
        Ok(())
    }

    /// Get dataset statistics
    pub fn stats(&self) -> LazyDatasetStats {
        LazyDatasetStats {
            file_loader_stats: self.file_loader.cache_stats(),
            position_cache_stats: self.position_cache.cache_stats(),
        }
    }
}

/// Statistics for lazy dataset
#[derive(Debug, Clone)]
pub struct LazyDatasetStats {
    pub file_loader_stats: LazyFileLoaderStats,
    pub position_cache_stats: LazyCollectionStats,
}

/// Global lazy resources manager
pub struct LazyResourceManager {
    resources: RwLock<HashMap<String, Box<dyn std::any::Any + Send + Sync>>>,
    initializers: RwLock<
        HashMap<String, Box<dyn Fn() -> Box<dyn std::any::Any + Send + Sync> + Send + Sync>>,
    >,
}

impl LazyResourceManager {
    /// Create a new resource manager
    pub fn new() -> Self {
        Self {
            resources: RwLock::new(HashMap::new()),
            initializers: RwLock::new(HashMap::new()),
        }
    }

    /// Register a lazy resource
    pub fn register<T, F>(&self, name: &str, initializer: F)
    where
        T: Send + Sync + 'static,
        F: Fn() -> T + Send + Sync + 'static,
    {
        let initializers = &mut *self.initializers.write().unwrap();
        initializers.insert(name.to_string(), Box::new(move || Box::new(initializer())));
    }

    /// Get a resource, initializing it if necessary
    pub fn get<T>(&self, name: &str) -> Option<Arc<T>>
    where
        T: Send + Sync + 'static,
    {
        // Try to get existing resource
        {
            let resources = self.resources.read().unwrap();
            if let Some(resource) = resources.get(name) {
                if let Some(typed_resource) = resource.downcast_ref::<Arc<T>>() {
                    return Some(Arc::clone(typed_resource));
                }
            }
        }

        // Initialize the resource
        let initializer = {
            let initializers = self.initializers.read().unwrap();
            initializers.get(name)?.as_ref()
                as *const dyn Fn() -> Box<dyn std::any::Any + Send + Sync>
        };

        let resource = unsafe { (*initializer)() };

        if let Ok(typed_resource) = resource.downcast::<Arc<T>>() {
            let result = Arc::clone(&typed_resource);

            // Cache the resource
            let mut resources = self.resources.write().unwrap();
            resources.insert(name.to_string(), typed_resource);

            Some(result)
        } else {
            None
        }
    }

    /// Clear all resources
    pub fn clear(&self) {
        self.resources.write().unwrap().clear();
    }
}

impl Default for LazyResourceManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Global lazy resource manager instance
static GLOBAL_RESOURCE_MANAGER: std::sync::OnceLock<LazyResourceManager> =
    std::sync::OnceLock::new();

/// Get the global resource manager
pub fn global_resource_manager() -> &'static LazyResourceManager {
    GLOBAL_RESOURCE_MANAGER.get_or_init(|| LazyResourceManager::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_lazy_value() {
        let counter = Arc::new(Mutex::new(0));
        let counter_clone = Arc::clone(&counter);

        let lazy = Lazy::new(move || {
            let mut count = counter_clone.lock().unwrap();
            *count += 1;
            Ok(*count)
        });

        assert!(!lazy.is_initialized());

        // First access should initialize
        let value1 = lazy.get().unwrap();
        assert_eq!(*value1, 1);
        assert!(lazy.is_initialized());

        // Second access should return cached value
        let value2 = lazy.get().unwrap();
        assert_eq!(*value2, 1);

        // Counter should only be incremented once
        assert_eq!(*counter.lock().unwrap(), 1);
    }

    #[test]
    fn test_lazy_collection() {
        let collection = LazyCollection::new(|key: &String| Ok(format!("Value for {}", key)));

        let value1 = collection.get(&"test".to_string()).unwrap();
        assert_eq!(value1, "Value for test");

        assert!(collection.is_cached(&"test".to_string()));
        assert!(!collection.is_cached(&"other".to_string()));
    }

    #[test]
    fn test_lazy_file_loader() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test.txt");
        fs::write(&file_path, "Hello, World!").unwrap();

        let loader = LazyFileLoader::new(temp_dir.path(), 1024);

        let data = loader.load_file("test.txt").unwrap();
        assert_eq!(data, b"Hello, World!");

        let stats = loader.cache_stats();
        assert_eq!(stats.cached_files, 1);
        assert!(stats.cache_size > 0);
    }

    #[test]
    fn test_resource_manager() {
        let manager = LazyResourceManager::new();

        manager.register("counter", || Arc::new(Mutex::new(42)));

        let counter1: Arc<Mutex<i32>> = manager.get("counter").unwrap();
        let counter2: Arc<Mutex<i32>> = manager.get("counter").unwrap();

        // Both should point to the same resource
        assert_eq!(*counter1.lock().unwrap(), 42);
        assert_eq!(*counter2.lock().unwrap(), 42);

        *counter1.lock().unwrap() = 100;
        assert_eq!(*counter2.lock().unwrap(), 100);
    }
}
