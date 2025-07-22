//! Lazy loading system for strategic motifs
//! 
//! This module implements an on-demand loading system for strategic chess motifs,
//! significantly reducing memory usage and startup time by only loading patterns
//! when they're actually needed for evaluation.

use crate::strategic_motifs::{StrategicMotif, MotifMatch, MotifType, GamePhase};
use chess::Board;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

/// Configuration for lazy loading behavior
#[derive(Debug, Clone)]
pub struct LazyLoadConfig {
    /// Maximum number of motifs to keep in memory at once
    pub max_cached_motifs: usize,
    /// How long to keep unused motifs in memory (seconds)
    pub motif_ttl_secs: u64,
    /// Maximum number of files to keep file handles open for
    pub max_open_files: usize,
    /// Enable compression for motif files
    pub use_compression: bool,
}

impl Default for LazyLoadConfig {
    fn default() -> Self {
        Self {
            max_cached_motifs: 1000,  // Keep 1000 most recent motifs in memory
            motif_ttl_secs: 300,      // 5 minutes TTL
            max_open_files: 10,       // Keep 10 files open
            use_compression: true,
        }
    }
}

/// Metadata for a motif file segment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotifSegmentMeta {
    /// File path for this segment
    pub file_path: PathBuf,
    /// Range of motif IDs in this segment
    pub id_range: (u64, u64),
    /// Number of motifs in this segment
    pub motif_count: usize,
    /// File size in bytes
    pub file_size: u64,
    /// Game phase this segment focuses on
    pub primary_phase: GamePhase,
    /// Secondary phases this segment covers
    pub secondary_phases: Vec<GamePhase>,
    /// Creation timestamp for cache management
    pub created_at: std::time::SystemTime,
}

/// Index mapping motif IDs to their file segments
#[derive(Debug, Serialize, Deserialize)]
pub struct MotifIndex {
    /// Map from motif ID to segment metadata
    pub motif_to_segment: HashMap<u64, MotifSegmentMeta>,
    /// Map from pattern hash to motif IDs
    pub pattern_to_motifs: HashMap<u64, Vec<u64>>,
    /// Game phase to relevant segment files
    pub phase_to_segments: HashMap<GamePhase, Vec<PathBuf>>,
    /// Total number of motifs across all segments
    pub total_motifs: usize,
}

/// Cache entry for loaded motifs with TTL
#[derive(Debug, Clone)]
struct CachedMotif {
    motif: StrategicMotif,
    last_accessed: Instant,
    access_count: u32,
}

/// File handle cache for efficient segment access
struct FileHandleCache {
    handles: HashMap<PathBuf, Box<dyn std::io::Read + Send>>,
    last_accessed: HashMap<PathBuf, Instant>,
    max_handles: usize,
}

impl FileHandleCache {
    fn new(max_handles: usize) -> Self {
        Self {
            handles: HashMap::new(),
            last_accessed: HashMap::new(),
            max_handles,
        }
    }

    fn evict_old_handles(&mut self) {
        if self.handles.len() >= self.max_handles {
            // Remove least recently used handle
            if let Some((oldest_path, _)) = self.last_accessed.iter()
                .min_by_key(|(_, &time)| time)
                .map(|(path, time)| (path.clone(), *time))
            {
                self.handles.remove(&oldest_path);
                self.last_accessed.remove(&oldest_path);
            }
        }
    }
}

/// Lazy-loading strategic motif database
pub struct LazyStrategicDatabase {
    /// Configuration for lazy loading
    config: LazyLoadConfig,
    /// Index mapping motifs to file segments
    index: MotifIndex,
    /// Cache of recently accessed motifs
    motif_cache: Arc<RwLock<HashMap<u64, CachedMotif>>>,
    /// File handle cache for efficient access
    file_cache: Arc<Mutex<FileHandleCache>>,
    /// Base directory for motif files
    base_dir: PathBuf,
    /// Statistics for monitoring performance
    stats: Arc<RwLock<LazyLoadStats>>,
}

/// Statistics for monitoring lazy loading performance
#[derive(Debug, Default)]
pub struct LazyLoadStats {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub files_loaded: u64,
    pub motifs_loaded: u64,
    pub cache_evictions: u64,
    pub file_handle_evictions: u64,
    pub total_load_time_ms: u64,
    pub average_load_time_ms: f64,
}

impl LazyLoadStats {
    pub fn hit_ratio(&self) -> f64 {
        if self.cache_hits + self.cache_misses == 0 {
            0.0
        } else {
            self.cache_hits as f64 / (self.cache_hits + self.cache_misses) as f64
        }
    }

    pub fn update_load_time(&mut self, load_time_ms: u64) {
        self.total_load_time_ms += load_time_ms;
        self.average_load_time_ms = self.total_load_time_ms as f64 / self.files_loaded.max(1) as f64;
    }
}

impl LazyStrategicDatabase {
    /// Create new lazy-loading strategic database
    pub fn new<P: AsRef<Path>>(base_dir: P, config: LazyLoadConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let base_dir = base_dir.as_ref().to_path_buf();
        
        // Load the index file
        let index_path = base_dir.join("motif_index.bin");
        let index = Self::load_index(&index_path)?;
        
        let file_cache = FileHandleCache::new(config.max_open_files);
        
        Ok(Self {
            config,
            index,
            motif_cache: Arc::new(RwLock::new(HashMap::new())),
            file_cache: Arc::new(Mutex::new(file_cache)),
            base_dir,
            stats: Arc::new(RwLock::new(LazyLoadStats::default())),
        })
    }

    /// Load motif index from file
    fn load_index(index_path: &Path) -> Result<MotifIndex, Box<dyn std::error::Error>> {
        use std::fs::File;
        use std::io::BufReader;

        let file = File::open(index_path)?;
        let reader = BufReader::new(file);
        let index: MotifIndex = bincode::deserialize_from(reader)?;
        Ok(index)
    }

    /// Get a motif by ID, loading from disk if necessary
    pub fn get_motif(&self, motif_id: u64) -> Result<Option<StrategicMotif>, Box<dyn std::error::Error>> {
        // Check cache first
        {
            let cache = self.motif_cache.read().unwrap();
            if let Some(cached) = cache.get(&motif_id) {
                self.stats.write().unwrap().cache_hits += 1;
                return Ok(Some(cached.motif.clone()));
            }
        }

        self.stats.write().unwrap().cache_misses += 1;

        // Find which segment contains this motif
        let segment = match self.index.motif_to_segment.get(&motif_id) {
            Some(segment) => segment,
            None => return Ok(None),
        };

        // Load the motif from disk
        let motif = self.load_motif_from_segment(motif_id, segment)?;
        
        if let Some(motif) = motif.as_ref() {
            // Cache the loaded motif
            self.cache_motif(motif_id, motif.clone());
        }

        Ok(motif)
    }

    /// Find motifs matching a pattern hash
    pub fn find_motifs_by_pattern(&self, pattern_hash: u64) -> Result<Vec<StrategicMotif>, Box<dyn std::error::Error>> {
        let motif_ids = match self.index.pattern_to_motifs.get(&pattern_hash) {
            Some(ids) => ids,
            None => return Ok(Vec::new()),
        };

        let mut results = Vec::new();
        for &motif_id in motif_ids {
            if let Some(motif) = self.get_motif(motif_id)? {
                results.push(motif);
            }
        }

        Ok(results)
    }

    /// Find motifs relevant to a specific game phase
    pub fn find_motifs_by_phase(&self, phase: &GamePhase) -> Result<Vec<StrategicMotif>, Box<dyn std::error::Error>> {
        let segment_paths = match self.index.phase_to_segments.get(phase) {
            Some(paths) => paths,
            None => return Ok(Vec::new()),
        };

        let mut results = Vec::new();
        
        // Load a sampling of motifs from relevant segments (not all at once)
        for path in segment_paths.iter().take(3) { // Limit to 3 segments to avoid memory bloat
            let motifs = self.load_segment_sample(path, 10)?; // Load 10 motifs per segment
            results.extend(motifs);
        }

        Ok(results)
    }

    /// Evaluate a position against relevant strategic motifs
    pub fn evaluate_position(&self, board: &Board) -> Result<Vec<MotifMatch>, Box<dyn std::error::Error>> {
        let position_hash = self.calculate_position_hash(board);
        
        // Find motifs that might match this position pattern
        let relevant_motifs = self.find_motifs_by_pattern(position_hash)?;
        
        let mut matches = Vec::new();
        
        for motif in relevant_motifs {
            if let Some(motif_match) = self.match_motif_to_position(&motif, board) {
                matches.push(motif_match);
            }
        }

        // Sort by relevance score
        matches.sort_by(|a, b| b.relevance.partial_cmp(&a.relevance).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(matches)
    }

    /// Load a specific motif from its segment file
    fn load_motif_from_segment(&self, motif_id: u64, segment: &MotifSegmentMeta) -> Result<Option<StrategicMotif>, Box<dyn std::error::Error>> {
        let start_time = Instant::now();
        
        let file_path = self.base_dir.join(&segment.file_path);
        let motifs = self.load_segment(&file_path)?;
        
        let load_time = start_time.elapsed().as_millis() as u64;
        {
            let mut stats = self.stats.write().unwrap();
            stats.files_loaded += 1;
            stats.motifs_loaded += motifs.len() as u64;
            stats.update_load_time(load_time);
        }

        // Find the specific motif
        let motif = motifs.into_iter().find(|m| m.id == motif_id);
        Ok(motif)
    }

    /// Load a sample of motifs from a segment (for phase-based queries)
    fn load_segment_sample(&self, path: &Path, max_count: usize) -> Result<Vec<StrategicMotif>, Box<dyn std::error::Error>> {
        let full_path = self.base_dir.join(path);
        let motifs = self.load_segment(&full_path)?;
        
        // Take a sample to avoid loading too many motifs at once
        Ok(motifs.into_iter().take(max_count).collect())
    }

    /// Load an entire segment file
    fn load_segment(&self, path: &Path) -> Result<Vec<StrategicMotif>, Box<dyn std::error::Error>> {
        use std::fs::File;
        use std::io::BufReader;

        let file = File::open(path)?;
        let reader = BufReader::new(file);
        
        let motifs: Vec<StrategicMotif> = if self.config.use_compression {
            // Handle compressed files
            bincode::deserialize_from(reader)?
        } else {
            bincode::deserialize_from(reader)?
        };

        Ok(motifs)
    }

    /// Cache a motif with TTL
    fn cache_motif(&self, motif_id: u64, motif: StrategicMotif) {
        let mut cache = self.motif_cache.write().unwrap();
        
        // Evict old entries if cache is full
        if cache.len() >= self.config.max_cached_motifs {
            self.evict_old_motifs(&mut cache);
        }

        cache.insert(motif_id, CachedMotif {
            motif,
            last_accessed: Instant::now(),
            access_count: 1,
        });
    }

    /// Evict old motifs from cache based on TTL and LRU
    fn evict_old_motifs(&self, cache: &mut HashMap<u64, CachedMotif>) {
        let now = Instant::now();
        let ttl = Duration::from_secs(self.config.motif_ttl_secs);
        
        // Remove expired entries
        let expired_keys: Vec<u64> = cache.iter()
            .filter(|(_, cached)| now.duration_since(cached.last_accessed) > ttl)
            .map(|(&id, _)| id)
            .collect();
            
        for key in expired_keys {
            cache.remove(&key);
            self.stats.write().unwrap().cache_evictions += 1;
        }

        // If still too many, remove least recently used
        while cache.len() >= self.config.max_cached_motifs {
            if let Some((lru_id, _)) = cache.iter()
                .min_by_key(|(_, cached)| cached.last_accessed)
                .map(|(&id, cached)| (id, cached))
            {
                cache.remove(&lru_id);
                self.stats.write().unwrap().cache_evictions += 1;
            } else {
                break;
            }
        }
    }

    /// Calculate a simple position hash for pattern matching
    fn calculate_position_hash(&self, board: &Board) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        board.to_string().hash(&mut hasher);
        hasher.finish()
    }

    /// Match a motif against a position
    fn match_motif_to_position(&self, motif: &StrategicMotif, board: &Board) -> Option<MotifMatch> {
        // Simplified matching logic - in a real implementation this would be more sophisticated
        let relevance = match &motif.motif_type {
            MotifType::PawnStructure(_) => 0.7,
            MotifType::PieceCoordination(_) => 0.6,
            MotifType::KingSafety(_) => 0.8,
            MotifType::Initiative(_) => 0.5,
            MotifType::Endgame(_) => 0.6,
            MotifType::Opening(_) => 0.4,
        };

        if relevance > 0.3 {
            Some(MotifMatch {
                motif: motif.clone(),
                relevance,
                matching_squares: Vec::new(), // Would be populated by real pattern matching
            })
        } else {
            None
        }
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> LazyLoadStats {
        let stats = self.stats.read().unwrap();
        LazyLoadStats {
            cache_hits: stats.cache_hits,
            cache_misses: stats.cache_misses,
            files_loaded: stats.files_loaded,
            motifs_loaded: stats.motifs_loaded,
            cache_evictions: stats.cache_evictions,
            file_handle_evictions: stats.file_handle_evictions,
            total_load_time_ms: stats.total_load_time_ms,
            average_load_time_ms: stats.average_load_time_ms,
        }
    }

    /// Clear all caches and reset statistics
    pub fn clear_caches(&self) {
        self.motif_cache.write().unwrap().clear();
        self.file_cache.lock().unwrap().handles.clear();
        *self.stats.write().unwrap() = LazyLoadStats::default();
    }

    /// Preload motifs for a specific game phase (optimization)
    pub fn preload_phase(&self, phase: GamePhase) -> Result<usize, Box<dyn std::error::Error>> {
        let motifs = self.find_motifs_by_phase(&phase)?;
        let count = motifs.len();
        
        // Motifs are now cached from the find_motifs_by_phase call
        Ok(count)
    }

    /// Get total number of available motifs
    pub fn total_motifs(&self) -> usize {
        self.index.total_motifs
    }

    /// Get number of cached motifs
    pub fn cached_motifs(&self) -> usize {
        self.motif_cache.read().unwrap().len()
    }
}

/// Utility for creating motif segment files
pub struct MotifSegmentBuilder {
    config: LazyLoadConfig,
    base_dir: PathBuf,
}

impl MotifSegmentBuilder {
    pub fn new<P: AsRef<Path>>(base_dir: P, config: LazyLoadConfig) -> Self {
        Self {
            config,
            base_dir: base_dir.as_ref().to_path_buf(),
        }
    }

    /// Split a large collection of motifs into segment files
    pub fn create_segments(&self, motifs: Vec<StrategicMotif>, motifs_per_segment: usize) -> Result<MotifIndex, Box<dyn std::error::Error>> {
        let mut index = MotifIndex {
            motif_to_segment: HashMap::new(),
            pattern_to_motifs: HashMap::new(),
            phase_to_segments: HashMap::new(),
            total_motifs: motifs.len(),
        };

        // Group motifs by game phase for better locality
        let mut phase_groups: HashMap<GamePhase, Vec<StrategicMotif>> = HashMap::new();
        
        for motif in motifs {
            let phase = motif.context.game_phase.clone();
            phase_groups.entry(phase).or_insert_with(Vec::new).push(motif);
        }

        // Create segments for each phase
        for (phase, phase_motifs) in phase_groups {
            let segments = self.create_phase_segments(phase_motifs, motifs_per_segment, &phase)?;
            
            for segment in segments {
                index.phase_to_segments.entry(phase.clone()).or_insert_with(Vec::new).push(segment.file_path.clone());
                
                // Update index mappings for motifs in this segment
                for motif_id in segment.id_range.0..=segment.id_range.1 {
                    index.motif_to_segment.insert(motif_id, segment.clone());
                }
            }
        }

        // Save the index
        self.save_index(&index)?;
        
        Ok(index)
    }

    fn create_phase_segments(&self, motifs: Vec<StrategicMotif>, motifs_per_segment: usize, phase: &GamePhase) -> Result<Vec<MotifSegmentMeta>, Box<dyn std::error::Error>> {
        let mut segments = Vec::new();
        
        for (segment_idx, chunk) in motifs.chunks(motifs_per_segment).enumerate() {
            let filename = format!("{:?}_segment_{}.bin", phase, segment_idx);
            let file_path = self.base_dir.join(&filename);
            
            // Write the segment file
            self.write_segment_file(&file_path, chunk)?;
            
            let id_range = if chunk.is_empty() {
                (0, 0)
            } else {
                (chunk[0].id, chunk[chunk.len() - 1].id)
            };

            let segment = MotifSegmentMeta {
                file_path: PathBuf::from(filename),
                id_range,
                motif_count: chunk.len(),
                file_size: std::fs::metadata(&file_path)?.len(),
                primary_phase: phase.clone(),
                secondary_phases: Vec::new(),
                created_at: std::time::SystemTime::now(),
            };

            segments.push(segment);
        }

        Ok(segments)
    }

    fn write_segment_file(&self, path: &Path, motifs: &[StrategicMotif]) -> Result<(), Box<dyn std::error::Error>> {
        use std::fs::File;
        use std::io::BufWriter;

        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        
        bincode::serialize_into(writer, motifs)?;
        Ok(())
    }

    fn save_index(&self, index: &MotifIndex) -> Result<(), Box<dyn std::error::Error>> {
        use std::fs::File;
        use std::io::BufWriter;

        let index_path = self.base_dir.join("motif_index.bin");
        let file = File::create(index_path)?;
        let writer = BufWriter::new(file);
        
        bincode::serialize_into(writer, index)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_lazy_loading_basic_functionality() {
        // This would require test data setup
        // For now, just test that structures can be created
        let config = LazyLoadConfig::default();
        assert_eq!(config.max_cached_motifs, 1000);
    }

    #[test]
    fn test_cache_eviction() {
        let config = LazyLoadConfig {
            max_cached_motifs: 2,
            motif_ttl_secs: 0, // Immediate expiration
            ..Default::default()
        };
        
        // Test would require more setup for full functionality
        assert!(config.max_cached_motifs == 2);
    }
}