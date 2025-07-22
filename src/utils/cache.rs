use lru::LruCache;
use std::collections::HashMap;
use std::hash::Hash;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Thread-safe LRU cache with time-based expiration
pub struct TimedLruCache<K, V> {
    cache: Arc<Mutex<LruCache<K, CacheEntry<V>>>>,
    ttl: Duration,
}

/// Cache entry with timestamp for TTL support
#[derive(Debug, Clone)]
struct CacheEntry<V> {
    value: V,
    timestamp: Instant,
}

impl<K, V> TimedLruCache<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    /// Create a new timed LRU cache
    pub fn new(capacity: usize, ttl: Duration) -> Self {
        let non_zero_capacity =
            NonZeroUsize::new(capacity).unwrap_or(NonZeroUsize::new(1).unwrap());
        Self {
            cache: Arc::new(Mutex::new(LruCache::new(non_zero_capacity))),
            ttl,
        }
    }

    /// Insert a value into the cache
    pub fn insert(&self, key: K, value: V) {
        let entry = CacheEntry {
            value,
            timestamp: Instant::now(),
        };

        if let Ok(mut cache) = self.cache.lock() {
            cache.put(key, entry);
        }
    }

    /// Get a value from the cache
    pub fn get(&self, key: &K) -> Option<V> {
        if let Ok(mut cache) = self.cache.lock() {
            if let Some(entry) = cache.get(key) {
                // Check if entry has expired
                if entry.timestamp.elapsed() < self.ttl {
                    return Some(entry.value.clone());
                } else {
                    // Remove expired entry
                    cache.pop(key);
                }
            }
        }
        None
    }

    /// Check if a key exists in cache (without updating LRU order)
    pub fn contains(&self, key: &K) -> bool {
        if let Ok(cache) = self.cache.lock() {
            if let Some(entry) = cache.peek(key) {
                return entry.timestamp.elapsed() < self.ttl;
            }
        }
        false
    }

    /// Clear all entries from the cache
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        if let Ok(cache) = self.cache.lock() {
            let capacity = cache.cap().get();
            let size = cache.len();
            let expired_count = cache
                .iter()
                .filter(|(_, entry)| entry.timestamp.elapsed() >= self.ttl)
                .count();

            CacheStats {
                capacity,
                size,
                expired_count,
                hit_ratio: 0.0, // Would need hit/miss tracking for accurate ratio
            }
        } else {
            CacheStats {
                capacity: 0,
                size: 0,
                expired_count: 0,
                hit_ratio: 0.0,
            }
        }
    }

    /// Clean up expired entries
    pub fn cleanup_expired(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            let now = Instant::now();
            let expired_keys: Vec<K> = cache
                .iter()
                .filter(|(_, entry)| now.duration_since(entry.timestamp) >= self.ttl)
                .map(|(k, _)| k.clone())
                .collect();

            for key in expired_keys {
                cache.pop(&key);
            }
        }
    }
}

/// High-performance similarity cache for chess positions
pub struct SimilarityCache {
    cache: TimedLruCache<(usize, usize), f32>,
    hit_count: Arc<Mutex<u64>>,
    miss_count: Arc<Mutex<u64>>,
}

impl SimilarityCache {
    /// Create a new similarity cache
    pub fn new(capacity: usize, ttl: Duration) -> Self {
        Self {
            cache: TimedLruCache::new(capacity, ttl),
            hit_count: Arc::new(Mutex::new(0)),
            miss_count: Arc::new(Mutex::new(0)),
        }
    }

    /// Get similarity from cache
    pub fn get_similarity(&self, pos1: usize, pos2: usize) -> Option<f32> {
        // Normalize key order (similarity is symmetric)
        let key = if pos1 <= pos2 {
            (pos1, pos2)
        } else {
            (pos2, pos1)
        };

        if let Some(similarity) = self.cache.get(&key) {
            if let Ok(mut hits) = self.hit_count.lock() {
                *hits += 1;
            }
            Some(similarity)
        } else {
            if let Ok(mut misses) = self.miss_count.lock() {
                *misses += 1;
            }
            None
        }
    }

    /// Store similarity in cache
    pub fn store_similarity(&self, pos1: usize, pos2: usize, similarity: f32) {
        // Normalize key order (similarity is symmetric)
        let key = if pos1 <= pos2 {
            (pos1, pos2)
        } else {
            (pos2, pos1)
        };
        self.cache.insert(key, similarity);
    }

    /// Get cache statistics with hit ratio
    pub fn stats(&self) -> CacheStats {
        let mut base_stats = self.cache.stats();

        let hits = self.hit_count.lock().map(|h| *h).unwrap_or(0);
        let misses = self.miss_count.lock().map(|m| *m).unwrap_or(0);

        base_stats.hit_ratio = if hits + misses > 0 {
            hits as f64 / (hits + misses) as f64
        } else {
            0.0
        };

        base_stats
    }

    /// Clear cache and reset statistics
    pub fn clear(&self) {
        self.cache.clear();
        if let Ok(mut hits) = self.hit_count.lock() {
            *hits = 0;
        }
        if let Ok(mut misses) = self.miss_count.lock() {
            *misses = 0;
        }
    }
}

/// Evaluation result cache for chess positions
pub struct EvaluationCache {
    cache: TimedLruCache<String, f32>,
    hit_count: Arc<Mutex<u64>>,
    miss_count: Arc<Mutex<u64>>,
}

impl EvaluationCache {
    /// Create a new evaluation cache
    pub fn new(capacity: usize, ttl: Duration) -> Self {
        Self {
            cache: TimedLruCache::new(capacity, ttl),
            hit_count: Arc::new(Mutex::new(0)),
            miss_count: Arc::new(Mutex::new(0)),
        }
    }

    /// Get evaluation from cache using FEN string as key
    pub fn get_evaluation(&self, fen: &str) -> Option<f32> {
        if let Some(evaluation) = self.cache.get(&fen.to_string()) {
            if let Ok(mut hits) = self.hit_count.lock() {
                *hits += 1;
            }
            Some(evaluation)
        } else {
            if let Ok(mut misses) = self.miss_count.lock() {
                *misses += 1;
            }
            None
        }
    }

    /// Store evaluation in cache
    pub fn store_evaluation(&self, fen: &str, evaluation: f32) {
        self.cache.insert(fen.to_string(), evaluation);
    }

    /// Get cache statistics with hit ratio
    pub fn stats(&self) -> CacheStats {
        let mut base_stats = self.cache.stats();

        let hits = self.hit_count.lock().map(|h| *h).unwrap_or(0);
        let misses = self.miss_count.lock().map(|m| *m).unwrap_or(0);

        base_stats.hit_ratio = if hits + misses > 0 {
            hits as f64 / (hits + misses) as f64
        } else {
            0.0
        };

        base_stats
    }

    /// Clear cache and reset statistics
    pub fn clear(&self) {
        self.cache.clear();
        if let Ok(mut hits) = self.hit_count.lock() {
            *hits = 0;
        }
        if let Ok(mut misses) = self.miss_count.lock() {
            *misses = 0;
        }
    }
}

/// Write-through cache for pattern data
pub struct PatternCache<K, V> {
    cache: Arc<Mutex<HashMap<K, V>>>,
    backing_store: Arc<Mutex<HashMap<K, V>>>,
    max_size: usize,
}

impl<K, V> PatternCache<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    /// Create a new pattern cache with backing store
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
            backing_store: Arc::new(Mutex::new(HashMap::new())),
            max_size,
        }
    }

    /// Insert a value (writes to both cache and backing store)
    pub fn insert(&self, key: K, value: V) {
        // Write to backing store first
        if let Ok(mut store) = self.backing_store.lock() {
            store.insert(key.clone(), value.clone());
        }

        // Then write to cache
        if let Ok(mut cache) = self.cache.lock() {
            // If cache is full, remove random entry
            if cache.len() >= self.max_size {
                if let Some(key_to_remove) = cache.keys().next().cloned() {
                    cache.remove(&key_to_remove);
                }
            }
            cache.insert(key, value);
        }
    }

    /// Get a value (checks cache first, then backing store)
    pub fn get(&self, key: &K) -> Option<V> {
        // Check cache first
        if let Ok(cache) = self.cache.lock() {
            if let Some(value) = cache.get(key) {
                return Some(value.clone());
            }
        }

        // Check backing store
        if let Ok(store) = self.backing_store.lock() {
            if let Some(value) = store.get(key) {
                let value = value.clone();

                // Promote to cache
                if let Ok(mut cache) = self.cache.lock() {
                    if cache.len() >= self.max_size {
                        if let Some(key_to_remove) = cache.keys().next().cloned() {
                            cache.remove(&key_to_remove);
                        }
                    }
                    cache.insert(key.clone(), value.clone());
                }

                return Some(value);
            }
        }

        None
    }

    /// Check if key exists (cache or backing store)
    pub fn contains(&self, key: &K) -> bool {
        if let Ok(cache) = self.cache.lock() {
            if cache.contains_key(key) {
                return true;
            }
        }

        if let Ok(store) = self.backing_store.lock() {
            return store.contains_key(key);
        }

        false
    }

    /// Clear both cache and backing store
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
        if let Ok(mut store) = self.backing_store.lock() {
            store.clear();
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> PatternCacheStats {
        let cache_size = self.cache.lock().map(|c| c.len()).unwrap_or(0);
        let backing_size = self.backing_store.lock().map(|s| s.len()).unwrap_or(0);

        PatternCacheStats {
            cache_size,
            backing_size,
            max_cache_size: self.max_size,
            cache_hit_ratio: 0.0, // Would need hit/miss tracking
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub capacity: usize,
    pub size: usize,
    pub expired_count: usize,
    pub hit_ratio: f64,
}

/// Pattern cache statistics
#[derive(Debug, Clone)]
pub struct PatternCacheStats {
    pub cache_size: usize,
    pub backing_size: usize,
    pub max_cache_size: usize,
    pub cache_hit_ratio: f64,
}

/// Batch cache operations for improved performance
pub struct BatchCache<K, V> {
    cache: Arc<Mutex<HashMap<K, V>>>,
    batch_size: usize,
    pending_inserts: Arc<Mutex<HashMap<K, V>>>,
}

impl<K, V> BatchCache<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    /// Create a new batch cache
    pub fn new(batch_size: usize) -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
            batch_size,
            pending_inserts: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Add item to pending batch
    pub fn batch_insert(&self, key: K, value: V) {
        if let Ok(mut pending) = self.pending_inserts.lock() {
            pending.insert(key, value);

            // Flush if batch is full
            if pending.len() >= self.batch_size {
                self.flush_batch();
            }
        }
    }

    /// Flush pending batch to main cache
    pub fn flush_batch(&self) {
        if let (Ok(mut cache), Ok(mut pending)) = (self.cache.lock(), self.pending_inserts.lock()) {
            for (key, value) in pending.drain() {
                cache.insert(key, value);
            }
        }
    }

    /// Get value from cache (including pending batch)
    pub fn get(&self, key: &K) -> Option<V> {
        // Check main cache first
        if let Ok(cache) = self.cache.lock() {
            if let Some(value) = cache.get(key) {
                return Some(value.clone());
            }
        }

        // Check pending batch
        if let Ok(pending) = self.pending_inserts.lock() {
            if let Some(value) = pending.get(key) {
                return Some(value.clone());
            }
        }

        None
    }

    /// Clear cache and pending batch
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
        if let Ok(mut pending) = self.pending_inserts.lock() {
            pending.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_timed_lru_cache() {
        let cache = TimedLruCache::new(3, Duration::from_millis(100));

        // Insert values
        cache.insert("key1", "value1");
        cache.insert("key2", "value2");
        cache.insert("key3", "value3");

        // Should be able to retrieve all
        assert_eq!(cache.get(&"key1"), Some("value1"));
        assert_eq!(cache.get(&"key2"), Some("value2"));
        assert_eq!(cache.get(&"key3"), Some("value3"));

        // Insert one more (should evict LRU)
        cache.insert("key4", "value4");
        assert_eq!(cache.get(&"key1"), None); // Should be evicted
        assert_eq!(cache.get(&"key4"), Some("value4"));
    }

    #[test]
    fn test_similarity_cache() {
        let cache = SimilarityCache::new(100, Duration::from_secs(1));

        // Store similarity
        cache.store_similarity(1, 2, 0.8);

        // Should be able to retrieve it (order shouldn't matter)
        assert_eq!(cache.get_similarity(1, 2), Some(0.8));
        assert_eq!(cache.get_similarity(2, 1), Some(0.8));

        // Non-existent similarity
        assert_eq!(cache.get_similarity(3, 4), None);

        // Check statistics
        let stats = cache.stats();
        assert_eq!(stats.hit_ratio, 2.0 / 3.0); // 2 hits out of 3 total requests
    }

    #[test]
    fn test_evaluation_cache() {
        let cache = EvaluationCache::new(100, Duration::from_secs(1));

        // Store evaluation
        cache.store_evaluation(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            0.0,
        );

        // Should be able to retrieve it
        assert_eq!(
            cache.get_evaluation("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            Some(0.0)
        );

        // Non-existent evaluation
        assert_eq!(cache.get_evaluation("8/8/8/8/8/8/8/8 w - - 0 1"), None);
    }

    #[test]
    fn test_pattern_cache() {
        let cache = PatternCache::new(2);

        // Insert values
        cache.insert("pattern1", "data1");
        cache.insert("pattern2", "data2");

        // Should be able to retrieve
        assert_eq!(cache.get(&"pattern1"), Some("data1"));
        assert_eq!(cache.get(&"pattern2"), Some("data2"));

        // Insert one more (should evict from cache but keep in backing store)
        cache.insert("pattern3", "data3");

        // Should still be able to retrieve all (from backing store)
        assert_eq!(cache.get(&"pattern1"), Some("data1"));
        assert_eq!(cache.get(&"pattern2"), Some("data2"));
        assert_eq!(cache.get(&"pattern3"), Some("data3"));
    }

    #[test]
    fn test_batch_cache() {
        let cache = BatchCache::new(2);

        // Add items to batch
        cache.batch_insert("key1", "value1");
        cache.batch_insert("key2", "value2");

        // Should be able to retrieve from pending batch
        assert_eq!(cache.get(&"key1"), Some("value1"));
        assert_eq!(cache.get(&"key2"), Some("value2"));

        // Add one more (should trigger flush)
        cache.batch_insert("key3", "value3");

        // Should still be able to retrieve all
        assert_eq!(cache.get(&"key1"), Some("value1"));
        assert_eq!(cache.get(&"key2"), Some("value2"));
        assert_eq!(cache.get(&"key3"), Some("value3"));
    }
}
