// Removed unused imports
use ndarray::Array1;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Thread-safe object pool for reusing expensive-to-create objects
pub struct ObjectPool<T> {
    pool: Arc<Mutex<VecDeque<T>>>,
    factory: Arc<dyn Fn() -> T + Send + Sync>,
    max_size: usize,
}

impl<T> ObjectPool<T> {
    /// Create a new object pool with a factory function
    pub fn new<F>(factory: F, max_size: usize) -> Self
    where
        F: Fn() -> T + Send + Sync + 'static,
    {
        Self {
            pool: Arc::new(Mutex::new(VecDeque::new())),
            factory: Arc::new(factory),
            max_size,
        }
    }

    /// Get an object from the pool, creating one if necessary
    pub fn get(&self) -> PooledObject<T> {
        let obj = {
            let mut pool = self.pool.lock().unwrap();
            pool.pop_front().unwrap_or_else(|| (self.factory)())
        };

        PooledObject {
            object: Some(obj),
            pool: Arc::clone(&self.pool),
            max_size: self.max_size,
        }
    }

    /// Get the current pool size
    pub fn size(&self) -> usize {
        self.pool.lock().unwrap().len()
    }

    /// Clear the pool
    pub fn clear(&self) {
        self.pool.lock().unwrap().clear();
    }
}

/// A pooled object that returns to the pool when dropped
pub struct PooledObject<T> {
    object: Option<T>,
    pool: Arc<Mutex<VecDeque<T>>>,
    max_size: usize,
}

impl<T> PooledObject<T> {
    /// Get a reference to the pooled object
    pub fn get(&self) -> &T {
        self.object.as_ref().unwrap()
    }

    /// Get a mutable reference to the pooled object
    pub fn get_mut(&mut self) -> &mut T {
        self.object.as_mut().unwrap()
    }
}

impl<T> Drop for PooledObject<T> {
    fn drop(&mut self) {
        if let Some(obj) = self.object.take() {
            let mut pool = self.pool.lock().unwrap();
            if pool.len() < self.max_size {
                pool.push_back(obj);
            }
        }
    }
}

impl<T> std::ops::Deref for PooledObject<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.get()
    }
}

impl<T> std::ops::DerefMut for PooledObject<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.get_mut()
    }
}

/// Thread-local object pool for single-threaded performance
pub struct ThreadLocalPool<T> {
    pool: RefCell<VecDeque<T>>,
    factory: Box<dyn Fn() -> T>,
    max_size: usize,
}

impl<T> ThreadLocalPool<T> {
    /// Create a new thread-local pool
    pub fn new<F>(factory: F, max_size: usize) -> Self
    where
        F: Fn() -> T + 'static,
    {
        Self {
            pool: RefCell::new(VecDeque::new()),
            factory: Box::new(factory),
            max_size,
        }
    }

    /// Get an object from the pool
    pub fn get(&self) -> ThreadLocalPooledObject<T> {
        let obj = {
            let mut pool = self.pool.borrow_mut();
            pool.pop_front().unwrap_or_else(|| (self.factory)())
        };

        ThreadLocalPooledObject {
            object: Some(obj),
            pool: &self.pool,
            max_size: self.max_size,
        }
    }

    /// Get the current pool size
    pub fn size(&self) -> usize {
        self.pool.borrow().len()
    }

    /// Clear the pool
    pub fn clear(&self) {
        self.pool.borrow_mut().clear();
    }
}

/// Thread-local pooled object
pub struct ThreadLocalPooledObject<'a, T> {
    object: Option<T>,
    pool: &'a RefCell<VecDeque<T>>,
    max_size: usize,
}

impl<'a, T> ThreadLocalPooledObject<'a, T> {
    /// Get a reference to the pooled object
    pub fn get(&self) -> &T {
        self.object.as_ref().unwrap()
    }

    /// Get a mutable reference to the pooled object
    pub fn get_mut(&mut self) -> &mut T {
        self.object.as_mut().unwrap()
    }
}

impl<'a, T> Drop for ThreadLocalPooledObject<'a, T> {
    fn drop(&mut self) {
        if let Some(obj) = self.object.take() {
            let mut pool = self.pool.borrow_mut();
            if pool.len() < self.max_size {
                pool.push_back(obj);
            }
        }
    }
}

impl<'a, T> std::ops::Deref for ThreadLocalPooledObject<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.get()
    }
}

impl<'a, T> std::ops::DerefMut for ThreadLocalPooledObject<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.get_mut()
    }
}

/// Specialized vector pool for chess engine operations
pub struct VectorPool {
    pool: ThreadLocalPool<Array1<f32>>,
    vector_size: usize,
}

impl VectorPool {
    /// Create a new vector pool
    pub fn new(vector_size: usize, max_size: usize) -> Self {
        let pool = ThreadLocalPool::new(move || Array1::zeros(vector_size), max_size);

        Self { pool, vector_size }
    }

    /// Get a zeroed vector from the pool
    pub fn get_zeroed(&self) -> ThreadLocalPooledObject<Array1<f32>> {
        let mut vec = self.pool.get();
        vec.fill(0.0);
        vec
    }

    /// Get a vector from the pool (contents undefined)
    pub fn get(&self) -> ThreadLocalPooledObject<Array1<f32>> {
        self.pool.get()
    }

    /// Get the vector size
    pub fn vector_size(&self) -> usize {
        self.vector_size
    }

    /// Get the current pool size
    pub fn size(&self) -> usize {
        self.pool.size()
    }

    /// Clear the pool
    pub fn clear(&self) {
        self.pool.clear();
    }
}

/// Global vector pool manager
pub struct VectorPoolManager {
    pools: std::collections::HashMap<usize, VectorPool>,
    max_pool_size: usize,
}

impl VectorPoolManager {
    /// Create a new vector pool manager
    pub fn new(max_pool_size: usize) -> Self {
        Self {
            pools: std::collections::HashMap::new(),
            max_pool_size,
        }
    }

    /// Get or create a vector pool for a specific size
    pub fn get_pool(&mut self, vector_size: usize) -> &VectorPool {
        self.pools
            .entry(vector_size)
            .or_insert_with(|| VectorPool::new(vector_size, self.max_pool_size))
    }

    /// Clear all pools
    pub fn clear_all(&mut self) {
        for pool in self.pools.values() {
            pool.clear();
        }
    }
}

/// Thread-local vector pool instance
thread_local! {
    static VECTOR_POOL_MANAGER: RefCell<VectorPoolManager> = RefCell::new(VectorPoolManager::new(16));
}

/// Thread-local vector pool for efficient reuse
thread_local! {
    static VECTOR_POOL_1024: std::cell::RefCell<VecDeque<Array1<f32>>> = std::cell::RefCell::new(VecDeque::new());
    static VECTOR_POOL_512: std::cell::RefCell<VecDeque<Array1<f32>>> = std::cell::RefCell::new(VecDeque::new());
    static VECTOR_POOL_256: std::cell::RefCell<VecDeque<Array1<f32>>> = std::cell::RefCell::new(VecDeque::new());
    static VECTOR_POOL_128: std::cell::RefCell<VecDeque<Array1<f32>>> = std::cell::RefCell::new(VecDeque::new());
    static VECTOR_POOL_64: std::cell::RefCell<VecDeque<Array1<f32>>> = std::cell::RefCell::new(VecDeque::new());
}

/// Get a vector from the appropriate thread-local pool
pub fn get_vector(size: usize) -> Array1<f32> {
    match size {
        1024 => get_vector_from_pool(&VECTOR_POOL_1024, size),
        512 => get_vector_from_pool(&VECTOR_POOL_512, size),
        256 => get_vector_from_pool(&VECTOR_POOL_256, size),
        128 => get_vector_from_pool(&VECTOR_POOL_128, size),
        64 => get_vector_from_pool(&VECTOR_POOL_64, size),
        _ => Array1::zeros(size), // For non-standard sizes, just create new
    }
}

/// Get a zeroed vector from the thread-local pool
pub fn get_zeroed_vector(size: usize) -> Array1<f32> {
    let mut vec = get_vector(size);
    vec.fill(0.0);
    vec
}

/// Helper function to get vector from specific pool
fn get_vector_from_pool(
    pool: &'static std::thread::LocalKey<std::cell::RefCell<VecDeque<Array1<f32>>>>,
    size: usize,
) -> Array1<f32> {
    pool.with(|pool_ref| {
        let mut pool = pool_ref.borrow_mut();
        pool.pop_front().unwrap_or_else(|| Array1::zeros(size))
    })
}

/// Return a vector to the appropriate thread-local pool
pub fn return_vector(mut vec: Array1<f32>) {
    let size = vec.len();

    // Only pool commonly used sizes to prevent memory bloat
    let pool = match size {
        1024 => Some(&VECTOR_POOL_1024),
        512 => Some(&VECTOR_POOL_512),
        256 => Some(&VECTOR_POOL_256),
        128 => Some(&VECTOR_POOL_128),
        64 => Some(&VECTOR_POOL_64),
        _ => None,
    };

    if let Some(pool) = pool {
        // Reset the vector to zeros for reuse
        vec.fill(0.0);

        pool.with(|pool_ref| {
            let mut pool = pool_ref.borrow_mut();

            // Limit pool size to prevent memory bloat (max 10 vectors per size)
            if pool.len() < 10 {
                pool.push_back(vec);
            }
            // If pool is full, just drop the vector
        });
    }
    // For non-standard sizes, just let the vector drop
}

/// RAII wrapper for automatic return to pool
pub struct PooledVector {
    vec: Option<Array1<f32>>,
}

impl PooledVector {
    /// Create a new pooled vector
    pub fn new(size: usize) -> Self {
        Self {
            vec: Some(get_vector(size)),
        }
    }

    /// Create a new zeroed pooled vector
    pub fn zeroed(size: usize) -> Self {
        Self {
            vec: Some(get_zeroed_vector(size)),
        }
    }

    /// Get a reference to the underlying vector
    pub fn as_ref(&self) -> &Array1<f32> {
        self.vec.as_ref().expect("Vector should always be present")
    }

    /// Get a mutable reference to the underlying vector
    pub fn as_mut(&mut self) -> &mut Array1<f32> {
        self.vec.as_mut().expect("Vector should always be present")
    }

    /// Take ownership of the vector (prevents automatic return to pool)
    pub fn take(mut self) -> Array1<f32> {
        self.vec.take().expect("Vector should always be present")
    }
}

impl Drop for PooledVector {
    fn drop(&mut self) {
        if let Some(vec) = self.vec.take() {
            return_vector(vec);
        }
    }
}

impl std::ops::Deref for PooledVector {
    type Target = Array1<f32>;

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl std::ops::DerefMut for PooledVector {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut()
    }
}

/// Clear all thread-local vector pools (useful for testing and cleanup)
pub fn clear_vector_pools() {
    VECTOR_POOL_1024.with(|pool| pool.borrow_mut().clear());
    VECTOR_POOL_512.with(|pool| pool.borrow_mut().clear());
    VECTOR_POOL_256.with(|pool| pool.borrow_mut().clear());
    VECTOR_POOL_128.with(|pool| pool.borrow_mut().clear());
    VECTOR_POOL_64.with(|pool| pool.borrow_mut().clear());
}

/// Get statistics about thread-local vector pools
pub fn get_vector_pool_stats() -> std::collections::HashMap<usize, usize> {
    let mut stats = std::collections::HashMap::new();

    VECTOR_POOL_1024.with(|pool| {
        stats.insert(1024, pool.borrow().len());
    });
    VECTOR_POOL_512.with(|pool| {
        stats.insert(512, pool.borrow().len());
    });
    VECTOR_POOL_256.with(|pool| {
        stats.insert(256, pool.borrow().len());
    });
    VECTOR_POOL_128.with(|pool| {
        stats.insert(128, pool.borrow().len());
    });
    VECTOR_POOL_64.with(|pool| {
        stats.insert(64, pool.borrow().len());
    });

    stats
}

/// Pool for chess move vectors
pub type MovePool = ObjectPool<Vec<chess::ChessMove>>;

/// Create a move pool
pub fn create_move_pool(max_size: usize) -> MovePool {
    ObjectPool::new(Vec::new, max_size)
}

/// Pool for hash maps
pub type HashMapPool<K, V> = ObjectPool<std::collections::HashMap<K, V>>;

/// Create a hash map pool
pub fn create_hashmap_pool<K, V>(max_size: usize) -> HashMapPool<K, V>
where
    K: std::hash::Hash + Eq + 'static,
    V: 'static,
{
    ObjectPool::new(std::collections::HashMap::new, max_size)
}

/// Trait for resettable objects (objects that can be reused)
pub trait Resettable {
    /// Reset the object to its initial state
    fn reset(&mut self);
}

impl<T> Resettable for Vec<T> {
    fn reset(&mut self) {
        self.clear();
    }
}

impl<K, V> Resettable for std::collections::HashMap<K, V>
where
    K: std::hash::Hash + Eq,
{
    fn reset(&mut self) {
        self.clear();
    }
}

impl Resettable for Array1<f32> {
    fn reset(&mut self) {
        self.fill(0.0);
    }
}

/// Pool for resettable objects
pub struct ResettablePool<T: Resettable> {
    pool: ObjectPool<T>,
}

impl<T: Resettable> ResettablePool<T> {
    /// Create a new resettable pool
    pub fn new<F>(factory: F, max_size: usize) -> Self
    where
        F: Fn() -> T + Send + Sync + 'static,
    {
        Self {
            pool: ObjectPool::new(factory, max_size),
        }
    }

    /// Get a reset object from the pool
    pub fn get_reset(&self) -> PooledObject<T> {
        let mut obj = self.pool.get();
        obj.reset();
        obj
    }

    /// Get an object from the pool (contents undefined)
    pub fn get(&self) -> PooledObject<T> {
        self.pool.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_object_pool() {
        let pool = ObjectPool::new(|| Vec::<i32>::new(), 10);

        // Test getting and returning objects
        {
            let mut obj1 = pool.get();
            obj1.push(1);
            obj1.push(2);
            assert_eq!(pool.size(), 0);
        }

        // Object should be returned to pool
        assert_eq!(pool.size(), 1);

        // Test reusing object
        {
            let obj2 = pool.get();
            assert_eq!(obj2.len(), 2); // Should contain previous data
        }
    }

    #[test]
    fn test_thread_local_pool() {
        let pool = ThreadLocalPool::new(|| Vec::<i32>::new(), 5);

        {
            let mut obj = pool.get();
            obj.push(42);
            assert_eq!(pool.size(), 0);
        }

        assert_eq!(pool.size(), 1);

        {
            let obj = pool.get();
            assert_eq!(obj.len(), 1);
            assert_eq!(obj[0], 42);
        }
    }

    #[test]
    fn test_vector_pool() {
        let pool = VectorPool::new(100, 5);

        {
            let mut vec = pool.get_zeroed();
            vec[0] = 1.0;
            vec[1] = 2.0;
            assert_eq!(pool.size(), 0);
        }

        assert_eq!(pool.size(), 1);

        {
            let vec = pool.get_zeroed();
            assert_eq!(vec[0], 0.0); // Should be zeroed
            assert_eq!(vec[1], 0.0);
        }
    }

    #[test]
    fn test_resettable_pool() {
        let pool = ResettablePool::new(|| Vec::<i32>::new(), 3);

        {
            let mut obj = pool.get_reset();
            obj.push(1);
            obj.push(2);
        }

        {
            let obj = pool.get_reset();
            assert_eq!(obj.len(), 0); // Should be reset
        }
    }

    #[test]
    fn test_pool_max_size() {
        let pool = ObjectPool::new(|| Vec::<i32>::new(), 2);

        // Fill pool to capacity
        {
            let _obj1 = pool.get();
            let _obj2 = pool.get();
            let _obj3 = pool.get();
        }

        // Should only store 2 objects
        assert_eq!(pool.size(), 2);
    }

    #[test]
    fn test_global_vector_pool() {
        let vec1 = get_zeroed_vector(1024);
        assert_eq!(vec1.len(), 1024);

        let vec2 = get_vector(512);
        assert_eq!(vec2.len(), 512);
    }

    #[test]
    fn test_thread_local_vector_pooling() {
        // Clear pools to start fresh
        clear_vector_pools();

        // Get vectors of different sizes
        let vec1024 = get_vector(1024);
        let vec512 = get_vector(512);
        let vec256 = get_vector(256);

        assert_eq!(vec1024.len(), 1024);
        assert_eq!(vec512.len(), 512);
        assert_eq!(vec256.len(), 256);

        // Return vectors to pool
        return_vector(vec1024);
        return_vector(vec512);
        return_vector(vec256);

        // Check pool stats
        let stats = get_vector_pool_stats();
        assert_eq!(stats.get(&1024), Some(&1));
        assert_eq!(stats.get(&512), Some(&1));
        assert_eq!(stats.get(&256), Some(&1));

        // Get vectors again - should reuse from pool
        let vec1024_reused = get_vector(1024);
        let vec512_reused = get_vector(512);

        assert_eq!(vec1024_reused.len(), 1024);
        assert_eq!(vec512_reused.len(), 512);

        // Pool should now have one fewer vector
        let stats_after = get_vector_pool_stats();
        assert_eq!(stats_after.get(&1024), Some(&0));
        assert_eq!(stats_after.get(&512), Some(&0));
        assert_eq!(stats_after.get(&256), Some(&1)); // This one wasn't reused
    }

    #[test]
    fn test_pooled_vector_raii() {
        clear_vector_pools();

        // Create a pooled vector in scope
        {
            let mut pooled = PooledVector::new(1024);
            assert_eq!(pooled.len(), 1024);

            // Modify the vector
            pooled[0] = 42.0;
            assert_eq!(pooled[0], 42.0);
        } // pooled goes out of scope, should return to pool

        // Check that it was returned to pool
        let stats = get_vector_pool_stats();
        assert_eq!(stats.get(&1024), Some(&1));

        // Get the vector again - should be zeroed
        let vec = get_vector(1024);
        assert_eq!(vec[0], 0.0); // Should be reset to zero
    }

    #[test]
    fn test_pooled_vector_take() {
        clear_vector_pools();

        // Create a pooled vector and take ownership
        let pooled = PooledVector::new(512);
        let vec = pooled.take(); // Take ownership, won't return to pool

        assert_eq!(vec.len(), 512);

        // Pool should still be empty since we took ownership
        let stats = get_vector_pool_stats();
        assert_eq!(stats.get(&512), Some(&0));
    }

    #[test]
    fn test_pool_size_limit() {
        clear_vector_pools();

        // Return more vectors than the pool limit (10)
        for _ in 0..15 {
            let vec = get_vector(128);
            return_vector(vec);
        }

        // Pool should be limited to 10 vectors
        let stats = get_vector_pool_stats();
        let pool_size = stats.get(&128).unwrap_or(&0);
        // The pool should have at least 1 vector but no more than 10
        assert!(*pool_size > 0, "Pool should have at least 1 vector");
        assert!(
            *pool_size <= 10,
            "Pool size should be limited to 10, but got {}",
            pool_size
        );

        // Test that we can get vectors from the pool
        let vec = get_vector(128);
        assert_eq!(vec.len(), 128);
    }

    #[test]
    fn test_non_standard_size_vectors() {
        // Non-standard sizes should not be pooled
        let vec = get_vector(100); // Non-standard size
        assert_eq!(vec.len(), 100);

        // Return it (should not be pooled)
        return_vector(vec);

        // Pool stats should not include this size
        let stats = get_vector_pool_stats();
        assert_eq!(stats.get(&100), None);
    }

    #[test]
    fn test_zeroed_vector_function() {
        let vec = get_zeroed_vector(256);
        assert_eq!(vec.len(), 256);

        // All elements should be zero
        for &value in vec.iter() {
            assert_eq!(value, 0.0);
        }
    }
}
