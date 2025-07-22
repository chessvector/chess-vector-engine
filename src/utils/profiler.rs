use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// High-performance profiler for chess engine operations
pub struct Profiler {
    timers: RwLock<HashMap<String, TimerData>>,
    counters: RwLock<HashMap<String, CounterData>>,
    memory_snapshots: RwLock<Vec<MemorySnapshot>>,
    start_time: Instant,
    enabled: bool,
}

/// Timer measurement data
#[derive(Debug, Clone)]
struct TimerData {
    total_time: Duration,
    count: u64,
    min_time: Duration,
    max_time: Duration,
    current_start: Option<Instant>,
}

/// Counter data
#[derive(Debug, Clone)]
struct CounterData {
    value: i64,
    min_value: i64,
    max_value: i64,
    total_increments: u64,
}

/// Memory usage snapshot
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    pub timestamp: Instant,
    pub allocated_bytes: usize,
    pub peak_bytes: usize,
    pub active_allocations: usize,
}

impl Profiler {
    /// Create a new profiler
    pub fn new() -> Self {
        Self {
            timers: RwLock::new(HashMap::new()),
            counters: RwLock::new(HashMap::new()),
            memory_snapshots: RwLock::new(Vec::new()),
            start_time: Instant::now(),
            enabled: true,
        }
    }

    /// Enable or disable profiling
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Start timing an operation
    pub fn start_timer(&self, name: &str) {
        if !self.enabled {
            return;
        }

        let mut timers = self.timers.write().unwrap();
        let timer = timers.entry(name.to_string()).or_insert_with(|| TimerData {
            total_time: Duration::ZERO,
            count: 0,
            min_time: Duration::MAX,
            max_time: Duration::ZERO,
            current_start: None,
        });

        timer.current_start = Some(Instant::now());
    }

    /// End timing an operation
    pub fn end_timer(&self, name: &str) {
        if !self.enabled {
            return;
        }

        let end_time = Instant::now();
        let mut timers = self.timers.write().unwrap();

        if let Some(timer) = timers.get_mut(name) {
            if let Some(start_time) = timer.current_start.take() {
                let elapsed = end_time.duration_since(start_time);

                timer.total_time += elapsed;
                timer.count += 1;
                timer.min_time = timer.min_time.min(elapsed);
                timer.max_time = timer.max_time.max(elapsed);
            }
        }
    }

    /// Time a closure
    pub fn time<F, R>(&self, name: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        self.start_timer(name);
        let result = f();
        self.end_timer(name);
        result
    }

    /// Increment a counter
    pub fn increment_counter(&self, name: &str, value: i64) {
        if !self.enabled {
            return;
        }

        let mut counters = self.counters.write().unwrap();
        let counter = counters
            .entry(name.to_string())
            .or_insert_with(|| CounterData {
                value: 0,
                min_value: i64::MAX,
                max_value: i64::MIN,
                total_increments: 0,
            });

        counter.value += value;
        counter.min_value = counter.min_value.min(counter.value);
        counter.max_value = counter.max_value.max(counter.value);
        counter.total_increments += 1;
    }

    /// Set a counter to a specific value
    pub fn set_counter(&self, name: &str, value: i64) {
        if !self.enabled {
            return;
        }

        let mut counters = self.counters.write().unwrap();
        let counter = counters
            .entry(name.to_string())
            .or_insert_with(|| CounterData {
                value: 0,
                min_value: i64::MAX,
                max_value: i64::MIN,
                total_increments: 0,
            });

        counter.value = value;
        counter.min_value = counter.min_value.min(value);
        counter.max_value = counter.max_value.max(value);
    }

    /// Take a memory snapshot
    pub fn memory_snapshot(
        &self,
        allocated_bytes: usize,
        peak_bytes: usize,
        active_allocations: usize,
    ) {
        if !self.enabled {
            return;
        }

        let mut snapshots = self.memory_snapshots.write().unwrap();
        snapshots.push(MemorySnapshot {
            timestamp: Instant::now(),
            allocated_bytes,
            peak_bytes,
            active_allocations,
        });

        // Keep only last 1000 snapshots
        if snapshots.len() > 1000 {
            let len = snapshots.len();
            snapshots.drain(0..len - 1000);
        }
    }

    /// Get timer statistics
    pub fn get_timer_stats(&self, name: &str) -> Option<TimerStats> {
        let timers = self.timers.read().unwrap();
        timers.get(name).map(|timer| TimerStats {
            name: name.to_string(),
            total_time: timer.total_time,
            count: timer.count,
            average_time: if timer.count > 0 {
                timer.total_time / timer.count as u32
            } else {
                Duration::ZERO
            },
            min_time: if timer.min_time == Duration::MAX {
                Duration::ZERO
            } else {
                timer.min_time
            },
            max_time: timer.max_time,
        })
    }

    /// Get all timer statistics
    pub fn get_all_timer_stats(&self) -> Vec<TimerStats> {
        let timers = self.timers.read().unwrap();
        timers
            .iter()
            .map(|(name, timer)| TimerStats {
                name: name.clone(),
                total_time: timer.total_time,
                count: timer.count,
                average_time: if timer.count > 0 {
                    timer.total_time / timer.count as u32
                } else {
                    Duration::ZERO
                },
                min_time: if timer.min_time == Duration::MAX {
                    Duration::ZERO
                } else {
                    timer.min_time
                },
                max_time: timer.max_time,
            })
            .collect()
    }

    /// Get counter statistics
    pub fn get_counter_stats(&self, name: &str) -> Option<CounterStats> {
        let counters = self.counters.read().unwrap();
        counters.get(name).map(|counter| CounterStats {
            name: name.to_string(),
            current_value: counter.value,
            min_value: if counter.min_value == i64::MAX {
                0
            } else {
                counter.min_value
            },
            max_value: if counter.max_value == i64::MIN {
                0
            } else {
                counter.max_value
            },
            total_increments: counter.total_increments,
        })
    }

    /// Get all counter statistics
    pub fn get_all_counter_stats(&self) -> Vec<CounterStats> {
        let counters = self.counters.read().unwrap();
        counters
            .iter()
            .map(|(name, counter)| CounterStats {
                name: name.clone(),
                current_value: counter.value,
                min_value: if counter.min_value == i64::MAX {
                    0
                } else {
                    counter.min_value
                },
                max_value: if counter.max_value == i64::MIN {
                    0
                } else {
                    counter.max_value
                },
                total_increments: counter.total_increments,
            })
            .collect()
    }

    /// Get recent memory snapshots
    pub fn get_memory_snapshots(&self, last_n: Option<usize>) -> Vec<MemorySnapshot> {
        let snapshots = self.memory_snapshots.read().unwrap();
        if let Some(n) = last_n {
            if snapshots.len() > n {
                snapshots[snapshots.len() - n..].to_vec()
            } else {
                snapshots.clone()
            }
        } else {
            snapshots.clone()
        }
    }

    /// Generate a performance report
    pub fn generate_report(&self) -> PerformanceReport {
        let uptime = self.start_time.elapsed();
        let timer_stats = self.get_all_timer_stats();
        let counter_stats = self.get_all_counter_stats();
        let memory_snapshots = self.get_memory_snapshots(Some(100));

        PerformanceReport {
            uptime,
            timer_stats,
            counter_stats,
            memory_snapshots,
            report_time: Instant::now(),
        }
    }

    /// Reset all statistics
    pub fn reset(&self) {
        self.timers.write().unwrap().clear();
        self.counters.write().unwrap().clear();
        self.memory_snapshots.write().unwrap().clear();
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Timer statistics
#[derive(Debug, Clone)]
pub struct TimerStats {
    pub name: String,
    pub total_time: Duration,
    pub count: u64,
    pub average_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
}

/// Counter statistics
#[derive(Debug, Clone)]
pub struct CounterStats {
    pub name: String,
    pub current_value: i64,
    pub min_value: i64,
    pub max_value: i64,
    pub total_increments: u64,
}

/// Performance report
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    pub uptime: Duration,
    pub timer_stats: Vec<TimerStats>,
    pub counter_stats: Vec<CounterStats>,
    pub memory_snapshots: Vec<MemorySnapshot>,
    pub report_time: Instant,
}

/// RAII timer for automatic timing
pub struct ScopedTimer<'a> {
    profiler: &'a Profiler,
    name: String,
}

impl<'a> ScopedTimer<'a> {
    /// Create a new scoped timer
    pub fn new(profiler: &'a Profiler, name: &str) -> Self {
        profiler.start_timer(name);
        Self {
            profiler,
            name: name.to_string(),
        }
    }
}

impl<'a> Drop for ScopedTimer<'a> {
    fn drop(&mut self) {
        self.profiler.end_timer(&self.name);
    }
}

/// Macro for easy scoped timing
#[macro_export]
macro_rules! profile_scope {
    ($profiler:expr, $name:expr) => {
        let _timer = $crate::utils::profiler::ScopedTimer::new($profiler, $name);
    };
}

/// Performance monitoring for specific chess engine operations
pub struct ChessEngineProfiler {
    profiler: Profiler,
    search_metrics: RwLock<SearchMetrics>,
    evaluation_metrics: RwLock<EvaluationMetrics>,
}

#[derive(Debug, Clone, Default)]
struct SearchMetrics {
    nodes_searched: u64,
    positions_evaluated: u64,
    cache_hits: u64,
    cache_misses: u64,
    pruned_branches: u64,
    transposition_hits: u64,
}

#[derive(Debug, Clone, Default)]
struct EvaluationMetrics {
    nnue_evaluations: u64,
    pattern_evaluations: u64,
    hybrid_evaluations: u64,
    similarity_searches: u64,
    vector_operations: u64,
}

impl ChessEngineProfiler {
    /// Create a new chess engine profiler
    pub fn new() -> Self {
        Self {
            profiler: Profiler::new(),
            search_metrics: RwLock::new(SearchMetrics::default()),
            evaluation_metrics: RwLock::new(EvaluationMetrics::default()),
        }
    }

    /// Record search metrics
    pub fn record_search(&self, nodes: u64, positions: u64, cache_hits: u64, cache_misses: u64) {
        let mut metrics = self.search_metrics.write().unwrap();
        metrics.nodes_searched += nodes;
        metrics.positions_evaluated += positions;
        metrics.cache_hits += cache_hits;
        metrics.cache_misses += cache_misses;

        self.profiler
            .increment_counter("search.nodes_total", nodes as i64);
        self.profiler
            .increment_counter("search.positions_total", positions as i64);
        self.profiler.set_counter(
            "search.cache_hit_ratio",
            if cache_hits + cache_misses > 0 {
                (cache_hits * 100 / (cache_hits + cache_misses)) as i64
            } else {
                0
            },
        );
    }

    /// Record evaluation metrics
    pub fn record_evaluation(&self, eval_type: &str) {
        let mut metrics = self.evaluation_metrics.write().unwrap();
        match eval_type {
            "nnue" => metrics.nnue_evaluations += 1,
            "pattern" => metrics.pattern_evaluations += 1,
            "hybrid" => metrics.hybrid_evaluations += 1,
            "similarity" => metrics.similarity_searches += 1,
            "vector" => metrics.vector_operations += 1,
            _ => {}
        }

        self.profiler
            .increment_counter(&format!("eval.{}", eval_type), 1);
    }

    /// Time a search operation
    pub fn time_search<F, R>(&self, operation: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        self.profiler.time(&format!("search.{}", operation), f)
    }

    /// Time an evaluation operation
    pub fn time_evaluation<F, R>(&self, operation: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        self.profiler.time(&format!("eval.{}", operation), f)
    }

    /// Get chess-specific performance metrics
    pub fn get_chess_metrics(&self) -> ChessMetrics {
        let search_metrics = self.search_metrics.read().unwrap().clone();
        let evaluation_metrics = self.evaluation_metrics.read().unwrap().clone();

        ChessMetrics {
            search_metrics,
            evaluation_metrics,
            nodes_per_second: self.calculate_nodes_per_second(),
            evaluations_per_second: self.calculate_evaluations_per_second(),
        }
    }

    /// Calculate nodes per second
    fn calculate_nodes_per_second(&self) -> f64 {
        if let Some(timer_stats) = self.profiler.get_timer_stats("search.tactical") {
            if timer_stats.total_time.as_secs_f64() > 0.0 {
                let search_metrics = self.search_metrics.read().unwrap();
                return search_metrics.nodes_searched as f64 / timer_stats.total_time.as_secs_f64();
            }
        }
        0.0
    }

    /// Calculate evaluations per second
    fn calculate_evaluations_per_second(&self) -> f64 {
        if let Some(timer_stats) = self.profiler.get_timer_stats("eval.total") {
            if timer_stats.total_time.as_secs_f64() > 0.0 {
                let eval_metrics = self.evaluation_metrics.read().unwrap();
                let total_evals = eval_metrics.nnue_evaluations
                    + eval_metrics.pattern_evaluations
                    + eval_metrics.hybrid_evaluations;
                return total_evals as f64 / timer_stats.total_time.as_secs_f64();
            }
        }
        0.0
    }

    /// Generate comprehensive chess engine report
    pub fn generate_chess_report(&self) -> ChessEngineReport {
        let base_report = self.profiler.generate_report();
        let chess_metrics = self.get_chess_metrics();

        ChessEngineReport {
            base_report,
            chess_metrics,
        }
    }

    /// Get the underlying profiler
    pub fn profiler(&self) -> &Profiler {
        &self.profiler
    }
}

impl Default for ChessEngineProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Chess-specific performance metrics
#[derive(Debug, Clone)]
pub struct ChessMetrics {
    pub search_metrics: SearchMetrics,
    pub evaluation_metrics: EvaluationMetrics,
    pub nodes_per_second: f64,
    pub evaluations_per_second: f64,
}

/// Comprehensive chess engine performance report
#[derive(Debug, Clone)]
pub struct ChessEngineReport {
    pub base_report: PerformanceReport,
    pub chess_metrics: ChessMetrics,
}

/// Global profiler instance
static GLOBAL_PROFILER: std::sync::OnceLock<Arc<ChessEngineProfiler>> = std::sync::OnceLock::new();

/// Get the global profiler
pub fn global_profiler() -> &'static Arc<ChessEngineProfiler> {
    GLOBAL_PROFILER.get_or_init(|| Arc::new(ChessEngineProfiler::new()))
}

/// Simple benchmark utility
pub struct Benchmark {
    name: String,
    start_time: Instant,
    iterations: u64,
}

impl Benchmark {
    /// Start a new benchmark
    pub fn start(name: &str) -> Self {
        Self {
            name: name.to_string(),
            start_time: Instant::now(),
            iterations: 0,
        }
    }

    /// Record an iteration
    pub fn iteration(&mut self) {
        self.iterations += 1;
    }

    /// Finish the benchmark and return results
    pub fn finish(self) -> BenchmarkResult {
        let elapsed = self.start_time.elapsed();
        let iterations_per_second = if elapsed.as_secs_f64() > 0.0 {
            self.iterations as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        BenchmarkResult {
            name: self.name,
            total_time: elapsed,
            iterations: self.iterations,
            iterations_per_second,
            time_per_iteration: if self.iterations > 0 {
                elapsed / self.iterations as u32
            } else {
                Duration::ZERO
            },
        }
    }
}

/// Benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub total_time: Duration,
    pub iterations: u64,
    pub iterations_per_second: f64,
    pub time_per_iteration: Duration,
}

impl BenchmarkResult {
    /// Print benchmark results
    pub fn print(&self) {
        println!("Benchmark: {}", self.name);
        println!("  Total time: {:?}", self.total_time);
        println!("  Iterations: {}", self.iterations);
        println!("  Iterations/sec: {:.2}", self.iterations_per_second);
        println!("  Time/iteration: {:?}", self.time_per_iteration);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_profiler_timers() {
        let profiler = Profiler::new();

        profiler.start_timer("test_operation");
        thread::sleep(Duration::from_millis(10));
        profiler.end_timer("test_operation");

        let stats = profiler.get_timer_stats("test_operation").unwrap();
        assert_eq!(stats.count, 1);
        assert!(stats.total_time >= Duration::from_millis(10));
        assert!(stats.average_time >= Duration::from_millis(10));
    }

    #[test]
    fn test_profiler_counters() {
        let profiler = Profiler::new();

        profiler.increment_counter("test_counter", 5);
        profiler.increment_counter("test_counter", 3);
        profiler.set_counter("test_counter", 10);

        let stats = profiler.get_counter_stats("test_counter").unwrap();
        assert_eq!(stats.current_value, 10);
        assert_eq!(stats.max_value, 10);
        assert_eq!(stats.total_increments, 2);
    }

    #[test]
    fn test_scoped_timer() {
        let profiler = Profiler::new();

        {
            let _timer = ScopedTimer::new(&profiler, "scoped_test");
            thread::sleep(Duration::from_millis(5));
        }

        let stats = profiler.get_timer_stats("scoped_test").unwrap();
        assert_eq!(stats.count, 1);
        assert!(stats.total_time >= Duration::from_millis(5));
    }

    #[test]
    fn test_chess_engine_profiler() {
        let profiler = ChessEngineProfiler::new();

        profiler.record_search(1000, 500, 300, 200);
        profiler.record_evaluation("nnue");
        profiler.record_evaluation("pattern");

        let metrics = profiler.get_chess_metrics();
        assert_eq!(metrics.search_metrics.nodes_searched, 1000);
        assert_eq!(metrics.search_metrics.positions_evaluated, 500);
        assert_eq!(metrics.evaluation_metrics.nnue_evaluations, 1);
        assert_eq!(metrics.evaluation_metrics.pattern_evaluations, 1);
    }

    #[test]
    fn test_benchmark() {
        let mut bench = Benchmark::start("test_benchmark");

        for _ in 0..100 {
            bench.iteration();
            // Simulate some work
            thread::sleep(Duration::from_micros(100));
        }

        let result = bench.finish();
        assert_eq!(result.iterations, 100);
        assert!(result.total_time > Duration::from_millis(10));
        assert!(result.iterations_per_second > 0.0);
    }

    #[test]
    fn test_memory_snapshots() {
        let profiler = Profiler::new();

        profiler.memory_snapshot(1024, 2048, 10);
        profiler.memory_snapshot(1536, 2048, 15);

        let snapshots = profiler.get_memory_snapshots(Some(2));
        assert_eq!(snapshots.len(), 2);
        assert_eq!(snapshots[0].allocated_bytes, 1024);
        assert_eq!(snapshots[1].allocated_bytes, 1536);
    }

    #[test]
    fn test_performance_report() {
        let profiler = Profiler::new();

        profiler.start_timer("operation1");
        thread::sleep(Duration::from_millis(5));
        profiler.end_timer("operation1");

        profiler.increment_counter("counter1", 42);
        profiler.memory_snapshot(1024, 1024, 5);

        let report = profiler.generate_report();
        assert!(!report.timer_stats.is_empty());
        assert!(!report.counter_stats.is_empty());
        assert!(!report.memory_snapshots.is_empty());
    }
}
