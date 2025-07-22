// Removed unused import
use ndarray::Array1;
use rayon::prelude::*;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Work-stealing thread pool for high-performance parallel operations
pub struct WorkStealingPool {
    sender: Sender<Task>,
    workers: Vec<WorkerHandle>,
    shutdown: Arc<Mutex<bool>>,
}

/// Task that can be executed in parallel
pub enum Task {
    VectorSimilarity {
        query: Array1<f32>,
        targets: Vec<Array1<f32>>,
        result_sender: Sender<Vec<f32>>,
    },
    BatchEvaluation {
        positions: Vec<String>, // FEN strings
        result_sender: Sender<Vec<f32>>,
    },
    DataProcessing {
        data: Vec<u8>,
        processor: Box<dyn Fn(&[u8]) -> Vec<u8> + Send + Sync>,
        result_sender: Sender<Vec<u8>>,
    },
    Shutdown,
}

/// Handle to a worker thread
struct WorkerHandle {
    handle: thread::JoinHandle<()>,
    id: usize,
}

impl WorkStealingPool {
    /// Create a new work-stealing pool with specified number of workers
    pub fn new(num_workers: usize) -> Self {
        let (sender, receiver) = mpsc::channel();
        let receiver = Arc::new(Mutex::new(receiver));
        let shutdown = Arc::new(Mutex::new(false));

        let mut workers = Vec::new();

        for id in 0..num_workers {
            let receiver = Arc::clone(&receiver);
            let shutdown = Arc::clone(&shutdown);

            let handle = thread::spawn(move || {
                Self::worker_loop(id, receiver, shutdown);
            });

            workers.push(WorkerHandle { handle, id });
        }

        Self {
            sender,
            workers,
            shutdown,
        }
    }

    /// Submit a task for parallel execution
    pub fn submit(&self, task: Task) -> Result<(), &'static str> {
        self.sender.send(task).map_err(|_| "Failed to submit task")
    }

    /// Worker thread main loop
    fn worker_loop(
        _worker_id: usize,
        receiver: Arc<Mutex<Receiver<Task>>>,
        shutdown: Arc<Mutex<bool>>,
    ) {
        loop {
            // Check for shutdown signal
            if let Ok(shutdown_flag) = shutdown.lock() {
                if *shutdown_flag {
                    break;
                }
            }

            // Try to receive a task
            let task = {
                if let Ok(receiver) = receiver.lock() {
                    match receiver.try_recv() {
                        Ok(task) => Some(task),
                        Err(_) => None,
                    }
                } else {
                    None
                }
            };

            if let Some(task) = task {
                match task {
                    Task::VectorSimilarity {
                        query,
                        targets,
                        result_sender,
                    } => {
                        let similarities = Self::compute_vector_similarities(&query, &targets);
                        let _ = result_sender.send(similarities);
                    }
                    Task::BatchEvaluation {
                        positions,
                        result_sender,
                    } => {
                        let evaluations = Self::compute_batch_evaluations(&positions);
                        let _ = result_sender.send(evaluations);
                    }
                    Task::DataProcessing {
                        data,
                        processor,
                        result_sender,
                    } => {
                        let result = processor(&data);
                        let _ = result_sender.send(result);
                    }
                    Task::Shutdown => break,
                }
            } else {
                // No work available, sleep briefly
                thread::sleep(Duration::from_millis(1));
            }
        }
    }

    /// Compute vector similarities in parallel
    fn compute_vector_similarities(query: &Array1<f32>, targets: &[Array1<f32>]) -> Vec<f32> {
        use crate::utils::simd::SimdVectorOps;

        targets
            .par_iter()
            .map(|target| SimdVectorOps::cosine_similarity(query, target))
            .collect()
    }

    /// Compute batch evaluations (placeholder implementation)
    fn compute_batch_evaluations(_positions: &[String]) -> Vec<f32> {
        // This would integrate with the actual chess engine evaluation
        // For now, return dummy values
        vec![0.0; _positions.len()]
    }

    /// Shutdown the thread pool
    pub fn shutdown(self) {
        // Signal all workers to shutdown
        if let Ok(mut shutdown_flag) = self.shutdown.lock() {
            *shutdown_flag = true;
        }

        // Send shutdown tasks to wake up sleeping workers
        for _ in 0..self.workers.len() {
            let _ = self.sender.send(Task::Shutdown);
        }

        // Wait for all workers to finish
        for worker in self.workers {
            let _ = worker.handle.join();
        }
    }
}

/// Parallel batch processor for similarity searches
pub struct ParallelSimilarityProcessor {
    pool: WorkStealingPool,
    batch_size: usize,
}

impl ParallelSimilarityProcessor {
    /// Create a new parallel similarity processor
    pub fn new(num_workers: usize, batch_size: usize) -> Self {
        Self {
            pool: WorkStealingPool::new(num_workers),
            batch_size,
        }
    }

    /// Process similarity searches in parallel batches
    pub fn process_similarities(&self, query: Array1<f32>, targets: Vec<Array1<f32>>) -> Vec<f32> {
        let chunk_size = self.batch_size;
        let chunks: Vec<_> = targets.chunks(chunk_size).collect();
        let mut result_receivers = Vec::new();

        // Submit all chunks as parallel tasks
        for chunk in chunks {
            let (result_sender, result_receiver) = mpsc::channel();

            let task = Task::VectorSimilarity {
                query: query.clone(),
                targets: chunk.to_vec(),
                result_sender,
            };

            if self.pool.submit(task).is_ok() {
                result_receivers.push(result_receiver);
            }
        }

        // Collect results from all chunks
        let mut all_similarities = Vec::new();
        for receiver in result_receivers {
            if let Ok(similarities) = receiver.recv() {
                all_similarities.extend(similarities);
            }
        }

        all_similarities
    }
}

/// Parallel data pipeline for processing large datasets
pub struct ParallelDataPipeline<T, U> {
    input_queue: Arc<Mutex<Vec<T>>>,
    output_queue: Arc<Mutex<Vec<U>>>,
    processors: Vec<thread::JoinHandle<()>>,
    shutdown: Arc<Mutex<bool>>,
}

impl<T, U> ParallelDataPipeline<T, U>
where
    T: Send + 'static,
    U: Send + 'static,
{
    /// Create a new parallel data pipeline
    pub fn new<F>(num_processors: usize, processor: F) -> Self
    where
        F: Fn(T) -> U + Send + Sync + Clone + 'static,
    {
        let input_queue = Arc::new(Mutex::new(Vec::new()));
        let output_queue = Arc::new(Mutex::new(Vec::new()));
        let shutdown = Arc::new(Mutex::new(false));
        let mut processors = Vec::new();

        for _ in 0..num_processors {
            let input_queue = Arc::clone(&input_queue);
            let output_queue = Arc::clone(&output_queue);
            let shutdown = Arc::clone(&shutdown);
            let processor = processor.clone();

            let handle = thread::spawn(move || {
                loop {
                    // Check for shutdown
                    if let Ok(shutdown_flag) = shutdown.lock() {
                        if *shutdown_flag {
                            break;
                        }
                    }

                    // Get work item
                    let work_item = {
                        if let Ok(mut queue) = input_queue.lock() {
                            queue.pop()
                        } else {
                            None
                        }
                    };

                    if let Some(item) = work_item {
                        // Process the item
                        let result = processor(item);

                        // Store result
                        if let Ok(mut queue) = output_queue.lock() {
                            queue.push(result);
                        }
                    } else {
                        // No work available, sleep briefly
                        thread::sleep(Duration::from_millis(1));
                    }
                }
            });

            processors.push(handle);
        }

        Self {
            input_queue,
            output_queue,
            processors,
            shutdown,
        }
    }

    /// Add items to the processing queue
    pub fn enqueue(&self, items: Vec<T>) {
        if let Ok(mut queue) = self.input_queue.lock() {
            queue.extend(items);
        }
    }

    /// Retrieve processed results
    pub fn dequeue_results(&self) -> Vec<U> {
        if let Ok(mut queue) = self.output_queue.lock() {
            std::mem::take(&mut *queue)
        } else {
            Vec::new()
        }
    }

    /// Get the number of pending input items
    pub fn pending_count(&self) -> usize {
        self.input_queue.lock().map(|q| q.len()).unwrap_or(0)
    }

    /// Get the number of available results
    pub fn result_count(&self) -> usize {
        self.output_queue.lock().map(|q| q.len()).unwrap_or(0)
    }

    /// Shutdown the pipeline
    pub fn shutdown(self) {
        // Signal shutdown
        if let Ok(mut flag) = self.shutdown.lock() {
            *flag = true;
        }

        // Wait for all processors to finish
        for handle in self.processors {
            let _ = handle.join();
        }
    }
}

/// Parallel position evaluator with load balancing
pub struct ParallelPositionEvaluator {
    workers: Vec<EvaluationWorker>,
    current_worker: Arc<Mutex<usize>>,
}

struct EvaluationWorker {
    sender: Sender<EvaluationRequest>,
    _handle: thread::JoinHandle<()>,
}

struct EvaluationRequest {
    position: String, // FEN
    response: Sender<f32>,
}

impl ParallelPositionEvaluator {
    /// Create a new parallel position evaluator
    pub fn new(num_workers: usize) -> Self {
        let mut workers = Vec::new();

        for _ in 0..num_workers {
            let (sender, receiver) = mpsc::channel::<EvaluationRequest>();

            let handle = thread::spawn(move || {
                for request in receiver {
                    let evaluation = Self::evaluate_position_sync(&request.position);
                    let _ = request.response.send(evaluation);
                }
            });

            workers.push(EvaluationWorker {
                sender,
                _handle: handle,
            });
        }

        Self {
            workers,
            current_worker: Arc::new(Mutex::new(0)),
        }
    }

    /// Evaluate positions in parallel
    pub fn evaluate_positions(&self, positions: Vec<String>) -> Vec<f32> {
        let mut response_receivers = Vec::new();

        // Distribute work across workers using round-robin
        for position in positions {
            let worker_idx = {
                if let Ok(mut idx) = self.current_worker.lock() {
                    let current = *idx;
                    *idx = (current + 1) % self.workers.len();
                    current
                } else {
                    0
                }
            };

            let (response_sender, response_receiver) = mpsc::channel();
            let request = EvaluationRequest {
                position,
                response: response_sender,
            };

            if self.workers[worker_idx].sender.send(request).is_ok() {
                response_receivers.push(response_receiver);
            }
        }

        // Collect results
        let mut evaluations = Vec::new();
        for receiver in response_receivers {
            if let Ok(evaluation) = receiver.recv() {
                evaluations.push(evaluation);
            }
        }

        evaluations
    }

    /// Synchronous position evaluation (placeholder)
    fn evaluate_position_sync(_fen: &str) -> f32 {
        // This would integrate with the actual chess evaluation engine
        // For now, return a dummy value
        0.0
    }
}

/// High-performance parallel vector operations
pub struct ParallelVectorOps;

impl ParallelVectorOps {
    /// Parallel dot products for multiple vector pairs
    pub fn parallel_dot_products(vectors_a: &[Array1<f32>], vectors_b: &[Array1<f32>]) -> Vec<f32> {
        use crate::utils::simd::SimdVectorOps;

        vectors_a
            .par_iter()
            .zip(vectors_b.par_iter())
            .map(|(a, b)| SimdVectorOps::dot_product(a, b))
            .collect()
    }

    /// Parallel similarity matrix computation
    pub fn parallel_similarity_matrix(vectors: &[Array1<f32>]) -> Vec<Vec<f32>> {
        use crate::utils::simd::SimdVectorOps;

        // Use rayon's parallel iterator for outer loop
        vectors
            .par_iter()
            .enumerate()
            .map(|(_i, vec_a)| {
                // Inner loop can be sequential or parallel depending on size
                if vectors.len() > 100 {
                    vectors
                        .par_iter()
                        .map(|vec_b| SimdVectorOps::cosine_similarity(vec_a, vec_b))
                        .collect()
                } else {
                    vectors
                        .iter()
                        .map(|vec_b| SimdVectorOps::cosine_similarity(vec_a, vec_b))
                        .collect()
                }
            })
            .collect()
    }

    /// Parallel k-nearest neighbors search
    pub fn parallel_knn_search(
        query: &Array1<f32>,
        dataset: &[Array1<f32>],
        k: usize,
    ) -> Vec<(usize, f32)> {
        use crate::utils::simd::SimdVectorOps;

        // Compute similarities in parallel
        let similarities: Vec<(usize, f32)> = dataset
            .par_iter()
            .enumerate()
            .map(|(idx, vector)| {
                let similarity = SimdVectorOps::cosine_similarity(query, vector);
                (idx, similarity)
            })
            .collect();

        // Sort and take top k (this part is sequential)
        let mut similarities = similarities;
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(k);

        similarities
    }

    /// Parallel batch normalization
    pub fn parallel_batch_normalize(vectors: &mut [Array1<f32>]) {
        use crate::utils::simd::SimdVectorOps;

        vectors.par_iter_mut().for_each(|vector| {
            let norm = SimdVectorOps::squared_norm(vector).sqrt();
            if norm > 0.0 {
                *vector = SimdVectorOps::scale_vector(vector, 1.0 / norm);
            }
        });
    }
}

/// Performance monitoring for parallel operations
pub struct ParallelPerformanceMonitor {
    start_time: Instant,
    task_counts: Arc<Mutex<Vec<usize>>>,
    total_tasks: Arc<Mutex<usize>>,
}

impl ParallelPerformanceMonitor {
    /// Create a new performance monitor
    pub fn new(num_workers: usize) -> Self {
        Self {
            start_time: Instant::now(),
            task_counts: Arc::new(Mutex::new(vec![0; num_workers])),
            total_tasks: Arc::new(Mutex::new(0)),
        }
    }

    /// Record task completion for a worker
    pub fn record_task_completion(&self, worker_id: usize) {
        if let Ok(mut counts) = self.task_counts.lock() {
            if worker_id < counts.len() {
                counts[worker_id] += 1;
            }
        }

        if let Ok(mut total) = self.total_tasks.lock() {
            *total += 1;
        }
    }

    /// Get performance statistics
    pub fn get_stats(&self) -> ParallelPerformanceStats {
        let elapsed = self.start_time.elapsed();
        let total_tasks = self.total_tasks.lock().map(|t| *t).unwrap_or(0);
        let task_counts = self
            .task_counts
            .lock()
            .map(|counts| counts.clone())
            .unwrap_or_default();

        let tasks_per_second = if elapsed.as_secs_f64() > 0.0 {
            total_tasks as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        // Calculate load balance (standard deviation of task counts)
        let mean_tasks = if !task_counts.is_empty() {
            task_counts.iter().sum::<usize>() as f64 / task_counts.len() as f64
        } else {
            0.0
        };

        let variance = if !task_counts.is_empty() {
            task_counts
                .iter()
                .map(|&count| {
                    let diff = count as f64 - mean_tasks;
                    diff * diff
                })
                .sum::<f64>()
                / task_counts.len() as f64
        } else {
            0.0
        };

        let load_balance = variance.sqrt() / mean_tasks.max(1.0);

        ParallelPerformanceStats {
            elapsed_time: elapsed,
            total_tasks,
            tasks_per_second,
            worker_task_counts: task_counts,
            load_balance_factor: load_balance,
        }
    }
}

/// Performance statistics for parallel operations
#[derive(Debug, Clone)]
pub struct ParallelPerformanceStats {
    pub elapsed_time: Duration,
    pub total_tasks: usize,
    pub tasks_per_second: f64,
    pub worker_task_counts: Vec<usize>,
    pub load_balance_factor: f64, // Lower is better (0 = perfect balance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_work_stealing_pool() {
        let pool = WorkStealingPool::new(2);
        let (result_sender, result_receiver) = mpsc::channel();

        let query = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let targets = vec![
            Array1::from_vec(vec![1.0, 0.0, 0.0]),
            Array1::from_vec(vec![0.0, 1.0, 0.0]),
        ];

        let task = Task::VectorSimilarity {
            query,
            targets,
            result_sender,
        };

        pool.submit(task).unwrap();

        let result = result_receiver.recv_timeout(Duration::from_secs(1));
        assert!(result.is_ok());

        pool.shutdown();
    }

    #[test]
    fn test_parallel_similarity_processor() {
        let processor = ParallelSimilarityProcessor::new(2, 10);

        let query = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let targets = vec![
            Array1::from_vec(vec![1.0, 0.0, 0.0]),
            Array1::from_vec(vec![0.0, 1.0, 0.0]),
            Array1::from_vec(vec![0.0, 0.0, 1.0]),
        ];

        let similarities = processor.process_similarities(query, targets);
        assert_eq!(similarities.len(), 3);

        // First similarity should be 1.0 (identical vectors)
        assert!((similarities[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_parallel_data_pipeline() {
        let pipeline = ParallelDataPipeline::new(2, |x: i32| x * 2);

        // Add some work
        pipeline.enqueue(vec![1, 2, 3, 4, 5]);

        // Wait a bit for processing
        thread::sleep(Duration::from_millis(100));

        // Check results
        let results = pipeline.dequeue_results();
        assert!(!results.is_empty());

        pipeline.shutdown();
    }

    #[test]
    fn test_parallel_vector_ops() {
        let vectors_a = vec![
            Array1::from_vec(vec![1.0, 2.0]),
            Array1::from_vec(vec![3.0, 4.0]),
        ];
        let vectors_b = vec![
            Array1::from_vec(vec![2.0, 1.0]),
            Array1::from_vec(vec![1.0, 2.0]),
        ];

        let dot_products = ParallelVectorOps::parallel_dot_products(&vectors_a, &vectors_b);
        assert_eq!(dot_products.len(), 2);

        // Verify first dot product: [1,2] · [2,1] = 1*2 + 2*1 = 4
        assert!((dot_products[0] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_parallel_knn_search() {
        let query = Array1::from_vec(vec![1.0, 0.0]);
        let dataset = vec![
            Array1::from_vec(vec![1.0, 0.0]), // Identical
            Array1::from_vec(vec![0.0, 1.0]), // Orthogonal
            Array1::from_vec(vec![0.5, 0.5]), // Similar
        ];

        let results = ParallelVectorOps::parallel_knn_search(&query, &dataset, 2);
        assert_eq!(results.len(), 2);

        // First result should be the identical vector
        assert_eq!(results[0].0, 0);
        assert!((results[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_performance_monitor() {
        let monitor = ParallelPerformanceMonitor::new(3);

        // Simulate some task completions
        monitor.record_task_completion(0);
        monitor.record_task_completion(1);
        monitor.record_task_completion(0);
        monitor.record_task_completion(2);

        let stats = monitor.get_stats();
        assert_eq!(stats.total_tasks, 4);
        assert_eq!(stats.worker_task_counts[0], 2);
        assert_eq!(stats.worker_task_counts[1], 1);
        assert_eq!(stats.worker_task_counts[2], 1);
    }
}
