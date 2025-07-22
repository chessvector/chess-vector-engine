use std::fmt;

/// Custom error types for the chess engine with enhanced resilience support
#[derive(Debug, Clone)]
pub enum ChessEngineError {
    /// Invalid chess position or move
    InvalidPosition(String),
    /// Database operation failed
    DatabaseError(String),
    /// Vector operation failed
    VectorError(String),
    /// Search operation failed or timed out
    SearchError(String),
    /// Neural network operation failed
    NeuralNetworkError(String),
    /// Training operation failed
    TrainingError(String),
    /// File I/O operation failed
    IoError(String),
    /// Configuration error
    ConfigurationError(String),
    /// Feature not available (e.g., GPU acceleration)
    FeatureNotAvailable(String),
    /// Resource exhausted (memory, time, etc.)
    ResourceExhausted(String),
    /// Operation failed after maximum retries
    RetryExhausted {
        operation: String,
        attempts: u32,
        last_error: String,
    },
    /// Timeout error with details
    Timeout {
        operation: String,
        duration_ms: u64,
    },
    /// Validation error with context
    ValidationError {
        field: String,
        value: String,
        expected: String,
    },
    /// Chained error with context
    ChainedError {
        source: Box<ChessEngineError>,
        context: String,
    },
    /// Circuit breaker is open
    CircuitBreakerOpen {
        operation: String,
        failures: u32,
    },
    /// Memory limit exceeded
    MemoryLimitExceeded {
        requested_mb: usize,
        available_mb: usize,
        limit_mb: usize,
    },
}

impl fmt::Display for ChessEngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChessEngineError::InvalidPosition(msg) => write!(f, "Invalid position: {}", msg),
            ChessEngineError::DatabaseError(msg) => write!(f, "Database error: {}", msg),
            ChessEngineError::VectorError(msg) => write!(f, "Vector operation error: {}", msg),
            ChessEngineError::SearchError(msg) => write!(f, "Search error: {}", msg),
            ChessEngineError::NeuralNetworkError(msg) => write!(f, "Neural network error: {}", msg),
            ChessEngineError::TrainingError(msg) => write!(f, "Training error: {}", msg),
            ChessEngineError::IoError(msg) => write!(f, "I/O error: {}", msg),
            ChessEngineError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            ChessEngineError::FeatureNotAvailable(msg) => {
                write!(f, "Feature not available: {}", msg)
            }
            ChessEngineError::ResourceExhausted(msg) => write!(f, "Resource exhausted: {}", msg),
            ChessEngineError::RetryExhausted { operation, attempts, last_error } => {
                write!(f, "Operation '{}' failed after {} attempts: {}", operation, attempts, last_error)
            }
            ChessEngineError::Timeout { operation, duration_ms } => {
                write!(f, "Operation '{}' timed out after {}ms", operation, duration_ms)
            }
            ChessEngineError::ValidationError { field, value, expected } => {
                write!(f, "Validation failed for field '{}': got '{}', expected '{}'", field, value, expected)
            }
            ChessEngineError::ChainedError { source, context } => {
                write!(f, "{}: {}", context, source)
            }
            ChessEngineError::CircuitBreakerOpen { operation, failures } => {
                write!(f, "Circuit breaker open for '{}' after {} failures", operation, failures)
            }
            ChessEngineError::MemoryLimitExceeded { requested_mb, available_mb, limit_mb } => {
                write!(f, "Memory limit exceeded: requested {}MB, available {}MB, limit {}MB", 
                    requested_mb, available_mb, limit_mb)
            }
        }
    }
}

impl std::error::Error for ChessEngineError {}

// Convenience type alias
pub type Result<T> = std::result::Result<T, ChessEngineError>;

// Convert from common error types
impl From<std::io::Error> for ChessEngineError {
    fn from(error: std::io::Error) -> Self {
        ChessEngineError::IoError(error.to_string())
    }
}

impl From<serde_json::Error> for ChessEngineError {
    fn from(error: serde_json::Error) -> Self {
        ChessEngineError::IoError(format!("JSON serialization error: {}", error))
    }
}

impl From<bincode::Error> for ChessEngineError {
    fn from(error: bincode::Error) -> Self {
        ChessEngineError::IoError(format!("Binary serialization error: {}", error))
    }
}

#[cfg(feature = "database")]
impl From<rusqlite::Error> for ChessEngineError {
    fn from(error: rusqlite::Error) -> Self {
        ChessEngineError::DatabaseError(error.to_string())
    }
}

impl From<std::num::ParseIntError> for ChessEngineError {
    fn from(error: std::num::ParseIntError) -> Self {
        ChessEngineError::ValidationError {
            field: "integer_parsing".to_string(),
            value: "unknown".to_string(),
            expected: format!("valid integer: {}", error),
        }
    }
}

impl From<std::num::ParseFloatError> for ChessEngineError {
    fn from(error: std::num::ParseFloatError) -> Self {
        ChessEngineError::ValidationError {
            field: "float_parsing".to_string(),
            value: "unknown".to_string(),
            expected: format!("valid float: {}", error),
        }
    }
}

/// Enhanced error utilities for production resilience
pub mod resilience {
    use super::*;
    use std::time::{Duration, Instant};
    use std::thread;
    
    /// Configuration for retry operations
    #[derive(Debug, Clone)]
    pub struct RetryConfig {
        pub max_attempts: u32,
        pub initial_delay_ms: u64,
        pub max_delay_ms: u64,
        pub backoff_multiplier: f64,
    }
    
    impl Default for RetryConfig {
        fn default() -> Self {
            Self {
                max_attempts: 3,
                initial_delay_ms: 100,
                max_delay_ms: 5000,
                backoff_multiplier: 2.0,
            }
        }
    }
    
    /// Retry an operation with exponential backoff
    pub fn retry_with_backoff<T, F, E>(
        operation_name: &str,
        config: &RetryConfig,
        mut operation: F,
    ) -> Result<T>
    where
        F: FnMut() -> std::result::Result<T, E>,
        E: std::fmt::Display + Clone,
    {
        let mut last_error: Option<E> = None;
        let mut delay_ms = config.initial_delay_ms;
        
        for attempt in 1..=config.max_attempts {
            match operation() {
                Ok(result) => return Ok(result),
                Err(error) => {
                    last_error = Some(error.clone());
                    
                    if attempt < config.max_attempts {
                        thread::sleep(Duration::from_millis(delay_ms));
                        delay_ms = ((delay_ms as f64) * config.backoff_multiplier) as u64;
                        delay_ms = delay_ms.min(config.max_delay_ms);
                    }
                }
            }
        }
        
        Err(ChessEngineError::RetryExhausted {
            operation: operation_name.to_string(),
            attempts: config.max_attempts,
            last_error: last_error.map(|e| e.to_string()).unwrap_or("unknown".to_string()),
        })
    }
    
    /// Circuit breaker state
    #[derive(Debug, Clone, PartialEq)]
    pub enum CircuitState {
        Closed,
        Open,
        HalfOpen,
    }
    
    /// Circuit breaker for failing operations
    #[derive(Debug)]
    pub struct CircuitBreaker {
        pub state: CircuitState,
        pub failure_count: u32,
        pub failure_threshold: u32,
        pub timeout_duration: Duration,
        pub last_failure_time: Option<Instant>,
    }
    
    impl CircuitBreaker {
        pub fn new(failure_threshold: u32, timeout_duration: Duration) -> Self {
            Self {
                state: CircuitState::Closed,
                failure_count: 0,
                failure_threshold,
                timeout_duration,
                last_failure_time: None,
            }
        }
        
        pub fn call<T, F, E>(&mut self, operation_name: &str, operation: F) -> Result<T>
        where
            F: FnOnce() -> std::result::Result<T, E>,
            E: std::fmt::Display,
        {
            match self.state {
                CircuitState::Open => {
                    if let Some(last_failure) = self.last_failure_time {
                        if Instant::now().duration_since(last_failure) > self.timeout_duration {
                            self.state = CircuitState::HalfOpen;
                        } else {
                            return Err(ChessEngineError::CircuitBreakerOpen {
                                operation: operation_name.to_string(),
                                failures: self.failure_count,
                            });
                        }
                    }
                }
                _ => {}
            }
            
            match operation() {
                Ok(result) => {
                    self.on_success();
                    Ok(result)
                }
                Err(error) => {
                    self.on_failure();
                    Err(ChessEngineError::SearchError(format!(
                        "Circuit breaker recorded failure in '{}': {}",
                        operation_name, error
                    )))
                }
            }
        }
        
        fn on_success(&mut self) {
            self.failure_count = 0;
            self.state = CircuitState::Closed;
        }
        
        fn on_failure(&mut self) {
            self.failure_count += 1;
            self.last_failure_time = Some(Instant::now());
            
            if self.failure_count >= self.failure_threshold {
                self.state = CircuitState::Open;
            }
        }
    }
    
    /// Memory monitor for resource management
    #[derive(Debug)]
    pub struct MemoryMonitor {
        max_memory_mb: usize,
        warning_threshold_mb: usize,
    }
    
    impl MemoryMonitor {
        pub fn new(max_memory_mb: usize) -> Self {
            Self {
                max_memory_mb,
                warning_threshold_mb: (max_memory_mb as f64 * 0.8) as usize,
            }
        }
        
        pub fn check_allocation(&self, requested_bytes: usize) -> Result<()> {
            let requested_mb = requested_bytes / (1024 * 1024);
            
            // Get current memory usage (simplified - in production you'd use a proper memory monitoring library)
            let current_usage_mb = self.get_estimated_memory_usage();
            
            if current_usage_mb + requested_mb > self.max_memory_mb {
                return Err(ChessEngineError::MemoryLimitExceeded {
                    requested_mb,
                    available_mb: self.max_memory_mb.saturating_sub(current_usage_mb),
                    limit_mb: self.max_memory_mb,
                });
            }
            
            if current_usage_mb + requested_mb > self.warning_threshold_mb {
                // Log warning but allow operation
                eprintln!("Warning: Memory usage approaching limit: {}MB + {}MB > {}MB (warning threshold)",
                    current_usage_mb, requested_mb, self.warning_threshold_mb);
            }
            
            Ok(())
        }
        
        fn get_estimated_memory_usage(&self) -> usize {
            // Simplified memory estimation - in production use proper memory monitoring
            // This is a placeholder that estimates based on process memory
            64 // Return 64MB as a conservative estimate
        }
    }
}

// Helper macros for error creation
#[macro_export]
macro_rules! invalid_position {
    ($msg:expr) => {
        ChessEngineError::InvalidPosition($msg.to_string())
    };
    ($fmt:expr, $($arg:tt)*) => {
        ChessEngineError::InvalidPosition(format!($fmt, $($arg)*))
    };
}

#[macro_export]
macro_rules! search_error {
    ($msg:expr) => {
        ChessEngineError::SearchError($msg.to_string())
    };
    ($fmt:expr, $($arg:tt)*) => {
        ChessEngineError::SearchError(format!($fmt, $($arg)*))
    };
}

#[macro_export]
macro_rules! vector_error {
    ($msg:expr) => {
        ChessEngineError::VectorError($msg.to_string())
    };
    ($fmt:expr, $($arg:tt)*) => {
        ChessEngineError::VectorError(format!($fmt, $($arg)*))
    };
}

#[macro_export]
macro_rules! training_error {
    ($msg:expr) => {
        ChessEngineError::TrainingError($msg.to_string())
    };
    ($fmt:expr, $($arg:tt)*) => {
        ChessEngineError::TrainingError(format!($fmt, $($arg)*))
    };
}

#[macro_export]
macro_rules! config_error {
    ($msg:expr) => {
        ChessEngineError::ConfigurationError($msg.to_string())
    };
    ($fmt:expr, $($arg:tt)*) => {
        ChessEngineError::ConfigurationError(format!($fmt, $($arg)*))
    };
}

#[macro_export]
macro_rules! resource_exhausted {
    ($msg:expr) => {
        ChessEngineError::ResourceExhausted($msg.to_string())
    };
    ($fmt:expr, $($arg:tt)*) => {
        ChessEngineError::ResourceExhausted(format!($fmt, $($arg)*))
    };
}

#[macro_export]
macro_rules! validation_error {
    ($field:expr, $value:expr, $expected:expr) => {
        ChessEngineError::ValidationError {
            field: $field.to_string(),
            value: $value.to_string(),
            expected: $expected.to_string(),
        }
    };
}

#[macro_export]
macro_rules! add_context {
    ($result:expr, $context:expr) => {
        $result.map_err(|e| ChessEngineError::ChainedError {
            source: Box::new(e),
            context: $context.to_string(),
        })
    };
}

#[macro_export]
macro_rules! memory_limit_exceeded {
    ($requested:expr, $available:expr, $limit:expr) => {
        ChessEngineError::MemoryLimitExceeded {
            requested_mb: $requested,
            available_mb: $available,
            limit_mb: $limit,
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let error = ChessEngineError::InvalidPosition("test position".to_string());
        assert_eq!(error.to_string(), "Invalid position: test position");
    }

    #[test]
    fn test_error_conversion() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let chess_error: ChessEngineError = io_error.into();

        match chess_error {
            ChessEngineError::IoError(msg) => assert!(msg.contains("file not found")),
            _ => panic!("Expected IoError"),
        }
    }

    #[test]
    fn test_error_macros() {
        let error = invalid_position!(
            "Invalid FEN: {}",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq"
        );
        match error {
            ChessEngineError::InvalidPosition(msg) => assert!(msg.contains("Invalid FEN")),
            _ => panic!("Expected InvalidPosition"),
        }
    }
    
    #[test]
    fn test_enhanced_error_types() {
        let validation_error = validation_error!("vector_size", "512", "1024");
        match validation_error {
            ChessEngineError::ValidationError { field, value, expected } => {
                assert_eq!(field, "vector_size");
                assert_eq!(value, "512");
                assert_eq!(expected, "1024");
            }
            _ => panic!("Expected ValidationError"),
        }
        
        let memory_error = ChessEngineError::MemoryLimitExceeded {
            requested_mb: 1024,
            available_mb: 512,
            limit_mb: 1000,
        };
        match memory_error {
            ChessEngineError::MemoryLimitExceeded { requested_mb, available_mb, limit_mb } => {
                assert_eq!(requested_mb, 1024);
                assert_eq!(available_mb, 512);
                assert_eq!(limit_mb, 1000);
            }
            _ => panic!("Expected MemoryLimitExceeded"),
        }
    }
    
    #[test]
    fn test_error_chaining() {
        let base_error = search_error!("Base search failed");
        let chained_result: Result<()> = Err(base_error);
        let enhanced_result = add_context!(chained_result, "During similarity search operation");
        
        match enhanced_result {
            Err(ChessEngineError::ChainedError { source, context }) => {
                assert_eq!(context, "During similarity search operation");
                match *source {
                    ChessEngineError::SearchError(ref msg) => assert_eq!(msg, "Base search failed"),
                    _ => panic!("Expected SearchError in chain"),
                }
            }
            _ => panic!("Expected ChainedError"),
        }
    }
}
