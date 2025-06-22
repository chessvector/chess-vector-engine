# Contributing to Chess Vector Engine

Thank you for your interest in contributing to Chess Vector Engine! This document provides guidelines and information for contributors.

## 🏗️ Open-Core Architecture

Chess Vector Engine uses an **open-core business model** with both open-source and commercial features. Please understand the project structure:

### 🆓 Open Source Contributions (MIT/Apache-2.0)
- **Core engine functionality** - Position encoding, basic similarity search, UCI engine
- **Opening book** - Expanding chess openings database
- **Basic tactical search** - Improvements to 6-ply search algorithms
- **Training data formats** - JSON and basic binary format support
- **Documentation** - API docs, examples, tutorials
- **Bug fixes** - Any fixes to existing open source features
- **Testing** - Unit tests, integration tests, benchmarks

### 💎 Commercial Features (Internal Development)
- **Advanced NNUE networks** - Neural network evaluation
- **GPU acceleration** - CUDA/Metal optimizations
- **Ultra-fast loading** - Memory-mapped files, advanced compression
- **Advanced search** - 10+ ply search with sophisticated pruning
- **License system** - Subscription and feature gating

## 🚀 Getting Started

### Prerequisites

- Rust 1.75.0 or later
- Git
- A GitHub account

### Setting Up Your Development Environment

1. **Fork the repository**
   ```bash
   # Fork via GitHub web interface, then clone your fork
   git clone https://github.com/YOUR_USERNAME/chess-vector-engine.git
   cd chess-vector-engine
   ```

2. **Install dependencies**
   ```bash
   # Build the project
   cargo build
   
   # Run tests to ensure everything works
   cargo test
   ```

3. **Verify functionality**
   ```bash
   # Run the demo
   cargo run --bin demo
   
   # Test the UCI engine
   cargo run --bin uci_engine
   
   # Analyze a position
   cargo run --bin analyze "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
   ```

## 📝 Contributing Guidelines

### Before You Start

1. **Check existing issues** - Look for existing issues or feature requests
2. **Discuss large changes** - Open an issue to discuss significant changes before implementing
3. **Understand the scope** - Focus on open source features unless you're a team member
4. **Follow the license model** - Respect the open-core architecture

### Contribution Process

1. **Create a branch**
   ```bash
   git checkout -b your-feature-branch
   ```

2. **Make your changes**
   - Write clean, documented code
   - Follow existing code style
   - Add tests for new functionality
   - Update documentation if needed

3. **Test your changes**
   ```bash
   # Run all tests
   cargo test
   
   # Run benchmarks if performance-related
   cargo run --bin benchmark
   
   # Check code formatting
   cargo fmt --check
   
   # Run clippy for linting
   cargo clippy
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add description of your change"
   ```

5. **Push and create PR**
   ```bash
   git push origin your-feature-branch
   ```
   Then create a pull request via GitHub interface.

### Code Style

- **Rust formatting**: Use `cargo fmt` to format your code
- **Linting**: Run `cargo clippy` and fix any warnings
- **Documentation**: Add rustdoc comments for public APIs
- **Error handling**: Use `Result<T, E>` for fallible operations
- **Testing**: Write unit tests for new functionality

### Commit Messages

Use conventional commit format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test improvements
- `refactor:` for code refactoring
- `perf:` for performance improvements

## 🧪 Testing

### Running Tests

```bash
# Run all tests
cargo test

# Run tests for a specific module
cargo test position_encoder

# Run tests with output
cargo test -- --nocapture

# Run benchmarks
cargo run --bin benchmark
```

### Writing Tests

- **Unit tests**: Place tests in the same file using `#[cfg(test)]`
- **Integration tests**: Place in `tests/` directory
- **Benchmarks**: Use criterion.rs in `benches/` directory

Example test:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_encoding() {
        let mut engine = ChessVectorEngine::new(1024);
        let board = Board::default();
        let vector = engine.encode_position(&board);
        assert_eq!(vector.len(), 1024);
    }
}
```

## 📚 Documentation

### Types of Documentation

- **Code comments**: For complex algorithms or business logic
- **Rustdoc**: For public APIs and modules
- **README updates**: For significant feature additions
- **Examples**: Demonstrate usage patterns

### Writing Documentation

```rust
/// Encodes a chess position into a high-dimensional vector.
/// 
/// # Arguments
/// 
/// * `board` - The chess position to encode
/// 
/// # Returns
/// 
/// A 1024-dimensional vector representing the position
/// 
/// # Examples
/// 
/// ```
/// use chess_vector_engine::ChessVectorEngine;
/// use chess::Board;
/// 
/// let mut engine = ChessVectorEngine::new(1024);
/// let board = Board::default();
/// let vector = engine.encode_position(&board);
/// ```
pub fn encode_position(&mut self, board: &Board) -> Vec<f32> {
    // Implementation
}
```

## 🐛 Reporting Issues

### Bug Reports

Use the bug report template and include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Rust version, etc.)
- Code sample if applicable

### Feature Requests

Use the feature request template and include:
- Problem statement
- Proposed solution
- Use cases
- Target license tier (open source preferred)

## 🎯 Areas for Contribution

### High-Priority Areas

1. **Opening Book Expansion**
   - Add more chess openings
   - Improve opening classification
   - Add opening statistics and evaluations

2. **Performance Optimization**
   - SIMD optimizations for vector operations
   - Memory usage improvements
   - Algorithm efficiency improvements

3. **Testing and Benchmarking**
   - Expand test coverage
   - Add more integration tests
   - Improve benchmark accuracy

4. **Documentation**
   - API documentation improvements
   - Tutorial and guide writing
   - Code example expansion

### Beginner-Friendly Issues

Look for issues labeled:
- `good-first-issue`
- `help-wanted`
- `documentation`
- `easy`

## 💡 Development Tips

### Understanding the Codebase

- **`src/lib.rs`** - Main library interface and ChessVectorEngine struct
- **`src/position_encoder.rs`** - Chess position to vector conversion
- **`src/similarity_search.rs`** - K-nearest neighbor search algorithms
- **`src/tactical_search.rs`** - Chess move search and evaluation
- **`src/opening_book.rs`** - Opening book implementation
- **`src/uci.rs`** - UCI protocol implementation

### Feature Gating

Open source contributors should avoid:
- GPU-specific code (marked with `#[cfg(feature = "gpu")]`)
- Advanced neural network features
- Commercial license verification code
- Ultra-fast loading optimizations

### Debugging

```bash
# Enable debug logging
RUST_LOG=debug cargo run --bin demo

# Run with debugging symbols
cargo build --features debug

# Use rust-gdb for debugging
rust-gdb target/debug/demo
```

## 🤝 Code Review Process

### What to Expect

1. **Automated checks** - CI will run tests, formatting, and linting
2. **Manual review** - Maintainers will review code quality and design
3. **Discussion** - Be prepared to discuss and iterate on your changes
4. **Testing** - Ensure your changes don't break existing functionality

### Review Criteria

- **Code quality** - Clean, readable, well-documented code
- **Test coverage** - Adequate testing for new functionality
- **Performance** - No significant performance regressions
- **Compatibility** - Maintains backward compatibility
- **License compliance** - Respects open-core model

## 📄 License and Legal

### Contributor License Agreement

By contributing, you agree that:
- Your contributions will be licensed under MIT OR Apache-2.0
- You have the right to license your contributions
- Your contributions are your original work

### Copyright

- Add copyright notices to new files
- Respect existing copyright notices
- Don't include copyrighted code without permission

## 🆘 Getting Help

### Community Support

- **GitHub Discussions** - Ask questions about development
- **Issues** - Report bugs or request features
- **Discord** - Join our community chat (link in README)

### Maintainer Contact

For complex contributions or design discussions, reach out to maintainers via:
- GitHub issues (preferred)
- Email (for sensitive topics)

## 🎉 Recognition

Contributors are recognized in:
- **README acknowledgments** - For significant contributions
- **Release notes** - For features and fixes
- **Hall of fame** - For outstanding contributors

Thank you for contributing to Chess Vector Engine! Every contribution, no matter how small, helps make the project better for everyone.