use chess_vector_engine::run_uci_engine;

/// UCI chess engine binary
///
/// This binary provides a UCI (Universal Chess Interface) compliant chess engine
/// that can be used with any UCI-compatible chess GUI such as:
/// - Arena
/// - ChessBase
/// - Fritz
/// - Scid vs. PC
/// - BanksiaGUI
/// - And many others
///
/// Usage:
/// 1. Compile: `cargo build --release --bin uci_engine`
/// 2. Add the resulting binary to your chess GUI as a new engine
/// 3. The engine will communicate using standard UCI protocol
///
/// Features:
/// - Hybrid evaluation combining pattern recognition and tactical search
/// - GPU acceleration when available (CUDA/Metal) with CPU fallback
/// - Opening book integration
/// - Configurable search depth and time management
/// - Advanced pattern-based move recommendations
/// - LSH-accelerated similarity search
///
/// UCI Options:
/// - Hash: Hash table size in MB (1-2048, default 128)
/// - Threads: Number of search threads (1-64, default 1)
/// - MultiPV: Number of principal variations (1-10, default 1)
/// - Pattern_Weight: Weight for pattern evaluation vs tactical (0-100, default 60)
/// - Tactical_Depth: Maximum tactical search depth (1-10, default 3)
/// - Pattern_Confidence_Threshold: Minimum confidence for pattern-only evaluation (0-100, default 75)
/// - Enable_LSH: Use LSH acceleration for similarity search (true/false, default true)
/// - Enable_GPU: Use GPU acceleration when available (true/false, default true)
fn main() {
    run_uci_engine();
}
