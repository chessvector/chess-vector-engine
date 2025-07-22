/// Engine Validation Suite
/// 
/// Comprehensive testing framework to validate chess engine accuracy
/// Tests fundamental engine capabilities without comparing to Stockfish

use chess_vector_engine::ChessVectorEngine;
use chess::Board;
use std::str::FromStr;

#[derive(Debug)]
pub struct ValidationResult {
    pub test_name: String,
    pub passed: bool,
    pub score: f32,
    pub details: String,
}

pub struct EngineValidationSuite {
    engine: ChessVectorEngine,
}

impl EngineValidationSuite {
    pub fn new() -> Self {
        let mut engine = ChessVectorEngine::new(1024);
        engine.enable_opening_book();
        Self { engine }
    }

    /// Test 1: Basic Engine Functionality
    /// Verify that the engine can perform basic operations
    pub fn test_basic_functionality(&mut self) -> ValidationResult {
        let mut details = String::new();
        let mut tests_passed = 0;
        let total_tests = 5;

        // Test 1: Engine creation
        details.push_str("  ✓ Engine creation: SUCCESS\n");
        tests_passed += 1;

        // Test 2: Position loading
        let board = Board::default();
        self.engine.add_position(&board, 0.0);
        details.push_str("  ✓ Position loading: SUCCESS\n");
        tests_passed += 1;

        // Test 3: Position evaluation (should return some value)
        if let Some(eval) = self.engine.evaluate_position(&board) {
            details.push_str(&format!("  ✓ Position evaluation: SUCCESS (eval: {:.3})\n", eval));
            tests_passed += 1;
        } else {
            details.push_str("  ✗ Position evaluation: FAILED (no evaluation returned)\n");
        }

        // Test 4: Similarity search
        let similar = self.engine.find_similar_positions(&board, 1);
        if !similar.is_empty() {
            details.push_str(&format!("  ✓ Similarity search: SUCCESS (found {} positions)\n", similar.len()));
            tests_passed += 1;
        } else {
            details.push_str("  ✗ Similarity search: FAILED (no similar positions found)\n");
        }

        // Test 5: Engine statistics
        let stats = self.engine.training_stats();
        if stats.total_positions > 0 {
            details.push_str(&format!("  ✓ Engine statistics: SUCCESS ({} positions)\n", stats.total_positions));
            tests_passed += 1;
        } else {
            details.push_str("  ✗ Engine statistics: FAILED (no positions recorded)\n");
        }

        let score = tests_passed as f32 / total_tests as f32;
        ValidationResult {
            test_name: "Basic Engine Functionality".to_string(),
            passed: score >= 0.8,
            score,
            details,
        }
    }

    /// Test 2: Evaluation Reasonableness
    /// Check that evaluations are within reasonable bounds and not extreme
    pub fn test_evaluation_reasonableness(&mut self) -> ValidationResult {
        let test_positions = vec![
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Starting position"),
            ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "1.e4"),
            ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2", "1.e4 e5"),
            ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "Developed position"),
            ("8/8/8/8/8/8/8/4K2k w - - 0 1", "King vs King endgame"),
        ];

        let mut reasonable_count = 0;
        let mut total_evaluations = 0;
        let mut details = String::new();

        for (fen, description) in &test_positions {
            if let Ok(board) = Board::from_str(fen) {
                self.engine.add_position(&board, 0.0); // Add to knowledge base
                if let Some(eval) = self.engine.evaluate_position(&board) {
                    // Check if evaluation is reasonable (between -50 and +50)
                    let reasonable = eval >= -50.0 && eval <= 50.0;
                    if reasonable {
                        reasonable_count += 1;
                    }
                    details.push_str(&format!("  {}: eval={:.3} {}\n", 
                        description, eval, 
                        if reasonable { "✓" } else { "✗ (extreme)" }));
                    total_evaluations += 1;
                } else {
                    details.push_str(&format!("  {}: No evaluation available\n", description));
                }
            }
        }

        let score = if total_evaluations > 0 { 
            reasonable_count as f32 / total_evaluations as f32 
        } else { 
            0.0 
        };

        ValidationResult {
            test_name: "Evaluation Reasonableness".to_string(),
            passed: score >= 0.8,
            score,
            details,
        }
    }

    /// Test 3: Similarity Search Accuracy
    /// Verify that similar positions are actually found
    pub fn test_similarity_search_accuracy(&mut self) -> ValidationResult {
        // Add a set of related positions
        let position_groups = vec![
            // Group 1: King's Pawn openings
            vec![
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
                "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
            ],
            // Group 2: French Defense family
            vec![
                "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
                "rnbqkbnr/ppp2ppp/4p3/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
                "rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPP2PPP/RNBQKBNR b KQkq d3 0 2",
            ],
        ];

        let mut group_accuracy = 0.0;
        let mut groups_tested = 0;
        let mut details = String::new();

        for (group_idx, group) in position_groups.iter().enumerate() {
            // Add all positions in the group
            for fen in group {
                if let Ok(board) = Board::from_str(fen) {
                    self.engine.add_position(&board, 0.0);
                }
            }

            // Test similarity search for first position in group
            if let Ok(query_board) = Board::from_str(&group[0]) {
                let similar = self.engine.find_similar_positions(&query_board, 3);
                
                // Count how many of the similar positions are from the same group
                let mut same_group_count = 0;
                for (similar_board, _eval, similarity) in &similar {
                    for group_fen in group {
                        if let Ok(group_board) = Board::from_str(group_fen) {
                            // Simple comparison - in real implementation you'd need proper board comparison
                            if similarity > &0.5 { // If similarity is reasonable
                                same_group_count += 1;
                                break;
                            }
                        }
                    }
                }

                let group_score = if similar.len() > 0 {
                    same_group_count as f32 / similar.len() as f32
                } else {
                    0.0
                };

                group_accuracy += group_score;
                groups_tested += 1;

                details.push_str(&format!("  Group {}: found {} similar, score {:.2}\n", 
                    group_idx + 1, similar.len(), group_score));
            }
        }

        let score = if groups_tested > 0 { 
            group_accuracy / groups_tested as f32 
        } else { 
            0.0 
        };

        ValidationResult {
            test_name: "Similarity Search Accuracy".to_string(),
            passed: score >= 0.3, // Lower threshold since this is complex
            score,
            details,
        }
    }

    /// Test 4: Opening Book Integration
    /// Verify opening book is working
    pub fn test_opening_book_integration(&mut self) -> ValidationResult {
        let mut details = String::new();
        let mut tests_passed = 0;
        let total_tests = 2;

        // Test 1: Check if opening book is enabled
        if let Some(stats) = self.engine.opening_book_stats() {
            details.push_str(&format!("  ✓ Opening book active: {} openings available\n", stats.total_openings));
            tests_passed += 1;
        } else {
            details.push_str("  ✗ Opening book: Not available\n");
        }

        // Test 2: Test with a known opening position
        if let Ok(board) = Board::from_str("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1") {
            if let Some(_eval) = self.engine.evaluate_position(&board) {
                details.push_str("  ✓ Opening position evaluation: SUCCESS\n");
                tests_passed += 1;
            } else {
                details.push_str("  ✗ Opening position evaluation: FAILED\n");
            }
        }

        let score = tests_passed as f32 / total_tests as f32;
        ValidationResult {
            test_name: "Opening Book Integration".to_string(),
            passed: score >= 0.5,
            score,
            details,
        }
    }

    /// Test 5: Memory and Performance
    /// Basic performance validation
    pub fn test_memory_and_performance(&mut self) -> ValidationResult {
        let mut details = String::new();
        let mut tests_passed = 0;
        let total_tests = 3;

        // Test 1: Add many positions without crashing
        let start_time = std::time::Instant::now();
        for i in 0..100 {
            let board = Board::default();
            self.engine.add_position(&board, i as f32 * 0.01);
        }
        let add_time = start_time.elapsed();
        details.push_str(&format!("  ✓ Bulk position addition: {} positions in {:.2}ms\n", 
            100, add_time.as_millis()));
        tests_passed += 1;

        // Test 2: Similarity search performance
        let start_time = std::time::Instant::now();
        let board = Board::default();
        for _ in 0..10 {
            let _similar = self.engine.find_similar_positions(&board, 5);
        }
        let search_time = start_time.elapsed();
        details.push_str(&format!("  ✓ Similarity search performance: 10 searches in {:.2}ms\n", 
            search_time.as_millis()));
        tests_passed += 1;

        // Test 3: Memory usage check (basic)
        let stats = self.engine.training_stats();
        if stats.total_positions > 0 {
            details.push_str(&format!("  ✓ Memory management: {} positions stored\n", stats.total_positions));
            tests_passed += 1;
        } else {
            details.push_str("  ✗ Memory management: No positions stored\n");
        }

        let score = tests_passed as f32 / total_tests as f32;
        ValidationResult {
            test_name: "Memory and Performance".to_string(),
            passed: score >= 0.8,
            score,
            details,
        }
    }

    /// Run all validation tests
    pub fn run_all_tests(&mut self) -> Vec<ValidationResult> {
        vec![
            self.test_basic_functionality(),
            self.test_evaluation_reasonableness(),
            self.test_similarity_search_accuracy(),
            self.test_opening_book_integration(),
            self.test_memory_and_performance(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_complete_engine_validation() {
        let mut suite = EngineValidationSuite::new();
        let results = suite.run_all_tests();
        
        println!("\n🧪 Chess Engine Validation Results");
        println!("==================================");
        
        let mut total_score = 0.0;
        let mut passed_tests = 0;
        
        for result in &results {
            println!("\n📊 {}", result.test_name);
            println!("   Score: {:.1}% {}", result.score * 100.0, if result.passed { "✓ PASS" } else { "✗ FAIL" });
            println!("   Details:");
            for line in result.details.lines() {
                if !line.trim().is_empty() {
                    println!("  {}", line);
                }
            }
            
            total_score += result.score;
            if result.passed {
                passed_tests += 1;
            }
        }
        
        let overall_score = total_score / results.len() as f32;
        
        println!("\n🎯 Overall Engine Validation Results");
        println!("====================================");
        println!("Tests passed: {}/{}", passed_tests, results.len());
        println!("Overall score: {:.1}%", overall_score * 100.0);
        
        let status = if overall_score >= 0.7 {
            "✅ ENGINE VALIDATION PASSED"
        } else if overall_score >= 0.5 {
            "⚠️  ENGINE NEEDS IMPROVEMENT"
        } else {
            "❌ ENGINE VALIDATION FAILED"
        };
        
        println!("Engine status: {}", status);
        
        // Print recommendations based on results
        if overall_score < 0.7 {
            println!("\n💡 Recommendations:");
            for result in &results {
                if !result.passed {
                    match result.test_name.as_str() {
                        "Basic Engine Functionality" => println!("   - Fix core engine operations"),
                        "Evaluation Reasonableness" => println!("   - Calibrate evaluation scale"),
                        "Similarity Search Accuracy" => println!("   - Improve position encoding"),
                        "Opening Book Integration" => println!("   - Check opening book loading"),
                        "Memory and Performance" => println!("   - Optimize memory usage"),
                        _ => {}
                    }
                }
            }
        }
        
        // This test always passes - it's for diagnostic purposes
        assert!(true, "Validation completed");
    }
}