use chess::{Board, Color, MoveGen, Piece, Square};
use chess_vector_engine::PositionEncoder;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};
use rayon::prelude::*;
use std::collections::HashMap;
use std::time::Instant;
use serde::{Deserialize, Serialize};

#[derive(Parser)]
#[command(name = "intelligent_curation")]
#[command(about = "Intelligently curate chess positions from large datasets")]
struct Cli {
    /// Input file with positions to curate
    #[arg(long)]
    input: String,
    
    /// Target number of positions to keep
    #[arg(long, default_value = "100000")]
    target_size: usize,
    
    /// Output file for curated positions
    #[arg(long)]
    output: String,
    
    /// Minimum tactical score threshold
    #[arg(long, default_value = "0.3")]
    min_tactical_score: f32,
    
    /// Similarity threshold for clustering (0.0-1.0)
    #[arg(long, default_value = "0.85")]
    similarity_threshold: f32,
    
    /// Enable detailed progress reporting
    #[arg(long)]
    _verbose: bool,
    
    /// Force GPU acceleration for similarity calculations
    #[arg(long)]
    force_gpu: bool,
}

#[derive(Debug, Clone)]
struct ScoredPosition {
    board: Board,
    evaluation: f32,
    tactical_score: f32,
    educational_score: f32,
    uniqueness_score: f32,
    total_score: f32,
    game_phase: GamePhase,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
enum GamePhase {
    Opening,
    Middlegame,
    Endgame,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let start_time = Instant::now();
    
    println!("🧠 Intelligent Chess Position Curation System");
    println!("📁 Input: {}", cli.input);
    println!("🎯 Target: {} positions", cli.target_size);
    println!("📊 Tactical threshold: {:.2}", cli.min_tactical_score);
    println!("🔍 Similarity threshold: {:.2}", cli.similarity_threshold);
    println!();
    
    // Initialize encoder for similarity calculations
    let encoder = PositionEncoder::new(1024);
    
    // Load input positions
    println!("📚 Loading positions from {}...", cli.input);
    let input_positions = load_positions(&cli.input)?;
    println!("✅ Loaded {} positions", input_positions.len());
    
    if input_positions.len() <= cli.target_size {
        println!("ℹ️  Input size ({}) already within target ({}), copying file", 
                 input_positions.len(), cli.target_size);
        save_positions(&input_positions, &cli.output)?;
        return Ok(());
    }
    
    // Run progressive curation pipeline
    let input_len = input_positions.len();
    let curated = progressive_curation(
        input_positions, 
        cli.target_size,
        &encoder,
        cli.min_tactical_score,
        cli.similarity_threshold,
        cli._verbose
    )?;
    
    // Save curated positions
    println!("💾 Saving {} curated positions to {}...", curated.len(), cli.output);
    save_positions(&curated, &cli.output)?;
    
    let total_time = start_time.elapsed();
    println!();
    println!("🎉 Curation completed successfully!");
    println!("⏱️  Total time: {:?}", total_time);
    println!("📊 Reduction ratio: {:.1}x", 
             input_len as f32 / curated.len() as f32);
    
    // Print curation statistics
    print_curation_stats(&curated);
    
    Ok(())
}

fn progressive_curation(
    positions: Vec<TrainingData>,
    target: usize,
    encoder: &PositionEncoder,
    min_tactical_score: f32,
    similarity_threshold: f32,
    _verbose: bool
) -> Result<Vec<TrainingData>, Box<dyn std::error::Error>> {
    
    let multi_progress = MultiProgress::new();
    
    // Stage 1: Quick elimination filter
    println!("🔍 Stage 1: Quick elimination filter");
    let stage1_pb = multi_progress.add(ProgressBar::new(positions.len() as u64));
    stage1_pb.set_style(
        ProgressStyle::default_bar()
            .template("   Filtering [{elapsed_precise}] [{bar:40.red/blue}] {pos}/{len} ({percent}%) {msg}")?
            .progress_chars("██░")
    );
    
    let filtered = quick_filter_pass(positions.clone(), &stage1_pb)?;
    stage1_pb.finish_with_message(format!("✅ Kept {} positions", filtered.len()));
    
    // Stage 2: Tactical and educational scoring
    println!("🎯 Stage 2: Intelligent scoring");
    let stage2_pb = multi_progress.add(ProgressBar::new(filtered.len() as u64));
    stage2_pb.set_style(
        ProgressStyle::default_bar()
            .template("   Scoring [{elapsed_precise}] [{bar:40.yellow/blue}] {pos}/{len} ({percent}%) {msg}")?
            .progress_chars("██░")
    );
    
    let scored = score_positions_parallel(filtered, encoder, min_tactical_score, &stage2_pb)?;
    stage2_pb.finish_with_message(format!("✅ Scored {} positions", scored.len()));
    
    // Stage 3: Similarity-based clustering
    if scored.len() > target * 2 {
        println!("🎭 Stage 3: Similarity clustering");
        let stage3_pb = multi_progress.add(ProgressBar::new(scored.len() as u64));
        stage3_pb.set_style(
            ProgressStyle::default_bar()
                .template("   Clustering [{elapsed_precise}] [{bar:40.green/blue}] {pos}/{len} ({percent}%) {msg}")?
                .progress_chars("██░")
        );
        
        let clustered = cluster_similar_positions(scored, encoder, similarity_threshold, &stage3_pb)?;
        stage3_pb.finish_with_message(format!("✅ Clustered to {} positions", clustered.len()));
        
        // Stage 4: Final intelligent selection
        println!("⚡ Stage 4: Final selection");
        let final_positions = intelligent_selection(clustered, target)?;
        
        println!("✅ Curation complete: {} → {} positions", 
                 positions.len(), final_positions.len());
        
        Ok(final_positions)
    } else {
        // Stage 3: Direct intelligent selection (no clustering needed)
        println!("⚡ Stage 3: Final selection (skipping clustering)");
        let final_positions = intelligent_selection(scored, target)?;
        
        println!("✅ Curation complete: {} → {} positions", 
                 positions.len(), final_positions.len());
        
        Ok(final_positions)
    }
}

fn quick_filter_pass(
    positions: Vec<TrainingData>, 
    progress: &ProgressBar
) -> Result<Vec<TrainingData>, Box<dyn std::error::Error>> {
    
    let filtered: Vec<_> = positions
        .into_par_iter()
        .enumerate()
        .filter_map(|(i, pos)| {
            if i % 1000 == 0 {
                progress.set_position(i as u64);
                progress.set_message(format!("filtering..."));
            }
            
            if is_worth_keeping(&pos.board) {
                Some(pos)
            } else {
                None
            }
        })
        .collect();
    
    progress.set_position(progress.length().unwrap_or(0));
    Ok(filtered)
}

fn is_worth_keeping(board: &Board) -> bool {
    // Quick elimination criteria
    
    // 1. Must have legal moves (not already mate/stalemate)
    if MoveGen::new_legal(board).count() == 0 {
        return false;
    }
    
    // 2. Not trivially winning/losing (>8 material advantage)
    let material_balance = calculate_material_balance(board);
    if material_balance.abs() > 8.0 {
        return false;
    }
    
    // 3. Not in the first 3 moves of ultra-common openings
    if is_trivial_opening_position(board) {
        return false;
    }
    
    // 4. Must have some pieces left (not bare kings)
    let piece_count = count_total_pieces(board);
    if piece_count < 6 {
        return false;
    }
    
    true
}

fn score_positions_parallel(
    positions: Vec<TrainingData>,
    encoder: &PositionEncoder,
    min_tactical_score: f32,
    progress: &ProgressBar
) -> Result<Vec<ScoredPosition>, Box<dyn std::error::Error>> {
    
    // Pre-encode all positions for similarity calculations
    let vectors: Vec<_> = positions.par_iter()
        .map(|pos| encoder.encode(&pos.board))
        .collect();
    
    let scored: Vec<_> = positions
        .into_par_iter()
        .enumerate()
        .filter_map(|(i, pos)| {
            if i % 500 == 0 {
                progress.set_position(i as u64);
                progress.set_message(format!("scoring..."));
            }
            
            let tactical_score = calculate_tactical_score(&pos.board);
            let educational_score = calculate_educational_value(&pos.board);
            let game_phase = determine_game_phase(&pos.board);
            
            // Calculate uniqueness by comparing to nearby positions
            let vectors_slice: Vec<Vec<f32>> = vectors.iter().map(|v| v.to_vec()).collect();
            let uniqueness_score = calculate_uniqueness_score(i, &vectors_slice, 0.9);
            
            // Weighted combination
            let total_score = tactical_score * 0.4 +
                             educational_score * 0.3 +
                             uniqueness_score * 0.3;
            
            if total_score >= min_tactical_score {
                Some(ScoredPosition {
                    board: pos.board,
                    evaluation: pos.evaluation,
                    tactical_score,
                    educational_score,
                    uniqueness_score,
                    total_score,
                    game_phase,
                })
            } else {
                None
            }
        })
        .collect();
    
    progress.set_position(progress.length().unwrap_or(0));
    Ok(scored)
}

fn calculate_tactical_score(board: &Board) -> f32 {
    let mut score: f32 = 0.0;
    
    // High-value tactical indicators
    if board.checkers().0 != 0 { score += 2.0; }  // Position is check
    if has_hanging_pieces(board) { score += 1.5; }  // Hanging pieces
    if has_pins_and_skewers(board) { score += 1.0; }  // Tactical motifs
    if is_promotion_imminent(board) { score += 1.0; }  // Pawn about to promote
    
    // Medium-value indicators
    let material_imbalance = calculate_material_balance(board).abs();
    if material_imbalance > 1.0 && material_imbalance < 3.0 { score += 0.5; }
    
    if has_weak_king_safety(board) { score += 0.5; }
    if has_advanced_passed_pawns(board) { score += 0.3; }
    if has_piece_activity_imbalance(board) { score += 0.2; }
    
    score.min(5.0)  // Cap at 5.0
}

fn calculate_educational_value(board: &Board) -> f32 {
    let mut value = 0.0;
    let phase = determine_game_phase(board);
    
    match phase {
        GamePhase::Opening => {
            if demonstrates_opening_principles(board) { value += 1.0; }
            if has_development_lesson(board) { value += 0.5; }
        }
        GamePhase::Middlegame => {
            value += count_strategic_themes(board) * 0.3;
            if has_pawn_structure_lesson(board) { value += 0.5; }
            if has_piece_coordination(board) { value += 0.3; }
        }
        GamePhase::Endgame => {
            if demonstrates_endgame_technique(board) { value += 1.5; }  // Endgames crucial
            if is_theoretical_endgame(board) { value += 1.0; }
        }
    }
    
    value.min(3.0)  // Cap at 3.0
}

fn calculate_uniqueness_score(index: usize, vectors: &[Vec<f32>], threshold: f32) -> f32 {
    let target_vector = &vectors[index];
    let mut max_similarity: f32 = 0.0;
    
    // Check similarity to nearby positions (optimization)
    let start = index.saturating_sub(50);
    let end = (index + 50).min(vectors.len());
    
    for (i, other_vector) in vectors[start..end].iter().enumerate() {
        if start + i == index { continue; }
        
        let similarity = cosine_similarity(target_vector.as_slice(), other_vector.as_slice());
        max_similarity = max_similarity.max(similarity);
        
        if max_similarity > threshold {
            break;  // Early exit if too similar
        }
    }
    
    1.0 - max_similarity  // Return uniqueness (inverse of similarity)
}

fn cluster_similar_positions(
    positions: Vec<ScoredPosition>,
    encoder: &PositionEncoder,
    threshold: f32,
    progress: &ProgressBar
) -> Result<Vec<ScoredPosition>, Box<dyn std::error::Error>> {
    
    let mut clusters: Vec<Vec<usize>> = Vec::new();
    let mut assigned = vec![false; positions.len()];
    
    let vectors: Vec<_> = positions.par_iter()
        .map(|pos| encoder.encode(&pos.board))
        .collect();
    
    for i in 0..positions.len() {
        if i % 1000 == 0 {
            progress.set_position(i as u64);
            progress.set_message(format!("clustering..."));
        }
        
        if assigned[i] { continue; }
        
        let mut cluster = vec![i];
        assigned[i] = true;
        
        // Find similar positions
        for j in (i + 1)..positions.len() {
            if assigned[j] { continue; }
            
            let similarity = cosine_similarity(vectors[i].as_slice().unwrap(), vectors[j].as_slice().unwrap());
            if similarity >= threshold {
                cluster.push(j);
                assigned[j] = true;
            }
        }
        
        clusters.push(cluster);
    }
    
    // Keep the best position from each cluster
    let clustered: Vec<_> = clusters
        .into_par_iter()
        .map(|cluster| {
            cluster.into_iter()
                .map(|idx| &positions[idx])
                .max_by(|a, b| a.total_score.partial_cmp(&b.total_score).unwrap())
                .unwrap()
                .clone()
        })
        .collect();
    
    progress.set_position(progress.length().unwrap_or(0));
    Ok(clustered)
}

fn intelligent_selection(
    mut scored: Vec<ScoredPosition>, 
    target: usize
) -> Result<Vec<TrainingData>, Box<dyn std::error::Error>> {
    
    // Sort by total score
    scored.sort_by(|a, b| b.total_score.partial_cmp(&a.total_score).unwrap());
    
    // Ensure balanced representation across game phases
    let opening_target = (target * 20) / 100;    // 20% openings
    let middlegame_target = (target * 50) / 100; // 50% middlegames
    let endgame_target = (target * 30) / 100;    // 30% endgames
    
    let mut selected = Vec::new();
    let mut phase_counts = HashMap::new();
    
    for pos in &scored {
        let current_count = phase_counts.get(&pos.game_phase).unwrap_or(&0);
        let target_for_phase = match pos.game_phase {
            GamePhase::Opening => opening_target,
            GamePhase::Middlegame => middlegame_target,
            GamePhase::Endgame => endgame_target,
        };
        
        if *current_count < target_for_phase || selected.len() < target {
            selected.push(TrainingData {
                board: pos.board,
                evaluation: pos.evaluation,
                depth: 1,
                game_id: 0,
            });
            phase_counts.insert(pos.game_phase.clone(), current_count + 1);
            
            if selected.len() >= target {
                break;
            }
        }
    }
    
    // Fill remaining slots with best available if needed
    if selected.len() < target {
        for pos in &scored {
            if selected.len() >= target { break; }
            
            let already_selected = selected.iter()
                .any(|s| s.board == pos.board);
            
            if !already_selected {
                selected.push(TrainingData {
                    board: pos.board,
                    evaluation: pos.evaluation,
                    depth: 1,
                    game_id: 0,
                });
            }
        }
    }
    
    println!("📊 Phase distribution:");
    for (phase, count) in phase_counts {
        println!("   {:?}: {} positions", phase, count);
    }
    
    Ok(selected)
}

// Helper functions for tactical analysis
fn has_hanging_pieces(board: &Board) -> bool {
    // Simplified hanging piece detection
    for square in chess::ALL_SQUARES {
        if let Some(piece) = board.piece_on(square) {
            if is_hanging(board, square, piece) {
                return true;
            }
        }
    }
    false
}

fn is_hanging(board: &Board, square: Square, piece: Piece) -> bool {
    // Simplified: piece is attacked and not defended
    let _color = board.color_on(square).unwrap();
    // Simplified implementation - just check if piece is undefended
    let piece_value = match piece {
        Piece::Pawn => 1.0,
        Piece::Knight | Piece::Bishop => 3.0,
        Piece::Rook => 5.0,
        Piece::Queen => 9.0,
        Piece::King => 100.0,
    };
    
    // Very basic check - consider it hanging if it's a valuable piece
    let attackers = true; // Simplified
    let defenders = piece_value < 3.0; // Simplified
    
    attackers && !defenders
}

fn has_pins_and_skewers(board: &Board) -> bool {
    // Simplified pin/skewer detection
    for square in chess::ALL_SQUARES {
        if let Some(piece) = board.piece_on(square) {
            if piece == Piece::Bishop || piece == Piece::Rook || piece == Piece::Queen {
                if creates_pin_or_skewer(board, square) {
                    return true;
                }
            }
        }
    }
    false
}

fn creates_pin_or_skewer(board: &Board, square: Square) -> bool {
    // Simplified implementation
    let piece = board.piece_on(square).unwrap();
    let _color = board.color_on(square).unwrap();
    
    let _attacks = match piece {
        Piece::Bishop => chess::get_bishop_rays(square),
        Piece::Rook => chess::get_rook_rays(square),
        Piece::Queen => return false, // Simplified implementation
        _ => return false,
    };
    
    false // Simplified implementation
}

fn is_promotion_imminent(board: &Board) -> bool {
    let pawns = board.pieces(Piece::Pawn);
    let white_pawns = pawns & board.color_combined(Color::White);
    let black_pawns = pawns & board.color_combined(Color::Black);
    
    // Check for pawns on 7th rank (about to promote)
    (white_pawns & chess::get_rank(chess::Rank::Seventh)).0 != 0 ||
    (black_pawns & chess::get_rank(chess::Rank::Second)).0 != 0
}

fn calculate_material_balance(board: &Board) -> f32 {
    let mut balance = 0.0;
    
    let piece_values = [
        (Piece::Pawn, 1.0),
        (Piece::Knight, 3.0),
        (Piece::Bishop, 3.0),
        (Piece::Rook, 5.0),
        (Piece::Queen, 9.0),
    ];
    
    for (piece, value) in piece_values {
        let white_count = (board.pieces(piece) & board.color_combined(Color::White)).0.count_ones() as f32;
        let black_count = (board.pieces(piece) & board.color_combined(Color::Black)).0.count_ones() as f32;
        balance += (white_count - black_count) * value;
    }
    
    balance
}

fn count_total_pieces(board: &Board) -> u32 {
    board.combined().0.count_ones()
}

fn is_trivial_opening_position(board: &Board) -> bool {
    // Check if this is one of the first few moves of ultra-common openings
    let fen = format!("{}", board);
    
    // Starting position
    if fen.starts_with("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq") {
        return true;
    }
    
    // Very common first moves
    let trivial_positions = [
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3", // 1.e4
        "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3", // 1.d4
        "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq", // 1.Nf3
    ];
    
    trivial_positions.iter().any(|&pos| fen.starts_with(pos))
}

fn determine_game_phase(board: &Board) -> GamePhase {
    let total_pieces = count_total_pieces(board);
    let queens = (board.pieces(Piece::Queen) & *board.combined()).0.count_ones();
    let major_pieces = (board.pieces(Piece::Rook) & *board.combined()).0.count_ones() + 
                      (board.pieces(Piece::Queen) & *board.combined()).0.count_ones();
    
    if total_pieces <= 10 || (queens == 0 && major_pieces <= 4) {
        GamePhase::Endgame
    } else if total_pieces >= 28 {
        GamePhase::Opening
    } else {
        GamePhase::Middlegame
    }
}

fn demonstrates_opening_principles(board: &Board) -> bool {
    // Check for common opening principles
    let center_control = has_center_control(board);
    let piece_development = has_good_development(board);
    let king_safety = has_castled_or_can_castle(board);
    
    center_control || piece_development || king_safety
}

fn has_development_lesson(board: &Board) -> bool {
    // Check if position teaches development
    let knights_developed = count_developed_knights(board);
    let bishops_developed = count_developed_bishops(board);
    
    knights_developed > 0 || bishops_developed > 0
}

fn count_strategic_themes(board: &Board) -> f32 {
    let mut themes = 0.0;
    
    if has_outposts(board) { themes += 1.0; }
    if has_weak_squares(board) { themes += 1.0; }
    if has_pawn_chains(board) { themes += 0.5; }
    if has_piece_activity_differences(board) { themes += 0.5; }
    
    themes
}

fn has_pawn_structure_lesson(board: &Board) -> bool {
    has_isolated_pawns(board) || has_doubled_pawns(board) || has_passed_pawns(board)
}

fn has_piece_coordination(board: &Board) -> bool {
    // Simplified coordination check
    count_total_pieces(board) > 16 && has_multiple_piece_attacks(board)
}

fn demonstrates_endgame_technique(board: &Board) -> bool {
    let phase = determine_game_phase(board);
    if phase != GamePhase::Endgame { return false; }
    
    has_king_activity(board) || has_pawn_endgame_patterns(board) || has_piece_endgame_patterns(board)
}

fn is_theoretical_endgame(board: &Board) -> bool {
    let total_pieces = count_total_pieces(board);
    total_pieces <= 7 && has_known_endgame_pattern(board)
}

// Additional helper functions (simplified implementations)
fn has_weak_king_safety(_board: &Board) -> bool { false } // Placeholder
fn has_advanced_passed_pawns(_board: &Board) -> bool { false } // Placeholder  
fn has_piece_activity_imbalance(_board: &Board) -> bool { false } // Placeholder
fn has_center_control(_board: &Board) -> bool { false } // Placeholder
fn has_good_development(_board: &Board) -> bool { false } // Placeholder
fn has_castled_or_can_castle(_board: &Board) -> bool { false } // Placeholder
fn count_developed_knights(_board: &Board) -> u32 { 0 } // Placeholder
fn count_developed_bishops(_board: &Board) -> u32 { 0 } // Placeholder
fn has_outposts(_board: &Board) -> bool { false } // Placeholder
fn has_weak_squares(_board: &Board) -> bool { false } // Placeholder
fn has_pawn_chains(_board: &Board) -> bool { false } // Placeholder
fn has_piece_activity_differences(_board: &Board) -> bool { false } // Placeholder
fn has_isolated_pawns(_board: &Board) -> bool { false } // Placeholder
fn has_doubled_pawns(_board: &Board) -> bool { false } // Placeholder
fn has_passed_pawns(_board: &Board) -> bool { false } // Placeholder
fn has_multiple_piece_attacks(_board: &Board) -> bool { false } // Placeholder
fn has_king_activity(_board: &Board) -> bool { false } // Placeholder
fn has_pawn_endgame_patterns(_board: &Board) -> bool { false } // Placeholder
fn has_piece_endgame_patterns(_board: &Board) -> bool { false } // Placeholder
fn has_known_endgame_pattern(_board: &Board) -> bool { false } // Placeholder

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

fn print_curation_stats(positions: &[TrainingData]) {
    println!();
    println!("📊 Curation Statistics:");
    println!("   Final position count: {}", positions.len());
    
    // Count by game phase
    let mut phase_counts = HashMap::new();
    for pos in positions {
        let phase = determine_game_phase(&pos.board);
        *phase_counts.entry(phase).or_insert(0) += 1;
    }
    
    for (phase, count) in phase_counts {
        let percentage = (count as f32 / positions.len() as f32) * 100.0;
        println!("   {:?}: {} ({:.1}%)", phase, count, percentage);
    }
}

// Data structures for loading/saving
#[derive(Debug, Clone)]
struct TrainingData {
    board: Board,
    evaluation: f32,
    depth: u8,
    game_id: usize,
}

fn load_positions(path: &str) -> Result<Vec<TrainingData>, Box<dyn std::error::Error>> {
    use chess_vector_engine::training::TrainingDataset;
    
    let dataset = TrainingDataset::load(path)?;
    let positions = dataset.data.into_iter()
        .map(|data| TrainingData {
            board: data.board,
            evaluation: data.evaluation,
            depth: data.depth as u8,
            game_id: data.game_id as usize,
        })
        .collect();
    
    Ok(positions)
}

fn save_positions(positions: &[TrainingData], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    use chess_vector_engine::training::{TrainingDataset, TrainingData as EngineTrainingData};
    
    let dataset_data: Vec<EngineTrainingData> = positions.iter()
        .map(|pos| EngineTrainingData {
            board: pos.board,
            evaluation: pos.evaluation,
            depth: pos.depth,
            game_id: pos.game_id,
        })
        .collect();
    
    let dataset = TrainingDataset { data: dataset_data };
    dataset.save(path)?;
    
    Ok(())
}