use chess_vector_engine::persistence::Database;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing database creation...");
    
    // Try to create an in-memory database first
    let db = Database::in_memory()?;
    println!("In-memory database created successfully!");
    
    // Test a simple operation
    let count = db.get_position_count()?;
    println!("Position count: {}", count);
    
    // Try to create a file database
    match Database::new("test.db") {
        Ok(db_file) => {
            println!("File database created successfully!");
            let count2 = db_file.get_position_count()?;
            println!("File database position count: {}", count2);
        }
        Err(e) => {
            println!("File database creation failed: {}", e);
            return Err(Box::new(e));
        }
    }
    
    println!("Database test completed successfully!");
    Ok(())
}