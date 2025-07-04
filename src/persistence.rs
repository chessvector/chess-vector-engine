#![allow(clippy::type_complexity)]
use rusqlite::{params, Connection, Result as SqlResult};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionData {
    pub fen: String,
    pub vector: Vec<f64>,
    pub evaluation: Option<f64>,
    pub compressed_vector: Option<Vec<f64>>,
    pub created_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSHHashFunction {
    pub random_vector: Vec<f64>,
    pub threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSHTableData {
    pub hash_functions: Vec<LSHHashFunction>,
    pub num_tables: usize,
    pub num_hash_functions: usize,
    pub vector_dim: usize,
}

pub struct Database {
    conn: Connection,
}

impl Database {
    pub fn new<P: AsRef<Path>>(db_path: P) -> SqlResult<Self> {
        let conn = Connection::open(db_path)?;

        // Enable basic optimizations
        conn.execute("PRAGMA foreign_keys=ON", [])?;

        let db = Database { conn };
        db.create_tables()?;
        Ok(db)
    }

    pub fn in_memory() -> SqlResult<Self> {
        let conn = Connection::open_in_memory()?;
        let db = Database { conn };
        db.create_tables()?;
        Ok(db)
    }

    fn create_tables(&self) -> SqlResult<()> {
        // Positions table - stores chess positions and their vectors
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fen TEXT NOT NULL UNIQUE,
                vector BLOB NOT NULL,
                evaluation REAL,
                compressed_vector BLOB,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL DEFAULT 0
            )",
            [],
        )?;

        // LSH tables data - stores LSH configuration and hash functions
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS lsh_config (
                id INTEGER PRIMARY KEY,
                num_tables INTEGER NOT NULL,
                num_hash_functions INTEGER NOT NULL,
                vector_dim INTEGER NOT NULL,
                hash_functions BLOB NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL DEFAULT 0
            )",
            [],
        )?;

        // LSH buckets - stores position assignments to hash buckets
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS lsh_buckets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_id INTEGER NOT NULL,
                bucket_hash TEXT NOT NULL,
                position_id INTEGER NOT NULL,
                UNIQUE(table_id, bucket_hash, position_id)
            )",
            [],
        )?;

        // Manifold model data - stores trained autoencoder weights
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS manifold_models (
                id INTEGER PRIMARY KEY,
                input_dim INTEGER NOT NULL,
                compressed_dim INTEGER NOT NULL,
                model_weights BLOB NOT NULL,
                training_metadata BLOB,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL DEFAULT 0
            )",
            [],
        )?;

        // Create indexes for better query performance
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_positions_fen ON positions(fen)",
            [],
        )?;

        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_lsh_buckets_table_bucket ON lsh_buckets(table_id, bucket_hash)",
            [],
        )?;

        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_positions_created_at ON positions(created_at)",
            [],
        )?;

        Ok(())
    }

    pub fn save_position(&self, position_data: &PositionData) -> SqlResult<i64> {
        let vector_bytes = bincode::serialize(&position_data.vector)
            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;

        let compressed_vector_bytes = position_data
            .compressed_vector
            .as_ref()
            .map(bincode::serialize)
            .transpose()
            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;

        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?
            .as_secs() as i64;

        self.conn.execute(
            "INSERT OR REPLACE INTO positions (fen, vector, evaluation, compressed_vector, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                position_data.fen,
                vector_bytes,
                position_data.evaluation,
                compressed_vector_bytes,
                position_data.created_at,
                current_time
            ],
        )?;

        Ok(self.conn.last_insert_rowid())
    }

    pub fn load_position(&self, fen: &str) -> SqlResult<Option<PositionData>> {
        let mut stmt = self.conn.prepare(
            "SELECT fen, vector, evaluation, compressed_vector, created_at 
             FROM positions WHERE fen = ?1",
        )?;

        let mut rows = stmt.query_map([fen], |row| {
            let vector_bytes: Vec<u8> = row.get(1)?;
            let vector: Vec<f64> = bincode::deserialize(&vector_bytes).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    1,
                    rusqlite::types::Type::Blob,
                    Box::new(e),
                )
            })?;

            let compressed_vector =
                if let Ok(Some(compressed_bytes)) = row.get::<_, Option<Vec<u8>>>(3) {
                    Some(bincode::deserialize(&compressed_bytes).map_err(|e| {
                        rusqlite::Error::FromSqlConversionFailure(
                            3,
                            rusqlite::types::Type::Blob,
                            Box::new(e),
                        )
                    })?)
                } else {
                    None
                };

            Ok(PositionData {
                fen: row.get(0)?,
                vector,
                evaluation: row.get(2)?,
                compressed_vector,
                created_at: row.get(4)?,
            })
        })?;

        match rows.next() {
            Some(Ok(position)) => Ok(Some(position)),
            Some(Err(e)) => Err(e),
            None => Ok(None),
        }
    }

    pub fn load_all_positions(&self) -> SqlResult<Vec<PositionData>> {
        let mut stmt = self.conn.prepare(
            "SELECT fen, vector, evaluation, compressed_vector, created_at 
             FROM positions ORDER BY created_at",
        )?;

        let rows = stmt.query_map([], |row| {
            let vector_bytes: Vec<u8> = row.get(1)?;
            let vector: Vec<f64> = bincode::deserialize(&vector_bytes).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    1,
                    rusqlite::types::Type::Blob,
                    Box::new(e),
                )
            })?;

            let compressed_vector =
                if let Ok(Some(compressed_bytes)) = row.get::<_, Option<Vec<u8>>>(3) {
                    Some(bincode::deserialize(&compressed_bytes).map_err(|e| {
                        rusqlite::Error::FromSqlConversionFailure(
                            3,
                            rusqlite::types::Type::Blob,
                            Box::new(e),
                        )
                    })?)
                } else {
                    None
                };

            Ok(PositionData {
                fen: row.get(0)?,
                vector,
                evaluation: row.get(2)?,
                compressed_vector,
                created_at: row.get(4)?,
            })
        })?;

        rows.collect()
    }

    pub fn save_lsh_config(&self, config: &LSHTableData) -> SqlResult<()> {
        let hash_functions_bytes = bincode::serialize(&config.hash_functions)
            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;

        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?
            .as_secs() as i64;

        self.conn.execute(
            "INSERT OR REPLACE INTO lsh_config (id, num_tables, num_hash_functions, vector_dim, hash_functions, created_at, updated_at)
             VALUES (1, ?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                config.num_tables,
                config.num_hash_functions,
                config.vector_dim,
                hash_functions_bytes,
                current_time,
                current_time
            ],
        )?;

        Ok(())
    }

    pub fn load_lsh_config(&self) -> SqlResult<Option<LSHTableData>> {
        let mut stmt = self.conn.prepare(
            "SELECT num_tables, num_hash_functions, vector_dim, hash_functions 
             FROM lsh_config WHERE id = 1",
        )?;

        let mut rows = stmt.query_map([], |row| {
            let hash_functions_bytes: Vec<u8> = row.get(3)?;
            let hash_functions: Vec<LSHHashFunction> = bincode::deserialize(&hash_functions_bytes)
                .map_err(|e| {
                    rusqlite::Error::FromSqlConversionFailure(
                        3,
                        rusqlite::types::Type::Blob,
                        Box::new(e),
                    )
                })?;

            Ok(LSHTableData {
                num_tables: row.get(0)?,
                num_hash_functions: row.get(1)?,
                vector_dim: row.get(2)?,
                hash_functions,
            })
        })?;

        match rows.next() {
            Some(Ok(config)) => Ok(Some(config)),
            Some(Err(e)) => Err(e),
            None => Ok(None),
        }
    }

    pub fn save_lsh_bucket(
        &self,
        table_id: usize,
        bucket_hash: &str,
        position_id: i64,
    ) -> SqlResult<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO lsh_buckets (table_id, bucket_hash, position_id)
             VALUES (?1, ?2, ?3)",
            params![table_id, bucket_hash, position_id],
        )?;
        Ok(())
    }

    pub fn load_lsh_buckets(&self, table_id: usize, bucket_hash: &str) -> SqlResult<Vec<i64>> {
        let mut stmt = self.conn.prepare(
            "SELECT position_id FROM lsh_buckets WHERE table_id = ?1 AND bucket_hash = ?2",
        )?;

        let rows = stmt.query_map(params![table_id, bucket_hash], |row| row.get(0))?;

        rows.collect()
    }

    pub fn clear_lsh_buckets(&self) -> SqlResult<()> {
        self.conn.execute("DELETE FROM lsh_buckets", [])?;
        Ok(())
    }

    pub fn get_position_count(&self) -> SqlResult<i64> {
        let mut stmt = self.conn.prepare("SELECT COUNT(*) FROM positions")?;
        let count: i64 = stmt.query_row([], |row| row.get(0))?;
        Ok(count)
    }

    pub fn vacuum(&self) -> SqlResult<()> {
        self.conn.execute("VACUUM", [])?;
        Ok(())
    }

    pub fn save_manifold_model(
        &self,
        input_dim: usize,
        compressed_dim: usize,
        model_weights: &[u8],
        training_metadata: Option<&[u8]>,
    ) -> SqlResult<()> {
        let metadata_bytes = training_metadata.unwrap_or(&[]);

        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?
            .as_secs() as i64;

        self.conn.execute(
            "INSERT OR REPLACE INTO manifold_models (id, input_dim, compressed_dim, model_weights, training_metadata, created_at, updated_at)
             VALUES (1, ?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                input_dim,
                compressed_dim,
                model_weights,
                metadata_bytes,
                current_time,
                current_time
            ],
        )?;
        Ok(())
    }

    pub fn load_manifold_model(&self) -> SqlResult<Option<(usize, usize, Vec<u8>, Vec<u8>)>> {
        let mut stmt = self.conn.prepare(
            "SELECT input_dim, compressed_dim, model_weights, training_metadata 
             FROM manifold_models WHERE id = 1",
        )?;

        let mut rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, usize>(0)?,
                row.get::<_, usize>(1)?,
                row.get::<_, Vec<u8>>(2)?,
                row.get::<_, Vec<u8>>(3)?,
            ))
        })?;

        match rows.next() {
            Some(Ok(model)) => Ok(Some(model)),
            Some(Err(e)) => Err(e),
            None => Ok(None),
        }
    }

    pub fn get_position_by_id(&self, id: i64) -> SqlResult<Option<PositionData>> {
        let mut stmt = self.conn.prepare(
            "SELECT fen, vector, evaluation, compressed_vector, created_at 
             FROM positions WHERE id = ?1",
        )?;

        let mut rows = stmt.query_map([id], |row| {
            let vector_bytes: Vec<u8> = row.get(1)?;
            let vector: Vec<f64> = bincode::deserialize(&vector_bytes).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    1,
                    rusqlite::types::Type::Blob,
                    Box::new(e),
                )
            })?;

            let compressed_vector =
                if let Ok(Some(compressed_bytes)) = row.get::<_, Option<Vec<u8>>>(3) {
                    Some(bincode::deserialize(&compressed_bytes).map_err(|e| {
                        rusqlite::Error::FromSqlConversionFailure(
                            3,
                            rusqlite::types::Type::Blob,
                            Box::new(e),
                        )
                    })?)
                } else {
                    None
                };

            Ok(PositionData {
                fen: row.get(0)?,
                vector,
                evaluation: row.get(2)?,
                compressed_vector,
                created_at: row.get(4)?,
            })
        })?;

        match rows.next() {
            Some(Ok(position)) => Ok(Some(position)),
            Some(Err(e)) => Err(e),
            None => Ok(None),
        }
    }

    /// Save multiple positions in a single transaction for much better performance
    pub fn save_positions_batch(&self, positions: &[PositionData]) -> SqlResult<usize> {
        if positions.is_empty() {
            return Ok(0);
        }

        let tx = self.conn.unchecked_transaction()?;

        {
            let mut stmt = tx.prepare(
                "INSERT OR REPLACE INTO positions (fen, vector, evaluation, compressed_vector, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)"
            )?;

            let current_time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?
                .as_secs() as i64;

            for position_data in positions {
                let vector_bytes = bincode::serialize(&position_data.vector)
                    .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;

                let compressed_vector_bytes = position_data
                    .compressed_vector
                    .as_ref()
                    .map(bincode::serialize)
                    .transpose()
                    .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;

                stmt.execute(params![
                    position_data.fen,
                    vector_bytes,
                    position_data.evaluation,
                    compressed_vector_bytes,
                    position_data.created_at,
                    current_time
                ])?;
            }
        }

        tx.commit()?;
        Ok(positions.len())
    }

    /// Load positions in batches for better memory efficiency
    pub fn load_positions_batch(
        &self,
        limit: usize,
        offset: usize,
    ) -> SqlResult<Vec<PositionData>> {
        let mut stmt = self.conn.prepare(
            "SELECT fen, vector, evaluation, compressed_vector, created_at 
             FROM positions ORDER BY id LIMIT ?1 OFFSET ?2",
        )?;

        let rows = stmt.query_map([limit, offset], |row| {
            let vector_bytes: Vec<u8> = row.get(1)?;
            let vector: Vec<f64> = bincode::deserialize(&vector_bytes).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    1,
                    rusqlite::types::Type::Blob,
                    Box::new(e),
                )
            })?;

            let compressed_vector =
                if let Ok(Some(compressed_bytes)) = row.get::<_, Option<Vec<u8>>>(3) {
                    Some(bincode::deserialize(&compressed_bytes).map_err(|e| {
                        rusqlite::Error::FromSqlConversionFailure(
                            3,
                            rusqlite::types::Type::Blob,
                            Box::new(e),
                        )
                    })?)
                } else {
                    None
                };

            Ok(PositionData {
                fen: row.get(0)?,
                vector,
                evaluation: row.get(2)?,
                compressed_vector,
                created_at: row.get(4)?,
            })
        })?;

        rows.collect()
    }

    /// Get the total count of positions in the database (as usize)
    pub fn get_total_position_count(&self) -> SqlResult<usize> {
        let mut stmt = self.conn.prepare("SELECT COUNT(*) FROM positions")?;
        let count: i64 = stmt.query_row([], |row| row.get(0))?;
        Ok(count as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_database_creation() {
        let db = Database::in_memory().unwrap();
        assert_eq!(db.get_position_count().unwrap(), 0);
    }

    #[test]
    fn test_position_storage() {
        let db = Database::in_memory().unwrap();

        let position = PositionData {
            fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string(),
            vector: vec![1.0, 2.0, 3.0],
            evaluation: Some(0.5),
            compressed_vector: Some(vec![0.1, 0.2]),
            created_at: 1234567890,
        };

        let id = db.save_position(&position).unwrap();
        assert!(id > 0);

        let loaded = db.load_position(&position.fen).unwrap().unwrap();
        assert_eq!(loaded.fen, position.fen);
        assert_eq!(loaded.vector, position.vector);
        assert_eq!(loaded.evaluation, position.evaluation);
        assert_eq!(loaded.compressed_vector, position.compressed_vector);
    }

    #[test]
    fn test_lsh_config_storage() {
        let db = Database::in_memory().unwrap();

        let config = LSHTableData {
            num_tables: 10,
            num_hash_functions: 5,
            vector_dim: 1024,
            hash_functions: vec![LSHHashFunction {
                random_vector: vec![1.0, -1.0, 0.5],
                threshold: 0.0,
            }],
        };

        db.save_lsh_config(&config).unwrap();

        let loaded = db.load_lsh_config().unwrap().unwrap();
        assert_eq!(loaded.num_tables, config.num_tables);
        assert_eq!(loaded.num_hash_functions, config.num_hash_functions);
        assert_eq!(loaded.vector_dim, config.vector_dim);
        assert_eq!(loaded.hash_functions.len(), config.hash_functions.len());
    }
}
