//! # Checkpoint module
//!
//! This module provides functionality for saving and loading OQNLP optimization state,
//! allowing users to resume long-running optimizations from where they left off.

use crate::types::{CheckpointConfig, OQNLPCheckpoint};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
/// Checkpoint-related errors
///
/// Errors related to checkpointing functionality, including IO errors,
/// serialization issues, and missing checkpoints.
pub enum CheckpointError {
    /// IO error when reading/writing checkpoint files
    #[error("IO error: {0}")]
    IoError(#[from] io::Error),

    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] bincode::Error),

    /// Checkpoint file not found
    #[error("Checkpoint file not found: {0}")]
    CheckpointNotFound(PathBuf),

    /// Invalid checkpoint data
    #[error("Invalid checkpoint data: {0}")]
    InvalidCheckpoint(String),
}

/// Checkpoint manager for OQNLP optimization
///
/// This struct manages saving and loading checkpoints for OQNLP optimizations.
/// It handles the configuration for checkpoints, including directory, naming, and frequency.
/// It provides methods to save the current optimization state and load the latest checkpoint.
pub struct CheckpointManager {
    config: CheckpointConfig,
}

impl CheckpointManager {
    /// Create a new checkpoint manager with the given configuration
    pub fn new(config: CheckpointConfig) -> Result<Self, CheckpointError> {
        if !config.checkpoint_dir.exists() {
            fs::create_dir_all(&config.checkpoint_dir)?;
        }

        Ok(Self { config })
    }

    /// Save a checkpoint to disk
    pub fn save_checkpoint(
        &self,
        checkpoint: &OQNLPCheckpoint,
        iteration: usize,
    ) -> Result<PathBuf, CheckpointError> {
        let filename = if self.config.keep_all {
            format!("{}_{:06}.bin", self.config.checkpoint_name, iteration)
        } else {
            format!("{}.bin", self.config.checkpoint_name)
        };

        let filepath = self.config.checkpoint_dir.join(filename);
        let encoded = bincode::serialize(checkpoint)?;
        fs::write(&filepath, encoded)?;

        Ok(filepath)
    }

    /// Load the latest checkpoint from disk
    pub fn load_latest_checkpoint(&self) -> Result<OQNLPCheckpoint, CheckpointError> {
        let checkpoint_path = if self.config.keep_all {
            self.find_latest_checkpoint()?
        } else {
            let filename = format!("{}.bin", self.config.checkpoint_name);
            self.config.checkpoint_dir.join(filename)
        };

        self.load_checkpoint_from_path(&checkpoint_path)
    }

    /// Load a specific checkpoint from a file path
    pub fn load_checkpoint_from_path(
        &self,
        path: &Path,
    ) -> Result<OQNLPCheckpoint, CheckpointError> {
        if !path.exists() {
            return Err(CheckpointError::CheckpointNotFound(path.to_path_buf()));
        }

        let encoded = fs::read(path)?;
        let checkpoint: OQNLPCheckpoint = bincode::deserialize(&encoded)?;

        Ok(checkpoint)
    }

    /// Check if a checkpoint exists
    pub fn checkpoint_exists(&self) -> bool {
        if self.config.keep_all {
            self.find_latest_checkpoint().is_ok()
        } else {
            let filename = format!("{}.bin", self.config.checkpoint_name);
            self.config.checkpoint_dir.join(filename).exists()
        }
    }

    /// Find the latest checkpoint file when keep_all is enabled
    fn find_latest_checkpoint(&self) -> Result<PathBuf, CheckpointError> {
        let entries = fs::read_dir(&self.config.checkpoint_dir)?;
        let pattern = format!("{}_", self.config.checkpoint_name);

        let mut latest_iteration = 0;
        let mut latest_path = None;

        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                if filename.starts_with(&pattern) && filename.ends_with(".bin") {
                    let iteration_str = &filename[pattern.len()..filename.len() - 4];
                    if let Ok(iteration) = iteration_str.parse::<usize>() {
                        if iteration > latest_iteration {
                            latest_iteration = iteration;
                            latest_path = Some(path);
                        }
                    }
                }
            }
        }

        latest_path
            .ok_or_else(|| CheckpointError::CheckpointNotFound(self.config.checkpoint_dir.clone()))
    }

    /// Clean up old checkpoint files (keep only the latest N files)
    pub fn cleanup_old_checkpoints(&self, keep_count: usize) -> Result<(), CheckpointError> {
        if !self.config.keep_all || keep_count == 0 {
            return Ok(());
        }

        let entries = fs::read_dir(&self.config.checkpoint_dir)?;
        let pattern = format!("{}_", self.config.checkpoint_name);

        let mut checkpoints = Vec::new();

        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                if filename.starts_with(&pattern) && filename.ends_with(".bin") {
                    let iteration_str = &filename[pattern.len()..filename.len() - 4];
                    if let Ok(iteration) = iteration_str.parse::<usize>() {
                        checkpoints.push((iteration, path));
                    }
                }
            }
        }

        checkpoints.sort_by(|a, b| b.0.cmp(&a.0));
        for (_, path) in checkpoints.iter().skip(keep_count) {
            fs::remove_file(path)?;
        }

        Ok(())
    }

    /// Get the checkpoint configuration
    pub fn config(&self) -> &CheckpointConfig {
        &self.config
    }
}

impl Default for CheckpointManager {
    fn default() -> Self {
        Self::new(CheckpointConfig::default()).unwrap()
    }
}

/// Read a checkpoint file directly from a given path
///
/// This is a convenience function that allows reading checkpoint files
/// without creating a `CheckpointManager` instance.
/// # Example
///
/// ```rust,no_run
/// use globalsearch::checkpoint::read_checkpoint_file;
/// use std::path::Path;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let checkpoint = read_checkpoint_file(Path::new("./checkpoints/optimization.bin"))?;
/// println!("Loaded checkpoint:");
/// println!("{}", checkpoint);
/// # Ok(())
/// # }
/// ```
pub fn read_checkpoint_file(path: &Path) -> Result<OQNLPCheckpoint, CheckpointError> {
    if !path.exists() {
        return Err(CheckpointError::CheckpointNotFound(path.to_path_buf()));
    }

    let encoded = fs::read(path)?;
    let checkpoint: OQNLPCheckpoint = bincode::deserialize(&encoded)?;

    Ok(checkpoint)
}

#[cfg(test)]
mod tests_checkpointing {
    use crate::checkpoint::{CheckpointConfig, CheckpointManager, OQNLPCheckpoint};
    use crate::types::{LocalSolution, OQNLPParams, SolutionSet};
    use ndarray::{array, Array1};
    use std::env;
    use std::fs;

    #[test]
    fn test_checkpoint_manager_creation() {
        let temp_dir = env::temp_dir().join("test_checkpoints");
        let config = CheckpointConfig {
            checkpoint_dir: temp_dir.clone(),
            checkpoint_name: "test".to_string(),
            save_frequency: 5,
            keep_all: false,
            auto_resume: true,
        };

        let _manager = CheckpointManager::new(config).unwrap();
        assert!(temp_dir.exists());

        // Cleanup
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_save_and_load_checkpoint() {
        let temp_dir = env::temp_dir().join("test_checkpoints_2");
        let config = CheckpointConfig {
            checkpoint_dir: temp_dir.clone(),
            checkpoint_name: "test".to_string(),
            save_frequency: 5,
            keep_all: false,
            auto_resume: true,
        };

        let manager = CheckpointManager::new(config).unwrap();

        // Create a test checkpoint
        let checkpoint = OQNLPCheckpoint {
            params: OQNLPParams::default(),
            current_iteration: 42,
            merit_threshold: 1.5,
            solution_set: Some(SolutionSet {
                solutions: Array1::from(vec![LocalSolution {
                    point: array![1.0, 2.0],
                    objective: -1.0,
                }]),
            }),
            reference_set: vec![array![1.0, 2.0], array![3.0, 4.0]],
            unchanged_cycles: 5,
            elapsed_time: 120.5,
            distance_filter_solutions: vec![],
            current_seed: 10,
            timestamp: "2025-01-01T00:00:00Z".to_string(),
        };

        // Save checkpoint
        let saved_path = manager.save_checkpoint(&checkpoint, 42).unwrap();
        assert!(saved_path.exists());

        // Load checkpoint
        let loaded_checkpoint = manager.load_latest_checkpoint().unwrap();
        assert_eq!(loaded_checkpoint.current_iteration, 42);
        assert_eq!(loaded_checkpoint.merit_threshold, 1.5);

        // Cleanup
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_checkpoint_exists() {
        let temp_dir = env::temp_dir().join("test_checkpoints_3");
        let config = CheckpointConfig {
            checkpoint_dir: temp_dir.clone(),
            checkpoint_name: "test".to_string(),
            save_frequency: 5,
            keep_all: false,
            auto_resume: true,
        };

        let manager = CheckpointManager::new(config).unwrap();
        assert!(!manager.checkpoint_exists());

        // Create a dummy checkpoint
        let checkpoint = OQNLPCheckpoint {
            params: OQNLPParams::default(),
            current_iteration: 0,
            merit_threshold: f64::INFINITY,
            solution_set: None,
            reference_set: vec![],
            unchanged_cycles: 0,
            elapsed_time: 0.0,
            distance_filter_solutions: vec![],
            current_seed: 0,
            timestamp: "2025-01-01T00:00:00Z".to_string(),
        };

        manager.save_checkpoint(&checkpoint, 0).unwrap();
        assert!(manager.checkpoint_exists());

        // Cleanup
        let _ = fs::remove_dir_all(temp_dir);
    }
}
