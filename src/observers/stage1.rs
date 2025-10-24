//! Stage 1 observer state tracking
//!
//! Tracks metrics during Stage 1 (reference set construction and initial scatter search)

use std::time::Instant;

/// State tracker for Stage 1 of the algorithm
#[derive(Debug, Clone)]
pub struct Stage1State {
    /// Current size of reference set
    reference_set_size: usize,

    /// Best objective function value found so far
    best_objective: f64,

    /// Current stage within Stage 1 (e.g., "initialization", "diversification", "trial_generation")
    current_substage: String,

    /// Time spent in current substage (seconds)
    substage_time: Option<f64>,

    /// Start time of current substage
    substage_start: Option<Instant>,

    /// Total number of objective function evaluations
    function_evaluations: usize,

    /// Number of trial points generated
    trial_points_generated: usize,

    /// Total time spent in Stage 1
    total_time: Option<f64>,

    /// Start time for Stage 1
    stage_start: Option<Instant>,
}

impl Stage1State {
    /// Create a new Stage 1 state tracker
    pub fn new() -> Self {
        Self {
            reference_set_size: 0,
            best_objective: f64::NAN,
            current_substage: "not_started".to_string(),
            substage_time: None,
            substage_start: None,
            function_evaluations: 0,
            trial_points_generated: 0,
            total_time: None,
            stage_start: None,
        }
    }

    /// Start Stage 1 timing
    pub fn start(&mut self) {
        self.stage_start = Some(Instant::now());
    }

    /// End Stage 1 and calculate total time
    pub fn end(&mut self) {
        if let Some(start) = self.stage_start {
            self.total_time = Some(start.elapsed().as_secs_f64());
        }
    }

    /// Update reference set size
    pub fn set_reference_set_size(&mut self, size: usize) {
        self.reference_set_size = size;
    }

    /// Update best objective value
    pub fn set_best_objective(&mut self, objective: f64) {
        if self.best_objective.is_nan() || objective < self.best_objective {
            self.best_objective = objective;
        }
    }

    /// Enter a new substage
    pub fn enter_substage(&mut self, name: &str) {
        // End previous substage if it exists
        if let Some(start) = self.substage_start {
            self.substage_time = Some(start.elapsed().as_secs_f64());
        }

        self.current_substage = name.to_string();
        self.substage_start = Some(Instant::now());
    }

    /// Set the substage time explicitly (for when timing is done externally)
    /// Increment function evaluation counter
    pub fn add_function_evaluations(&mut self, count: usize) {
        self.function_evaluations += count;
    }

    /// Increment trial points generated counter
    pub fn add_trial_points(&mut self, count: usize) {
        self.trial_points_generated += count;
    }

    /// Get reference set size
    pub fn reference_set_size(&self) -> usize {
        self.reference_set_size
    }

    /// Get best objective value
    pub fn best_objective(&self) -> f64 {
        self.best_objective
    }

    /// Get current substage name
    pub fn current_substage(&self) -> &str {
        &self.current_substage
    }

    /// Get total elapsed time since Stage 1 started (seconds)
    /// Returns cumulative time from the beginning of Stage 1
    pub fn total_time(&self) -> Option<f64> {
        if let Some(start) = self.stage_start {
            Some(start.elapsed().as_secs_f64())
        } else {
            self.total_time
        }
    }

    /// Get total number of function evaluations
    pub fn function_evaluations(&self) -> usize {
        self.function_evaluations
    }

    /// Get number of trial points generated
    pub fn trial_points_generated(&self) -> usize {
        self.trial_points_generated
    }
}

impl Default for Stage1State {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage1_state_creation() {
        let state = Stage1State::new();
        assert_eq!(state.reference_set_size(), 0);
        assert!(state.best_objective().is_nan());
        assert_eq!(state.current_substage(), "not_started");
        assert_eq!(state.function_evaluations(), 0);
        assert_eq!(state.trial_points_generated(), 0);
    }

    #[test]
    fn test_stage1_state_updates() {
        let mut state = Stage1State::new();

        state.set_reference_set_size(10);
        assert_eq!(state.reference_set_size(), 10);

        state.set_best_objective(1.5);
        assert_eq!(state.best_objective(), 1.5);

        // Best objective should only update if lower
        state.set_best_objective(2.0);
        assert_eq!(state.best_objective(), 1.5);

        state.add_function_evaluations(5);
        assert_eq!(state.function_evaluations(), 5);

        state.add_trial_points(20);
        assert_eq!(state.trial_points_generated(), 20);
    }

    #[test]
    fn test_stage1_substages() {
        let mut state = Stage1State::new();

        state.start();
        state.enter_substage("initialization");
        assert_eq!(state.current_substage(), "initialization");

        std::thread::sleep(std::time::Duration::from_millis(10));

        state.enter_substage("diversification");
        assert_eq!(state.current_substage(), "diversification");

        // Should have cumulative time since start
        assert!(state.total_time().is_some());
    }

    #[test]
    fn test_stage1_timing() {
        let mut state = Stage1State::new();

        state.start();
        std::thread::sleep(std::time::Duration::from_millis(10));
        state.end();

        let total_time = state.total_time();
        assert!(total_time.is_some());
        assert!(total_time.unwrap() > 0.0);
    }
}
