//! Stage 2 observer state tracking
//!
//! Tracks metrics during Stage 2 (iterative improvement phase)

use std::time::Instant;

/// State tracker for Stage 2 of the algorithm
#[derive(Debug, Clone)]
pub struct Stage2State {
    /// Best objective function value found so far
    best_objective: f64,

    /// Current number of solutions in solution set
    solution_set_size: usize,

    /// Current iteration number
    current_iteration: usize,

    /// Current threshold value for merit filter
    threshold_value: f64,

    /// Number of local solver calls made
    local_solver_calls: usize,

    /// Number of local solver calls that improved the solution set
    improved_local_calls: usize,

    /// Total number of function evaluations in this stage (includes trial points and local solver evaluations)
    function_evaluations: usize,

    /// Number of unchanged cycles
    unchanged_cycles: usize,

    /// Total time spent in Stage 2
    total_time: Option<f64>,

    /// Start time for Stage 2
    stage_start: Option<Instant>,
}

impl Stage2State {
    /// Create a new Stage 2 state tracker
    pub fn new() -> Self {
        Self {
            best_objective: f64::NAN,
            solution_set_size: 0,
            current_iteration: 0,
            threshold_value: f64::INFINITY,
            local_solver_calls: 0,
            improved_local_calls: 0,
            function_evaluations: 0,
            unchanged_cycles: 0,
            total_time: None,
            stage_start: None,
        }
    }

    /// Start Stage 2 timing
    pub fn start(&mut self) {
        self.stage_start = Some(Instant::now());
    }

    /// End Stage 2 and calculate total time
    pub fn end(&mut self) {
        if let Some(start) = self.stage_start {
            self.total_time = Some(start.elapsed().as_secs_f64());
        }
    }

    /// Update current iteration
    pub fn set_iteration(&mut self, iteration: usize) {
        self.current_iteration = iteration;
    }

    /// Update best objective value
    pub fn set_best_objective(&mut self, objective: f64) {
        if self.best_objective.is_nan() || objective < self.best_objective {
            self.best_objective = objective;
        }
    }

    /// Update solution set size
    pub fn set_solution_set_size(&mut self, size: usize) {
        self.solution_set_size = size;
    }

    /// Update threshold value
    pub fn set_threshold_value(&mut self, threshold: f64) {
        self.threshold_value = threshold;
    }

    /// Increment local solver call counter
    pub fn add_local_solver_call(&mut self, improved: bool) {
        self.local_solver_calls += 1;
        if improved {
            self.improved_local_calls += 1;
        }
    }

    /// Add function evaluations (from trial points or local solvers)
    pub fn add_function_evaluations(&mut self, count: usize) {
        self.function_evaluations += count;
    }

    /// Set unchanged cycles count
    pub fn set_unchanged_cycles(&mut self, count: usize) {
        self.unchanged_cycles = count;
    }

    /// Get best objective value
    pub fn best_objective(&self) -> f64 {
        self.best_objective
    }

    /// Get solution set size
    pub fn solution_set_size(&self) -> usize {
        self.solution_set_size
    }

    /// Get current iteration
    pub fn current_iteration(&self) -> usize {
        self.current_iteration
    }

    /// Get threshold value
    pub fn threshold_value(&self) -> f64 {
        self.threshold_value
    }

    /// Get number of local solver calls
    pub fn local_solver_calls(&self) -> usize {
        self.local_solver_calls
    }

    /// Get number of local solver calls that improved the solution set
    pub fn improved_local_calls(&self) -> usize {
        self.improved_local_calls
    }

    /// Get total function evaluations
    pub fn function_evaluations(&self) -> usize {
        self.function_evaluations
    }

    /// Get unchanged cycles count
    pub fn unchanged_cycles(&self) -> usize {
        self.unchanged_cycles
    }

    /// Get total time spent in Stage 2 (seconds)
    pub fn total_time(&self) -> Option<f64> {
        if let Some(start) = self.stage_start {
            Some(start.elapsed().as_secs_f64())
        } else {
            self.total_time
        }
    }
}

impl Default for Stage2State {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage2_state_creation() {
        let state = Stage2State::new();
        assert!(state.best_objective().is_nan());
        assert_eq!(state.solution_set_size(), 0);
        assert_eq!(state.current_iteration(), 0);
        assert!(state.best_objective().is_nan());
        assert_eq!(state.local_solver_calls(), 0);
        assert_eq!(state.improved_local_calls(), 0);
        assert_eq!(state.function_evaluations(), 0);
        assert_eq!(state.unchanged_cycles(), 0);
    }

    #[test]
    fn test_stage2_state_updates() {
        let mut state = Stage2State::new();

        state.set_iteration(5);
        assert_eq!(state.current_iteration(), 5);

        state.set_best_objective(1.5);
        assert_eq!(state.best_objective(), 1.5);

        // Best objective should only update if lower
        state.set_best_objective(2.0);
        assert_eq!(state.best_objective(), 1.5);

        state.set_solution_set_size(3);
        assert_eq!(state.solution_set_size(), 3);

        state.set_threshold_value(2.5);
        assert_eq!(state.threshold_value(), 2.5);

        state.add_local_solver_call(true);
        state.add_local_solver_call(false);
        assert_eq!(state.local_solver_calls(), 2);
        assert_eq!(state.improved_local_calls(), 1);

        state.add_function_evaluations(10);
        state.add_function_evaluations(50);
        assert_eq!(state.function_evaluations(), 60);

        state.set_unchanged_cycles(3);
        assert_eq!(state.unchanged_cycles(), 3);
    }

    #[test]
    fn test_stage2_history_tracking() {
        // History tracking removed — this test no longer applies
    }

    #[test]
    fn test_stage2_timing() {
        let mut state = Stage2State::new();

        state.start();
        std::thread::sleep(std::time::Duration::from_millis(10));
        state.end();

        let total_time = state.total_time();
        assert!(total_time.is_some());
        assert!(total_time.unwrap() > 0.0);
    }

    #[test]
    fn test_stage2_improved_local_calls() {
        let mut state = Stage2State::new();

        // No calls yet
        assert_eq!(state.improved_local_calls(), 0);

        // All improved
        state.add_local_solver_call(true);
        state.add_local_solver_call(true);
        assert_eq!(state.local_solver_calls(), 2);
        assert_eq!(state.improved_local_calls(), 2);

        // Some did not improve
        state.add_local_solver_call(false);
        state.add_local_solver_call(false);
        assert_eq!(state.local_solver_calls(), 4);
        assert_eq!(state.improved_local_calls(), 2);
    }
}
