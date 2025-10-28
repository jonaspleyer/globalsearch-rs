//! Stage 2 observer state tracking
//!
//! Tracks metrics during Stage 2 (iterative improvement phase).
//! Stage 2 performs iterative refinement of the solution set through
//! merit filtering, distance-based selection, and local optimization.
//!
//! ## Stage 2 Overview
//!
//! Stage 2 iteratively improves the solution set by:
//!
//! 1. **Merit Filtering**: Select best solutions based on quality and diversity
//! 2. **Trial Point Generation**: Create new candidates from current solutions
//! 3. **Local Optimization**: Apply local solvers to improve candidate solutions
//! 4. **Solution Set Update**: Replace inferior solutions with improvements
//! 5. **Convergence Check**: Monitor for termination conditions
//!
//! ## Tracked Metrics
//!
//! - **Solution Set**: Size and quality of the maintained solution collection
//! - **Iterations**: Progress through the iterative improvement process
//! - **Local Solver Calls**: Frequency and success of local optimization
//! - **Function Evaluations**: Computational cost of Stage 2
//! - **Convergence Metrics**: Threshold values and stagnation detection
//! - **Timing**: Duration of Stage 2 execution (when enabled)

use std::time::Instant;

/// State tracker for Stage 2 of the algorithm
///
/// Tracks comprehensive metrics during the iterative refinement phase that
/// improves the solution set through merit filtering and local optimization.
/// This phase focuses on intensifying search around high-quality regions.
///
/// # Key Metrics
///
/// - **Best Objective**: Best objective function value found across all solutions
/// - **Solution Set Size**: Number of solutions maintained in the working set
/// - **Current Iteration**: Progress through the iterative improvement process
/// - **Threshold Value**: Merit filter threshold for solution acceptance
/// - **Local Solver Calls**: Total calls made to local optimization algorithms
/// - **Improved Calls**: Number of local solver calls that improved solutions
/// - **Function Evaluations**: Total objective function calls in Stage 2
/// - **Unchanged Cycles**: Number of iterations without solution improvement
/// - **Timing**: Total duration of Stage 2 execution (when enabled)
///
/// # Interpretation
///
/// - **Best Objective**: Overall quality of solutions found
/// - **Solution Set Size**: Diversity and coverage maintained
/// - **Local Solver Calls**: Intensity of local search effort
/// - **Unchanged Cycles**: Convergence indicator (increasing suggests stagnation)
/// - **Threshold Value**: Adaptivity of merit filtering (lower = more selective)
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
    ///
    /// Initializes all metrics to default values:
    /// - Best objective: NaN (no valid solutions yet)
    /// - Solution set size: 0
    /// - Current iteration: 0
    /// - Threshold value: +∞ (accept all initially)
    /// - Local solver calls: 0
    /// - Improved calls: 0
    /// - Function evaluations: 0
    /// - Unchanged cycles: 0
    /// - Timing: None
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
    ///
    /// Returns the best (lowest) objective function value found across all
    /// solutions in the current solution set. This represents the highest
    /// quality solution discovered during Stage 2.
    ///
    /// # Returns
    ///
    /// - `f64`: Best objective value in the current solution set
    /// - `NaN`: If no valid solutions exist in the solution set
    ///
    /// # Interpretation
    ///
    /// - **Decreasing values**: Algorithm is finding better solutions (good)
    /// - **Stable values**: Algorithm has converged or is exploring
    /// - **NaN**: Solution set is empty or uninitialized
    pub fn best_objective(&self) -> f64 {
        self.best_objective
    }

    /// Get solution set size
    ///
    /// Returns the current number of solutions maintained in the working solution set.
    /// The solution set maintains a diverse collection of high-quality solutions
    /// that balance quality and coverage of the search space.
    ///
    /// # Interpretation
    ///
    /// - **Stable values**: Algorithm maintaining target solution set size
    /// - **Increasing values**: Solution set growing (may indicate exploration)
    /// - **Decreasing values**: Solutions being filtered out (may indicate intensification)
    /// - **Zero**: Solution set is empty (algorithm may have issues)
    pub fn solution_set_size(&self) -> usize {
        self.solution_set_size
    }

    /// Get current iteration
    ///
    /// Returns the current iteration number in Stage 2. Each iteration represents
    /// a complete cycle of selection, generation, evaluation, and filtering.
    ///
    /// # Interpretation
    ///
    /// - **Increasing values**: Algorithm progressing through Stage 2
    /// - **Higher values**: More computational effort invested
    /// - **Zero**: Stage 2 hasn't started or is initializing
    ///
    /// # Relationship to Termination
    ///
    /// The algorithm terminates when either:
    /// - Maximum iterations reached
    /// - Convergence criteria met (unchanged cycles exceed limit)
    /// - Target objective achieved
    pub fn current_iteration(&self) -> usize {
        self.current_iteration
    }

    /// Get threshold value
    ///
    /// Returns the current merit filter threshold value. Solutions must have
    /// an objective value better than this threshold to be accepted into the
    /// solution set during filtering operations.
    ///
    /// # Interpretation
    ///
    /// - **Lower values**: More selective filtering (higher quality requirement)
    /// - **Higher values**: Less selective filtering (accepts more solutions)
    /// - **∞ (infinity)**: Accept all solutions (initial state)
    /// - **Decreasing over time**: Algorithm becoming more selective as it improves
    ///
    /// # Merit Filtering
    ///
    /// The threshold controls the trade-off between solution quality and diversity.
    /// Lower thresholds maintain higher quality solutions but may reduce diversity.
    pub fn threshold_value(&self) -> f64 {
        self.threshold_value
    }

    /// Get number of local solver calls
    ///
    /// Returns the total number of times local optimization algorithms have been
    /// invoked during Stage 2. Each call attempts to improve a candidate solution
    /// through gradient-based or derivative-free local search.
    ///
    /// # Interpretation
    ///
    /// - **Higher values**: More intensive local search effort
    /// - **Increasing over time**: Algorithm actively applying local optimization
    /// - **Computational cost**: Local solver calls are typically expensive
    ///
    /// # Relationship to Improvements
    ///
    /// Compare with `improved_local_calls()` to assess local solver effectiveness.
    /// A high ratio of improved to total calls indicates efficient local search.
    pub fn local_solver_calls(&self) -> usize {
        self.local_solver_calls
    }

    /// Get number of local solver calls that improved the solution set
    ///
    /// Returns the number of local solver calls that successfully improved the
    /// solution set by finding better solutions. This measures the effectiveness
    /// of local optimization in finding improvements.
    ///
    /// # Interpretation
    ///
    /// - **Higher values**: Local solvers frequently finding improvements
    /// - **Ratio to total calls**: Efficiency of local search (improved/total)
    /// - **Increasing over time**: Local solvers still effective
    /// - **Stable/low values**: Local solvers not finding significant improvements
    ///
    /// # Success Metrics
    ///
    /// - **High ratio (>50%)**: Local solvers very effective
    /// - **Moderate ratio (20-50%)**: Local solvers moderately effective
    /// - **Low ratio (<20%)**: Local solvers rarely improving (may indicate convergence)
    pub fn improved_local_calls(&self) -> usize {
        self.improved_local_calls
    }

    /// Get total function evaluations
    ///
    /// Returns the cumulative count of objective function evaluations performed
    /// during Stage 2. This includes evaluations for trial points generated
    /// during each iteration and function evaluations performed by local solvers.
    ///
    /// # Interpretation
    ///
    /// - **Higher values**: More thorough exploration and local optimization
    /// - **Increasing over time**: Algorithm actively evaluating candidates
    /// - **Computational cost**: Primary measure of Stage 2 resource usage
    ///
    /// # Components
    ///
    /// Function evaluations include:
    /// - Trial point evaluations during each iteration
    /// - Local solver function evaluations (gradient computations, line searches, etc.)
    pub fn function_evaluations(&self) -> usize {
        self.function_evaluations
    }

    /// Get unchanged cycles count
    ///
    /// Returns the number of consecutive iterations where the solution set
    /// has not improved. This is a key convergence indicator used to detect
    /// when the algorithm should terminate due to stagnation.
    pub fn unchanged_cycles(&self) -> usize {
        self.unchanged_cycles
    }

    /// Get total time spent in Stage 2 (seconds)
    ///
    /// Returns the time elapsed since Stage 2 began. If Stage 2 is still running,
    /// returns the current elapsed time. If Stage 2 has completed, returns the
    /// total time spent in Stage 2.
    ///
    /// # Returns
    ///
    /// - `Some(f64)`: Elapsed time in seconds
    /// - `None`: If Stage 2 timing was not started
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
mod tests_observers_stage2 {
    use super::*;

    #[test]
    /// Test Stage2State creation with default values
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
    /// Test basic Stage2State updates and best objective minimization
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
    /// Test timing functionality with start/end
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
    /// Test local solver call tracking with improved/not improved calls
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

    #[test]
    /// Test best objective updates with edge cases (NaN, negative values, equal values)
    fn test_stage2_best_objective_edge_cases() {
        let mut state = Stage2State::new();

        // Test with NaN initial value
        assert!(state.best_objective().is_nan());

        // First valid objective should be accepted
        state.set_best_objective(10.0);
        assert_eq!(state.best_objective(), 10.0);

        // Better objective should update
        state.set_best_objective(5.0);
        assert_eq!(state.best_objective(), 5.0);

        // Worse objective should not update
        state.set_best_objective(7.0);
        assert_eq!(state.best_objective(), 5.0);

        // Equal objective should not update
        state.set_best_objective(5.0);
        assert_eq!(state.best_objective(), 5.0);

        // Negative objectives should work
        state.set_best_objective(-2.0);
        assert_eq!(state.best_objective(), -2.0);

        // Positive worse objective should not update
        state.set_best_objective(1.0);
        assert_eq!(state.best_objective(), -2.0);

        // Very small improvements
        state.set_best_objective(-2.000001);
        assert_eq!(state.best_objective(), -2.000001);
    }

    #[test]
    /// Test iteration progression and non-sequential updates
    fn test_stage2_iteration_progression() {
        let mut state = Stage2State::new();

        assert_eq!(state.current_iteration(), 0);

        for i in 1..=10 {
            state.set_iteration(i);
            assert_eq!(state.current_iteration(), i);
        }

        // Test non-sequential updates
        state.set_iteration(50);
        assert_eq!(state.current_iteration(), 50);

        state.set_iteration(25);
        assert_eq!(state.current_iteration(), 25);
    }

    #[test]
    /// Test solution set size updates with various values
    fn test_stage2_solution_set_size_variations() {
        let mut state = Stage2State::new();

        assert_eq!(state.solution_set_size(), 0);

        // Test various sizes
        let sizes = vec![1, 5, 10, 0, 100, 50];
        for &size in &sizes {
            state.set_solution_set_size(size);
            assert_eq!(state.solution_set_size(), size);
        }
    }

    #[test]
    /// Test threshold value updates including infinity
    fn test_stage2_threshold_value_updates() {
        let mut state = Stage2State::new();

        // Initial should be infinity
        assert_eq!(state.threshold_value(), f64::INFINITY);

        // Test various threshold values
        let thresholds = vec![10.0, 5.5, 1.0, 0.1, 0.0, -1.0];
        for &threshold in &thresholds {
            state.set_threshold_value(threshold);
            assert_eq!(state.threshold_value(), threshold);
        }

        // Test infinity again
        state.set_threshold_value(f64::INFINITY);
        assert_eq!(state.threshold_value(), f64::INFINITY);
    }

    #[test]
    /// Test local solver call patterns with different improvement sequences
    fn test_stage2_local_solver_call_patterns() {
        let mut state = Stage2State::new();

        // Test alternating pattern
        let pattern = vec![true, false, true, true, false, false, true];
        let mut expected_improved = 0;

        for &improved in &pattern {
            state.add_local_solver_call(improved);
            if improved {
                expected_improved += 1;
            }
        }

        assert_eq!(state.local_solver_calls(), pattern.len());
        assert_eq!(state.improved_local_calls(), expected_improved);

        // Test all false
        let mut state2 = Stage2State::new();
        for _ in 0..5 {
            state2.add_local_solver_call(false);
        }
        assert_eq!(state2.local_solver_calls(), 5);
        assert_eq!(state2.improved_local_calls(), 0);

        // Test all true
        let mut state3 = Stage2State::new();
        for _ in 0..3 {
            state3.add_local_solver_call(true);
        }
        assert_eq!(state3.local_solver_calls(), 3);
        assert_eq!(state3.improved_local_calls(), 3);
    }

    #[test]
    /// Test function evaluations accumulation with various increments
    fn test_stage2_function_evaluations_accumulation() {
        let mut state = Stage2State::new();

        assert_eq!(state.function_evaluations(), 0);

        // Test incremental additions
        state.add_function_evaluations(100);
        assert_eq!(state.function_evaluations(), 100);

        state.add_function_evaluations(250);
        assert_eq!(state.function_evaluations(), 350);

        state.add_function_evaluations(0);
        assert_eq!(state.function_evaluations(), 350);

        // Large numbers
        state.add_function_evaluations(10000);
        assert_eq!(state.function_evaluations(), 10350);
    }

    #[test]
    /// Test unchanged cycles tracking for convergence detection
    fn test_stage2_unchanged_cycles_tracking() {
        let mut state = Stage2State::new();

        assert_eq!(state.unchanged_cycles(), 0);

        // Test increasing cycles
        for i in 1..=5 {
            state.set_unchanged_cycles(i);
            assert_eq!(state.unchanged_cycles(), i);
        }

        // Test reset to zero
        state.set_unchanged_cycles(0);
        assert_eq!(state.unchanged_cycles(), 0);

        // Test large numbers
        state.set_unchanged_cycles(1000);
        assert_eq!(state.unchanged_cycles(), 1000);
    }

    #[test]
    /// Test timing edge cases (no start, ongoing timing, end timing)
    fn test_stage2_timing_edge_cases() {
        let mut state = Stage2State::new();

        // No timing started
        assert!(state.total_time().is_none());

        // End without start
        state.end();
        assert!(state.total_time().is_none());

        // Start and check ongoing time
        state.start();
        let time1 = state.total_time().unwrap();
        assert!(time1 >= 0.0);

        std::thread::sleep(std::time::Duration::from_millis(5));

        let time2 = state.total_time().unwrap();
        assert!(time2 > time1);

        // End and check final time
        state.end();
        let final_time = state.total_time().unwrap();
        assert!(final_time >= time2);
    }

    #[test]
    /// Test that clone preserves all state correctly
    fn test_stage2_clone_behavior() {
        let mut state = Stage2State::new();
        state.set_iteration(10);
        state.set_best_objective(3.5);
        state.set_solution_set_size(8);
        state.set_threshold_value(2.0);
        state.add_local_solver_call(true);
        state.add_local_solver_call(false);
        state.add_function_evaluations(500);
        state.set_unchanged_cycles(2);

        let cloned = state.clone();

        assert_eq!(cloned.current_iteration(), 10);
        assert_eq!(cloned.best_objective(), 3.5);
        assert_eq!(cloned.solution_set_size(), 8);
        assert_eq!(cloned.threshold_value(), 2.0);
        assert_eq!(cloned.local_solver_calls(), 2);
        assert_eq!(cloned.improved_local_calls(), 1);
        assert_eq!(cloned.function_evaluations(), 500);
        assert_eq!(cloned.unchanged_cycles(), 2);
    }

    #[test]
    /// Test Default implementation creates same state as new()
    fn test_stage2_default_implementation() {
        let state = Stage2State::default();
        assert!(state.best_objective().is_nan());
        assert_eq!(state.solution_set_size(), 0);
        assert_eq!(state.current_iteration(), 0);
        assert_eq!(state.threshold_value(), f64::INFINITY);
        assert_eq!(state.local_solver_calls(), 0);
        assert_eq!(state.improved_local_calls(), 0);
        assert_eq!(state.function_evaluations(), 0);
        assert_eq!(state.unchanged_cycles(), 0);
    }

    #[test]
    /// Test convergence scenario with improving then stagnant iterations
    fn test_stage2_convergence_scenario() {
        let mut state = Stage2State::new();

        // Simulate a convergence scenario
        state.set_best_objective(10.0);
        state.set_solution_set_size(5);
        state.set_threshold_value(8.0);

        // Early iterations with improvements
        for iter in 1..=5 {
            state.set_iteration(iter);
            state.add_local_solver_call(true);
            state.add_function_evaluations(20);
        }

        assert_eq!(state.current_iteration(), 5);
        assert_eq!(state.local_solver_calls(), 5);
        assert_eq!(state.improved_local_calls(), 5);
        assert_eq!(state.function_evaluations(), 100);

        // Later iterations with no improvements (convergence)
        for iter in 6..=10 {
            state.set_iteration(iter);
            state.add_local_solver_call(false);
            state.add_function_evaluations(15);
            state.set_unchanged_cycles(iter - 5); // Increasing unchanged cycles
        }

        assert_eq!(state.current_iteration(), 10);
        assert_eq!(state.local_solver_calls(), 10);
        assert_eq!(state.improved_local_calls(), 5); // Still 5 improved
        assert_eq!(state.function_evaluations(), 100 + 5 * 15); // 175
        assert_eq!(state.unchanged_cycles(), 5);
    }
}
