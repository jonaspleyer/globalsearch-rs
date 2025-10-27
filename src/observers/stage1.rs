//! Stage 1 observer state tracking
//!
//! Tracks metrics during Stage 1 (reference set construction and initial scatter search).
//! Stage 1 consists of several substages that build the initial reference set used
//! in Stage 2 iterative refinement.
//!
//! ## Stage 1 Overview
//!
//! Stage 1 performs scatter search to explore the parameter space and construct
//! a diverse reference set of high-quality solutions. The process includes:
//!
//! 1. **Initialization**: Generate initial points and evaluate them
//! 2. **Diversification**: Build initial reference set through systematic sampling
//! 3. **Intensification**: Generate trial points through combination and improvement
//! 4. **Scatter Search**: Apply scatter search operators to refine the reference set
//! 5. **Local Optimization**: Apply local solvers to improve reference set quality
//!
//! ## Tracked Metrics
//!
//! - **Reference Set**: Size and quality of the solution reference set
//! - **Function Evaluations**: Total objective function calls during Stage 1
//! - **Trial Points**: Number of candidate solutions generated and tested
//! - **Substage Tracking**: Current phase of Stage 1 execution
//! - **Timing**: Duration of Stage 1 and substages (when enabled)
//!
//! ## Usage
//!
//! ```rust
//! use globalsearch::observers::Stage1State;
//!
//! // Access Stage 1 state from an observer
//! if let Some(stage1) = observer.stage1_final() {
//!     println!("Stage 1 Results:");
//!     println!("  Reference set size: {}", stage1.reference_set_size());
//!     println!("  Best objective: {:.6}", stage1.best_objective());
//!     println!("  Function evaluations: {}", stage1.function_evaluations());
//!     println!("  Trial points generated: {}", stage1.trial_points_generated());
//! }
//! ```

use std::time::Instant;

/// State tracker for Stage 1 of the algorithm
///
/// Tracks comprehensive metrics during the scatter search phase that builds
/// the initial reference set. This includes reference set construction,
/// trial point generation, function evaluations, and substage progression.
///
/// # Key Metrics
///
/// - **Reference Set Size**: Number of solutions in the current reference set
/// - **Best Objective**: Best objective function value found so far
/// - **Function Evaluations**: Total number of objective function calls
/// - **Trial Points**: Number of candidate solutions generated during intensification
/// - **Current Substage**: Current phase of Stage 1 execution
/// - **Timing**: Total and substage timing information (when enabled)
///
/// # Substages
///
/// Stage 1 progresses through several substages:
///
/// - `"not_started"`: Stage 1 has not begun
/// - `"scatter_search_running"`: Scatter search is initializing
/// - `"initialization_complete"`: Initial points have been evaluated
/// - `"diversification_complete"`: Initial reference set is built
/// - `"intensification_complete"`: Trial point generation is finished
/// - `"scatter_search_complete"`: Scatter search has converged
/// - `"local_optimization_complete"`: Local optimization of reference set is done
/// - `"stage1_complete"`: Stage 1 has finished (internal marker)
///
/// # Interpretation
///
/// - **Reference Set Size**: Indicates diversity and coverage of the search space
/// - **Function Evaluations**: Measures computational cost of Stage 1
/// - **Trial Points Generated**: Shows exploration intensity during intensification
/// - **Best Objective**: Tracks improvement in solution quality over time
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
    ///
    /// Initializes all metrics to default values:
    /// - Reference set size: 0
    /// - Best objective: NaN (no valid solutions yet)
    /// - Current substage: "not_started"
    /// - Function evaluations: 0
    /// - Trial points: 0
    /// - Timing: None
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
    ///
    /// Updates the current substage name and records timing for the previous substage.
    /// This method is called internally as Stage 1 progresses through its phases.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the new substage being entered
    ///
    /// # Timing Behavior
    ///
    /// If a previous substage was active, its duration is calculated and stored.
    /// The start time for the new substage is recorded.
    pub fn enter_substage(&mut self, name: &str) {
        // End previous substage if it exists
        if let Some(start) = self.substage_start {
            self.substage_time = Some(start.elapsed().as_secs_f64());
        }

        self.current_substage = name.to_string();
        self.substage_start = Some(Instant::now());
    }

    /// Increment function evaluation counter
    ///
    /// Adds to the cumulative count of objective function evaluations performed
    /// during Stage 1. This should be called whenever the objective function
    /// is evaluated, including for initial points, trial points, and local optimization.
    ///
    /// # Arguments
    ///
    /// * `count` - Number of function evaluations to add
    pub fn add_function_evaluations(&mut self, count: usize) {
        self.function_evaluations += count;
    }

    /// Increment trial points generated counter
    ///
    /// Adds to the count of trial points generated during the intensification phase.
    /// Trial points are candidate solutions created by combining reference set members.
    ///
    /// # Arguments
    ///
    /// * `count` - Number of trial points to add
    pub fn add_trial_points(&mut self, count: usize) {
        self.trial_points_generated += count;
    }

    /// Get reference set size
    ///
    /// Returns the current number of solutions in the reference set.
    /// The reference set maintains a diverse collection of high-quality solutions
    /// that guide the intensification phase of scatter search.
    ///
    /// # Interpretation
    ///
    /// - **Increasing values**: Reference set is growing through diversification
    /// - **Stable values**: Reference set has reached target size
    /// - **Higher values**: Better coverage of the search space (generally positive)
    pub fn reference_set_size(&self) -> usize {
        self.reference_set_size
    }

    /// Get best objective value
    ///
    /// Returns the best (lowest) objective function value found so far in Stage 1.
    /// This represents the highest quality solution discovered during scatter search.
    ///
    /// # Returns
    ///
    /// - `f64`: Best objective value found
    /// - `NaN`: If no valid solutions have been evaluated yet
    ///
    /// # Interpretation
    ///
    /// - **Decreasing values**: Algorithm is finding better solutions (good)
    /// - **Stable values**: Algorithm has converged on current region
    /// - **NaN**: Stage 1 hasn't started or no solutions evaluated yet
    pub fn best_objective(&self) -> f64 {
        self.best_objective
    }

    /// Get current substage name
    ///
    /// Returns a string identifier for the current phase of Stage 1 execution.
    /// This helps track progress through the scatter search algorithm.
    ///
    /// # Possible Values
    ///
    /// - `"not_started"`: Stage 1 initialization pending
    /// - `"scatter_search_running"`: Scatter search algorithm starting
    /// - `"initialization_complete"`: Initial point evaluation finished
    /// - `"diversification_complete"`: Initial reference set constructed
    /// - `"intensification_complete"`: Trial point generation finished
    /// - `"scatter_search_complete"`: Scatter search convergence achieved
    /// - `"local_optimization_complete"`: Reference set local optimization done
    /// - `"stage1_complete"`: Stage 1 finished (internal marker)
    ///
    /// # Usage
    ///
    /// ```rust
    /// match stage1.current_substage() {
    ///     "scatter_search_complete" => println!("Scatter search finished"),
    ///     "local_optimization_complete" => println!("Stage 1 complete"),
    ///     _ => println!("Stage 1 in progress..."),
    /// }
    /// ```
    pub fn current_substage(&self) -> &str {
        &self.current_substage
    }

    /// Get total elapsed time since Stage 1 started (seconds)
    ///
    /// Returns the time elapsed since Stage 1 began. If Stage 1 is still running,
    /// returns the current elapsed time. If Stage 1 has completed, returns the
    /// total time spent in Stage 1.
    ///
    /// # Returns
    ///
    /// - `Some(f64)`: Elapsed time in seconds
    /// - `None`: If Stage 1 timing was not started
    pub fn total_time(&self) -> Option<f64> {
        if let Some(start) = self.stage_start {
            Some(start.elapsed().as_secs_f64())
        } else {
            self.total_time
        }
    }

    /// Get total number of function evaluations
    ///
    /// Returns the cumulative count of objective function evaluations performed
    /// during Stage 1. This includes evaluations for initial points, diversification,
    /// intensification trial points, and local optimization.
    ///
    /// # Interpretation
    ///
    /// - **Higher values**: More thorough exploration of the search space
    /// - **Increasing over time**: Algorithm is actively evaluating candidates
    /// - **Computational cost**: Primary measure of Stage 1 resource usage
    ///
    /// # Components
    ///
    /// Function evaluations include:
    /// - Initial point evaluations during diversification
    /// - Trial point evaluations during intensification
    /// - Local solver function evaluations (if applicable)
    pub fn function_evaluations(&self) -> usize {
        self.function_evaluations
    }

    /// Get number of trial points generated
    ///
    /// Returns the total number of trial points generated during the intensification
    /// phase of scatter search. Trial points are candidate solutions created by
    /// combining and perturbing reference set members.
    ///
    /// # Interpretation
    ///
    /// - **Higher values**: More intensive exploration during intensification
    /// - **Increasing during intensification**: Algorithm is generating candidates
    /// - **Stable after intensification**: Phase completed, moving to next stage
    ///
    /// # Relationship to Reference Set
    ///
    /// Only a subset of trial points are accepted into the reference set.
    /// The ratio of accepted to generated trial points indicates intensification effectiveness.
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
