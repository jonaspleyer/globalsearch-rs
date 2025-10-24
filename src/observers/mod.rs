//! # Observers Module
//!
//! This module provides observer functionality to monitor and track the optimization
//! algorithm state during execution. Observers can track various metrics such as:
//!
//! - Reference set size and best objective values in Stage 1
//! - Solution set size and iterations in Stage 2  
//! - Number of function evaluations
//! - Threshold values and local solver calls
//!
//! ## Example Usage
//!
//! ```rust
//! use globalsearch::observers::{Observer, ObserverMode};
//! use globalsearch::oqnlp::OQNLP;
//! use globalsearch::types::OQNLPParams;
//! # use globalsearch::problem::Problem;
//! # use globalsearch::types::EvaluationError;
//! # use ndarray::{Array1, Array2, array};
//! #
//! # #[derive(Clone)]
//! # struct TestProblem;
//! # impl Problem for TestProblem {
//! #     fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
//! #         Ok(x[0].powi(2) + x[1].powi(2))
//! #     }
//! #     fn variable_bounds(&self) -> Array2<f64> {
//! #         array![[-5.0, 5.0], [-5.0, 5.0]]
//! #     }
//! # }
//!
//! let problem = TestProblem;
//! let params = OQNLPParams::default();
//!
//! // Create an observer with tracking for both stages
//! let observer = Observer::new()
//!     .with_stage1_tracking()
//!     .with_stage2_tracking()
//!     .with_timing();
//!
//! let mut optimizer = OQNLP::new(problem, params)
//!     .unwrap()
//!     .add_observer(observer);
//!
//! let solutions = optimizer.run();
//!
//! // After optimization, access observer metrics
//! // You can also use the observer's callback functionality to log metrics during optimization
//! if let Some(observer) = optimizer.observer() {
//!        if let Some(stage1) = observer.stage1_final() {
//!            println!("\nStage 1 (Scatter Search):");
//!            println!("  Reference set size: {}", stage1.reference_set_size());
//!            println!("  Best objective: {:.8}", stage1.best_objective());
//!            println!("  Function evaluations: {}", stage1.function_evaluations());
//!            println!("  Trial points generated: {}", stage1.trial_points_generated());
//!            if let Some(time) = stage1.total_time() {
//!                println!("  Total time: {:.3}s", time);
//!            }
//!        }
//!
//!        if let Some(stage2) = observer.stage2() {
//!            println!("\nStage 2 (Local Refinement):");
//!            println!("  Iterations completed: {}", stage2.current_iteration() + 1);
//!            println!("  Best objective: {:.8}", stage2.best_objective());
//!            println!("  Solutions found: {}", stage2.solution_set_size());
//!            println!(
//!                "  Local solver calls: {} (improved: {})",
//!                stage2.local_solver_calls(),
//!                stage2.improved_local_calls()
//!            );
//!            println!("  Function evaluations: {}", stage2.function_evaluations());
//!            println!("  Unchanged cycles: {}", stage2.unchanged_cycles());
//!            if let Some(time) = stage2.total_time() {
//!                println!("  Total time: {:.3}s", time);
//!          }
//!        }
//!    }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::time::Instant;

mod stage1;
mod stage2;

pub use stage1::Stage1State;
pub use stage2::Stage2State;

/// Observer mode determines which stages to track
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObserverMode {
    /// Only track Stage 1 (reference set construction)
    Stage1Only,
    /// Only track Stage 2 (iterative improvement)
    Stage2Only,
    /// Track both stages
    Both,
}

/// Callback function type for observer updates
///
/// The callback receives a reference to the Observer, allowing access to
/// all tracked metrics during optimization.
pub type ObserverCallback = Box<dyn Fn(&Observer) + Send + Sync>;

/// Main observer struct that tracks algorithm state
///
/// The observer can be configured to track different metrics during
/// Stage 1 (reference set construction) and Stage 2 (iterative improvement).
pub struct Observer {
    /// Observer mode determines which stages to track
    mode: ObserverMode,

    /// Stage 1 tracking state
    stage1: Option<Stage1State>,

    /// Stage 2 tracking state
    stage2: Option<Stage2State>,

    /// Whether to track timing information
    track_timing: bool,

    /// Start time for the optimization
    start_time: Option<Instant>,

    /// Optional callback function to be called during optimization
    callback: Option<ObserverCallback>,

    /// Frequency of callback invocation (every N iterations in Stage 2)
    callback_frequency: usize,

    /// Flag to track if Stage 1 has completed (to avoid repeated logging)
    stage1_completed: bool,

    /// Flag to track if Stage 2 has started (to avoid premature logging)
    stage2_started: bool,
}

// Manual Debug implementation since ObserverCallback doesn't implement Debug
impl std::fmt::Debug for Observer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Observer")
            .field("mode", &self.mode)
            .field("stage1", &self.stage1)
            .field("stage2", &self.stage2)
            .field("track_timing", &self.track_timing)
            .field("start_time", &self.start_time)
            .field("callback", &self.callback.as_ref().map(|_| "Some(...)"))
            .field("callback_frequency", &self.callback_frequency)
            .field("stage1_completed", &self.stage1_completed)
            .field("stage2_started", &self.stage2_started)
            .finish()
    }
}

// Manual Clone implementation since ObserverCallback cannot be cloned
impl Clone for Observer {
    fn clone(&self) -> Self {
        Self {
            mode: self.mode,
            stage1: self.stage1.clone(),
            stage2: self.stage2.clone(),
            track_timing: self.track_timing,
            start_time: self.start_time,
            callback: None, // Callbacks cannot be cloned
            callback_frequency: self.callback_frequency,
            stage1_completed: self.stage1_completed,
            stage2_started: self.stage2_started,
        }
    }
}

impl Observer {
    /// Create a new observer with no tracking enabled
    pub fn new() -> Self {
        Self {
            mode: ObserverMode::Both,
            stage1: None,
            stage2: None,
            track_timing: false,
            start_time: None,
            callback: None,
            callback_frequency: 1,
            stage1_completed: false,
            stage2_started: false,
        }
    }

    /// Enable Stage 1 tracking
    pub fn with_stage1_tracking(mut self) -> Self {
        self.stage1 = Some(Stage1State::new());
        self
    }

    /// Enable Stage 2 tracking
    pub fn with_stage2_tracking(mut self) -> Self {
        self.stage2 = Some(Stage2State::new());
        self
    }

    /// Enable timing tracking for stages
    pub fn with_timing(mut self) -> Self {
        self.track_timing = true;
        self
    }

    /// Set observer mode
    pub fn with_mode(mut self, mode: ObserverMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set a callback function to be called during optimization
    ///
    /// The callback receives a reference to the Observer, allowing you to
    /// access all tracked metrics in real-time during optimization.
    ///
    /// # Arguments
    ///
    /// * `callback` - Function to call during optimization
    ///
    /// # Example
    ///
    /// ```rust
    /// use globalsearch::observers::Observer;
    ///
    /// let observer = Observer::new()
    ///     .with_stage2_tracking()
    ///     .with_callback(|obs| {
    ///         if let Some(stage2) = obs.stage2() {
    ///             println!("Iteration {}: Best = {:.6}",
    ///                 stage2.current_iteration(),
    ///                 stage2.best_objective());
    ///         }
    ///     });
    /// ```
    pub fn with_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(&Observer) + Send + Sync + 'static,
    {
        self.callback = Some(Box::new(callback));
        self
    }

    /// Set the frequency for callback invocation
    ///
    /// The callback will be called every N iterations in Stage 2.
    /// Default is 1 (every iteration).
    ///
    /// **Note:** If no callback has been set with `with_callback()`, this method will
    /// automatically use the default callback. This allows you to simply call
    /// `with_callback_frequency(n)` to get default logging at the specified frequency.
    ///
    /// # Arguments
    ///
    /// * `frequency` - Number of iterations between callback calls
    ///
    /// # Example
    ///
    /// ```rust
    /// use globalsearch::observers::Observer;
    ///
    /// // This will automatically use the default callback
    /// let observer = Observer::new()
    ///     .with_stage2_tracking()
    ///     .with_callback_frequency(10); // Logs every 10 iterations with default callback
    /// ```
    pub fn with_callback_frequency(mut self, frequency: usize) -> Self {
        self.callback_frequency = frequency;
        // If no callback has been set, use the default one
        if self.callback.is_none() {
            self = self.with_default_callback();
        }
        self
    }

    /// Use a default console logging callback for Stage 1 and Stage 2
    ///
    /// This is a convenience method that provides sensible default logging
    /// for both stages of the optimization. Stage 1 updates are printed
    /// after scatter search and local optimization. Stage 2 updates are
    /// printed according to the callback frequency.
    ///
    /// # Example
    ///
    /// ```rust
    /// use globalsearch::observers::Observer;
    ///
    /// let observer = Observer::new()
    ///     .with_stage1_tracking()
    ///     .with_stage2_tracking()
    ///     .with_default_callback();
    /// ```
    pub fn with_default_callback(self) -> Self {
        self.with_callback(|obs| {
            // Stage 1 updates
            if let Some(stage1) = obs.stage1() {
                let substage = stage1.current_substage();
                if substage == "scatter_search_running" {
                    eprintln!("[Stage 1] Starting Scatter Search...");
                } else if substage == "initialization_complete" {
                    eprintln!(
                        "[Stage 1] Initialization Complete | Initial Points: {}",
                        stage1.function_evaluations()
                    );
                } else if substage == "diversification_complete" {
                    eprintln!(
                        "[Stage 1] Diversification Complete | Ref. Set Size: {}",
                        stage1.reference_set_size()
                    );
                } else if substage == "intensification_complete" {
                    eprintln!(
                        "[Stage 1] Intensification Complete | Trial Points Generated: {} | Accepted: {}",
                        stage1.trial_points_generated(),
                        stage1.reference_set_size()
                    );
                } else if substage == "scatter_search_complete" {
                    eprintln!(
                        "[Stage 1] Scatter Search Complete | Best: {:.6}",
                        stage1.best_objective()
                    );
                } else if substage == "local_optimization_complete" {
                    eprintln!(
                        "[Stage 1] Local Optimization Complete | Best: {:.6} | Total Fn Evals: {}",
                        stage1.best_objective(),
                        stage1.function_evaluations()
                    );
                }
            }
            // Stage 2 updates (only when started)
            if let Some(stage2) = obs.stage2() {
                if stage2.current_iteration() > 0 {
                    eprintln!(
                        "[Stage 2] Iter {} | Best: {:.6} | Solutions: {} | Threshold: {:.6} | Local Solver Calls: {} | Fn Evals: {}",
                        stage2.current_iteration(),
                        stage2.best_objective(),
                        stage2.solution_set_size(),
                        stage2.threshold_value(),
                        stage2.local_solver_calls(),
                        stage2.function_evaluations()
                    );
                }
            }
        })
    }

    /// Use a default console logging callback for Stage 1 only
    ///
    /// This prints updates during scatter search and local optimization in Stage 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// use globalsearch::observers::Observer;
    ///
    /// let observer = Observer::new()
    ///     .with_stage1_tracking()
    ///     .with_stage1_callback();
    /// ```
    pub fn with_stage1_callback(self) -> Self {
        self.with_callback(|obs| {
            if let Some(stage1) = obs.stage1() {
                let substage = stage1.current_substage();
                if substage == "scatter_search_running" {
                    eprintln!("[Stage 1] Starting Scatter Search...");
                } else if substage == "initialization_complete" {
                    eprintln!(
                        "[Stage 1] Initialization Complete | Initial Points: {}",
                        stage1.function_evaluations()
                    );
                } else if substage == "diversification_complete" {
                    eprintln!(
                        "[Stage 1] Diversification Complete | Ref. Set Size: {}",
                        stage1.reference_set_size()
                    );
                } else if substage == "intensification_complete" {
                    eprintln!(
                        "[Stage 1] Intensification Complete | Trial Points Generated: {} | Accepted: {}",
                        stage1.trial_points_generated(),
                        stage1.reference_set_size()
                    );
                } else if substage == "scatter_search_complete" {
                    eprintln!(
                        "[Stage 1] Scatter Search Complete | Best: {:.6}",
                        stage1.best_objective()
                    );
                } else if substage == "local_optimization_complete" {
                    eprintln!(
                        "[Stage 1] Local Optimization Complete | Best: {:.6} | TotalFnEvals: {}",
                        stage1.best_objective(),
                        stage1.function_evaluations()
                    );
                }
                // Don't print for "stage1_complete" - it's just an internal marker
            }
        })
    }

    /// Use a default console logging callback for Stage 2 only
    ///
    /// This prints iteration progress during Stage 2. Use `with_callback_frequency()`
    /// to control how often updates are printed.
    ///
    /// # Example
    ///
    /// ```rust
    /// use globalsearch::observers::Observer;
    ///
    /// let observer = Observer::new()
    ///     .with_stage2_tracking()
    ///     .with_stage2_callback()
    ///     .with_callback_frequency(10); // Print every 10 iterations
    /// ```
    pub fn with_stage2_callback(self) -> Self {
        self.with_callback(|obs| {
            if let Some(stage2) = obs.stage2() {
                if stage2.current_iteration() > 0 {
                    eprintln!(
                        "[Stage 2] Iter {} | Best: {:.6} | Solutions: {} | Threshold: {:.6} | Local Calls: {} | Fn Evals: {}",
                        stage2.current_iteration(),
                        stage2.best_objective(),
                        stage2.solution_set_size(),
                        stage2.threshold_value(),
                        stage2.local_solver_calls(),
                        stage2.function_evaluations()
                    );
                }
            }
        })
    }

    /// Start timing
    pub(crate) fn start_timer(&mut self) {
        if self.track_timing {
            self.start_time = Some(Instant::now());
        }
    }

    /// Get elapsed time in seconds
    pub fn elapsed_time(&self) -> Option<f64> {
        self.start_time.map(|start| start.elapsed().as_secs_f64())
    }

    /// Check if Stage 1 should be observed
    pub fn should_observe_stage1(&self) -> bool {
        matches!(self.mode, ObserverMode::Stage1Only | ObserverMode::Both) && self.stage1.is_some()
    }

    /// Check if Stage 2 should be observed
    pub fn should_observe_stage2(&self) -> bool {
        matches!(self.mode, ObserverMode::Stage2Only | ObserverMode::Both) && self.stage2.is_some()
    }

    /// Get Stage 1 state reference
    pub fn stage1(&self) -> Option<&Stage1State> {
        // Don't return Stage 1 state after it's completed to prevent repeated logging
        if self.stage1_completed {
            None
        } else {
            self.stage1.as_ref()
        }
    }

    /// Get Stage 1 state reference even after completion (for final statistics)
    pub fn stage1_final(&self) -> Option<&Stage1State> {
        self.stage1.as_ref()
    }

    /// Get mutable Stage 1 state reference
    pub(crate) fn stage1_mut(&mut self) -> Option<&mut Stage1State> {
        self.stage1.as_mut()
    }

    /// Mark Stage 1 as completed (prevents further Stage 1 callback invocations)
    pub(crate) fn mark_stage1_complete(&mut self) {
        self.stage1_completed = true;
    }

    /// Get Stage 2 state reference
    pub fn stage2(&self) -> Option<&Stage2State> {
        // Don't return Stage 2 state until it has started to prevent premature logging
        if self.stage2_started {
            self.stage2.as_ref()
        } else {
            None
        }
    }

    /// Get mutable Stage 2 state reference
    pub(crate) fn stage2_mut(&mut self) -> Option<&mut Stage2State> {
        self.stage2.as_mut()
    }

    /// Mark Stage 2 as started (allows Stage 2 callback invocations)
    pub(crate) fn mark_stage2_started(&mut self) {
        self.stage2_started = true;
    }

    /// Check if timing is enabled
    pub fn is_timing_enabled(&self) -> bool {
        self.track_timing
    }

    /// Invoke the callback if one is set
    ///
    /// This is called internally by the OQNLP algorithm during optimization.
    pub(crate) fn invoke_callback(&self) {
        if let Some(ref callback) = self.callback {
            callback(self);
        }
    }

    /// Check if callback should be invoked for the current iteration
    pub(crate) fn should_invoke_callback(&self, iteration: usize) -> bool {
        self.callback.is_some() && (iteration % self.callback_frequency == 0)
    }
}

impl Default for Observer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observer_creation() {
        let observer = Observer::new();
        assert!(!observer.should_observe_stage1());
        assert!(!observer.should_observe_stage2());
        assert!(!observer.is_timing_enabled());
    }

    #[test]
    fn test_observer_with_stage1() {
        let observer = Observer::new().with_stage1_tracking();
        assert!(observer.should_observe_stage1());
        assert!(!observer.should_observe_stage2());
    }

    #[test]
    fn test_observer_with_stage2() {
        let observer = Observer::new().with_stage2_tracking();
        assert!(!observer.should_observe_stage1());
        assert!(observer.should_observe_stage2());
    }

    #[test]
    fn test_observer_with_both_stages() {
        let observer = Observer::new().with_stage1_tracking().with_stage2_tracking();
        assert!(observer.should_observe_stage1());
        assert!(observer.should_observe_stage2());
    }

    #[test]
    fn test_observer_with_timing() {
        let observer = Observer::new().with_timing();
        assert!(observer.is_timing_enabled());
    }

    #[test]
    fn test_observer_modes() {
        let observer = Observer::new()
            .with_mode(ObserverMode::Stage1Only)
            .with_stage1_tracking()
            .with_stage2_tracking();

        assert!(observer.should_observe_stage1());
        assert!(!observer.should_observe_stage2());

        let observer = Observer::new()
            .with_mode(ObserverMode::Stage2Only)
            .with_stage1_tracking()
            .with_stage2_tracking();

        assert!(!observer.should_observe_stage1());
        assert!(observer.should_observe_stage2());

        let observer = Observer::new()
            .with_mode(ObserverMode::Both)
            .with_stage1_tracking()
            .with_stage2_tracking();

        assert!(observer.should_observe_stage1());
        assert!(observer.should_observe_stage2());
    }
}
