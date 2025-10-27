//! # Observers Module
//!
//! The observers module provides comprehensive monitoring and tracking capabilities
//! for the OQNLP global optimization algorithm. Observers allow you to monitor the
//! algorithm's progress in real-time, collect detailed metrics about each stage of
//! the optimization process, and implement custom logging or visualization.
//!
//! ## Overview
//!
//! The OQNLP algorithm operates in two main stages:
//!
//! 1. **Stage 1 (Scatter Search)**: Explores the parameter space using scatter search
//!    metaheuristics to identify promising regions and build an initial reference set.
//! 2. **Stage 2 (Iterative Refinement)**: Performs local optimization from multiple
//!    starting points, iteratively improving the solution set through merit filtering
//!    and distance-based selection.
//!
//! Observers track key metrics for each stage, providing insights into algorithm
//! behavior, convergence patterns, and computational efficiency.
//!
//! ## Key Features
//!
//! - **Real-time Monitoring**: Track algorithm progress with customizable callbacks
//! - **Detailed Metrics**: Comprehensive statistics for both optimization stages
//! - **Flexible Configuration**: Choose which stages and metrics to monitor
//! - **Performance Tracking**: Monitor function evaluations, timing, and convergence
//! - **Custom Callbacks**: Implement custom logging, visualization, or early stopping
//!
//! ## Architecture
//!
//! The observer system consists of three main components:
//!
//! - [`Observer`]: Main coordinator that manages tracking configuration and callbacks
//! - [`Stage1State`]: Tracks metrics during scatter search and reference set construction
//! - [`Stage2State`]: Tracks metrics during iterative local refinement
//!
//! ## Example Usage
//!
//! ```rust
//! use globalsearch::observers::{Observer, ObserverMode};
//! use globalsearch::oqnlp::OQNLP;
//! use globalsearch::types::OQNLPParams;
//! use globalsearch::problem::Problem;
//! use globalsearch::types::EvaluationError;
//! use ndarray::{Array1, Array2, array};
//!
//! [derive(Clone)]
//! struct TestProblem;
//! impl Problem for TestProblem {
//!    fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
//!        Ok(x[0].powi(2) + x[1].powi(2))
//!    }
//!    fn variable_bounds(&self) -> Array2<f64> {
//!         array![[-5.0, 5.0], [-5.0, 5.0]]
//!     }
//! }
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
//! Ok::<(), Box<dyn::error::Error>>(())
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
/// It supports real-time monitoring through callbacks and provides detailed
/// statistics about algorithm performance and convergence.
///
/// # Configuration Options
///
/// Observers are configured using the builder pattern:
///
/// ```rust
/// use globalsearch::observers::Observer;
///
/// // Basic observer with no tracking
/// let observer = Observer::new();
///
/// // Track both stages with default logging
/// let observer = Observer::new()
///     .with_stage1_tracking()
///     .with_stage2_tracking()
///     .with_default_callback();
///
/// // Custom configuration
/// let observer = Observer::new()
///     .with_stage1_tracking()
///     .with_timing()
///     .with_callback(|obs| {
///         // Custom callback logic
///     });
/// ```
///
/// # Configuration Options
///
/// Observers are configured using the builder pattern:
///
/// ```rust
/// use globalsearch::observers::Observer;
///
/// // Basic observer with no tracking
/// let observer = Observer::new();
///
/// // Track both stages with default logging
/// let observer = Observer::new()
///     .with_stage1_tracking()
///     .with_stage2_tracking()
///     .with_default_callback();
///
/// // Custom configuration
/// let observer = Observer::new()
///     .with_stage1_tracking()
///     .with_timing()
///     .with_callback(|obs| {
///         // Custom callback logic
///     });
/// ```
///
/// # Observer Modes
///
/// The observer can operate in different modes to control which stages are tracked:
///
/// - [`ObserverMode::Both`]: Track both Stage 1 and Stage 2 (default)
/// - [`ObserverMode::Stage1Only`]: Track only Stage 1 scatter search
/// - [`ObserverMode::Stage2Only`]: Track only Stage 2 local refinement
///
/// # Callback System
///
/// Callbacks allow real-time monitoring of the optimization process. They receive
/// a reference to the observer and can access all tracked metrics. Callbacks can be:
///
/// - **Default callbacks**: Pre-built logging functions for common use cases
/// - **Custom callbacks**: User-defined functions for specialized monitoring
/// - **Frequency-controlled**: Callbacks can be invoked every N iterations
///
/// # Timing Information
///
/// When timing is enabled with `with_timing()`, the observer tracks:
///
/// - Total time spent in each stage
/// - Time spent in sub-phases within Stage 1
/// - Cumulative timing information accessible via `stage1_final()` and `stage2()`
///
/// # Accessing Metrics
///
/// Metrics can be accessed in two ways:
///
/// 1. **During optimization**: Via callbacks that receive the observer reference
/// 2. **After optimization**: Via the observer stored in the OQNLP instance
///
/// ```rust
/// // During optimization (in callback)
/// let observer = Observer::new()
///     .with_stage2_tracking()
///     .with_callback(|obs| {
///         if let Some(stage2) = obs.stage2() {
///             println!("Current best: {}", stage2.best_objective());
///         }
///     });
///
/// // After optimization
/// let solutions = optimizer.run();
/// if let Some(observer) = optimizer.observer() {
///     if let Some(stage1) = observer.stage1_final() {
///         println!("Stage 1 completed in {} evaluations",
///             stage1.function_evaluations());
///     }
/// }
/// ```
pub struct Observer {
    /// Observer mode determines which stages to track
    mode: ObserverMode,

    /// Stage 1 tracking state (None if not tracking Stage 1)
    stage1: Option<Stage1State>,

    /// Stage 2 tracking state (None if not tracking Stage 2)
    stage2: Option<Stage2State>,

    /// Whether to track timing information for stages
    track_timing: bool,

    /// Start time for the overall optimization (used for elapsed time calculations)
    start_time: Option<Instant>,

    /// Optional callback function invoked during optimization
    callback: Option<ObserverCallback>,

    /// Frequency of callback invocation (every N iterations in Stage 2)
    callback_frequency: usize,

    /// Flag to track if Stage 1 has completed (prevents repeated logging)
    stage1_completed: bool,

    /// Flag to track if Stage 2 has started (prevents premature logging)
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
    ///
    /// Returns a minimal observer that tracks nothing by default.
    /// Use the builder methods to enable specific tracking features.
    ///
    /// # Example
    ///
    /// ```rust
    /// use globalsearch::observers::Observer;
    ///
    /// let observer = Observer::new();
    /// // No tracking enabled - use builder methods to configure
    /// ```
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
    ///
    /// Enables tracking of scatter search metrics including:
    /// - Reference set size and composition
    /// - Best objective values found
    /// - Function evaluation counts
    /// - Trial point generation statistics
    /// - Sub-stage progression (initialization, diversification, intensification)
    ///
    /// Stage 1 tracking is required for `stage1()` and `stage1_final()` to return data.
    ///
    /// # Example
    ///
    /// ```rust
    /// use globalsearch::observers::Observer;
    ///
    /// let observer = Observer::new()
    ///     .with_stage1_tracking();
    /// ```
    pub fn with_stage1_tracking(mut self) -> Self {
        self.stage1 = Some(Stage1State::new());
        self
    }

    /// Enable Stage 2 tracking
    ///
    /// Enables tracking of iterative refinement metrics including:
    /// - Current iteration number
    /// - Solution set size and composition
    /// - Best objective values
    /// - Local solver call statistics
    /// - Function evaluation counts
    /// - Threshold values and merit filtering
    /// - Convergence metrics (unchanged cycles)
    ///
    /// Stage 2 tracking is required for `stage2()` to return data.
    ///
    /// # Example
    ///
    /// ```rust
    /// use globalsearch::observers::Observer;
    ///
    /// let observer = Observer::new()
    ///     .with_stage2_tracking();
    /// ```
    pub fn with_stage2_tracking(mut self) -> Self {
        self.stage2 = Some(Stage2State::new());
        self
    }

    /// Enable timing tracking for stages
    ///
    /// When enabled, tracks elapsed time for:
    /// - Total Stage 1 duration
    /// - Total Stage 2 duration
    /// - Sub-stage timing within Stage 1
    ///
    /// Timing data is accessible via the `total_time()` methods on
    /// [`Stage1State`] and [`Stage2State`].
    ///
    /// # Performance Impact
    ///
    /// Timing has minimal performance impact but requires system clock access.
    ///
    /// # Example
    ///
    /// ```rust
    /// use globalsearch::observers::Observer;
    ///
    /// let observer = Observer::new()
    ///     .with_stage1_tracking()
    ///     .with_stage2_tracking()
    ///     .with_timing();
    ///
    /// // Later, access timing data
    /// if let Some(stage1) = observer.stage1_final() {
    ///     if let Some(time) = stage1.total_time() {
    ///         println!("Stage 1 took {:.3} seconds", time);
    ///     }
    /// }
    /// ```
    pub fn with_timing(mut self) -> Self {
        self.track_timing = true;
        self
    }

    /// Set observer mode
    ///
    /// Controls which stages of the optimization algorithm are monitored.
    /// This allows fine-grained control over tracking scope and performance.
    ///
    /// # Arguments
    ///
    /// * `mode` - The observer mode determining which stages to track
    ///
    /// # Performance Considerations
    ///
    /// Using [`ObserverMode::Stage1Only`] or [`ObserverMode::Stage2Only`] can
    /// reduce memory usage and callback overhead when only specific stage
    /// information is needed.
    ///
    /// # Example
    ///
    /// ```rust
    /// use globalsearch::observers::{Observer, ObserverMode};
    ///
    /// // Track only Stage 2 for performance monitoring
    /// let observer = Observer::new()
    ///     .with_mode(ObserverMode::Stage2Only)
    ///     .with_stage2_tracking()
    ///     .with_default_callback();
    ///
    /// // Track both stages (default behavior)
    /// let observer = Observer::new()
    ///     .with_mode(ObserverMode::Both)
    ///     .with_stage1_tracking()
    ///     .with_stage2_tracking();
    /// ```
    pub fn with_mode(mut self, mode: ObserverMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set a callback function to be called during optimization
    ///
    /// The callback receives a reference to the Observer, allowing access to
    /// all tracked metrics in real-time during optimization. Callbacks are invoked
    /// at key points during the algorithm execution.
    ///
    /// # Callback Timing
    ///
    /// - **Stage 1**: Called after major substages (initialization, diversification,
    ///   intensification, scatter search completion, local optimization completion)
    /// - **Stage 2**: Called according to the callback frequency (default: every iteration)
    ///
    /// # Arguments
    ///
    /// * `callback` - Function to call during optimization
    ///
    /// # Thread Safety
    ///
    /// Callbacks must be thread-safe (`Send + Sync`) as they may be called from
    /// parallel execution contexts.
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
    ///
    /// # Advanced Example
    ///
    /// ```rust
    /// use globalsearch::observers::Observer;
    ///
    /// let observer = Observer::new()
    ///     .with_stage1_tracking()
    ///     .with_stage2_tracking()
    ///     .with_callback(|obs| {
    ///         // Log Stage 1 progress
    ///         if let Some(stage1) = obs.stage1() {
    ///             println!("Stage 1: {} evaluations, best = {:.6}",
    ///                 stage1.function_evaluations(),
    ///                 stage1.best_objective());
    ///         }
    ///
    ///         // Log Stage 2 progress
    ///         if let Some(stage2) = obs.stage2() {
    ///             println!("Stage 2: Iteration {}, {} solutions",
    ///                 stage2.current_iteration(),
    ///                 stage2.solution_set_size());
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
    /// Controls how often the callback is invoked during Stage 2. For example,
    /// a frequency of 10 means the callback is called every 10 iterations.
    ///
    /// # Arguments
    ///
    /// * `frequency` - Number of iterations between callback calls
    ///
    /// # Default Behavior
    ///
    /// - Default frequency is 1 (callback called every iteration)
    /// - If no callback has been set with `with_callback()`, this method will
    ///   automatically use the default callback
    ///
    /// # Performance Considerations
    ///
    /// Lower frequencies reduce callback overhead but provide less detailed monitoring.
    /// Higher frequencies provide more detailed progress information but may impact
    /// performance for very fast optimization problems.
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
    ///
    /// // Custom callback with custom frequency
    /// let observer = Observer::new()
    ///     .with_stage2_tracking()
    ///     .with_callback(|obs| {
    ///         // Custom logging logic
    ///     })
    ///     .with_callback_frequency(25); // Custom callback every 25 iterations
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
    /// for both stages of the optimization. The default callback prints progress
    /// information to stderr (using `eprintln!`).
    ///
    /// # Stage 1 Logging
    ///
    /// Logs major substages:
    /// - Scatter search start
    /// - Initialization completion
    /// - Diversification completion
    /// - Intensification completion
    /// - Scatter search completion
    /// - Local optimization completion
    ///
    /// # Stage 2 Logging
    ///
    /// Logs iteration progress according to callback frequency:
    /// - Current iteration number
    /// - Best objective value found
    /// - Current solution set size
    /// - Merit filter threshold value
    /// - Local solver call counts
    /// - Function evaluation counts
    ///
    /// # Output Format
    ///
    /// ```
    /// [Stage 1] Scatter Search Complete | Best: 1.234567
    /// [Stage 2] Iter 50 | Best: 0.123456 | Solutions: 8 | Threshold: 0.500000 | Local Calls: 25 | Fn Evals: 1250
    /// ```
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
    ///
    /// # Controlling Frequency
    ///
    /// Use `with_callback_frequency()` to control how often Stage 2 updates are printed:
    ///
    /// ```rust
    /// let observer = Observer::new()
    ///     .with_stage1_tracking()
    ///     .with_stage2_tracking()
    ///     .with_default_callback()
    ///     .with_callback_frequency(10); // Print every 10 iterations
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
    ///
    /// Records the current time as the start time for the optimization.
    /// This is called internally when optimization begins.
    pub(crate) fn start_timer(&mut self) {
        if self.track_timing {
            self.start_time = Some(Instant::now());
        }
    }

    /// Get elapsed time in seconds
    ///
    /// Returns the time elapsed since `start_timer()` was called.
    /// Returns `None` if timing is not enabled or timer hasn't started.
    pub fn elapsed_time(&self) -> Option<f64> {
        self.start_time.map(|start| start.elapsed().as_secs_f64())
    }

    /// Check if Stage 1 should be observed
    ///
    /// Returns true if Stage 1 tracking is enabled and the observer mode
    /// allows Stage 1 observation (Stage1Only or Both modes).
    pub fn should_observe_stage1(&self) -> bool {
        matches!(self.mode, ObserverMode::Stage1Only | ObserverMode::Both) && self.stage1.is_some()
    }

    /// Check if Stage 2 should be observed
    ///
    /// Returns true if Stage 2 tracking is enabled and the observer mode
    /// allows Stage 2 observation (Stage2Only or Both modes).
    pub fn should_observe_stage2(&self) -> bool {
        matches!(self.mode, ObserverMode::Stage2Only | ObserverMode::Both) && self.stage2.is_some()
    }

    /// Get Stage 1 state reference
    ///
    /// Returns the current Stage 1 state if Stage 1 tracking is enabled and
    /// Stage 1 is still active. Returns `None` after Stage 1 completes to
    /// prevent repeated callback invocations.
    ///
    /// For final Stage 1 statistics after completion, use `stage1_final()`.
    ///
    /// # Returns
    ///
    /// - `Some(&Stage1State)` if Stage 1 is active and tracking is enabled
    /// - `None` if Stage 1 has completed or tracking is disabled
    ///
    /// # Example
    ///
    /// ```rust
    /// use globalsearch::observers::Observer;
    ///
    /// let observer = Observer::new().with_stage1_tracking();
    ///
    /// // In a callback during Stage 1
    /// if let Some(stage1) = observer.stage1() {
    ///     println!("Current best: {}", stage1.best_objective());
    ///     println!("Reference set size: {}", stage1.reference_set_size());
    /// }
    /// ```
    pub fn stage1(&self) -> Option<&Stage1State> {
        // Don't return Stage 1 state after it's completed to prevent repeated logging
        if self.stage1_completed {
            None
        } else {
            self.stage1.as_ref()
        }
    }

    /// Get Stage 1 state reference even after completion (for final statistics)
    ///
    /// Returns the final Stage 1 state regardless of whether Stage 1 is still
    /// active. This method should be used for accessing final statistics after
    /// optimization completes.
    ///
    /// # Returns
    ///
    /// - `Some(&Stage1State)` if Stage 1 tracking was enabled
    /// - `None` if Stage 1 tracking was not enabled
    ///
    /// # Difference from `stage1()`
    ///
    /// - `stage1()` returns `None` after Stage 1 completes (to prevent repeated callbacks)
    /// - `stage1_final()` always returns the final state when available
    ///
    /// # Example
    ///
    /// ```rust
    /// use globalsearch::observers::Observer;
    ///
    /// // After optimization completes
    /// if let Some(observer) = optimizer.observer() {
    ///     if let Some(stage1) = observer.stage1_final() {
    ///         println!("Stage 1 Summary:");
    ///         println!("  Total function evaluations: {}", stage1.function_evaluations());
    ///         println!("  Trial points generated: {}", stage1.trial_points_generated());
    ///         println!("  Final reference set size: {}", stage1.reference_set_size());
    ///         if let Some(time) = stage1.total_time() {
    ///             println!("  Total time: {:.3}s", time);
    ///         }
    ///     }
    /// }
    /// ```
    pub fn stage1_final(&self) -> Option<&Stage1State> {
        self.stage1.as_ref()
    }

    /// Get mutable Stage 1 state reference
    ///
    /// Used internally by the OQNLP algorithm to update Stage 1 metrics.
    /// Returns None if Stage 1 tracking is not enabled.
    pub(crate) fn stage1_mut(&mut self) -> Option<&mut Stage1State> {
        self.stage1.as_mut()
    }

    /// Mark Stage 1 as completed (prevents further Stage 1 callback invocations)
    ///
    /// Called internally when Stage 1 finishes. This prevents the observer
    /// from returning Stage 1 state in subsequent `stage1()` calls,
    /// avoiding repeated callback invocations for completed stages.
    pub(crate) fn mark_stage1_complete(&mut self) {
        self.stage1_completed = true;
    }

    /// Get Stage 2 state reference
    ///
    /// Returns the current Stage 2 state if Stage 2 tracking is enabled and
    /// Stage 2 has started. Returns `None` before Stage 2 begins to prevent
    /// premature callback invocations.
    ///
    /// # Returns
    ///
    /// - `Some(&Stage2State)` if Stage 2 is active and tracking is enabled
    /// - `None` if Stage 2 hasn't started yet or tracking is disabled
    ///
    /// # Example
    ///
    /// ```rust
    /// use globalsearch::observers::Observer;
    ///
    /// let observer = Observer::new().with_stage2_tracking();
    ///
    /// // In a callback during Stage 2
    /// if let Some(stage2) = observer.stage2() {
    ///     println!("Iteration: {}", stage2.current_iteration());
    ///     println!("Best objective: {}", stage2.best_objective());
    ///     println!("Solution set size: {}", stage2.solution_set_size());
    /// }
    /// ```
    pub fn stage2(&self) -> Option<&Stage2State> {
        // Don't return Stage 2 state until it has started to prevent premature logging
        if self.stage2_started {
            self.stage2.as_ref()
        } else {
            None
        }
    }

    /// Get mutable Stage 2 state reference
    ///
    /// Used internally by the OQNLP algorithm to update Stage 2 metrics.
    /// Returns None if Stage 2 tracking is not enabled.
    pub(crate) fn stage2_mut(&mut self) -> Option<&mut Stage2State> {
        self.stage2.as_mut()
    }

    /// Mark Stage 2 as started (allows Stage 2 callback invocations)
    ///
    /// Called internally when Stage 2 begins. This allows the observer
    /// to return Stage 2 state in subsequent `stage2()` calls,
    /// enabling callback invocations for active Stage 2 operation.
    pub(crate) fn mark_stage2_started(&mut self) {
        self.stage2_started = true;
    }

    /// Check if timing is enabled
    ///
    /// Returns true if the observer is configured to track timing information.
    pub fn is_timing_enabled(&self) -> bool {
        self.track_timing
    }

    /// Invoke the callback if one is set
    ///
    /// Called internally by the OQNLP algorithm at appropriate points during
    /// optimization. The callback receives a reference to this observer,
    /// allowing access to all current metrics.
    pub(crate) fn invoke_callback(&self) {
        if let Some(ref callback) = self.callback {
            callback(self);
        }
    }

    /// Check if callback should be invoked for the current iteration
    ///
    /// Determines whether the callback should be called based on the current
    /// iteration number and the configured callback frequency.
    ///
    /// # Arguments
    ///
    /// * `iteration` - Current iteration number in Stage 2
    ///
    /// # Returns
    ///
    /// True if a callback is configured and the iteration is a multiple of
    /// the callback frequency.
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
