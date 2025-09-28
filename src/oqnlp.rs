//! # OQNLP (OptQuest for Nonlinear Programming) Module
//!
//! The OQNLP (OptQuest/NLP) algorithm is a global optimization algorithm that combines scatter search with local optimization methods.
//! This module implements the OQNLP global optimization algorithm, which combines
//! the systematic exploration of scatter search with the local refinement capabilities
//! of classical optimization methods.
//!
//! ## Algorithm Overview
//!
//! OQNLP is a hybrid metaheuristic that operates through a two-stage process:
//!
//! ### Stage 1: Reference Set Construction
//! - **Population Initialization**: Generate diverse candidate solutions within bounds
//! - **Quality Evaluation**: Assess all candidates using the objective function
//! - **Local Optimization**: Apply local solvers to the best candidate
//!
//! ### Stage 2: Iterative Improvement
//! - **Merit Filtering**: Optimize only solutions meeting quality thresholds
//! - **Distance Filtering**: Ensure sufficient diversity between solutions
//! - **Local Refinement**: Use local solvers to polish selected candidates
//! - **Solution Set Update**: Replace inferior solutions with new discoveries
//!
//! ## Key Features
//!
//! ### Global Search Capability
//! - **Systematic Exploration**: Scatter search prevents premature convergence
//! - **Diversity Maintenance**: Distance filters ensure broad search coverage
//! - **Multi-modal Optimization**: Can find multiple local optima simultaneously
//!
//! ### Local Refinement
//! - **Gradient-Based Methods**: L-BFGS, trust region, Newton-CG for smooth problems
//! - **Derivative-Free Methods**: Nelder-Mead, COBYLA for non-smooth problems
//! - **Flexible Configuration**: Easily switch between different local solvers
//!
//! ### Robustness Features
//! - **Checkpointing Support**: Resume long optimizations from saved states
//! - **Constraint Handling**: Support for bound and general nonlinear constraints using COBYLA
//! - **Error Recovery**: Graceful handling of function evaluation failures
//!
//! ## Mathematical Formulation
//!
//! OQNLP solves optimization problems of the form:
//!
//! ```text
//! minimize    f(x)
//! subject to  lₙ ≤ xₙ ≤ uₙ    (bound constraints)
//!             gₙ(x) ≥ 0        (inequality constraints)
//! ```
//!
//! Where:
//! - `f(x)`: Objective function to minimize
//! - `x`: Decision variables vector
//! - `lₙ, uₙ`: Lower and upper bounds for variable i
//! - `gₙ(x)`: Inequality constraint functions
//!
//! ## Example: Six-Hump Camel Function
//!
//! ```rust
//! // Molga, M., & Smutnicki, C. Test functions for optimization needs (April 3, 2005), pp. 27-28. Retrieved January 2025, from https://robertmarks.org/Classes/ENGR5358/Papers/functions.pdf
//! use globalsearch::local_solver::builders::{TrustRegionBuilder, TrustRegionRadiusMethod};
//! use globalsearch::problem::Problem;
//! use globalsearch::{
//!     oqnlp::OQNLP,
//!     types::{EvaluationError, LocalSolverType, OQNLPParams, SolutionSet},
//! };
//! use ndarray::{array, Array1, Array2};
//!
//! #[derive(Debug, Clone)]
//! pub struct SixHumpCamel;
//!
//! impl Problem for SixHumpCamel {
//!    fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
//!       Ok(
//!          (4.0 - 2.1 * x[0].powi(2) + x[0].powi(4) / 3.0) * x[0].powi(2)
//!            + x[0] * x[1]
//!            + (-4.0 + 4.0 * x[1].powi(2)) * x[1].powi(2),
//!          )
//!     }
//!
//!     // Calculated analytically, reference didn't provide gradient
//!     fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
//!         Ok(array![
//!             (8.0 - 8.4 * x[0].powi(2) + 2.0 * x[0].powi(4)) * x[0] + x[1],
//!             x[0] + (-8.0 + 16.0 * x[1].powi(2)) * x[1]
//!         ])
//!     }
//!
//!     // Calculated analytically, reference didn't provide hessian
//!     fn hessian(&self, x: &Array1<f64>) -> Result<Array2<f64>, EvaluationError> {
//!         Ok(array![
//!             [(4.0 * x[0].powi(2) - 4.2) * x[0].powi(2)
//!                 + 4.0 * (4.0 / 3.0 * x[0].powi(3) - 4.2 * x[0]) * x[0]
//!                 + 2.0 * (x[0].powi(4) / 3.0 - 2.1 * x[0].powi(2) + 4.0),
//!             1.0],
//!             [1.0, 40.0 * x[1].powi(2) + 2.0 * (4.0 * x[1].powi(2) - 4.0)]])
//!     }
//!
//!     fn variable_bounds(&self) -> Array2<f64> {
//!         array![[-3.0, 3.0], [-2.0, 2.0]]
//!     }
//! }
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Solving the Six-Hump Camel problem using globalsearch-rs
//!     // using the Trust Region method as a local solver
//!     let problem: SixHumpCamel = SixHumpCamel;
//!
//!     // Set the parameters for the OQNLP algorithm
//!     // It is recommended that you adjust these parameters to your problem
//!     // and the desired behavior of the algorithm, instead of using the default values.
//!     let params: OQNLPParams = OQNLPParams {
//!         local_solver_type: LocalSolverType::TrustRegion,
//!         local_solver_config: TrustRegionBuilder::default()
//!             .method(TrustRegionRadiusMethod::Steihaug)
//!             .build(),
//!             ..OQNLPParams::default()
//!     };
//!
//!     let mut oqnlp: OQNLP<SixHumpCamel> = OQNLP::new(problem, params).unwrap();
//!     let sol_set: SolutionSet = oqnlp.run().unwrap();
//!     println!("Solution set:");
//!     println!("{}", sol_set);
//!     Ok(())
//!
//! }
//! ```

use crate::filters::{DistanceFilter, MeritFilter};
use crate::local_solver::runner::LocalSolver;
use crate::problem::Problem;
use crate::scatter_search::ScatterSearch;
use crate::types::{FilterParams, LocalSolution, OQNLPParams, SolutionSet};
#[cfg(feature = "checkpointing")]
use crate::{
    checkpoint::{CheckpointError, CheckpointManager},
    types::{CheckpointConfig, OQNLPCheckpoint},
};
#[cfg(feature = "checkpointing")]
use chrono;
#[cfg(feature = "progress_bar")]
use kdam::{Bar, BarExt};
use ndarray::Array1;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use thiserror::Error;

// TODO: We could do batched stage 2 to implement rayon parallelism

#[derive(Debug, Error)]
/// ONQLP errors
pub enum OQNLPError {
    /// Error when the local solver fails to find a solution
    #[error("OQNLP Error: Local solver failed to find a solution. {0}")]
    LocalSolverError(String),

    /// Error when OQNLP fails to find a feasible solution
    #[error("OQNLP Error: No feasible solution found.")]
    NoFeasibleSolution,

    /// Error when the objective function evaluation fails
    #[error("OQNLP Error: Objective function evaluation failed.")]
    ObjectiveFunctionEvaluationFailed,

    /// Error when creating a new ScatterSearch instance
    #[error("OQNLP Error: Failed to create a new ScatterSearch instance: {0}")]
    ScatterSearchError(#[from] crate::scatter_search::ScatterSearchError),

    /// Error when running the ScatterSearch instance
    #[error("OQNLP Error: Failed to run the ScatterSearch instance: {0}")]
    ScatterSearchRunError(crate::scatter_search::ScatterSearchError),

    /// Error when the population size is invalid
    ///
    /// Population size should be at least 3, since the reference set
    /// pushes the bounds and the midpoint.
    #[error("OQNLP Error: Population size should be at least 3, got {0}. Reference Set size should be at least 3, since it pushes the bounds and the midpoint.")]
    InvalidPopulationSize(usize),

    /// Error when the iterations are invalid
    ///
    /// Iterations should be less than or equal to the population size.
    #[error("OQNLP Error: Iterations should be less than or equal to population size. OQNLP received `iterations`: {0}, `population size`: {1}.")]
    InvalidIterations(usize, usize),

    /// Error when creating the distance filter
    #[error("OQNLP Error: Failed to create distance filter. {0}")]
    DistanceFilterError(String),

    /// Error related to checkpointing operations
    #[cfg(feature = "checkpointing")]
    #[error("OQNLP Error: Checkpointing error: {0}")]
    CheckpointError(#[from] CheckpointError),
}

/// Main OQNLP optimization algorithm implementation.
///
/// This struct encapsulates the complete OQNLP (OptQuest for Nonlinear Programming) algorithm,
/// which combines scatter search with local optimization methods to solve global optimization
/// problems efficiently.
///
/// ## Algorithm Components
///
/// The OQNLP struct manages several key algorithmic components:
///
/// ### Core Algorithm Elements
/// - **Problem Definition**: The optimization problem implementing the [`Problem`] trait
/// - **Algorithm Parameters**: Configuration via [`OQNLPParams`] controlling behavior
/// - **Filtering Mechanisms**: Merit and distance filters for solution quality and diversity
/// - **Local Solver**: Refinement of candidate solutions using classical optimization methods
///
/// ### Solution Management
/// - **Solution Set**: Collection of high-quality solutions discovered during optimization
/// - **Reference Set**: Internal population of candidate solutions for scatter search
/// - **Checkpointing**: Optional state persistence for long-running optimizations
///
/// ## Optimization Process
///
/// The algorithm operates through a two-stage process:
///
/// 1. **Stage 1 (Initialization)**: Build initial reference set via scatter search
/// 2. **Stage 2 (Refinement)**: Iteratively improve solutions through local optimization
///
/// ## Configuration Options
///
/// ### Time Control
/// - [`max_time()`](OQNLP::max_time): Set maximum optimization time
/// - [`target_objective()`](OQNLP::target_objective): Early stopping criterion
///
/// ### Solution Quality
/// - [`exclude_out_of_bounds()`](OQNLP::exclude_out_of_bounds): Enforce variable bounds
/// - [`verbose()`](OQNLP::verbose): Enable detailed progress output
///
#[cfg_attr(feature = "checkpointing", doc = "### State Persistence")]
#[cfg_attr(feature = "checkpointing", doc = "- [`with_checkpointing()`](Self::with_checkpointing): Enable automatic state saving")]
#[cfg_attr(feature = "checkpointing", doc = "- [`resume_with_modified_params()`](Self::resume_with_modified_params): Continue with new parameters")]
///
/// ## Example Usage
///
/// ```rust
/// use globalsearch::{oqnlp::OQNLP, types::OQNLPParams, problem::Problem, types::EvaluationError};
/// use ndarray::{array, Array1, Array2};
///
/// #[derive(Clone)]
/// struct QuadraticProblem;
///
/// impl Problem for QuadraticProblem {
///     fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
///         Ok(x[0].powi(2) + x[1].powi(2))  // Minimize x² + y²
///     }
///
///     fn variable_bounds(&self) -> Array2<f64> {
///         array![[-5.0, 5.0], [-5.0, 5.0]]  // x, y ∈ [-5, 5]
///     }
/// }
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let problem = QuadraticProblem;
/// let params = OQNLPParams::default();
///
/// let mut optimizer = OQNLP::new(problem, params)?
///     .target_objective(0.01)  // Stop when f(x) ≤ 0.01
///     .max_time(60.0)          // Maximum 60 seconds
///     .verbose();              // Show progress
///
/// let solutions = optimizer.run()?;
/// println!("Found {} solutions", solutions.len());
/// # Ok(())
/// # }
/// ```
pub struct OQNLP<P: Problem + Clone> {
    /// Optimization problem definition.
    ///
    /// Defines the objective function, constraints, and variable bounds for the optimization.
    problem: P,

    /// Algorithm configuration parameters.
    ///
    /// Controls population size, iterations, convergence criteria, and local solver selection.
    params: OQNLPParams,
    /// Merit filter for solution quality control.
    ///
    /// Maintains dynamic threshold to accept only improving or competitive solutions.
    merit_filter: MeritFilter,

    /// Distance filter for population diversity.
    ///
    /// Enforces minimum separation between solutions to prevent clustering.
    distance_filter: DistanceFilter,

    /// Local optimization solver interface.
    ///
    /// Refines candidate solutions using classical optimization algorithms.
    local_solver: LocalSolver<P>,

    /// Collection of high-quality solutions discovered.
    ///
    /// Maintains the best solutions found during optimization. Initially `None`.
    solution_set: Option<SolutionSet>,

    /// Maximum optimization time limit (seconds).
    ///
    /// When set, optimization stops after this duration in Stage 2.
    /// Timer starts after the first local search completes.
    max_time: Option<f64>,

    /// Verbose flag to enable additional output during the optimization process.
    verbose: bool,

    /// Early stopping criterion based on objective value.
    ///
    /// Optimization stops when a solution with objective ≤ this value is found.
    /// Useful when the global optimum or "good enough" threshold is known.
    target_objective: Option<f64>,

    /// Variable bounds enforcement flag.
    ///
    /// When `true`, solutions outside variable bounds are rejected.
    /// Combined with `target_objective`, both conditions must be satisfied.
    exclude_out_of_bounds: bool,

    /// Checkpoint manager for state persistence (optional).
    ///
    /// Handles saving and loading optimization state for long-running problems.
    #[cfg(feature = "checkpointing")]
    checkpoint_manager: Option<CheckpointManager>,

    /// Current iteration counter for checkpointing.
    #[cfg(feature = "checkpointing")]
    current_iteration: usize,

    /// Active reference set for scatter search state.
    #[cfg(feature = "checkpointing")]
    current_reference_set: Option<Vec<Array1<f64>>>,

    /// Counter for cycles without solution improvement.
    #[cfg(feature = "checkpointing")]
    unchanged_cycles: usize,

    /// Optimization start timestamp for elapsed time tracking.
    #[cfg(feature = "checkpointing")]
    start_time: Option<std::time::Instant>,

    /// Current random seed for reproducible continuation.
    #[cfg(feature = "checkpointing")]
    current_seed: u64,
}

impl<P: Problem + Clone + Send + Sync> OQNLP<P> {
    /// # OQNLP Constructor
    ///
    /// This method creates a new OQNLP instance with the given optimization problem and parameters.
    ///
    /// ## Errors
    ///
    /// Returns an error if the population size is less than 3 or if the iterations are greater than the population size.
    ///
    /// Returns an error if the distance filter fails to be created.
    ///
    /// ## Warnings
    ///
    /// If the `wait_cycle` parameter is greater than or equal to the `iterations` parameter, a warning is printed.
    pub fn new(problem: P, params: OQNLPParams) -> Result<Self, OQNLPError> {
        if params.population_size <= 3 {
            return Err(OQNLPError::InvalidPopulationSize(params.population_size));
        }

        if params.iterations > params.population_size {
            return Err(OQNLPError::InvalidIterations(
                params.iterations,
                params.population_size,
            ));
        }

        if params.wait_cycle >= params.iterations {
            eprintln!(
                "Warning: `wait_cycle` is greater than or equal to `iterations`. This may lead to suboptimal results."
            );
        }

        let filter_params: FilterParams = FilterParams {
            distance_factor: params.distance_factor,
            wait_cycle: params.wait_cycle,
            threshold_factor: params.threshold_factor,
        };

        Ok(Self {
            problem: problem.clone(),
            params: params.clone(),
            merit_filter: MeritFilter::new(),
            distance_filter: DistanceFilter::new(filter_params)
                .map_err(|e| OQNLPError::DistanceFilterError(e.to_string()))?,
            local_solver: LocalSolver::new(
                problem,
                params.local_solver_type.clone(),
                params.local_solver_config.clone(),
            ),
            solution_set: None,
            max_time: None,
            verbose: false,
            target_objective: None,
            exclude_out_of_bounds: false,
            #[cfg(feature = "checkpointing")]
            checkpoint_manager: None,
            #[cfg(feature = "checkpointing")]
            current_iteration: 0,
            #[cfg(feature = "checkpointing")]
            current_reference_set: None,
            #[cfg(feature = "checkpointing")]
            unchanged_cycles: 0,
            #[cfg(feature = "checkpointing")]
            start_time: None,
            #[cfg(feature = "checkpointing")]
            current_seed: params.seed,
        })
    }

    /// Run the OQNLP algorithm and return the solution set
    pub fn run(&mut self) -> Result<SolutionSet, OQNLPError> {
        // Try to resume from checkpoint if enabled
        #[cfg(feature = "checkpointing")]
        let resumed_from_checkpoint = self.try_resume_from_checkpoint()?;
        #[cfg(not(feature = "checkpointing"))]
        let resumed_from_checkpoint = false;

        // Set start time for elapsed time calculation
        #[cfg(feature = "checkpointing")]
        if self.start_time.is_none() {
            self.start_time = Some(std::time::Instant::now());
        }

        // If we resumed from checkpoint, skip Stage 1 and jump to Stage 2
        let (mut ref_set, mut unchanged_cycles) = if resumed_from_checkpoint {
            #[cfg(feature = "checkpointing")]
            {
                if self.verbose {
                    println!(
                        "Resuming from checkpoint at iteration {}",
                        self.current_iteration
                    );
                }
                let ref_set = self.current_reference_set.clone().unwrap_or_default();
                let unchanged_cycles = self.unchanged_cycles;
                (ref_set, unchanged_cycles)
            }
            #[cfg(not(feature = "checkpointing"))]
            unreachable!()
        } else {
            // Stage 1: Initial ScatterSearch iterations and first local call
            if self.verbose {
                println!("Starting Stage 1");
            }

            let ss: ScatterSearch<P> =
                ScatterSearch::new(self.problem.clone(), self.params.clone())?;
            let (ref_set, scatter_candidate) =
                ss.run().map_err(|e| OQNLPError::ScatterSearchRunError(e))?;
            let local_sol: LocalSolution = self
                .local_solver
                .solve(scatter_candidate)
                .map_err(|e| OQNLPError::LocalSolverError(e.to_string()))?;

            self.merit_filter.update_threshold(local_sol.objective);

            if self.verbose {
                println!(
                    "Stage 1: Local solution found with objective = {:.8}",
                    local_sol.objective
                );
            }

            self.process_local_solution(local_sol)?;

            // Check if target objective has been reached after Stage 1
            if self.target_objective_reached() {
                if self.verbose {
                    println!(
                        "Stage 1: Target objective {:.8} reached. Stopping optimization.",
                        self.target_objective.unwrap()
                    );
                }
                return self
                    .solution_set
                    .clone()
                    .ok_or(OQNLPError::NoFeasibleSolution);
            }

            // Store reference set for checkpointing
            #[cfg(feature = "checkpointing")]
            {
                self.current_reference_set = Some(ref_set.clone());
            }

            (ref_set, 0)
        };

        if self.verbose {
            println!("Starting Stage 2");
        }

        #[cfg(feature = "progress_bar")]
        let mut stage2_bar = Bar::builder()
            .total(self.params.iterations)
            .desc("Stage 2")
            .unit("it")
            .postfix(&format!(
                "Objective function: {:.6}",
                self.solution_set
                    .as_ref()
                    .and_then(|s| s.best_solution())
                    .map_or(f64::INFINITY, |s| s.objective)
            ))
            .build()
            .expect("Failed to create progress bar");

        // Adjust progress bar for resumed runs
        #[cfg(all(feature = "progress_bar", feature = "checkpointing"))]
        if resumed_from_checkpoint {
            for _ in 0..self.current_iteration {
                stage2_bar.update(1).expect("Failed to update progress bar");
            }
        }

        // Stage 2: Main iterative loop
        #[cfg(feature = "checkpointing")]
        let mut rng: StdRng = if resumed_from_checkpoint {
            // Use the saved seed when resuming from checkpoint
            StdRng::seed_from_u64(self.current_seed)
        } else {
            // Start fresh with base seed + current iteration
            let seed = self.params.seed + self.current_iteration as u64;
            self.current_seed = seed;
            StdRng::seed_from_u64(seed)
        };

        #[cfg(not(feature = "checkpointing"))]
        let mut rng: StdRng = StdRng::seed_from_u64(self.params.seed);

        // Shuffle reference set if starting fresh
        if !resumed_from_checkpoint {
            ref_set.shuffle(&mut rng);
        }

        let start_timer: Option<std::time::Instant> =
            self.max_time.map(|_| std::time::Instant::now());

        // Start from current iteration if resumed, otherwise from 0
        #[cfg(feature = "checkpointing")]
        let start_iter = if resumed_from_checkpoint {
            self.current_iteration
        } else {
            0
        };
        #[cfg(not(feature = "checkpointing"))]
        let start_iter = 0;

        for (local_iter, trial) in ref_set
            .iter()
            .take(self.params.iterations)
            .enumerate()
            .skip(start_iter)
        {
            #[cfg(feature = "checkpointing")]
            {
                self.current_iteration = local_iter;
                self.unchanged_cycles = unchanged_cycles;
            }

            // Update the current seed for checkpointing
            #[cfg(feature = "checkpointing")]
            {
                self.current_seed = self.params.seed + self.current_iteration as u64;
            }

            #[cfg(feature = "progress_bar")]
            if !resumed_from_checkpoint || local_iter >= start_iter {
                stage2_bar.update(1).expect("Failed to update progress bar");
            }

            if let (Some(max_secs), Some(start)) = (self.max_time, start_timer) {
                if start.elapsed().as_secs_f64() > max_secs {
                    if self.verbose {
                        println!("Timeout reached after {} seconds", max_secs);
                    }
                    break;
                }
            }

            let trial = trial.clone();
            let obj: f64 = self
                .problem
                .objective(&trial)
                .map_err(|_| OQNLPError::ObjectiveFunctionEvaluationFailed)?;
            if self.should_start_local(&trial, obj)? {
                self.merit_filter.update_threshold(obj);
                let local_trial: LocalSolution = self
                    .local_solver
                    .solve(trial)
                    .map_err(|e| OQNLPError::LocalSolverError(e.to_string()))?;
                let added: bool = self.process_local_solution(local_trial.clone())?;

                if self.verbose && added {
                    println!(
                        "Stage 2, iteration {}: Added local solution found with objective = {:.8}",
                        local_iter, local_trial.objective
                    );
                    println!("x0 = {}", local_trial.point);
                }

                #[cfg(feature = "progress_bar")]
                if added {
                    stage2_bar
                        .set_postfix(&format!("Objective function: {:.6}", local_trial.objective));
                    stage2_bar
                        .refresh()
                        .expect("Failed to refresh progress bar");
                }

                // Check if target objective has been reached
                if self.target_objective_reached() {
                    if self.verbose {
                        println!(
                            "Stage 2, iteration {}: Target objective {:.8} reached. Stopping optimization.",
                            local_iter,
                            self.target_objective.unwrap()
                        );
                    }
                    break;
                }
            } else {
                unchanged_cycles += 1;

                if unchanged_cycles >= self.params.wait_cycle {
                    if self.verbose {
                        println!(
                            "Stage 2, iteration {}: Adjusting threshold from {:.8} to {:.8}",
                            local_iter,
                            self.merit_filter.threshold,
                            self.merit_filter.threshold + 0.1 * self.merit_filter.threshold.abs()
                        );
                    }

                    self.adjust_threshold(self.merit_filter.threshold);
                    unchanged_cycles = 0;
                }
            }

            // Save checkpoint if enabled and conditions are met
            #[cfg(feature = "checkpointing")]
            self.maybe_save_checkpoint()?;
        }

        // Save final checkpoint at the end of optimization
        #[cfg(feature = "checkpointing")]
        self.maybe_save_final_checkpoint()?;

        self.solution_set
            .clone()
            .ok_or(OQNLPError::NoFeasibleSolution)
    }

    // Helper methods
    /// Check if a local search should be started based on the merit and distance filters
    fn should_start_local(&self, point: &Array1<f64>, obj: f64) -> Result<bool, OQNLPError> {
        let passes_merit: bool = obj <= self.merit_filter.threshold;
        let passes_distance: bool = self.distance_filter.check(point);
        Ok(passes_merit && passes_distance)
    }

    /// Check if the target objective has been reached
    fn target_objective_reached(&self) -> bool {
        if let (Some(target), Some(solution_set)) = (self.target_objective, &self.solution_set) {
            if let Some(best) = solution_set.best_solution() {
                let target_reached = best.objective <= target;

                // If exclude_out_of_bounds is enabled, also check if the solution is within bounds
                if self.exclude_out_of_bounds {
                    return target_reached && self.is_within_bounds(&best.point);
                }

                return target_reached;
            }
        }
        false
    }

    /// Check if a point is within the variable bounds
    fn is_within_bounds(&self, point: &Array1<f64>) -> bool {
        let bounds = self.problem.variable_bounds();

        for (i, &value) in point.iter().enumerate() {
            let lower_bound = bounds[[i, 0]];
            let upper_bound = bounds[[i, 1]];

            if value < lower_bound || value > upper_bound {
                return false;
            }
        }

        true
    }

    /// Process a local solution, updating the best solution and filters
    fn process_local_solution(&mut self, solution: LocalSolution) -> Result<bool, OQNLPError> {
        const ABS_TOL: f64 = 1e-8;
        const REL_TOL: f64 = 1e-6;

        // If exclude_out_of_bounds is enabled, check if the solution is within bounds
        if self.exclude_out_of_bounds && !self.is_within_bounds(&solution.point) {
            if self.verbose {
                println!(
                    "Solution with objective {:.8} rejected: out of bounds",
                    solution.objective
                );
            }
            // Still add to distance filter to maintain diversity, but don't add to solution set
            self.distance_filter.add_solution(solution);
            return Ok(false);
        }

        let solutions = if let Some(existing) = &self.solution_set {
            existing.solutions.clone()
        } else {
            // First solution, initialize solution set
            self.solution_set = Some(SolutionSet {
                solutions: Array1::from(vec![solution.clone()]),
            });
            self.merit_filter.update_threshold(solution.objective);
            self.distance_filter.add_solution(solution);
            return Ok(true);
        };

        let current_best: &LocalSolution = &solutions[0];
        let obj1 = solution.objective;
        let obj2 = current_best.objective;

        let obj_diff = (obj1 - obj2).abs();
        let tol = ABS_TOL.max(REL_TOL * obj1.abs().max(obj2.abs()));

        let added: bool = if obj1 < obj2 - tol {
            // Found new best solution
            self.solution_set = Some(SolutionSet {
                solutions: Array1::from(vec![solution.clone()]),
            });
            self.merit_filter.update_threshold(solution.objective);
            false
        } else if obj_diff <= tol && !self.is_duplicate_in_set(&solution, &solutions) {
            // Similar objective value and not duplicate, add to set
            let mut new_solutions: Vec<LocalSolution> = solutions.to_vec();
            new_solutions.push(solution.clone());
            self.solution_set = Some(SolutionSet {
                solutions: Array1::from(new_solutions),
            });

            true
        } else {
            false
        };

        self.distance_filter.add_solution(solution);
        Ok(added)
    }

    /// Check if a candidate solution is a duplicate in a set of solutions
    fn is_duplicate_in_set(&self, candidate: &LocalSolution, set: &Array1<LocalSolution>) -> bool {
        let distance_threshold = self.params.distance_factor;

        set.iter().any(|s| {
            let diff = &candidate.point - &s.point;
            let dist_squared = diff.dot(&diff);
            dist_squared < distance_threshold * distance_threshold
        })
    }

    /// Set the maximum time for the stage 2 of the OQNLP algorithm and return self for chaining.
    ///
    /// If the time limit is reached, the algorithm will stop and return the
    /// solution set found so far.
    ///
    /// The time limit is in seconds and it starts timing after the
    /// first local search.
    pub fn max_time(mut self, max_time: f64) -> Self {
        self.max_time = Some(max_time);
        self
    }

    /// Enable verbose output for the OQNLP algorithm
    ///
    /// This will print additional information about the optimization process
    pub fn verbose(mut self) -> Self {
        self.verbose = true;
        self
    }

    /// Set a target objective function value to stop optimization early
    ///
    /// If the best solution found has an objective function value less than or equal to this target,
    /// the optimization will stop early. This can be useful when you know the global optimum
    /// or want to stop when a "good enough" solution is found.
    ///
    /// # Arguments
    ///
    /// * `target` - The target objective function value
    ///
    /// # Example
    ///
    /// ```rust
    /// # use globalsearch::oqnlp::OQNLP;
    /// # use globalsearch::types::OQNLPParams;
    /// # use globalsearch::problem::Problem;
    /// # use globalsearch::types::EvaluationError;
    /// # use ndarray::{Array1, Array2, array};
    /// # #[derive(Clone)]
    /// # struct TestProblem;
    /// # impl Problem for TestProblem {
    /// #     fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> { Ok(x.sum().powi(2)) }
    /// #     fn variable_bounds(&self) -> Array2<f64> { array![[-1.0, 1.0]] }
    /// # }
    /// let mut oqnlp = OQNLP::new(TestProblem, OQNLPParams::default())
    ///     .unwrap()
    ///     .target_objective(0.0);  // Stop when objective <= 0.0
    /// ```
    pub fn target_objective(mut self, target: f64) -> Self {
        self.target_objective = Some(target);
        self
    }

    /// Enable or disable exclusion of out-of-bounds solutions
    ///
    /// When enabled, the algorithm will check if solutions are within the variable bounds
    /// before adding them to the solution set.
    pub fn set_exclude_out_of_bounds(mut self, enable: bool) -> Self {
        self.exclude_out_of_bounds = enable;
        self
    }

    /// Enable exclusion of out-of-bounds solutions
    ///
    /// This is a convenience method equivalent to calling `set_exclude_out_of_bounds(true)`.
    /// When enabled, the algorithm will check if solutions are within the variable bounds
    /// before adding them to the solution set.
    pub fn exclude_out_of_bounds(self) -> Self {
        self.set_exclude_out_of_bounds(true)
    }

    /// Enable checkpointing with the given configuration
    ///
    /// This allows saving and resuming the optimization state
    #[cfg(feature = "checkpointing")]
    pub fn with_checkpointing(mut self, config: CheckpointConfig) -> Result<Self, OQNLPError> {
        self.checkpoint_manager = Some(CheckpointManager::new(config)?);
        Ok(self)
    }

    /// Try to resume from the latest checkpoint if available
    ///
    /// Returns true if resumed from checkpoint, false if starting fresh
    #[cfg(feature = "checkpointing")]
    pub fn try_resume_from_checkpoint(&mut self) -> Result<bool, OQNLPError> {
        if let Some(ref manager) = self.checkpoint_manager {
            if manager.config().auto_resume && manager.checkpoint_exists() {
                let checkpoint = manager.load_latest_checkpoint()?;
                self.restore_from_checkpoint(checkpoint)?;
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Resume from the latest checkpoint with modified parameters
    ///
    /// This allows you to continue optimization with different parameters
    /// (e.g., more iterations, different population size, etc.)
    #[cfg(feature = "checkpointing")]
    pub fn resume_with_modified_params(
        &mut self,
        new_params: OQNLPParams,
    ) -> Result<bool, OQNLPError> {
        if let Some(ref manager) = self.checkpoint_manager {
            if manager.checkpoint_exists() {
                let mut checkpoint = manager.load_latest_checkpoint()?;

                // Keep the old params for comparison
                let old_iterations = checkpoint.params.iterations;
                let old_population_size = checkpoint.params.population_size;

                // Handle population size changes
                if new_params.population_size != old_population_size {
                    if new_params.population_size > old_population_size {
                        self.expand_reference_set(
                            &mut checkpoint.reference_set,
                            old_population_size,
                            new_params.population_size,
                        )?;

                        if self.verbose {
                            println!(
                                "Expanded reference set from {} to {} points",
                                old_population_size, new_params.population_size
                            );
                        }
                    } else {
                        // Population size decreased, issue warning but continue
                        eprintln!(
                            "Warning: New population size ({}) is smaller than original ({}). Using original reference set with {} points.",
                            new_params.population_size, old_population_size, old_population_size
                        );

                        if self.verbose {
                            println!(
                                "Keeping original reference set size of {} points despite smaller population_size parameter",
                                old_population_size
                            );
                        }
                    }
                }

                // Update with new parameters
                checkpoint.params = new_params.clone();

                self.restore_from_checkpoint(checkpoint)?;

                if self.verbose {
                    println!(
                        "Resumed with modified parameters. Iterations changed from {} to {}",
                        old_iterations, new_params.iterations
                    );
                }

                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Expand the reference set to a larger population size
    ///
    /// This method generates additional points to expand an existing reference set
    /// while maintaining diversity using the same strategy as ScatterSearch
    #[cfg(feature = "checkpointing")]
    fn expand_reference_set(
        &self,
        ref_set: &mut Vec<Array1<f64>>,
        old_size: usize,
        new_size: usize,
    ) -> Result<(), OQNLPError> {
        if new_size <= old_size {
            return Ok(());
        }

        // Create a temporary ScatterSearch instance to reuse its methods
        let temp_params = OQNLPParams {
            population_size: new_size,
            seed: self.current_seed + ref_set.len() as u64,
            ..self.params.clone()
        };

        let mut scatter_search = ScatterSearch::new(self.problem.clone(), temp_params)
            .map_err(|_| OQNLPError::ScatterSearchError)?;

        // ScatterSearch's diversify_reference_set method to expand our existing set
        scatter_search
            .diversify_reference_set(ref_set)
            .map_err(|_| OQNLPError::ScatterSearchError)?;

        Ok(())
    }

    /// Resume from a specific checkpoint file with modified parameters
    #[cfg(feature = "checkpointing")]
    pub fn resume_from_checkpoint_with_params(
        &mut self,
        checkpoint_path: &std::path::Path,
        new_params: OQNLPParams,
    ) -> Result<(), OQNLPError> {
        if let Some(ref manager) = self.checkpoint_manager {
            let mut checkpoint = manager.load_checkpoint_from_path(checkpoint_path)?;

            // Keep the old params for comparison
            let old_iterations = checkpoint.params.iterations;

            // Update with new parameters
            checkpoint.params = new_params.clone();

            self.restore_from_checkpoint(checkpoint)?;

            if self.verbose {
                println!(
                    "Resumed from {} with modified parameters. Iterations changed from {} to {}",
                    checkpoint_path.display(),
                    old_iterations,
                    new_params.iterations
                );
            }
        }
        Ok(())
    }

    /// Restore state from a checkpoint
    #[cfg(feature = "checkpointing")]
    fn restore_from_checkpoint(&mut self, checkpoint: OQNLPCheckpoint) -> Result<(), OQNLPError> {
        let solution_count = checkpoint.solution_set.as_ref().map_or(0, |s| s.len());

        self.params = checkpoint.params;
        self.current_iteration = checkpoint.current_iteration;
        self.merit_filter
            .update_threshold(checkpoint.merit_threshold);
        self.solution_set = checkpoint.solution_set;
        self.current_reference_set = Some(checkpoint.reference_set);
        self.unchanged_cycles = checkpoint.unchanged_cycles;

        // Restore the current seed for continuing RNG sequence
        self.current_seed = checkpoint.current_seed;

        // Restore target objective
        self.target_objective = checkpoint.target_objective;

        // Restore exclude_out_of_bounds setting
        self.exclude_out_of_bounds = checkpoint.exclude_out_of_bounds;

        // Restore distance filter solutions
        self.distance_filter
            .set_solutions(checkpoint.distance_filter_solutions);

        if self.verbose {
            println!(
                "Resumed from checkpoint at iteration {} with {} solutions",
                checkpoint.current_iteration, solution_count
            );
        }

        Ok(())
    }

    /// Create a checkpoint of the current state
    #[cfg(feature = "checkpointing")]
    fn create_checkpoint(&self) -> OQNLPCheckpoint {
        let elapsed_time = self
            .start_time
            .map(|start| start.elapsed().as_secs_f64())
            .unwrap_or(0.0);

        OQNLPCheckpoint {
            params: self.params.clone(),
            current_iteration: self.current_iteration,
            merit_threshold: self.merit_filter.threshold,
            solution_set: self.solution_set.clone(),
            reference_set: self.current_reference_set.clone().unwrap_or_default(),
            unchanged_cycles: self.unchanged_cycles,
            elapsed_time,
            distance_filter_solutions: self.distance_filter.get_solutions().clone(),
            current_seed: self.current_seed,
            target_objective: self.target_objective,
            exclude_out_of_bounds: self.exclude_out_of_bounds,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Save a checkpoint if checkpointing is enabled and conditions are met
    #[cfg(feature = "checkpointing")]
    fn maybe_save_checkpoint(&self) -> Result<(), OQNLPError> {
        if let Some(ref manager) = self.checkpoint_manager {
            if self.current_iteration % manager.config().save_frequency == 0 {
                let checkpoint = self.create_checkpoint();
                let saved_path = manager.save_checkpoint(&checkpoint, self.current_iteration)?;

                if self.verbose {
                    println!("Checkpoint saved to: {}", saved_path.display());
                }
            }
        }
        Ok(())
    }

    /// Save final checkpoint at the end of optimization (regardless of save_frequency)
    #[cfg(feature = "checkpointing")]
    fn maybe_save_final_checkpoint(&self) -> Result<(), OQNLPError> {
        if let Some(manager) = &self.checkpoint_manager {
            let checkpoint = self.create_checkpoint();
            let saved_path = manager.save_checkpoint(&checkpoint, self.current_iteration)?;

            if self.verbose {
                println!("Final checkpoint saved to: {}", saved_path.display());
            }
        }
        Ok(())
    }

    /// Adjust the threshold for the merit filter
    ///
    /// The threshold is adjusted using the equation:
    /// `threshold = threshold + threshold_factor * (1 + abs(threshold))`
    fn adjust_threshold(&mut self, current_threshold: f64) {
        let new_threshold: f64 =
            current_threshold + self.params.threshold_factor * (1.0 + current_threshold.abs());
        self.merit_filter.update_threshold(new_threshold);
    }
}

#[cfg(test)]
mod tests_oqnlp {
    use super::*;
    use crate::types::EvaluationError;
    use ndarray::{array, Array1, Array2};

    // Dummy problem for testing, sum of variables with bounds from -5 to 5
    #[derive(Clone)]
    struct DummyProblem;
    impl Problem for DummyProblem {
        fn objective(&self, trial: &Array1<f64>) -> Result<f64, EvaluationError> {
            Ok(trial.sum())
        }

        fn gradient(&self, trial: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
            Ok(Array1::ones(trial.len()))
        }

        fn variable_bounds(&self) -> Array2<f64> {
            array![[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]]
        }
    }

    #[derive(Clone)]
    struct SixHumpCamel;
    impl Problem for SixHumpCamel {
        fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
            Ok(
                (4.0 - 2.1 * x[0].powi(2) + x[0].powi(4) / 3.0) * x[0].powi(2)
                    + x[0] * x[1]
                    + (-4.0 + 4.0 * x[1].powi(2)) * x[1].powi(2),
            )
        }

        // Calculated analytically, reference didn't provide gradient
        fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
            Ok(array![
                (8.0 - 8.4 * x[0].powi(2) + 2.0 * x[0].powi(4)) * x[0] + x[1],
                x[0] + (-8.0 + 16.0 * x[1].powi(2)) * x[1]
            ])
        }

        fn variable_bounds(&self) -> Array2<f64> {
            array![[-3.0, 3.0], [-2.0, 2.0]]
        }
    }

    #[test]
    /// Test processing a new local solution
    fn test_process_local_solution_new() {
        let problem: DummyProblem = DummyProblem;
        let params: OQNLPParams = OQNLPParams::default();
        // Create a new OQNLP instance manually.
        let mut oqnlp: OQNLP<DummyProblem> = OQNLP::new(problem, params).unwrap();

        let trial = Array1::from(vec![1.0, 2.0, 3.0]);
        let ls: LocalSolution = LocalSolution {
            objective: trial.sum(),
            point: trial.clone(),
        };

        let added: bool = oqnlp.process_local_solution(ls.clone()).unwrap();
        // For the first solution, no duplicate exists so added should be true
        // and the solution set should contain one solution
        assert!(added);

        let sol_set: SolutionSet = oqnlp.solution_set.unwrap();
        assert_eq!(sol_set.len(), 1);
        assert!((sol_set[0].objective - ls.objective).abs() < 1e-6);
    }

    #[test]
    /// Test that processing a duplicate solution does not add it
    fn test_process_local_solution_duplicate() {
        let problem: DummyProblem = DummyProblem;
        let params: OQNLPParams = OQNLPParams::default();
        let mut oqnlp: OQNLP<DummyProblem> = OQNLP::new(problem, params).unwrap();

        let trial = Array1::from(vec![1.0, 2.0, 3.0]);
        let ls: LocalSolution = LocalSolution {
            objective: trial.sum(),
            point: trial.clone(),
        };

        // Process the solution for the first time
        oqnlp.process_local_solution(ls.clone()).unwrap();

        // Process the duplicate solution, it should not be added
        let added: bool = oqnlp.process_local_solution(ls.clone()).unwrap();
        let sol_set = oqnlp.solution_set.unwrap();
        assert_eq!(sol_set.len(), 1);
        assert!(!added);
    }

    #[test]
    /// Test that a better (lower objective) solution replaces the previous best.
    fn test_process_local_solution_better() {
        let problem: DummyProblem = DummyProblem;
        let params: OQNLPParams = OQNLPParams::default();
        let mut oqnlp: OQNLP<DummyProblem> = OQNLP::new(problem, params).unwrap();

        // First solution with objective 6.0
        let trial1 = Array1::from(vec![2.0, 2.0, 2.0]);
        let ls1: LocalSolution = LocalSolution {
            objective: trial1.sum(),
            point: trial1.clone(),
        };
        oqnlp.process_local_solution(ls1).unwrap();

        // Second solution with objective 3.0 (better)
        let trial2 = Array1::from(vec![1.0, 1.0, 1.0]);
        let ls2: LocalSolution = LocalSolution {
            objective: trial2.sum(),
            point: trial2.clone(),
        };
        let added: bool = oqnlp.process_local_solution(ls2.clone()).unwrap();

        // When a new best is found, the solution set is replaced
        let sol_set: SolutionSet = oqnlp.solution_set.unwrap();
        assert_eq!(sol_set.len(), 1);
        assert!((sol_set[0].objective - ls2.objective).abs() < 1e-6);

        // The best solution replacement does not mark the solution as "added"
        // since it changes the existing solution set
        assert!(!added);
    }

    #[test]
    /// Test the helper for deciding whether to start a local search or not
    fn test_should_start_local() {
        let problem: DummyProblem = DummyProblem;
        let params: OQNLPParams = OQNLPParams::default();

        // Create a new OQNLP instance manually
        let mut oqnlp: OQNLP<DummyProblem> = OQNLP::new(problem, params).unwrap();

        // Set the merit filter threshold to a specific value for testing
        oqnlp.merit_filter.update_threshold(10.0);

        // A trial far enough in space but with objective above the threshold
        let trial = Array1::from(vec![10.0, 10.0, 10.0]);
        let obj: f64 = trial.sum(); // 30.0 > 10.0 threshold
        let start: bool = oqnlp.should_start_local(&trial, obj).unwrap();
        assert!(!start);

        // A trial with objective below threshold and spatially acceptable
        let trial2 = Array1::from(vec![1.0, 1.0, 1.0]);
        let obj2: f64 = trial2.sum(); // 3.0 < 10.0 threshold
        let start2: bool = oqnlp.should_start_local(&trial2, obj2).unwrap();
        assert!(start2);
    }

    #[test]
    /// Test adjusting the merit filter threshold.
    fn test_adjust_threshold() {
        let problem: DummyProblem = DummyProblem;
        let params: OQNLPParams = OQNLPParams::default();

        // Create a new OQNLP instance manually
        let mut oqnlp: OQNLP<DummyProblem> = OQNLP::new(problem, params).unwrap();

        oqnlp.adjust_threshold(10.0);

        // New threshold = 10.0 + 0.2*(1 + 10.0) = 12.2
        assert!((oqnlp.merit_filter.threshold - 12.2).abs() < f64::EPSILON);
    }

    #[test]
    /// Test max_time method for setting the time limit
    fn test_max_time() {
        let problem: DummyProblem = DummyProblem;
        let params: OQNLPParams = OQNLPParams::default();

        // Create a new OQNLP instance manually
        let oqnlp: OQNLP<DummyProblem> = OQNLP::new(problem, params).unwrap();

        // Test setting the time limit
        let oqnlp: OQNLP<DummyProblem> = oqnlp.max_time(10.0);
        assert_eq!(oqnlp.max_time, Some(10.0));

        // Test using it in SixHumpCamel example
        let problem: SixHumpCamel = SixHumpCamel;
        let params: OQNLPParams = OQNLPParams {
            iterations: 100,
            population_size: 500,
            ..Default::default()
        };

        // With normal time limits, the algorithm should find both solutions
        let mut oqnlp: OQNLP<SixHumpCamel> = OQNLP::new(problem.clone(), params.clone())
            .unwrap()
            .verbose()
            .max_time(20.0);

        let sol_set: SolutionSet = oqnlp.run().unwrap();
        assert_eq!(sol_set.len(), 2);

        // With a very low time limit, the algorithm should stop before finding both solutions
        let mut oqnlp: OQNLP<SixHumpCamel> = OQNLP::new(problem, params)
            .unwrap()
            .verbose()
            .max_time(0.000001);

        let sol_set: SolutionSet = oqnlp.run().unwrap();
        assert_eq!(sol_set.len(), 1);
    }

    #[test]
    /// Test the invalid population size for the OQNLP algorithm
    fn test_oqnlp_params_invalid_population_size() {
        let problem: DummyProblem = DummyProblem {};
        let params: OQNLPParams = OQNLPParams {
            population_size: 1, // Population size must be greater or equal to 3
            ..OQNLPParams::default()
        };

        let oqnlp = OQNLP::new(problem, params);
        assert!(matches!(oqnlp, Err(OQNLPError::InvalidPopulationSize(1))));
    }

    #[test]
    #[cfg(feature = "progress_bar")]
    /// Test the progress bar functionality
    fn test_progress_bar() {
        use kdam::term;
        use std::io::{stderr, IsTerminal};

        // Initialize terminal
        term::init(stderr().is_terminal());

        let problem = SixHumpCamel;
        let params = OQNLPParams {
            iterations: 5,       // Small number for testing
            population_size: 10, // Must be >= iterations
            ..Default::default()
        };

        let mut oqnlp = OQNLP::new(problem, params).unwrap().verbose();

        // Run OQNLP with progress bar
        let result = oqnlp.run();

        assert!(
            result.is_ok(),
            "OQNLP should run successfully with progress bar"
        );

        // Verify that a solution was found
        let sol_set = result.unwrap();
        assert!(sol_set.len() > 0, "Should find at least one solution");
    }

    #[test]
    /// Test the target objective functionality
    fn test_target_objective() {
        let problem = SixHumpCamel;
        let params = OQNLPParams {
            iterations: 50,
            population_size: 100,
            ..Default::default()
        };

        // Test with a target that should be reached (SixHumpCamel has global minimum around -1.03)
        let mut oqnlp = OQNLP::new(problem.clone(), params.clone())
            .unwrap()
            .target_objective(-0.5); // Set target higher than global minimum

        let result = oqnlp.run();
        assert!(result.is_ok(), "OQNLP should run successfully");

        let sol_set = result.unwrap();
        let best = sol_set.best_solution().unwrap();

        // The algorithm should have stopped when it found a solution <= -0.5
        assert!(
            best.objective <= -0.5,
            "Best objective {} should be <= target -0.5",
            best.objective
        );

        // Test with a target that will never be reached
        let mut oqnlp2 = OQNLP::new(problem, params).unwrap().target_objective(-10.0); // Set target much lower than possible

        let result2 = oqnlp2.run();
        assert!(
            result2.is_ok(),
            "OQNLP should run successfully even if target not reached"
        );

        let sol_set2 = result2.unwrap();
        let best2 = sol_set2.best_solution().unwrap();

        // The algorithm should have run all iterations without reaching the target
        assert!(
            best2.objective > -10.0,
            "Best objective {} should be > impossible target -10.0",
            best2.objective
        );
    }

    #[test]
    #[cfg(feature = "checkpointing")]
    /// Test the resume_with_modified_params functionality
    fn test_resume_with_modified_params() {
        use crate::types::CheckpointConfig;
        use std::env;

        let checkpoint_dir = env::temp_dir().join("globalsearch_test_resume");
        std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create test directory");

        let problem = SixHumpCamel;
        let initial_params = OQNLPParams {
            iterations: 10,
            population_size: 20,
            ..Default::default()
        };

        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: checkpoint_dir.clone(),
            checkpoint_name: "test_resume".to_string(),
            save_frequency: 2,
            keep_all: false,
            auto_resume: true,
        };

        // Create and run initial OQNLP with checkpointing to create a checkpoint
        let mut oqnlp = OQNLP::new(problem.clone(), initial_params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config.clone())
            .unwrap()
            .verbose();

        // Run for a few iterations to create checkpoints
        let _result = oqnlp.run();

        // Create a new OQNLP instance with modified parameters
        let modified_params = OQNLPParams {
            iterations: 25,      // Increased iterations
            population_size: 30, // Increased population size
            ..initial_params
        };

        let mut oqnlp2 = OQNLP::new(problem.clone(), modified_params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config)
            .unwrap()
            .verbose();

        // Test resume with modified parameters
        let resumed = oqnlp2.resume_with_modified_params(modified_params.clone());

        assert!(
            resumed.is_ok(),
            "Resume with modified params should succeed"
        );
        assert!(resumed.unwrap(), "Should have resumed from checkpoint");

        // Verify that the parameters were updated
        assert_eq!(oqnlp2.params.iterations, 25);
        assert_eq!(oqnlp2.params.population_size, 30);

        // Test case where no checkpoint exists
        let empty_checkpoint_dir = env::temp_dir().join("globalsearch_test_empty");
        std::fs::create_dir_all(&empty_checkpoint_dir).expect("Failed to create test directory");

        let empty_checkpoint_config = CheckpointConfig {
            checkpoint_dir: empty_checkpoint_dir.clone(),
            checkpoint_name: "nonexistent".to_string(),
            save_frequency: 2,
            keep_all: false,
            auto_resume: true,
        };

        let mut oqnlp3 = OQNLP::new(problem, modified_params.clone())
            .unwrap()
            .with_checkpointing(empty_checkpoint_config)
            .unwrap();

        let not_resumed = oqnlp3.resume_with_modified_params(modified_params);
        assert!(
            not_resumed.is_ok(),
            "Should handle no checkpoint gracefully"
        );
        assert!(
            !not_resumed.unwrap(),
            "Should return false when no checkpoint exists"
        );

        // Clean up test directories
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
        let _ = std::fs::remove_dir_all(&empty_checkpoint_dir);
    }

    #[test]
    /// Test the exclude_out_of_bounds functionality
    fn test_exclude_out_of_bounds() {
        let problem = DummyProblem;
        let params = OQNLPParams {
            iterations: 5,
            population_size: 10,
            ..Default::default()
        };

        // Test with exclude_out_of_bounds enabled
        let mut oqnlp = OQNLP::new(problem.clone(), params.clone())
            .unwrap()
            .exclude_out_of_bounds()
            .verbose();

        // Verify the flag is set
        assert!(oqnlp.exclude_out_of_bounds);

        // Create a solution that is out of bounds
        // Problem bounds are [-5.0, 5.0] for each dimension
        let out_of_bounds_solution = LocalSolution {
            point: Array1::from(vec![10.0, 10.0, 10.0]), // All values exceed upper bound of 5.0
            objective: 30.0,
        };

        // Create a solution that is within bounds
        let within_bounds_solution = LocalSolution {
            point: Array1::from(vec![1.0, 2.0, 3.0]), // All values within [-5.0, 5.0]
            objective: 6.0,
        };

        // Test that out-of-bounds solution is rejected
        let added_out_of_bounds = oqnlp
            .process_local_solution(out_of_bounds_solution)
            .unwrap();
        assert!(!added_out_of_bounds);
        assert!(oqnlp.solution_set.is_none()); // No solution should be added

        // Test that within-bounds solution is accepted
        let added_within_bounds = oqnlp
            .process_local_solution(within_bounds_solution.clone())
            .unwrap();
        assert!(added_within_bounds);
        assert!(oqnlp.solution_set.is_some());
        let sol_set = oqnlp.solution_set.unwrap();
        assert_eq!(sol_set.len(), 1);
        assert!((sol_set[0].objective - within_bounds_solution.objective).abs() < 1e-6);

        // Test with exclude_out_of_bounds disabled (default behavior)
        let mut oqnlp2 = OQNLP::new(problem, params).unwrap();
        assert!(!oqnlp2.exclude_out_of_bounds); // Should be false by default

        // Out-of-bounds solution should be accepted when flag is disabled
        let out_of_bounds_solution2 = LocalSolution {
            point: Array1::from(vec![15.0, 20.0, 25.0]),
            objective: 60.0,
        };

        let added_out_of_bounds2 = oqnlp2
            .process_local_solution(out_of_bounds_solution2.clone())
            .unwrap();
        assert!(added_out_of_bounds2);
        assert!(oqnlp2.solution_set.is_some());
        let sol_set2 = oqnlp2.solution_set.unwrap();
        assert_eq!(sol_set2.len(), 1);
        assert!((sol_set2[0].objective - out_of_bounds_solution2.objective).abs() < 1e-6);
    }

    #[test]
    /// Test exclude_out_of_bounds with target_objective
    fn test_exclude_out_of_bounds_with_target_objective() {
        let problem = DummyProblem;
        let params = OQNLPParams {
            iterations: 5,
            population_size: 10,
            ..Default::default()
        };

        let mut oqnlp = OQNLP::new(problem, params)
            .unwrap()
            .exclude_out_of_bounds()
            .target_objective(50.0)
            .verbose();

        // Create an out-of-bounds solution that meets the target objective
        let out_of_bounds_good_obj = LocalSolution {
            point: Array1::from(vec![10.0, 10.0, 10.0]), // Out of bounds
            objective: 30.0,                             // Meets target objective (< 50.0)
        };

        // Create a within-bounds solution that meets the target objective
        let within_bounds_good_obj = LocalSolution {
            point: Array1::from(vec![1.0, 2.0, 3.0]), // Within bounds
            objective: 40.0,                          // Meets target objective (< 50.0)
        };

        // Create a within-bounds solution that doesn't meet the target objective
        let within_bounds_bad_obj = LocalSolution {
            point: Array1::from(vec![0.0, 0.0, 0.0]), // Within bounds
            objective: 60.0,                          // Doesn't meet target objective (> 50.0)
        };

        // Process out-of-bounds solution - should be rejected even if it meets target
        oqnlp
            .process_local_solution(out_of_bounds_good_obj)
            .unwrap();
        assert!(!oqnlp.target_objective_reached()); // Should not reach target due to bounds

        // Process within-bounds solution that meets target - should be accepted
        oqnlp
            .process_local_solution(within_bounds_good_obj)
            .unwrap();
        assert!(oqnlp.target_objective_reached()); // Should reach target

        // Reset for next test
        oqnlp.solution_set = None;

        // Process within-bounds solution that doesn't meet target - should be accepted but target not reached
        oqnlp.process_local_solution(within_bounds_bad_obj).unwrap();
        assert!(!oqnlp.target_objective_reached()); // Should not reach target due to objective
    }

    #[test]
    /// Test is_within_bounds helper method
    fn test_is_within_bounds() {
        let problem = DummyProblem; // Bounds are [-5.0, 5.0] for each dimension
        let params = OQNLPParams::default();
        let oqnlp = OQNLP::new(problem, params).unwrap();

        // Test point within bounds
        let within_bounds = Array1::from(vec![1.0, 2.0, 3.0]);
        assert!(oqnlp.is_within_bounds(&within_bounds));

        // Test point at lower bound
        let at_lower_bound = Array1::from(vec![-5.0, -5.0, -5.0]);
        assert!(oqnlp.is_within_bounds(&at_lower_bound));

        // Test point at upper bound
        let at_upper_bound = Array1::from(vec![5.0, 5.0, 5.0]);
        assert!(oqnlp.is_within_bounds(&at_upper_bound));

        // Test point below lower bound
        let below_lower_bound = Array1::from(vec![-6.0, 0.0, 0.0]);
        assert!(!oqnlp.is_within_bounds(&below_lower_bound));

        // Test point above upper bound
        let above_upper_bound = Array1::from(vec![0.0, 6.0, 0.0]);
        assert!(!oqnlp.is_within_bounds(&above_upper_bound));

        // Test empty point (edge case)
        let empty_point = Array1::from(vec![]);
        assert!(oqnlp.is_within_bounds(&empty_point)); // Empty point should be considered within bounds
    }

    #[test]
    #[cfg(feature = "checkpointing")]
    /// Test that exclude_out_of_bounds is properly saved and restored in checkpoints
    fn test_exclude_out_of_bounds_checkpointing() {
        use crate::types::CheckpointConfig;
        use std::env;

        let checkpoint_dir = env::temp_dir().join("globalsearch_test_exclude_bounds");
        std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create test directory");

        let problem = DummyProblem;
        let params = OQNLPParams {
            iterations: 5,
            population_size: 10,
            ..Default::default()
        };

        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: checkpoint_dir.clone(),
            checkpoint_name: "test_exclude_bounds".to_string(),
            save_frequency: 1,
            keep_all: false,
            auto_resume: false,
        };

        // Create OQNLP with exclude_out_of_bounds enabled
        let oqnlp = OQNLP::new(problem.clone(), params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config.clone())
            .unwrap()
            .exclude_out_of_bounds()
            .verbose();

        // Create a checkpoint
        let checkpoint = oqnlp.create_checkpoint();
        assert!(checkpoint.exclude_out_of_bounds);

        // Create new OQNLP instance and restore from checkpoint
        let mut oqnlp2 = OQNLP::new(problem, params)
            .unwrap()
            .with_checkpointing(checkpoint_config)
            .unwrap();

        // Before restoration, exclude_out_of_bounds should be false
        assert!(!oqnlp2.exclude_out_of_bounds);

        // Restore from checkpoint
        oqnlp2.restore_from_checkpoint(checkpoint).unwrap();

        // After restoration, exclude_out_of_bounds should be true
        assert!(oqnlp2.exclude_out_of_bounds);

        // Clean up
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }

    #[test]
    #[cfg(feature = "checkpointing")]
    /// Test resume_with_modified_params with decreased population size
    fn test_resume_with_modified_params_decreased_population() {
        use crate::types::CheckpointConfig;
        use std::env;

        let checkpoint_dir = env::temp_dir().join("globalsearch_test_decrease");
        std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create test directory");

        let problem = SixHumpCamel;
        let initial_params = OQNLPParams {
            iterations: 10,
            population_size: 30,
            ..Default::default()
        };

        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: checkpoint_dir.clone(),
            checkpoint_name: "test_decrease".to_string(),
            save_frequency: 2,
            keep_all: false,
            auto_resume: true,
        };

        // Create and run initial OQNLP with checkpointing
        let mut oqnlp = OQNLP::new(problem.clone(), initial_params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config.clone())
            .unwrap()
            .verbose();

        let _result = oqnlp.run();

        // Create modified parameters with decreased population size
        let modified_params = OQNLPParams {
            iterations: 15,
            population_size: 20, // Decreased from 30 to 20
            ..initial_params
        };

        let mut oqnlp2 = OQNLP::new(problem, modified_params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config)
            .unwrap()
            .verbose();

        // Test resume with decreased population size
        let resumed = oqnlp2.resume_with_modified_params(modified_params);

        assert!(
            resumed.is_ok(),
            "Resume with decreased population should succeed"
        );
        assert!(resumed.unwrap(), "Should have resumed from checkpoint");

        // The reference set should keep the original size despite the smaller population_size parameter
        // This is tested implicitly by ensuring the function succeeds without error

        // Clean up test directory
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }

    #[test]
    #[cfg(feature = "checkpointing")]
    /// Test the expand_reference_set method functionality
    fn test_expand_reference_set() {
        use crate::types::CheckpointConfig;
        use std::env;

        let checkpoint_dir = env::temp_dir().join("globalsearch_test_expand");
        std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create test directory");

        let problem = SixHumpCamel;
        let params = OQNLPParams {
            iterations: 5,
            population_size: 10,
            ..Default::default()
        };

        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: checkpoint_dir.clone(),
            checkpoint_name: "test_expand".to_string(),
            save_frequency: 1,
            keep_all: false,
            auto_resume: true,
        };

        // Create OQNLP instance
        let oqnlp = OQNLP::new(problem.clone(), params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config)
            .unwrap();

        // Create a small reference set to expand
        let mut ref_set = vec![
            Array1::from(vec![-1.0, -1.0]),
            Array1::from(vec![0.0, 0.0]),
            Array1::from(vec![1.0, 1.0]),
        ];
        let old_size = ref_set.len();
        let new_size = 8;

        // Test expanding the reference set
        let result = oqnlp.expand_reference_set(&mut ref_set, old_size, new_size);

        assert!(result.is_ok(), "expand_reference_set should succeed");
        assert_eq!(
            ref_set.len(),
            new_size,
            "Reference set should be expanded to new size"
        );

        // Verify that original points are still there
        assert!(ref_set.contains(&Array1::from(vec![-1.0, -1.0])));
        assert!(ref_set.contains(&Array1::from(vec![0.0, 0.0])));
        assert!(ref_set.contains(&Array1::from(vec![1.0, 1.0])));

        // Verify that new points are within bounds
        let bounds = problem.variable_bounds();
        for point in &ref_set {
            assert!(
                point.len() == bounds.nrows(),
                "Point should have correct dimensions"
            );
            for (i, &val) in point.iter().enumerate() {
                assert!(
                    val >= bounds[[i, 0]] && val <= bounds[[i, 1]],
                    "Point value {} should be within bounds [{}, {}]",
                    val,
                    bounds[[i, 0]],
                    bounds[[i, 1]]
                );
            }
        }

        // Clean up test directory
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }

    #[test]
    #[cfg(feature = "checkpointing")]
    /// Test expand_reference_set with edge cases
    fn test_expand_reference_set_edge_cases() {
        use crate::types::CheckpointConfig;
        use std::env;

        let checkpoint_dir = env::temp_dir().join("globalsearch_test_expand_edge");
        std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create test directory");

        let problem = SixHumpCamel;
        let params = OQNLPParams {
            iterations: 5,
            population_size: 10,
            ..Default::default()
        };

        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: checkpoint_dir.clone(),
            checkpoint_name: "test_expand_edge".to_string(),
            save_frequency: 1,
            keep_all: false,
            auto_resume: true,
        };

        let oqnlp = OQNLP::new(problem, params)
            .unwrap()
            .with_checkpointing(checkpoint_config)
            .unwrap();

        // Test case 1: new_size <= old_size (should return immediately)
        let mut ref_set1 = vec![
            Array1::from(vec![-1.0, -1.0]),
            Array1::from(vec![0.0, 0.0]),
            Array1::from(vec![1.0, 1.0]),
        ];
        let original_len = ref_set1.len();

        let result1 = oqnlp.expand_reference_set(&mut ref_set1, 5, 3);
        assert!(
            result1.is_ok(),
            "expand_reference_set should handle new_size <= old_size"
        );
        assert_eq!(
            ref_set1.len(),
            original_len,
            "Reference set should not change when new_size <= old_size"
        );

        // Test case 2: new_size == old_size (should return immediately)
        let mut ref_set2 = vec![Array1::from(vec![-1.0, -1.0]), Array1::from(vec![0.0, 0.0])];
        let original_len2 = ref_set2.len();

        let result2 = oqnlp.expand_reference_set(&mut ref_set2, original_len2, original_len2);
        assert!(
            result2.is_ok(),
            "expand_reference_set should handle new_size == old_size"
        );
        assert_eq!(
            ref_set2.len(),
            original_len2,
            "Reference set should not change when new_size == old_size"
        );

        // Test case 3: Empty reference set expansion
        let mut ref_set3: Vec<Array1<f64>> = vec![];
        let result3 = oqnlp.expand_reference_set(&mut ref_set3, 0, 5);
        assert!(
            result3.is_ok(),
            "expand_reference_set should handle empty reference set"
        );
        assert_eq!(
            ref_set3.len(),
            5,
            "Reference set should be expanded from empty to new size"
        );

        // Clean up test directory
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }

    #[test]
    #[cfg(feature = "checkpointing")]
    /// Test expand_reference_set integration through resume_with_modified_params
    fn test_expand_reference_set_integration() {
        use crate::types::CheckpointConfig;
        use std::env;

        let checkpoint_dir = env::temp_dir().join("globalsearch_test_expand_integration");
        std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create test directory");

        let problem = SixHumpCamel;
        let initial_params = OQNLPParams {
            iterations: 5,
            population_size: 10,
            ..Default::default()
        };

        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: checkpoint_dir.clone(),
            checkpoint_name: "test_expand_integration".to_string(),
            save_frequency: 1,
            keep_all: false,
            auto_resume: true,
        };

        // Create and run initial OQNLP to create a checkpoint
        let mut oqnlp = OQNLP::new(problem.clone(), initial_params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config.clone())
            .unwrap()
            .verbose();

        let _result = oqnlp.run();

        // Create modified parameters with significantly larger population size
        let modified_params = OQNLPParams {
            iterations: 8,
            population_size: 25, // Significantly increased from 10 to 25
            ..initial_params
        };

        let mut oqnlp2 = OQNLP::new(problem, modified_params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config)
            .unwrap()
            .verbose();

        // Test resume with modified parameters (this should trigger expand_reference_set internally)
        let resumed = oqnlp2.resume_with_modified_params(modified_params);

        assert!(
            resumed.is_ok(),
            "Resume with larger population should succeed"
        );
        assert!(resumed.unwrap(), "Should have resumed from checkpoint");

        // Verify that the parameters were updated
        assert_eq!(oqnlp2.params.population_size, 25);

        // The internal reference set should have been expanded
        // This is tested indirectly by ensuring the resume operation succeeds
        // and the algorithm can continue with the larger population size

        // Clean up test directory
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }

    #[test]
    #[cfg(feature = "checkpointing")]
    /// Test resume_from_checkpoint_with_params functionality
    fn test_resume_from_checkpoint_with_params() {
        use crate::types::CheckpointConfig;
        use std::env;

        let checkpoint_dir = env::temp_dir().join("globalsearch_test_resume_specific");
        std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create test directory");

        let problem = SixHumpCamel;
        let initial_params = OQNLPParams {
            iterations: 8,
            population_size: 15,
            ..Default::default()
        };

        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: checkpoint_dir.clone(),
            checkpoint_name: "test_resume_specific".to_string(),
            save_frequency: 2,
            keep_all: true,     // Keep all checkpoints for specific file testing
            auto_resume: false, // Don't auto-resume to test specific file loading
        };

        // Create and run initial OQNLP with checkpointing to create checkpoint files
        let mut oqnlp = OQNLP::new(problem.clone(), initial_params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config.clone())
            .unwrap()
            .verbose();

        let _result = oqnlp.run();

        // Find the checkpoint file that was created
        let checkpoint_files: Vec<_> = std::fs::read_dir(&checkpoint_dir)
            .expect("Failed to read checkpoint directory")
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                if path.extension()? == "bin"
                    && path.file_name()?.to_str()?.contains("test_resume_specific")
                {
                    Some(path)
                } else {
                    None
                }
            })
            .collect();

        assert!(
            !checkpoint_files.is_empty(),
            "Should have created at least one checkpoint file"
        );

        // Use the first checkpoint file found
        let checkpoint_path = &checkpoint_files[0];

        // Create modified parameters
        let modified_params = OQNLPParams {
            iterations: 20,      // Increased iterations
            population_size: 25, // Increased population size
            ..initial_params
        };

        // Create a new OQNLP instance
        let mut oqnlp2 = OQNLP::new(problem.clone(), modified_params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config)
            .unwrap()
            .verbose();

        // Test resume from specific checkpoint with modified parameters
        let result =
            oqnlp2.resume_from_checkpoint_with_params(checkpoint_path, modified_params.clone());

        assert!(
            result.is_ok(),
            "Resume from specific checkpoint should succeed"
        );

        // Verify that the parameters were updated
        assert_eq!(oqnlp2.params.iterations, 20);
        assert_eq!(oqnlp2.params.population_size, 25);

        // Verify that the algorithm can continue running with the loaded state
        let continued_result = oqnlp2.run();
        assert!(
            continued_result.is_ok(),
            "Should be able to continue optimization after resume"
        );

        // Clean up test directory
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }

    #[test]
    #[cfg(feature = "checkpointing")]
    /// Test resume_from_checkpoint_with_params with nonexistent file
    fn test_resume_from_checkpoint_with_params_nonexistent_file() {
        use crate::types::CheckpointConfig;
        use std::env;
        use std::path::PathBuf;

        let checkpoint_dir = env::temp_dir().join("globalsearch_test_resume_nonexistent");
        std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create test directory");

        let problem = SixHumpCamel;
        let params = OQNLPParams {
            iterations: 5,
            population_size: 10,
            ..Default::default()
        };

        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: checkpoint_dir.clone(),
            checkpoint_name: "test_nonexistent".to_string(),
            save_frequency: 1,
            keep_all: false,
            auto_resume: false,
        };

        let mut oqnlp = OQNLP::new(problem, params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config)
            .unwrap();

        // Try to resume from a nonexistent checkpoint file
        let nonexistent_path = PathBuf::from("nonexistent_checkpoint.bin");
        let result = oqnlp.resume_from_checkpoint_with_params(&nonexistent_path, params);

        // Should return an error for nonexistent file
        assert!(
            result.is_err(),
            "Should return error for nonexistent checkpoint file"
        );

        // Clean up test directory
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }

    #[test]
    #[cfg(feature = "checkpointing")]
    /// Test resume_from_checkpoint_with_params without checkpoint manager
    fn test_resume_from_checkpoint_with_params_no_manager() {
        use std::path::PathBuf;

        let problem = SixHumpCamel;
        let params = OQNLPParams {
            iterations: 5,
            population_size: 10,
            ..Default::default()
        };

        // Create OQNLP without checkpointing
        let mut oqnlp = OQNLP::new(problem, params.clone()).unwrap();

        // Try to resume from checkpoint without checkpoint manager
        let dummy_path = PathBuf::from("dummy_checkpoint.bin");
        let result = oqnlp.resume_from_checkpoint_with_params(&dummy_path, params);

        // Should succeed but do nothing when no checkpoint manager is present
        assert!(
            result.is_ok(),
            "Should handle absence of checkpoint manager gracefully"
        );
    }

    #[test]
    #[cfg(feature = "checkpointing")]
    /// Test resume_from_checkpoint_with_params with different parameter combinations
    fn test_resume_from_checkpoint_with_params_various_params() {
        use crate::types::CheckpointConfig;
        use std::env;

        let checkpoint_dir = env::temp_dir().join("globalsearch_test_resume_various");
        std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create test directory");

        let problem = SixHumpCamel;
        let initial_params = OQNLPParams {
            iterations: 6,
            population_size: 12,
            distance_factor: 0.1,
            threshold_factor: 0.3,
            ..Default::default()
        };

        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: checkpoint_dir.clone(),
            checkpoint_name: "test_resume_various".to_string(),
            save_frequency: 1,
            keep_all: true,
            auto_resume: false,
        };

        // Create and run initial OQNLP to create a checkpoint
        let mut oqnlp = OQNLP::new(problem.clone(), initial_params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config.clone())
            .unwrap()
            .verbose();

        let _result = oqnlp.run();

        // Find a checkpoint file
        let checkpoint_files: Vec<_> = std::fs::read_dir(&checkpoint_dir)
            .expect("Failed to read checkpoint directory")
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                if path.extension()? == "bin" {
                    Some(path)
                } else {
                    None
                }
            })
            .collect();

        assert!(
            !checkpoint_files.is_empty(),
            "Should have created checkpoint files"
        );
        let checkpoint_path = &checkpoint_files[0];

        // Test 1: Modify only iterations (keeping it within population_size constraint)
        let modified_params1 = OQNLPParams {
            iterations: 10, // Increase iterations but keep <= population_size (12)
            ..initial_params.clone()
        };

        let mut oqnlp1 = OQNLP::new(problem.clone(), modified_params1.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config.clone())
            .unwrap();

        let result1 = oqnlp1.resume_from_checkpoint_with_params(checkpoint_path, modified_params1);
        assert!(
            result1.is_ok(),
            "Should handle iterations-only modification"
        );
        assert_eq!(oqnlp1.params.iterations, 10);

        // Test 2: Modify multiple parameters
        let modified_params2 = OQNLPParams {
            iterations: 25,
            population_size: 30,
            distance_factor: 0.05,
            threshold_factor: 0.4,
            ..initial_params
        };

        let mut oqnlp2 = OQNLP::new(problem, modified_params2.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config)
            .unwrap();

        let result2 = oqnlp2.resume_from_checkpoint_with_params(checkpoint_path, modified_params2);
        assert!(
            result2.is_ok(),
            "Should handle multiple parameter modifications"
        );
        assert_eq!(oqnlp2.params.iterations, 25);
        assert_eq!(oqnlp2.params.population_size, 30);
        assert!((oqnlp2.params.distance_factor - 0.05).abs() < 1e-10);
        assert!((oqnlp2.params.threshold_factor - 0.4).abs() < 1e-10);

        // Clean up test directory
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }

    #[test]
    #[cfg(feature = "checkpointing")]
    /// Test restore_from_checkpoint functionality
    fn test_restore_from_checkpoint() {
        use crate::types::{CheckpointConfig, OQNLPCheckpoint};
        use chrono;
        use std::env;

        let checkpoint_dir = env::temp_dir().join("globalsearch_test_restore");
        std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create test directory");

        let problem = SixHumpCamel;
        let initial_params = OQNLPParams {
            iterations: 15,
            population_size: 20,
            distance_factor: 0.15,
            threshold_factor: 0.25,
            ..Default::default()
        };

        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: checkpoint_dir.clone(),
            checkpoint_name: "test_restore".to_string(),
            save_frequency: 1,
            keep_all: false,
            auto_resume: false,
        };

        // Create OQNLP instance
        let mut oqnlp = OQNLP::new(problem.clone(), initial_params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config)
            .unwrap()
            .verbose()
            .target_objective(-0.8);

        // Create some solutions for the checkpoint
        let solution1 = LocalSolution {
            objective: -0.5,
            point: Array1::from(vec![0.1, -0.7]),
        };
        let solution2 = LocalSolution {
            objective: -0.3,
            point: Array1::from(vec![-0.9, 0.2]),
        };

        let solution_set = SolutionSet {
            solutions: Array1::from(vec![solution1.clone(), solution2.clone()]),
        };

        // Create a mock checkpoint with comprehensive state
        let checkpoint = OQNLPCheckpoint {
            params: OQNLPParams {
                iterations: 25,
                population_size: 30,
                distance_factor: 0.08,
                threshold_factor: 0.35,
                ..initial_params.clone()
            },
            current_iteration: 12,
            merit_threshold: -0.2,
            solution_set: Some(solution_set.clone()),
            reference_set: vec![
                Array1::from(vec![-1.5, 1.0]),
                Array1::from(vec![0.5, -1.2]),
                Array1::from(vec![2.0, 0.8]),
            ],
            unchanged_cycles: 3,
            elapsed_time: 45.67,
            distance_filter_solutions: vec![solution1.clone(), solution2.clone()],
            current_seed: 98765,
            target_objective: Some(-0.9),
            exclude_out_of_bounds: true,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        // Test restore_from_checkpoint
        let result = oqnlp.restore_from_checkpoint(checkpoint.clone());

        assert!(result.is_ok(), "restore_from_checkpoint should succeed");

        // Verify all state was restored correctly
        assert_eq!(oqnlp.params.iterations, 25, "Iterations should be restored");
        assert_eq!(
            oqnlp.params.population_size, 30,
            "Population size should be restored"
        );
        assert!(
            (oqnlp.params.distance_factor - 0.08).abs() < 1e-10,
            "Distance factor should be restored"
        );
        assert!(
            (oqnlp.params.threshold_factor - 0.35).abs() < 1e-10,
            "Threshold factor should be restored"
        );

        assert_eq!(
            oqnlp.current_iteration, 12,
            "Current iteration should be restored"
        );
        assert!(
            (oqnlp.merit_filter.threshold - (-0.2)).abs() < 1e-10,
            "Merit threshold should be restored"
        );
        assert_eq!(
            oqnlp.unchanged_cycles, 3,
            "Unchanged cycles should be restored"
        );
        assert_eq!(oqnlp.current_seed, 98765, "Current seed should be restored");
        assert_eq!(
            oqnlp.target_objective,
            Some(-0.9),
            "Target objective should be restored"
        );

        // Verify solution set was restored
        let restored_solution_set = oqnlp
            .solution_set
            .as_ref()
            .expect("Solution set should be restored");
        assert_eq!(
            restored_solution_set.len(),
            2,
            "Solution set should have 2 solutions"
        );
        assert!(
            (restored_solution_set[0].objective - (-0.5)).abs() < 1e-10,
            "First solution objective should match"
        );
        assert!(
            (restored_solution_set[1].objective - (-0.3)).abs() < 1e-10,
            "Second solution objective should match"
        );

        // Verify reference set was restored
        let restored_ref_set = oqnlp
            .current_reference_set
            .as_ref()
            .expect("Reference set should be restored");
        assert_eq!(
            restored_ref_set.len(),
            3,
            "Reference set should have 3 points"
        );
        assert_eq!(
            restored_ref_set[0],
            Array1::from(vec![-1.5, 1.0]),
            "First reference point should match"
        );
        assert_eq!(
            restored_ref_set[1],
            Array1::from(vec![0.5, -1.2]),
            "Second reference point should match"
        );
        assert_eq!(
            restored_ref_set[2],
            Array1::from(vec![2.0, 0.8]),
            "Third reference point should match"
        );

        // Verify distance filter solutions were restored
        let distance_filter_solutions = oqnlp.distance_filter.get_solutions();
        assert_eq!(
            distance_filter_solutions.len(),
            2,
            "Distance filter should have 2 solutions"
        );

        // Clean up test directory
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }

    #[test]
    #[cfg(feature = "checkpointing")]
    /// Test restore_from_checkpoint with empty solution set
    fn test_restore_from_checkpoint_empty_solution_set() {
        use crate::types::{CheckpointConfig, OQNLPCheckpoint};
        use chrono;
        use std::env;

        let checkpoint_dir = env::temp_dir().join("globalsearch_test_restore_empty");
        std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create test directory");

        let problem = SixHumpCamel;
        let params = OQNLPParams {
            iterations: 10,
            population_size: 15,
            ..Default::default()
        };

        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: checkpoint_dir.clone(),
            checkpoint_name: "test_restore_empty".to_string(),
            save_frequency: 1,
            keep_all: false,
            auto_resume: false,
        };

        let mut oqnlp = OQNLP::new(problem, params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config)
            .unwrap()
            .verbose();

        // Create checkpoint with no solution set
        let checkpoint = OQNLPCheckpoint {
            params: params.clone(),
            current_iteration: 5,
            merit_threshold: 100.0,
            solution_set: None, // No solutions
            reference_set: vec![Array1::from(vec![0.0, 0.0])],
            unchanged_cycles: 0,
            elapsed_time: 10.0,
            distance_filter_solutions: vec![],
            current_seed: 12345,
            target_objective: None,
            exclude_out_of_bounds: false,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        let result = oqnlp.restore_from_checkpoint(checkpoint);

        assert!(
            result.is_ok(),
            "Should handle empty solution set gracefully"
        );
        assert!(
            oqnlp.solution_set.is_none(),
            "Solution set should remain None"
        );
        assert_eq!(
            oqnlp.current_iteration, 5,
            "Current iteration should be restored"
        );
        assert!(
            (oqnlp.merit_filter.threshold - 100.0).abs() < 1e-10,
            "Merit threshold should be restored"
        );

        // Clean up test directory
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }

    #[test]
    #[cfg(feature = "checkpointing")]
    /// Test restore_from_checkpoint with verbose output
    fn test_restore_from_checkpoint_verbose_output() {
        use crate::types::{CheckpointConfig, OQNLPCheckpoint};
        use chrono;
        use std::env;

        let checkpoint_dir = env::temp_dir().join("globalsearch_test_restore_verbose");
        std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create test directory");

        let problem = SixHumpCamel;
        let params = OQNLPParams {
            iterations: 8,
            population_size: 12,
            ..Default::default()
        };

        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: checkpoint_dir.clone(),
            checkpoint_name: "test_restore_verbose".to_string(),
            save_frequency: 1,
            keep_all: false,
            auto_resume: false,
        };

        // Test with verbose enabled
        let mut oqnlp_verbose = OQNLP::new(problem.clone(), params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config.clone())
            .unwrap()
            .verbose(); // Enable verbose output

        let solution = LocalSolution {
            objective: -1.2,
            point: Array1::from(vec![0.5, -0.3]),
        };

        let solution_set = SolutionSet {
            solutions: Array1::from(vec![solution]),
        };

        let checkpoint_with_solutions = OQNLPCheckpoint {
            params: params.clone(),
            current_iteration: 7,
            merit_threshold: -1.0,
            solution_set: Some(solution_set),
            reference_set: vec![Array1::from(vec![1.0, 1.0])],
            unchanged_cycles: 2,
            elapsed_time: 25.0,
            distance_filter_solutions: vec![],
            current_seed: 54321,
            target_objective: None,
            exclude_out_of_bounds: false,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        // This should print verbose output (iteration 7 with 1 solutions)
        let result_verbose = oqnlp_verbose.restore_from_checkpoint(checkpoint_with_solutions);
        assert!(result_verbose.is_ok(), "Verbose restore should succeed");

        // Test with verbose disabled
        let mut oqnlp_quiet = OQNLP::new(problem, params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config)
            .unwrap();
        // No .verbose() call - should be quiet

        let checkpoint_no_solutions = OQNLPCheckpoint {
            params,
            current_iteration: 3,
            merit_threshold: 50.0,
            solution_set: None,
            reference_set: vec![Array1::from(vec![-1.0, -1.0])],
            unchanged_cycles: 1,
            elapsed_time: 15.0,
            distance_filter_solutions: vec![],
            current_seed: 11111,
            target_objective: None,
            exclude_out_of_bounds: false,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        // This should not print any output (verbose is false)
        let result_quiet = oqnlp_quiet.restore_from_checkpoint(checkpoint_no_solutions);
        assert!(result_quiet.is_ok(), "Quiet restore should succeed");

        // Clean up test directory
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }

    #[test]
    #[cfg(feature = "checkpointing")]
    /// Test restore_from_checkpoint with various edge cases
    fn test_restore_from_checkpoint_edge_cases() {
        use crate::types::{CheckpointConfig, OQNLPCheckpoint};
        use chrono;
        use std::env;

        let checkpoint_dir = env::temp_dir().join("globalsearch_test_restore_edge");
        std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create test directory");

        let problem = SixHumpCamel;
        let params = OQNLPParams {
            iterations: 5,
            population_size: 8,
            ..Default::default()
        };

        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: checkpoint_dir.clone(),
            checkpoint_name: "test_restore_edge".to_string(),
            save_frequency: 1,
            keep_all: false,
            auto_resume: false,
        };

        let mut oqnlp = OQNLP::new(problem, params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config)
            .unwrap();

        // Test case 1: Checkpoint at iteration 0
        let checkpoint_zero = OQNLPCheckpoint {
            params: params.clone(),
            current_iteration: 0,
            merit_threshold: 1000.0,
            solution_set: None,
            reference_set: vec![],
            unchanged_cycles: 0,
            elapsed_time: 0.0,
            distance_filter_solutions: vec![],
            current_seed: 1,
            target_objective: None,
            exclude_out_of_bounds: false,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        let result1 = oqnlp.restore_from_checkpoint(checkpoint_zero);
        assert!(result1.is_ok(), "Should handle iteration 0 checkpoint");
        assert_eq!(oqnlp.current_iteration, 0);
        assert_eq!(oqnlp.unchanged_cycles, 0);

        // Test case 2: Checkpoint with very large merit threshold
        let checkpoint_large_threshold = OQNLPCheckpoint {
            params: params.clone(),
            current_iteration: 2,
            merit_threshold: f64::MAX / 2.0,
            solution_set: None,
            reference_set: vec![Array1::from(vec![0.0, 0.0])],
            unchanged_cycles: 1,
            elapsed_time: 5.0,
            distance_filter_solutions: vec![],
            current_seed: 999,
            target_objective: Some(f64::MIN / 2.0),
            exclude_out_of_bounds: true,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        let result2 = oqnlp.restore_from_checkpoint(checkpoint_large_threshold);
        assert!(result2.is_ok(), "Should handle large threshold values");
        assert!((oqnlp.merit_filter.threshold - (f64::MAX / 2.0)).abs() < f64::MAX / 4.0);
        assert_eq!(oqnlp.target_objective, Some(f64::MIN / 2.0));

        // Test case 3: Checkpoint with many unchanged cycles
        let checkpoint_many_cycles = OQNLPCheckpoint {
            params,
            current_iteration: 4,
            merit_threshold: 0.0,
            solution_set: None,
            reference_set: vec![Array1::from(vec![1.0, -1.0])],
            unchanged_cycles: 1000,
            elapsed_time: 100.0,
            distance_filter_solutions: vec![],
            current_seed: 777,
            target_objective: Some(0.0),
            exclude_out_of_bounds: false,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        let result3 = oqnlp.restore_from_checkpoint(checkpoint_many_cycles);
        assert!(result3.is_ok(), "Should handle many unchanged cycles");
        assert_eq!(oqnlp.unchanged_cycles, 1000);
        assert_eq!(oqnlp.current_seed, 777);

        // Clean up test directory
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }

    #[test]
    #[cfg(feature = "checkpointing")]
    /// Test create_checkpoint functionality
    fn test_create_checkpoint() {
        use crate::types::CheckpointConfig;
        use std::env;

        let checkpoint_dir = env::temp_dir().join("globalsearch_test_create_checkpoint");
        std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create test directory");

        let problem = SixHumpCamel;
        let params = OQNLPParams {
            iterations: 20,
            population_size: 25,
            distance_factor: 0.12,
            threshold_factor: 0.28,
            seed: 42,
            ..Default::default()
        };

        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: checkpoint_dir.clone(),
            checkpoint_name: "test_create".to_string(),
            save_frequency: 1,
            keep_all: false,
            auto_resume: false,
        };

        // Create OQNLP instance and set up some state
        let mut oqnlp = OQNLP::new(problem.clone(), params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config)
            .unwrap()
            .target_objective(-0.75);

        // Manually set up some internal state to test checkpoint creation
        oqnlp.current_iteration = 8;
        oqnlp.unchanged_cycles = 2;
        oqnlp.current_seed = 98765;
        oqnlp.merit_filter.update_threshold(-0.3);

        // Set up some solutions
        let solution1 = LocalSolution {
            objective: -0.6,
            point: Array1::from(vec![0.2, -0.8]),
        };
        let solution2 = LocalSolution {
            objective: -0.4,
            point: Array1::from(vec![-0.7, 0.3]),
        };

        let solution_set = SolutionSet {
            solutions: Array1::from(vec![solution1.clone(), solution2.clone()]),
        };
        oqnlp.solution_set = Some(solution_set);

        // Set up reference set
        oqnlp.current_reference_set = Some(vec![
            Array1::from(vec![-2.0, 1.5]),
            Array1::from(vec![1.0, -1.0]),
            Array1::from(vec![0.0, 0.0]),
        ]);

        // Add solutions to distance filter
        oqnlp.distance_filter.add_solution(solution1.clone());
        oqnlp.distance_filter.add_solution(solution2.clone());

        // Set start time to test elapsed time calculation
        oqnlp.start_time = Some(std::time::Instant::now() - std::time::Duration::from_secs(30));

        // Create checkpoint
        let checkpoint = oqnlp.create_checkpoint();

        // Verify all checkpoint fields are correct
        assert_eq!(
            checkpoint.params.iterations, 20,
            "Parameters should be captured"
        );
        assert_eq!(
            checkpoint.params.population_size, 25,
            "Parameters should be captured"
        );
        assert!(
            (checkpoint.params.distance_factor - 0.12).abs() < 1e-10,
            "Parameters should be captured"
        );
        assert!(
            (checkpoint.params.threshold_factor - 0.28).abs() < 1e-10,
            "Parameters should be captured"
        );
        assert_eq!(checkpoint.params.seed, 42, "Parameters should be captured");

        assert_eq!(
            checkpoint.current_iteration, 8,
            "Current iteration should be captured"
        );
        assert!(
            (checkpoint.merit_threshold - (-0.3)).abs() < 1e-10,
            "Merit threshold should be captured"
        );
        assert_eq!(
            checkpoint.unchanged_cycles, 2,
            "Unchanged cycles should be captured"
        );
        assert_eq!(
            checkpoint.current_seed, 98765,
            "Current seed should be captured"
        );
        assert_eq!(
            checkpoint.target_objective,
            Some(-0.75),
            "Target objective should be captured"
        );

        // Verify solution set
        let checkpoint_solutions = checkpoint
            .solution_set
            .as_ref()
            .expect("Solution set should be captured");
        assert_eq!(
            checkpoint_solutions.len(),
            2,
            "Solution set should have 2 solutions"
        );
        assert!(
            (checkpoint_solutions[0].objective - (-0.6)).abs() < 1e-10,
            "First solution should match"
        );
        assert!(
            (checkpoint_solutions[1].objective - (-0.4)).abs() < 1e-10,
            "Second solution should match"
        );

        // Verify reference set
        assert_eq!(
            checkpoint.reference_set.len(),
            3,
            "Reference set should have 3 points"
        );
        assert_eq!(
            checkpoint.reference_set[0],
            Array1::from(vec![-2.0, 1.5]),
            "Reference point should match"
        );
        assert_eq!(
            checkpoint.reference_set[1],
            Array1::from(vec![1.0, -1.0]),
            "Reference point should match"
        );
        assert_eq!(
            checkpoint.reference_set[2],
            Array1::from(vec![0.0, 0.0]),
            "Reference point should match"
        );

        // Verify distance filter solutions
        assert_eq!(
            checkpoint.distance_filter_solutions.len(),
            2,
            "Distance filter should have 2 solutions"
        );

        // Verify elapsed time (should be approximately 30 seconds, give or take a few milliseconds)
        assert!(
            checkpoint.elapsed_time >= 29.0 && checkpoint.elapsed_time <= 31.0,
            "Elapsed time should be approximately 30 seconds, got {}",
            checkpoint.elapsed_time
        );

        // Verify timestamp is valid RFC3339 format
        assert!(
            chrono::DateTime::parse_from_rfc3339(&checkpoint.timestamp).is_ok(),
            "Timestamp should be valid RFC3339 format"
        );

        // Clean up test directory
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }

    #[test]
    #[cfg(feature = "checkpointing")]
    /// Test create_checkpoint with empty state
    fn test_create_checkpoint_empty_state() {
        use crate::types::CheckpointConfig;
        use std::env;

        let checkpoint_dir = env::temp_dir().join("globalsearch_test_create_empty");
        std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create test directory");

        let problem = SixHumpCamel;
        let params = OQNLPParams {
            iterations: 5,
            population_size: 8,
            ..Default::default()
        };

        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: checkpoint_dir.clone(),
            checkpoint_name: "test_create_empty".to_string(),
            save_frequency: 1,
            keep_all: false,
            auto_resume: false,
        };

        // Create OQNLP instance with minimal state
        let oqnlp = OQNLP::new(problem, params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config)
            .unwrap();

        // Create checkpoint with default/empty state
        let checkpoint = oqnlp.create_checkpoint();

        // Verify checkpoint captures empty/default state correctly
        assert_eq!(
            checkpoint.params.iterations, 5,
            "Parameters should be captured"
        );
        assert_eq!(
            checkpoint.params.population_size, 8,
            "Parameters should be captured"
        );
        assert_eq!(
            checkpoint.current_iteration, 0,
            "Should start at iteration 0"
        );
        assert_eq!(
            checkpoint.unchanged_cycles, 0,
            "Should start with 0 unchanged cycles"
        );
        assert_eq!(
            checkpoint.current_seed, params.seed,
            "Should have initial seed"
        );
        assert_eq!(
            checkpoint.target_objective, None,
            "Should have no target objective"
        );

        // Verify empty collections
        assert!(
            checkpoint.solution_set.is_none(),
            "Solution set should be None initially"
        );
        assert!(
            checkpoint.reference_set.is_empty(),
            "Reference set should be empty initially"
        );
        assert!(
            checkpoint.distance_filter_solutions.is_empty(),
            "Distance filter should be empty initially"
        );

        // Verify elapsed time is 0 when no start time is set
        assert_eq!(
            checkpoint.elapsed_time, 0.0,
            "Elapsed time should be 0 when no start time"
        );

        // Verify timestamp is valid
        assert!(
            chrono::DateTime::parse_from_rfc3339(&checkpoint.timestamp).is_ok(),
            "Timestamp should be valid RFC3339 format"
        );

        // Clean up test directory
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }

    #[test]
    #[cfg(feature = "checkpointing")]
    /// Test create_checkpoint with various edge cases
    fn test_create_checkpoint_edge_cases() {
        use crate::types::CheckpointConfig;
        use std::env;

        let checkpoint_dir = env::temp_dir().join("globalsearch_test_create_edge");
        std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create test directory");

        let problem = SixHumpCamel;
        let params = OQNLPParams {
            iterations: 10,
            population_size: 15,
            ..Default::default()
        };

        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: checkpoint_dir.clone(),
            checkpoint_name: "test_create_edge".to_string(),
            save_frequency: 1,
            keep_all: false,
            auto_resume: false,
        };

        let mut oqnlp = OQNLP::new(problem, params)
            .unwrap()
            .with_checkpointing(checkpoint_config)
            .unwrap()
            .target_objective(f64::NEG_INFINITY); // Edge case: extreme target

        // Test case 1: Very large iteration number
        oqnlp.current_iteration = usize::MAX / 2;
        oqnlp.unchanged_cycles = 999999;
        oqnlp.current_seed = u64::MAX / 2;

        // Set extreme merit threshold
        oqnlp.merit_filter.update_threshold(f64::MAX / 2.0);

        // Set start time in the past (edge case for elapsed time)
        oqnlp.start_time = Some(
            std::time::Instant::now()
                .checked_sub(std::time::Duration::from_secs(60))
                .unwrap_or(std::time::Instant::now()),
        ); // 1 minute ago

        let checkpoint1 = oqnlp.create_checkpoint();

        assert_eq!(
            checkpoint1.current_iteration,
            usize::MAX / 2,
            "Should handle large iteration numbers"
        );
        assert_eq!(
            checkpoint1.unchanged_cycles, 999999,
            "Should handle large unchanged cycles"
        );
        assert_eq!(
            checkpoint1.current_seed,
            u64::MAX / 2,
            "Should handle large seed values"
        );
        assert!(
            (checkpoint1.merit_threshold - (f64::MAX / 2.0)).abs() < f64::MAX / 4.0,
            "Should handle large threshold"
        );
        assert_eq!(
            checkpoint1.target_objective,
            Some(f64::NEG_INFINITY),
            "Should handle extreme target objective"
        );
        assert!(
            checkpoint1.elapsed_time >= 55.0 && checkpoint1.elapsed_time <= 65.0,
            "Should handle elapsed time, got {}",
            checkpoint1.elapsed_time
        );

        // Test case 2: Reset to minimal values
        oqnlp.current_iteration = 0;
        oqnlp.unchanged_cycles = 0;
        oqnlp.current_seed = 1;
        oqnlp.merit_filter.update_threshold(0.0);
        oqnlp.target_objective = Some(0.0);
        oqnlp.start_time = Some(std::time::Instant::now()); // Just started

        let checkpoint2 = oqnlp.create_checkpoint();

        assert_eq!(
            checkpoint2.current_iteration, 0,
            "Should handle zero iteration"
        );
        assert_eq!(
            checkpoint2.unchanged_cycles, 0,
            "Should handle zero unchanged cycles"
        );
        assert_eq!(checkpoint2.current_seed, 1, "Should handle minimal seed");
        assert!(
            (checkpoint2.merit_threshold - 0.0).abs() < 1e-10,
            "Should handle zero threshold"
        );
        assert_eq!(
            checkpoint2.target_objective,
            Some(0.0),
            "Should handle zero target objective"
        );
        assert!(
            checkpoint2.elapsed_time >= 0.0 && checkpoint2.elapsed_time <= 1.0,
            "Should handle minimal elapsed time, got {}",
            checkpoint2.elapsed_time
        );

        // Verify timestamps are different (showing time progression)
        assert_ne!(
            checkpoint1.timestamp, checkpoint2.timestamp,
            "Timestamps should be different"
        );

        // Clean up test directory
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }

    #[test]
    #[cfg(feature = "checkpointing")]
    /// Test create_checkpoint multiple times to ensure consistency
    fn test_create_checkpoint_consistency() {
        use crate::types::CheckpointConfig;
        use std::env;

        let checkpoint_dir = env::temp_dir().join("globalsearch_test_create_consistency");
        std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create test directory");

        let problem = SixHumpCamel;
        let params = OQNLPParams {
            iterations: 12,
            population_size: 18,
            distance_factor: 0.09,
            ..Default::default()
        };

        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: checkpoint_dir.clone(),
            checkpoint_name: "test_create_consistency".to_string(),
            save_frequency: 1,
            keep_all: false,
            auto_resume: false,
        };

        let mut oqnlp = OQNLP::new(problem, params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config)
            .unwrap();

        // Set up consistent state
        oqnlp.current_iteration = 6;
        oqnlp.unchanged_cycles = 1;
        oqnlp.current_seed = 54321;
        oqnlp.merit_filter.update_threshold(-0.5);

        let solution = LocalSolution {
            objective: -0.7,
            point: Array1::from(vec![0.1, 0.2]),
        };
        let solution_set = SolutionSet {
            solutions: Array1::from(vec![solution.clone()]),
        };
        oqnlp.solution_set = Some(solution_set);
        oqnlp.distance_filter.add_solution(solution);

        // Create multiple checkpoints in quick succession
        let checkpoint1 = oqnlp.create_checkpoint();
        std::thread::sleep(std::time::Duration::from_millis(10)); // Small delay
        let checkpoint2 = oqnlp.create_checkpoint();

        // Most fields should be identical (state hasn't changed)
        assert_eq!(checkpoint1.params.iterations, checkpoint2.params.iterations);
        assert_eq!(
            checkpoint1.params.population_size,
            checkpoint2.params.population_size
        );
        assert_eq!(checkpoint1.current_iteration, checkpoint2.current_iteration);
        assert_eq!(checkpoint1.merit_threshold, checkpoint2.merit_threshold);
        assert_eq!(checkpoint1.unchanged_cycles, checkpoint2.unchanged_cycles);
        assert_eq!(checkpoint1.current_seed, checkpoint2.current_seed);
        assert_eq!(checkpoint1.target_objective, checkpoint2.target_objective);

        // Solution sets should be identical
        let sol1 = checkpoint1.solution_set.as_ref().unwrap();
        let sol2 = checkpoint2.solution_set.as_ref().unwrap();
        assert_eq!(sol1.len(), sol2.len());
        assert!((sol1[0].objective - sol2[0].objective).abs() < 1e-10);

        // Reference sets should be identical
        assert_eq!(checkpoint1.reference_set, checkpoint2.reference_set);

        // Distance filter solutions should be identical
        assert_eq!(
            checkpoint1.distance_filter_solutions.len(),
            checkpoint2.distance_filter_solutions.len()
        );

        // Timestamps should be different (time has passed)
        assert_ne!(
            checkpoint1.timestamp, checkpoint2.timestamp,
            "Timestamps should be different"
        );

        // Elapsed times should be very close (but not necessarily identical due to precision)
        let time_diff = (checkpoint1.elapsed_time - checkpoint2.elapsed_time).abs();
        assert!(
            time_diff < 1.0,
            "Elapsed times should be very close, difference: {}",
            time_diff
        );

        // Clean up test directory
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }
}
