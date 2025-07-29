//! # OQNLP module
//!
//! The OQNLP (OptQuest/NLP) algorithm is a global optimization algorithm that combines scatter search with local optimization methods.
//!
//! ## Example
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

// TODO: Set penalty functions?
// How should we do this? Two different OQNLP implementations?
// -> UnconstrainedOQNLP
// -> ConstrainedOQNLP

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
    #[error("OQNLP Error: Failed to create a new ScatterSearch instance.")]
    ScatterSearchError,

    /// Error when running the ScatterSearch instance
    #[error("OQNLP Error: Failed to run the ScatterSearch instance.")]
    ScatterSearchRunError,

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

/// # The main struct for the OQNLP algorithm.
///
/// This struct contains the optimization problem, algorithm parameters, filtering mechanisms,
/// and local solver, managing the optimization process.
///
/// The OQNLP algorithm is a global optimization algorithm that combines scatter search with local optimization methods.
///
/// The struct contains the following fields:
/// - `problem`: The optimization problem to be solved.
/// - `params`: Parameters controlling the behavior of the OQNLP algorithm.
/// - `merit_filter`: The merit filter used to maintain a threshold for the objective function.
/// - `distance_filter`: The distance filter used to maintain diversity among solutions.
/// - `local_solver`: The local solver responsible for refining solutions.
/// - `solution_set`: The set of best solutions found during the optimization process.
/// - `max_time`: Max time for the stage 2 of the OQNLP algorithm.
/// - `verbose`: Verbose flag to enable additional output during the optimization process.
///
/// It also contains methods to run the optimization process and adjust the threshold for the merit and distance filters.
///
/// The new method creates a new OQNLP instance with the given optimization problem and parameters,
/// while the run method executes the optimization process and returns the solution set.
pub struct OQNLP<P: Problem + Clone> {
    /// The optimization problem to be solved.
    ///
    /// This defines the objective function, gradient and constraints for the optimization process.
    problem: P,

    /// Parameters controlling the behavior of the OQNLP algorithm.
    ///
    /// These include total iterations, stage 1 iterations, waitcycles, and population settings.
    params: OQNLPParams,
    merit_filter: MeritFilter,

    /// The distance filter used to maintain diversity among solutions.
    ///
    /// It ensures that solutions are well-spaced.
    distance_filter: DistanceFilter,

    /// The local solver responsible for refining solutions.
    ///
    /// It applies a local optimization method, using argmin to find the best solution.
    local_solver: LocalSolver<P>,

    /// The set of best solutions found during the optimization process.
    ///
    /// If no solution has been found yet, it remains `None`.
    solution_set: Option<SolutionSet>,

    /// Max time for the stage 2 of the OQNLP algorithm.
    ///
    /// If the time limit is reached, the algorithm will stop and return the
    /// solution set found so far.
    ///
    /// The time limit is in seconds and it starts timing after the
    /// first local search.
    ///
    /// It is optional and can be set to `None` to disable the time limit.
    max_time: Option<f64>,

    /// Verbose flag to enable additional output during the optimization process.
    verbose: bool,

    /// Target objective function value to stop optimization early
    ///
    /// If set, the optimization will stop when a solution with an objective function value
    /// less than or equal to this value is found.
    /// This is useful for problems where a specific target is known and
    /// can be used to stop the optimization early.
    target_objective: Option<f64>,

    /// Checkpoint manager for saving and loading optimization state
    #[cfg(feature = "checkpointing")]
    checkpoint_manager: Option<CheckpointManager>,

    /// Current iteration number (for checkpointing)
    #[cfg(feature = "checkpointing")]
    current_iteration: usize,

    /// Current reference set (for checkpointing)
    #[cfg(feature = "checkpointing")]
    current_reference_set: Option<Vec<Array1<f64>>>,

    /// Number of unchanged cycles (for checkpointing)
    #[cfg(feature = "checkpointing")]
    unchanged_cycles: usize,

    /// Start time for elapsed time calculation
    #[cfg(feature = "checkpointing")]
    start_time: Option<std::time::Instant>,

    /// Current seed for RNG continuation (for checkpointing)
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
                ScatterSearch::new(self.problem.clone(), self.params.clone())
                    .map_err(|_| OQNLPError::ScatterSearchError)?;
            let (ref_set, scatter_candidate) =
                ss.run().map_err(|_| OQNLPError::ScatterSearchRunError)?;
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
                return best.objective <= target;
            }
        }
        false
    }

    /// Process a local solution, updating the best solution and filters
    fn process_local_solution(&mut self, solution: LocalSolution) -> Result<bool, OQNLPError> {
        const ABS_TOL: f64 = 1e-8;
        const REL_TOL: f64 = 1e-6;

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
        // Create an OQNLP instance manually.
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
        let oqnlp: OQNLP<DummyProblem> = OQNLP {
            problem: problem.clone(),
            params: params.clone(),
            merit_filter: {
                let mut mf: MeritFilter = MeritFilter::new();
                mf.update_threshold(10.0);
                mf
            },
            distance_filter: {
                // Add an existing solution to the distance filter
                let mut df: DistanceFilter = DistanceFilter::new(FilterParams {
                    distance_factor: params.distance_factor,
                    wait_cycle: params.wait_cycle,
                    threshold_factor: params.threshold_factor,
                })
                .expect("Failed to create DistanceFilter");
                let dummy_sol: LocalSolution = LocalSolution {
                    objective: 5.0,
                    point: Array1::from(vec![0.0, 0.0, 5.0]),
                };
                df.add_solution(dummy_sol);
                df
            },
            local_solver: LocalSolver::new(
                problem.clone(),
                params.local_solver_type.clone(),
                params.local_solver_config.clone(),
            ),
            solution_set: None,
            max_time: None,
            verbose: false,
            target_objective: None,
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
        };

        // A trial far enough in space but with objective above the threshold
        let trial = Array1::from(vec![10.0, 10.0, 10.0]);
        let obj: f64 = trial.sum(); // 20.0 > 10.0 threshold
        let start: bool = oqnlp.should_start_local(&trial, obj).unwrap();
        assert!(!start);

        // A trial with objective below threshold and spatially acceptable
        let trial2 = Array1::from(vec![1.0, 1.0, 5.0]);
        let obj2: f64 = trial2.sum(); // 2.0 < 10.0 threshold
        let start2: bool = oqnlp.should_start_local(&trial2, obj2).unwrap();
        assert!(start2);
    }

    #[test]
    /// Test adjusting the merit filter threshold.
    fn test_adjust_threshold() {
        let problem: DummyProblem = DummyProblem;
        let params: OQNLPParams = OQNLPParams::default();

        // Crate a new OQNLP instance manually
        let mut oqnlp: OQNLP<DummyProblem> = OQNLP {
            problem: problem.clone(),
            params: params.clone(),
            merit_filter: {
                let mut mf: MeritFilter = MeritFilter::new();
                mf.update_threshold(10.0);
                mf
            },
            distance_filter: DistanceFilter::new(FilterParams {
                distance_factor: params.distance_factor,
                wait_cycle: params.wait_cycle,
                threshold_factor: params.threshold_factor,
            })
            .unwrap(),
            local_solver: LocalSolver::new(
                problem.clone(),
                params.local_solver_type.clone(),
                params.local_solver_config.clone(),
            ),
            solution_set: None,
            max_time: None,
            verbose: false,
            target_objective: None,
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
        };

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
        let oqnlp: OQNLP<DummyProblem> = OQNLP {
            problem: problem.clone(),
            params: params.clone(),
            merit_filter: {
                let mut mf: MeritFilter = MeritFilter::new();
                mf.update_threshold(10.0);
                mf
            },
            distance_filter: DistanceFilter::new(FilterParams {
                distance_factor: params.distance_factor,
                wait_cycle: params.wait_cycle,
                threshold_factor: params.threshold_factor,
            })
            .unwrap(),
            local_solver: LocalSolver::new(
                problem.clone(),
                params.local_solver_type.clone(),
                params.local_solver_config.clone(),
            ),
            solution_set: None,
            max_time: None,
            verbose: false,
            target_objective: None,
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
        };

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
}
