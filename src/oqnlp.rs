//! # OQNLP module
//!
//! The OQNLP (OptQuest/NLP) algorithm is a global optimization algorithm that combines scatter search with local optimization methods.

use crate::filters::{DistanceFilter, MeritFilter};
use crate::local_solver::runner::LocalSolver;
use crate::problem::Problem;
use crate::scatter_search::ScatterSearch;
use crate::types::{FilterParams, LocalSolution, OQNLPParams};
use ndarray::Array1;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use thiserror::Error;

// TODO: Where can we use rayon?
// use rayon::prelude::*;

// TODO: Set penalty functions?
// How should we do this? Two different OQNLP implementations?
// -> UnconstrainedOQNLP
// -> ConstrainedOQNLP

/// Error to be thrown if an issue occurs during the OQNLP optimization process
#[derive(Debug, Error)]
pub enum OQNLPError {
    /// Iterations should be less than or equal to population size
    #[error("OQNLP Error: Iterations should be less than or equal to population size. OQNLP received `iterations`: {0}, `population size`: {1}")]
    Iterations(usize, usize),

    /// Error when the local solver fails to find a solution
    #[error("OQNLP Error: Local solver failed to find a solution")]
    LocalSolverError,

    /// Error when OQNLP fails to find a feasible solution
    #[error("OQNLP Error: No feasible solution found")]
    NoFeasibleSolution,

    /// Error when the objective function evaluation fails
    #[error("OQNLP Error: Objective function evaluation failed")]
    ObjectiveFunctionEvaluationFailed,

    /// Error when creating a new ScatterSearch instance
    #[error("OQNLP Error: Failed to create a new ScatterSearch instance")]
    ScatterSearchError,

    /// Error when running the ScatterSearch instance
    #[error("OQNLP Error: Failed to run the ScatterSearch instance")]
    ScatterSearchRunError,
}

/// The main struct for the OQNLP algorithm.
///
/// This struct contains the optimization problem, algorithm parameters, filtering mechanisms,
/// and local solver, managing the optimization process.
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
    solution_set: Option<Array1<LocalSolution>>,

    /// Verbose flag to enable additional output during the optimization process.
    verbose: bool,
}

// TODO: Check implementation with the paper
impl<P: Problem + Clone + Send + Sync> OQNLP<P> {
    /// Create a new OQNLP instance with the given problem and parameters
    pub fn new(problem: P, params: OQNLPParams) -> Result<Self, OQNLPError> {
        let filter_params: FilterParams = FilterParams {
            distance_factor: params.distance_factor,
            wait_cycle: params.wait_cycle,
            threshold_factor: params.threshold_factor,
        };

        Ok(Self {
            problem: problem.clone(),
            params: params.clone(),
            merit_filter: MeritFilter::new(),
            distance_filter: DistanceFilter::new(filter_params),
            local_solver: LocalSolver::new(
                problem,
                params.local_solver_type.clone(),
                params.local_solver_config.clone(),
            ),
            solution_set: None,
            verbose: false,
        })
    }

    /// Run the OQNLP algorithm and return the solution set
    pub fn run(&mut self) -> Result<Array1<LocalSolution>, OQNLPError> {
        if self.params.wait_cycle >= self.params.iterations {
            eprintln!(
                "Warning: `wait_cycle` is greater than or equal to `iterations`. This may lead to suboptimal results."
            );
        }

        if self.params.iterations > self.params.population_size {
            return Err(OQNLPError::Iterations(
                self.params.iterations,
                self.params.population_size,
            ));
        }

        // Stage 1: Initial ScatterSearch iterations and first local call
        if self.verbose {
            println!("Starting Stage 1");
        }

        let ss: ScatterSearch<P> = ScatterSearch::new(self.problem.clone(), self.params.clone())
            .map_err(|_| OQNLPError::ScatterSearchError)?;
        let (mut ref_set, scatter_candidate) =
            ss.run().map_err(|_| OQNLPError::ScatterSearchRunError)?;
        let local_sol: LocalSolution = self
            .local_solver
            .solve(scatter_candidate)
            .map_err(|_| OQNLPError::LocalSolverError)?;

        self.merit_filter.update_threshold(local_sol.objective);

        if self.verbose {
            println!(
                "Stage 1: Local solution found with objective = {:.8}",
                local_sol.objective
            );

            println!("Starting Stage 2");
        }

        self.process_local_solution(local_sol)?;

        // Stage 2: Main iterative loop
        let mut unchanged_cycles: usize = 0;
        let mut rng: StdRng = StdRng::seed_from_u64(self.params.seed);
        // TODO: Do we need to shuffle the reference set? Is this necessary?
        ref_set.shuffle(&mut rng);

        for (iter, trial) in ref_set.iter().take(self.params.iterations).enumerate() {
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
                    .map_err(|_| OQNLPError::LocalSolverError)?;
                let added: bool = self.process_local_solution(local_trial.clone())?;

                if self.verbose && added {
                    println!(
                        "Stage 2, iteration {}: Added local solution found with objective = {:.8}",
                        iter, local_trial.objective
                    );
                    println!("x0 = {}", local_trial.point);
                }
            } else {
                unchanged_cycles += 1;

                if unchanged_cycles >= self.params.wait_cycle {
                    if self.verbose {
                        println!(
                            "Stage 2, iteration {}: Adjusting threshold from {:.8} to {:.8}",
                            iter,
                            self.merit_filter.threshold,
                            self.merit_filter.threshold + 0.1 * self.merit_filter.threshold.abs()
                        );
                    }

                    self.adjust_threshold(self.merit_filter.threshold);
                    unchanged_cycles = 0;
                }
            }
        }

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

    /// Process a local solution, updating the best solution and filters
    fn process_local_solution(&mut self, solution: LocalSolution) -> Result<bool, OQNLPError> {
        const EPS: f64 = 1e-6; // TODO: Should we let the user select this?

        let solutions = if let Some(existing) = &self.solution_set {
            existing
        } else {
            // First solution, initialize solution set
            self.solution_set = Some(Array1::from(vec![solution.clone()]));
            self.merit_filter.update_threshold(solution.objective);
            self.distance_filter.add_solution(solution);
            return Ok(true);
        };

        let current_best: &LocalSolution = &solutions[0];
        let obj_diff: f64 = solution.objective - current_best.objective;

        let added: bool = if obj_diff < -EPS {
            // Found new best solution
            self.solution_set = Some(Array1::from(vec![solution.clone()]));
            self.merit_filter.update_threshold(solution.objective);
            false
        } else if obj_diff.abs() <= EPS && !self.is_duplicate_in_set(&solution, solutions) {
            // Similar objective value and not duplicate, add to set
            let mut new_solutions = solutions.to_vec();
            new_solutions.push(solution.clone());
            self.solution_set = Some(Array1::from(new_solutions));
            true
        } else {
            false
        };

        self.distance_filter.add_solution(solution);
        Ok(added)
    }

    /// Check if a candidate solution is a duplicate in a set of solutions
    fn is_duplicate_in_set(&self, candidate: &LocalSolution, set: &Array1<LocalSolution>) -> bool {
        for s in set.iter() {
            let diff = &candidate.point - &s.point;
            let dist = diff.dot(&diff).sqrt();
            if dist < self.params.distance_factor {
                return true;
            }
        }
        false
    }

    /// Enable verbose output for the OQNLP algorithm
    ///
    /// This will print additional information about the optimization process
    pub fn verbose(mut self) -> Self {
        self.verbose = true;
        self
    }

    /// Adjust the threshold for the merit filter
    ///
    /// The threshold is adjusted using the formula:
    /// `threshold = threshold + threshold_factor * (1 + abs(threshold))`
    fn adjust_threshold(&mut self, current_threshold: f64) {
        let new_threshold: f64 =
            current_threshold + self.params.threshold_factor * (1.0 + current_threshold.abs());
        self.merit_filter.update_threshold(new_threshold);
    }
}
