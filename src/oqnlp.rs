//! # OQNLP module
//!
//! The OQNLP (OptQuest/NLP) algorithm is a global optimization algorithm that combines scatter search with local optimization methods.

use crate::filters::{DistanceFilter, MeritFilter};
use crate::local_solver::LocalSolver;
use crate::problem::Problem;
use crate::scatter_search::ScatterSearch;
use crate::types::{FilterParams, LocalSolution, OQNLPParams, Result};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

// TODO: Where can we use rayon?
// use rayon::prelude::*;

use ndarray::Array1;

// TODO: Save multiple global optimums; For an example of this check Cross in Tray problem:
//
// Stage 2, iter 97: Improved local solution found with objective = -2.0626118708227397
// x0 = [-1.3494066142838494, -1.3494066186040463], shape=[2], strides=[1], layout=CFcf (0xf), const ndim=1
// Stage 2, iter 97: Improved local solution found with objective = -2.0626118708227392
// x0 = [1.3494066219800123, -1.3494066169687304], shape=[2], strides=[1], layout=CFcf (0xf), const ndim=1

// TODO: Set penalty functions?
// How should we do this? Two different OQNLP implementations?
// -> UnconstrainedOQNLP
// -> ConstrainedOQNLP

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

    /// The best solution found during the optimization process.
    ///
    /// This is updated as better solutions are discovered. If no solution has been found yet, it remains `None`.
    best_solution: Option<LocalSolution>,
    verbose: bool,
}

// TODO: Check implementation with the paper
impl<P: Problem + Clone + Send + Sync> OQNLP<P> {
    /// Create a new OQNLP instance with the given problem and parameters
    pub fn new(problem: P, params: OQNLPParams) -> Result<Self> {
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
            best_solution: None,
            verbose: false,
        })
    }

    /// Run the OQNLP algorithm and return the best solution found
    pub fn run(&mut self) -> Result<LocalSolution> {
        if self.params.wait_cycle >= self.params.iterations {
            eprintln!(
                "Warning: wait_cycle is greater than or equal to iterations. This may lead to suboptimal results."
            );
        }

        if self.params.iterations > self.params.population_size {
            return Err(anyhow::anyhow!(
                "Error: Iterations should be less than or equal to population size."
            ));
        }

        // Stage 1: Initial ScatterSearch iterations and first local call
        if self.verbose {
            println!("Starting Stage 1");
        }

        let mut ss: ScatterSearch<P> =
            ScatterSearch::new(self.problem.clone(), self.params.clone())?;
        let (mut ref_set, scatter_candidate) = ss.run()?;
        let local_sol: LocalSolution = self.local_solver.solve(scatter_candidate)?;

        self.merit_filter.update_threshold(local_sol.objective);
        self.process_local_solution(local_sol.clone())?;

        if self.verbose {
            println!(
                "Stage 1: Local solution found with objective = {:.8}",
                local_sol.objective
            );

            println!("Starting Stage 2");
        }

        // Stage 2: Main iterative loop
        let mut unchanged_cycles: usize = 0;
        let mut rng: StdRng = StdRng::seed_from_u64(self.params.seed);
        // TODO: Do we need to shuffle the reference set? Is this necessary?
        ref_set.shuffle(&mut rng);

        for (iter, trial) in ref_set.into_iter().enumerate() {
            let trial = trial.clone();
            let obj: f64 = self.problem.objective(&trial)?;
            if self.should_start_local(&trial, obj)? {
                self.merit_filter.update_threshold(obj);
                let local_trial: LocalSolution = self.local_solver.solve(trial)?;
                self.process_local_solution(local_trial.clone())?;

                if self.verbose {
                    println!(
                        "Stage 2, iteration {}: Improved local solution found with objective = {:.8}",
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

        self.best_solution
            .clone()
            .ok_or_else(|| anyhow::anyhow!("No feasible solution found"))
    }

    // Helper methods
    /// Check if a local search should be started based on the merit and distance filters
    fn should_start_local(&self, point: &Array1<f64>, obj: f64) -> Result<bool> {
        let passes_merit: bool = obj <= self.merit_filter.threshold;
        let passes_distance: bool = self.distance_filter.check(point);
        Ok(passes_merit && passes_distance)
    }

    /// Process a local solution, updating the best solution and filters
    fn process_local_solution(&mut self, solution: LocalSolution) -> Result<()> {
        // Update best solution
        if self
            .best_solution
            .as_ref()
            .map_or(true, |best| solution.objective < best.objective)
        {
            self.best_solution = Some(solution.clone());
            self.merit_filter.update_threshold(solution.objective);
        }
        self.distance_filter.add_solution(solution);
        Ok(())
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
