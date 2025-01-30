use crate::filters::{DistanceFilter, MeritFilter};
use crate::local_solver::LocalSolver;
use crate::problem::Problem;
use crate::scatter_search::ScatterSearch;
use crate::types::{LocalSolution, OQNLPParams, Result};
use rayon::prelude::*;

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
    scatter_search: ScatterSearch<P>,
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
}

// TODO: Check implementation with the paper
// TODO: Add penalty weights to problem
impl<P: Problem + Clone + Send + Sync> OQNLP<P> {
    /// Create a new OQNLP instance with the given problem and parameters
    pub fn new(problem: P, params: OQNLPParams) -> Result<Self> {
        let filter_params = crate::types::FilterParams {
            distance_factor: params.distance_factor,
            wait_cycle: params.wait_cycle,
            threshold_factor: params.threshold_factor,
        };

        Ok(Self {
            problem: problem.clone(),
            params: params.clone(),
            scatter_search: ScatterSearch::new(problem.clone(), params.clone()),
            merit_filter: MeritFilter::new(filter_params.clone()),
            distance_filter: DistanceFilter::new(filter_params),
            local_solver: LocalSolver::new(problem, params.solver_type.clone()),
            best_solution: None,
        })
    }

    /// Run the OQNLP algorithm and return the best solution found
    pub fn run(&mut self) -> Result<LocalSolution> {
        // Stage 1: Generate initial points

        // TODO: This is wrong, this should take population size and run it for stage 1 iterations
        // Save best solutions in array and run local solver on all of them
        let stage_1_points = self
            .scatter_search
            .generate_trial_points(self.params.stage_1_iterations)?;

        let best_stage_1 = stage_1_points
            .into_par_iter()
            .map(|point| {
                let obj: f64 = self.problem.objective(&point)?;
                Ok((point, obj))
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(point, _)| point)
            .ok_or_else(|| anyhow::anyhow!("No stage 1 points"))?;

        // Run local solver on best stage 1 point
        let solution: LocalSolution = self.local_solver.solve(&best_stage_1)?;
        let objective: f64 = solution.objective.clone();
        self.best_solution = Some(solution.clone());
        self.distance_filter.add_solution(solution);
        self.merit_filter.update_threshold(objective);

        // Stage 2: Main loop
        for _ in 0..(self.params.total_iterations - self.params.stage_1_iterations) {
            let points = self.scatter_search.generate_trial_points(1)?;
            let point = points[0].clone(); // TODO: Remove this

            // TODO: If new solution has +-eps same objective as best solution, save it as well (two global optimum)
            if self.merit_filter.check(self.problem.objective(&point)?)
                && self.distance_filter.check(&point)
            {
                match self.local_solver.solve(&point) {
                    Ok(solution) => {
                        if self
                            .best_solution
                            .as_ref()
                            .map(|s| solution.objective < s.objective)
                            .unwrap_or(true)
                        {
                            self.best_solution = Some(solution.clone());
                        }
                        self.distance_filter.add_solution(solution);
                    }
                    Err(e) => anyhow::bail!("Local solver failed: {}", e),
                }
            }
        }

        self.best_solution
            .clone()
            .ok_or_else(|| anyhow::anyhow!("No solution found"))
    }
}
