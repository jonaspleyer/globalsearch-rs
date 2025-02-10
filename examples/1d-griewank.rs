/// 1-D Griewank function
/// The 1-D Griewank function is defined as:
///
/// $ f(x) = 1 + \frac{x^2}{4000} - \cos(x) $
///
/// The function is defined on the domain `[-600, 600]`.
/// The function has a global minimum at `x = 0` with `f(x) = 0`.
/// The function is continuous, differentiable and non-convex.
///
/// References:
///
/// Molga, M., & Smutnicki, C. Test functions for optimization needs (April 3, 2005), pp. 11-12. Retrieved January 2025, from https://robertmarks.org/Classes/ENGR5358/Papers/functions.pdf
use globalsearch_rs::problem::Problem;
use globalsearch_rs::types::{HagerZhangBuilder, LBFGSBuilder};
use globalsearch_rs::{
    oqnlp::OQNLP,
    types::{EvaluationError, LocalSolution, LocalSolverType, OQNLPParams},
};
use ndarray::{array, Array1, Array2};

#[derive(Debug, Clone)]
pub struct OneDGriewank;

impl Problem for OneDGriewank {
    fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
        Ok(1.0 + x[0].powi(2) / 4000.0 - x[0].cos())
    }

    // Calculated analytically, reference didn't provide gradient
    fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
        Ok(array![x[0] / 2000.0 + x[0].sin()])
    }

    fn variable_bounds(&self) -> Array2<f64> {
        array![[-600.0, 600.0]]
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let problem: OneDGriewank = OneDGriewank;

    let params: OQNLPParams = OQNLPParams {
        iterations: 50,
        wait_cycle: 20,
        threshold_factor: 0.1,
        distance_factor: 0.75,
        population_size: 100,
        local_solver_type: LocalSolverType::LBFGS,
        local_solver_config: LBFGSBuilder::default().build(),
        seed: 0,
    };

    let mut oqnlp: OQNLP<OneDGriewank> = OQNLP::new(problem.clone(), params)?;
    let solution_set: Array1<LocalSolution> = oqnlp.run()?;

    println!("Best solution found:");
    for solution in solution_set.iter() {
        println!("Point: {}", solution.point);
        println!("Objective: {}", solution.objective);
    }

    let modified_params = OQNLPParams {
        iterations: 100,
        wait_cycle: 10,
        threshold_factor: 0.1,
        distance_factor: 0.75,
        population_size: 500,
        local_solver_type: LocalSolverType::LBFGS,
        local_solver_config: LBFGSBuilder::default()
            .max_iter(100)
            .history_size(15)
            .line_search_params(HagerZhangBuilder::default().build())
            .build(),
        seed: 0,
    };

    let mut modified_oqnlp: OQNLP<OneDGriewank> = OQNLP::new(problem, modified_params)?;
    let modified_solution_set: Array1<LocalSolution> = modified_oqnlp.run()?;

    println!("Best solution found:");
    for solution in modified_solution_set.iter() {
        println!("Point: {}", solution.point);
        println!("Objective: {}", solution.objective);
    }

    Ok(())
}
