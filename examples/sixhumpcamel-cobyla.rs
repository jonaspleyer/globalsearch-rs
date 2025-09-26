/// Six-Hump Camel Function with COBYLA Local Solver
/// The Six-Hump Camel function is defined as follows:
///
/// $f(x) = (4 - 2.1 x_1^2 + x_1^4 / 3) x_1^2 + x_1 x_2 + (-4 + 4 x_2^2) x_2^2$
///
/// The function is defined on the domain $x_1 \in [-3, 3]$ and $x_2 \in [-2, 2]$.
/// The function has two global minima at $f(0.0898, -0.7126) = -1.0316$ and $f(-0.0898, 0.7126) = -1.0316$.
/// The function is continuous, differentiable and non-convex.
///
/// This example shows how to use COBYLA as the local solver in the global search algorithm.
/// COBYLA is a constrained derivative-free optimization local solver.
///
/// References:
///
/// Molga, M., & Smutnicki, C. Test functions for optimization needs (April 3, 2005), pp. 27-28. Retrieved January 2025, from https://robertmarks.org/Classes/ENGR5358/Papers/functions.pdf
use globalsearch::local_solver::builders::COBYLABuilder;
use globalsearch::problem::Problem;
use globalsearch::{
    oqnlp::OQNLP,
    types::{EvaluationError, LocalSolverType, OQNLPParams, SolutionSet},
};
use ndarray::{array, Array1, Array2};

#[derive(Debug, Clone)]
pub struct SixHumpCamel;

impl Problem for SixHumpCamel {
    fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
        Ok(
            (4.0 - 2.1 * x[0].powi(2) + x[0].powi(4) / 3.0) * x[0].powi(2)
                + x[0] * x[1]
                + (-4.0 + 4.0 * x[1].powi(2)) * x[1].powi(2),
        )
    }

    fn variable_bounds(&self) -> Array2<f64> {
        array![[-3.0, 3.0], [-2.0, 2.0]]
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let problem: SixHumpCamel = SixHumpCamel;
    
    // Configure COBYLA with custom parameters
    let cobyla_config = COBYLABuilder::default()
        .max_iter(350)                     // Increase iterations for better convergence
        .initial_step_size(0.25)           // Smaller initial step size
        .ftol_rel(1e-10)                   // Relative function tolerance
        .ftol_abs(1e-12)                   // Absolute function tolerance
        .build();
    
    let params: OQNLPParams = OQNLPParams {
        iterations: 150,                             // Global search iterations
        wait_cycle: 8,                               // Wait cycles before expanding
        threshold_factor: 0.15,                      // Merit filter threshold
        distance_factor: 0.8,                        // Distance filter factor
        population_size: 300,                        // Scatter search population
        local_solver_type: LocalSolverType::COBYLA,  // Use COBYLA solver
        local_solver_config: cobyla_config,          // COBYLA configuration
        seed: 0,                                     // Random seed for reproducibility
    };

    let mut oqnlp: OQNLP<SixHumpCamel> = OQNLP::new(problem, params)?.verbose();
    let solution_set: SolutionSet = oqnlp.run()?;

    println!("\n{}", solution_set);
    Ok(())
}