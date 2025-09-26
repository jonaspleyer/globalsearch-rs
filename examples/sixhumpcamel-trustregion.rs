/// Six-Hump Camel Function
/// The Six-Hump Camel function is defined as follows:
///
/// $f(x) = (4 - 2.1 x_1^2 + x_1^4 / 3) x_1^2 + x_1 x_2 + (-4 + 4 x_2^2) x_2^2$
///
/// The function is defined on the domain $x_1 \in [-3, 3]$ and $x_2 \in [-2, 2]$.
/// The function has two global minima at $f(0.0898, -0.7126) = -1.0316$ and $f(-0.0898, 0.7126) = -1.0316$.
/// The function is continuous, differentiable and non-convex.
///
/// References:
///
/// Molga, M., & Smutnicki, C. Test functions for optimization needs (April 3, 2005), pp. 27-28. Retrieved January 2025, from https://robertmarks.org/Classes/ENGR5358/Papers/functions.pdf
use globalsearch::local_solver::builders::{TrustRegionBuilder, TrustRegionRadiusMethod};
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

    // Calculated analytically, reference didn't provide gradient
    fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
        Ok(array![
            (8.0 - 8.4 * x[0].powi(2) + 2.0 * x[0].powi(4)) * x[0] + x[1],
            x[0] + (-8.0 + 16.0 * x[1].powi(2)) * x[1]
        ])
    }

    // Calculated analytically, reference didn't provide hessian
    fn hessian(&self, x: &Array1<f64>) -> Result<Array2<f64>, EvaluationError> {
        Ok(array![
            [
                (4.0 * x[0].powi(2) - 4.2) * x[0].powi(2)
                    + 4.0 * (4.0 / 3.0 * x[0].powi(3) - 4.2 * x[0]) * x[0]
                    + 2.0 * (x[0].powi(4) / 3.0 - 2.1 * x[0].powi(2) + 4.0),
                1.0
            ],
            [1.0, 40.0 * x[1].powi(2) + 2.0 * (4.0 * x[1].powi(2) - 4.0)],
        ])
    }

    fn variable_bounds(&self) -> Array2<f64> {
        array![[-3.0, 3.0], [-2.0, 2.0]]
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let problem: SixHumpCamel = SixHumpCamel;
    let params: OQNLPParams = OQNLPParams {
        iterations: 125,
        wait_cycle: 10,
        threshold_factor: 0.2,
        distance_factor: 0.75,
        population_size: 250,
        local_solver_type: LocalSolverType::TrustRegion,
        local_solver_config: TrustRegionBuilder::default()
            .method(TrustRegionRadiusMethod::Steihaug)
            .build(),
        seed: 0,
    };

    let mut oqnlp: OQNLP<SixHumpCamel> = OQNLP::new(problem.clone(), params)?;
    let solution_set: SolutionSet = oqnlp.run()?;

    println!("Solution set found using Steihaug:");
    println!("{}", solution_set);

    let params: OQNLPParams = OQNLPParams {
        iterations: 125,
        wait_cycle: 10,
        threshold_factor: 0.2,
        distance_factor: 0.75,
        population_size: 250,
        local_solver_type: LocalSolverType::TrustRegion,
        local_solver_config: TrustRegionBuilder::default()
            .method(TrustRegionRadiusMethod::Cauchy)
            .build(),
        seed: 0,
    };

    let mut oqnlp: OQNLP<SixHumpCamel> = OQNLP::new(problem, params)?;
    let solution_set: SolutionSet = oqnlp.run()?;

    println!("Solution set found using Cauchy:");
    println!("{}", solution_set);

    Ok(())
}
