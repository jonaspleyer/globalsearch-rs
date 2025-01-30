/// References:
///
/// Molga, M., & Smutnicki, C. Test functions for optimization needs (April 3, 2005), pp. 27-28. Retrieved January 2025, from https://robertmarks.org/Classes/ENGR5358/Papers/functions.pdf
use anyhow::Result;
use globalsearch_rs::problem::Problem;
use globalsearch_rs::{
    oqnlp::OQNLP,
    types::{LocalSolution, LocalSolverType, OQNLPParams},
};
use ndarray::{array, Array1, Array2};

#[derive(Debug, Clone)]
pub struct SixHumpCamel;

impl Problem for SixHumpCamel {
    fn objective(&self, x: &Array1<f64>) -> Result<f64> {
        Ok(
            (4.0 - 2.1 * x[0].powi(2) + x[0].powi(4) / 3.0) * x[0].powi(2)
                + x[0] * x[1]
                + (-4.0 + 4.0 * x[1].powi(2)) * x[1].powi(2),
        )
    }

    // Calculated analytically, reference didn't provide gradient
    fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>> {
        Ok(array![
            (8.0 - 8.4 * x[0].powi(2) + 2.0 * x[0].powi(4)) * x[0] + x[1],
            x[0] + (-8.0 + 16.0 * x[1].powi(2)) * x[1]
        ])
    }

    fn variable_bounds(&self) -> Array2<f64> {
        array![[-3.0, 3.0], [-2.0, 2.0]]
    }
}

fn main() -> Result<()> {
    let problem: SixHumpCamel = SixHumpCamel;
    let params: OQNLPParams = OQNLPParams {
        total_iterations: 1000,
        stage_1_iterations: 200,
        wait_cycle: 20,
        threshold_factor: 0.2,
        distance_factor: 0.75,
        population_size: 10,
        solver_type: LocalSolverType::SteepestDescent,
    };

    let mut oqnlp: OQNLP<SixHumpCamel> = OQNLP::new(problem, params)?;
    let solution: LocalSolution = oqnlp.run()?;

    println!("Best solution found:");
    println!("Point: {:?}", solution.point);
    println!("Objective: {}", solution.objective);

    Ok(())
}
