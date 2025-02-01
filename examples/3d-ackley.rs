/// References:
///
/// Molga, M., & Smutnicki, C. Test functions for optimization needs (April 3, 2005), pp. 15-16. Retrieved January 2025, from https://robertmarks.org/Classes/ENGR5358/Papers/functions.pdf
use anyhow::Result;
use globalsearch_rs::problem::Problem;
use globalsearch_rs::{
    oqnlp::OQNLP,
    types::{LocalSolution, LocalSolverType, OQNLPParams, SteepestDescentBuilder},
};
use ndarray::{array, Array1, Array2};

#[derive(Debug, Clone)]
pub struct ThreeDAckley {
    a: f64,
    b: f64,
    c: f64,
}

impl ThreeDAckley {
    pub fn new(a: f64, b: f64, c: f64) -> Self {
        Self { a, b, c }
    }
}

impl Problem for ThreeDAckley {
    fn objective(&self, x: &Array1<f64>) -> Result<f64> {
        let norm = (x[0].powi(2) + x[1].powi(2) + x[2].powi(2)) / 3.0;
        let cos_sum = (x[0] * self.c).cos() + (x[1] * self.c).cos() + (x[2] * self.c).cos();

        Ok(
            -self.a * (-self.b * norm.sqrt()).exp() - (cos_sum / 3.0).exp()
                + self.a
                + std::f64::consts::E,
        )
    }

    // Calculated analytically, reference didn't provide gradient
    fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>> {
        let norm = (x[0].powi(2) + x[1].powi(2) + x[2].powi(2)) / 3.0;
        let sqrt_norm = norm.sqrt();
        let exp_term1 = (-self.b * sqrt_norm).exp();
        let exp_term2 =
            (((x[0] * self.c).cos() + (x[1] * self.c).cos() + (x[2] * self.c).cos()) / 3.0).exp();

        Ok(array![
            (self.a * self.b * x[0] / (3.0 * sqrt_norm)) * exp_term1
                + self.c / 3.0 * (x[0] * self.c).sin() * exp_term2,
            (self.a * self.b * x[1] / (3.0 * sqrt_norm)) * exp_term1
                + self.c / 3.0 * (x[1] * self.c).sin() * exp_term2,
            (self.a * self.b * x[2] / (3.0 * sqrt_norm)) * exp_term1
                + self.c / 3.0 * (x[2] * self.c).sin() * exp_term2,
        ])
    }

    fn variable_bounds(&self) -> Array2<f64> {
        array![[-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768]]
    }
}

fn main() -> Result<()> {
    let a: f64 = 20.0;
    let b: f64 = 0.2;
    let c: f64 = 2.0 * std::f64::consts::PI;

    let problem = ThreeDAckley::new(a, b, c);
    let params: OQNLPParams = OQNLPParams {
        total_iterations: 10000,
        stage_1_iterations: 500,
        wait_cycle: 20,
        threshold_factor: 0.2,
        distance_factor: 0.75,
        population_size: 100,
        local_solver_type: LocalSolverType::SteepestDescent,
        local_solver_config: SteepestDescentBuilder::default().build(),
    };

    let mut oqnlp: OQNLP<ThreeDAckley> = OQNLP::new(problem, params)?;
    let solution: LocalSolution = oqnlp.run()?;

    println!("Best solution found:");
    println!("Point: {:?}", solution.point);
    println!("Objective: {}", solution.objective);

    Ok(())
}
