use criterion::{criterion_group, criterion_main, Criterion};

/// Six-Hump Camel Back Function
/// The Six-Hump Camel Back function is defined as follows:
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
use anyhow::Result;
use globalsearch_rs::problem::Problem;
use globalsearch_rs::types::SteepestDescentBuilder;
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

fn six_hump_camel() -> Result<LocalSolution, anyhow::Error> {
    let problem: SixHumpCamel = SixHumpCamel;
    let params: OQNLPParams = OQNLPParams {
        total_iterations: 500,
        stage_1_iterations: 100,
        wait_cycle: 20,
        threshold_factor: 0.2,
        distance_factor: 0.75,
        population_size: 25,
        local_solver_type: LocalSolverType::SteepestDescent,
        local_solver_config: SteepestDescentBuilder::default().build(),
        seed: 0,
    };

    let mut oqnlp: OQNLP<SixHumpCamel> = OQNLP::new(problem, params)?;
    let solution: LocalSolution = oqnlp.run()?;

    Ok(solution)
}

fn run_six_hump_camel(c: &mut Criterion) {
    c.bench_function("six_hump_camel", |b| b.iter(|| six_hump_camel()));
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(500).measurement_time(std::time::Duration::from_secs(200));
    targets = run_six_hump_camel
}
criterion_main!(benches);
