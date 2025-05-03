use criterion::{criterion_group, criterion_main, Criterion};
use globalsearch::problem::Problem;
use globalsearch::scatter_search::ScatterSearch;
use globalsearch::types::{EvaluationError, OQNLPParams};
use ndarray::{array, Array1, Array2};
use std::hint::black_box;

#[derive(Debug, Clone)]
struct SixHumpCamel;

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

fn run_scatter_search_population(c: &mut Criterion) {
    use criterion::BenchmarkId;
    let sizes: [usize; 3] = [1000, 10000, 100000];
    for &size in &sizes {
        c.bench_with_input(
            BenchmarkId::new("scatter_search", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let problem = SixHumpCamel;
                    let params = OQNLPParams {
                        population_size: size,
                        ..OQNLPParams::default()
                    };
                    let ss = ScatterSearch::new(problem, params).unwrap();
                    let (ref_set, best) = ss.run().unwrap();
                    black_box((ref_set, best));
                })
            },
        );
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(5))
        .sample_size(30);
    targets = run_scatter_search_population
}
criterion_main!(benches);
