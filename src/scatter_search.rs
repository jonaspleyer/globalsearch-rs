//! # Scatter Search module
//!
//! This module contains the implementation of the Scatter Search algorithm.

use crate::problem::Problem;
use crate::types::OQNLPParams;
use ndarray::{Array1, Axis};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Mutex;
use thiserror::Error;

// TODO: For some reason, in my PC it runs faster without rayon (using bench)
// What am I doing wrong? Is it overhead?

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Struct to hold the bounds of the variables
#[derive(Debug, Clone)]
pub struct VariableBounds {
    pub lower: Array1<f64>,
    pub upper: Array1<f64>,
}

#[derive(Debug, Error)]
/// Error type for Scatter Search
pub enum ScatterSearchError {
    /// Error when the reference set is empty
    #[error("Scatter Search Error: No candidates left.")]
    NoCandidates,

    /// Error when evaluating the objective function
    #[error("Scatter Search Error: Evaluation error: {0}")]
    EvaluationError(#[from] crate::types::EvaluationError),
}

/// Scatter Search algorithm implementation struct
pub struct ScatterSearch<P: Problem> {
    problem: P,
    params: OQNLPParams,
    reference_set: Vec<Array1<f64>>,
    bounds: VariableBounds,
    rng: Mutex<StdRng>, // Wrapped in a Mutex for thread safety
}

impl<P: Problem + Sync + Send> ScatterSearch<P> {
    pub fn new(problem: P, params: OQNLPParams) -> Result<Self, ScatterSearchError> {
        let lower = problem
            .variable_bounds()
            .slice_axis(Axis(1), ndarray::Slice::from(0..1))
            .into_owned();
        let upper = problem
            .variable_bounds()
            .slice_axis(Axis(1), ndarray::Slice::from(1..2))
            .into_owned();

        let bounds = VariableBounds {
            lower: lower.remove_axis(Axis(1)),
            upper: upper.remove_axis(Axis(1)),
        };

        let seed: u64 = params.seed;
        let mut ss = Self {
            problem,
            params,
            reference_set: Vec::new(),
            bounds,
            rng: Mutex::new(StdRng::seed_from_u64(seed)),
        };

        ss.initialize_reference_set()?;
        Ok(ss)
    }

    /// Returns a new, thread-local RNG
    /// I think this doesn't work as intended
    fn local_rng(&self) -> StdRng {
        let mut master_rng = self.rng.lock().unwrap();
        let new_seed: u64 = master_rng.random();
        StdRng::seed_from_u64(new_seed)
    }

    /// Run the Scatter Search algorithm
    ///
    /// Returns the reference set and the best solution found
    pub fn run(&mut self) -> Result<(Vec<Array1<f64>>, Array1<f64>), ScatterSearchError> {
        let ref_set = self.reference_set.clone();
        let best = self.best_solution()?;
        Ok((ref_set, best))
    }

    pub fn initialize_reference_set(&mut self) -> Result<(), ScatterSearchError> {
        let mut ref_set = Vec::with_capacity(self.params.population_size);

        ref_set.push(self.bounds.lower.clone());
        ref_set.push(self.bounds.upper.clone());
        ref_set.push((&self.bounds.lower + &self.bounds.upper) / 2.0);

        self.diversify_reference_set(&mut ref_set)?;
        self.reference_set = ref_set;
        Ok(())
    }

    pub fn diversify_reference_set(
        &mut self,
        ref_set: &mut Vec<Array1<f64>>,
    ) -> Result<(), ScatterSearchError> {
        // Generate candidate points using stratified sampling.
        let mut candidates = self.generate_stratified_samples(self.params.population_size)?;

        while ref_set.len() < self.params.population_size {
            let farthest = candidates
                .iter()
                .max_by(|a, b| {
                    self.min_distance(a, ref_set)
                        .partial_cmp(&self.min_distance(b, ref_set))
                        .unwrap()
                })
                .ok_or_else(|| ScatterSearchError::NoCandidates)?
                .clone();

            ref_set.push(farthest.clone());
            candidates.retain(|c| c != &farthest);
        }
        Ok(())
    }

    pub fn generate_stratified_samples(
        &self,
        n: usize,
    ) -> Result<Vec<Array1<f64>>, ScatterSearchError> {
        let dim = self.bounds.lower.len();

        #[cfg(feature = "rayon")]
        let samples = (0..n)
            .into_par_iter()
            .map(|_| {
                // Create a thread-local RNG once per iteration.
                let mut rng = self.local_rng();
                Ok(Array1::from_shape_fn(dim, |i| {
                    rng.random_range(self.bounds.lower[i]..=self.bounds.upper[i])
                }))
            })
            .collect::<Result<Vec<_>, ScatterSearchError>>()?;

        #[cfg(not(feature = "rayon"))]
        let samples = (0..n)
            .map(|_| {
                let mut rng = self.rng.lock().unwrap();
                Ok(Array1::from_shape_fn(dim, |i| {
                    rng.random_range(self.bounds.lower[i]..=self.bounds.upper[i])
                }))
            })
            .collect::<Result<Vec<_>, ScatterSearchError>>()?;

        Ok(samples)
    }

    pub fn min_distance(&self, point: &Array1<f64>, ref_set: &[Array1<f64>]) -> f64 {
        #[cfg(feature = "rayon")]
        {
            ref_set
                .par_iter()
                .map(|p| euclidean_distance(point, p))
                .reduce(|| f64::INFINITY, f64::min)
        }
        #[cfg(not(feature = "rayon"))]
        {
            ref_set
                .iter()
                .map(|p| euclidean_distance(point, p))
                .fold(f64::INFINITY, f64::min)
        }
    }

    pub fn generate_trial_points(&mut self) -> Result<Vec<Array1<f64>>, ScatterSearchError> {
        // Create all index pairs.
        let indices: Vec<(usize, usize)> = (0..self.reference_set.len())
            .flat_map(|i| ((i + 1)..self.reference_set.len()).map(move |j| (i, j)))
            .collect();

        #[cfg(feature = "rayon")]
        let trial_points: Vec<Vec<Array1<f64>>> = indices
            .par_iter()
            .map(|&(i, j)| {
                let point1 = self.reference_set[i].clone();
                let point2 = self.reference_set[j].clone();
                self.combine_points(&point1, &point2)
            })
            .collect::<Result<Vec<_>, ScatterSearchError>>()?;

        #[cfg(not(feature = "rayon"))]
        let trial_points: Vec<Vec<Array1<f64>>> = indices
            .iter()
            .map(|&(i, j)| {
                let point1 = self.reference_set[i].clone();
                let point2 = self.reference_set[j].clone();
                self.combine_points(&point1, &point2)
            })
            .collect::<Result<Vec<_>, ScatterSearchError>>()?;

        Ok(trial_points.into_iter().flatten().collect())
    }

    /// Combines two points into several trial points.
    pub fn combine_points(
        &self,
        a: &Array1<f64>,
        b: &Array1<f64>,
    ) -> Result<Vec<Array1<f64>>, ScatterSearchError> {
        let mut points = Vec::new();

        // Linear combinations.
        let directions = vec![0.25, 0.5, 0.75, 1.25];
        for &alpha in &directions {
            let mut point = a.clone() * alpha + b.clone() * (1.0 - alpha);
            self.apply_bounds(&mut point);
            points.push(point);
        }

        // Random perturbations using a thread-local RNG.
        // Not sure this works as intended
        let mut rng = self.local_rng();
        for _ in 0..2 {
            let mut point = (a + b) / 2.0;
            point.iter_mut().enumerate().for_each(|(i, x)| {
                *x += rng.random_range(-0.1..0.1) * (self.bounds.upper[i] - self.bounds.lower[i]);
            });
            self.apply_bounds(&mut point);
            points.push(point);
        }

        Ok(points)
    }

    pub fn apply_bounds(&self, point: &mut Array1<f64>) {
        for i in 0..point.len() {
            point[i] = point[i].clamp(self.bounds.lower[i], self.bounds.upper[i]);
        }
    }

    pub fn update_reference_set(
        &mut self,
        trials: Vec<Array1<f64>>,
    ) -> Result<(), ScatterSearchError> {
        #[cfg(feature = "rayon")]
        {
            // Evaluate objective values in parallel.
            let mut evaluated: Vec<(Array1<f64>, f64)> = self
                .reference_set
                .iter()
                .chain(trials.iter())
                .par_bridge()
                .map(|point| {
                    let obj = self.problem.objective(point)?;
                    Ok((point.clone(), obj))
                })
                .collect::<Result<Vec<(Array1<f64>, f64)>, ScatterSearchError>>()?;

            evaluated.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            self.reference_set = evaluated
                .into_iter()
                .take(self.params.population_size)
                .map(|(p, _)| p)
                .collect();

            Ok(())
        }
        #[cfg(not(feature = "rayon"))]
        {
            let mut evaluated: Vec<(Array1<f64>, f64)> = self
                .reference_set
                .iter()
                .chain(trials.iter())
                .map(|point| {
                    let obj = self.problem.objective(point)?;
                    Ok((point.clone(), obj))
                })
                .collect::<Result<Vec<(Array1<f64>, f64)>, ScatterSearchError>>()?;

            evaluated.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            self.reference_set = evaluated
                .into_iter()
                .take(self.params.population_size)
                .map(|(p, _)| p)
                .collect();

            Ok(())
        }
    }

    pub fn best_solution(&self) -> Result<Array1<f64>, ScatterSearchError> {
        #[cfg(feature = "rayon")]
        let best = self.reference_set.par_iter().min_by(|a, b| {
            let obj_a = self.problem.objective(a).unwrap();
            let obj_b = self.problem.objective(b).unwrap();
            obj_a.partial_cmp(&obj_b).unwrap()
        });
        #[cfg(not(feature = "rayon"))]
        let best = self.reference_set.iter().min_by(|a, b| {
            let obj_a = self.problem.objective(a).unwrap();
            let obj_b = self.problem.objective(b).unwrap();
            obj_a.partial_cmp(&obj_b).unwrap()
        });
        best.cloned().ok_or(ScatterSearchError::NoCandidates)
    }

    pub fn store_trial(&mut self, trial: Array1<f64>) {
        self.reference_set.push(trial);
    }
}

fn euclidean_distance(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::OQNLPParams;
    use crate::types::{EvaluationError, LocalSolverType, SteepestDescentBuilder};
    use ndarray::{array, Array2};

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

        fn variable_bounds(&self) -> Array2<f64> {
            array![[-3.0, 3.0], [-2.0, 2.0]]
        }
    }

    #[test]
    /// Test if the population size is correctly set in the `ScatterSearch` struct
    fn test_population_size() {
        let problem: SixHumpCamel = SixHumpCamel;
        let params: OQNLPParams = OQNLPParams {
            iterations: 50,
            wait_cycle: 30,
            threshold_factor: 0.2,
            distance_factor: 0.75,
            population_size: 100,
            local_solver_type: LocalSolverType::SteepestDescent,
            local_solver_config: SteepestDescentBuilder::default().build(),
            seed: 0,
        };

        let ss: ScatterSearch<SixHumpCamel> = ScatterSearch::new(problem, params).unwrap();
        assert_eq!(ss.reference_set.len(), 100);
    }

    #[test]
    /// Test if the bounds are correctly set in the `ScatterSearch` struct and
    /// all the points in the reference set are within the bounds
    fn test_bounds_in_reference_set() {
        let problem: SixHumpCamel = SixHumpCamel;
        let params: OQNLPParams = OQNLPParams {
            iterations: 50,
            wait_cycle: 30,
            threshold_factor: 0.2,
            distance_factor: 0.75,
            population_size: 100,
            local_solver_type: LocalSolverType::SteepestDescent,
            local_solver_config: SteepestDescentBuilder::default().build(),
            seed: 0,
        };

        let ss: ScatterSearch<SixHumpCamel> = ScatterSearch::new(problem, params).unwrap();
        let bounds: VariableBounds = ss.bounds;
        let ref_set = ss.reference_set;

        for point in ref_set {
            for i in 0..point.len() {
                assert!(point[i] >= bounds.lower[i]);
                assert!(point[i] <= bounds.upper[i]);
            }
        }
    }

    #[test]
    /// Test that, given the same seed and population size, the reference set
    /// is the same for two different `ScatterSearch` instances
    fn test_same_reference_set() {
        let problem: SixHumpCamel = SixHumpCamel;
        let params: OQNLPParams = OQNLPParams {
            iterations: 50,
            wait_cycle: 30,
            threshold_factor: 0.2,
            distance_factor: 0.75,
            population_size: 100,
            local_solver_type: LocalSolverType::SteepestDescent,
            local_solver_config: SteepestDescentBuilder::default().build(),
            seed: 0,
        };

        let ss1: ScatterSearch<SixHumpCamel> =
            ScatterSearch::new(problem.clone(), params.clone()).unwrap();
        let ss2: ScatterSearch<SixHumpCamel> = ScatterSearch::new(problem, params).unwrap();

        let ref_set1 = ss1.reference_set.clone();
        let ref_set2 = ss2.reference_set.clone();

        for i in 0..ref_set1.len() {
            assert_eq!(ref_set1[i], ref_set2[i]);
        }
    }
}
