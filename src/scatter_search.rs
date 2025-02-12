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
    rng: Mutex<StdRng>,
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

        let bounds: VariableBounds = VariableBounds {
            lower: lower.remove_axis(Axis(1)),
            upper: upper.remove_axis(Axis(1)),
        };

        let seed: u64 = params.seed;
        let mut ss: ScatterSearch<P> = Self {
            problem,
            params,
            reference_set: Vec::new(),
            bounds,
            rng: Mutex::new(StdRng::seed_from_u64(seed)),
        };

        ss.initialize_reference_set()?;
        Ok(ss)
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
        let mut ref_set: Vec<Array1<f64>> = Vec::with_capacity(self.params.population_size);

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
        let mut candidates = self.generate_stratified_samples(self.params.population_size)?;

        #[cfg(feature = "rayon")]
        let mut min_dists: Vec<f64> = candidates
            .par_iter()
            .map(|c| self.min_distance(c, ref_set))
            .collect();

        #[cfg(not(feature = "rayon"))]
        let mut min_dists: Vec<f64> = candidates
            .iter()
            .map(|c| self.min_distance(c, ref_set))
            .collect();

        while ref_set.len() < self.params.population_size {
            let (max_idx, _) = min_dists
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .ok_or(ScatterSearchError::NoCandidates)?;

            let farthest = candidates.swap_remove(max_idx);
            min_dists.swap_remove(max_idx);
            ref_set.push(farthest);

            if ref_set.len() >= self.params.population_size {
                break;
            }

            #[cfg(feature = "rayon")]
            let updater_iter = candidates.par_iter().zip(min_dists.par_iter_mut());
            #[cfg(not(feature = "rayon"))]
            let updater_iter = candidates.iter().zip(min_dists.iter_mut());

            #[cfg(feature = "rayon")]
            updater_iter.for_each(|(c, current_min)| {
                let dist = euclidean_distance(c, ref_set.last().unwrap());
                if dist < *current_min {
                    *current_min = dist;
                }
            });

            #[cfg(not(feature = "rayon"))]
            updater_iter.for_each(|(c, current_min)| {
                let dist: f64 = euclidean_distance(c, ref_set.last().unwrap());
                if dist < *current_min {
                    *current_min = dist;
                }
            });
        }

        Ok(())
    }

    pub fn generate_stratified_samples(
        &self,
        n: usize,
    ) -> Result<Vec<Array1<f64>>, ScatterSearchError> {
        let dim: usize = self.bounds.lower.len();

        // Precompute seeds while holding the mutex once
        let seeds: Vec<u64> = {
            let mut rng = self.rng.lock().unwrap();
            (0..n).map(|_| rng.random::<u64>()).collect::<Vec<_>>()
        };

        #[cfg(feature = "rayon")]
        let samples = seeds
            .into_par_iter()
            .map(|seed| {
                let mut rng = StdRng::seed_from_u64(seed);
                Ok(Array1::from_shape_fn(dim, |i| {
                    rng.random_range(self.bounds.lower[i]..=self.bounds.upper[i])
                }))
            })
            .collect::<Result<Vec<_>, ScatterSearchError>>()?;

        #[cfg(not(feature = "rayon"))]
        let samples = seeds
            .into_iter()
            .map(|seed: u64| {
                let mut rng = StdRng::seed_from_u64(seed);
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

        // Precompute seeds for each combine_points call
        let seeds: Vec<u64> = {
            let mut rng = self.rng.lock().unwrap();
            (0..indices.len())
                .map(|_| rng.random::<u64>())
                .collect::<Vec<_>>()
        };

        #[cfg(feature = "rayon")]
        let trial_points: Vec<Vec<Array1<f64>>> = indices
            .par_iter()
            .zip(seeds.par_iter())
            .map(|(&(i, j), &seed)| {
                let point1 = self.reference_set[i].clone();
                let point2 = self.reference_set[j].clone();
                self.combine_points(&point1, &point2, seed)
            })
            .collect::<Result<Vec<_>, ScatterSearchError>>()?;

        #[cfg(not(feature = "rayon"))]
        let trial_points: Vec<Vec<Array1<f64>>> = indices
            .iter()
            .zip(seeds.iter())
            .map(|(&(i, j), &seed)| {
                let point1 = self.reference_set[i].clone();
                let point2 = self.reference_set[j].clone();
                self.combine_points(&point1, &point2, seed)
            })
            .collect::<Result<Vec<_>, ScatterSearchError>>()?;

        Ok(trial_points.into_iter().flatten().collect())
    }

    /// Combines two points into several trial points.
    pub fn combine_points(
        &self,
        a: &Array1<f64>,
        b: &Array1<f64>,
        seed: u64,
    ) -> Result<Vec<Array1<f64>>, ScatterSearchError> {
        let mut points = Vec::new();

        // Linear combinations.
        let directions = vec![0.25, 0.5, 0.75, 1.25];
        for &alpha in &directions {
            let mut point = a.clone() * alpha + b.clone() * (1.0 - alpha);
            self.apply_bounds(&mut point);
            points.push(point);
        }

        // Random perturbations using the provided seed
        let mut rng = StdRng::seed_from_u64(seed);
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
            let obj_a: f64 = self.problem.objective(a).unwrap();
            let obj_b: f64 = self.problem.objective(b).unwrap();
            obj_a.partial_cmp(&obj_b).unwrap()
        });
        #[cfg(not(feature = "rayon"))]
        let best = self.reference_set.iter().min_by(|a, b| {
            let obj_a: f64 = self.problem.objective(a).unwrap();
            let obj_b: f64 = self.problem.objective(b).unwrap();
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
mod tests_scatter_search {
    use super::*;
    use crate::local_solver::builders::SteepestDescentBuilder;
    use crate::types::OQNLPParams;
    use crate::types::{EvaluationError, LocalSolverType};
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
