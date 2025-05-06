//! # Scatter Search module
//!
//! This module contains the implementation of the Scatter Search algorithm.
//!
//! The Scatter Search algorithm is a population-based optimization algorithm
//! that uses a reference set of solutions to generate new candidate solutions.
//!
//! The algorithm is divided into three main steps:
//!  1. Initialization of the reference set
//!  2. Diversification of the reference set
//!  3. Generation of trial points and evaluation

use crate::problem::Problem;
use crate::types::OQNLPParams;
use ndarray::Array1;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Mutex;
use thiserror::Error;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[cfg(feature = "progress_bar")]
use kdam::{Bar, BarExt};

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
    #[error("Scatter Search Error: Evaluation error: {0}.")]
    EvaluationError(#[from] crate::types::EvaluationError),
}

/// Scatter Search algorithm implementation struct
pub struct ScatterSearch<P: Problem> {
    problem: P,
    params: OQNLPParams,
    reference_set: Vec<Array1<f64>>,
    bounds: VariableBounds,
    rng: Mutex<StdRng>,
    #[cfg(feature = "progress_bar")]
    progress_bar: Option<Bar>,
}

impl<P: Problem + Sync + Send> ScatterSearch<P> {
    pub fn new(problem: P, params: OQNLPParams) -> Result<Self, ScatterSearchError> {
        let var_bounds = problem.variable_bounds();
        let bounds = VariableBounds {
            lower: var_bounds.column(0).to_owned(),
            upper: var_bounds.column(1).to_owned(),
        };

        let seed: u64 = params.seed;
        let ss: ScatterSearch<P> = Self {
            problem,
            params,
            reference_set: Vec::new(),
            bounds,
            rng: Mutex::new(StdRng::seed_from_u64(seed)),
            #[cfg(feature = "progress_bar")]
            progress_bar: None,
        };

        Ok(ss)
    }

    /// Run the Scatter Search algorithm
    ///
    /// Returns the reference set and the best solution found
    pub fn run(mut self) -> Result<(Vec<Array1<f64>>, Array1<f64>), ScatterSearchError> {
        #[cfg(feature = "progress_bar")]
        {
            self.progress_bar = Some(
                Bar::builder()
                    .total(3)
                    .desc("Stage 1")
                    .unit("steps")
                    .build()
                    .expect("Failed to create progress bar"),
            );
        }

        self.initialize_reference_set()?;
        let best = self.best_solution()?;

        #[cfg(feature = "progress_bar")]
        if let Some(pb) = &mut self.progress_bar {
            pb.set_description("Stage 1, found best solution");
            pb.update(1).expect("Failed to update progress bar");
        }
        Ok((self.reference_set, best))
    }

    pub fn initialize_reference_set(&mut self) -> Result<(), ScatterSearchError> {
        let mut ref_set: Vec<Array1<f64>> = Vec::with_capacity(self.params.population_size);

        ref_set.push(self.bounds.lower.to_owned());
        ref_set.push(self.bounds.upper.to_owned());
        ref_set.push((&self.bounds.lower + &self.bounds.upper) / 2.0);

        #[cfg(feature = "progress_bar")]
        if let Some(pb) = &mut self.progress_bar {
            pb.set_description("Stage 1, initialized reference set");
            pb.update(1).expect("Failed to update progress bar");
        }

        self.diversify_reference_set(&mut ref_set)?;
        self.reference_set = ref_set;

        #[cfg(feature = "progress_bar")]
        if let Some(pb) = &mut self.progress_bar {
            pb.set_description("Stage 1, diversified reference set");
            pb.update(1).expect("Failed to update progress bar");
        }
        Ok(())
    }

    /// Diversify the reference set by adding new points to it
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

            updater_iter.for_each(|(c, current_min)| {
                if let Some(last) = ref_set.last() {
                    let dist: f64 = euclidean_distance_squared(c, last);
                    if dist < *current_min {
                        *current_min = dist;
                    }
                } else {
                    unreachable!("Reference set is empty");
                }
            });
        }

        Ok(())
    }

    /// Generate stratified samples within the bounds
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

    /// Compute the minimum distance between a point and a reference set
    pub fn min_distance(&self, point: &Array1<f64>, ref_set: &[Array1<f64>]) -> f64 {
        #[cfg(feature = "rayon")]
        {
            ref_set
                .par_iter()
                .map(|p| euclidean_distance_squared(point, p))
                .reduce(|| f64::INFINITY, f64::min)
        }
        #[cfg(not(feature = "rayon"))]
        {
            ref_set
                .iter()
                .map(|p| euclidean_distance_squared(point, p))
                .fold(f64::INFINITY, f64::min)
        }
    }

    pub fn generate_trial_points(&mut self) -> Result<Vec<Array1<f64>>, ScatterSearchError> {
        // Create all index pairs.
        let indices: Vec<(usize, usize)> = (0..self.reference_set.len())
            .flat_map(|i| ((i + 1)..self.reference_set.len()).map(move |j| (i, j)))
            .collect();

        // Pre-allocate the result vector to avoid reallocations
        let n_combinations = indices.len();
        let n_trial_points_per_combo = 6; // 4 linear combinations + 2 random
        let mut trial_points: Vec<Array1<f64>> =
            Vec::with_capacity(n_combinations * n_trial_points_per_combo);

        // Precompute seeds for each combine_points call
        let seeds: Vec<u64> = {
            let mut rng = self.rng.lock().unwrap();
            (0..indices.len())
                .map(|_| rng.random::<u64>())
                .collect::<Vec<_>>()
        };

        #[cfg(feature = "rayon")]
        {
            let points_per_combo: Vec<Vec<Array1<f64>>> = indices
                .par_iter()
                .zip(seeds.par_iter())
                .map(|(&(i, j), &seed)| {
                    self.combine_points(&self.reference_set[i], &self.reference_set[j], seed)
                })
                .collect::<Result<Vec<_>, ScatterSearchError>>()?;

            for points in points_per_combo {
                trial_points.extend(points);
            }
        }

        #[cfg(not(feature = "rayon"))]
        {
            for (&(i, j), &seed) in indices.iter().zip(seeds.iter()) {
                let points =
                    self.combine_points(&self.reference_set[i], &self.reference_set[j], seed)?;
                trial_points.extend(points);
            }
        }

        Ok(trial_points)
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
        let directions: Vec<f64> = vec![0.25, 0.5, 0.75, 1.25];
        for &alpha in &directions {
            let mut point = a * alpha + b * (1.0 - alpha);
            self.apply_bounds(&mut point);
            points.push(point);
        }

        // Random perturbations using the provided seed
        let mut rng: StdRng = StdRng::seed_from_u64(seed);
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

            let pop_size = self.params.population_size;
            evaluated.select_nth_unstable_by(pop_size, |a, b| a.1.total_cmp(&b.1));
            evaluated.truncate(pop_size);
            self.reference_set = evaluated.into_iter().map(|(p, _)| p).collect();

            Ok(())
        }
        #[cfg(not(feature = "rayon"))]
        {
            let mut evaluated: Vec<(Array1<f64>, f64)> = self
                .reference_set
                .iter()
                .chain(trials.iter())
                .map(|point| {
                    let obj: f64 = self.problem.objective(point)?;
                    Ok((point.clone(), obj))
                })
                .collect::<Result<Vec<(Array1<f64>, f64)>, ScatterSearchError>>()?;

            let pop_size = self.params.population_size;
            evaluated.select_nth_unstable_by(pop_size, |a, b| a.1.total_cmp(&b.1));
            evaluated.truncate(pop_size);
            self.reference_set = evaluated.into_iter().map(|(p, _)| p).collect();

            Ok(())
        }
    }

    pub fn best_solution(&self) -> Result<Array1<f64>, ScatterSearchError> {
        #[cfg(feature = "rayon")]
        {
            let best = self
                .reference_set
                .par_iter()
                .min_by(|a, b| {
                    let obj_a: f64 = self.problem.objective(a).unwrap();
                    let obj_b: f64 = self.problem.objective(b).unwrap();
                    obj_a.partial_cmp(&obj_b).unwrap()
                })
                .ok_or(ScatterSearchError::NoCandidates)?;
            Ok(best.clone())
        }
        #[cfg(not(feature = "rayon"))]
        {
            let mut best_point: Option<(&Array1<f64>, f64)> = None;
            for point in &self.reference_set {
                let obj: f64 = self.problem.objective(point)?;
                best_point = match best_point {
                    None => Some((point, obj)),
                    Some((_, best_obj)) if obj < best_obj => Some((point, obj)),
                    Some(current) => Some(current),
                };
            }
            best_point
                .map(|(p, _)| p.clone())
                .ok_or(ScatterSearchError::NoCandidates)
        }
    }

    pub fn store_trial(&mut self, trial: Array1<f64>) {
        self.reference_set.push(trial);
    }
}

/// Compute the squared Euclidean distance between two points
///
/// Use this function for performance since we don't use the square root
fn euclidean_distance_squared(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    (a - b).mapv(|x| x * x).sum()
}

// The following code allows to compute the Euclidean distance between two points
// but it is not used in the current implementation
// Could there be some cases where we need to compute the Euclidean distance?
// due to overflow or numerical stability?
//
// /// Compute the Euclidean distance between two points.
// ///
// /// Use euclidean_distance_squared when only comparing distances for better performance
// /// given that if a < b, then a² < b²
// fn euclidean_distance(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
//     euclidean_distance_squared(a, b).sqrt()
// }

#[cfg(test)]
mod tests_scatter_search {
    use super::*;
    use crate::types::EvaluationError;
    use crate::types::OQNLPParams;
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
            seed: 0,
            ..OQNLPParams::default()
        };

        let ss: ScatterSearch<SixHumpCamel> = ScatterSearch::new(problem, params).unwrap();
        let (ref_set, _) = ss.run().unwrap();
        assert_eq!(ref_set.len(), 100);
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
            seed: 0,
            ..OQNLPParams::default()
        };

        let ss: ScatterSearch<SixHumpCamel> = ScatterSearch::new(problem, params).unwrap();
        let bounds: VariableBounds = ss.bounds.clone();
        let (ref_set, _) = ss.run().unwrap();

        assert_eq!(ref_set.len(), 100);

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
            seed: 0,
            ..OQNLPParams::default()
        };

        let ss1: ScatterSearch<SixHumpCamel> =
            ScatterSearch::new(problem.clone(), params.clone()).unwrap();
        let ss2: ScatterSearch<SixHumpCamel> = ScatterSearch::new(problem, params).unwrap();

        let (ref_set1, _) = ss1.run().unwrap();
        let (ref_set2, _) = ss2.run().unwrap();

        assert_eq!(ref_set1.len(), 100);
        assert_eq!(ref_set1.len(), ref_set2.len());

        for i in 0..ref_set1.len() {
            assert_eq!(ref_set1[i], ref_set2[i]);
        }
    }

    #[test]
    /// Test generating trial points for a `ScatterSearch` instance
    fn test_generate_trial_points() {
        let problem: SixHumpCamel = SixHumpCamel;
        let params: OQNLPParams = OQNLPParams {
            iterations: 1,
            wait_cycle: 30,
            threshold_factor: 0.2,
            distance_factor: 0.75,
            population_size: 10,
            seed: 0,
            ..OQNLPParams::default()
        };

        let mut ss: ScatterSearch<SixHumpCamel> = ScatterSearch::new(problem, params).unwrap();
        ss.initialize_reference_set().unwrap();

        let trial_points: Vec<Array1<f64>> = ss.generate_trial_points().unwrap();

        // With 10 points in the reference set, we should have C(10, 2) = 45 combinations
        // Each combination produces 6 trial points (4 linear combinations + 2 random)
        // So the reference set has 45 * 6 = 270 trial points
        assert_eq!(trial_points.len(), 45 * 6);
    }

    #[test]
    /// Test combining two points into trial points
    fn test_combine_points() {
        let problem: SixHumpCamel = SixHumpCamel;
        let params: OQNLPParams = OQNLPParams {
            iterations: 1,
            wait_cycle: 30,
            threshold_factor: 0.2,
            distance_factor: 0.75,
            population_size: 10,
            seed: 0,
            ..OQNLPParams::default()
        };

        let ss: ScatterSearch<SixHumpCamel> = ScatterSearch::new(problem, params).unwrap();
        let a: Array1<f64> = array![1.0, 1.0];
        let b: Array1<f64> = array![2.0, 2.0];

        let trial_points: Vec<Array1<f64>> = ss.combine_points(&a, &b, 0).unwrap();

        // 4 linear combinations and 2 random perturbations
        assert_eq!(trial_points.len(), 6);
    }

    #[test]
    /// Test storing trials in the reference set
    fn test_store_trials() {
        let problem: SixHumpCamel = SixHumpCamel;
        let params: OQNLPParams = OQNLPParams {
            iterations: 1,
            wait_cycle: 30,
            threshold_factor: 0.2,
            distance_factor: 0.75,
            population_size: 4,
            seed: 0,
            ..OQNLPParams::default()
        };

        let mut ss: ScatterSearch<SixHumpCamel> = ScatterSearch::new(problem, params).unwrap();

        // Initially empty reference set (not initialized)
        assert_eq!(ss.reference_set.len(), 0);

        let trial: Array1<f64> = array![1.0, 1.0];
        ss.store_trial(trial.clone());

        // Verify trial was stored
        assert_eq!(ss.reference_set.len(), 1);
        assert_eq!(ss.reference_set[0], trial);
    }

    #[test]
    /// Test updating the reference set with new trials
    fn test_update_reference_set() {
        let problem: SixHumpCamel = SixHumpCamel;
        let params: OQNLPParams = OQNLPParams {
            iterations: 1,
            wait_cycle: 30,
            threshold_factor: 0.2,
            distance_factor: 0.75,
            population_size: 4,
            seed: 0,
            ..OQNLPParams::default()
        };

        let mut ss: ScatterSearch<SixHumpCamel> = ScatterSearch::new(problem, params).unwrap();
        ss.initialize_reference_set().unwrap();

        let trials: Vec<Array1<f64>> = vec![array![1.0, 1.0], array![2.0, 2.0]];
        ss.update_reference_set(trials).unwrap();

        assert_eq!(ss.reference_set.len(), 4);
    }

    #[test]
    /// Test computing the minimum distance between a point and a reference set
    fn test_min_distance() {
        let problem: SixHumpCamel = SixHumpCamel;
        let params: OQNLPParams = OQNLPParams {
            iterations: 1,
            wait_cycle: 30,
            threshold_factor: 0.2,
            distance_factor: 0.75,
            population_size: 4,
            seed: 0,
            ..OQNLPParams::default()
        };

        let mut ss: ScatterSearch<SixHumpCamel> = ScatterSearch::new(problem, params).unwrap();
        ss.initialize_reference_set().unwrap();

        let point: Array1<f64> = array![-3.0, -2.0];
        let min_dist: f64 = ss.min_distance(&point, &ss.reference_set);

        // The minimum distance should be 0 since the point is in the reference set
        assert_eq!(min_dist, 0.0);
    }

    #[test]
    /// Test euclidean distance squared
    fn test_euclidean_distance_squared() {
        let a: Array1<f64> = array![1.0, 2.0];
        let b: Array1<f64> = array![3.0, 4.0];
        let dist: f64 = euclidean_distance_squared(&a, &b);
        assert_eq!(dist, 8.0);
    }

    #[cfg(feature = "rayon")]
    #[test]
    /// Test generating trial points using rayon
    fn test_generate_trial_points_rayon() {
        let problem: SixHumpCamel = SixHumpCamel;
        let params: OQNLPParams = OQNLPParams {
            iterations: 1,
            wait_cycle: 30,
            threshold_factor: 0.2,
            distance_factor: 0.75,
            population_size: 10,
            seed: 0,
            ..OQNLPParams::default()
        };

        let mut ss: ScatterSearch<SixHumpCamel> = ScatterSearch::new(problem, params).unwrap();
        ss.initialize_reference_set().unwrap();

        let trial_points: Vec<Array1<f64>> = ss.generate_trial_points().unwrap();

        // With 10 points in the reference set, we should have C(10, 2) = 45 combinations
        // Each combination produces 6 trial points (4 linear combinations + 2 random)
        // So the reference set has 45 * 6 = 270 trial points
        assert_eq!(trial_points.len(), 45 * 6);
    }

    #[cfg(feature = "rayon")]
    #[test]
    /// Test updating the reference set using rayon
    fn test_update_reference_set_rayon() {
        let problem: SixHumpCamel = SixHumpCamel;
        let params: OQNLPParams = OQNLPParams {
            iterations: 1,
            wait_cycle: 30,
            threshold_factor: 0.2,
            distance_factor: 0.75,
            population_size: 4,
            seed: 0,
            ..OQNLPParams::default()
        };

        let mut ss: ScatterSearch<SixHumpCamel> = ScatterSearch::new(problem, params).unwrap();
        ss.initialize_reference_set().unwrap();

        let trials: Vec<Array1<f64>> = vec![array![1.0, 1.0], array![2.0, 2.0]];
        ss.update_reference_set(trials).unwrap();

        assert_eq!(ss.reference_set.len(), 4);
    }

    #[cfg(feature = "rayon")]
    #[test]
    /// Test computing the minimum distance between a point and a reference set using rayon
    fn test_min_distance_rayon() {
        let problem: SixHumpCamel = SixHumpCamel;
        let params: OQNLPParams = OQNLPParams {
            iterations: 1,
            wait_cycle: 30,
            threshold_factor: 0.2,
            distance_factor: 0.75,
            population_size: 4,
            seed: 0,
            ..OQNLPParams::default()
        };

        let mut ss: ScatterSearch<SixHumpCamel> = ScatterSearch::new(problem, params).unwrap();
        ss.initialize_reference_set().unwrap();

        let point: Array1<f64> = array![-3.0, -2.0];
        let min_dist: f64 = ss.min_distance(&point, &ss.reference_set);

        // The minimum distance should be 0 since the point is in the reference set
        assert_eq!(min_dist, 0.0);
    }
}
