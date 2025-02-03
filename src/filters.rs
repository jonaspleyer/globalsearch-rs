//! # Filters module
//!
//! This module contains the implementation of the filters used in the OQNLP algorithm. The filters are used to maintain diversity among solutions and to check if a solution is below a certain threshold.

use crate::types::{FilterParams, LocalSolution};
use ndarray::Array1;

pub struct MeritFilter {
    pub threshold: f64,
    params: FilterParams,
}

// TODO: Implement filters, use rayon? Update threshold?
// Paper suggests (for waitcycle): threshold = threshold + threshold_factor * (1 + abs(threshold))
impl MeritFilter {
    /// Create a new MeritFilter with the given parameters
    pub fn new(params: FilterParams) -> Self {
        Self {
            threshold: f64::INFINITY, // TODO: Should this be params.threshold_factor? or is it updated later?
            params,
        }
    }

    pub fn update_threshold(&mut self, threshold: f64) {
        self.threshold = threshold;
    }

    /// Check if the given value is below the threshold
    pub fn check(&self, value: f64) -> bool {
        value <= self.threshold
    }
}

pub struct DistanceFilter {
    solutions: Vec<LocalSolution>, // TODO: Change to ndarray?
    params: FilterParams,
}

impl DistanceFilter {
    /// Create a new DistanceFilter with the given parameters
    pub fn new(params: FilterParams) -> Self {
        Self {
            solutions: Vec::new(), // Use ndarray?
            params,
        }
    }

    pub fn min_distance(&self, point: &Array1<f64>) -> f64 {
        self.solutions
            .iter()
            .map(|s| euclidean_distance(point, &s.point))
            .fold(f64::INFINITY, |a, b| a.min(b))
    }

    /// Add a solution to DistanceFilter
    pub fn add_solution(&mut self, solution: LocalSolution) {
        self.solutions.push(solution);
    }

    /// Check if the given point is far enough from all solutions in DistanceFilter
    pub fn check(&self, point: &Array1<f64>) -> bool {
        self.solutions
            .iter()
            .all(|s| euclidean_distance(point, &s.point) > self.params.distance_factor)
    }
}

/// RefSet Euclidean distance
fn euclidean_distance(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}
