//! # Filters module
//!
//! This module contains the implementation of the filters used in the OQNLP algorithm. The filters are used to maintain diversity among solutions and to check if a solution is below a certain threshold.

use crate::types::{FilterParams, LocalSolution};
use ndarray::Array1;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum FiltersErrors {
    /// Distance factor must be positive or equal to zero
    #[error("Distance factor must be positive or equal to zero, got {0}.")]
    NegativeDistanceFactor(f64),
}

/// Merit filter
pub struct MeritFilter {
    pub threshold: f64,
}

impl Default for MeritFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl MeritFilter {
    /// Create a new MeritFilter
    pub fn new() -> Self {
        Self {
            threshold: f64::INFINITY,
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

/// Distance filter
pub struct DistanceFilter {
    solutions: Vec<LocalSolution>, // TODO: Change to ndarray?
    params: FilterParams,
}

impl DistanceFilter {
    /// # Create a new DistanceFilter with the given parameters
    ///
    /// Create a new DistanceFilter with the given parameters and an empty solution set
    /// to store the solutions.
    ///
    /// ## Errors
    ///
    /// Returns an error if the distance factor is negative
    pub fn new(params: FilterParams) -> Result<Self, FiltersErrors> {
        if params.distance_factor < 0.0 {
            return Err(FiltersErrors::NegativeDistanceFactor(
                params.distance_factor,
            ));
        }

        Ok(Self {
            solutions: Vec::new(), // Use ndarray?
            params,
        })
    }

    /// Get the minimum distance between the given point and all solutions in DistanceFilter
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

/// Euclidean distance
fn euclidean_distance(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod test_filters {
    use super::*;
    
    #[test]
    /// Test the invalid distance factor for the Distance Filter
    fn test_filter_params_invalid_distance_factor() {
        let params: FilterParams = FilterParams {
            distance_factor: -0.5, // Distance Factor should be greater or equal to 0.0
            wait_cycle: 10,
            threshold_factor: 0.1,
        };

        let df: Result<DistanceFilter, FiltersErrors> = DistanceFilter::new(params);

        assert!(matches!(
            df,
            Err(FiltersErrors::NegativeDistanceFactor(-0.5))
        ));
    }
}
