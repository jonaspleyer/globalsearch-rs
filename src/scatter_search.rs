//! # Scatter Search module
//!
//! This module contains the implementation of the Scatter Search algorithm.

use crate::problem::Problem;
use crate::types::{OQNLPParams, Result};
use ndarray::{Array1, Axis};
use rand::Rng;
use rayon::prelude::*;

// TODO: Check params
pub struct ScatterSearch<P: Problem> {
    problem: P,
    params: OQNLPParams,
}

// TODO: Set seedable RNG

// TODO: Check paper "Scatter Search implementation in C" and add reference
// Provide different implementations of Scatter Search
impl<P: Problem> ScatterSearch<P> {
    pub fn new(problem: P, params: OQNLPParams) -> Self {
        Self { problem, params }
    }

    pub fn generate_trial_points(&self, count: usize) -> Result<Vec<Array1<f64>>> {
        let bounds = self.problem.variable_bounds();
        let dim: usize = bounds.len_of(Axis(0));

        Ok((0..count)
            .into_par_iter()
            .map(|_| {
                Array1::from_shape_fn(dim, |i| {
                    let lower: f64 = bounds[[i, 0]];
                    let upper: f64 = bounds[[i, 1]];
                    rand::rng().random_range(lower..=upper)
                })
            })
            .collect())
    }
}
