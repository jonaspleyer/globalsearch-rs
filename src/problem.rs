//! # Optimization problem trait module
//!
//! This module contains the `Problem` trait, which defines the methods that an optimization problem must implement, including the objective function, gradient and variable bounds.
//!
//! ## Example
//! ```rust
//! /// References:
//! ///
//! /// Molga, M., & Smutnicki, C. Test functions for optimization needs (April 3, 2005), pp. 11-12. Retrieved January 2025, from https://robertmarks.org/Classes/ENGR5358/Papers/functions.pdf
//!
//! use globalsearch::problem::Problem;
//! use globalsearch::types::EvaluationError;
//! use ndarray::{array, Array1, Array2};
//!
//! #[derive(Debug, Clone)]
//! pub struct OneDGriewank;
//!
//! impl Problem for OneDGriewank {
//!    fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
//!       Ok(1.0 + x[0].powi(2) / 4000.0 - x[0].cos())
//!    }
//!
//!    // Calculated analytically, reference didn't provide gradient
//!    fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
//!        Ok(array![x[0] / 2000.0 + x[0].sin()])
//!    }
//!
//!    fn variable_bounds(&self) -> Array2<f64> {
//!        array![[-600.0, 600.0]]
//!    }
//! }
use crate::types::EvaluationError;
use ndarray::{Array1, Array2};

/// Trait for optimization problems
///
/// This trait defines the methods that an optimization problem must implement, including the objective function, gradient and variable bounds.
pub trait Problem {
    /// Objective function to minimize, given at point x (`Array1<f64>`)
    ///
    /// Returns a `Result<f64, EvaluationError>` of the value of the objective function at x
    fn objective(&self, _x: &Array1<f64>) -> Result<f64, EvaluationError> {
        Err(EvaluationError::ObjectiveFunctionNotImplemented)
    }

    /// Gradient of the objective function at point x (`Array1<f64>`)
    ///
    /// Returns a `Result<Array1<f64>, EvaluationError>` of the gradient of the objective function at x
    ///
    /// The default implementation returns an error indicating the gradient is not implemented
    /// in case it is needed
    fn gradient(&self, _x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
        Err(EvaluationError::GradientNotImplemented)
    }

    /// Returns the Hessian at point x (`Array1<f64>`).
    ///
    /// Returns a `Result<Array2<f64>, EvaluationError>` of the hessian of the objective function at x
    ///
    /// The default implementation returns an error indicating the hessian is not implemented
    /// in case it is needed
    fn hessian(&self, _x: &Array1<f64>) -> Result<Array2<f64>, EvaluationError> {
        Err(EvaluationError::HessianNotImplemented)
    }

    /// Variable bounds for the optimization problem
    ///
    /// Returns a `Result<Array2<f64>>` of the variable bounds for the optimization problem.
    ///
    /// This bounds are only used in the scatter search phase of the algorithm.
    /// The local solver is unconstrained (See [argmin issue #137](https://github.com/argmin-rs/argmin/issues/137)) and therefor can return solutions out of the bounds.
    /// You may be able to guide your solutions to your desired bounds/constraints by using a penalty method.
    fn variable_bounds(&self) -> Array2<f64>;
}
