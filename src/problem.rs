use crate::types::Result;
use ndarray::{Array1, Array2};

/// Trait for optimization problems
///
/// This trait defines the methods that an optimization problem must implement, including the objective function, gradient, variable bounds, and dimension.
pub trait Problem {
    /// Objective function to minimize, given at point x (`Array1<f64>`)
    ///
    /// Returns a `Result<f64>` of the value of the objective function at x
    fn objective(&self, x: &Array1<f64>) -> Result<f64>;

    /// Gradient of the objective function at point x (`Array1<f64>`)
    ///
    /// Returns a `Result<Array1<f64>>` of the gradient of the objective function at x
    fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>>;

    // TODO: Should variable bounds be optional? Also, just set it as an array, not function (struct?)
    // The variable bounds is only being used in the scatter search
    // and not in the minimization algorithm; this should either be explained or changed

    /// Variable bounds for the optimization problem
    ///
    /// Returns a `Result<Array2<f64>>` of the variable bounds for the optimization problem
    fn variable_bounds(&self) -> Array2<f64>;
}
