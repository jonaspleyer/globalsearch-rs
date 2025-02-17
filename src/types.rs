//! # Types module
//!
//! This module contains the types and structs used in the OQNLP algorithm,
//! including the parameters for the algorithm, filtering mechanisms,
//! solution set, local solutions, local solver types, and local solver configurations.

use crate::local_solver::builders::{LBFGSBuilder, LocalSolverConfig};
use ndarray::Array1;
use std::fmt;
use std::ops::Index;
use thiserror::Error;

// TODO: Implement SR1 when it is fixed in argmin (https://github.com/argmin-rs/argmin/issues/221)
// Or add it now and print a warning that it is not working as expected in some cases

// TODO: Implement methods with Hessians

#[derive(Debug, Clone)]
/// Parameters for the OQNLP algorithm
///
/// These parameters control the behavior of the optimization process,
/// including the total number of iterations, the number of iterations for stage 1,
/// the wait cycle, threshold factor, distance factor, and population size.
pub struct OQNLPParams {
    /// Total number of iterations for the optimization process
    pub iterations: usize,

    /// Number of population size
    ///
    /// Population size is the number of points in the reference set.
    /// The reference set is created in Stage 1, where we optimize the best objective
    /// function value found so far.
    ///
    /// In stage 2, we optimize random `iterations` points of the reference set.
    pub population_size: usize,

    /// Number of iterations to wait before updating the threshold criteria and reference set
    ///
    /// This is used to determine the number of iterations to wait before updating the
    /// threshold criteria (Stage 2) and the reference set (Stage 1).
    pub wait_cycle: usize,

    /// Threshold factor multiplier
    ///
    /// The new threshold is calculated as `threshold = threshold + threshold_factor * (1 + abs(threshold))`
    pub threshold_factor: f64,

    /// Factor that influences the minimum required distance between candidate solutions
    pub distance_factor: f64,

    /// Type of local solver to use from argmin
    pub local_solver_type: LocalSolverType,

    /// Configuration for the local solver
    pub local_solver_config: LocalSolverConfig,

    /// Random seed for the algorithm
    pub seed: u64,
}

impl Default for OQNLPParams {
    /// Default parameters for the OQNLP algorithm
    ///
    /// It is highly recommended to change these parameters based on the problem at hand.
    ///
    /// The default parameters are:
    /// - `iterations`: 300
    /// - `population_size`: 1000
    /// - `wait_cycle`: 15
    /// - `threshold_factor`: 0.2
    /// - `distance_factor`: 0.75
    /// - `local_solver_type`: `LocalSolverType::LBFGS`
    /// - `local_solver_config`: `LBFGSBuilder::default().build()`
    /// - `seed`: 0
    fn default() -> Self {
        Self {
            iterations: 300,
            population_size: 1000,
            wait_cycle: 15,
            threshold_factor: 0.2,
            distance_factor: 0.75,
            local_solver_type: LocalSolverType::LBFGS,
            local_solver_config: LBFGSBuilder::default().build(),
            seed: 0,
        }
    }
}

#[derive(Debug, Clone)]
/// Parameters for the filtering mechanisms
///
/// These parameters control the behavior of the filtering mechanisms, including the distance factor, wait cycle, and threshold factor.
/// The distance factor influences the minimum required distance between candidate solutions, the wait cycle determines the number of iterations to wait before updating the threshold criteria, and the threshold factor is used to update the threshold criteria.
pub struct FilterParams {
    /// Factor that influences the minimum required distance between candidate solutions
    ///
    /// The distance factor is used to determine the minimum required distance between candidate solutions.
    /// If the distance between two solutions is less than the distance factor, one of the solutions is removed.
    ///
    /// The distance factor is used in the `DistanceFilter` mechanism and it is a positive value or zero.
    pub distance_factor: f64,
    /// Number of iterations to wait before updating the threshold criteria
    pub wait_cycle: usize,
    /// Threshold factor
    pub threshold_factor: f64,
}

#[derive(Debug, Clone)]
/// A local solution in the parameter space
///
/// This struct represents a solution point in the parameter space along with the objective function value at that point.
pub struct LocalSolution {
    /// The solution point in the parameter space
    pub point: Array1<f64>,
    /// The objective function value at the solution point
    pub objective: f64,
}

#[derive(Debug, Clone)]
/// A set of local solutions
///
/// This struct represents a set of local solutions in the parameter space
/// including the solution points and their corresponding objective function values.
///
/// The solutions are stored in an `Array1` of `LocalSolution` structs.
///
/// The `SolutionSet` struct implements the `Index` trait to allow indexing into
/// the set and the `Display` trait to allow pretty printing.
///
/// It also provides a method to get the number of solutions stored in the set using `len(&self)`.
pub struct SolutionSet {
    pub solutions: Array1<LocalSolution>,
}

impl SolutionSet {
    /// Returns the number of solutions stored in the set.
    pub fn len(&self) -> usize {
        self.solutions.len()
    }

    /// Returns true if the solution set contains no solutions.
    pub fn is_empty(&self) -> bool {
        self.solutions.is_empty()
    }
}

impl Index<usize> for SolutionSet {
    type Output = LocalSolution;

    /// Returns the solution at the given index.
    fn index(&self, index: usize) -> &Self::Output {
        &self.solutions[index]
    }
}

impl fmt::Display for SolutionSet {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let len: usize = self.solutions.len();
        writeln!(f, "━━━━━━━━━━━ Solution Set ━━━━━━━━━━━")?;
        writeln!(f, "Total solutions: {}", self.solutions.len())?;
        if len > 0 {
            // Since all the solutions have the same objective value (+- eps)
            // we can just print the first one
            writeln!(f, "Objective value {:.8e}", self.solutions[0].objective)?;
        }
        writeln!(f, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")?;

        for (i, solution) in self.solutions.iter().enumerate() {
            writeln!(f, "Solution #{}", i + 1)?;
            writeln!(f, "  Objective: {:.8e}", solution.objective)?;
            writeln!(f, "  Parameters:")?;
            writeln!(f, "    {:.8e}", solution.point)?;

            if i < self.solutions.len() - 1 {
                writeln!(f, "――――――――――――――――――――――――――――――――――――")?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
/// Local solver implementation types for the OQNLP algorithm
///
/// This enum defines the types of local solvers that can be used in the OQNLP algorithm, including L-BFGS, Nelder-Mead, and Gradient Descent (argmin's implementations).
pub enum LocalSolverType {
    /// L-BFGS local solver
    ///
    /// Requires `CostFunction` and `Gradient`
    LBFGS,

    /// Nelder-Mead local solver
    ///
    /// Requires `CostFunction`
    NelderMead,

    /// Steepest Descent local solver
    ///
    /// Requires `CostFunction` and `Gradient`
    SteepestDescent,
}

#[derive(Debug, Error)]
/// Error type for function, gradient and hessian evaluation
pub enum EvaluationError {
    /// Error when the input is invalid
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Error when dividing by zero
    #[error("Division by zero found")]
    DivisionByZero,

    /// Error when having a negative square root
    #[error("Negative square root found")]
    NegativeSqrt,

    /// Error when the objective function is not implemented
    #[error("Objective function not implemented and needed for local solver")]
    ObjectiveFunctionNotImplemented,

    /// Error when the gradient is not implemented
    #[error("Gradient not implemented and needed for local solver")]
    GradientNotImplemented,

    /// Error when the hessian is not implemented
    #[error("Hessian not implemented and needed for local solver")]
    HessianNotImplemented,

    /// Error when the objective function can't be evaluated
    #[error("Objective function evaluation failed")]
    ObjectiveFunctionEvaluationFailed,

    /// Error when the gradient can't be evaluated
    #[error("Gradient evaluation failed")]
    GradientEvaluationFailed,

    /// Error when the hessian can't be evaluated
    #[error("Hessian evaluation failed")]
    HessianEvaluationFailed,
}
