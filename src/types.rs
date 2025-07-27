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

#[cfg(feature = "checkpointing")]
use std::path::PathBuf;

// TODO: Implement SR1 when it is fixed in argmin (https://github.com/argmin-rs/argmin/issues/221)
// Or add it now and print a warning that it is not working as expected in some cases

#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "checkpointing",
    derive(serde::Serialize, serde::Deserialize)
)]
/// Parameters for the OQNLP algorithm
///
/// These parameters control the behavior of the optimization process,
/// including the total number of iterations, the population size
/// the wait cycle, threshold factor, distance factor, local solver type
/// and settings and the seed.
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
#[cfg_attr(
    feature = "checkpointing",
    derive(serde::Serialize, serde::Deserialize)
)]
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
#[cfg_attr(
    feature = "checkpointing",
    derive(serde::Serialize, serde::Deserialize)
)]
/// A local solution in the parameter space
///
/// This struct represents a solution point in the parameter space along with the objective function value at that point.
pub struct LocalSolution {
    /// The solution point in the parameter space
    pub point: Array1<f64>,
    /// The objective function value at the solution point
    pub objective: f64,
}

impl LocalSolution {
    /// Returns the objective function value (f64) at the solution point
    ///
    /// Same as `objective` field
    ///
    /// This method is similar to the `fun` method in `SciPy.optimize` result
    pub fn fun(&self) -> f64 {
        self.objective
    }

    /// Returns the solution point (`Array1<f64>`) in the parameter space
    ///
    /// Same as `point` field
    /// Returns a clone of the point to avoid moving it
    ///
    /// This method is similar to the `x` method in `SciPy.optimize` result
    pub fn x(&self) -> Array1<f64> {
        self.point.clone()
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "checkpointing",
    derive(serde::Serialize, serde::Deserialize)
)]
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

    /// Returns the best solution in the set based on the objective function value.
    pub fn best_solution(&self) -> Option<&LocalSolution> {
        self.solutions
            .iter()
            .min_by(|a, b| a.objective.partial_cmp(&b.objective).unwrap())
    }

    /// Returns an iterator over the solutions in the set.
    pub fn solutions(&self) -> impl Iterator<Item = &LocalSolution> {
        self.solutions.iter()
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
            if let Some(best) = self.best_solution() {
                writeln!(f, "Best objective value: {:.8e}", best.objective)?;
            }
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

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "checkpointing",
    derive(serde::Serialize, serde::Deserialize)
)]
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

    /// Trust Region local solver
    ///
    /// Requires `CostFunction`, `Gradient` and `Hessian`
    TrustRegion,

    /// Newton-Conjugate-Gradient method local solver
    ///
    /// Requires `CostFunction`, `Gradient` and `Hessian`
    NewtonCG,
}

impl LocalSolverType {
    /// Returns the local solver type from a string
    ///
    /// This method is used to convert a string to a `LocalSolverType` enum.
    /// It is used to set the local solver type for the Python bindings.
    pub fn from_string(s: &str) -> Result<Self, &'static str> {
        match s.to_lowercase().as_str() {
            "lbfgs" => Ok(Self::LBFGS),
            "nelder-mead" => Ok(Self::NelderMead),
            "neldermead" => Ok(Self::NelderMead),
            "steepestdescent" => Ok(Self::SteepestDescent),
            "trustregion" => Ok(Self::TrustRegion),
            "newton-cg" => Ok(Self::NewtonCG),
            "newtoncg" => Ok(Self::NewtonCG),
            _ => Err("Invalid solver type."),
        }
    }
}

#[derive(Debug, Error)]
/// Error type for function, gradient and hessian evaluation
pub enum EvaluationError {
    /// Error when the input is invalid
    #[error("Invalid input: {0}.")]
    InvalidInput(String),

    /// Error when dividing by zero
    #[error("Division by zero found.")]
    DivisionByZero,

    /// Error when having a negative square root
    #[error("Negative square root found.")]
    NegativeSqrt,

    /// Error when the objective function is not implemented
    #[error("Objective function not implemented and needed for local solver.")]
    ObjectiveFunctionNotImplemented,

    /// Error when the gradient is not implemented
    #[error("Gradient not implemented and needed for local solver.")]
    GradientNotImplemented,

    /// Error when the hessian is not implemented
    #[error("Hessian not implemented and needed for local solver.")]
    HessianNotImplemented,

    /// Error when the objective function can't be evaluated
    #[error("Objective function evaluation failed.")]
    ObjectiveFunctionEvaluationFailed,

    /// Error when the gradient can't be evaluated
    #[error("Gradient evaluation failed.")]
    GradientEvaluationFailed,

    /// Error when the hessian can't be evaluated
    #[error("Hessian evaluation failed.")]
    HessianEvaluationFailed,
}

#[cfg(feature = "checkpointing")]
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "checkpointing",
    derive(serde::Serialize, serde::Deserialize)
)]
/// Configuration for checkpointing behavior
///
/// This struct defines the configuration for checkpointing behavior in the OQNLP algorithm.
/// It includes the directory where checkpoint files will be saved, the base name for checkpoint files,
/// the frequency of checkpointing, whether to keep all checkpoints or only the latest one,
/// and whether to automatically resume from the latest checkpoint if found.
pub struct CheckpointConfig {
    /// Directory where checkpoint files will be saved
    pub checkpoint_dir: PathBuf,

    /// Base name for checkpoint files (without extension)
    pub checkpoint_name: String,

    /// Frequency of checkpointing (every N iterations)
    pub save_frequency: usize,

    /// Whether to keep all checkpoints or only the latest one
    pub keep_all: bool,

    /// Whether to automatically resume from the latest checkpoint if found
    pub auto_resume: bool,
}

#[cfg(feature = "checkpointing")]
impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            checkpoint_dir: PathBuf::from("./checkpoints"),
            checkpoint_name: "oqnlp_checkpoint".to_string(),
            save_frequency: 25,
            keep_all: false,
            auto_resume: true,
        }
    }
}

#[cfg(feature = "checkpointing")]
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "checkpointing",
    derive(serde::Serialize, serde::Deserialize)
)]
/// Complete OQNLP state that can be saved and restored
///
/// This struct represents the complete state of the OQNLP algorithm at a given point in time,
/// including the algorithm parameters, current iteration, merit threshold, solution set,
/// current reference set from scatter search, number of unchanged cycles,
/// elapsed time, distance filter solutions for maintaining diversity, current seed value,
/// timestamp of the checkpoint, and other relevant information.
/// It is used to save the state of the algorithm to a file and restore it later.
pub struct OQNLPCheckpoint {
    /// Algorithm parameters
    pub params: OQNLPParams,

    /// Current iteration number
    pub current_iteration: usize,

    /// Current threshold value for merit filter
    pub merit_threshold: f64,

    /// Current solution set (if any)
    pub solution_set: Option<SolutionSet>,

    /// Current reference set from scatter search
    #[cfg_attr(
        feature = "checkpointing",
        serde(
            serialize_with = "serialize_vec_array1",
            deserialize_with = "deserialize_vec_array1"
        )
    )]
    pub reference_set: Vec<Array1<f64>>,

    /// Number of unchanged cycles
    pub unchanged_cycles: usize,

    /// Elapsed time in seconds
    pub elapsed_time: f64,

    /// Distance filter solutions for maintaining diversity
    pub distance_filter_solutions: Vec<LocalSolution>,

    /// Current seed value for continuing RNG sequence
    pub current_seed: u64,

    /// Timestamp of the checkpoint
    pub timestamp: String,
}

#[cfg(feature = "checkpointing")]
/// Serializes a vector of `Array1<f64>` to a format suitable for serialization
fn serialize_vec_array1<S>(vec: &Vec<Array1<f64>>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::ser::SerializeSeq;
    let mut seq = serializer.serialize_seq(Some(vec.len()))?;
    for array in vec {
        let vec_data: Vec<f64> = array.to_vec();
        seq.serialize_element(&vec_data)?;
    }
    seq.end()
}

#[cfg(feature = "checkpointing")]
/// Deserializes a vector of `Array1<f64>` from a format suitable for serialization
fn deserialize_vec_array1<'de, D>(deserializer: D) -> Result<Vec<Array1<f64>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    let vec_of_vecs: Vec<Vec<f64>> = Vec::deserialize(deserializer)?;
    Ok(vec_of_vecs.into_iter().map(Array1::from).collect())
}

#[cfg(feature = "checkpointing")]
/// Implements the Display trait for OQNLPCheckpoint
///
/// This trait provides a formatted string representation of the OQNLP checkpoint,
/// including the timestamp, current iteration, elapsed time, unchanged cycles,
/// merit threshold, reference set size, distance filter solutions,
/// current seed value, and parameters.
impl fmt::Display for OQNLPCheckpoint {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "━━━━━━━━━━━ OQNLP Checkpoint ━━━━━━━━━━━")?;
        writeln!(f, "Timestamp: {}", self.timestamp)?;
        writeln!(f, "Current iteration: {}", self.current_iteration)?;
        writeln!(f, "Elapsed time: {:.2}s", self.elapsed_time)?;
        writeln!(f, "Unchanged cycles: {}", self.unchanged_cycles)?;
        writeln!(f, "Merit threshold: {:.8e}", self.merit_threshold)?;
        writeln!(f, "Reference set size: {}", self.reference_set.len())?;
        writeln!(
            f,
            "Distance filter solutions: {}",
            self.distance_filter_solutions.len()
        )?;
        writeln!(f, "Current seed: {}", self.current_seed)?;

        if let Some(ref solution_set) = self.solution_set {
            writeln!(f, "Solution set: {} solutions", solution_set.len())?;
            if let Some(best) = solution_set.best_solution() {
                writeln!(f, "Best objective: {:.8e}", best.objective)?;
            }
        } else {
            writeln!(f, "Solution set: None")?;
        }

        writeln!(f, "Parameters:")?;
        writeln!(f, "  Population size: {}", self.params.population_size)?;
        writeln!(f, "  Iterations: {}", self.params.iterations)?;
        writeln!(f, "  Wait cycle: {}", self.params.wait_cycle)?;
        writeln!(f, "  Threshold factor: {}", self.params.threshold_factor)?;
        writeln!(f, "  Distance factor: {}", self.params.distance_factor)?;
        writeln!(f, "  Local solver: {:?}", self.params.local_solver_type)?;
        writeln!(f, "  Seed: {}", self.params.seed)?;
        writeln!(f, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")?;

        Ok(())
    }
}

#[cfg(test)]
mod tests_types {
    use super::*;
    use ndarray::array;

    #[test]
    /// Test the default parameters for the OQNLP algorithm
    fn test_oqnlp_params_default() {
        let params = OQNLPParams::default();
        assert_eq!(params.iterations, 300);
        assert_eq!(params.population_size, 1000);
        assert_eq!(params.wait_cycle, 15);
        assert_eq!(params.threshold_factor, 0.2);
        assert_eq!(params.distance_factor, 0.75);
        assert_eq!(params.seed, 0);
    }

    #[test]
    /// Test the len method for the SolutionSet struct
    fn test_solution_set_len() {
        let solutions = Array1::from_vec(vec![
            LocalSolution {
                point: array![1.0, 2.0],
                objective: -1.0,
            },
            LocalSolution {
                point: array![3.0, 4.0],
                objective: -2.0,
            },
        ]);
        let solution_set: SolutionSet = SolutionSet { solutions };
        assert_eq!(solution_set.len(), 2);
    }

    #[test]
    /// Test the is_empty method for the SolutionSet struct
    fn test_solution_set_is_empty() {
        let solutions: Array1<LocalSolution> = Array1::from_vec(vec![]);
        let solution_set: SolutionSet = SolutionSet { solutions };
        assert!(solution_set.is_empty());

        let solutions: Array1<LocalSolution> = Array1::from_vec(vec![LocalSolution {
            point: array![1.0],
            objective: -1.0,
        }]);
        let solution_set: SolutionSet = SolutionSet { solutions };
        assert!(!solution_set.is_empty());
    }

    #[test]
    /// Test indexing into the SolutionSet struct
    fn test_solution_set_index() {
        let solutions: Array1<LocalSolution> = Array1::from_vec(vec![
            LocalSolution {
                point: array![1.0, 2.0],
                objective: -1.0,
            },
            LocalSolution {
                point: array![3.0, 4.0],
                objective: -2.0,
            },
        ]);
        let solution_set: SolutionSet = SolutionSet { solutions };

        assert_eq!(solution_set[0].objective, -1.0);
        assert_eq!(solution_set[1].objective, -2.0);
    }

    #[test]
    /// Test the Display trait for the SolutionSet struct
    fn test_solution_set_display() {
        let solutions: Array1<LocalSolution> = Array1::from_vec(vec![LocalSolution {
            point: array![1.0],
            objective: -1.0,
        }]);
        let solution_set: SolutionSet = SolutionSet { solutions };

        println!("{}", solution_set);

        let display_output: String = format!("{}", solution_set);
        assert!(display_output.contains("Solution Set"));
        assert!(display_output.contains("Total solutions: 1"));
        assert!(display_output.contains("Best objective value"));
        assert!(display_output.contains("Solution #1"));
    }

    #[test]
    /// Test the display of empty solution set
    fn test_empty_solution_set_display() {
        let solutions: Array1<LocalSolution> = Array1::from_vec(vec![]);
        let solution_set: SolutionSet = SolutionSet { solutions };

        let display_output: String = format!("{}", solution_set);
        assert!(display_output.contains("Solution Set"));
        assert!(display_output.contains("Total solutions: 0"));
    }

    #[test]
    #[should_panic]
    fn test_solution_set_index_out_of_bounds() {
        let solutions: Array1<LocalSolution> = Array1::from_vec(vec![]);
        let solution_set: SolutionSet = SolutionSet { solutions };
        let _should_panic: LocalSolution = solution_set[0].clone();
    }

    #[test]
    /// Test the from_string method for the LocalSolverType enum
    fn test_local_solver_type_from_string() {
        assert_eq!(
            LocalSolverType::from_string("LBFGS"),
            Ok(LocalSolverType::LBFGS)
        );
        assert_eq!(
            LocalSolverType::from_string("Nelder-Mead"),
            Ok(LocalSolverType::NelderMead)
        );
        assert_eq!(
            LocalSolverType::from_string("SteepestDescent"),
            Ok(LocalSolverType::SteepestDescent)
        );
        assert_eq!(
            LocalSolverType::from_string("TrustRegion"),
            Ok(LocalSolverType::TrustRegion)
        );
        assert_eq!(
            LocalSolverType::from_string("NewtonCG"),
            Ok(LocalSolverType::NewtonCG)
        );
        assert_eq!(
            LocalSolverType::from_string("Invalid"),
            Err("Invalid solver type.")
        );
    }

    #[test]
    /// Test f() and x() methods from LocalSolution
    fn test_local_solution_f_x() {
        let local_solution = LocalSolution {
            point: array![1.0],
            objective: -1.0,
        };

        assert_eq!(local_solution.fun(), -1.0);
        assert_eq!(local_solution.x(), array![1.0]);
    }

    #[test]
    /// Test best_solution from SolutionSet
    fn test_solution_set_best_solution() {
        let solutions: Array1<LocalSolution> = Array1::from_vec(vec![
            LocalSolution {
                point: array![1.0],
                objective: -1.0,
            },
            LocalSolution {
                point: array![2.0],
                objective: -1.0,
            },
            LocalSolution {
                point: array![3.0],
                objective: -1.0,
            },
        ]);
        let solution_set: SolutionSet = SolutionSet { solutions };

        let best_solution = solution_set.best_solution().unwrap();
        assert_eq!(best_solution.objective, -1.0);
    }

    #[cfg(feature = "checkpointing")]
    #[test]
    /// Test the Display trait for OQNLPCheckpoint
    fn test_oqnlp_checkpoint_display() {
        let solution_set = SolutionSet {
            solutions: Array1::from_vec(vec![
                LocalSolution {
                    point: array![1.0, 2.0],
                    objective: -1.5,
                },
                LocalSolution {
                    point: array![3.0, 4.0],
                    objective: -2.0,
                },
            ]),
        };

        let checkpoint = OQNLPCheckpoint {
            params: OQNLPParams {
                iterations: 100,
                population_size: 500,
                wait_cycle: 20,
                threshold_factor: 0.3,
                distance_factor: 0.8,
                seed: 42,
                local_solver_type: LocalSolverType::LBFGS,
                local_solver_config: crate::local_solver::builders::LBFGSBuilder::default().build(),
            },
            current_iteration: 50,
            merit_threshold: 1.25,
            solution_set: Some(solution_set),
            reference_set: vec![array![1.0, 2.0], array![3.0, 4.0]],
            unchanged_cycles: 5,
            elapsed_time: 123.45,
            distance_filter_solutions: vec![],
            current_seed: 42,
            timestamp: "2025-07-27T12:00:00Z".to_string(),
        };

        let display_output = format!("{}", checkpoint);

        // Check that display output contains expected elements
        assert!(display_output.contains("OQNLP Checkpoint"));
        assert!(display_output.contains("2025-07-27T12:00:00Z"));
        assert!(display_output.contains("Current iteration: 50"));
        assert!(display_output.contains("Merit threshold: 1.25"));
        assert!(display_output.contains("Solution set: 2 solutions"));
        assert!(display_output.contains("Best objective: -2"));
        assert!(display_output.contains("Reference set size: 2"));
        assert!(display_output.contains("Unchanged cycles: 5"));
        assert!(display_output.contains("Elapsed time: 123.45s"));
        assert!(display_output.contains("Population size: 500"));
        assert!(display_output.contains("Wait cycle: 20"));
        assert!(display_output.contains("Local solver: LBFGS"));
    }

    #[cfg(feature = "checkpointing")]
    #[test]
    /// Test the Display trait for OQNLPCheckpoint with no solutions
    fn test_oqnlp_checkpoint_display_no_solutions() {
        let checkpoint = OQNLPCheckpoint {
            params: OQNLPParams::default(),
            current_iteration: 10,
            merit_threshold: f64::INFINITY,
            solution_set: None,
            reference_set: vec![],
            unchanged_cycles: 0,
            elapsed_time: 15.5,
            distance_filter_solutions: vec![],
            current_seed: 0,
            timestamp: "2025-07-27T10:00:00Z".to_string(),
        };

        let display_output = format!("{}", checkpoint);

        // Check that display output handles no solutions case
        assert!(display_output.contains("OQNLP Checkpoint"));
        assert!(display_output.contains("2025-07-27T10:00:00Z"));
        assert!(display_output.contains("Current iteration: 10"));
        assert!(display_output.contains("Solution set: None"));
        assert!(display_output.contains("Reference set size: 0"));
    }
}
