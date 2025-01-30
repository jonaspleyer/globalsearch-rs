use ndarray::Array1;

// TODO: Implement local solver options

// TODO: Implement SR1 when it is fixed in argmin (https://github.com/argmin-rs/argmin/issues/221)

// TODO: Implement methods with Hessians (How do we get the Hessian and how do we specify when we need it and when not?)

#[derive(Debug, Clone)]
/// Local solver implementation types for the OQNLP algorithm
///
/// This enum defines the types of local solvers that can be used in the OQNLP algorithm, including L-BFGS, Nelder-Mead, and Gradient Descent (argmin's implementations).
pub enum LocalSolverType {
    LBFGS,
    NelderMead,
    SteepestDescent,
}

#[derive(Debug, Clone)]
/// Parameters for the OQNLP algorithm
///
/// These parameters control the behavior of the optimization process, including the total number of iterations,
/// the number of iterations for stage 1, the wait cycle, threshold factor, distance factor, and population size.
pub struct OQNLPParams {
    /// Total number of iterations for the optimization process
    ///
    /// The main loop is defined as `total_iterations - stage_1_iterations`
    pub total_iterations: usize,
    /// Number of iterations for stage 1
    pub stage_1_iterations: usize,
    /// Number of iterations to wait before updating the threshold criteria
    pub wait_cycle: usize,
    /// Threshold factor
    pub threshold_factor: f64,
    /// Factor that influences the minimum required distance between candidate solutions
    pub distance_factor: f64,
    /// Number of candidate solutions considered in each iteration (population size)
    pub population_size: usize,
    /// Type of local solver to use from argmin
    pub solver_type: LocalSolverType,
}

#[derive(Debug, Clone)]
/// Parameters for the filtering mechanisms
///
/// These parameters control the behavior of the filtering mechanisms, including the distance factor, wait cycle, and threshold factor.
/// The distance factor influences the minimum required distance between candidate solutions, the wait cycle determines the number of iterations to wait before updating the threshold criteria, and the threshold factor is used to update the threshold criteria.
pub struct FilterParams {
    /// Factor that influences the minimum required distance between candidate solutions
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

pub type Result<T> = anyhow::Result<T>;
