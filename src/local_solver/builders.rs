//! # Local Solver Builders module.
//!
//! This module contains the builders.
//! The builders allow for the creation and configuration of the local solvers, including the L-BFGS, Nelder-Mead, and Steepest Descent methods.
//!
//! ## Example
//! ```rust
//! use globalsearch::local_solver::builders::{HagerZhangBuilder, LBFGSBuilder};
//!
//! // L-BFGS local solver configuration
//! let lbfgs = LBFGSBuilder::default()
//!             .max_iter(300)
//!             .tolerance_grad(1e-8)
//!             .build();
//!
//! // Hager-Zhang line search configuration
//! let hager_zhang = HagerZhangBuilder::default()
//!                     .delta(0.1)
//!                     .sigma(0.9)
//!                     .build();
//! ```
use ndarray::{array, Array1};

#[derive(Debug, Clone)]
/// Local solver configuration for the OQNLP algorithm
///
/// This enum defines the configuration options for the local solver used in the optimizer, depending on the method used.
pub enum LocalSolverConfig {
    LBFGS {
        /// Maximum number of iterations for the L-BFGS local solver
        max_iter: u64,
        /// Tolerance for the gradient
        tolerance_grad: f64,
        /// Tolerance for the cost function
        tolerance_cost: f64,
        /// Number of previous iterations to store in the history
        history_size: usize,

        // l1_regularization: f64, Should this be included? As bool? https://docs.rs/argmin/0.10.0/argmin/solver/quasinewton/struct.LBFGS.html
        /// Line search parameters for the L-BFGS local solver
        line_search_params: LineSearchParams,
    },
    NelderMead {
        /// Simplex delta
        ///
        /// Sets the step size for generating the simplex from a given point.
        ///
        /// We add the point as the first vertex of the simplex.
        /// Then, for each dimension of the point, we create a new point by cloning the initial point and
        /// then incrementing the value at the given index by the fixed offset, simplex_delta.
        /// This results in a simplex with one vertex for each coordinate direction offset from the initial point.
        ///
        /// The default value is 0.1.
        simplex_delta: f64,
        /// Sample standard deviation tolerance
        sd_tolerance: f64,
        /// Maximum number of iterations for the Nelder-Mead local solver
        max_iter: u64,
        /// Reflection coefficient
        alpha: f64,
        /// Expansion coefficient
        gamma: f64,
        /// Contraction coefficient
        rho: f64,
        /// Shrinkage coefficient
        sigma: f64,
    },
    SteepestDescent {
        /// Maximum number of iterations for the Steepest Descent local solver
        max_iter: u64,
        /// Line search parameters for the Steepest Descent local solver
        line_search_params: LineSearchParams,
    },
}

impl LocalSolverConfig {
    pub fn lbfgs() -> LBFGSBuilder {
        LBFGSBuilder::default()
    }

    pub fn neldermead() -> NelderMeadBuilder {
        NelderMeadBuilder::default()
    }

    pub fn steepestdescent() -> SteepestDescentBuilder {
        SteepestDescentBuilder::default()
    }
}

#[derive(Debug, Clone)]
/// L-BFGS builder struct
///
/// This struct allows for the configuration of the L-BFGS local solver.
pub struct LBFGSBuilder {
    max_iter: u64,
    tolerance_grad: f64,
    tolerance_cost: f64,
    history_size: usize,
    line_search_params: LineSearchParams,
}

/// L-BFGS Builder
///
/// This builder allows for the configuration of the L-BFGS local solver.
impl LBFGSBuilder {
    /// Build the L-BFGS local solver configuration
    pub fn build(self) -> LocalSolverConfig {
        LocalSolverConfig::LBFGS {
            max_iter: self.max_iter,
            tolerance_grad: self.tolerance_grad,
            tolerance_cost: self.tolerance_cost,
            history_size: self.history_size,
            line_search_params: self.line_search_params,
        }
    }

    /// Set the maximum number of iterations for the L-BFGS local solver
    pub fn max_iter(mut self, max_iter: u64) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the tolerance for the gradient
    pub fn tolerance_grad(mut self, tolerance_grad: f64) -> Self {
        self.tolerance_grad = tolerance_grad;
        self
    }

    /// Set the tolerance for the cost function
    pub fn tolerance_cost(mut self, tolerance_cost: f64) -> Self {
        self.tolerance_cost = tolerance_cost;
        self
    }

    /// Set the number of previous iterations to store in the history
    pub fn history_size(mut self, history_size: usize) -> Self {
        self.history_size = history_size;
        self
    }

    /// Set the line search parameters for the L-BFGS local solver
    pub fn line_search_params(mut self, line_search_params: LineSearchParams) -> Self {
        self.line_search_params = line_search_params;
        self
    }
}

/// Default implementation for the L-BFGS builder
///
/// This implementation sets the default values for the L-BFGS builder.
/// Default values:
/// - `max_iter`: 300
/// - `tolerance_grad`: sqrt(EPSILON)
/// - `tolerance_cost`: EPSILON
/// - `history_size`: 10
/// - `line_search_params`: Default LineSearchParams
impl Default for LBFGSBuilder {
    fn default() -> Self {
        LBFGSBuilder {
            max_iter: 300,
            tolerance_grad: f64::EPSILON.sqrt(),
            tolerance_cost: f64::EPSILON,
            history_size: 10,
            line_search_params: LineSearchParams::default(),
        }
    }
}

#[derive(Debug, Clone)]
/// Nelder-Mead builder struct
///
/// This struct allows for the configuration of the Nelder-Mead local solver, including the simplex delta, sample standard deviation tolerance, the reflection coefficient, the expansion coefficient, the contraction coefficient, and the shrinkage coefficient.
pub struct NelderMeadBuilder {
    simplex_delta: f64,
    sd_tolerance: f64,
    max_iter: u64,
    alpha: f64,
    gamma: f64,
    rho: f64,
    sigma: f64,
}

/// Nelder-Mead Builder
///
/// This builder allows for the configuration of the Nelder-Mead local solver.
impl NelderMeadBuilder {
    /// Build the Nelder-Mead local solver configuration
    pub fn build(self) -> LocalSolverConfig {
        LocalSolverConfig::NelderMead {
            simplex_delta: self.simplex_delta,
            sd_tolerance: self.sd_tolerance,
            max_iter: self.max_iter,
            alpha: self.alpha,
            gamma: self.gamma,
            rho: self.rho,
            sigma: self.sigma,
        }
    }

    /// Set the simplex delta parameter
    pub fn simplex_delta(mut self, simplex_delta: f64) -> Self {
        self.simplex_delta = simplex_delta;
        self
    }

    /// Set the sample standard deviation tolerance
    pub fn sd_tolerance(mut self, sd_tolerance: f64) -> Self {
        self.sd_tolerance = sd_tolerance;
        self
    }

    /// Set the maximum number of iterations for the Nelder-Mead local solver
    pub fn max_iter(mut self, max_iter: u64) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the reflection coefficient
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the expansion coefficient
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set the contraction coefficient
    pub fn rho(mut self, rho: f64) -> Self {
        self.rho = rho;
        self
    }

    /// Set the shrinkage coefficient
    pub fn sigma(mut self, sigma: f64) -> Self {
        self.sigma = sigma;
        self
    }
}

/// Default implementation for the Nelder-Mead builder
///
/// This implementation sets the default values for the Nelder-Mead builder.
/// Default values:
/// - `simplex_delta`: 0.1
/// - `sd_tolerance`: EPSILON
/// - `alpha`: 1.0
/// - `gamma`: 2.0
/// - `rho`: 0.5
/// - `sigma`: 0.5
impl Default for NelderMeadBuilder {
    fn default() -> Self {
        NelderMeadBuilder {
            simplex_delta: 0.1,
            sd_tolerance: f64::EPSILON,
            max_iter: 300,
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
        }
    }
}

#[derive(Debug, Clone)]
/// Steepest Descent builder struct
///
/// This struct allows for the configuration of the Steepest Descent local solver, including the maximum number of iterations and the line search parameters.
pub struct SteepestDescentBuilder {
    max_iter: u64,
    line_search_params: LineSearchParams,
}

/// Steepest Descent Builder
///
/// This builder allows for the configuration of the Steepest Descent local solver.
impl SteepestDescentBuilder {
    /// Build the Steepest Descent local solver configuration
    pub fn build(self) -> LocalSolverConfig {
        LocalSolverConfig::SteepestDescent {
            max_iter: self.max_iter,
            line_search_params: self.line_search_params,
        }
    }

    /// Set the maximum number of iterations for the Steepest Descent local solver
    pub fn max_iter(mut self, max_iter: u64) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the line search parameters for the Steepest Descent local solver
    pub fn line_search_params(mut self, line_search_params: LineSearchParams) -> Self {
        self.line_search_params = line_search_params;
        self
    }
}

/// Default implementation for the Steepest Descent builder
///
/// This implementation sets the default values for the Steepest Descent builder.
/// Default values:
/// - `max_iter`: 300
/// - `line_search_params`: Default LineSearchParams
impl Default for SteepestDescentBuilder {
    fn default() -> Self {
        SteepestDescentBuilder {
            max_iter: 300,
            line_search_params: LineSearchParams::default(),
        }
    }
}

#[derive(Debug, Clone)]
/// Line search methods for the local solver
///
/// This enum defines the types of line search methods that can be used in some of the local solver, including MoreThuente, HagerZhang, and Backtracking.
pub enum LineSearchMethod {
    MoreThuente {
        c1: f64,
        c2: f64,
        width_tolerance: f64,
        bounds: Array1<f64>,
    },
    HagerZhang {
        delta: f64,
        sigma: f64,
        epsilon: f64,
        theta: f64,
        gamma: f64,
        eta: f64,
        bounds: Array1<f64>,
    },
}

#[derive(Debug, Clone)]
/// Line search parameters for the local solver
///
/// This struct defines the parameters for the line search algorithm used in the local solver. It is only needed for the optimizers that use line search methods.
pub struct LineSearchParams {
    pub method: LineSearchMethod,
}

impl LineSearchParams {
    pub fn morethuente() -> MoreThuenteBuilder {
        MoreThuenteBuilder::default()
    }

    pub fn hagerzhang() -> HagerZhangBuilder {
        HagerZhangBuilder::default()
    }
}

/// Default implementation for the line search parameters
///
/// This implementation sets the default values for the line search parameters.
/// Default values:
/// - `c1`: 1e-4
/// - `c2`: 0.9
/// - `width_tolerance`: 1e-10
/// - `bounds`: [sqrt(EPSILON), INFINITY]
/// - `method`: MoreThuente
impl Default for LineSearchParams {
    fn default() -> Self {
        LineSearchParams {
            method: LineSearchMethod::MoreThuente {
                c1: 1e-4,
                c2: 0.9,
                width_tolerance: 1e-10,
                bounds: array![f64::EPSILON.sqrt(), f64::INFINITY],
            },
        }
    }
}

#[derive(Debug, Clone)]
/// More-Thuente builder struct
///
/// This struct allows for the configuration of the More-Thuente line search method, including the strong wolfe conditions parameter c1 and c2, the width tolerance, and the bounds.
pub struct MoreThuenteBuilder {
    c1: f64,
    c2: f64,
    width_tolerance: f64,
    bounds: Array1<f64>,
}

/// More-Thuente Builder
///
/// This builder allows for the configuration of the More-Thuente line search method.
impl MoreThuenteBuilder {
    /// Build the More-Thuente line search parameters
    pub fn build(self) -> LineSearchParams {
        LineSearchParams {
            method: LineSearchMethod::MoreThuente {
                c1: self.c1,
                c2: self.c2,
                width_tolerance: self.width_tolerance,
                bounds: self.bounds,
            },
        }
    }

    /// Set the strong Wolfe conditions parameter c1
    pub fn c1(mut self, c1: f64) -> Self {
        self.c1 = c1;
        self
    }

    /// Set the strong Wolfe conditions parameter c2
    pub fn c2(mut self, c2: f64) -> Self {
        self.c2 = c2;
        self
    }

    /// Set the width tolerance
    pub fn width_tolerance(mut self, width_tolerance: f64) -> Self {
        self.width_tolerance = width_tolerance;
        self
    }

    /// Set the bounds
    pub fn bounds(mut self, bounds: Array1<f64>) -> Self {
        self.bounds = bounds;
        self
    }
}

/// Default implementation for the More-Thuente builder
///
/// This implementation sets the default values for the More-Thuente builder.
/// Default values:
/// - `c1`: 1e-4
/// - `c2`: 0.9
/// - `width_tolerance`: 1e-10
/// - `bounds`: [sqrt(EPSILON), INFINITY]
impl Default for MoreThuenteBuilder {
    fn default() -> Self {
        MoreThuenteBuilder {
            c1: 1e-4,
            c2: 0.9,
            width_tolerance: 1e-10,
            bounds: array![f64::EPSILON.sqrt(), f64::INFINITY],
        }
    }
}

#[derive(Debug, Clone)]
/// Hager-Zhang builder struct
///
/// This struct allows for the configuration of the Hager-Zhang line search method, including delta, sigma, epsilon, theta, gamma, eta and the bounds.
pub struct HagerZhangBuilder {
    delta: f64,
    sigma: f64,
    epsilon: f64,
    theta: f64,
    gamma: f64,
    eta: f64,
    bounds: Array1<f64>,
}

/// Hager-Zhang Builder
///
/// This builder allows for the configuration of the Hager-Zhang line search method.
impl HagerZhangBuilder {
    /// Build the Hager-Zhang line search parameters
    pub fn build(self) -> LineSearchParams {
        LineSearchParams {
            method: LineSearchMethod::HagerZhang {
                delta: self.delta,
                sigma: self.sigma,
                epsilon: self.epsilon,
                theta: self.theta,
                gamma: self.gamma,
                eta: self.eta,
                bounds: self.bounds,
            },
        }
    }

    /// Set the delta parameter
    pub fn delta(mut self, delta: f64) -> Self {
        self.delta = delta;
        self
    }

    /// Set the sigma parameter
    pub fn sigma(mut self, sigma: f64) -> Self {
        self.sigma = sigma;
        self
    }

    /// Set the epsilon parameter
    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set the theta parameter
    pub fn theta(mut self, theta: f64) -> Self {
        self.theta = theta;
        self
    }

    /// Set the gamma parameter
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set the eta parameter
    pub fn eta(mut self, eta: f64) -> Self {
        self.eta = eta;
        self
    }

    /// Set the bounds
    pub fn bounds(mut self, bounds: Array1<f64>) -> Self {
        self.bounds = bounds;
        self
    }
}

/// Default implementation for the Hager-Zhang builder
///
/// This implementation sets the default values for the Hager-Zhang builder.
/// Default values:
/// - `c1`: 1e-4
/// - `c2`: 0.9
/// - `width_tolerance`: 1e-10
/// - `bounds`: [sqrt(EPSILON), INFINITY]
impl Default for HagerZhangBuilder {
    fn default() -> Self {
        HagerZhangBuilder {
            delta: 0.1,
            sigma: 0.9,
            epsilon: 1e-6,
            theta: 0.5,
            gamma: 0.66,
            eta: 0.01,
            bounds: array![f64::EPSILON, 1e5],
        }
    }
}
