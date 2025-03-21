//! # Local Solver Builders module.
//!
//! This module contains the builders.
//! The builders allow for the creation and configuration of the local solvers,
//! including the L-BFGS, Nelder-Mead, Steepest Descent and Trust Region methods.
//!
//! ## Example
//! ```rust
//! use globalsearch::local_solver::builders::{HagerZhangBuilder, LBFGSBuilder};
//!
//! // L-BFGS local solver configuration
//! let lbfgs = LBFGSBuilder::default()
//!             .max_iter(500)
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

#[derive(Debug, Clone, PartialEq)]
/// Trust Region Radius Method
///
/// This enum defines the types of trust region radius methods that can be
/// used in the Trust Region local solver, including Cauchy and Steihaug.
pub enum TrustRegionRadiusMethod {
    Cauchy,
    Steihaug,
}

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
    TrustRegion {
        /// Trust Region radius method to use to compute the step length and direction
        trust_region_radius_method: TrustRegionRadiusMethod,
        /// The maximum number of iterations for the Trust Region local solver
        max_iter: u64,
        /// The radius for the Trust Region local solver
        radius: f64,
        /// The maximum radius for the Trust Region local solver
        max_radius: f64,
        /// The parameter that determines the acceptance threshold for the trust region step
        ///
        /// Must lie in [0, 1/4) and defaults to 0.125
        eta: f64,
        // TODO: Steihaug's method can take with_epsilon, but Cauchy doesn't
        // Should we include it here?
        // TODO: Currently I don't set Dogleg as a method since it would require using linalg from
        // ndarray. If more methods use ArgminInv then it would be a good idea to switch to using linalg
        // and implement it
    },
    NewtonCG {
        /// Maximum number of iterations for the Newton local solver
        max_iter: u64,
        /// Curvature threshold
        ///
        /// The curvature threshold for the Newton-CG method. If the curvature is below this threshold,
        /// the step is considered to be a Newton step. The default value is 0.0.
        curvature_threshold: f64,
        /// Tolerance for the Newton-CG method
        tolerance: f64,
        /// Line search parameters for the Newton-CG method
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

    pub fn trustregion() -> TrustRegionBuilder {
        TrustRegionBuilder::default()
    }

    pub fn newton_cg() -> NewtonCGBuilder {
        NewtonCGBuilder::default()
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
    /// Create a new L-BFGS builder
    pub fn new(
        max_iter: u64,
        tolerance_grad: f64,
        tolerance_cost: f64,
        history_size: usize,
        line_search_params: LineSearchParams,
    ) -> Self {
        LBFGSBuilder {
            max_iter,
            tolerance_grad,
            tolerance_cost,
            history_size,
            line_search_params,
        }
    }

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
    /// Create a new Nelder-Mead builder
    pub fn new(
        simplex_delta: f64,
        sd_tolerance: f64,
        max_iter: u64,
        alpha: f64,
        gamma: f64,
        rho: f64,
        sigma: f64,
    ) -> Self {
        NelderMeadBuilder {
            simplex_delta,
            sd_tolerance,
            max_iter,
            alpha,
            gamma,
            rho,
            sigma,
        }
    }

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
/// - `max_iter`: 300
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
    /// Create a new Steepest Descent builder
    pub fn new(max_iter: u64, line_search_params: LineSearchParams) -> Self {
        SteepestDescentBuilder {
            max_iter,
            line_search_params,
        }
    }

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
/// Trust Region builder struct
///
/// This struct allows for the configuration of the Trust Region local solver.
pub struct TrustRegionBuilder {
    trust_region_radius_method: TrustRegionRadiusMethod,
    max_iter: u64,
    radius: f64,
    max_radius: f64,
    eta: f64,
}

/// Trust Region Builder
///
/// This builder allows for the configuration of the Trust Region local solver.
impl TrustRegionBuilder {
    /// Create a new Trust Region builder
    pub fn new(
        trust_region_radius_method: TrustRegionRadiusMethod,
        max_iter: u64,
        radius: f64,
        max_radius: f64,
        eta: f64,
    ) -> Self {
        TrustRegionBuilder {
            trust_region_radius_method,
            max_iter,
            radius,
            max_radius,
            eta,
        }
    }

    /// Build the Trust Region local solver configuration
    pub fn build(self) -> LocalSolverConfig {
        LocalSolverConfig::TrustRegion {
            trust_region_radius_method: self.trust_region_radius_method,
            max_iter: self.max_iter,
            radius: self.radius,
            max_radius: self.max_radius,
            eta: self.eta,
        }
    }

    /// Set the Trust Region Method for the Trust Region local solver
    pub fn method(mut self, method: TrustRegionRadiusMethod) -> Self {
        self.trust_region_radius_method = method;
        self
    }

    /// Set the maximum number of iterations for the Trust Region local solver
    pub fn max_iter(mut self, max_iter: u64) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the Trust Region radius for the Trust Region local solver
    pub fn radius(mut self, radius: f64) -> Self {
        self.radius = radius;
        self
    }

    /// Set the maximum Trust Region radius for the Trust Region local solver
    pub fn max_radius(mut self, max_radius: f64) -> Self {
        self.max_radius = max_radius;
        self
    }

    /// Set eta for the Trust Region local solver
    ///
    /// The parameter that determines the acceptance threshold for the trust region step.
    /// Must lie in [0, 1/4) and defaults to 0.125
    pub fn eta(mut self, eta: f64) -> Self {
        self.eta = eta;
        self
    }
}

/// Default implementation for the Trust Region builder
///
/// This implementation sets the default values for the Trust Region builder.
/// Default values:
/// - `trust_region_radius_method`: Cauchy
/// - `radius`: 1.0
/// - `max_radius`: 100.0
/// - `eta`: 0.125
impl Default for TrustRegionBuilder {
    fn default() -> Self {
        TrustRegionBuilder {
            trust_region_radius_method: TrustRegionRadiusMethod::Cauchy,
            max_iter: 300,
            radius: 1.0,
            max_radius: 100.0,
            eta: 0.125,
        }
    }
}

#[derive(Debug, Clone)]
/// Newton-CG builder struct
///
/// This struct allows for the configuration of the Newton-CG method local solver.
pub struct NewtonCGBuilder {
    max_iter: u64,
    curvature_threshold: f64,
    tolerance: f64,
    line_search_params: LineSearchParams,
}

/// Newton-CG method Builder
///
/// This builder allows for the configuration of the Newton-CG method local solver.
impl NewtonCGBuilder {
    /// Create a new Newton-CG builder
    pub fn new(
        max_iter: u64,
        curvature_threshold: f64,
        tolerance: f64,
        line_search_params: LineSearchParams,
    ) -> Self {
        NewtonCGBuilder {
            max_iter,
            curvature_threshold,
            tolerance,
            line_search_params,
        }
    }

    /// Build the Newton-CG method local solver configuration
    pub fn build(self) -> LocalSolverConfig {
        LocalSolverConfig::NewtonCG {
            max_iter: self.max_iter,
            curvature_threshold: self.curvature_threshold,
            tolerance: self.tolerance,
            line_search_params: self.line_search_params,
        }
    }

    /// Set the maximum number of iterations for the L-BFGS local solver
    pub fn max_iter(mut self, max_iter: u64) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the curvature threshold
    pub fn curvature_threshold(mut self, curvature_threshold: f64) -> Self {
        self.curvature_threshold = curvature_threshold;
        self
    }

    /// Set the tolerance
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set the line search parameters for the Newton-CG method local solver
    pub fn line_search_params(mut self, line_search_params: LineSearchParams) -> Self {
        self.line_search_params = line_search_params;
        self
    }
}

/// Default implementation for Newton-CG builder
///
/// This implementation sets the default values for Newton-CG builder.
/// Default values:
/// - `max_iter`: 300
/// - `curvature_threshold`: 0.0
/// - `tolerance`: EPSILON
/// - `line_search_params`: Default LineSearchParams
impl Default for NewtonCGBuilder {
    fn default() -> Self {
        NewtonCGBuilder {
            max_iter: 300,
            curvature_threshold: 0.0,
            tolerance: f64::EPSILON,
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
    /// Create a new More-Thuente builder
    pub fn new(c1: f64, c2: f64, width_tolerance: f64, bounds: Array1<f64>) -> Self {
        MoreThuenteBuilder {
            c1,
            c2,
            width_tolerance,
            bounds,
        }
    }

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
    /// Create a new Hager-Zhang builder
    pub fn new(
        delta: f64,
        sigma: f64,
        epsilon: f64,
        theta: f64,
        gamma: f64,
        eta: f64,
        bounds: Array1<f64>,
    ) -> Self {
        HagerZhangBuilder {
            delta,
            sigma,
            epsilon,
            theta,
            gamma,
            eta,
            bounds,
        }
    }

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
/// - `delta`: 0.1
/// - `sigma`: 0.9
/// - `epsilon`: 1e-6
/// - `theta`: 0.5
/// - `gamma`: 0.66
/// - `eta`: 0.01
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

#[cfg(test)]
mod tests_builders {
    use super::*;

    #[test]
    /// Test the default values for the L-BFGS builder
    ///
    /// The default values are:
    /// - `max_iter`: 300
    /// - `tolerance_grad`: sqrt(EPSILON)
    /// - `tolerance_cost`: EPSILON
    /// - `history_size`: 10
    /// - `line_search_params`: Default LineSearchParams
    fn test_default_lbfgs() {
        let lbfgs: LocalSolverConfig = LBFGSBuilder::default().build();
        match lbfgs {
            LocalSolverConfig::LBFGS {
                max_iter,
                tolerance_grad,
                tolerance_cost,
                history_size,
                line_search_params,
            } => {
                assert_eq!(max_iter, 300);
                assert_eq!(tolerance_grad, f64::EPSILON.sqrt());
                assert_eq!(tolerance_cost, f64::EPSILON);
                assert_eq!(history_size, 10);
                match line_search_params.method {
                    LineSearchMethod::MoreThuente {
                        c1,
                        c2,
                        width_tolerance,
                        bounds,
                    } => {
                        assert_eq!(c1, 1e-4);
                        assert_eq!(c2, 0.9);
                        assert_eq!(width_tolerance, 1e-10);
                        assert_eq!(bounds, array![f64::EPSILON.sqrt(), f64::INFINITY]);
                    }
                    _ => panic!("Expected MoreThuente line search method"),
                }
            }
            _ => panic!("Expected L-BFGS local solver"),
        }
    }

    #[test]
    /// Test the default values for the Nelder-Mead builder
    ///
    /// The default values are:
    /// - `simplex_delta`: 0.1
    /// - `sd_tolerance`: EPSILON
    /// - `alpha`: 1.0
    /// - `gamma`: 2.0
    /// - `rho`: 0.5
    /// - `sigma`: 0.5
    fn test_default_neldermead() {
        let neldermead: LocalSolverConfig = NelderMeadBuilder::default().build();
        match neldermead {
            LocalSolverConfig::NelderMead {
                simplex_delta,
                sd_tolerance,
                max_iter,
                alpha,
                gamma,
                rho,
                sigma,
            } => {
                assert_eq!(simplex_delta, 0.1);
                assert_eq!(sd_tolerance, f64::EPSILON);
                assert_eq!(max_iter, 300);
                assert_eq!(alpha, 1.0);
                assert_eq!(gamma, 2.0);
                assert_eq!(rho, 0.5);
                assert_eq!(sigma, 0.5);
            }
            _ => panic!("Expected Nelder-Mead local solver"),
        }
    }

    #[test]
    /// Test the default values for the Steepest Descent builder
    ///
    /// The default values are:
    /// - `max_iter`: 300
    /// - `line_search_params`: Default LineSearchParams
    fn test_default_steepestdescent() {
        let steepestdescent: LocalSolverConfig = SteepestDescentBuilder::default().build();
        match steepestdescent {
            LocalSolverConfig::SteepestDescent {
                max_iter,
                line_search_params,
            } => {
                assert_eq!(max_iter, 300);
                match line_search_params.method {
                    LineSearchMethod::MoreThuente {
                        c1,
                        c2,
                        width_tolerance,
                        bounds,
                    } => {
                        assert_eq!(c1, 1e-4);
                        assert_eq!(c2, 0.9);
                        assert_eq!(width_tolerance, 1e-10);
                        assert_eq!(bounds, array![f64::EPSILON.sqrt(), f64::INFINITY]);
                    }
                    _ => panic!("Expected MoreThuente line search method"),
                }
            }
            _ => panic!("Expected Steepest Descent local solver"),
        }
    }

    #[test]
    /// Test the default values for the Trust Region builder
    ///
    /// The default values are:
    /// - `trust_region_radius_method`: Cauchy
    /// - `radius`: 1.0
    /// - `max_radius`: 100.0
    /// - `eta`: 0.125
    fn test_default_trustregion() {
        let trustregion: LocalSolverConfig = TrustRegionBuilder::default().build();
        match trustregion {
            LocalSolverConfig::TrustRegion {
                trust_region_radius_method,
                max_iter,
                radius,
                max_radius,
                eta,
            } => {
                assert_eq!(trust_region_radius_method, TrustRegionRadiusMethod::Cauchy);
                assert_eq!(max_iter, 300);
                assert_eq!(radius, 1.0);
                assert_eq!(max_radius, 100.0);
                assert_eq!(eta, 0.125);
            }
            _ => panic!("Expected Trust Region local solver"),
        }
    }

    #[test]
    /// Test the default values for the Newton-CG builder
    ///
    /// The default values are:
    /// - `max_iter`: 300
    /// - `curvature_threshold`: 0.0
    /// - `tolerance`: EPSILON
    /// - `line_search_params`: Default LineSearchParams
    fn test_default_newton_cg() {
        let newtoncg: LocalSolverConfig = NewtonCGBuilder::default().build();
        match newtoncg {
            LocalSolverConfig::NewtonCG {
                max_iter,
                curvature_threshold,
                tolerance,
                line_search_params,
            } => {
                assert_eq!(max_iter, 300);
                assert_eq!(curvature_threshold, 0.0);
                assert_eq!(tolerance, f64::EPSILON);
                match line_search_params.method {
                    LineSearchMethod::MoreThuente {
                        c1,
                        c2,
                        width_tolerance,
                        bounds,
                    } => {
                        assert_eq!(c1, 1e-4);
                        assert_eq!(c2, 0.9);
                        assert_eq!(width_tolerance, 1e-10);
                        assert_eq!(bounds, array![f64::EPSILON.sqrt(), f64::INFINITY]);
                    }
                    _ => panic!("Expected MoreThuente line search method"),
                }
            }
            _ => panic!("Expected Newton-CG local solver"),
        }
    }

    /// Test the default values for the More-Thuente builder
    ///
    /// The default values are:
    /// - `c1`: 1e-4
    /// - `c2`: 0.9
    /// - `width_tolerance`: 1e-10
    /// - `bounds`: [sqrt(EPSILON), INFINITY]
    #[test]
    fn test_default_morethuente() {
        let morethuente: LineSearchParams = MoreThuenteBuilder::default().build();
        match morethuente.method {
            LineSearchMethod::MoreThuente {
                c1,
                c2,
                width_tolerance,
                bounds,
            } => {
                assert_eq!(c1, 1e-4);
                assert_eq!(c2, 0.9);
                assert_eq!(width_tolerance, 1e-10);
                assert_eq!(bounds, array![f64::EPSILON.sqrt(), f64::INFINITY]);
            }
            _ => panic!("Expected MoreThuente line search method"),
        }
    }

    #[test]
    /// Test the default values for the Hager-Zhang builder
    ///
    /// The default values are:
    /// - `delta`: 0.1
    /// - `sigma`: 0.9
    /// - `epsilon`: 1e-6
    /// - `theta`: 0.5
    /// - `gamma`: 0.66
    /// - `eta`: 0.01
    /// - `bounds`: [sqrt(EPSILON), 1e5]
    fn test_default_hagerzhang() {
        let hagerzhang: LineSearchParams = HagerZhangBuilder::default().build();
        match hagerzhang.method {
            LineSearchMethod::HagerZhang {
                delta,
                sigma,
                epsilon,
                theta,
                gamma,
                eta,
                bounds,
            } => {
                assert_eq!(delta, 0.1);
                assert_eq!(sigma, 0.9);
                assert_eq!(epsilon, 1e-6);
                assert_eq!(theta, 0.5);
                assert_eq!(gamma, 0.66);
                assert_eq!(eta, 0.01);
                assert_eq!(bounds, array![f64::EPSILON, 1e5]);
            }
            _ => panic!("Expected HagerZhang line search method"),
        }
    }

    #[test]
    /// Test changing the parameters of L-BFGS builder
    fn change_params_lbfgs() {
        let linesearch: LineSearchParams = MoreThuenteBuilder::default().c1(1e-5).c2(0.8).build();
        let lbfgs: LocalSolverConfig = LBFGSBuilder::default()
            .max_iter(500)
            .tolerance_grad(1e-8)
            .tolerance_cost(1e-8)
            .history_size(5)
            .line_search_params(linesearch)
            .build();
        match lbfgs {
            LocalSolverConfig::LBFGS {
                max_iter,
                tolerance_grad,
                tolerance_cost,
                history_size,
                line_search_params,
            } => {
                assert_eq!(max_iter, 500);
                assert_eq!(tolerance_grad, 1e-8);
                assert_eq!(tolerance_cost, 1e-8);
                assert_eq!(history_size, 5);
                match line_search_params.method {
                    LineSearchMethod::MoreThuente {
                        c1,
                        c2,
                        width_tolerance,
                        bounds,
                    } => {
                        assert_eq!(c1, 1e-5);
                        assert_eq!(c2, 0.8);
                        assert_eq!(width_tolerance, 1e-10);
                        assert_eq!(bounds, array![f64::EPSILON.sqrt(), f64::INFINITY]);
                    }
                    _ => panic!("Expected MoreThuente line search method"),
                }
            }
            _ => panic!("Expected L-BFGS local solver"),
        }
    }

    #[test]
    /// Test changing the parameters of Nelder-Mead builder
    fn change_params_neldermead() {
        let neldermead: LocalSolverConfig = NelderMeadBuilder::default()
            .simplex_delta(0.5)
            .sd_tolerance(1e-5)
            .max_iter(1000)
            .alpha(1.5)
            .gamma(3.0)
            .rho(0.6)
            .sigma(0.6)
            .build();
        match neldermead {
            LocalSolverConfig::NelderMead {
                simplex_delta,
                sd_tolerance,
                max_iter,
                alpha,
                gamma,
                rho,
                sigma,
            } => {
                assert_eq!(simplex_delta, 0.5);
                assert_eq!(sd_tolerance, 1e-5);
                assert_eq!(max_iter, 1000);
                assert_eq!(alpha, 1.5);
                assert_eq!(gamma, 3.0);
                assert_eq!(rho, 0.6);
                assert_eq!(sigma, 0.6);
            }
            _ => panic!("Expected Nelder-Mead local solver"),
        }
    }

    #[test]
    /// Test changing the parameters of Steepest Descent builder
    fn change_params_steepestdescent() {
        let linesearch: LineSearchParams = MoreThuenteBuilder::default().c1(1e-5).c2(0.8).build();
        let steepestdescent: LocalSolverConfig = SteepestDescentBuilder::default()
            .max_iter(500)
            .line_search_params(linesearch)
            .build();
        match steepestdescent {
            LocalSolverConfig::SteepestDescent {
                max_iter,
                line_search_params,
            } => {
                assert_eq!(max_iter, 500);
                match line_search_params.method {
                    LineSearchMethod::MoreThuente {
                        c1,
                        c2,
                        width_tolerance,
                        bounds,
                    } => {
                        assert_eq!(c1, 1e-5);
                        assert_eq!(c2, 0.8);
                        assert_eq!(width_tolerance, 1e-10);
                        assert_eq!(bounds, array![f64::EPSILON.sqrt(), f64::INFINITY]);
                    }
                    _ => panic!("Expected MoreThuente line search method"),
                }
            }
            _ => panic!("Expected Steepest Descent local solver"),
        }
    }

    #[test]
    /// Test changing the parameters of Trust Region builder
    fn change_params_trustregion() {
        let trustregion: LocalSolverConfig = TrustRegionBuilder::default()
            .method(TrustRegionRadiusMethod::Steihaug)
            .max_iter(500)
            .radius(2.0)
            .max_radius(200.0)
            .eta(0.1)
            .build();
        match trustregion {
            LocalSolverConfig::TrustRegion {
                trust_region_radius_method,
                max_iter,
                radius,
                max_radius,
                eta,
            } => {
                assert_eq!(
                    trust_region_radius_method,
                    TrustRegionRadiusMethod::Steihaug
                );
                assert_eq!(max_iter, 500);
                assert_eq!(radius, 2.0);
                assert_eq!(max_radius, 200.0);
                assert_eq!(eta, 0.1);
            }
            _ => panic!("Expected Trust Region local solver"),
        }
    }

    #[test]
    /// Test changing the parameters of Newton-CG builder
    fn change_params_newton_cg() {
        let linesearch: LineSearchParams = MoreThuenteBuilder::default().c1(1e-5).c2(0.8).build();
        let newtoncg: LocalSolverConfig = NewtonCGBuilder::default()
            .max_iter(500)
            .curvature_threshold(0.1)
            .tolerance(1e-7)
            .line_search_params(linesearch)
            .build();
        match newtoncg {
            LocalSolverConfig::NewtonCG {
                max_iter,
                curvature_threshold,
                tolerance,
                line_search_params,
            } => {
                assert_eq!(max_iter, 500);
                assert_eq!(curvature_threshold, 0.1);
                assert_eq!(tolerance, 1e-7);
                match line_search_params.method {
                    LineSearchMethod::MoreThuente {
                        c1,
                        c2,
                        width_tolerance,
                        bounds,
                    } => {
                        assert_eq!(c1, 1e-5);
                        assert_eq!(c2, 0.8);
                        assert_eq!(width_tolerance, 1e-10);
                        assert_eq!(bounds, array![f64::EPSILON.sqrt(), f64::INFINITY]);
                    }
                    _ => panic!("Expected MoreThuente line search method"),
                }
            }
            _ => panic!("Expected Newton-CG local solver"),
        }
    }

    #[test]
    /// Test changing the parameters of More-Thuente builder
    fn change_params_morethuente() {
        let morethuente: LineSearchParams = MoreThuenteBuilder::default()
            .c1(1e-5)
            .c2(0.8)
            .width_tolerance(1e-8)
            .bounds(array![1e-5, 1e5])
            .build();
        match morethuente.method {
            LineSearchMethod::MoreThuente {
                c1,
                c2,
                width_tolerance,
                bounds,
            } => {
                assert_eq!(c1, 1e-5);
                assert_eq!(c2, 0.8);
                assert_eq!(width_tolerance, 1e-8);
                assert_eq!(bounds, array![1e-5, 1e5]);
            }
            _ => panic!("Expected MoreThuente line search method"),
        }
    }

    #[test]
    /// Test changing the parameters of Hager-Zhang builder
    fn change_params_hagerzhang() {
        let hagerzhang = HagerZhangBuilder::default()
            .delta(0.2)
            .sigma(0.8)
            .epsilon(1e-7)
            .theta(0.6)
            .gamma(0.7)
            .eta(0.05)
            .bounds(array![1e-6, 1e6])
            .build();

        match hagerzhang.method {
            LineSearchMethod::HagerZhang {
                delta,
                sigma,
                epsilon,
                theta,
                gamma,
                eta,
                bounds,
            } => {
                assert_eq!(delta, 0.2);
                assert_eq!(sigma, 0.8);
                assert_eq!(epsilon, 1e-7);
                assert_eq!(theta, 0.6);
                assert_eq!(gamma, 0.7);
                assert_eq!(eta, 0.05);
                assert_eq!(bounds, array![1e-6, 1e6]);
            }
            _ => panic!("Expected HagerZhang line search method"),
        }
    }

    #[test]
    /// Test creating a LBFGSdBuilder using new()
    fn test_lbfgs_new() {
        let ls = LineSearchParams::morethuente().c1(1e-5).c2(0.8).build();
        let lbfgs = LBFGSBuilder::new(500, 1e-8, 1e-8, 5, ls).build();
        match lbfgs {
            LocalSolverConfig::LBFGS {
                max_iter,
                tolerance_grad,
                tolerance_cost,
                history_size,
                line_search_params,
            } => {
                assert_eq!(max_iter, 500);
                assert_eq!(tolerance_grad, 1e-8);
                assert_eq!(tolerance_cost, 1e-8);
                assert_eq!(history_size, 5);
                match line_search_params.method {
                    LineSearchMethod::MoreThuente {
                        c1,
                        c2,
                        width_tolerance,
                        bounds,
                    } => {
                        assert_eq!(c1, 1e-5);
                        assert_eq!(c2, 0.8);
                        assert_eq!(width_tolerance, 1e-10);
                        assert_eq!(bounds, array![f64::EPSILON.sqrt(), f64::INFINITY]);
                    }
                    _ => panic!("Expected MoreThuente line search method"),
                }
            }
            _ => panic!("Expected L-BFGS local solver"),
        }
    }

    #[test]
    /// Test creating a NelderMeadBuilder using new()
    fn test_neldermead_new() {
        let nm = NelderMeadBuilder::new(0.5, 1e-5, 1000, 1.5, 3.0, 0.6, 0.6).build();
        match nm {
            LocalSolverConfig::NelderMead {
                simplex_delta,
                sd_tolerance,
                max_iter,
                alpha,
                gamma,
                rho,
                sigma,
            } => {
                assert_eq!(simplex_delta, 0.5);
                assert_eq!(sd_tolerance, 1e-5);
                assert_eq!(max_iter, 1000);
                assert_eq!(alpha, 1.5);
                assert_eq!(gamma, 3.0);
                assert_eq!(rho, 0.6);
                assert_eq!(sigma, 0.6);
            }
            _ => panic!("Expected Nelder-Mead local solver"),
        }
    }

    #[test]
    /// Test creating a SteepestDescentBuilder using new()
    fn test_steepestdescent_new() {
        let ls = LineSearchParams::morethuente().c1(1e-5).c2(0.8).build();
        let sd = SteepestDescentBuilder::new(500, ls).build();
        match sd {
            LocalSolverConfig::SteepestDescent {
                max_iter,
                line_search_params,
            } => {
                assert_eq!(max_iter, 500);
                match line_search_params.method {
                    LineSearchMethod::MoreThuente {
                        c1,
                        c2,
                        width_tolerance,
                        bounds,
                    } => {
                        assert_eq!(c1, 1e-5);
                        assert_eq!(c2, 0.8);
                        assert_eq!(width_tolerance, 1e-10);
                        assert_eq!(bounds, array![f64::EPSILON.sqrt(), f64::INFINITY]);
                    }
                    _ => panic!("Expected MoreThuente line search method"),
                }
            }
            _ => panic!("Expected Steepest Descent local solver"),
        }
    }

    #[test]
    /// Test creating a TrustRegionBuilder using new()
    fn test_trustregion_new() {
        let tr = TrustRegionBuilder::new(TrustRegionRadiusMethod::Steihaug, 500, 2.0, 200.0, 0.1)
            .build();
        match tr {
            LocalSolverConfig::TrustRegion {
                trust_region_radius_method,
                max_iter,
                radius,
                max_radius,
                eta,
            } => {
                assert_eq!(
                    trust_region_radius_method,
                    TrustRegionRadiusMethod::Steihaug
                );
                assert_eq!(max_iter, 500);
                assert_eq!(radius, 2.0);
                assert_eq!(max_radius, 200.0);
                assert_eq!(eta, 0.1);
            }
            _ => panic!("Expected Trust Region local solver"),
        }
    }

    #[test]
    /// Test creating a NewtonCGBuilder using new()
    fn test_newtoncg_new() {
        let ls = LineSearchParams::morethuente().c1(1e-5).c2(0.8).build();
        let ncg = NewtonCGBuilder::new(500, 0.1, 1e-7, ls).build();
        match ncg {
            LocalSolverConfig::NewtonCG {
                max_iter,
                curvature_threshold,
                tolerance,
                line_search_params,
            } => {
                assert_eq!(max_iter, 500);
                assert_eq!(curvature_threshold, 0.1);
                assert_eq!(tolerance, 1e-7);
                match line_search_params.method {
                    LineSearchMethod::MoreThuente {
                        c1,
                        c2,
                        width_tolerance,
                        bounds,
                    } => {
                        assert_eq!(c1, 1e-5);
                        assert_eq!(c2, 0.8);
                        assert_eq!(width_tolerance, 1e-10);
                        assert_eq!(bounds, array![f64::EPSILON.sqrt(), f64::INFINITY]);
                    }
                    _ => panic!("Expected MoreThuente line search method"),
                }
            }
            _ => panic!("Expected Newton-CG local solver"),
        }
    }

    #[test]
    /// Test creating a MoreThuenteBuilder using new()
    fn test_morethuente_new() {
        let mt = MoreThuenteBuilder::new(1e-5, 0.8, 1e-8, array![1e-5, 1e5]).build();
        match mt.method {
            LineSearchMethod::MoreThuente {
                c1,
                c2,
                width_tolerance,
                bounds,
            } => {
                assert_eq!(c1, 1e-5);
                assert_eq!(c2, 0.8);
                assert_eq!(width_tolerance, 1e-8);
                assert_eq!(bounds, array![1e-5, 1e5]);
            }
            _ => panic!("Expected MoreThuente line search method"),
        }
    }

    #[test]
    /// Test creating a HagerZhangBuilder using new()
    fn test_hagerzhang_new() {
        let hz = HagerZhangBuilder::new(0.2, 0.8, 1e-7, 0.6, 0.7, 0.05, array![1e-6, 1e6]).build();
        match hz.method {
            LineSearchMethod::HagerZhang {
                delta,
                sigma,
                epsilon,
                theta,
                gamma,
                eta,
                bounds,
            } => {
                assert_eq!(delta, 0.2);
                assert_eq!(sigma, 0.8);
                assert_eq!(epsilon, 1e-7);
                assert_eq!(theta, 0.6);
                assert_eq!(gamma, 0.7);
                assert_eq!(eta, 0.05);
                assert_eq!(bounds, array![1e-6, 1e6]);
            }
            _ => panic!("Expected HagerZhang line search method"),
        }
    }
}
