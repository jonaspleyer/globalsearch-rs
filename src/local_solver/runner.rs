//! # Local Solver Runner module
//!
//! This module contains the implementation of the local solver runner, which is used to solve optimization problems locally.
//!
//! The local solver runner uses the `argmin` crate to run the local solvers.
//!
//! ## Local Solvers
//!
//! The local solvers currently supported are:
//!  - L-BFGS: Requires gradient and linesearch
//!  - Nelder-Mead: Only requires the objective function
//!  - Steepest Descent: Requires gradient and linesearch
//!  - Trust Region: Requires gradient and hessian

use crate::local_solver::builders::{LineSearchMethod, LocalSolverConfig, TrustRegionRadiusMethod};
use crate::problem::Problem;
use crate::types::{LocalSolution, LocalSolverType};
use argmin::core::{CostFunction, Error, Executor, Gradient, Hessian};
use argmin::solver::{
    gradientdescent::SteepestDescent,
    linesearch::{HagerZhangLineSearch, MoreThuenteLineSearch},
    neldermead::NelderMead,
    quasinewton::LBFGS,
    trustregion::{CauchyPoint, Steihaug, TrustRegion},
};
use ndarray::{Array1, Array2};
use thiserror::Error;

// TODO: Do not repeat code in the linesearch branch, use helper function?

#[derive(Error, Debug, PartialEq)]
/// Local solver error enum
pub enum LocalSolverError {
    #[error("Local Solver Error: Invalid LocalSolverConfig for L-BFGS solver. {0}")]
    InvalidLBFGSConfig(String),

    #[error("Local Solver Error: Invalid LocalSolverConfig for Nelder-Mead solver. {0}")]
    InvalidNelderMeadConfig(String),

    #[error("Local Solver Error: Invalid LocalSolverConfig for Steepest Descent solver. {0}")]
    InvalidSteepestDescentConfig(String),

    #[error("Local Solver Error: Invalid LocalSolverConfig for Trust Region solver. {0}")]
    InvalidTrustRegionConfig(String),

    #[error("Local Solver Error: Failed to run local solver. {0}")]
    RunFailed(String),

    #[error("Local Solver Error: No solution found")]
    NoSolution,
}

/// # Local solver struct
///
/// This struct contains the problem to solve and the local solver type and configuration.
///
/// It has a `solve` method that uses a match to select the local solver function to use based on the `LocalSolverType` enum.
/// The `solve` method returns a `LocalSolution` struct.
///
/// The `LocalSolver` struct is generic over the `Problem` trait.
/// It has a problem field of type `P` and a local solver type field of type `LocalSolverType`.
/// It also has a local solver configuration field of type `LocalSolverConfig` to configure the local solver.
pub struct LocalSolver<P: Problem> {
    problem: P,
    local_solver_type: LocalSolverType,
    local_solver_config: LocalSolverConfig,
}

impl<P: Problem> LocalSolver<P> {
    pub fn new(
        problem: P,
        local_solver_type: LocalSolverType,
        local_solver_config: LocalSolverConfig,
    ) -> Self {
        Self {
            problem,
            local_solver_type,
            local_solver_config,
        }
    }

    /// Solve the optimization problem using the local solver
    ///
    /// This function uses a match to select the local solver function to use based on the `LocalSolverType` enum.
    pub fn solve(&self, initial_point: Array1<f64>) -> Result<LocalSolution, LocalSolverError> {
        match self.local_solver_type {
            LocalSolverType::LBFGS => self.solve_lbfgs(initial_point, &self.local_solver_config),
            LocalSolverType::NelderMead => {
                self.solve_nelder_mead(initial_point, &self.local_solver_config)
            }
            LocalSolverType::SteepestDescent => {
                self.solve_steepestdescent(initial_point, &self.local_solver_config)
            }
            LocalSolverType::TrustRegion => {
                self.solve_trust_region(initial_point, &self.local_solver_config)
            }
        }
    }

    /// Solve the optimization problem using the L-BFGS local solver
    fn solve_lbfgs(
        &self,
        initial_point: Array1<f64>,
        solver_config: &LocalSolverConfig,
    ) -> Result<LocalSolution, LocalSolverError> {
        struct ProblemCost<'a, P: Problem> {
            problem: &'a P,
        }

        impl<P: Problem> CostFunction for ProblemCost<'_, P> {
            type Param = Array1<f64>;
            type Output = f64;

            fn cost(&self, param: &Self::Param) -> std::result::Result<Self::Output, Error> {
                self.problem
                    .objective(param)
                    .map_err(|e| Error::msg(e.to_string()))
            }
        }

        impl<P: Problem> Gradient for ProblemCost<'_, P> {
            type Param = Array1<f64>;
            type Gradient = Array1<f64>;

            fn gradient(&self, param: &Self::Param) -> std::result::Result<Self::Gradient, Error> {
                self.problem
                    .gradient(param)
                    .map_err(|e| Error::msg(e.to_string()))
            }
        }

        let cost = ProblemCost {
            problem: &self.problem,
        };

        // TODO: For now we don't support l1 regularization
        // Should we support it and default to 0.0? Does it change something in the minimization?
        // Seems we can pass None?

        if let LocalSolverConfig::LBFGS {
            max_iter,
            tolerance_grad,
            tolerance_cost,
            history_size,
            line_search_params,
        } = solver_config
        {
            // Match line search method
            match &line_search_params.method {
                LineSearchMethod::MoreThuente {
                    c1,
                    c2,
                    width_tolerance,
                    bounds,
                } => {
                    let linesearch = MoreThuenteLineSearch::new()
                        .with_c(*c1, *c2)
                        .map_err(|e: Error| LocalSolverError::InvalidLBFGSConfig(e.to_string()))?
                        .with_bounds(bounds[0], bounds[1])
                        .map_err(|e: Error| LocalSolverError::InvalidLBFGSConfig(e.to_string()))?
                        .with_width_tolerance(*width_tolerance)
                        .map_err(|e: Error| LocalSolverError::InvalidLBFGSConfig(e.to_string()))?;

                    let solver = LBFGS::new(linesearch, *history_size)
                        .with_tolerance_cost(*tolerance_cost)
                        .map_err(|e: Error| LocalSolverError::InvalidLBFGSConfig(e.to_string()))?
                        .with_tolerance_grad(*tolerance_grad)
                        .map_err(|e: Error| LocalSolverError::InvalidLBFGSConfig(e.to_string()))?;

                    let res = Executor::new(cost, solver)
                        .configure(|state| state.param(initial_point.clone()).max_iters(*max_iter))
                        .run()
                        .map_err(|e: Error| LocalSolverError::RunFailed(e.to_string()))?;

                    Ok(LocalSolution {
                        point: res
                            .state()
                            .best_param
                            .as_ref()
                            .ok_or(LocalSolverError::NoSolution)?
                            .clone(),
                        objective: res.state().best_cost,
                    })
                }
                LineSearchMethod::HagerZhang {
                    delta,
                    sigma,
                    epsilon,
                    theta,
                    gamma,
                    eta,
                    bounds,
                } => {
                    let linesearch = HagerZhangLineSearch::new()
                        .with_delta_sigma(*delta, *sigma)
                        .map_err(|e: Error| LocalSolverError::InvalidLBFGSConfig(e.to_string()))?
                        .with_epsilon(*epsilon)
                        .map_err(|e: Error| LocalSolverError::InvalidLBFGSConfig(e.to_string()))?
                        .with_theta(*theta)
                        .map_err(|e: Error| LocalSolverError::InvalidLBFGSConfig(e.to_string()))?
                        .with_gamma(*gamma)
                        .map_err(|e: Error| LocalSolverError::InvalidLBFGSConfig(e.to_string()))?
                        .with_eta(*eta)
                        .map_err(|e: Error| LocalSolverError::InvalidLBFGSConfig(e.to_string()))?
                        .with_bounds(bounds[0], bounds[1])
                        .map_err(|e: Error| LocalSolverError::InvalidLBFGSConfig(e.to_string()))?;

                    let solver = LBFGS::new(linesearch, *history_size)
                        .with_tolerance_cost(*tolerance_cost)
                        .map_err(|e: Error| LocalSolverError::InvalidLBFGSConfig(e.to_string()))?
                        .with_tolerance_grad(*tolerance_grad)
                        .map_err(|e: Error| LocalSolverError::InvalidLBFGSConfig(e.to_string()))?;

                    let res = Executor::new(cost, solver)
                        .configure(|state| state.param(initial_point.clone()).max_iters(*max_iter))
                        .run()
                        .map_err(|e: Error| LocalSolverError::RunFailed(e.to_string()))?;

                    Ok(LocalSolution {
                        point: res
                            .state()
                            .best_param
                            .as_ref()
                            .ok_or(LocalSolverError::NoSolution)?
                            .clone(),
                        objective: res.state().best_cost,
                    })
                }
            }
        } else {
            Err(LocalSolverError::InvalidLBFGSConfig(
                "Error parsing solver config".to_string(),
            ))
        }
    }

    /// Solve the optimization problem using the Nelder-Mead local solver
    fn solve_nelder_mead(
        &self,
        initial_point: Array1<f64>,
        solver_config: &LocalSolverConfig,
    ) -> Result<LocalSolution, LocalSolverError> {
        struct ProblemCost<'a, P: Problem> {
            problem: &'a P,
        }

        impl<P: Problem> CostFunction for ProblemCost<'_, P> {
            type Param = Array1<f64>;
            type Output = f64;

            fn cost(&self, param: &Self::Param) -> std::result::Result<Self::Output, Error> {
                self.problem
                    .objective(param)
                    .map_err(|e| Error::msg(e.to_string()))
            }
        }

        let cost = ProblemCost {
            problem: &self.problem,
        };

        if let LocalSolverConfig::NelderMead {
            simplex_delta,
            sd_tolerance,
            max_iter,
            alpha,
            gamma,
            rho,
            sigma,
        } = solver_config
        {
            // Generate initial simplex
            let mut simplex = vec![initial_point.clone()];
            for i in 0..initial_point.len() {
                let mut point = initial_point.clone();
                point[i] += simplex_delta;
                simplex.push(point);
            }

            let solver = NelderMead::new(simplex)
                .with_sd_tolerance(*sd_tolerance)
                .map_err(|e: Error| LocalSolverError::InvalidNelderMeadConfig(e.to_string()))?
                .with_alpha(*alpha)
                .map_err(|e: Error| LocalSolverError::InvalidNelderMeadConfig(e.to_string()))?
                .with_gamma(*gamma)
                .map_err(|e: Error| LocalSolverError::InvalidNelderMeadConfig(e.to_string()))?
                .with_rho(*rho)
                .map_err(|e: Error| LocalSolverError::InvalidNelderMeadConfig(e.to_string()))?
                .with_sigma(*sigma)
                .map_err(|e: Error| LocalSolverError::InvalidNelderMeadConfig(e.to_string()))?;

            let res = Executor::new(cost, solver)
                .configure(|state| state.max_iters(*max_iter))
                .run()
                .map_err(|e: Error| LocalSolverError::RunFailed(e.to_string()))?;

            Ok(LocalSolution {
                point: res
                    .state()
                    .best_param
                    .as_ref()
                    .ok_or(LocalSolverError::NoSolution)?
                    .clone(),
                objective: res.state().best_cost,
            })
        } else {
            Err(LocalSolverError::InvalidNelderMeadConfig(
                "Error parsing solver configuration".to_string(),
            ))
        }
    }

    /// Solve the optimization problem using the Steepest Descent local solver
    fn solve_steepestdescent(
        &self,
        initial_point: Array1<f64>,
        solver_config: &LocalSolverConfig,
    ) -> Result<LocalSolution, LocalSolverError> {
        struct ProblemCost<'a, P: Problem> {
            problem: &'a P,
        }

        impl<P: Problem> CostFunction for ProblemCost<'_, P> {
            type Param = Array1<f64>;
            type Output = f64;

            fn cost(&self, param: &Self::Param) -> std::result::Result<Self::Output, Error> {
                self.problem
                    .objective(param)
                    .map_err(|e| Error::msg(e.to_string()))
            }
        }

        impl<P: Problem> Gradient for ProblemCost<'_, P> {
            type Param = Array1<f64>;
            type Gradient = Array1<f64>;

            fn gradient(&self, param: &Self::Param) -> std::result::Result<Self::Gradient, Error> {
                self.problem
                    .gradient(param)
                    .map_err(|e| Error::msg(e.to_string()))
            }
        }

        let cost = ProblemCost {
            problem: &self.problem,
        };

        if let LocalSolverConfig::SteepestDescent {
            max_iter,
            line_search_params,
        } = solver_config
        {
            // Match line search method
            match &line_search_params.method {
                LineSearchMethod::MoreThuente {
                    c1,
                    c2,
                    width_tolerance,
                    bounds,
                } => {
                    let linesearch = MoreThuenteLineSearch::new()
                        .with_c(*c1, *c2)
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidSteepestDescentConfig(e.to_string())
                        })?
                        .with_bounds(bounds[0], bounds[1])
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidSteepestDescentConfig(e.to_string())
                        })?
                        .with_width_tolerance(*width_tolerance)
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidSteepestDescentConfig(e.to_string())
                        })?;

                    let solver = SteepestDescent::new(linesearch);

                    let res = Executor::new(cost, solver)
                        .configure(|state| state.param(initial_point.clone()).max_iters(*max_iter))
                        .run()
                        .map_err(|e: Error| LocalSolverError::RunFailed(e.to_string()))?;

                    Ok(LocalSolution {
                        point: res
                            .state()
                            .best_param
                            .as_ref()
                            .ok_or(LocalSolverError::NoSolution)?
                            .clone(),
                        objective: res.state().best_cost,
                    })
                }
                LineSearchMethod::HagerZhang {
                    delta,
                    sigma,
                    epsilon,
                    theta,
                    gamma,
                    eta,
                    bounds,
                } => {
                    let linesearch = HagerZhangLineSearch::new()
                        .with_delta_sigma(*delta, *sigma)
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidSteepestDescentConfig(e.to_string())
                        })?
                        .with_epsilon(*epsilon)
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidSteepestDescentConfig(e.to_string())
                        })?
                        .with_theta(*theta)
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidSteepestDescentConfig(e.to_string())
                        })?
                        .with_gamma(*gamma)
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidSteepestDescentConfig(e.to_string())
                        })?
                        .with_eta(*eta)
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidSteepestDescentConfig(e.to_string())
                        })?
                        .with_bounds(bounds[0], bounds[1])
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidSteepestDescentConfig(e.to_string())
                        })?;

                    let solver = SteepestDescent::new(linesearch);

                    let res = Executor::new(cost, solver)
                        .configure(|state| state.param(initial_point.clone()).max_iters(*max_iter))
                        .run()
                        .map_err(|e: Error| LocalSolverError::RunFailed(e.to_string()))?;

                    Ok(LocalSolution {
                        point: res
                            .state()
                            .best_param
                            .as_ref()
                            .ok_or(LocalSolverError::NoSolution)?
                            .clone(),
                        objective: res.state().best_cost,
                    })
                }
            }
        } else {
            Err(LocalSolverError::InvalidSteepestDescentConfig(
                "Error parsing solver configuration".to_string(),
            ))
        }
    }

    /// Solve the optimization problem using the Nelder-Mead local solver
    fn solve_trust_region(
        &self,
        initial_point: Array1<f64>,
        solver_config: &LocalSolverConfig,
    ) -> Result<LocalSolution, LocalSolverError> {
        struct ProblemCost<'a, P: Problem> {
            problem: &'a P,
        }

        impl<P: Problem> CostFunction for ProblemCost<'_, P> {
            type Param = Array1<f64>;
            type Output = f64;

            fn cost(&self, param: &Self::Param) -> std::result::Result<Self::Output, Error> {
                self.problem
                    .objective(param)
                    .map_err(|e| Error::msg(e.to_string()))
            }
        }

        impl<P: Problem> Gradient for ProblemCost<'_, P> {
            type Param = Array1<f64>;
            type Gradient = Array1<f64>;

            fn gradient(&self, param: &Self::Param) -> std::result::Result<Self::Gradient, Error> {
                self.problem
                    .gradient(param)
                    .map_err(|e| Error::msg(e.to_string()))
            }
        }

        impl<P: Problem> Hessian for ProblemCost<'_, P> {
            type Param = Array1<f64>;
            type Hessian = Array2<f64>;

            fn hessian(&self, param: &Self::Param) -> std::result::Result<Self::Hessian, Error> {
                self.problem
                    .hessian(param)
                    .map_err(|e| Error::msg(e.to_string()))
            }
        }

        let cost = ProblemCost {
            problem: &self.problem,
        };

        if let LocalSolverConfig::TrustRegion {
            trust_region_radius_method,
            max_iter,
            radius,
            max_radius,
            eta,
        } = solver_config
        {
            match trust_region_radius_method {
                TrustRegionRadiusMethod::Cauchy => {
                    let subproblem = CauchyPoint::new();
                    let solver = TrustRegion::new(subproblem)
                        .with_radius(*radius)
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidTrustRegionConfig(e.to_string())
                        })?
                        .with_max_radius(*max_radius)
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidTrustRegionConfig(e.to_string())
                        })?
                        .with_eta(*eta)
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidTrustRegionConfig(e.to_string())
                        })?;
                    let res = Executor::new(cost, solver)
                        .configure(|state| state.param(initial_point.clone()).max_iters(*max_iter))
                        .run()
                        .map_err(|e: Error| LocalSolverError::RunFailed(e.to_string()))?;

                    Ok(LocalSolution {
                        point: res
                            .state()
                            .best_param
                            .as_ref()
                            .ok_or(LocalSolverError::NoSolution)?
                            .clone(),
                        objective: res.state().best_cost,
                    })
                }
                TrustRegionRadiusMethod::Steihaug => {
                    let subproblem = Steihaug::new();
                    let solver = TrustRegion::new(subproblem)
                        .with_radius(*radius)
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidTrustRegionConfig(e.to_string())
                        })?
                        .with_max_radius(*max_radius)
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidTrustRegionConfig(e.to_string())
                        })?
                        .with_eta(*eta)
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidTrustRegionConfig(e.to_string())
                        })?;
                    let res = Executor::new(cost, solver)
                        .configure(|state| state.param(initial_point.clone()).max_iters(*max_iter))
                        .run()
                        .map_err(|e: Error| LocalSolverError::RunFailed(e.to_string()))?;

                    Ok(LocalSolution {
                        point: res
                            .state()
                            .best_param
                            .as_ref()
                            .ok_or(LocalSolverError::NoSolution)?
                            .clone(),
                        objective: res.state().best_cost,
                    })
                }
            }
        } else {
            Err(LocalSolverError::InvalidTrustRegionConfig(
                "Error parsing solver configuration".to_string(),
            ))
        }
    }
}

#[cfg(test)]
mod tests_local_solvers {
    use super::*;
    use crate::local_solver::builders::{HagerZhangBuilder, LBFGSBuilder, SteepestDescentBuilder};
    use crate::types::{EvaluationError, LocalSolverType};
    use ndarray::{array, Array2};

    #[derive(Debug, Clone)]
    pub struct NoGradientSixHumpCamel;

    impl Problem for NoGradientSixHumpCamel {
        fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
            Ok(
                (4.0 - 2.1 * x[0].powi(2) + x[0].powi(4) / 3.0) * x[0].powi(2)
                    + x[0] * x[1]
                    + (-4.0 + 4.0 * x[1].powi(2)) * x[1].powi(2),
            )
        }

        fn variable_bounds(&self) -> Array2<f64> {
            array![[-3.0, 3.0], [-2.0, 2.0]]
        }
    }

    #[test]
    /// Test the Nelder-Mead local solver with a problem that doesn't
    /// have a gradient. Since Nelder-Mead doesn't require a gradient,
    /// the local solver should run without an error.
    fn test_nelder_mead_no_gradient() {
        let problem: NoGradientSixHumpCamel = NoGradientSixHumpCamel;

        let local_solver: LocalSolver<NoGradientSixHumpCamel> = LocalSolver::new(
            problem.clone(),
            LocalSolverType::NelderMead,
            LocalSolverConfig::NelderMead {
                simplex_delta: 0.1,
                sd_tolerance: 1e-6,
                max_iter: 1000,
                alpha: 1.0,
                gamma: 2.0,
                rho: 0.5,
                sigma: 0.5,
            },
        );

        let initial_point: Array1<f64> = array![0.0, 0.0];
        let res: LocalSolution = local_solver.solve(initial_point).unwrap();
        assert_eq!(res.objective, -1.0316278623977673);
    }

    #[test]
    /// Test the Steepest Descent local solver with a problem that doesn't
    /// have a gradient. Since Steepest Descent requires a gradient,
    /// the local solver should return an error.
    fn test_steepest_descent_no_gradient() {
        let problem: NoGradientSixHumpCamel = NoGradientSixHumpCamel;

        let local_solver: LocalSolver<NoGradientSixHumpCamel> = LocalSolver::new(
            problem,
            LocalSolverType::SteepestDescent,
            SteepestDescentBuilder::default().build(),
        );

        let initial_point: Array1<f64> = array![0.0, 0.0];
        let error: LocalSolverError = local_solver.solve(initial_point).unwrap_err();
        assert_eq!(
            error,
            LocalSolverError::RunFailed(
                "Gradient not implemented and needed for local solver".to_string()
            )
        );
    }

    #[test]
    /// Test the L-BFGS local solver with a problem that doesn't
    /// have a gradient. Since L-BFGS requires a gradient,
    /// the local solver should return an error.
    fn test_lbfgs_no_gradient() {
        let problem: NoGradientSixHumpCamel = NoGradientSixHumpCamel;

        let local_solver: LocalSolver<NoGradientSixHumpCamel> = LocalSolver::new(
            problem,
            LocalSolverType::LBFGS,
            LBFGSBuilder::default().build(),
        );

        let initial_point: Array1<f64> = array![0.0, 0.0];
        let error: LocalSolverError = local_solver.solve(initial_point).unwrap_err();
        assert_eq!(
            error,
            LocalSolverError::RunFailed(
                "Gradient not implemented and needed for local solver".to_string()
            )
        );
    }

    #[test]
    /// Test creating a local solver with an invalid configuration
    /// In this case, for HagerZhangLineSearch, delta must be in (0, 1) and we set it to 2.0
    /// It should return the following error:
    /// `HagerZhangLineSearch`: delta must be in (0, 1) and sigma must be in [delta, 1)
    fn invalid_hagerzhang() {
        let problem: NoGradientSixHumpCamel = NoGradientSixHumpCamel;

        let local_solver: LocalSolver<NoGradientSixHumpCamel> = LocalSolver::new(
            problem,
            LocalSolverType::LBFGS,
            LocalSolverConfig::LBFGS {
                max_iter: 1000,
                tolerance_grad: 1e-6,
                tolerance_cost: 1e-6,
                history_size: 5,
                line_search_params: HagerZhangBuilder::default().delta(2.0).build(),
            },
        );

        let initial_point: Array1<f64> = array![0.0, 0.0];
        let error: LocalSolverError = local_solver.solve(initial_point).unwrap_err();

        assert_eq!(
            error,
            LocalSolverError::InvalidLBFGSConfig(
                "Invalid parameter: \"`HagerZhangLineSearch`: delta must be in (0, 1) and sigma must be in [delta, 1).\""
                    .to_string()
            )
        );
    }

    #[test]
    /// Test creating a Trust Region solver using an invalid eta value
    /// In this case, eta must be in [0, 1/4) and we set it to 1.0
    fn invalid_trust_region_eta() {
        let problem: NoGradientSixHumpCamel = NoGradientSixHumpCamel;

        let local_solver: LocalSolver<NoGradientSixHumpCamel> = LocalSolver::new(
            problem,
            LocalSolverType::TrustRegion,
            LocalSolverConfig::TrustRegion {
                trust_region_radius_method: TrustRegionRadiusMethod::Steihaug,
                max_iter: 1000,
                radius: 0.1,
                max_radius: 1.0,
                eta: 1.0,
            },
        );

        let initial_point: Array1<f64> = array![0.0, 0.0];
        let error: LocalSolverError = local_solver.solve(initial_point).unwrap_err();

        assert_eq!(
            error,
            LocalSolverError::InvalidTrustRegionConfig(
                "Invalid parameter: \"`TrustRegion`: eta must be in [0, 1/4).\"".to_string()
            )
        );
    }
}
