//! # Local solver module
//!
//! This module contains the implementation to interface with the local solver. The local solver is used to solve the optimization problem in the neighborhood of a given point. The local solver is implemented using the `argmin` crate.

use crate::problem::Problem;
use crate::types::{LineSearchMethod, LocalSolution, LocalSolverConfig, LocalSolverType, Result};
use argmin::core::{CostFunction, Error, Executor, Gradient};
use argmin::solver::{
    gradientdescent::SteepestDescent,
    linesearch::{HagerZhangLineSearch, MoreThuenteLineSearch},
    neldermead::NelderMead,
    quasinewton::LBFGS,
};
use ndarray::Array1;

// TODO: Do not repeat code in the linesearch branch, use helper function?

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
    pub fn solve(&self, initial_point: &Array1<f64>) -> Result<LocalSolution> {
        match self.local_solver_type {
            LocalSolverType::LBFGS => self.solve_lbfgs(initial_point, &self.local_solver_config),
            LocalSolverType::NelderMead => {
                self.solve_nelder_mead(initial_point, &self.local_solver_config)
            }
            LocalSolverType::SteepestDescent => {
                self.solve_steepestdescent(initial_point, &self.local_solver_config)
            }
        }
    }

    /// Solve the optimization problem using the L-BFGS local solver
    fn solve_lbfgs(
        &self,
        initial_point: &Array1<f64>,
        solver_config: &LocalSolverConfig,
    ) -> Result<LocalSolution> {
        struct ProblemCost<'a, P: Problem> {
            problem: &'a P,
        }

        impl<'a, P: Problem> CostFunction for ProblemCost<'a, P> {
            type Param = Array1<f64>;
            type Output = f64;

            fn cost(&self, param: &Self::Param) -> std::result::Result<Self::Output, Error> {
                self.problem
                    .objective(param)
                    .map_err(|e| Error::msg(e.to_string()))
            }
        }

        impl<'a, P: Problem> Gradient for ProblemCost<'a, P> {
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
                        .with_c(*c1, *c2)?
                        .with_bounds(bounds[0], bounds[1])?
                        .with_width_tolerance(*width_tolerance)?;

                    let solver = LBFGS::new(linesearch, *history_size)
                        .with_tolerance_cost(*tolerance_cost)?
                        .with_tolerance_grad(*tolerance_grad)?;

                    let res = Executor::new(cost, solver)
                        .configure(|state| state.param(initial_point.clone()).max_iters(*max_iter))
                        .run()?;

                    return Ok(LocalSolution {
                        point: res
                            .state()
                            .best_param
                            .as_ref()
                            .ok_or_else(|| anyhow::anyhow!("No solution"))?
                            .clone(),
                        objective: res.state().best_cost,
                    });
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
                        .with_delta_sigma(*delta, *sigma)?
                        .with_epsilon(*epsilon)?
                        .with_theta(*theta)?
                        .with_gamma(*gamma)?
                        .with_eta(*eta)?
                        .with_bounds(bounds[0], bounds[1])?;

                    let solver = LBFGS::new(linesearch, *history_size)
                        .with_tolerance_cost(*tolerance_cost)?
                        .with_tolerance_grad(*tolerance_grad)?;

                    let res = Executor::new(cost, solver)
                        .configure(|state| state.param(initial_point.clone()).max_iters(*max_iter))
                        .run()?;

                    return Ok(LocalSolution {
                        point: res
                            .state()
                            .best_param
                            .as_ref()
                            .ok_or_else(|| anyhow::anyhow!("No solution"))?
                            .clone(),
                        objective: res.state().best_cost,
                    });
                }
            };
        } else {
            Err(anyhow::anyhow!(
                "Invalid LocalSolverConfig for L-BFGS solver"
            ))
        }
    }

    fn solve_nelder_mead(
        &self,
        initial_point: &Array1<f64>,
        solver_config: &LocalSolverConfig,
    ) -> Result<LocalSolution> {
        struct ProblemCost<'a, P: Problem> {
            problem: &'a P,
        }

        impl<'a, P: Problem> CostFunction for ProblemCost<'a, P> {
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
            sd_tolerance,
            max_iter,
            alpha,
            gamma,
            rho,
            sigma,
        } = solver_config
        {
            // Generate initial simplex
            // TODO: Set user settings for this
            let mut simplex = vec![initial_point.clone()];
            for i in 0..initial_point.len() {
                let mut point = initial_point.clone();
                point[i] += 0.1;
                simplex.push(point);
            }

            let solver = NelderMead::new(simplex)
                .with_sd_tolerance(*sd_tolerance)?
                .with_alpha(*alpha)?
                .with_gamma(*gamma)?
                .with_rho(*rho)?
                .with_sigma(*sigma)?;

            let res = Executor::new(cost, solver)
                .configure(|state| state.max_iters(*max_iter))
                .run()?;

            Ok(LocalSolution {
                point: res
                    .state()
                    .best_param
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("No solution"))?
                    .clone(),
                objective: res.state().best_cost,
            })
        } else {
            Err(anyhow::anyhow!(
                "Invalid LocalSolverConfig for Nelder-Mead solver"
            ))
        }
    }

    fn solve_steepestdescent(
        &self,
        initial_point: &Array1<f64>,
        solver_config: &LocalSolverConfig,
    ) -> Result<LocalSolution> {
        struct ProblemCost<'a, P: Problem> {
            problem: &'a P,
        }

        impl<'a, P: Problem> CostFunction for ProblemCost<'a, P> {
            type Param = Array1<f64>;
            type Output = f64;

            fn cost(&self, param: &Self::Param) -> std::result::Result<Self::Output, Error> {
                self.problem
                    .objective(param)
                    .map_err(|e| Error::msg(e.to_string()))
            }
        }

        impl<'a, P: Problem> Gradient for ProblemCost<'a, P> {
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
                        .with_c(*c1, *c2)?
                        .with_bounds(bounds[0], bounds[1])?
                        .with_width_tolerance(*width_tolerance)?;

                    let solver = SteepestDescent::new(linesearch);

                    let res = Executor::new(cost, solver)
                        .configure(|state| state.param(initial_point.clone()).max_iters(*max_iter))
                        .run()?;

                    return Ok(LocalSolution {
                        point: res
                            .state()
                            .best_param
                            .as_ref()
                            .ok_or_else(|| anyhow::anyhow!("No solution"))?
                            .clone(),
                        objective: res.state().best_cost,
                    });
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
                        .with_delta_sigma(*delta, *sigma)?
                        .with_epsilon(*epsilon)?
                        .with_theta(*theta)?
                        .with_gamma(*gamma)?
                        .with_eta(*eta)?
                        .with_bounds(bounds[0], bounds[1])?;

                    let solver = SteepestDescent::new(linesearch);

                    let res = Executor::new(cost, solver)
                        .configure(|state| state.param(initial_point.clone()).max_iters(*max_iter))
                        .run()?;

                    return Ok(LocalSolution {
                        point: res
                            .state()
                            .best_param
                            .as_ref()
                            .ok_or_else(|| anyhow::anyhow!("No solution"))?
                            .clone(),
                        objective: res.state().best_cost,
                    });
                }
            };
        } else {
            Err(anyhow::anyhow!(
                "Invalid LocalSolverConfig for Steepest Descent solver"
            ))
        }
    }
}
