use crate::problem::Problem;
use crate::types::{LocalSolution, LocalSolverType, Result};
use argmin::core::{CostFunction, Error, Executor, Gradient};
use argmin::solver::{
    gradientdescent::SteepestDescent, linesearch::MoreThuenteLineSearch, neldermead::NelderMead,
    quasinewton::LBFGS,
};
use ndarray::Array1;

pub struct LocalSolver<P: Problem> {
    problem: P,
    solver_type: LocalSolverType,
}

impl<P: Problem> LocalSolver<P> {
    pub fn new(problem: P, solver_type: LocalSolverType) -> Self {
        Self {
            problem,
            solver_type,
        }
    }

    pub fn solve(&self, initial_point: &Array1<f64>) -> Result<LocalSolution> {
        match self.solver_type {
            LocalSolverType::LBFGS => self.solve_lbfgs(initial_point),
            LocalSolverType::NelderMead => self.solve_nelder_mead(initial_point),
            LocalSolverType::SteepestDescent => self.solve_steepestdescent(initial_point),
        }
    }

    fn solve_lbfgs(&self, initial_point: &Array1<f64>) -> Result<LocalSolution> {
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

        // TODO: Change this settings? Let user select local solver settings and linesearch?
        let linesearch = MoreThuenteLineSearch::new().with_c(1e-4, 0.9)?;
        let solver = LBFGS::new(linesearch, 7);

        let res = Executor::new(cost, solver)
            .configure(|state| state.param(initial_point.clone()).max_iters(1000))
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
    }

    fn solve_nelder_mead(&self, initial_point: &Array1<f64>) -> Result<LocalSolution> {
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

        // Generate initial simplex
        // TODO: Set user settings for this
        let mut simplex = vec![initial_point.clone()];
        for i in 0..initial_point.len() {
            let mut point = initial_point.clone();
            point[i] += 0.1;
            simplex.push(point);
        }

        let solver = NelderMead::new(simplex).with_sd_tolerance(1e-6)?;

        let res = Executor::new(cost, solver)
            .configure(|state| state.max_iters(1000))
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
    }

    fn solve_steepestdescent(&self, initial_point: &Array1<f64>) -> Result<LocalSolution> {
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

        // TODO: Change this settings? Let user select local solver settings and linesearch?
        let linesearch = MoreThuenteLineSearch::new().with_c(1e-4, 0.9)?;
        let solver = SteepestDescent::new(linesearch);

        let res = Executor::new(cost, solver)
            .configure(|state| state.param(initial_point.clone()).max_iters(1000))
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
    }
}
