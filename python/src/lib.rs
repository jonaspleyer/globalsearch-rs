mod builders;

use globalsearch::local_solver::builders::{
    LBFGSBuilder, NelderMeadBuilder, NewtonCGBuilder, SteepestDescentBuilder, TrustRegionBuilder,
};
use globalsearch::oqnlp::OQNLP;
use globalsearch::problem::Problem;
use globalsearch::types::{EvaluationError, LocalSolverType, OQNLPParams};
use ndarray::{Array1, Array2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyOQNLPParams {
    #[pyo3(get, set)]
    pub iterations: usize,
    #[pyo3(get, set)]
    pub population_size: usize,
    #[pyo3(get, set)]
    pub wait_cycle: usize,
    #[pyo3(get, set)]
    pub threshold_factor: f64,
    #[pyo3(get, set)]
    pub distance_factor: f64,
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyLocalSolution {
    #[pyo3(get, set)]
    pub point: Vec<f64>,
    #[pyo3(get, set)]
    pub objective: f64,
}

#[pymethods]
impl PyLocalSolution {
    #[new]
    fn new(point: Vec<f64>, objective: f64) -> Self {
        PyLocalSolution { point, objective }
    }

    /// Returns the objective function value at the solution point
    /// 
    /// Same as `objective` field
    /// 
    /// This method is similar to the `fun` method in `SciPy.optimize` result
    fn fun(&self) -> f64 {
        self.objective
    }

    /// Returns the solution point as a list of float values
    ///
    /// Same as `point` field
    ///
    /// This method is similar to the `x` method in `SciPy.optimize` result
    fn x(&self) -> Vec<f64> {
        self.point.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "PyLocalSolution(point={:?}, objective={})",
            self.point, self.objective
        )
    }

    fn __str__(&self) -> String {
        format!(
            "Solution(x={:?}, fun={})",
            self.point, self.objective
        )
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PySolutionSet {
    #[pyo3(get)]
    pub solutions: Vec<PyLocalSolution>,
}

#[pymethods]
impl PySolutionSet {
    #[new]
    fn new(solutions: Vec<PyLocalSolution>) -> Self {
        PySolutionSet { solutions }
    }

    /// Returns the number of solutions stored in the set.
    fn __len__(&self) -> usize {
        self.solutions.len()
    }

    /// Returns true if the solution set contains no solutions.
    fn is_empty(&self) -> bool {
        self.solutions.is_empty()
    }

    /// Returns the best solution in the set based on the objective function value.
    fn best_solution(&self) -> Option<PyLocalSolution> {
        self.solutions
            .iter()
            .min_by(|a, b| a.objective.partial_cmp(&b.objective).unwrap())
            .cloned()
    }

    /// Returns the solution at the given index.
    fn __getitem__(&self, index: usize) -> PyResult<PyLocalSolution> {
        self.solutions
            .get(index)
            .cloned()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyIndexError, _>("Index out of range"))
    }

    /// Returns an iterator over the solutions in the set.
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PySolutionSetIterator>> {
        let iter = PySolutionSetIterator {
            inner: slf.solutions.clone(),
            index: 0,
        };
        Py::new(slf.py(), iter)
    }

    fn __repr__(&self) -> String {
        format!("PySolutionSet(solutions={:?})", self.solutions)
    }

    fn __str__(&self) -> String {
        let mut result = String::from("Solution Set\n");
        result.push_str(&format!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"));
        result.push_str(&format!("Total solutions: {}\n", self.solutions.len()));
        
        if !self.solutions.is_empty() {
            if let Some(best) = self.best_solution() {
                result.push_str(&format!("Best objective value: {:.8e}\n", best.objective));
            }
        }
        result.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        for (i, solution) in self.solutions.iter().enumerate() {
            result.push_str(&format!("Solution #{}\n", i + 1));
            result.push_str(&format!("  Objective: {:.8e}\n", solution.objective));
            result.push_str(&format!("  Parameters: {:?}\n", solution.point));

            if i < self.solutions.len() - 1 {
                result.push_str("――――――――――――――――――――――――――――――――――――\n");
            }
        }
        
        result
    }
}

#[pyclass]
struct PySolutionSetIterator {
    inner: Vec<PyLocalSolution>,
    index: usize,
}

#[pymethods]
impl PySolutionSetIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<PyLocalSolution> {
        if slf.index < slf.inner.len() {
            let result = slf.inner[slf.index].clone();
            slf.index += 1;
            Some(result)
        } else {
            None
        }
    }
}

#[pymethods]
impl PyOQNLPParams {
    #[new]
    #[pyo3(signature = (
        iterations = 300,
        population_size = 1000,
        wait_cycle = 15,
        threshold_factor = 0.2,
        distance_factor = 0.75,
    ))]
    #[pyo3(
        text_signature = "(iterations=300, population_size=1000, wait_cycle=15, threshold_factor=0.2, distance_factor=0.75)"
    )]
    fn new(
        iterations: usize,
        population_size: usize,
        wait_cycle: usize,
        threshold_factor: f64,
        distance_factor: f64,
    ) -> Self {
        PyOQNLPParams {
            iterations,
            population_size,
            wait_cycle,
            threshold_factor,
            distance_factor,
        }
    }
}

#[pyclass]
#[derive(Debug)]
pub struct PyProblem {
    #[pyo3(get, set)]
    objective: PyObject,
    #[pyo3(get, set)]
    variable_bounds: PyObject,
    #[pyo3(get, set)]
    gradient: Option<PyObject>,
    #[pyo3(get, set)]
    hessian: Option<PyObject>,
}

impl Clone for PyProblem {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            objective: self.objective.clone_ref(py),
            variable_bounds: self.variable_bounds.clone_ref(py),
            gradient: self.gradient.as_ref().map(|g| g.clone_ref(py)),
            hessian: self.hessian.as_ref().map(|h| h.clone_ref(py)),
        })
    }
}

#[pymethods]
impl PyProblem {
    #[new]
    #[pyo3(signature = (objective, variable_bounds, gradient=None, hessian=None))]
    fn new(
        objective: PyObject,
        variable_bounds: PyObject,
        gradient: Option<PyObject>,
        hessian: Option<PyObject>,
    ) -> Self {
        PyProblem {
            objective,
            variable_bounds,
            gradient,
            hessian,
        }
    }
}

impl Problem for PyProblem {
    fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
        Python::with_gil(|py| {
            let x_py = x
                .to_vec()
                .into_pyobject(py)
                .map_err(|e| EvaluationError::InvalidInput(e.to_string()))?;
            let result = self
                .objective
                .call1(py, (x_py,))
                .map_err(|e| EvaluationError::InvalidInput(e.to_string()))?;
            result
                .extract(py)
                .map_err(|e| EvaluationError::InvalidInput(e.to_string()))
        })
    }

    fn variable_bounds(&self) -> Array2<f64> {
        Python::with_gil(|py| {
            let result = self
                .variable_bounds
                .call0(py)
                .map_err(|e| EvaluationError::InvalidInput(e.to_string()))
                .and_then(|res| {
                    res.extract::<Vec<Vec<f64>>>(py)
                        .map_err(|e| EvaluationError::InvalidInput(e.to_string()))
                });

            match result {
                Ok(bounds) => {
                    let rows = bounds.len();
                    let cols = if rows > 0 { bounds[0].len() } else { 0 };
                    Array2::from_shape_vec((rows, cols), bounds.into_iter().flatten().collect())
                        .unwrap()
                }
                Err(_) => panic!("Variable bounds must be a 2D array of floats"),
            }
        })
    }

    fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
        if let Some(grad_fn) = &self.gradient {
            Python::with_gil(|py| {
                let x_py = x
                    .to_vec()
                    .into_pyobject(py)
                    .map_err(|e| EvaluationError::InvalidInput(e.to_string()))?;
                let result = grad_fn
                    .call1(py, (x_py,))
                    .map_err(|e| EvaluationError::InvalidInput(e.to_string()))?;

                let grad_vec: Vec<f64> = result
                    .extract(py)
                    .map_err(|e| EvaluationError::InvalidInput(e.to_string()))?;

                Ok(Array1::from(grad_vec))
            })
        } else {
            Err(EvaluationError::GradientNotImplemented)
        }
    }

    fn hessian(&self, x: &Array1<f64>) -> Result<Array2<f64>, EvaluationError> {
        if let Some(hess_fn) = &self.hessian {
            Python::with_gil(|py| {
                let x_py = x
                    .to_vec()
                    .into_pyobject(py)
                    .map_err(|e| EvaluationError::InvalidInput(e.to_string()))?;
                let result = hess_fn
                    .call1(py, (x_py,))
                    .map_err(|e| EvaluationError::InvalidInput(e.to_string()))?;

                let hess_vec: Vec<Vec<f64>> = result
                    .extract(py)
                    .map_err(|e| EvaluationError::InvalidInput(e.to_string()))?;

                let size = hess_vec.len();
                let flat_hess: Vec<f64> = hess_vec.into_iter().flatten().collect();

                Array2::from_shape_vec((size, size), flat_hess).map_err(|_| {
                    EvaluationError::InvalidInput("Hessian shape mismatch".to_string())
                })
            })
        } else {
            Err(EvaluationError::HessianNotImplemented)
        }
    }
}

/// Python wrapper around the OQNLP optimizer
///
/// This function takes a problem, parameters and optionally a local solver and its configuration
/// and returns the best solution found by the optimizer.
#[pyfunction]
#[pyo3(signature = (problem, params, local_solver=None, local_solver_config=None, seed=None, target_objective=None, max_time=None, verbose=false, exclude_out_of_bounds=false))]
fn optimize(
    problem: PyProblem,
    params: PyOQNLPParams,
    local_solver: Option<&str>,
    local_solver_config: Option<PyObject>,
    seed: Option<u64>,
    target_objective: Option<f64>,
    max_time: Option<f64>,
    verbose: Option<bool>,
    exclude_out_of_bounds: Option<bool>,
) -> PyResult<PySolutionSet> {
    Python::with_gil(|py| {
        // Convert local_solver string to enum
        let solver_type = LocalSolverType::from_string(local_solver.unwrap_or("lbfgs"))
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

        let seed = seed.unwrap_or(0);

        // Create local solver configuration (default)
        let local_solver_config = if let Some(config) = local_solver_config {
            match solver_type {
                LocalSolverType::LBFGS => {
                    if let Ok(lbfgs_config) = config.extract::<crate::builders::PyLBFGS>(py) {
                        lbfgs_config.to_builder().build()
                    } else {
                        return Err(PyValueError::new_err(
                            "Expected PyLBFGS for LBFGS solver type".to_string(),
                        ));
                    }
                }
                LocalSolverType::NelderMead => {
                    if let Ok(neldermead_config) =
                        config.extract::<crate::builders::PyNelderMead>(py)
                    {
                        neldermead_config.to_builder().build()
                    } else {
                        return Err(PyValueError::new_err(
                            "Expected PyNelderMead for NelderMead solver type".to_string(),
                        ));
                    }
                }
                LocalSolverType::SteepestDescent => {
                    if let Ok(steepest_descent_config) =
                        config.extract::<crate::builders::PySteepestDescent>(py)
                    {
                        steepest_descent_config.to_builder().build()
                    } else {
                        return Err(PyValueError::new_err(
                            "Expected PySteepestDescent for SteepestDescent solver type"
                                .to_string(),
                        ));
                    }
                }
                LocalSolverType::NewtonCG => {
                    if let Ok(newtoncg_config) = config.extract::<crate::builders::PyNewtonCG>(py) {
                        newtoncg_config.to_builder().build()
                    } else {
                        return Err(PyValueError::new_err(
                            "Expected PyNewtonCG for NewtonCG solver type".to_string(),
                        ));
                    }
                }
                LocalSolverType::TrustRegion => {
                    if let Ok(trustregion_config) =
                        config.extract::<crate::builders::PyTrustRegion>(py)
                    {
                        trustregion_config.to_builder().build()
                    } else {
                        return Err(PyValueError::new_err(
                            "Expected PyTrustRegion for TrustRegion solver type".to_string(),
                        ));
                    }
                }
            }
        } else {
            // Create default local solver configuration
            match solver_type {
                LocalSolverType::LBFGS => LBFGSBuilder::default().build(),
                LocalSolverType::NewtonCG => NewtonCGBuilder::default().build(),
                LocalSolverType::TrustRegion => TrustRegionBuilder::default().build(),
                LocalSolverType::NelderMead => NelderMeadBuilder::default().build(),
                LocalSolverType::SteepestDescent => SteepestDescentBuilder::default().build(),
            }
        };

        let params: OQNLPParams = OQNLPParams {
            iterations: params.iterations,
            population_size: params.population_size,
            wait_cycle: params.wait_cycle,
            threshold_factor: params.threshold_factor,
            distance_factor: params.distance_factor,
            seed,
            local_solver_type: solver_type,
            local_solver_config,
        };

        let mut optimizer =
            OQNLP::new(problem, params).map_err(|e| PyValueError::new_err(e.to_string()))?;

        // Apply optional configurations
        if let Some(target) = target_objective {
            optimizer = optimizer.target_objective(target);
        }

        if let Some(max_secs) = max_time {
            optimizer = optimizer.max_time(max_secs);
        }

        if verbose.unwrap_or(false) {
            optimizer = optimizer.verbose();
        }

        if exclude_out_of_bounds.unwrap_or(false) {
            optimizer = optimizer.exclude_out_of_bounds();
        }

        let solution_set = optimizer.run();

        let binding = solution_set.map_err(|e| PyValueError::new_err(e.to_string()))?;

        // Convert Rust SolutionSet to Python PySolutionSet
        let py_solutions: Vec<PyLocalSolution> = binding
            .solutions()
            .map(|sol| PyLocalSolution {
                point: sol.point.to_vec(),
                objective: sol.objective,
            })
            .collect();

        Ok(PySolutionSet::new(py_solutions))
    })
}

#[pymodule]
/// PyGlobalSearch
///
/// This library provides a Python interface to the `globalsearch-rs` crate.
///
/// `globalsearch-rs` is a Rust implementation of the OQNLP (OptQuest/NLP) algorithm
/// with the core ideas from "Scatter Search and Local NLP Solvers: A Multistart Framework for Global Optimization"
/// by Ugray et al. (2007). It combines scatter search metaheuristics with local
/// minimization for global optimization of nonlinear problems.
///
/// It is similar to MATLAB's `GlobalSearch`.
fn pyglobalsearch(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(optimize, m)?)?;
    m.add_class::<PyOQNLPParams>()?;
    m.add_class::<PyProblem>()?;
    m.add_class::<PyLocalSolution>()?;
    m.add_class::<PySolutionSet>()?;

    // Builders submodule
    let builders = PyModule::new(_py, "builders")?;
    crate::builders::init_module(_py, &builders)?;
    m.add_submodule(&builders)?;

    Ok(())
}
