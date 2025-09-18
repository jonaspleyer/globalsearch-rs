use globalsearch::local_solver::builders::{
    COBYLABuilder, HagerZhangBuilder, LBFGSBuilder, LineSearchParams, MoreThuenteBuilder,
    NelderMeadBuilder, NewtonCGBuilder, SteepestDescentBuilder, TrustRegionBuilder,
    TrustRegionRadiusMethod,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::f64::EPSILON;
use std::f64::INFINITY;

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyHagerZhang {
    #[pyo3(get, set)]
    pub delta: f64,
    #[pyo3(get, set)]
    pub sigma: f64,
    #[pyo3(get, set)]
    pub epsilon: f64,
    #[pyo3(get, set)]
    pub theta: f64,
    #[pyo3(get, set)]
    pub gamma: f64,
    #[pyo3(get, set)]
    pub eta: f64,
    #[pyo3(get, set)]
    pub bounds: Vec<f64>,
}

#[pymethods]
impl PyHagerZhang {
    #[new]
    #[pyo3(signature = (
        delta = 0.1,
        sigma = 0.9,
        epsilon = 1e-6,
        theta = 0.5,
        gamma = 0.66,
        eta = 0.01,
        bounds = vec![EPSILON.sqrt(), INFINITY], 
    ))]
    fn new(
        delta: f64,
        sigma: f64,
        epsilon: f64,
        theta: f64,
        gamma: f64,
        eta: f64,
        bounds: Vec<f64>,
    ) -> Self {
        PyHagerZhang {
            delta,
            sigma,
            epsilon,
            theta,
            gamma,
            eta,
            bounds,
        }
    }
}

impl PyHagerZhang {
    pub fn to_builder(&self) -> HagerZhangBuilder {
        HagerZhangBuilder::new(
            self.delta,
            self.sigma,
            self.epsilon,
            self.theta,
            self.gamma,
            self.eta,
            self.bounds.clone().into(),
        )
    }
}

#[pyfunction]
#[pyo3(
    text_signature = "(delta: f64, sigma: f64, epsilon: f64, theta: f64, gamma: f64, eta: f64, bounds: List[float])"
)]
fn hagerzhang(
    delta: f64,
    sigma: f64,
    epsilon: f64,
    theta: f64,
    gamma: f64,
    eta: f64,
    bounds: Vec<f64>,
) -> PyHagerZhang {
    PyHagerZhang {
        delta,
        sigma,
        epsilon,
        theta,
        gamma,
        eta,
        bounds,
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyMoreThuente {
    #[pyo3(get, set)]
    pub c1: f64,
    #[pyo3(get, set)]
    pub c2: f64,
    #[pyo3(get, set)]
    pub width_tolerance: f64,
    #[pyo3(get, set)]
    pub bounds: Vec<f64>,
}

#[pymethods]
impl PyMoreThuente {
    #[new]
    #[pyo3(signature = (
        c1 = 1e-4,
        c2 = 0.9,
        width_tolerance = 1e-10,
        bounds = vec![EPSILON.sqrt(), INFINITY],
    ))]
    fn new(c1: f64, c2: f64, width_tolerance: f64, bounds: Vec<f64>) -> Self {
        PyMoreThuente {
            c1,
            c2,
            width_tolerance,
            bounds,
        }
    }
}

impl PyMoreThuente {
    pub fn to_builder(&self) -> MoreThuenteBuilder {
        MoreThuenteBuilder::new(
            self.c1,
            self.c2,
            self.width_tolerance,
            self.bounds.clone().into(),
        )
    }
}

#[pyfunction]
#[pyo3(text_signature = "(c1: f64, c2: f64, width_tolerance: f64, bounds: List[float])")]
fn morethuente(c1: f64, c2: f64, width_tolerance: f64, bounds: Vec<f64>) -> PyMoreThuente {
    PyMoreThuente {
        c1,
        c2,
        width_tolerance,
        bounds,
    }
}

#[derive(Debug, Clone)]
pub enum PyLineSearchMethod {
    MoreThuente(PyMoreThuente),
    HagerZhang(PyHagerZhang),
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyLineSearchParams {
    pub method: PyLineSearchMethod,
}

#[pymethods]
impl PyLineSearchParams {
    #[new]
    #[pyo3(signature = (method))]
    fn new(method: Py<pyo3::PyAny>, py: Python) -> PyResult<Self> {
        if let Ok(more_thuente) = method.extract::<PyMoreThuente>(py) {
            return Ok(PyLineSearchParams {
                method: PyLineSearchMethod::MoreThuente(more_thuente),
            });
        }

        if let Ok(hager_zhang) = method.extract::<PyHagerZhang>(py) {
            return Ok(PyLineSearchParams {
                method: PyLineSearchMethod::HagerZhang(hager_zhang),
            });
        }

        Err(PyTypeError::new_err(
            "Expected PyMoreThuente or PyHagerZhang",
        ))
    }

    #[staticmethod]
    fn morethuente(params: PyMoreThuente) -> Self {
        PyLineSearchParams {
            method: PyLineSearchMethod::MoreThuente(params),
        }
    }

    #[staticmethod]
    fn hagerzhang(params: PyHagerZhang) -> Self {
        PyLineSearchParams {
            method: PyLineSearchMethod::HagerZhang(params),
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyLBFGS {
    #[pyo3(get, set)]
    pub max_iter: u64,
    #[pyo3(get, set)]
    pub tolerance_grad: f64,
    #[pyo3(get, set)]
    pub tolerance_cost: f64,
    #[pyo3(get, set)]
    pub history_size: usize,
    #[pyo3(get, set)]
    pub l1_coefficient: Option<f64>,
    #[pyo3(get, set)]
    pub line_search_params: PyLineSearchParams,
}

#[pymethods]
impl PyLBFGS {
    #[new]
    #[pyo3(signature = (
        max_iter = 300,
        tolerance_grad = EPSILON.sqrt(),
        tolerance_cost = EPSILON,
        history_size = 10,
        l1_coefficient = None,
        line_search_params = PyLineSearchParams {
            method: PyLineSearchMethod::MoreThuente(PyMoreThuente {
                c1: 1e-4,
                c2: 0.9,
                width_tolerance: 1e-10,
                bounds: vec![EPSILON.sqrt(), INFINITY],
            }),
        },
    ))]
    fn new(
        max_iter: u64,
        tolerance_grad: f64,
        tolerance_cost: f64,
        history_size: usize,
        l1_coefficient: Option<f64>,
        line_search_params: PyLineSearchParams,
    ) -> Self {
        PyLBFGS {
            max_iter,
            tolerance_grad,
            tolerance_cost,
            history_size,
            l1_coefficient,
            line_search_params,
        }
    }
}

impl PyLBFGS {
    pub fn to_builder(&self) -> LBFGSBuilder {
        let line_search_params = match &self.line_search_params.method {
            PyLineSearchMethod::MoreThuente(params) => LineSearchParams::morethuente()
                .c1(params.c1)
                .c2(params.c2)
                .width_tolerance(params.width_tolerance)
                .bounds(params.bounds.clone().into())
                .build(),
            PyLineSearchMethod::HagerZhang(params) => LineSearchParams::hagerzhang()
                .delta(params.delta)
                .sigma(params.sigma)
                .epsilon(params.epsilon)
                .theta(params.theta)
                .gamma(params.gamma)
                .eta(params.eta)
                .bounds(params.bounds.clone().into())
                .build(),
        };

        LBFGSBuilder::new(
            self.max_iter,
            self.tolerance_grad,
            self.tolerance_cost,
            self.history_size,
            self.l1_coefficient,
            line_search_params,
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    max_iter,
    tolerance_grad,
    tolerance_cost,
    history_size,
    line_search_params,
    l1_coefficient = None,
))]
#[pyo3(
    text_signature = "(max_iter: u64, tolerance_grad: f64, tolerance_cost: f64, history_size: usize, line_search_params: Union[PyMoreThuente, PyHagerZhang], l1_coefficient: Optional[float] = None)"
)]
fn lbfgs(
    max_iter: u64,
    tolerance_grad: f64,
    tolerance_cost: f64,
    history_size: usize,
    line_search_params: Py<pyo3::PyAny>,
    l1_coefficient: Option<f64>,
    py: Python,
) -> PyResult<PyLBFGS> {
    let line_search_params =
        if let Ok(params) = line_search_params.extract::<PyLineSearchParams>(py) {
            params
        } else if let Ok(more_thuente) = line_search_params.extract::<PyMoreThuente>(py) {
            PyLineSearchParams {
                method: PyLineSearchMethod::MoreThuente(more_thuente),
            }
        } else if let Ok(hager_zhang) = line_search_params.extract::<PyHagerZhang>(py) {
            PyLineSearchParams {
                method: PyLineSearchMethod::HagerZhang(hager_zhang),
            }
        } else {
            return Err(PyTypeError::new_err(
                "Expected PyLineSearchParams, PyMoreThuente, or PyHagerZhang",
            ));
        };

    Ok(PyLBFGS {
        max_iter,
        tolerance_grad,
        tolerance_cost,
        history_size,
        l1_coefficient,
        line_search_params,
    })
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyNelderMead {
    #[pyo3(get, set)]
    pub simplex_delta: f64,
    #[pyo3(get, set)]
    pub sd_tolerance: f64,
    #[pyo3(get, set)]
    pub max_iter: u64,
    #[pyo3(get, set)]
    pub alpha: f64,
    #[pyo3(get, set)]
    pub gamma: f64,
    #[pyo3(get, set)]
    pub rho: f64,
    #[pyo3(get, set)]
    pub sigma: f64,
}

#[pymethods]
impl PyNelderMead {
    #[new]
    #[pyo3(signature = (
        simplex_delta = 0.1,
        sd_tolerance = EPSILON,
        max_iter = 300,
        alpha = 1.0,
        gamma = 2.0,
        rho = 0.5,
        sigma = 0.5,
    ))]
    fn new(
        simplex_delta: f64,
        sd_tolerance: f64,
        max_iter: u64,
        alpha: f64,
        gamma: f64,
        rho: f64,
        sigma: f64,
    ) -> Self {
        PyNelderMead {
            simplex_delta,
            sd_tolerance,
            max_iter,
            alpha,
            gamma,
            rho,
            sigma,
        }
    }
}

impl PyNelderMead {
    pub fn to_builder(&self) -> NelderMeadBuilder {
        NelderMeadBuilder::new(
            self.simplex_delta,
            self.sd_tolerance,
            self.max_iter,
            self.alpha,
            self.gamma,
            self.rho,
            self.sigma,
        )
    }
}

#[pyfunction]
#[pyo3(
    text_signature = "(simplex_delta: f64, sd_tolerance: f64, max_iter: u64, alpha: f64, gamma: f64, rho: f64, sigma: f64)"
)]
fn neldermead(
    simplex_delta: f64,
    sd_tolerance: f64,
    max_iter: u64,
    alpha: f64,
    gamma: f64,
    rho: f64,
    sigma: f64,
) -> PyNelderMead {
    PyNelderMead {
        simplex_delta,
        sd_tolerance,
        max_iter,
        alpha,
        gamma,
        rho,
        sigma,
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PySteepestDescent {
    #[pyo3(get, set)]
    pub max_iter: u64,
    #[pyo3(get, set)]
    pub line_search_params: PyLineSearchParams,
}

#[pymethods]
impl PySteepestDescent {
    #[new]
    #[pyo3(signature = (
        max_iter = 300,
        line_search_params = PyLineSearchParams {
            method: PyLineSearchMethod::MoreThuente(PyMoreThuente {
                c1: 1e-4,
                c2: 0.9,
                width_tolerance: 1e-10,
                bounds: vec![EPSILON.sqrt(), INFINITY],
            }),
        },
    ))]
    fn new(
        max_iter: u64,
        line_search_params: PyLineSearchParams,
    ) -> Self {
        PySteepestDescent {
            max_iter,
            line_search_params,
        }
    }
}

impl PySteepestDescent {
    pub fn to_builder(&self) -> SteepestDescentBuilder {
        let line_search_params = match &self.line_search_params.method {
            PyLineSearchMethod::MoreThuente(params) => LineSearchParams::morethuente()
                .c1(params.c1)
                .c2(params.c2)
                .width_tolerance(params.width_tolerance)
                .bounds(params.bounds.clone().into())
                .build(),
            PyLineSearchMethod::HagerZhang(params) => LineSearchParams::hagerzhang()
                .delta(params.delta)
                .sigma(params.sigma)
                .epsilon(params.epsilon)
                .theta(params.theta)
                .gamma(params.gamma)
                .eta(params.eta)
                .bounds(params.bounds.clone().into())
                .build(),
        };

        SteepestDescentBuilder::new(
            self.max_iter,
            line_search_params,
        )
    }
}

#[pyfunction]
#[pyo3(
    text_signature = "(max_iter: u64, line_search_params: Union[PyMoreThuente, PyHagerZhang])"
)]
fn steepestdescent(
    max_iter: u64,
    line_search_params: Py<pyo3::PyAny>,
    py: Python,
) -> PyResult<PySteepestDescent> {
    let line_search_params =
        if let Ok(params) = line_search_params.extract::<PyLineSearchParams>(py) {
            params
        } else if let Ok(more_thuente) = line_search_params.extract::<PyMoreThuente>(py) {
            PyLineSearchParams {
                method: PyLineSearchMethod::MoreThuente(more_thuente),
            }
        } else if let Ok(hager_zhang) = line_search_params.extract::<PyHagerZhang>(py) {
            PyLineSearchParams {
                method: PyLineSearchMethod::HagerZhang(hager_zhang),
            }
        } else {
            return Err(PyTypeError::new_err(
                "Expected PyLineSearchParams, PyMoreThuente, or PyHagerZhang",
            ));
        };

    Ok(PySteepestDescent {
        max_iter,
        line_search_params,
    })
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyNewtonCG {
    #[pyo3(get, set)]
    pub max_iter: u64,
    #[pyo3(get, set)]
    pub curvature_threshold: f64,
    #[pyo3(get, set)]
    pub tolerance: f64,
    #[pyo3(get, set)]
    pub line_search_params: PyLineSearchParams,
}

#[pymethods]
impl PyNewtonCG {
    #[new]
    #[pyo3(signature = (
        max_iter = 300,
        curvature_threshold = 0.0,
        tolerance = EPSILON,
        line_search_params = PyLineSearchParams {
            method: PyLineSearchMethod::MoreThuente(PyMoreThuente {
                c1: 1e-4,
                c2: 0.9,
                width_tolerance: 1e-10,
                bounds: vec![EPSILON.sqrt(), INFINITY],
            }),
        },
    ))]
    fn new(
        max_iter: u64,
        curvature_threshold: f64,
        tolerance: f64,
        line_search_params: PyLineSearchParams,
    ) -> Self {
        PyNewtonCG {
            max_iter,
            curvature_threshold,
            tolerance,
            line_search_params,
        }
    }
}

impl PyNewtonCG {
    pub fn to_builder(&self) -> NewtonCGBuilder {
        NewtonCGBuilder::new(
            self.max_iter,
            self.curvature_threshold,
            self.tolerance,
            match &self.line_search_params.method {
                PyLineSearchMethod::MoreThuente(params) => LineSearchParams::morethuente()
                    .c1(params.c1)
                    .c2(params.c2)
                    .width_tolerance(params.width_tolerance)
                    .bounds(params.bounds.clone().into())
                    .build(),
                PyLineSearchMethod::HagerZhang(params) => LineSearchParams::hagerzhang()
                    .delta(params.delta)
                    .sigma(params.sigma)
                    .epsilon(params.epsilon)
                    .theta(params.theta)
                    .gamma(params.gamma)
                    .eta(params.eta)
                    .bounds(params.bounds.clone().into())
                    .build(),
            },
        )
    }
}

#[pyfunction]
#[pyo3(
    text_signature = "(max_iter: u64, curvature_threshold: f64, tolerance: f64, line_search_params: Union[PyMoreThuente, PyHagerZhang])"
)]
fn newtoncg(
    max_iter: u64,
    curvature_threshold: f64,
    tolerance: f64,
    line_search_params: Py<pyo3::PyAny>,
    py: Python,
) -> PyResult<PyNewtonCG> {
let line_search_params =
        if let Ok(params) = line_search_params.extract::<PyLineSearchParams>(py) {
            params
        } else if let Ok(more_thuente) = line_search_params.extract::<PyMoreThuente>(py) {
            PyLineSearchParams {
                method: PyLineSearchMethod::MoreThuente(more_thuente),
            }
        } else if let Ok(hager_zhang) = line_search_params.extract::<PyHagerZhang>(py) {
            PyLineSearchParams {
                method: PyLineSearchMethod::HagerZhang(hager_zhang),
            }
        } else {
            return Err(PyTypeError::new_err(
                "Expected PyLineSearchParams, PyMoreThuente, or PyHagerZhang",
            ));
        };

    Ok(PyNewtonCG {
        max_iter,
        curvature_threshold,
        tolerance,
        line_search_params,
    })
}

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, PartialEq)]
pub enum PyTrustRegionRadiusMethod {
    Cauchy,
    Steihaug,
}

#[pymethods]
impl PyTrustRegionRadiusMethod {
    #[staticmethod]
    fn cauchy() -> Self {
        PyTrustRegionRadiusMethod::Cauchy
    }

    #[staticmethod]
    fn steihaug() -> Self {
        PyTrustRegionRadiusMethod::Steihaug
    }
}

impl From<PyTrustRegionRadiusMethod> for TrustRegionRadiusMethod {
    fn from(method: PyTrustRegionRadiusMethod) -> Self {
        match method {
            PyTrustRegionRadiusMethod::Cauchy => TrustRegionRadiusMethod::Cauchy,
            PyTrustRegionRadiusMethod::Steihaug => TrustRegionRadiusMethod::Steihaug,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyTrustRegion {
    #[pyo3(get, set)]
    pub trust_region_radius_method: PyTrustRegionRadiusMethod,
    #[pyo3(get, set)]
    pub max_iter: u64,
    #[pyo3(get, set)]
    pub radius: f64,
    #[pyo3(get, set)]
    pub max_radius: f64,
    #[pyo3(get, set)]
    pub eta: f64,
}

#[pymethods]
impl PyTrustRegion {
    #[new]
    #[pyo3(signature = (
        trust_region_radius_method = PyTrustRegionRadiusMethod::Cauchy,
        max_iter = 300,
        radius = 1.0,
        max_radius = 100.0,
        eta = 0.125,
    ))]
    fn new(
        trust_region_radius_method: PyTrustRegionRadiusMethod,
        max_iter: u64,
        radius: f64,
        max_radius: f64,
        eta: f64,
    ) -> Self {
        PyTrustRegion {
            trust_region_radius_method,
            max_iter,
            radius,
            max_radius,
            eta,
        }
    }
}

impl PyTrustRegion {
    pub fn to_builder(&self) -> TrustRegionBuilder {
        TrustRegionBuilder::new(
            self.trust_region_radius_method.clone().into(),
            self.max_iter,
            self.radius,
            self.max_radius,
            self.eta,
        )
    }
}

#[pyfunction]
#[pyo3(
    text_signature = "(trust_region_radius_method: PyTrustRegionRadiusMethod, max_iter: u64, radius: f64, max_radius: f64, eta: f64)"
)]
fn trustregion(
    trust_region_radius_method: PyTrustRegionRadiusMethod,
    max_iter: u64,
    radius: f64,
    max_radius: f64,
    eta: f64,
) -> PyTrustRegion {
    PyTrustRegion {
        trust_region_radius_method,
        max_iter,
        radius,
        max_radius,
        eta,
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyCOBYLA {
    #[pyo3(get, set)]
    pub max_iter: u64,
    #[pyo3(get, set)]
    pub step_size: f64,
    #[pyo3(get, set)]
    pub ftol_rel: Option<f64>,
    #[pyo3(get, set)]
    pub ftol_abs: Option<f64>,
    #[pyo3(get, set)]
    pub xtol_rel: Option<f64>,
    #[pyo3(get, set)]
    pub xtol_abs: Option<f64>,
}

#[pymethods]
impl PyCOBYLA {
    #[new]
    #[pyo3(signature = (
        max_iter = 300,
        step_size = 1.0,
        ftol_rel = None,
        ftol_abs = None,
        xtol_rel = None,
        xtol_abs = None,
    ))]
    fn new(
        max_iter: u64,
        step_size: f64,
        ftol_rel: Option<f64>,
        ftol_abs: Option<f64>,
        xtol_rel: Option<f64>,
        xtol_abs: Option<f64>,
    ) -> Self {
        PyCOBYLA {
            max_iter,
            step_size,
            ftol_rel,
            ftol_abs,
            xtol_rel,
            xtol_abs,
        }
    }
}

impl PyCOBYLA {
    pub fn to_builder(&self) -> COBYLABuilder {
        let mut builder = COBYLABuilder::new(
            self.max_iter,
            self.step_size,
        );
        
        if let Some(ftol_rel) = self.ftol_rel {
            builder = builder.ftol_rel(ftol_rel);
        }
        
        if let Some(ftol_abs) = self.ftol_abs {
            builder = builder.ftol_abs(ftol_abs);
        }
        
        if let Some(xtol_rel) = self.xtol_rel {
            builder = builder.xtol_rel(xtol_rel);
        }
        
        if let Some(xtol_abs) = self.xtol_abs {
            builder = builder.xtol_abs(xtol_abs);
        }
        
        builder
    }
}

#[pyfunction]
#[pyo3(signature = (
    max_iter = 300,
    step_size = 1.0,
    ftol_rel = None,
    ftol_abs = None,
    xtol_rel = None,
    xtol_abs = None,
))]
#[pyo3(
    text_signature = "(max_iter: int = 300, step_size: float = 1.0, ftol_rel: Optional[float] = None, ftol_abs: Optional[float] = None, xtol_rel: Optional[float] = None, xtol_abs: Optional[float] = None)"
)]
fn cobyla(
    max_iter: u64,
    step_size: f64,
    ftol_rel: Option<f64>,
    ftol_abs: Option<f64>,
    xtol_rel: Option<f64>,
    xtol_abs: Option<f64>,
) -> PyCOBYLA {
    PyCOBYLA {
        max_iter,
        step_size,
        ftol_rel,
        ftol_abs,
        xtol_rel,
        xtol_abs,
    }
}

/// Initialize the builders module
pub fn init_module(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLineSearchParams>()?;

    m.add_class::<PyHagerZhang>()?;
    m.add_function(wrap_pyfunction!(hagerzhang, m)?)?;
    m.setattr("HagerZhang", m.getattr("PyHagerZhang")?)?;

    m.add_class::<PyMoreThuente>()?;
    m.add_function(wrap_pyfunction!(morethuente, m)?)?;
    m.setattr("MoreThuente", m.getattr("PyMoreThuente")?)?;

    m.add_class::<PyLBFGS>()?;
    m.add_function(wrap_pyfunction!(lbfgs, m)?)?;
    m.setattr("LBFGS", m.getattr("PyLBFGS")?)?;

    m.add_class::<PyNelderMead>()?;
    m.add_function(wrap_pyfunction!(neldermead, m)?)?;
    m.setattr("nelder_mead", m.getattr("PyNelderMead")?)?;
    m.setattr("NelderMead", m.getattr("PyNelderMead")?)?;

    m.add_class::<PySteepestDescent>()?;
    m.add_function(wrap_pyfunction!(steepestdescent, m)?)?;
    m.setattr("SteepestDescent", m.getattr("PySteepestDescent")?)?;

    m.add_class::<PyNewtonCG>()?;
    m.add_function(wrap_pyfunction!(newtoncg, m)?)?;
    m.setattr("newton_cg", m.getattr("PyNewtonCG")?)?;
    m.setattr("NewtonCG", m.getattr("PyNewtonCG")?)?;

    m.add_class::<PyTrustRegionRadiusMethod>()?;
    m.setattr("TrustRegionRadiusMethod", m.getattr("PyTrustRegionRadiusMethod")?)?;
    m.setattr("PyTrustRegionRadiusMethod", m.getattr("PyTrustRegionRadiusMethod")?)?;

    m.add_class::<PyTrustRegion>()?;
    m.add_function(wrap_pyfunction!(trustregion, m)?)?;
    m.setattr("TrustRegion", m.getattr("PyTrustRegion")?)?;

    m.add_class::<PyCOBYLA>()?;
    m.add_function(wrap_pyfunction!(cobyla, m)?)?;
    m.setattr("COBYLA", m.getattr("PyCOBYLA")?)?;

    Ok(())
}
