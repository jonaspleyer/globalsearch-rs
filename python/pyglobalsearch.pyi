import numpy as np
from numpy.typing import NDArray
from typing import Callable, List, Optional, TypedDict, Union, Type

class Solution(TypedDict):
    """
    # Represents the result of an optimization process.

    Attributes:
        x: The optimal solution point as a list of float values
        fun: The objective function value at the solution
    """

    x: List[float]
    fun: float

class PyOQNLPParams:
    """
    # Parameters for the OQNLP global optimization algorithm.

    Controls the behavior of the optimizer including population size,
    number of iterations, wait cycle, threshold and distance factor
    and seed.
    """

    iterations: int
    population_size: int
    wait_cycle: int
    threshold_factor: float
    distance_factor: float
    def __init__(
        self,
        iterations: int = 300,
        population_size: int = 1000,
        wait_cycle: int = 15,
        threshold_factor: float = 0.2,
        distance_factor: float = 0.75,
    ) -> None:
        """
        # Initialize optimization parameters.

        Args:
            iterations: Maximum number of iterations to perform
            population_size: Size of the population for the global search
            wait_cycle: Number of iterations to wait before terminating if no improvement
            threshold_factor: Factor controlling the threshold for local searches
            distance_factor: Factor controlling the minimum distance between solutions
        """
        ...

class PyProblem:
    """
    # Defines an optimization problem to be solved.

    Contains the objective function, variable bounds, and optionally
    gradient and hessian functions, depending on the local solver used.

    The objective function, gradient and hessian should take a single
    argument, a numpy array of float values, and return a float, numpy array or
    numpy matrix, respectively.

    The variable bounds function should take no arguments and return a numpy
    array of float values representing the lower and upper bounds for each
    variable in the optimization problem.
    """

    objective: Callable[[NDArray[np.float64]], float]
    variable_bounds: Callable[[], NDArray[np.float64]]
    gradient: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]]
    hessian: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]]
    def __init__(
        self,
        objective: Callable[[NDArray[np.float64]], float],
        variable_bounds: Callable[[], NDArray[np.float64]],
        gradient: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]] = None,
        hessian: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]] = None,
    ) -> None:
        """
        # Initialize an optimization problem.

        The objective function and the variable bounds are required.

        The gradient and hessian functions are optional, but should be provided
        if the local solver requires them.

        Args:
            objective: Function that computes the objective value to be minimized
            variable_bounds: Function that returns an array of [lower, upper] bounds for each variable
            gradient: Optional function that computes the gradient of the objective
            hessian: Optional function that computes the Hessian of the objective
        """
        ...

class PyLineSearchMethod:
    @staticmethod
    def hagerzhang() -> "PyLineSearchMethod": ...
    @staticmethod
    def morethunte() -> "PyLineSearchMethod": ...

class HagerZhang(PyLineSearchMethod):
    delta: float
    sigma: float
    epsilon: float
    theta: float
    gamma: float
    eta: float
    bounds: List[float]
    def __init__(
        self,
        delta: float = 0.1,
        sigma: float = 0.9,
        epsilon: float = 1e-6,
        theta: float = 0.5,
        gamma: float = 0.66,
        eta: float = 0.01,
        bounds: List[float] = [1.490116119384766e-8, 10e20],
    ) -> None: ...

class MoreThuente(PyLineSearchMethod):
    c1: float
    c2: float
    width_tolerance: float
    bounds: List[float]
    def __init__(
        self,
        c1: float = 1e-4,
        c2: float = 0.9,
        width_tolerance: float = 1e-10,
        bounds: List[float] = [1.490116119384766e-8, 10e20],
    ) -> None: ...

class PyLineSearchParams(PyLineSearchMethod):
    params: Union[HagerZhang, MoreThuente]
    def __init__(self, params: Union[HagerZhang, MoreThuente]) -> None: ...

class PyLBFGS:
    max_iter: int
    tolerance_grad: float
    tolerance_cost: float
    history_size: int
    line_search_params: Union[
        PyLineSearchMethod, HagerZhang, MoreThuente, PyLineSearchParams
    ]
    def __init__(
        self,
        max_iter: int = 300,
        tolerance_grad: float = 1.490116119384766e-8,
        tolerance_cost: float = 2.220446049250313e-16,
        history_size: int = 10,
        line_search_params: Union[
            PyLineSearchMethod, HagerZhang, MoreThuente, PyLineSearchParams
        ] = MoreThuente(),
    ) -> None: ...

class PyNelderMead:
    simplex_delta: float
    sd_tolerance: float
    max_iter: int
    alpha: float
    gamma: float
    rho: float
    sigma: float
    def __init__(
        self,
        simplex_delta: float = 0.1,
        sd_tolerance: float = 2.220446049250313e-16,
        max_iter: int = 300,
        alpha: float = 1.0,
        gamma: float = 2.0,
        rho: float = 0.5,
        sigma: float = 0.5,
    ) -> None: ...

class PySteepestDescent:
    max_iter: int
    line_search_params: Union[
        PyLineSearchMethod, HagerZhang, MoreThuente, PyLineSearchParams
    ]
    def __init__(
        self,
        max_iter: int = 300,
        line_search_params: Union[
            PyLineSearchMethod, HagerZhang, MoreThuente, PyLineSearchParams
        ] = PyLineSearchMethod.morethunte(),
    ) -> None: ...

class PyNewtonCG:
    max_iter: int
    curvature_tolerance: float
    tolerance: float
    line_search_params: Union[
        PyLineSearchMethod, HagerZhang, MoreThuente, PyLineSearchParams
    ]
    def __init__(
        self,
        max_iter: int = 300,
        curvature_tolerance: float = 0.0,
        tolerance: float = 1.490116119384766e-8,
        line_search_params: Union[
            PyLineSearchMethod, HagerZhang, MoreThuente, PyLineSearchParams
        ] = PyLineSearchMethod.morethunte(),
    ) -> None: ...

class PyTrustRegionRadiusMethod:
    @staticmethod
    def cauchy() -> "PyTrustRegionRadiusMethod": ...
    @staticmethod
    def steihaug() -> "PyTrustRegionRadiusMethod": ...

class PyTrustRegion:
    trust_region_radius_method: PyTrustRegionRadiusMethod
    max_iter: int
    radius: float
    max_radius: float
    eta: float
    def __init__(
        self,
        trust_region_radius_method: PyTrustRegionRadiusMethod = PyTrustRegionRadiusMethod.cauchy(),
        max_iter: int = 300,
        radius: float = 1.0,
        max_radius: float = 100.0,
        eta: float = 0.125,
    ) -> None: ...

class builders:
    @staticmethod
    def hagerzhang(
        delta: float = 0.1,
        sigma: float = 0.9,
        epsilon: float = 1e-6,
        theta: float = 0.5,
        gamma: float = 0.66,
        eta: float = 0.01,
        bounds: List[float] = [1.490116119384766e-8, 10e20],
    ) -> HagerZhang: ...
    @staticmethod
    def morethunte(
        c1: float = 1e-4,
        c2: float = 0.9,
        width_tolerance: float = 1e-10,
        bounds: List[float] = [1.490116119384766e-8, 1e20],
    ) -> MoreThuente: ...
    @staticmethod
    def morethuente(
        c1: float = 1e-4,
        c2: float = 0.9,
        width_tolerance: float = 1e-10,
        bounds: List[float] = [1.490116119384766e-8, 1e20],
    ) -> MoreThuente: ...
    @staticmethod
    def lbfgs(
        max_iter: int = 300,
        tolerance_grad: float = 1.490116119384766e-8,
        tolerance_cost: float = 2.220446049250313e-16,
        history_size: int = 10,
        line_search_params: Union[
            PyLineSearchMethod, HagerZhang, MoreThuente, "PyLineSearchParams"
        ] = MoreThuente(),
    ) -> "PyLBFGS": ...
    @staticmethod
    def nelder_mead(
        simplex_delta: float = 0.1,
        sd_tolerance: float = 2.220446049250313e-16,
        max_iter: int = 300,
        alpha: float = 1.0,
        gamma: float = 2.0,
        rho: float = 0.5,
        sigma: float = 0.5,
    ) -> "PyNelderMead": ...
    @staticmethod
    def neldermead(
        simplex_delta: float = 0.1,
        sd_tolerance: float = 2.220446049250313e-16,
        max_iter: int = 300,
        alpha: float = 1.0,
        gamma: float = 2.0,
        rho: float = 0.5,
        sigma: float = 0.5,
    ) -> "PyNelderMead": ...
    @staticmethod
    def steepest_descent(
        max_iter: int = 300,
        line_search_params: Union[
            PyLineSearchMethod, HagerZhang, MoreThuente, "PyLineSearchParams"
        ] = PyLineSearchMethod.morethunte(),
    ) -> "PySteepestDescent": ...
    @staticmethod
    def newton_cg(
        max_iter: int = 300,
        curvature_tolerance: float = 0.0,
        tolerance: float = 1.490116119384766e-8,
        line_search_params: Union[
            PyLineSearchMethod, HagerZhang, MoreThuente, "PyLineSearchParams"
        ] = PyLineSearchMethod.morethunte(),
    ) -> "PyNewtonCG": ...
    @staticmethod
    def trustregion(
        trust_region_radius_method: PyTrustRegionRadiusMethod = PyTrustRegionRadiusMethod.cauchy(),
        max_iter: int = 300,
        radius: float = 1.0,
        max_radius: float = 100.0,
        eta: float = 0.125,
    ) -> PyTrustRegion: ...

    # Aliases to global class definitions
    PyHagerZhang: Type[HagerZhang]
    PyMoreThuente: Type[MoreThuente]
    PyLineSearchParams: Type[PyLineSearchParams]

    PyLBFGS: Type[PyLBFGS]
    PyNelderMead: Type[PyNelderMead]
    PySteepestDescent: Type[PySteepestDescent]
    PyNewtonCG: Type[PyNewtonCG]
    PyTrustRegionRadiusMethod: Type[PyTrustRegionRadiusMethod]
    PyTrustRegion: Type[PyTrustRegion]

def optimize(
    problem: PyProblem,
    params: PyOQNLPParams,
    local_solver: Optional[str] = "LBFGS",
    local_solver_config: Optional[
        Union[PyLBFGS, PyNelderMead, PySteepestDescent, PyNewtonCG, PyTrustRegion]
    ] = None,
    seed: Optional[int] = 0,
) -> Optional[List[Solution]]:
    """
    # Perform global optimization on the given problem.

    This function uses a hybrid global-local search algorithm to find the
    global minimum of the objective function defined in the problem.

    Args:
        problem: The optimization problem to solve
        params: Parameters controlling the optimization process
        local_solver: The algorithm to use for local optimization ("LBFGS" by default)
        seed: Seed for reproducibility (0 by default)

    Returns:
        A list of Solution objects containing multiple solutions found and their
        objective values or none if no solution is found
    """
    ...
