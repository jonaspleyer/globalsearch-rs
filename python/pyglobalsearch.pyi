import numpy as np
from numpy.typing import NDArray
from typing import Callable, List, Optional, TypedDict, Union, Type, Iterator

class Solution(TypedDict):
    """
    # Represents the result of an optimization process.

    Attributes:
        x: The optimal solution point as a list of float values
        fun: The objective function value at the solution
    """

    x: List[float]
    fun: float

class PyLocalSolution:
    """
    # A local solution in the parameter space

    This class represents a solution point in the parameter space along with the objective function value at that point.
    """

    point: List[float]
    objective: float

    def __init__(self, point: List[float], objective: float) -> None:
        """
        # Initialize a local solution.

        Args:
            point: The solution point in the parameter space
            objective: The objective function value at the solution point
        """
        ...

    def fun(self) -> float:
        """
        # Returns the objective function value at the solution point

        Same as `objective` field

        This method is similar to the `fun` method in `SciPy.optimize` result
        """
        ...

    def x(self) -> List[float]:
        """
        # Returns the solution point as a list of float values

        Same as `point` field

        This method is similar to the `x` method in `SciPy.optimize` result
        """
        ...

class PySolutionSet:
    """
    # A set of local solutions

    This class represents a set of local solutions in the parameter space
    including the solution points and their corresponding objective function values.

    The solutions are stored as a list of `PyLocalSolution` objects.

    The `PySolutionSet` class supports indexing, iteration, and provides methods
    to get the number of solutions and find the best solution.
    """

    solutions: List[PyLocalSolution]

    def __init__(self, solutions: List[PyLocalSolution]) -> None:
        """
        # Initialize a solution set.

        Args:
            solutions: List of PyLocalSolution objects
        """
        ...

    def __len__(self) -> int:
        """
        # Returns the number of solutions stored in the set.
        """
        ...

    def is_empty(self) -> bool:
        """
        # Returns true if the solution set contains no solutions.
        """
        ...

    def best_solution(self) -> Optional[PyLocalSolution]:
        """
        # Returns the best solution in the set based on the objective function value.
        """
        ...

    def __getitem__(self, index: int) -> PyLocalSolution:
        """
        # Returns the solution at the given index.
        """
        ...

    def __iter__(self) -> Iterator[PyLocalSolution]:
        """
        # Returns an iterator over the solutions in the set.
        """
        ...

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
    gradient, hessian, and constraint functions, depending on the local solver used.

    The objective function, gradient and hessian should take a single
    argument, a numpy array of float values, and return a float, numpy array or
    numpy matrix, respectively.

    The variable bounds function should take no arguments and return a numpy
    array of float values representing the lower and upper bounds for each
    variable in the optimization problem.

    The constraints should be a list of constraint functions. Each constraint
    function should take a single argument (a numpy array of float values)
    and return a float representing the constraint value. A constraint is
    satisfied when constraint(x) >= 0.
    """

    objective: Callable[[NDArray[np.float64]], float]
    variable_bounds: Callable[[], NDArray[np.float64]]
    gradient: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]]
    hessian: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]]
    constraints: Optional[List[Callable[[NDArray[np.float64]], float]]]
    def __init__(
        self,
        objective: Callable[[NDArray[np.float64]], float],
        variable_bounds: Callable[[], NDArray[np.float64]],
        gradient: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]] = None,
        hessian: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]] = None,
        constraints: Optional[List[Callable[[NDArray[np.float64]], float]]] = None,
    ) -> None:
        """
        # Initialize an optimization problem.

        The objective function and the variable bounds are required.

        The gradient and hessian functions are optional, but should be provided
        if the local solver requires them.

        The constraints are optional and should be provided as a list of constraint
        functions if the local solver supports constraints (e.g., COBYLA).

        Args:
            objective: Function that computes the objective value to be minimized
            variable_bounds: Function that returns an array of [lower, upper] bounds for each variable
            gradient: Optional function that computes the gradient of the objective
            hessian: Optional function that computes the Hessian of the objective
            constraints: Optional list of constraint functions. Each constraint is satisfied when constraint(x) >= 0
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
    l1_coefficient: Optional[float]
    line_search_params: Union[
        PyLineSearchMethod, HagerZhang, MoreThuente, PyLineSearchParams
    ]
    def __init__(
        self,
        max_iter: int = 300,
        tolerance_grad: float = 1.490116119384766e-8,
        tolerance_cost: float = 2.220446049250313e-16,
        history_size: int = 10,
        l1_coefficient: Optional[float] = None,
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

class PyCOBYLA:
    max_iter: int
    step_size: float
    ftol_rel: Optional[float]
    ftol_abs: Optional[float]
    xtol_rel: Optional[float]
    xtol_abs: Optional[float]
    def __init__(
        self,
        max_iter: int = 300,
        step_size: float = 1.0,
        ftol_rel: Optional[float] = None,
        ftol_abs: Optional[float] = None,
        xtol_rel: Optional[float] = None,
        xtol_abs: Optional[float] = None,
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
        l1_coefficient: Optional[float] = None,
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
    @staticmethod
    def cobyla(
        max_iter: int = 300,
        step_size: float = 1.0,
        ftol_rel: Optional[float] = None,
        ftol_abs: Optional[float] = None,
        xtol_rel: Optional[float] = None,
        xtol_abs: Optional[float] = None,
    ) -> PyCOBYLA: ...

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
    PyCOBYLA: Type[PyCOBYLA]

def optimize(
    problem: PyProblem,
    params: PyOQNLPParams,
    local_solver: Optional[str] = "COBYLA",
    local_solver_config: Optional[
        Union[
            PyLBFGS,
            PyNelderMead,
            PySteepestDescent,
            PyNewtonCG,
            PyTrustRegion,
            PyCOBYLA,
        ]
    ] = None,
    seed: Optional[int] = 0,
    target_objective: Optional[float] = None,
    max_time: Optional[float] = None,
    verbose: Optional[bool] = False,
    exclude_out_of_bounds: Optional[bool] = False,
) -> PySolutionSet:
    """
    # Perform global optimization on the given problem.

    This function uses a hybrid global-local search algorithm to find the
    global minimum of the objective function defined in the problem.

    Args:
        problem: The optimization problem to solve
        params: Parameters controlling the optimization process
        local_solver: The algorithm to use for local optimization ("COBYLA" by default)
        local_solver_config: Configuration for the local solver (None for default)
        seed: Seed for reproducibility (0 by default)
        target_objective: Target objective value to stop early when reached (None to disable)
        max_time: Maximum time in seconds for Stage 2 optimization (None for unlimited)
        verbose: Enable verbose output during optimization (False by default)
        exclude_out_of_bounds: Exclude out-of-bounds solutions from consideration (False by default)

    Returns:
        A PySolutionSet object containing multiple solutions found and their
        objective values
    """
    ...
