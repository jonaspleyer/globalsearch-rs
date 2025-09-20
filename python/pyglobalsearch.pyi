import numpy as np
from numpy.typing import NDArray
from typing import Callable, List, Optional, TypedDict, Union, Type, Iterator

class Solution(TypedDict):
    """
    Represents the result of an optimization process.

    This is a compatibility type that matches the format used by SciPy's optimization functions.
    Use `PyLocalSolution` for the more feature-rich solution representation.

    Example:
        >>> result = optimize(problem, params)
        >>> best = result.best_solution()
        >>> solution_dict = {"x": best.x(), "fun": best.fun()}

    Attributes:
        x: The optimal solution point as a list of float values
        fun: The objective function value at the solution
    """

    x: List[float]
    fun: float

class PyLocalSolution:
    """
    A local solution in the parameter space.

    This class represents a solution point found by the optimization algorithm,
    including both the parameter values and the corresponding objective function value.
    Multiple PyLocalSolution objects are typically returned in a PySolutionSet.

    The class provides SciPy-compatible methods (`x()` and `fun()`) alongside
    direct attribute access (`point` and `objective`).

    Example:
        >>> solution = PyLocalSolution([1.0, 2.0], 3.5)
        >>> print(f"Point: {solution.x()}, Value: {solution.fun()}")
        Point: [1.0, 2.0], Value: 3.5

    Attributes:
        point: The solution coordinates in parameter space
        objective: The objective function value at this point
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
    Defines an optimization problem to be solved.

    Contains the objective function, variable bounds, and optionally
    gradient, hessian, and constraint functions, depending on the local solver used.

    All functions should accept numpy arrays and return appropriate types:
    - objective: (x: np.ndarray) -> float
    - gradient: (x: np.ndarray) -> np.ndarray
    - hessian: (x: np.ndarray) -> np.ndarray (2D)
    - constraints: List[(x: np.ndarray) -> float] where constraint(x) >= 0 means satisfied
    - variable_bounds: () -> np.ndarray of shape (n_vars, 2) with [lower, upper] bounds

    Examples:
        Basic unconstrained problem:
        >>> def objective(x): return x[0]**2 + x[1]**2
        >>> def bounds(): return np.array([[-5, 5], [-5, 5]])
        >>> problem = PyProblem(objective, bounds)

        Problem with gradient:
        >>> def gradient(x): return np.array([2*x[0], 2*x[1]])
        >>> problem = PyProblem(objective, bounds, gradient=gradient)

        Constrained problem:
        >>> def constraint(x): return x[0] + x[1] - 1  # x[0] + x[1] >= 1
        >>> problem = PyProblem(objective, bounds, constraints=[constraint])

    Attributes:
        objective: Function that computes the objective value to be minimized
        variable_bounds: Function that returns an array of [lower, upper] bounds for each variable
        gradient: Optional function that computes the gradient of the objective
        hessian: Optional function that computes the Hessian of the objective
        constraints: Optional list of constraint functions
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
    """
    Configuration for the COBYLA (Constrained Optimization BY Linear Approximations) solver.

    COBYLA is a derivative-free optimization algorithm that can handle inequality constraints.
    It's particularly useful when gradients are not available or when dealing with noisy
    objective functions.

    Examples:
        Basic usage:
        >>> cobyla_config = PyCOBYLA(max_iter=500, step_size=0.1)

        With per-variable tolerances:
        >>> # Different tolerance for each variable
        >>> cobyla_config = PyCOBYLA(xtol_abs=[1e-6, 1e-8])

        Using builder pattern:
        >>> cobyla_config = gs.builders.cobyla(
        ...     max_iter=1000,
        ...     xtol_abs=[1e-8] * n_vars  # Same tolerance for all variables
        ... )

    Attributes:
        max_iter: Maximum number of iterations
        step_size: Initial step size for the algorithm
        ftol_rel: Relative tolerance for function value convergence
        ftol_abs: Absolute tolerance for function value convergence
        xtol_rel: Relative tolerance for parameter convergence
        xtol_abs: Per-variable absolute tolerances for parameter convergence
    """

    max_iter: int
    step_size: float
    ftol_rel: Optional[float]
    ftol_abs: Optional[float]
    xtol_rel: Optional[float]
    xtol_abs: Optional[List[float]]
    def __init__(
        self,
        max_iter: int = 300,
        step_size: float = 1.0,
        ftol_rel: Optional[float] = None,
        ftol_abs: Optional[float] = None,
        xtol_rel: Optional[float] = None,
        xtol_abs: Optional[List[float]] = None,
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
        xtol_abs: Optional[List[float]] = None,
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
    Perform global optimization on the given problem.

    This function implements the OQNLP (OptQuest/NLP) algorithm, which combines
    scatter search metaheuristics with local optimization to find global minima
    of nonlinear problems. It's particularly effective for multi-modal functions
    with multiple local minima.

    The algorithm works in two stages:
    1. Scatter search to explore the parameter space and identify promising regions
    2. Local optimization from multiple starting points to refine solutions

    Examples:
        Basic optimization:
        >>> result = gs.optimize(problem, params)
        >>> best = result.best_solution()

        With custom solver configuration:
        >>> cobyla_config = gs.builders.cobyla(max_iter=1000)
        >>> result = gs.optimize(problem, params,
        ...                     local_solver="COBYLA",
        ...                     local_solver_config=cobyla_config)

        With early stopping:
        >>> result = gs.optimize(problem, params,
        ...                     target_objective=-1.0316,  # Stop when reached
        ...                     max_time=60.0,            # Max 60 seconds
        ...                     verbose=True)              # Show progress

    Args:
        problem: The optimization problem to solve (objective, bounds, constraints, etc.)
        params: Parameters controlling the optimization algorithm behavior
        local_solver: Local optimization algorithm ("COBYLA", "LBFGS", "NewtonCG",
                     "TrustRegion", "NelderMead", "SteepestDescent")
        local_solver_config: Custom configuration for the local solver (None for defaults)
        seed: Random seed for reproducible results (0 by default)
        target_objective: Stop optimization when this objective value is reached
        max_time: Maximum time in seconds for Stage 2 optimization (None = unlimited)
        verbose: Print progress information during optimization (False by default)
        exclude_out_of_bounds: Filter out solutions that violate bounds (False by default)

    Returns:
        PySolutionSet containing all solutions found, sorted by objective value.
        Use .best_solution() to get the best result, or iterate over all solutions.

    Raises:
        ValueError: If solver configuration doesn't match the specified solver type,
                   or if the problem is not properly defined.
    """
    ...
