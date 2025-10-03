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

    .. attribute: `x`
        :type: List[float]

        Parameter values at the solution point

    .. attribute: `fun`
        :type: float

        Objective function value at the solution point
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

    .. py:attribute: `point`
        :type: List[float]

        Parameter values at the solution point
    .. py:attribute: `objective`
        :type: float

        Objective function value at the solution point
    """

    point: List[float]
    objective: float

    def __init__(self, point: List[float], objective: float) -> None:
        """
        Initialize a local solution.

        :param point: The solution point in the parameter space
        :type point: List[float]
        :param objective: The objective function value at the solution point
        :type objective: float
        """
        ...

    def fun(self) -> float:
        """
        Returns the objective function value at the solution point.

        Same as `objective` field

        This method is similar to the `fun` method in `SciPy.optimize` result

        :return: The objective function value
        :rtype: float
        """
        ...

    def x(self) -> List[float]:
        """
        Returns the solution point as a list of float values.

        Same as `point` field

        This method is similar to the `x` method in `SciPy.optimize` result

        :return: The solution point in parameter space
        :rtype: List[float]
        """
        ...

class PySolutionSet:
    """
    A set of local solutions.

    This class represents a set of local solutions in the parameter space
    including the solution points and their corresponding objective function values.

    The solutions are stored as a list of `PyLocalSolution` objects.

    The `PySolutionSet` class supports indexing, iteration, and provides methods
    to get the number of solutions and find the best solution.
    """

    solutions: List[PyLocalSolution]

    def __init__(self, solutions: List[PyLocalSolution]) -> None:
        """
        Initialize a solution set.

        :param solutions: List of PyLocalSolution objects
        :type solutions: List[PyLocalSolution]
        """
        ...

    def __len__(self) -> int:
        """
        Returns the number of solutions stored in the set.

        :return: Number of solutions
        :rtype: int
        """
        ...

    def is_empty(self) -> bool:
        """
        Returns true if the solution set contains no solutions.

        :return: True if the solution set is empty, False otherwise
        :rtype: bool
        """
        ...

    def best_solution(self) -> Optional[PyLocalSolution]:
        """
        Returns the best solution in the set based on the objective function value.

        If the set is empty, returns None.

        :return: The best PyLocalSolution or None if the set is empty
        :rtype: Optional[PyLocalSolution]
        """
        ...

    def __getitem__(self, index: int) -> PyLocalSolution:
        """
        Returns the solution at the given index.

        :param index: Index of the solution to retrieve
        :type index: int
        :return: The PyLocalSolution at the specified index
        :rtype: PyLocalSolution
        """
        ...

    def __iter__(self) -> Iterator[PyLocalSolution]:
        """
        Returns an iterator over the solutions in the set.

        :return: An iterator over PyLocalSolution objects
        :rtype: Iterator[PyLocalSolution]
        """
        ...

class PyOQNLPParams:
    """
    Parameters for the OQNLP global optimization algorithm.

    Controls the behavior of the optimizer including population size,
    number of iterations, wait cycle, threshold and distance factor
    and seed.

    :param iterations: Maximum number of iterations to perform (default 300)
    :type iterations: int
    :param population_size: Size of the population for the global search (default 1000)
    :type population_size: int
    :param wait_cycle: Number of iterations to wait before terminating if no improvement (default 15)
    :type wait_cycle: int
    :param threshold_factor: Factor controlling the threshold for local searches (default 0.2)
    :type threshold_factor: float
    :param distance_factor: Factor controlling the minimum distance between solutions (default 0.75)
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
        Initialize optimization parameters.

        :param iterations: Maximum number of iterations to perform (default 300)
        :type iterations: int
        :param population_size: Size of the population for the global search (default 1000)
        :type population_size: int
        :param wait_cycle: Number of iterations to wait before terminating if no improvement (default 15)
        :type wait_cycle: int
        :param threshold_factor: Factor controlling the threshold for local searches (default 0.2)
        :type threshold_factor: float
        :param distance_factor: Factor controlling the minimum distance between solutions (default 0.75)
        :type distance_factor: float
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

    **Examples**

    Basic unconstrained problem::

        >>> def objective(x): return x[0]**2 + x[1]**2
        >>> def bounds(): return np.array([[-5, 5], [-5, 5]])
        >>> problem = PyProblem(objective, bounds)

    Problem with gradient::

        >>> def gradient(x): return np.array([2*x[0], 2*x[1]])
        >>> problem = PyProblem(objective, bounds, gradient=gradient)

    Constrained problem::

        >>> def constraint(x): return x[0] + x[1] - 1  # x[0] + x[1] >= 1
        >>> problem = PyProblem(objective, bounds, constraints=[constraint])
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
        Initialize an optimization problem.

        The objective function and the variable bounds are required.

        The gradient and hessian functions are optional, but should be provided
        if the local solver requires them.

        The constraints are optional and should be provided as a list of constraint
        functions if the local solver supports constraints (e.g., COBYLA).

        :param objective: Function that computes the objective value to be minimized
        :type objective: Callable[[NDArray[np.float64]], float]
        :param variable_bounds: Function that returns an array of [lower, upper] bounds for each variable
        :type variable_bounds: Callable[[], NDArray[np.float64]]
        :param gradient: Optional function that computes the gradient of the objective
        :type gradient: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]]
        :param hessian: Optional function that computes the Hessian of the objective
        :type hessian: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]]
        :param constraints: Optional list of constraint functions. Each constraint is satisfied when constraint(x) >= 0
        :type constraints: Optional[List[Callable[[NDArray[np.float64]], float]]]
        """
        ...

class PyLineSearchMethod:
    """
    Base class for line search methods.

    Line search methods are used in gradient-based optimization algorithms
    to determine the step size along the search direction. This class provides
    factory methods for creating specific line search configurations.

    Available methods:
        - Hager-Zhang: Robust line search with strong Wolfe conditions
        - More-Thuente: Efficient line search with cubic interpolation

    Examples
    --------
        # Using factory methods
        hz_method = PyLineSearchMethod.hagerzhang()
        mt_method = PyLineSearchMethod.morethunte()
    """
    @staticmethod
    def hagerzhang() -> "PyLineSearchMethod": ...
    @staticmethod
    def morethunte() -> "PyLineSearchMethod": ...

class HagerZhang(PyLineSearchMethod):
    """
    Hager-Zhang line search configuration.

    Implements the Hager-Zhang line search algorithm, which is a robust
    line search method that satisfies the strong Wolfe conditions and
    provides good performance for gradient-based optimization methods.

    Examples
    --------
        >>> hagerzhang_config = HagerZhang(delta=0.05, sigma=0.95)
    """

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
    ) -> None:
        """
        Initialize Hager-Zhang line search configuration.

        :param delta: Armijo parameter for sufficient decrease condition (default 0.1)
        :type delta: float
        :param sigma: Wolfe parameter for curvature condition (default 0.9)
        :type sigma: float
        :param epsilon: Tolerance for the line search termination (default 1e-6)
        :type epsilon: float
        :param theta: Parameter controlling the bracketing phase (default 0.5)
        :type theta: float
        :param gamma: Expansion factor for the bracketing phase (default 0.66)
        :type gamma: float
        :param eta: Contraction factor for the sectioning phase (default 0.01)
        :type eta: float
        :param bounds: Step size bounds [min, max] (default [1.49e-8, 1e20])
        :type bounds: List[float]
        """
        ...

class MoreThuente(PyLineSearchMethod):
    """
    More-Thuente line search configuration.

    Implements the More-Thuente line search algorithm, which uses cubic
    interpolation to efficiently find step sizes that satisfy the Wolfe
    conditions. This method is widely used in optimization algorithms.

    Examples
    --------
        >>> morethuente_config = MoreThuente(c1=1e-3, c2=0.8)
    """

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
    ) -> None:
        """
        Initialize More-Thuente line search configuration.

        :param c1: Armijo parameter for sufficient decrease condition (default 1e-4)
        :type c1: float
        :param c2: Wolfe parameter for curvature condition (default 0.9)
        :type c2: float
        :param width_tolerance: Tolerance for the interval width (default 1e-10)
        :type width_tolerance: float
        :param bounds: Step size bounds [min, max] (default [1.49e-8, 1e20])
        :type bounds: List[float]
        """
        ...

class PyLineSearchParams(PyLineSearchMethod):
    """
    Wrapper for line search parameters.

    This class provides a unified interface for different line search
    parameter configurations (HagerZhang or MoreThuente).

    Examples
    --------
        >>> hagerzhang_params = HagerZhang(delta=0.1)
        >>> line_search = PyLineSearchParams(hagerzhang_params)
    """

    params: Union[HagerZhang, MoreThuente]
    def __init__(self, params: Union[HagerZhang, MoreThuente]) -> None:
        """
        Initialize line search parameters wrapper.

        :param params: Line search configuration (HagerZhang or MoreThuente)
        :type params: Union[HagerZhang, MoreThuente]
        """
        ...

class PyLBFGS:
    """
    Configuration for the L-BFGS (Limited-memory BFGS) solver.

    L-BFGS is a quasi-Newton optimization method that approximates the
    Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm using limited memory.
    It's efficient for large-scale optimization problems requiring gradients.

    Examples
    --------
        >>> lbfgs_config = PyLBFGS(max_iter=500, history_size=20)
    """

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
    ) -> None:
        """
        Initialize L-BFGS solver configuration.

        :param max_iter: Maximum number of iterations (default 300)
        :type max_iter: int
        :param tolerance_grad: Gradient tolerance for convergence (default 1.49e-8)
        :type tolerance_grad: float
        :param tolerance_cost: Cost function tolerance for convergence (default 2.22e-16)
        :type tolerance_cost: float
        :param history_size: Number of previous gradients to store (default 10)
        :type history_size: int
        :param l1_coefficient: L1 regularization coefficient (optional)
        :type l1_coefficient: Optional[float]
        :param line_search_params: Line search configuration (default MoreThuente)
        :type line_search_params: Union[PyLineSearchMethod, HagerZhang, MoreThuente, PyLineSearchParams]
        """
        ...

class PyNelderMead:
    """
    Configuration for the Nelder-Mead simplex solver.

    Nelder-Mead is a derivative-free optimization method that uses a simplex
    (a geometric figure with n+1 vertices in n dimensions) to iteratively
    search for the minimum. It's particularly useful when gradients are not available.

    Examples
    --------
        >>> nelder_mead_config = PyNelderMead(max_iter=1000, alpha=1.5)
    """

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
    ) -> None:
        """
        Initialize Nelder-Mead solver configuration.

        :param simplex_delta: Initial simplex size (default 0.1)
        :type simplex_delta: float
        :param sd_tolerance: Standard deviation tolerance for convergence (default 2.22e-16)
        :type sd_tolerance: float
        :param max_iter: Maximum number of iterations (default 300)
        :type max_iter: int
        :param alpha: Reflection coefficient (default 1.0)
        :type alpha: float
        :param gamma: Expansion coefficient (default 2.0)
        :type gamma: float
        :param rho: Contraction coefficient (default 0.5)
        :type rho: float
        :param sigma: Shrink coefficient (default 0.5)
        :type sigma: float
        """
        ...

class PySteepestDescent:
    """
    Configuration for the steepest descent (gradient descent) solver.

    Steepest descent is a first-order optimization algorithm that iteratively
    moves in the direction of steepest descent (negative gradient) to find
    local minima. It requires gradient information.

    Examples
    --------
        >>> steepest_config = PySteepestDescent(max_iter=1000)
    """

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
    ) -> None:
        """
        Initialize steepest descent solver configuration.

        :param max_iter: Maximum number of iterations (default 300)
        :type max_iter: int
        :param line_search_params: Line search configuration (default MoreThuente)
        :type line_search_params: Union[PyLineSearchMethod, HagerZhang, MoreThuente, PyLineSearchParams]
        """
        ...

class PyNewtonCG:
    """
    Configuration for the Newton-CG (Newton Conjugate Gradient) solver.

    Newton-CG is a second-order optimization method that uses the conjugate
    gradient algorithm to approximately solve the Newton step. It's efficient
    for problems where the Hessian is large but can be computed or approximated.

    Examples
    --------
        >>> newton_cg_config = PyNewtonCG(max_iter=500, tolerance=1e-10)
    """

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
    ) -> None:
        """
        Initialize Newton-CG solver configuration.

        :param max_iter: Maximum number of iterations (default 300)
        :type max_iter: int
        :param curvature_tolerance: Tolerance for negative curvature detection (default 0.0)
        :type curvature_tolerance: float
        :param tolerance: Convergence tolerance for the Newton step (default 1.49e-8)
        :type tolerance: float
        :param line_search_params: Line search configuration (default MoreThuente)
        :type line_search_params: Union[PyLineSearchMethod, HagerZhang, MoreThuente, PyLineSearchParams]
        """
        ...

class PyTrustRegionRadiusMethod:
    """
    Trust region radius computation methods.

    This class provides factory methods for different approaches to computing
    the trust region radius in trust region optimization methods.

    Available methods:
        - Cauchy: Uses Cauchy point for trust region radius computation
        - Steihaug: Uses Steihaug's conjugate gradient approach

    Examples
    --------
        >>> cauchy_method = PyTrustRegionRadiusMethod.cauchy()
        >>> steihaug_method = PyTrustRegionRadiusMethod.steihaug()
    """
    @staticmethod
    def cauchy() -> "PyTrustRegionRadiusMethod": ...
    @staticmethod
    def steihaug() -> "PyTrustRegionRadiusMethod": ...

class PyTrustRegion:
    """
    Configuration for the trust region optimization solver.

    Trust region methods solve optimization problems by restricting steps to
    within a "trust region" where the quadratic model is considered reliable.
    The method adjusts the trust region size based on the agreement between
    the model and the actual function.

    Examples
    --------
        >>> trustregion_config = PyTrustRegion(radius=2.0, max_radius=50.0)
    """

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
    ) -> None:
        """
        Initialize trust region solver configuration.

        :param trust_region_radius_method: Method for computing trust region radius (default Cauchy)
        :type trust_region_radius_method: PyTrustRegionRadiusMethod
        :param max_iter: Maximum number of iterations (default 300)
        :type max_iter: int
        :param radius: Initial trust region radius (default 1.0)
        :type radius: float
        :param max_radius: Maximum allowed trust region radius (default 100.0)
        :type max_radius: float
        :param eta: Threshold for accepting/rejecting steps (default 0.125)
        :type eta: float
        """
        ...

class PyCOBYLA:
    """
    Configuration for the COBYLA (Constrained Optimization BY Linear Approximations) solver.

    COBYLA is a derivative-free optimization algorithm that can handle inequality constraints.
    It's particularly useful when gradients are not available or when dealing with noisy
    objective functions.

    Examples
    --------
    Basic usage::

        >>> cobyla_config = PyCOBYLA(max_iter=500, step_size=0.1)

    With per-variable tolerances::

        >>> # Different tolerance for each variable
        >>> cobyla_config = PyCOBYLA(xtol_abs=[1e-6, 1e-8])

    Using builder pattern::

        >>> cobyla_config = gs.builders.cobyla(
        ...     max_iter=1000,
        ...     xtol_abs=[1e-8] * n_vars  # Same tolerance for all variables
        ... )

    **Attributes**

    max_iter
        Maximum number of iterations
    step_size
        Initial step size for the algorithm
    ftol_rel
        Relative tolerance for function value convergence
    ftol_abs
        Absolute tolerance for function value convergence
    xtol_rel
        Relative tolerance for parameter convergence
    xtol_abs
        Per-variable absolute tolerances for parameter convergence
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
    ) -> HagerZhang:
        """
        Create a Hager-Zhang line search configuration.

        This builder function allows easy creation of a Hager-Zhang line search
        configuration with custom parameters for gradient-based optimization methods.

        Examples
        --------
            >>> hagerzhang_config = gs.builders.hagerzhang(delta=0.05, sigma=0.95)

        :param delta: Armijo parameter for sufficient decrease condition (default 0.1)
        :type delta: float
        :param sigma: Wolfe parameter for curvature condition (default 0.9)
        :type sigma: float
        :param epsilon: Tolerance for the line search termination (default 1e-6)
        :type epsilon: float
        :param theta: Parameter controlling the bracketing phase (default 0.5)
        :type theta: float
        :param gamma: Expansion factor for the bracketing phase (default 0.66)
        :type gamma: float
        :param eta: Contraction factor for the sectioning phase (default 0.01)
        :type eta: float
        :param bounds: Step size bounds [min, max] (default [1.49e-8, 1e20])
        :type bounds: List[float]
        :return: Configured Hager-Zhang line search
        :rtype: HagerZhang
        """
        ...
    @staticmethod
    def morethuente(
        c1: float = 1e-4,
        c2: float = 0.9,
        width_tolerance: float = 1e-10,
        bounds: List[float] = [1.490116119384766e-8, 1e20],
    ) -> MoreThuente:
        """
        Create a Moré-Thuente line search configuration.

        This builder function allows easy creation of a Moré-Thuente line search
        configuration with custom parameters for gradient-based optimization methods.

        Examples
        --------
            >>> morethuente_config = gs.builders.morethuente(c1=1e-3, c2=0.8)

        :param c1: Armijo parameter for sufficient decrease condition (default 1e-4)
        :type c1: float
        :param c2: Wolfe parameter for curvature condition (default 0.9)
        :type c2: float
        :param width_tolerance: Tolerance for the interval width (default 1e-10)
        :type width_tolerance: float
        :param bounds: Step size bounds [min, max] (default [1.49e-8, 1e20])
        :type bounds: List[float]
        :return: Configured Moré-Thuente line search
        :rtype: MoreThuente
        """
        ...
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
    ) -> "PyLBFGS":
        """
        Create an L-BFGS solver configuration.

        This builder function allows easy creation of an L-BFGS (Limited-memory
        Broyden-Fletcher-Goldfarb-Shanno) configuration with custom parameters.

        Examples
        --------
            >>> lbfgs_config = gs.builders.lbfgs(max_iter=500, history_size=20)

        :param max_iter: Maximum number of iterations (default 300)
        :type max_iter: int
        :param tolerance_grad: Gradient tolerance for convergence (default 1.49e-8)
        :type tolerance_grad: float
        :param tolerance_cost: Cost function tolerance for convergence (default 2.22e-16)
        :type tolerance_cost: float
        :param history_size: Number of previous gradients to store (default 10)
        :type history_size: int
        :param line_search_params: Line search configuration (default MoreThuente)
        :type line_search_params: Union[PyLineSearchMethod, HagerZhang, MoreThuente, PyLineSearchParams]
        :param l1_coefficient: L1 regularization coefficient (optional)
        :type l1_coefficient: float
        :return: Configured L-BFGS solver
        :rtype: PyLBFGS
        """
        ...
    @staticmethod
    def nelder_mead(
        simplex_delta: float = 0.1,
        sd_tolerance: float = 2.220446049250313e-16,
        max_iter: int = 300,
        alpha: float = 1.0,
        gamma: float = 2.0,
        rho: float = 0.5,
        sigma: float = 0.5,
    ) -> "PyNelderMead":
        """
        Create a Nelder-Mead solver configuration.

        This builder function allows easy creation of a Nelder-Mead simplex algorithm
        configuration.

        Examples
        --------
            >>> nelder_mead_config = gs.builders.nelder_mead(max_iter=1000, alpha=1.5)

        :param simplex_delta: Initial simplex size (default 0.1)
        :type simplex_delta: float
        :param sd_tolerance: Standard deviation tolerance for convergence (default 2.22e-16)
        :type sd_tolerance: float
        :param max_iter: Maximum number of iterations (default 300)
        :type max_iter: int
        :param alpha: Reflection coefficient (default 1.0)
        :type alpha: float
        :param gamma: Expansion coefficient (default 2.0)
        :type gamma: float
        :param rho: Contraction coefficient (default 0.5)
        :type rho: float
        :param sigma: Shrink coefficient (default 0.5)
        :type sigma: float
        :return: Configured Nelder-Mead solver
        :rtype: PyNelderMead
        """
        ...
    @staticmethod
    def neldermead(
        simplex_delta: float = 0.1,
        sd_tolerance: float = 2.220446049250313e-16,
        max_iter: int = 300,
        alpha: float = 1.0,
        gamma: float = 2.0,
        rho: float = 0.5,
        sigma: float = 0.5,
    ) -> "PyNelderMead":
        """
        Create a Nelder-Mead solver configuration.

        This builder function allows easy creation of a Nelder-Mead simplex algorithm
        configuration.

        Examples
        --------
            >>> neldermead_config = gs.builders.neldermead(max_iter=1000, alpha=1.5)

        :param simplex_delta: Initial simplex size (default 0.1)
        :type simplex_delta: float
        :param sd_tolerance: Standard deviation tolerance for convergence (default 2.22e-16)
        :type sd_tolerance: float
        :param max_iter: Maximum number of iterations (default 300)
        :type max_iter: int
        :param alpha: Reflection coefficient (default 1.0)
        :type alpha: float
        :param gamma: Expansion coefficient (default 2.0)
        :type gamma: float
        :param rho: Contraction coefficient (default 0.5)
        :type rho: float
        :param sigma: Shrink coefficient (default 0.5)
        :type sigma: float
        :return: Configured Nelder-Mead solver
        :rtype: PyNelderMead
        """
        ...
    @staticmethod
    def steepest_descent(
        max_iter: int = 300,
        line_search_params: Union[
            PyLineSearchMethod, HagerZhang, MoreThuente, "PyLineSearchParams"
        ] = PyLineSearchMethod.morethunte(),
    ) -> "PySteepestDescent":
        """
        Create a steepest descent solver configuration.

        This builder function allows easy creation of a steepest descent (gradient descent)
        configuration.

        Examples
        --------
            >>> steepest_config = gs.builders.steepest_descent(max_iter=1000)

        :param max_iter: Maximum number of iterations (default 300)
        :type max_iter: int
        :param line_search_params: Line search configuration (default MoreThuente)
        :type line_search_params: Union[PyLineSearchMethod, HagerZhang, MoreThuente, PyLineSearchParams]
        :return: Configured steepest descent solver
        :rtype: PySteepestDescent
        """
        ...
    @staticmethod
    def newton_cg(
        max_iter: int = 300,
        curvature_tolerance: float = 0.0,
        tolerance: float = 1.490116119384766e-8,
        line_search_params: Union[
            PyLineSearchMethod, HagerZhang, MoreThuente, "PyLineSearchParams"
        ] = PyLineSearchMethod.morethunte(),
    ) -> "PyNewtonCG":
        """
        Create a Newton-CG solver configuration.

        This builder function allows easy creation of a Newton-CG (Newton Conjugate Gradient)
        configuration.

        Examples
        --------
            >>> newton_cg_config = gs.builders.newton_cg(max_iter=500, tolerance=1e-10)

        :param max_iter: Maximum number of iterations (default 300)
        :type max_iter: int
        :param curvature_tolerance: Tolerance for negative curvature detection (default 0.0)
        :type curvature_tolerance: float
        :param tolerance: Convergence tolerance for the Newton step (default 1.49e-8)
        :type tolerance: float
        :param line_search_params: Line search configuration (default MoreThuente)
        :type line_search_params: Union[PyLineSearchMethod, HagerZhang, MoreThuente, PyLineSearchParams]
        :return: Configured Newton-CG solver
        :rtype: PyNewtonCG
        """
        ...
    @staticmethod
    def trustregion(
        trust_region_radius_method: PyTrustRegionRadiusMethod = PyTrustRegionRadiusMethod.cauchy(),
        max_iter: int = 300,
        radius: float = 1.0,
        max_radius: float = 100.0,
        eta: float = 0.125,
    ) -> PyTrustRegion:
        """
        Create a trust region solver configuration.

        This builder function allows easy creation of a trust region method
        configuration.

        Examples
        --------
            >>> trustregion_config = gs.builders.trustregion(radius=2.0, max_radius=50.0)

        :param trust_region_radius_method: Method for computing trust region radius (default Cauchy)
        :type trust_region_radius_method: PyTrustRegionRadiusMethod
        :param max_iter: Maximum number of iterations (default 300)
        :type max_iter: int
        :param radius: Initial trust region radius (default 1.0)
        :type radius: float
        :param max_radius: Maximum allowed trust region radius (default 100.0)
        :type max_radius: float
        :param eta: Threshold for accepting/rejecting steps (default 0.125)
        :type eta: float
        :return: Configured trust region solver
        :rtype: PyTrustRegion
        """
        ...
    @staticmethod
    def cobyla(
        max_iter: int = 300,
        step_size: float = 1.0,
        ftol_rel: Optional[float] = None,
        ftol_abs: Optional[float] = None,
        xtol_rel: Optional[float] = None,
        xtol_abs: Optional[List[float]] = None,
    ) -> PyCOBYLA:
        """
        Create a COBYLA solver configuration.

        This builder function allows easy creation of a COBYLA configuration
        with custom tolerances and parameters.

        Examples
        --------
            >>> cobyla_config = gs.builders.cobyla(max_iter=500, step_size=0.5)

        :param max_iter: Maximum number of iterations (default 300)
        :type max_iter: int
        :param step_size: Initial step size (default 1.0)
        :type step_size: float
        :param ftol_rel: Relative tolerance for function value convergence (optional)
        :type ftol_rel: float
        :param ftol_abs: Absolute tolerance for function value convergence (optional)
        :type ftol_abs: float
        :param xtol_rel: Relative tolerance for parameter convergence (optional)
        :type xtol_rel: float
        :param xtol_abs: Per-variable absolute tolerances for parameter convergence (optional)
        :type xtol_abs: List[float]
        :return: Configured COBYLA solver
        :rtype: PyCOBYLA


        """
        ...

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
    parallel: Optional[bool] = False,
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

    **Examples**

    Basic optimization::

        >>> result = gs.optimize(problem, params)
        >>> best = result.best_solution()

    With custom solver configuration::

        >>> cobyla_config = gs.builders.cobyla(max_iter=1000)
        >>> result = gs.optimize(problem, params,
        ...                     local_solver="COBYLA",
        ...                     local_solver_config=cobyla_config)

    With early stopping::

        >>> result = gs.optimize(problem, params,
        ...                     target_objective=-1.0316,  # Stop when reached
        ...                     max_time=60.0,             # Max 60 seconds
        ...                     verbose=True)              # Show progress

    :param problem: The optimization problem to solve (objective, bounds, constraints, etc.)
    :type problem: PyProblem
    :param params: Parameters controlling the optimization algorithm behavior
    :type params: PyOQNLPParams
    :param local_solver: Local optimization algorithm ("COBYLA", "LBFGS", "NewtonCG",
                        "TrustRegion", "NelderMead", "SteepestDescent")
    :type local_solver: str
    :param local_solver_config: Custom configuration for the local solver (None for defaults)
    :type local_solver_config: Union[PyLBFGS, PyNelderMead, PySteepestDescent, PyNewtonCG, PyTrustRegion, PyCOBYLA]
    :param seed: Random seed for reproducible results (0 by default)
    :type seed: int
    :param target_objective: Stop optimization when this objective value is reached (None by default = no target)
    :type target_objective: float
    :param max_time: Maximum time in seconds for Stage 2 optimization (None by default = unlimited)
    :type max_time: float
    :param verbose: Print progress information during optimization (False by default)
    :type verbose: bool
    :param exclude_out_of_bounds: Filter out solutions that violate bounds (False by default)
    :type exclude_out_of_bounds: bool
    :param parallel: Enable parallel processing using rayon (False by default)
    :type parallel: bool
    :return: A set of local solutions found during optimization
    :rtype: PySolutionSet
    :raises ValueError: If solver configuration doesn't match the specified solver type,
                        or if the problem is not properly defined.
    """
    ...
