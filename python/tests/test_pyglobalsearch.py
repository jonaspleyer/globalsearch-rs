import pyglobalsearch as gs
import numpy as np
from numpy.typing import NDArray

# Create the optimization parameters
params = gs.PyOQNLPParams(
    iterations=150,
    population_size=500,
    wait_cycle=5,
    threshold_factor=0.75,
    distance_factor=0.1,
)


# Function, variable bounds and gradient definitions
# Objective function
def obj(x: NDArray[np.float64]) -> float:
    return x[0] ** 2 + x[1] ** 2


# Gradient
def grad(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array(
        [
            2 * x[0],
            2 * x[1],
        ]
    )


# Hessian
def hess(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array([[2, 0], [0, 2]])


# Variable bounds
def variable_bounds() -> NDArray[np.float64]:
    return np.array([[-3, 3], [-2, 2]])


# Create a problem without a gradient or hessian
problem = gs.PyProblem(obj, variable_bounds)

# Create a problem with the gradient but not with a hessian
problem_grad = gs.PyProblem(obj, variable_bounds, grad)

# Create a problem with the gradient and hessian
problem_full = gs.PyProblem(obj, variable_bounds, grad, hess)


# Test that LBFGS fails when the gradient is not implemented
def test_gradient_not_implemented_lbfgs():
    try:
        gs.optimize(problem, params, local_solver="LBFGS")
    except Exception as e:
        assert (
            str(e)
            == "OQNLP Error: Local solver failed to find a solution. Local Solver Error: Failed to run local solver. Gradient not implemented and needed for local solver."
        )


# Test that SteepestDescent fails when the gradient is not implemented
def test_gradient_not_implemented_steepestdescent():
    try:
        gs.optimize(problem, params, local_solver="SteepestDescent")
    except Exception as e:
        assert (
            str(e)
            == "OQNLP Error: Local solver failed to find a solution. Local Solver Error: Failed to run local solver. Gradient not implemented and needed for local solver."
        )


# Test that NewtonCG fails when the hessian is not implemented
def test_hessian_not_implemented_newtoncg():
    try:
        gs.optimize(problem_grad, params, local_solver="NewtonCG")
    except Exception as e:
        assert (
            str(e)
            == "OQNLP Error: Local solver failed to find a solution. Local Solver Error: Failed to run local solver. Hessian not implemented and needed for local solver."
        )


# Test that TrustRegion fails when the hessian is not implemented
def test_hessian_not_implemented_trustregion():
    try:
        gs.optimize(problem_grad, params, local_solver="TrustRegion")
    except Exception as e:
        assert (
            str(e)
            == "OQNLP Error: Local solver failed to find a solution. Local Solver Error: Failed to run local solver. Hessian not implemented and needed for local solver."
        )


# Test that we throw an error when the local solver is not implemented
def test_invalid_local_solver():
    try:
        gs.optimize(problem, params, local_solver="InvalidLocalSolver")
    except Exception as e:
        assert str(e) == "Invalid solver type."


# Test successful optimization with LBFGS
def test_optimize_lbfgs():
    result = gs.optimize(problem_grad, params, local_solver="LBFGS")
    assert result is not None, "Optimization returned None"
    assert len(result) > 0, "Optimization returned empty result"
    assert result[0]["x"] == [0.0, 0.0]
    assert result[0]["fun"] == 0.0


## Test successful optimization with SteepestDescent
# This fails due to a bug in argmin:
# https://github.com/argmin-rs/argmin/issues/400
# It should be easy to solve by changing the local solver configuration
# But that is a current limitation of the Python bindings
# def test_optimize_steepestdescent():
#     assert result is not None, "Optimization returned None"
#     assert len(result) > 0, "Optimization returned empty result"
#     assert result[0]["x"] == [0.0, 0.0]
#     assert result[0]["fun"] == 0.0


## Test successful optimization with NewtonCG
# This fails due to a bug in argmin:
# https://github.com/argmin-rs/argmin/issues/400
# It should be easy to solve by changing the local solver configuration
# But that is a current limitation of the Python bindings
# def test_optimize_newtoncg():
#     assert result is not None, "Optimization returned None"
#     assert len(result) > 0, "Optimization returned empty result"
#     assert result[0]["x"] == [0.0, 0.0]
#     assert result[0]["fun"] == 0.0


# Test successful optimization with TrustRegion
def test_optimize_trustregion():
    result = gs.optimize(problem_full, params, local_solver="TrustRegion")
    assert result is not None, "Optimization returned None"
    assert len(result) > 0, "Optimization returned empty result"
    assert result[0]["x"] == [0.0, 0.0]
    assert result[0]["fun"] == 0.0


# Test configuring PyOQNLPParams
def test_configure_params():
    custom_params = gs.PyOQNLPParams(
        iterations=100,
        population_size=200,
        wait_cycle=3,
        threshold_factor=0.5,
        distance_factor=0.2,
    )
    result = gs.optimize(problem_grad, custom_params, local_solver="LBFGS")
    assert result is not None, "Optimization returned None"
    assert len(result) > 0, "Optimization returned empty result"
    assert result[0]["x"] == [0.0, 0.0]
    assert result[0]["fun"] == 0.0


# Test default PyOQNLPParams
def test_default_params():
    default_params = gs.PyOQNLPParams()
    result = gs.optimize(problem_grad, default_params, local_solver="LBFGS")
    assert result is not None, "Optimization returned None"
    assert len(result) > 0, "Optimization returned empty result"
    assert result[0]["x"] == [0.0, 0.0]
    assert result[0]["fun"] == 0.0


# Test LBFGS with problem_full (should work, ignoring Hessian)
def test_optimize_lbfgs_full():
    result = gs.optimize(problem_full, params, local_solver="LBFGS")
    assert result is not None, "Optimization returned None"
    assert len(result) > 0, "Optimization returned empty result"
    assert result[0]["x"] == [0.0, 0.0]
    assert result[0]["fun"] == 0.0


# Test TrustRegion with problem_grad (should fail, needs Hessian)
def test_trustregion_requires_hessian():
    try:
        gs.optimize(problem_grad, params, local_solver="TrustRegion")
    except Exception as e:
        assert (
            str(e)
            == "OQNLP Error: Local solver failed to find a solution. Local Solver Error: Failed to run local solver. Hessian not implemented and needed for local solver."
        )


def test_default_local_solver():
    """Test optimization with default local solver (should be LBFGS)"""
    result = gs.optimize(problem_grad, params)  # No local_solver specified
    assert result is not None, "Optimization returned None"
    assert len(result) > 0, "Optimization returned empty result"
    assert result[0]["x"] == [0.0, 0.0]
    assert result[0]["fun"] == 0.0


def test_default_seed():
    """Test optimization with default seed"""
    result1 = gs.optimize(
        problem_grad, params, local_solver="LBFGS"
    )  # No seed specified
    result2 = gs.optimize(
        problem_grad, params, local_solver="LBFGS", seed=0
    )  # Default seed
    assert result1 is not None and result2 is not None
    # Results should be identical with same seed
    assert result1[0]["x"] == result2[0]["x"]
    assert result1[0]["fun"] == result2[0]["fun"]


def test_different_seeds():
    """Test that different seeds can produce different intermediate behavior"""
    result1 = gs.optimize(problem_grad, params, local_solver="LBFGS", seed=42)
    result2 = gs.optimize(problem_grad, params, local_solver="LBFGS", seed=123)
    assert result1 is not None and result2 is not None
    # Both should find the global minimum but potentially through different paths
    assert abs(result1[0]["fun"]) < 1e-10
    assert abs(result2[0]["fun"]) < 1e-10


def test_lbfgs_custom_config():
    """Test LBFGS with custom configuration using builders"""
    # Create custom line search parameters
    morethuente_params = gs.builders.PyMoreThuente(
        c1=1e-5, c2=0.8, width_tolerance=1e-12
    )
    line_search = gs.builders.PyLineSearchParams(morethuente_params)

    # Create custom LBFGS configuration
    custom_lbfgs = gs.builders.PyLBFGS(
        max_iter=50,
        tolerance_grad=1e-6,
        tolerance_cost=1e-12,
        history_size=5,
        line_search_params=line_search,
    )
    result = gs.optimize(
        problem_grad, params, local_solver="LBFGS", local_solver_config=custom_lbfgs
    )
    assert result is not None, "Optimization returned None"
    assert len(result) > 0, "Optimization returned empty result"
    assert abs(result[0]["fun"]) < 1e-6  # Should still find minimum


def test_lbfgs_with_hagerzhang_line_search():
    """Test LBFGS with HagerZhang line search"""
    hagerzhang_params = gs.builders.PyHagerZhang(
        delta=0.05, sigma=0.95, epsilon=1e-8, theta=0.4, gamma=0.7, eta=0.02
    )
    line_search = gs.builders.PyLineSearchParams(hagerzhang_params)

    custom_lbfgs = gs.builders.PyLBFGS(
        max_iter=100,
        tolerance_grad=1e-8,
        history_size=8,
        line_search_params=line_search,
    )
    result = gs.optimize(
        problem_grad, params, local_solver="LBFGS", local_solver_config=custom_lbfgs
    )
    assert result is not None, "Optimization returned None"
    assert abs(result[0]["fun"]) < 1e-6


def test_nelder_mead_custom_config():
    """Test NelderMead with custom configuration"""
    custom_nelder_mead = gs.builders.PyNelderMead(
        simplex_delta=0.05,
        sd_tolerance=1e-8,
        max_iter=200,
        alpha=1.5,
        gamma=2.5,
        rho=0.3,
        sigma=0.3,
    )
    result = gs.optimize(
        problem,
        params,
        local_solver="NelderMead",
        local_solver_config=custom_nelder_mead,
    )
    assert result is not None, "Optimization returned None"
    assert abs(result[0]["fun"]) < 1e-4  # NelderMead may be less precise


def test_trustregion_custom_config():
    """Test TrustRegion with custom configuration"""
    custom_trustregion = gs.builders.PyTrustRegion(
        trust_region_radius_method=gs.builders.PyTrustRegionRadiusMethod.steihaug(),
        max_iter=150,
        radius=0.5,
        max_radius=50.0,
        eta=0.1,
    )
    result = gs.optimize(
        problem_full,
        params,
        local_solver="TrustRegion",
        local_solver_config=custom_trustregion,
    )
    assert result is not None, "Optimization returned None"
    assert abs(result[0]["fun"]) < 1e-6


def test_trustregion_cauchy_method():
    """Test TrustRegion with Cauchy method"""
    custom_trustregion = gs.builders.PyTrustRegion(
        trust_region_radius_method=gs.builders.PyTrustRegionRadiusMethod.cauchy(),
        max_iter=100,
    )
    result = gs.optimize(
        problem_full,
        params,
        local_solver="TrustRegion",
        local_solver_config=custom_trustregion,
    )
    assert result is not None, "Optimization returned None"
    assert abs(result[0]["fun"]) < 1e-6


def test_builders_lbfgs():
    """Test LBFGS created using builders module"""
    # Create line search params first
    morethuente = gs.builders.morethuente(
        c1=1e-5, c2=0.85, width_tolerance=1e-11, bounds=[1e-10, 1e10]
    )

    lbfgs_config = gs.builders.lbfgs(
        max_iter=80,
        tolerance_grad=1e-7,
        tolerance_cost=1e-14,
        history_size=6,
        line_search_params=morethuente,
    )
    result = gs.optimize(
        problem_grad, params, local_solver="LBFGS", local_solver_config=lbfgs_config
    )
    assert result is not None, "Optimization returned None"
    assert abs(result[0]["fun"]) < 1e-6


def test_builders_nelder_mead():
    """Test NelderMead created using builders module"""
    nelder_mead_config = gs.builders.neldermead(
        simplex_delta=0.08,
        sd_tolerance=1e-10,
        max_iter=250,
        alpha=1.2,
        gamma=2.2,
        rho=0.4,
        sigma=0.4,
    )
    result = gs.optimize(
        problem,
        params,
        local_solver="NelderMead",
        local_solver_config=nelder_mead_config,
    )
    assert result is not None, "Optimization returned None"
    assert abs(result[0]["fun"]) < 1e-4


def test_builders_trustregion():
    """Test TrustRegion created using builders module"""
    trustregion_config = gs.builders.trustregion(
        trust_region_radius_method=gs.builders.PyTrustRegionRadiusMethod.steihaug(),
        max_iter=120,
        radius=0.8,
        max_radius=80.0,
        eta=0.15,
    )
    result = gs.optimize(
        problem_full,
        params,
        local_solver="TrustRegion",
        local_solver_config=trustregion_config,
    )
    assert result is not None, "Optimization returned None"
    assert abs(result[0]["fun"]) < 1e-6


def test_builders_hagerzhang():
    """Test HagerZhang line search created using builders module"""
    hagerzhang_config = gs.builders.hagerzhang(
        delta=0.08,
        sigma=0.92,
        epsilon=1e-7,
        theta=0.45,
        gamma=0.68,
        eta=0.015,
        bounds=[1e-12, 1e12],
    )

    lbfgs_config = gs.builders.lbfgs(
        max_iter=100,
        tolerance_grad=1e-8,
        tolerance_cost=1e-12,
        history_size=10,
        line_search_params=hagerzhang_config,
    )
    result = gs.optimize(
        problem_grad, params, local_solver="LBFGS", local_solver_config=lbfgs_config
    )
    assert result is not None, "Optimization returned None"
    assert abs(result[0]["fun"]) < 1e-6


def test_single_variable_problem():
    """Test optimization with a single variable"""

    def single_var_obj(x: NDArray[np.float64]) -> float:
        return x[0] ** 2 - 4 * x[0] + 3  # Minimum at x=2, value=-1

    def single_var_grad(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([2 * x[0] - 4])

    def single_var_bounds() -> NDArray[np.float64]:
        return np.array([[-5, 5]])

    single_var_problem = gs.PyProblem(
        single_var_obj, single_var_bounds, single_var_grad
    )
    result = gs.optimize(single_var_problem, params, local_solver="LBFGS")
    assert result is not None, "Optimization returned None"
    assert abs(result[0]["x"][0] - 2.0) < 1e-6, (
        f"Expected x near 2.0, got {result[0]['x'][0]}"
    )
    assert abs(result[0]["fun"] - (-1.0)) < 1e-6, (
        f"Expected fun near -1.0, got {result[0]['fun']}"
    )


def test_convergence_tolerance():
    """Test that optimization respects different tolerance settings"""
    # Create line search params for both configs
    loose_morethuente = gs.builders.PyMoreThuente(c1=1e-4, c2=0.9)
    tight_morethuente = gs.builders.PyMoreThuente(c1=1e-4, c2=0.9)

    loose_line_search = gs.builders.PyLineSearchParams(loose_morethuente)
    tight_line_search = gs.builders.PyLineSearchParams(tight_morethuente)

    loose_lbfgs = gs.builders.PyLBFGS(
        max_iter=300,
        tolerance_grad=1e-4,  # Loose tolerance
        tolerance_cost=1e-8,
        history_size=10,
        line_search_params=loose_line_search,
    )
    result_loose = gs.optimize(
        problem_grad, params, local_solver="LBFGS", local_solver_config=loose_lbfgs
    )

    tight_lbfgs = gs.builders.PyLBFGS(
        max_iter=300,
        tolerance_grad=1e-10,  # Tight tolerance
        tolerance_cost=1e-12,
        history_size=10,
        line_search_params=tight_line_search,
    )
    result_tight = gs.optimize(
        problem_grad, params, local_solver="LBFGS", local_solver_config=tight_lbfgs
    )

    assert result_loose is not None and result_tight is not None
    # Both should find the solution, potentially with different precision
    assert abs(result_loose[0]["fun"]) < 1e-3
    assert abs(result_tight[0]["fun"]) < 1e-8


def test_solution_format():
    """Test that returned solutions have correct format"""
    result = gs.optimize(problem_grad, params, local_solver="LBFGS")
    assert result is not None, "Optimization returned None"
    assert len(result) > 0, "Should return at least one solution"

    for solution in result:
        assert "x" in solution, "Solution should have 'x' key"
        assert "fun" in solution, "Solution should have 'fun' key"
        assert isinstance(solution["x"], list), "x should be a list"
        assert isinstance(solution["fun"], float), "fun should be a float"
        assert len(solution["x"]) == 2, "x should have 2 elements for this problem"


def test_target_objective_reached():
    """Test that optimization stops when target objective is reached"""
    # Target should be easily reached (global minimum is 0.0)
    target = 0.1
    result = gs.optimize(
        problem_grad, params, local_solver="LBFGS", target_objective=target
    )
    assert result is not None, "Optimization returned None"
    assert len(result) > 0, "Should return at least one solution"

    # The best solution should be at or below the target
    best_solution = min(result, key=lambda s: s["fun"])
    assert best_solution["fun"] <= target, (
        f"Best objective {best_solution['fun']} should be <= target {target}"
    )


def test_max_time_parameter():
    """Test that max_time parameter limits optimization time"""
    short_time = 0.1

    import time

    start_time = time.time()
    result = gs.optimize(
        problem_grad, params, local_solver="LBFGS", max_time=short_time
    )
    elapsed_time = time.time() - start_time

    assert result is not None, "Optimization returned None"
    # Note: The actual elapsed time might be slightly longer due to setup overhead
    # but it should be reasonably close to the limit
    assert elapsed_time < short_time + 1.0, (
        f"Elapsed time {elapsed_time} should be close to limit {short_time}"
    )


def test_verbose_parameter():
    """Test that verbose parameter works without errors"""
    # This test mainly ensures verbose doesn't crash the optimization
    result = gs.optimize(problem_grad, params, local_solver="LBFGS", verbose=True)
    assert result is not None, "Optimization with verbose=True returned None"
    assert len(result) > 0, "Should return at least one solution"


def test_target_objective_and_max_time_combined():
    """Test combination of target_objective and max_time"""
    result = gs.optimize(
        problem_grad,
        params,
        local_solver="LBFGS",
        target_objective=0.1,
        max_time=5.0,
        verbose=False,
    )
    assert result is not None, "Optimization returned None"
    assert len(result) > 0, "Should return at least one solution"

    best_solution = min(result, key=lambda s: s["fun"])
    # Should reach target quickly (global minimum is 0.0)
    assert best_solution["fun"] <= 0.1, (
        f"Best objective {best_solution['fun']} should be <= target 0.1"
    )


def test_target_objective_none():
    """Test that target_objective=None works (should be equivalent to no target)"""
    result1 = gs.optimize(problem_grad, params, local_solver="LBFGS", seed=42)
    result2 = gs.optimize(
        problem_grad, params, local_solver="LBFGS", seed=42, target_objective=None
    )

    assert result1 is not None and result2 is not None
    # Results should be identical when using same seed and no target
    assert len(result1) == len(result2)
    assert abs(result1[0]["fun"] - result2[0]["fun"]) < 1e-10


def test_max_time_none():
    """Test that max_time=None works (should be equivalent to no time limit)"""
    result1 = gs.optimize(problem_grad, params, local_solver="LBFGS", seed=42)
    result2 = gs.optimize(
        problem_grad, params, local_solver="LBFGS", seed=42, max_time=None
    )

    assert result1 is not None and result2 is not None
    # Results should be identical when using same seed and no time limit
    assert len(result1) == len(result2)
    assert abs(result1[0]["fun"] - result2[0]["fun"]) < 1e-10


def test_verbose_false():
    """Test that verbose=False works (should be equivalent to default)"""
    result1 = gs.optimize(problem_grad, params, local_solver="LBFGS", seed=42)
    result2 = gs.optimize(
        problem_grad, params, local_solver="LBFGS", seed=42, verbose=False
    )

    assert result1 is not None and result2 is not None
    # Results should be identical when using same seed
    assert len(result1) == len(result2)
    assert abs(result1[0]["fun"] - result2[0]["fun"]) < 1e-10
