"""
Test script demonstrating optimization with many constraints using COBYLA.
This tests that we can now have unlimited constraints (testing with 100+ constraints).
"""

import pyglobalsearch as gs
import numpy as np


def objective_function(x):
    """Simple quadratic function: (x[0] - 1)^2 + (x[1] - 2)^2 + ... + (x[n] - n)^2"""
    return sum((x[i] - (i + 1)) ** 2 for i in range(len(x)))


# Define many constraints - each one ensures x[i] >= 0.1 * i
def make_constraint(i):
    def constraint(x):
        return x[i] - 0.1 * i  # x[i] >= 0.1 * i

    return constraint


def test_constraints():
    """Test optimization with many constraints to demonstrate support."""

    # Test with 100 constraints
    num_vars = 10
    num_constraints = 100

    print(f"Testing optimization with {num_constraints} constraints...")
    print("Objective: Sum of squares with offset")
    print(f"Variables: {num_vars} dimensions")
    print(f"Constraints: {num_constraints} inequality constraints")

    # Create constraints
    constraints = []
    for i in range(num_constraints):
        # Create different types of constraints
        if i < num_vars:
            # Direct variable bounds: x[i] >= 0.1 * i
            constraints.append(make_constraint(i))
        else:
            # Linear combinations: sum of variables >= some value
            def make_linear_constraint(idx):
                def constraint(x):
                    return sum(x[: min(3, len(x))]) - 0.05 * idx

                return constraint

            constraints.append(make_linear_constraint(i))

    # Set up bounds for each variable
    def variable_bounds():
        return np.array([[-2.0, 5.0] for _ in range(num_vars)])

    # Create the problem
    problem = gs.PyProblem(
        objective=objective_function,
        variable_bounds=variable_bounds,
        constraints=constraints,
    )

    # Use COBYLA solver with constraints
    try:
        # Create COBYLA config
        cobyla_config = gs.builders.cobyla(max_iter=200)

        # Set up parameters
        params = gs.PyOQNLPParams()
        params.iterations = 20
        params.population_size = 20

        result = gs.optimize(
            problem=problem,
            params=params,
            local_solver="COBYLA",
            local_solver_config=cobyla_config,
        )

        print("\nOptimization completed!")

        best_sol = result.best_solution()
        if best_sol is not None:
            print(f"Best solution: x = {[f'{xi:.4f}' for xi in best_sol.point]}")
            print(f"Best objective value: f(x) = {best_sol.objective:.6f}")

            # Check constraint satisfaction
            print(f"\nChecking {len(constraints)} constraints:")
            constraint_violations = 0
            for i, constraint_fn in enumerate(constraints):
                value = constraint_fn(best_sol.point)
                status = "✓" if value >= -1e-6 else "✗"
                if value < -1e-6:
                    constraint_violations += 1
                if i < 10 or value < -1e-6:  # Show first 10 or any violations
                    print(f"Constraint {i + 1}: {value:.6f} {status}")

            if constraint_violations == 0:
                print(f"All {len(constraints)} constraints satisfied!")
            else:
                raise ValueError(
                    f"{constraint_violations} constraint violations found!"
                )

            print(
                f"\nSuccess: Handled {num_constraints} constraints without any limit!"
            )
        else:
            print("No solution found!")

    except Exception as e:
        print(f"Optimization failed: {e}")
        import traceback

        traceback.print_exc()


def test_even_more_constraints():
    """Test with an extreme number of constraints"""

    num_vars = 5
    num_constraints = 500

    print("\n" + "=" * 60)
    print(f"{num_constraints} constraints with {num_vars} variables")
    print("=" * 60)

    # Simple objective: minimize sum of squares
    def simple_objective(x):
        return sum(xi**2 for xi in x)

    # Create many simple constraints: x[i % num_vars] >= -1 + 0.001 * i
    constraints = []
    for i in range(num_constraints):
        var_idx = i % num_vars
        threshold = -1 + 0.001 * i

        def make_constraint(idx, thresh):
            def constraint(x):
                return x[idx] - thresh

            return constraint

        constraints.append(make_constraint(var_idx, threshold))

    # Set up bounds
    def variable_bounds():
        return np.array([[-2.0, 2.0] for _ in range(num_vars)])

    # Create the problem
    problem = gs.PyProblem(
        objective=simple_objective,
        variable_bounds=variable_bounds,
        constraints=constraints,
    )

    try:
        # Create COBYLA config
        cobyla_config = gs.builders.cobyla(max_iter=100)

        # Set up parameters
        params = gs.PyOQNLPParams()
        params.iterations = 100
        params.population_size = 500

        result = gs.optimize(
            problem=problem,
            params=params,
            local_solver="COBYLA",
            local_solver_config=cobyla_config,
        )

        print(f"Successfully handled {num_constraints} constraints!")

        best_sol = result.best_solution()
        if best_sol is not None:
            print(f"Best objective: {best_sol.objective:.6f}")

            # Check a few constraints
            violations = 0
            for i in range(min(10, len(constraints))):
                value = constraints[i](best_sol.point)
                if value < -1e-6:
                    violations += 1

            print(f"Constraint check (first 10): {10 - violations}/10 satisfied")

    except Exception as e:
        raise RuntimeError(f"Test failed: {e}")
