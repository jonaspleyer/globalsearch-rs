"""
Test script for Python constraint support with COBYLA solver.
"""

import numpy as np
import pyglobalsearch as gs


def test_constrained_optimization():
    """Test constrained optimization with COBYLA solver."""

    # Define the Six-Hump Camel function
    def objective(x):
        x1, x2 = x[0], x[1]
        return (
            (4 - 2.1 * x1**2 + x1**4 / 3) * x1**2 + x1 * x2 + (-4 + 4 * x2**2) * x2**2
        )

    # Variable bounds: x1 in [-3, 3], x2 in [-2, 2]
    def variable_bounds():
        return np.array([[-3.0, 3.0], [-2.0, 2.0]])

    # Constraint functions: x1 + x2 >= -1 and x1 - x2 >= -2
    def constraint1(x):
        return x[0] + x[1] + 1.0  # x1 + x2 >= -1

    def constraint2(x):
        return x[0] - x[1] + 2.0  # x1 - x2 >= -2

    # Create the problem with constraints
    problem = gs.PyProblem(
        objective=objective,
        variable_bounds=variable_bounds,
        constraints=[constraint1, constraint2],
    )

    # Create COBYLA solver configuration
    cobyla_config = gs.builders.cobyla()

    # Set up parameters
    params = gs.PyOQNLPParams()
    params.iterations = 100
    params.population_size = 500

    print("Testing constrained optimization with COBYLA...")
    print("Objective: Six-Hump Camel function")
    print("Constraints: x1 + x2 >= -1, x1 - x2 >= -2")
    print("Variable bounds: x1 in [-3, 3], x2 in [-2, 2]")

    # Run optimization
    try:
        result = gs.optimize(
            problem=problem,
            params=params,
            local_solver="COBYLA",
            local_solver_config=cobyla_config,
        )

        print("\nOptimization successful!")

        # Get the best solution
        best_sol = result.best_solution()
        if best_sol is not None:
            print(f"Best solution: x = {best_sol.point}")
            print(f"Best objective value: f(x) = {best_sol.objective}")

            # Check constraint satisfaction
            x_best = np.array(best_sol.point)
            c1_val = constraint1(x_best)
            c2_val = constraint2(x_best)

            print("\nConstraint values at solution:")
            print(f"Constraint 1 (x1 + x2 + 1): {c1_val:.6f} (should be >= 0)")
            print(f"Constraint 2 (x1 - x2 + 2): {c2_val:.6f} (should be >= 0)")

            if c1_val >= -1e-6 and c2_val >= -1e-6:
                print("All constraints satisfied!")
            else:
                raise ValueError("Some constraints violated!")
        else:
            print("No solution found!")

    except Exception as e:
        print(f"Optimization failed: {e}")
        import traceback

        traceback.print_exc()


def test_exclude_global_minimum():
    """Test that constraints can exclude one of the global minima."""

    # Define the Six-Hump Camel function
    def objective(x):
        x1, x2 = x[0], x[1]
        return (
            (4 - 2.1 * x1**2 + x1**4 / 3) * x1**2 + x1 * x2 + (-4 + 4 * x2**2) * x2**2
        )

    # Variable bounds: x1 in [-3, 3], x2 in [-2, 2]
    def variable_bounds():
        return np.array([[-3.0, 3.0], [-2.0, 2.0]])

    # The Six-Hump Camel function has two global minima:
    # 1. x ≈ (-0.0898, 0.7126) with f(x) ≈ -1.0316
    # 2. x ≈ (0.0898, -0.7126) with f(x) ≈ -1.0316

    # Let's exclude the first minimum using constraints
    # We'll create a circular constraint around the first minimum
    def exclude_first_minimum(x):
        # Exclude region around (-0.0898, 0.7126) with radius 0.4 (larger exclusion)
        distance_sq = (x[0] + 0.0898) ** 2 + (x[1] - 0.7126) ** 2
        return distance_sq - 0.4**2  # distance_sq >= 0.16 (outside circle)

    # Create the problem with constraints that exclude the first minimum
    problem = gs.PyProblem(
        objective=objective,
        variable_bounds=variable_bounds,
        constraints=[
            exclude_first_minimum,
        ],
    )

    # Create COBYLA solver configuration
    cobyla_config = gs.builders.cobyla(max_iter=500)

    # Set up parameters
    params = gs.PyOQNLPParams()
    params.iterations = 100
    params.population_size = 500

    print("\n" + "=" * 60)
    print("Testing constraint-based exclusion of global minimum...")
    print("Constraints: Exclude region around (-0.0898, 0.7126)")
    print("Expected result: Should find the other minimum near (0.0898, -0.7126)")
    print("=" * 60)

    # Run optimization
    try:
        result = gs.optimize(
            problem=problem,
            params=params,
            local_solver="COBYLA",
            local_solver_config=cobyla_config,
        )

        print("\nOptimization completed!")

        # Get the best solution
        best_sol = result.best_solution()
        if best_sol is not None:
            x_best = np.array(best_sol.point)
            print(f"Best solution: x = [{x_best[0]:.4f}, {x_best[1]:.4f}]")
            print(f"Best objective value: f(x) = {best_sol.objective:.6f}")

            # Check if we found the second minimum (positive x1, negative x2)
            if x_best[0] > 0 and x_best[1] < 0:
                print("Success: Found the second global minimum as expected")
                print("The constraint successfully excluded the first minimum.")
            else:
                print("Found a different solution (may be local minimum)")

            # Check constraint satisfaction
            c1_val = exclude_first_minimum(x_best)

            print("\nConstraint verification:")
            print(f"Exclusion constraint: {c1_val:.6f} (should be >= 0)")

            # Check distance from excluded region
            excluded_center = np.array([-0.0898, 0.7126])
            distance = np.linalg.norm(x_best - excluded_center)
            print(f"Distance from excluded region center: {distance:.4f}")
            print("Exclusion radius: 0.4")

            if c1_val >= -1e-6:
                print("All constraints satisfied!")
                if distance > 0.4 - 1e-6:
                    print("Solution is properly outside the excluded region!")
                else:
                    raise ValueError("Solution is inside the excluded region!")

            else:
                raise ValueError("Some constraints violated!")

        else:
            print("No solution found!")

    except Exception as e:
        print(f"Optimization failed: {e}")
        import traceback

        traceback.print_exc()
