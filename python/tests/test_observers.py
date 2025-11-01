"""
Observer tests for pyglobalsearch.

Tests the observer functionality for monitoring optimization progress.
"""

import pyglobalsearch as gs
import numpy as np
from numpy.typing import NDArray


# Test problem definitions
def simple_obj(x: NDArray[np.float64]) -> float:
    """Simple quadratic objective function."""
    return x[0] ** 2 + x[1] ** 2


def simple_bounds() -> NDArray[np.float64]:
    """Bounds for the simple quadratic problem."""
    return np.array([[-3, 3], [-3, 3]])


# Create test problem and parameters
simple_problem = gs.PyProblem(simple_obj, simple_bounds)
test_params = gs.PyOQNLPParams(
    iterations=50,  # Smaller for faster tests
    population_size=100,
    wait_cycle=3,
    threshold_factor=0.5,
    distance_factor=0.5,
)


class TestObserverCreation:
    """Test observer creation and basic configuration."""

    def test_observer_creation(self):
        """Test creating a new observer."""
        observer = gs.observers.Observer()
        assert observer is not None
        assert not observer.should_observe_stage1
        assert not observer.should_observe_stage2
        assert not observer.is_timing_enabled

    def test_observer_with_stage1_tracking(self):
        """Test observer with Stage 1 tracking enabled."""
        observer = gs.observers.Observer()
        observer.with_stage1_tracking()
        assert observer.should_observe_stage1
        assert not observer.should_observe_stage2

    def test_observer_with_stage2_tracking(self):
        """Test observer with Stage 2 tracking enabled."""
        observer = gs.observers.Observer()
        observer.with_stage2_tracking()
        assert not observer.should_observe_stage1
        assert observer.should_observe_stage2

    def test_observer_with_both_stages(self):
        """Test observer with both stages tracking enabled."""
        observer = gs.observers.Observer()
        observer.with_stage1_tracking()
        observer.with_stage2_tracking()
        assert observer.should_observe_stage1
        assert observer.should_observe_stage2

    def test_observer_with_timing(self):
        """Test observer with timing enabled."""
        observer = gs.observers.Observer()
        observer.with_timing()
        assert observer.is_timing_enabled

    def test_observer_modes(self):
        """Test observer mode configuration."""
        observer = gs.observers.Observer()
        observer.with_stage1_tracking()
        observer.with_stage2_tracking()

        # Test Stage1Only mode
        observer.with_mode(gs.observers.ObserverMode.Stage1Only)
        assert observer.should_observe_stage1
        assert not observer.should_observe_stage2

        # Test Stage2Only mode
        observer.with_mode(gs.observers.ObserverMode.Stage2Only)
        assert not observer.should_observe_stage1
        assert observer.should_observe_stage2

        # Test Both mode
        observer.with_mode(gs.observers.ObserverMode.Both)
        assert observer.should_observe_stage1
        assert observer.should_observe_stage2


class TestObserverCallbacks:
    """Test observer callback functionality."""

    def test_default_callbacks(self):
        """Test that default callbacks can be created without error."""
        observer = gs.observers.Observer()

        # Test default callback
        observer.with_default_callback()

        # Test stage-specific callbacks
        observer.with_stage1_callback()
        observer.with_stage2_callback()

    def test_callback_frequency(self):
        """Test callback frequency configuration."""
        observer = gs.observers.Observer()
        observer.with_callback_frequency(5)
        # Should automatically create default callback when frequency is set

    def test_custom_callback(self):
        """Test custom Python callback."""
        callback_calls = []

        def custom_callback(stage1, stage2):
            callback_calls.append((stage1, stage2))

        observer = gs.observers.Observer()
        observer.with_callback(custom_callback)

        # Run optimization with observer
        result = gs.optimize(simple_problem, test_params, observer=observer)

        # Callback should have been called at least once
        assert len(callback_calls) > 0

        # Check callback arguments
        for stage1, stage2 in callback_calls:
            # stage1 and stage2 can be None depending on stage
            if stage1 is not None:
                assert hasattr(stage1, "best_objective")
                assert hasattr(stage1, "function_evaluations")
            if stage2 is not None:
                assert hasattr(stage2, "best_objective")
                assert hasattr(stage2, "current_iteration")


class TestObserverStateAccess:
    """Test observer state access methods."""

    def test_stage1_state_access(self):
        """Test Stage 1 state access."""
        observer = gs.observers.Observer()
        observer.with_stage1_tracking()

        # Before optimization, stage1 should be accessible
        stage1 = observer.stage1()
        assert stage1 is not None
        assert hasattr(stage1, "best_objective")
        assert hasattr(stage1, "function_evaluations")
        assert hasattr(stage1, "reference_set_size")

    def test_stage2_state_access(self):
        """Test Stage 2 state access."""
        observer = gs.observers.Observer()
        observer.with_stage2_tracking()

        # Before Stage 2 starts, stage2 should be None
        stage2 = observer.stage2()
        assert stage2 is None

    def test_stage1_final_access(self):
        """Test Stage 1 final state access."""
        observer = gs.observers.Observer()
        observer.with_stage1_tracking()
        observer.with_mode(gs.observers.ObserverMode.Both)

        # Run optimization
        result = gs.optimize(simple_problem, test_params, observer=observer)

        # After optimization, stage1_final should be accessible
        stage1_final = observer.stage1_final()
        assert stage1_final is not None
        assert stage1_final.function_evaluations > 0
        assert not np.isnan(stage1_final.best_objective)


class TestObserverIntegration:
    """Test observer integration with optimization."""

    def test_observer_with_optimization_stage1(self):
        """Test observer tracking Stage 1 during optimization."""
        observer = gs.observers.Observer()
        observer.with_stage1_tracking()
        observer.with_timing()
        observer.with_mode(gs.observers.ObserverMode.Both)

        result = gs.optimize(simple_problem, test_params, observer=observer)

        # Check that we got results
        assert result is not None
        assert len(result) > 0

        # Check Stage 1 final statistics
        stage1 = observer.stage1_final()
        assert stage1 is not None
        assert stage1.function_evaluations > 0
        assert stage1.reference_set_size > 0
        assert stage1.trial_points_generated >= 0

        # Check timing if enabled
        if observer.is_timing_enabled:
            assert stage1.total_time is not None
            assert stage1.total_time > 0

    def test_observer_with_optimization_stage2(self):
        """Test observer tracking Stage 2 during optimization."""
        observer = gs.observers.Observer()
        observer.with_stage2_tracking()
        observer.with_timing()

        result = gs.optimize(simple_problem, test_params, observer=observer)

        # Check that we got results
        assert result is not None
        assert len(result) > 0

        # Check Stage 2 statistics
        stage2 = observer.stage2()
        if stage2 is not None:  # Stage 2 might not run for simple problems
            assert stage2.function_evaluations >= 0
            assert stage2.solution_set_size >= 0
            assert stage2.current_iteration >= 0
            assert not np.isnan(stage2.best_objective)

            # Check timing if enabled
            if observer.is_timing_enabled:
                assert stage2.total_time is not None
                assert stage2.total_time >= 0

    def test_observer_with_both_stages(self):
        """Test observer tracking both stages during optimization."""
        observer = gs.observers.Observer()
        observer.with_stage1_tracking()
        observer.with_stage2_tracking()
        observer.with_timing()
        observer.with_mode(gs.observers.ObserverMode.Both)

        result = gs.optimize(simple_problem, test_params, observer=observer)

        # Check that we got results
        assert result is not None
        assert len(result) > 0

        # Check Stage 1 final statistics
        stage1 = observer.stage1_final()
        assert stage1 is not None
        assert stage1.function_evaluations > 0

        # Check Stage 2 statistics (may be None if Stage 2 didn't run)
        stage2 = observer.stage2()
        if stage2 is not None:
            assert stage2.function_evaluations >= 0

    def test_observer_with_default_callback(self):
        """Test observer with default callback during optimization."""
        observer = gs.observers.Observer()
        observer.with_stage1_tracking()
        observer.with_stage2_tracking()
        observer.with_default_callback()

        # This should not raise an exception
        result = gs.optimize(simple_problem, test_params, observer=observer)
        assert result is not None
        assert len(result) > 0

    def test_observer_elapsed_time(self):
        """Test observer elapsed time tracking."""
        observer = gs.observers.Observer()
        observer.with_timing()
        observer.with_mode(gs.observers.ObserverMode.Both)

        # Before timing starts, elapsed_time should be None
        assert observer.elapsed_time is None

        # Run optimization
        result = gs.optimize(simple_problem, test_params, observer=observer)

        # After optimization, elapsed_time should be available
        elapsed = observer.elapsed_time
        assert elapsed is not None
        assert elapsed > 0


class TestObserverStateObjects:
    """Test the Stage1State and Stage2State objects."""

    def test_stage1_state_attributes(self):
        """Test Stage1State object attributes."""
        observer = gs.observers.Observer()
        observer.with_stage1_tracking()

        result = gs.optimize(simple_problem, test_params, observer=observer)

        stage1 = observer.stage1_final()
        assert stage1 is not None

        # Test all attributes are accessible and have reasonable values
        assert isinstance(stage1.reference_set_size, int)
        assert isinstance(stage1.best_objective, float)
        assert isinstance(stage1.current_substage, str)
        assert isinstance(stage1.function_evaluations, int)
        assert isinstance(stage1.trial_points_generated, int)

        # Test string representations
        str_repr = str(stage1)
        repr_repr = repr(stage1)
        assert len(str_repr) > 0
        assert len(repr_repr) > 0

    def test_stage2_state_attributes(self):
        """Test Stage2State object attributes."""
        observer = gs.observers.Observer()
        observer.with_stage2_tracking()

        result = gs.optimize(simple_problem, test_params, observer=observer)

        stage2 = observer.stage2()
        if stage2 is not None:  # Stage 2 might not run
            # Test all attributes are accessible and have reasonable values
            assert isinstance(stage2.best_objective, float)
            assert isinstance(stage2.solution_set_size, int)
            assert isinstance(stage2.current_iteration, int)
            assert isinstance(stage2.threshold_value, float)
            assert isinstance(stage2.local_solver_calls, int)
            assert isinstance(stage2.improved_local_calls, int)
            assert isinstance(stage2.function_evaluations, int)
            assert isinstance(stage2.unchanged_cycles, int)

            # Test string representations
            str_repr = str(stage2)
            repr_repr = repr(stage2)
            assert len(str_repr) > 0
            assert len(repr_repr) > 0


class TestObserverModes:
    """Test different observer modes."""

    def test_stage1_only_mode(self):
        """Test Stage1Only observer mode."""
        observer = gs.observers.Observer()
        observer.with_mode(gs.observers.ObserverMode.Stage1Only)
        observer.with_stage1_tracking()
        observer.with_stage2_tracking()  # This should be ignored due to mode

        assert observer.should_observe_stage1
        assert not observer.should_observe_stage2

    def test_stage2_only_mode(self):
        """Test Stage2Only observer mode."""
        observer = gs.observers.Observer()
        observer.with_mode(gs.observers.ObserverMode.Stage2Only)
        observer.with_stage1_tracking()  # This should be ignored due to mode
        observer.with_stage2_tracking()

        assert not observer.should_observe_stage1
        assert observer.should_observe_stage2

    def test_both_mode(self):
        """Test Both observer mode."""
        observer = gs.observers.Observer()
        observer.with_mode(gs.observers.ObserverMode.Both)
        observer.with_stage1_tracking()
        observer.with_stage2_tracking()

        assert observer.should_observe_stage1
        assert observer.should_observe_stage2


class TestObserverStringRepresentation:
    """Test observer string representation."""

    def test_observer_str_representation(self):
        """Test observer __str__ method."""
        observer = gs.observers.Observer()
        observer.with_stage1_tracking()
        observer.with_stage2_tracking()
        observer.with_timing()

        str_repr = str(observer)
        assert "Observer Configuration" in str_repr
        assert "Stage 1 Tracking: true" in str_repr
        assert "Stage 2 Tracking: true" in str_repr
        assert "Timing Enabled: true" in str_repr
