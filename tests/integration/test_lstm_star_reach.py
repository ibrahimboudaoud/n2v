"""
Integration tests: LSTM ONNX model reachability with Star sets.

Validates that the Minkowski-sum fix in _add_sets (commit 55204f9) allows
n2v to propagate Star sets through an unrolled LSTM ONNX graph without
crashing, and that the resulting bounds are sound.

Models used:
  - kinematics_lstm h8  (smallest model, 10 timesteps × 4 features = 40 inputs, 2 outputs)

Test plan:
  0. Fast Box/IBP sanity: model loads, ONNX graph runs end-to-end (< 1s).
  1. No-crash test (SLOW): Star approx reach() completes. Takes ~5 min due to
     LP calls in sigmoid per timestep. Mark with pytest -m slow to skip.
  2. Soundness test (SLOW): 200 random inputs all within Star bounds.
  3. nVar-growth test (SLOW): output nVar > input nVar.

Reference: Tran et al. HSCC 2023, doi:10.1145/3575870.3587128
"""

import os
import numpy as np
import pytest
import torch

from n2v.nn import NeuralNetwork
from n2v.sets import Star
from n2v.sets.box import Box
from n2v.utils.load_vnnlib import load_vnnlib
from n2v.utils.model_loader import load_onnx

pytestmark_slow = pytest.mark.slow

BENCH_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'benchmarks', 'kinematics_lstm'
)
ONNX_H8  = os.path.join(BENCH_DIR, 'onnx', 'lstm_kin_h8.onnx')
VNNLIB_DIR = os.path.join(BENCH_DIR, 'vnnlib')


def _load_first_unsat_property():
    """Return (lb, ub) for the first available vnnlib property."""
    prop_path = os.path.join(VNNLIB_DIR, 'prop_000.vnnlib')
    prop = load_vnnlib(prop_path)
    lb = np.asarray(prop['lb'], dtype=np.float64).flatten()
    ub = np.asarray(prop['ub'], dtype=np.float64).flatten()
    return lb, ub


@pytest.fixture(scope='module')
def kin_net_h8():
    model = load_onnx(ONNX_H8)
    model.eval()
    return NeuralNetwork(model)


@pytest.fixture(scope='module')
def kin_input_star():
    lb, ub = _load_first_unsat_property()
    return Star.from_bounds(lb, ub), lb, ub


class TestLSTMBoxReachFast:
    """
    Fast sanity tests using Box (IBP) sets — complete in < 1 second.
    Verifies the ONNX model loads correctly and can propagate sets end-to-end.
    Box sets bypass LP calls so these run instantly.
    """

    def test_box_reach_completes(self, kin_net_h8, kin_input_star):
        input_star, lb, ub = kin_input_star
        box = Box(lb, ub)
        result = kin_net_h8.reach(box, method='approx')
        assert len(result) >= 1
        out = result[0]
        assert hasattr(out, 'lb') and hasattr(out, 'ub')

    def test_box_soundness(self, kin_net_h8, kin_input_star):
        input_star, lb, ub = kin_input_star
        box = Box(lb, ub)
        result = kin_net_h8.reach(box, method='approx')
        out_lb, out_ub = result[0].lb.flatten(), result[0].ub.flatten()

        model = kin_net_h8.model
        np.random.seed(0)
        violations = 0
        for _ in range(200):
            x = np.random.uniform(lb, ub).astype(np.float32)
            t = torch.tensor(x).unsqueeze(0)
            with torch.no_grad():
                y = model(t).numpy().flatten()
            if np.any(y < out_lb - 1e-4) or np.any(y > out_ub + 1e-4):
                violations += 1
        assert violations == 0, f"{violations}/200 samples outside Box bounds"

    def test_box_bounds_finite(self, kin_net_h8, kin_input_star):
        input_star, lb, ub = kin_input_star
        box = Box(lb, ub)
        result = kin_net_h8.reach(box, method='approx')
        out_lb, out_ub = result[0].lb, result[0].ub
        assert np.all(np.isfinite(out_lb)), "Box LB contains inf/nan"
        assert np.all(np.isfinite(out_ub)), "Box UB contains inf/nan"


@pytest.mark.slow
class TestLSTMStarReachNoCrash:
    """The gate-addition Minkowski sum must not raise an error."""

    def test_approx_reach_completes(self, kin_net_h8, kin_input_star):
        input_star, lb, ub = kin_input_star
        result = kin_net_h8.reach(input_star, method='approx')
        assert len(result) >= 1, "reach() returned empty list"
        out = result[0]
        assert isinstance(out, Star), f"Expected Star output, got {type(out)}"
        assert out.dim == 2, f"Expected 2-class output, got dim={out.dim}"


@pytest.mark.slow
class TestLSTMStarReachSoundness:
    """
    Every point x in the input box must map to an output inside the Star bounds.
    This is the core soundness requirement for verification.
    """

    def test_output_bounds_contain_all_samples(self, kin_net_h8, kin_input_star):
        input_star, lb, ub = kin_input_star

        # Run approx reachability
        result = kin_net_h8.reach(input_star, method='approx')
        out_star = result[0]
        out_lb, out_ub = out_star.estimate_ranges()
        out_lb = out_lb.flatten()
        out_ub = out_ub.flatten()

        # Sample 200 random inputs from the property's input box
        np.random.seed(42)
        model = kin_net_h8.model
        violations = 0
        for _ in range(200):
            x = np.random.uniform(lb, ub).astype(np.float32)
            t = torch.tensor(x).unsqueeze(0)
            with torch.no_grad():
                y = model(t).numpy().flatten()

            if np.any(y < out_lb - 1e-4) or np.any(y > out_ub + 1e-4):
                violations += 1

        assert violations == 0, (
            f"{violations}/200 sampled outputs fell outside reachable-set bounds"
        )

    def test_bounds_are_finite(self, kin_net_h8, kin_input_star):
        input_star, lb, ub = kin_input_star
        result = kin_net_h8.reach(input_star, method='approx')
        out_lb, out_ub = result[0].estimate_ranges()
        assert np.all(np.isfinite(out_lb)), "Lower bounds contain inf/nan"
        assert np.all(np.isfinite(out_ub)), "Upper bounds contain inf/nan"
        assert np.all(out_lb <= out_ub + 1e-9), "Lower bound exceeds upper bound"


@pytest.mark.slow
class TestLSTMStarPredicateGrowth:
    """
    After propagating through T timesteps the output Star must carry more
    predicate variables than the input — confirming Minkowski sums fired.
    """

    def test_nvar_grows_across_timesteps(self, kin_net_h8, kin_input_star):
        input_star, lb, ub = kin_input_star
        result = kin_net_h8.reach(input_star, method='approx')
        out_star = result[0]
        assert out_star.nVar > input_star.nVar, (
            f"Output nVar ({out_star.nVar}) should exceed input nVar "
            f"({input_star.nVar}) after Minkowski sums across timesteps"
        )
