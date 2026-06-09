"""
Soundness tests for the Minkowski-sum fix in _add_sets (Star + Star).

Two scenarios are validated:

1. Shared predicates (same nVar) — two Stars derived from the same input set via
   independent affine maps. The element-wise path is used and must be sound.

2. Independent predicates (different nVar) — two Stars with unrelated predicate
   spaces, as in LSTM gate sums W_ih(x_t) + W_hh(h_{t-1}).  The Minkowski-sum
   path introduced in commit 55204f9 (Tran et al. HSCC 2023, Prop. 2.8) is used.

Soundness criterion: for every point (a, b) with a ∈ set_a, b ∈ set_b the sum
a + b must lie inside the bounds of _add_sets([set_a], [set_b], 'add')[0].
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.nn.reach import _add_sets
from n2v.nn.layer_ops.dispatcher import reach_layer
from n2v.sets import Star


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _make_star(lb, ub):
    return Star.from_bounds(np.asarray(lb, float), np.asarray(ub, float))


def _check_soundness(result_star, samples, tol=1e-5):
    lb, ub = result_star.estimate_ranges()
    lb = lb.flatten()
    ub = ub.flatten()
    for pt in samples:
        pt = np.asarray(pt).flatten()
        assert np.all(pt >= lb - tol), f"Point {pt} below LB {lb}"
        assert np.all(pt <= ub + tol), f"Point {pt} above UB {ub}"


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Shared-predicate path (same nVar) — regression
# ─────────────────────────────────────────────────────────────────────────────

class TestSharedPredicatesRegression:
    """
    Element-wise Star addition must still be sound after the nVar-dispatch fix.
    Both stars come from the same input, so they share predicate variables.
    """

    def test_two_linear_layers_same_input(self):
        torch.manual_seed(0)
        W1 = nn.Linear(4, 4, bias=True)
        W2 = nn.Linear(4, 4, bias=True)

        input_star = _make_star([0.0] * 4, [1.0] * 4)

        sa = reach_layer(W1, [input_star], 'approx')[0]
        sb = reach_layer(W2, [input_star], 'approx')[0]

        # Both stars were derived from the same input_star → same nVar
        assert sa.nVar == sb.nVar, "Expected shared-predicate path"

        result = _add_sets([sa], [sb], 'add')[0]

        np.random.seed(0)
        samples = []
        for _ in range(300):
            x = np.random.uniform(0.0, 1.0, 4).astype(np.float32)
            t = torch.tensor(x).unsqueeze(0)
            with torch.no_grad():
                out = (W1(t) + W2(t)).numpy().flatten()
            samples.append(out)

        _check_soundness(result, samples)

    def test_subtraction_same_input(self):
        torch.manual_seed(1)
        W1 = nn.Linear(3, 3, bias=False)
        W2 = nn.Linear(3, 3, bias=False)

        input_star = _make_star([-1.0] * 3, [1.0] * 3)

        sa = reach_layer(W1, [input_star], 'approx')[0]
        sb = reach_layer(W2, [input_star], 'approx')[0]

        assert sa.nVar == sb.nVar

        result = _add_sets([sa], [sb], 'sub')[0]

        np.random.seed(1)
        samples = []
        for _ in range(300):
            x = np.random.uniform(-1.0, 1.0, 3).astype(np.float32)
            t = torch.tensor(x).unsqueeze(0)
            with torch.no_grad():
                out = (W1(t) - W2(t)).numpy().flatten()
            samples.append(out)

        _check_soundness(result, samples)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Independent-predicate path (different nVar) — Minkowski sum
# ─────────────────────────────────────────────────────────────────────────────

class TestIndependentPredicatesMinkowskiSum:
    """
    Stars with unrelated predicate spaces must be combined via Minkowski sum.
    This is the HSCC 2023 fix for LSTM gate addition.
    """

    def _make_independent_stars(self, seed=42):
        """
        Simulate the LSTM situation: one Star has nVar from a fresh input box,
        the other has been through a ReLU split and has extra predicates.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Star A: direct affine map of an input box (nVar = 4)
        lb_a = np.random.uniform(-1, 0, 4)
        ub_a = lb_a + np.random.uniform(0.1, 0.5, 4)
        star_a = _make_star(lb_a, ub_a)
        W_a = nn.Linear(4, 4, bias=True)
        torch.nn.init.normal_(W_a.weight)
        torch.nn.init.normal_(W_a.bias)
        sa = reach_layer(W_a, [star_a], 'approx')[0]

        # Star B: different input box with different dimension → different nVar
        lb_b = np.random.uniform(-1, 0, 6)
        ub_b = lb_b + np.random.uniform(0.1, 0.5, 6)
        star_b_input = _make_star(lb_b, ub_b)
        W_b = nn.Linear(6, 4, bias=True)   # maps 6 → 4 (same output dim as sa)
        torch.nn.init.normal_(W_b.weight)
        torch.nn.init.normal_(W_b.bias)
        sb = reach_layer(W_b, [star_b_input], 'approx')[0]

        assert sa.nVar != sb.nVar, "Need different nVar for independent-predicate test"
        return sa, sb, lb_a, ub_a, lb_b, ub_b, W_a, W_b

    def test_minkowski_sum_no_crash(self):
        sa, sb, *_ = self._make_independent_stars()
        # Must not raise a shape error or assertion error
        result = _add_sets([sa], [sb], 'add')
        assert len(result) == 1
        assert result[0].dim == sa.dim

    def test_minkowski_sum_soundness_add(self):
        sa, sb, lb_a, ub_a, lb_b, ub_b, W_a, W_b = self._make_independent_stars()

        result = _add_sets([sa], [sb], 'add')[0]

        np.random.seed(99)
        samples = []
        for _ in range(400):
            xa = np.random.uniform(lb_a, ub_a).astype(np.float32)
            xb = np.random.uniform(lb_b, ub_b).astype(np.float32)
            ta = torch.tensor(xa).unsqueeze(0)
            tb = torch.tensor(xb).unsqueeze(0)
            with torch.no_grad():
                out = (W_a(ta) + W_b(tb)).numpy().flatten()
            samples.append(out)

        _check_soundness(result, samples)

    def test_minkowski_sum_soundness_sub(self):
        sa, sb, lb_a, ub_a, lb_b, ub_b, W_a, W_b = self._make_independent_stars(seed=7)

        result = _add_sets([sa], [sb], 'sub')[0]

        np.random.seed(77)
        samples = []
        for _ in range(400):
            xa = np.random.uniform(lb_a, ub_a).astype(np.float32)
            xb = np.random.uniform(lb_b, ub_b).astype(np.float32)
            ta = torch.tensor(xa).unsqueeze(0)
            tb = torch.tensor(xb).unsqueeze(0)
            with torch.no_grad():
                out = (W_a(ta) - W_b(tb)).numpy().flatten()
            samples.append(out)

        _check_soundness(result, samples)

    def test_result_nvar_grows(self):
        """Minkowski sum must introduce independent predicate variables (nVar = nVar_a + nVar_b)."""
        sa, sb, *_ = self._make_independent_stars()
        result = _add_sets([sa], [sb], 'add')[0]
        assert result.nVar == sa.nVar + sb.nVar

    def test_small_explicit_stars(self):
        """
        Manual construction: two 2-D stars with known values.
        sa = {x | x = [1,1] + [[1,0],[0,1]] @ a,  a in [-0.1, 0.1]^2}
        sb = {y | y = [2,2] + [[1,0],[0,1]] @ b,  b in [-0.2, 0.2]^2 + relu split adds 1 pred}
        Manually give sb 3 predicates so nVar differs.
        """
        # sa: center [1,1], 2 generators, box constraints
        V_a = np.array([[1.0, 1.0, 0.0],
                        [1.0, 0.0, 1.0]])
        C_a = np.zeros((0, 2))
        d_a = np.zeros((0, 1))
        lb_a = np.array([[-0.1], [-0.1]])
        ub_a = np.array([[0.1],  [0.1]])
        sa = Star(V_a, C_a, d_a, lb_a, ub_a)

        # sb: center [2,2], 3 generators (different nVar)
        V_b = np.array([[2.0, 1.0, 0.0, 0.5],
                        [2.0, 0.0, 1.0, 0.5]])
        C_b = np.zeros((0, 3))
        d_b = np.zeros((0, 1))
        lb_b = np.array([[-0.2], [-0.2], [-0.1]])
        ub_b = np.array([[0.2],  [0.2],  [0.1]])
        sb = Star(V_b, C_b, d_b, lb_b, ub_b)

        assert sa.nVar != sb.nVar

        result = _add_sets([sa], [sb], 'add')[0]

        # Center must be sum of centers
        expected_center = np.array([3.0, 3.0])
        np.testing.assert_allclose(result.V[:, 0], expected_center, atol=1e-9)

        # nVar of result = 2 + 3 = 5
        assert result.nVar == 5

        # Soundness: sum of extreme points must lie in result
        extremes_a = [
            np.array([1.0 + 0.1, 1.0 + 0.1]),
            np.array([1.0 - 0.1, 1.0 - 0.1]),
            np.array([1.0 + 0.1, 1.0 - 0.1]),
        ]
        extremes_b = [
            np.array([2.0 + 0.2 + 0.05, 2.0 + 0.2 + 0.05]),
            np.array([2.0 - 0.2 - 0.05, 2.0 - 0.2 - 0.05]),
        ]
        samples = [a + b for a in extremes_a for b in extremes_b]
        _check_soundness(result, samples)
