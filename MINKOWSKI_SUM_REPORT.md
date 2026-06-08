# Minkowski Sum Fix for Star-Set Addition — Implementation Report

**Author**: Ibrahim Boudaoud  
**Date**: 2026-06-08  
**Branch**: `main` (commit `55204f9`)  
**Reference**: Tran et al., "Verification of Recurrent Neural Networks with Star Reachability", HSCC 2023. DOI: 10.1145/3575870.3587128

---

## 1. Background

n2v is a Python neural network verification tool that uses set-based reachability analysis. Given an input set (e.g., an ε-ball around a test input), it propagates the set forward through the network to compute all possible outputs, then checks whether a safety property holds for every point in that output set.

The tool supports several set representations. The most precise is the **Star set**: a tuple `⟨c, V, C, d, l, u⟩` representing all points of the form:

```
x = V · [1, α]ᵀ,   subject to   C·α ≤ d,   l ≤ α ≤ u
```

where `V = [c | v₁ | v₂ | … | vₙ]` is the basis matrix (first column = center), and `α` is the vector of predicate variables.

---

## 2. The Problem This Fixes

### 2.1 LSTM gate addition requires Minkowski sum

An LSTM cell computes gates like:

```
gates = W_ih · x_t  +  W_hh · h_{t-1}  +  bias
```

When using Star sets for reachability, after the first timestep `h_{t-1}` is itself a Star (the reachable set of hidden states). So the addition involves two Star sets:

- **`sa`**: `W_ih · x_t` — predicates come from the current input `x_t ∈ Xₜ`
- **`sb`**: `W_hh · h_{t-1}` — predicates come from the accumulated history of previous inputs and ReLU splits

These two stars have **independent predicate variables** (different `nVar`) that have nothing to do with each other.

### 2.2 What n2v was doing (wrong)

The previous `_add_sets()` for Star + Star always did:

```python
V_out = sa.V + sb.V   # element-wise addition
out = Star(V_out, sa.C, sa.d, ...)  # only keeps sa's constraints
```

This assumes both stars share **the same predicate variables** — valid only for residual connections in feedforward nets, where both branches trace back to the same input. For LSTM gate sums this would either:
- Crash with a shape error when `sa.V` and `sb.V` have different shapes (different `nVar`)
- Silently produce a wrong set if the shapes happened to match (dropping `sb`'s constraints entirely)

### 2.3 What the paper says

Tran et al. HSCC 2023 (Proposition 2.8) defines the Minkowski sum of two stars with independent predicates as:

```
Θ₁ ⊕ Θ₂ = ⟨ c₁+c₂,  [V₁ | V₂],  P₁∧P₂,  [l₁; l₂],  [u₁; u₂] ⟩
```

Centers add, basis matrices **concatenate** (independent predicate spaces side by side), and constraints combine block-diagonally. Their key algorithm (Algorithm 3.1, Remark 1) builds the hidden-state reachable set as:

```
H_t = ReLU( (W_hh · H_{t-1} + b)  ⊕  W_hx · Xₜ )
```

This correctly encodes the independence of the new input and the previous hidden state.

---

## 3. What Was Implemented

**File changed**: `n2v/nn/reach.py`, function `_add_sets()` (lines 833–935)

The fix adds an `nVar` check before choosing the addition strategy:

```python
elif isinstance(sa, Star) and isinstance(sb, Star):
    if sa.nVar != sb.nVar:
        # Independent predicate spaces — use Minkowski sum
        out = sa.minkowski_sum(sb)          # for 'add'
        # or: out = sa.minkowski_sum(sb_neg) for 'sub'
    else:
        # Shared predicates (residual connections) — exact element-wise
        V_out = sa.V + sb.V
        out = Star(V_out, sa.C, sa.d, ...)
```

The same logic was applied to `ImageStar + ImageStar`.

**`star.minkowski_sum()`** (already existed in `n2v/sets/star.py:212`) performs the correct operation:

```python
new_c   = self.V[:, 0:1] + other.V[:, 0:1]          # c₁ + c₂
new_V   = hstack([new_c, self.V[:, 1:], other.V[:, 1:]])  # [c | V₁ | V₂]
new_C   = block_diag(self.C, other.C)                 # independent constraints
new_d   = vstack([self.d, other.d])
new_l/u = vstack([self.l, other.l]) / vstack([...])
```

No changes were made to `star.minkowski_sum()` — it was already correct.

---

## 4. What This Enables vs. What It Doesn't

### Enabled

- Star-set reachability on the **gate addition** step of an LSTM (`W_ih(xₜ) + W_hh(h)`), which is the first bottleneck when using Stars on unrolled LSTM ONNX graphs.
- Correctness (soundness) for residual connections is preserved — the `nVar`-match path is unchanged.

### Not Yet Solved

This fix handles one of several open problems for full LSTM star-set verification:

| Problem | Status |
|---|---|
| Gate addition across independent predicate spaces | **Fixed (this PR)** |
| Sigmoid/Tanh over-approximation for Stars | Partial — `sigmoid_reach.py` and `tanh_reach.py` exist but use interval-based relaxation, not the triangle approximation from the paper |
| Element-wise multiplication `f ⊙ c`, `o ⊙ tanh(c)` | Open — McCormick relaxation exists for Stars but assumes shared predicates; needs extension |
| Predicate explosion over long sequences | Open — each Minkowski sum and ReLU split adds predicates; LP solve cost grows polynomially with `T × hidden_size` |

---

## 5. Next Steps

### Step 1 — Validate the fix on the existing benchmarks (immediate)

Run the kinematics_lstm and psMNIST_lstm integration tests with `method='approx'` using Star sets (previously these ran on Box/IBP only). The Minkowski sum fix should allow the ONNX graph to propagate without crashing at the gate addition nodes.

```bash
python -m pytest tests/ -k "lstm" -v
```

### Step 2 — Triangle over-approximation for Sigmoid/Tanh on Stars

The paper uses a **triangle relaxation** for ReLU crossing neurons. The same idea applies to sigmoid and tanh for LSTM gates. Currently `sigmoid_reach.py` and `tanh_reach.py` compute interval bounds and return a Box-style approximation. They should instead add a new predicate variable `α_new` with three constraints (lower tangent, upper secant, and non-negativity) — the same pattern used for ReLU in `relu_reach.py`. This keeps the result a Star set rather than collapsing to a Box.

### Step 3 — Minkowski-sum-aware element-wise multiplication

The LSTM cell update:

```
c_new = f ⊙ c_old + i ⊙ g
h_new = o ⊙ tanh(c_new)
```

involves element-wise multiplication of Stars. The existing McCormick relaxation in `_mul_stars_mccormick()` was written assuming shared predicates. After the Minkowski sum fix, `f` and `c_old` will have independent predicate spaces. McCormick needs to be updated to construct bilinear envelope constraints across the concatenated predicate space.

### Step 4 — Predicate reduction / abstraction

After T timesteps with hidden size H, each star carries O(T × H) predicate variables (one per gate per timestep). LP solves over this space become expensive. Strategies to explore:
- **Zonotope prefilter** (already in `bounds_precomputation.py`) — use it before each ReLU/sigmoid split to prune always-active/always-inactive neurons
- **Predicate merging** — periodically collapse the star to a smaller over-approximation (e.g., tightest zonotope enclosure) to bound predicate growth
- **CROWN-style bounds** — linear relaxation that avoids per-neuron LP calls entirely

### Step 5 — VNN-COMP benchmark with Star/approx verification

Once Steps 2–3 are working, add a Star-based `approx` mode to the psMNIST_lstm benchmark and compare UNSAT counts versus the current IBP-only approach. Star sets should certify robustness on longer sequences where IBP blows up.

---

## 6. Summary

The paper by Tran et al. (HSCC 2023) establishes the theoretical foundation for using Minkowski sums of star sets to verify RNNs without unrolling. The core operation (Proposition 2.8) was already implemented in `star.minkowski_sum()` but was never being called in `_add_sets()`. This PR wires up the two: when `_add_sets()` encounters two Star sets with different `nVar` counts (independent predicate spaces), it now delegates to `minkowski_sum()` rather than incorrectly adding the basis matrices element-wise.

This is a necessary but not sufficient step toward full LSTM star-set verification. The remaining gaps are sigmoid/tanh triangle relaxation (Step 2), cross-space McCormick for element-wise multiplication (Step 3), and predicate growth management (Step 4).
