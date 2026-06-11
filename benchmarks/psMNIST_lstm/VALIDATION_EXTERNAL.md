# External UNSAT Validation Report — psMNIST_lstm v2

**Date**: 2026-06-11  
**Benchmark**: `benchmarks/psMNIST_lstm/` — 25 UNSAT instances, `prop_000–prop_024` on `lstm_psMNIST_h8.onnx`  
**Task**: confirm or refute the 25 UNSAT labels using a sound verifier independent of n2v's IBP

---

## ⚠️ CONTRADICTED INSTANCES

**None.** No UNSAT label was refuted. See Section 4 for full results.

---

## 1. Tool and Environment

### Primary tool attempted: alpha-beta-CROWN

alpha-beta-CROWN (the VNN-COMP reference verifier) was not installed in the active environment.
The tool requires a dedicated setup from its GitHub repository
(`https://github.com/Verified-Intelligence/alpha-beta-CROWN`) with a custom conda environment
and a complex YAML-based configuration system. This setup was not attempted because
**auto_LiRPA** — the bound-propagation library that alpha-beta-CROWN is built on — is
available as a standalone package and provides CROWN bounds directly. The tool used is
auto_LiRPA 0.7.2, CROWN mode — the sound bound-propagation core of alpha-beta-CROWN. The full
alpha-beta-CROWN wrapper (alpha-optimization + branch-and-bound) was not run, but is unnecessary
here given the minimum margin of +1.455.

### Fallback tool used: auto_LiRPA (CROWN method)

| Item | Value |
|---|---|
| Tool | auto_LiRPA |
| Version | 0.7.2 |
| Source | `pip install "auto-LiRPA @ git+https://github.com/Verified-Intelligence/auto_LiRPA.git"` |
| Method | `CROWN` (linear relaxation — sound, not just IBP) |
| PyTorch | 2.12.0 |
| onnx | 1.21.0 |
| onnx2torch | 1.5.15 |
| Python | 3.11.15 |
| OS / Hardware | macOS 15.2, Apple Silicon arm64 (CPU only) |
| Conda env | `abcrown` (Python 3.11, created for this task) |
| Date run | 2026-06-11 |

### Why CROWN is a sound, independent check

CROWN (Zhang et al., NeurIPS 2018) computes **linear relaxation bounds** on neural network
outputs. For Sigmoid and Tanh activations it uses triangle enclosures; for bilinear
multiplications it uses McCormick envelopes. These are over-approximations, making CROWN
**sound**: if CROWN returns `CERTIFIED`, the property is provably true (there is genuinely no
counterexample in the box). CROWN is strictly stronger than IBP: IBP bounds are always a
subset of CROWN bounds.

n2v's verification uses IBP (Interval Bound Propagation), an interval-arithmetic method that
is a special case of CROWN with no linear correction terms. The two methods share the same
mathematical goal but are separate implementations using separate libraries. A CROWN
certificate from auto_LiRPA is independent corroboration, not a circular check.

### Verification logic

For each instance with `true_cls = c` and output logits `Y_0 … Y_9`:

1. auto_LiRPA's CROWN computes output lower bounds `lb[i]` and upper bounds `ub[i]`
   for all logits simultaneously over the input box `[center − ε, center + ε]`.
2. The property is **certified** iff `lb[c] > ub[j]` for every `j ≠ c`.
3. The minimum margin is `min_j≠c (lb[c] − ub[j])`.
   - Positive margin → CONFIRMED.
   - Zero or negative margin → INCONCLUSIVE (CROWN is incomplete; this is not a contradiction).

---

## 2. Ingestion Check (Step 0)

**Result: loaded cleanly. No caveats.**

`lstm_psMNIST_h8.onnx` was converted to a PyTorch model via `onnx2torch.convert()` and then
wrapped in an `auto_LiRPA.BoundedModule`. Both steps completed without error.

```
onnx2torch.convert("lstm_psMNIST_h8.onnx")
  → output shape torch.Size([1, 10])  ✓

auto_LiRPA.BoundedModule(model, dummy, device='cpu')
  → BoundedModule wrapped successfully  ✓

CROWN smoke-test on dummy input: OK
  → output lb = [3.29, -8.43, 1.71, ...]  (finite, non-trivial)  ✓
```

One warning was emitted by onnx2torch:
```
UserWarning: Using a non-tuple sequence for multidimensional indexing is deprecated...
  (slice.py:63 — Slice operator handling)
```
This is an onnx2torch implementation warning unrelated to correctness. The conversion and
bound computation are not affected.

auto_LiRPA also emitted one batch-dimension warning on an `Add` node. This is a known
cosmetic warning in auto_LiRPA when constants have a batch dimension; it does not affect
soundness.

The BoundedModule initialisation took 8.4 s (model graph construction and operator
registration). Per-instance bound computation was approximately 2.0 s each.

---

## 3. Execution

### Commands

```bash
# Create and configure environment
conda create -n abcrown python=3.11 -y
conda run -n abcrown pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
conda run -n abcrown pip install "auto-LiRPA @ git+https://github.com/Verified-Intelligence/auto_LiRPA.git"
conda run -n abcrown pip install onnx onnx2torch

# Run verification
conda run -n abcrown python3 /tmp/verify_unsat.py
```

### Key parameters

```python
METHOD    = "CROWN"      # linear relaxation bounds
TIMEOUT_S = 120.0        # per-instance timeout (matches instances.csv)

# For each prop_k:
lb_np, ub_np, true_cls = parse_vnnlib(f"vnnlib/prop_{k:03d}.vnnlib")
x_lb  = torch.tensor(lb_np).unsqueeze(0)
x_ub  = torch.tensor(ub_np).unsqueeze(0)
x_ctr = (x_lb + x_ub) / 2.0

ptb = PerturbationLpNorm(norm=np.inf, x_L=x_lb, x_U=x_ub)
bt  = BoundedTensor(x_ctr, ptb)
lb_out, ub_out = bounded_model.compute_bounds(x=(bt,), method="CROWN")

certified = all(lb_out[0, true_cls] > ub_out[0, j]
                for j in range(10) if j != true_cls)
```

Box bounds were read directly from the VNN-LIB files — not re-derived from the model or
norm_params.npz.

---

## 4. Results — 25-instance table

| Instance | true\_cls | n2v label | CROWN verdict | Bucket | Min margin | Runtime |
|---|---|---|---|---|---|---|
| prop_000 | 2 | UNSAT | CERTIFIED | **CONFIRMED** | +5.6158 | 1.81 s |
| prop_001 | 0 | UNSAT | CERTIFIED | **CONFIRMED** | +5.3863 | 1.97 s |
| prop_002 | 6 | UNSAT | CERTIFIED | **CONFIRMED** | +5.2163 | 1.98 s |
| prop_003 | 6 | UNSAT | CERTIFIED | **CONFIRMED** | +5.2071 | 1.97 s |
| prop_004 | 0 | UNSAT | CERTIFIED | **CONFIRMED** | +5.1888 | 1.96 s |
| prop_005 | 0 | UNSAT | CERTIFIED | **CONFIRMED** | +5.1051 | 1.95 s |
| prop_006 | 0 | UNSAT | CERTIFIED | **CONFIRMED** | +4.7564 | 2.00 s |
| prop_007 | 7 | UNSAT | CERTIFIED | **CONFIRMED** | +4.5176 | 1.99 s |
| prop_008 | 5 | UNSAT | CERTIFIED | **CONFIRMED** | +4.4406 | 2.00 s |
| prop_009 | 0 | UNSAT | CERTIFIED | **CONFIRMED** | +4.4089 | 1.94 s |
| prop_010 | 7 | UNSAT | CERTIFIED | **CONFIRMED** | +4.3776 | 2.09 s |
| prop_011 | 6 | UNSAT | CERTIFIED | **CONFIRMED** | +4.3434 | 2.00 s |
| prop_012 | 3 | UNSAT | CERTIFIED | **CONFIRMED** | +4.0217 | 1.98 s |
| prop_013 | 8 | UNSAT | CERTIFIED | **CONFIRMED** | +3.2396 | 1.97 s |
| prop_014 | 7 | UNSAT | CERTIFIED | **CONFIRMED** | +2.7641 | 1.98 s |
| prop_015 | 0 | UNSAT | CERTIFIED | **CONFIRMED** | +3.0556 | 1.87 s |
| prop_016 | 6 | UNSAT | CERTIFIED | **CONFIRMED** | +3.0592 | 1.99 s |
| prop_017 | 5 | UNSAT | CERTIFIED | **CONFIRMED** | +2.9509 | 1.98 s |
| prop_018 | 6 | UNSAT | CERTIFIED | **CONFIRMED** | +2.9301 | 1.96 s |
| prop_019 | 7 | UNSAT | CERTIFIED | **CONFIRMED** | +2.8774 | 2.00 s |
| prop_020 | 2 | UNSAT | CERTIFIED | **CONFIRMED** | +2.7057 | 1.98 s |
| prop_021 | 7 | UNSAT | CERTIFIED | **CONFIRMED** | +2.2967 | 2.03 s |
| prop_022 | 7 | UNSAT | CERTIFIED | **CONFIRMED** | +2.1746 | 2.02 s |
| prop_023 | 2 | UNSAT | CERTIFIED | **CONFIRMED** | +2.0899 | 1.87 s |
| prop_024 | 6 | UNSAT | CERTIFIED | **CONFIRMED** | +1.4550 | 2.00 s |

**Min margin** = min over all j ≠ true_cls of `(lb[true_cls] − ub[j])`. A positive value
means CROWN proves the true class lower bound exceeds every other class upper bound by that
amount. No instance is even close to zero.

---

## 5. Summary counts

| Bucket | Count |
|---|---|
| **CONFIRMED** | **25 / 25** |
| INCONCLUSIVE | 0 / 25 |
| CONTRADICTED | 0 / 25 |

Total wall time: ~50 s (2.0 s/instance + 8.4 s model init).

---

## 6. Verdict

**The 25 UNSAT labels are independently validated.**

auto_LiRPA's CROWN method certifies all 25 instances at ε = 0.005 on `lstm_psMNIST_h8.onnx`.
The minimum certification margin across all instances is +1.455 logit units (prop_024), which
is a comfortable safety margin above zero — no instance is borderline. This result is not
surprising: the benchmark was designed by selecting only IBP-certifiable instances on the
h8 model, and CROWN is strictly stronger than IBP, so CROWN certifying all 25 is expected.
The value of this exercise is that it eliminates the circularity: a self-consistent
IBP-pass-at-generation + IBP-recheck at dry-run time could in principle both fail the same
way if n2v's IBP had a systematic bug. CROWN's independent implementation of linear
relaxation confirms there is no such bug — the instances are genuinely robust.

**For the VNN-COMP 2026 submission writeup**: the UNSAT labels are corroborated by auto_LiRPA
CROWN v0.7.2, an independent implementation of linear relaxation bounds. The SAT labels were
previously confirmed by concrete witnesses (see `VERIFY_v2.md`). The benchmark is sound.

**Remaining gap**: alpha-beta-CROWN (the actual competition verifier) was not run because its
full installation was not set up. Since auto_LiRPA is the computational core of alpha-beta-CROWN,
this gap is minimal — the underlying bound computation is the same. To fully close it, run
the instances through the `abcrown.py` script from
`https://github.com/Verified-Intelligence/alpha-beta-CROWN` with `method: CROWN-Optimized`
before the official submission.
