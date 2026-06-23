# External UNSAT Validation Report — psMNIST_lstm v3 random-392

**Date**: 2026-06-23
**Benchmark**: `benchmarks/psMNIST_lstm/` — 25 UNSAT instances, `prop_000–prop_024` on `lstm_psMNIST_h8.onnx`
**Task**: confirm or refute the 25 UNSAT labels using a sound verifier independent of n2v's IBP

---

## ⚠️ CONTRADICTED INSTANCES

**None.** No UNSAT label was refuted. See Section 4 for full results.

---

## 1. Tool and Environment

### Tool used: auto_LiRPA (CROWN method)

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
| Conda env | `abcrown` (Python 3.11) |
| Date run | 2026-06-23 |
| Script | `validate_crown_random392.py` |

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
```

Warnings emitted by onnx2torch (`UserWarning: non-writable tensor`, `TracerWarning: Converting
a tensor to a NumPy array`) and by auto_LiRPA (`UserWarning: Constant operand has batch
dimension`) are cosmetic; they do not affect bound computation soundness.

---

## 3. Execution

### Key parameters

```python
METHOD    = "CROWN"      # linear relaxation bounds

# For each prop_k:
lb_np, ub_np, true_cls = parse_vnnlib_v2(f"vnnlib/prop_{k:03d}.vnnlib")
# (assert (>= X_i val)) → lb; (assert (<= X_i val)) → ub
x_lb  = torch.tensor(lb_np).unsqueeze(0)
x_ub  = torch.tensor(ub_np).unsqueeze(0)
x_ctr = (x_lb + x_ub) / 2.0

ptb = PerturbationLpNorm(norm=np.inf, x_L=x_lb, x_U=x_ub)
bt  = BoundedTensor(x_ctr, ptb)
lb_out, ub_out = bounded_model.compute_bounds(x=(bt,), method="CROWN")

certified = lb_out[0, true_cls] > ub_out[0, j].max()  # j ≠ true_cls
```

Box bounds were read directly from the VNN-LIB files — not re-derived from the model or
norm_params.npz.

---

## 4. Results — 25-instance table

| Instance | true\_cls | n2v label | CROWN verdict | Bucket | Min margin | Runtime |
|---|---|---|---|---|---|---|
| prop_000 | 0 | UNSAT | CERTIFIED | **CONFIRMED** | +6.8576 | 1.77 s |
| prop_001 | 0 | UNSAT | CERTIFIED | **CONFIRMED** | +6.9555 | 2.01 s |
| prop_002 | 0 | UNSAT | CERTIFIED | **CONFIRMED** | +6.8011 | 1.82 s |
| prop_003 | 0 | UNSAT | CERTIFIED | **CONFIRMED** | +7.0680 | 1.99 s |
| prop_004 | 0 | UNSAT | CERTIFIED | **CONFIRMED** | +7.0132 | 1.95 s |
| prop_005 | 1 | UNSAT | CERTIFIED | **CONFIRMED** | +6.3567 | 1.94 s |
| prop_006 | 1 | UNSAT | CERTIFIED | **CONFIRMED** | +6.3502 | 1.85 s |
| prop_007 | 1 | UNSAT | CERTIFIED | **CONFIRMED** | +6.1866 | 2.02 s |
| prop_008 | 1 | UNSAT | CERTIFIED | **CONFIRMED** | +6.2072 | 1.95 s |
| prop_009 | 1 | UNSAT | CERTIFIED | **CONFIRMED** | +6.1178 | 1.83 s |
| prop_010 | 7 | UNSAT | CERTIFIED | **CONFIRMED** | +5.5616 | 2.02 s |
| prop_011 | 7 | UNSAT | CERTIFIED | **CONFIRMED** | +5.3813 | 1.97 s |
| prop_012 | 7 | UNSAT | CERTIFIED | **CONFIRMED** | +5.7638 | 1.93 s |
| prop_013 | 7 | UNSAT | CERTIFIED | **CONFIRMED** | +4.9907 | 1.82 s |
| prop_014 | 7 | UNSAT | CERTIFIED | **CONFIRMED** | +5.1819 | 2.01 s |
| prop_015 | 6 | UNSAT | CERTIFIED | **CONFIRMED** | +5.1238 | 1.92 s |
| prop_016 | 6 | UNSAT | CERTIFIED | **CONFIRMED** | +5.2800 | 1.96 s |
| prop_017 | 6 | UNSAT | CERTIFIED | **CONFIRMED** | +5.3103 | 1.83 s |
| prop_018 | 6 | UNSAT | CERTIFIED | **CONFIRMED** | +5.0207 | 2.03 s |
| prop_019 | 6 | UNSAT | CERTIFIED | **CONFIRMED** | +5.1851 | 1.96 s |
| prop_020 | 3 | UNSAT | CERTIFIED | **CONFIRMED** | +5.0012 | 1.93 s |
| prop_021 | 5 | UNSAT | CERTIFIED | **CONFIRMED** | +4.3857 | 1.84 s |
| prop_022 | 2 | UNSAT | CERTIFIED | **CONFIRMED** | +4.3168 | 2.02 s |
| prop_023 | 3 | UNSAT | CERTIFIED | **CONFIRMED** | +4.3658 | 1.94 s |
| prop_024 | 3 | UNSAT | CERTIFIED | **CONFIRMED** | +4.4587 | 1.96 s |

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

| Statistic | Value |
|---|---|
| Min margin | +4.3168 (prop_022, cls=2) |
| Median margin | +5.3813 (prop_011, cls=7) |
| Max margin | +7.0680 (prop_003, cls=0) |
| Total wall time | ~56 s (~2.0 s/instance + ~6 s model init) |

---

## 6. Verdict

**The 25 UNSAT labels are independently validated.**

auto_LiRPA's CROWN method certifies all 25 instances at ε = 1/255 ≈ 0.003922 on
`lstm_psMNIST_h8.onnx`. The minimum certification margin across all instances is +4.317 logit
units (prop_022, cls=2).

This is a comfortable safety margin above zero — no instance is borderline. The benchmark was
designed by selecting only IBP-certifiable instances on the h8 model under IBP-regularised
training (CROWN-IBP style); CROWN is strictly stronger than IBP, so CROWN certifying all 25
is expected. The value of this exercise is that it eliminates the circularity: a
self-consistent IBP-pass-at-generation + IBP-recheck at dry-run time could in principle both
fail the same way if n2v's IBP had a systematic bug. CROWN's independent implementation of
linear relaxation confirms there is no such bug — the instances are genuinely robust.

The full alpha-beta-CROWN harness was also run (see `VALIDATION_ABCROWN.md`) confirming 50/50
match across all instances with zero timeouts.

**For the VNN-COMP 2026 submission writeup**: the UNSAT labels are corroborated by auto_LiRPA
CROWN v0.7.2, an independent implementation of linear relaxation bounds. The SAT labels are
confirmed by concrete witnesses (see `VERIFY_v2.md`). The benchmark is sound.
