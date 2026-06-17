# psMNIST LSTM Benchmark Report

**Author**: Ibrahim Boudaoud  
**Date**: 2026-06-10  
**Benchmark**: `benchmarks/psMNIST_lstm/`  
**Target**: VNN-COMP 2026

---

## 1. What is psMNIST?

Permuted Sequential MNIST (psMNIST) is the LSTM analogue of the standard MNIST feedforward benchmark that every VNN-COMP verifier already handles. A fixed random permutation is applied to the flattened 784-pixel MNIST image before feeding it to the LSTM one row at a time. This breaks spatial locality completely — the model cannot rely on adjacent pixels being related and must learn long-range temporal dependencies. It is the standard recurrent network benchmark in the ML literature (Le et al. 2015, Arjovsky et al. 2016).

**Implementation in this repo**:
- 14 timesteps × 28 features = 392-dimensional input (first 14 rows of the permuted image)
- 10-class LSTM classifier
- Robustness property: L∞-ball around a test image, same semantics as ACAS Xu

---

## 2. Epsilon Rationale

### 2.1 ε = 1/255 ≈ 0.003922 — UNSAT instances (h8 model)

`EPS_UNSAT = 1/255` is a **fixed constant** set in `generate.py`. It corresponds to exactly
**1 pixel level** of L∞ perturbation in [0,1] input space (the standard VNN-COMP
image-benchmark convention: pixels range 0–255, so dividing by 255 means ε = k/255 = k pixel
levels).

Inputs are normalised by dividing by 255 only (no Z-score). This makes the epsilon
semantically meaningful: an adversary can perturb each pixel by at most 1 grey-level value.

The h8 model is trained with **IBP regularisation** (CROWN-IBP / DiffAI style): the training
loss combines clean cross-entropy with an IBP worst-case cross-entropy loss (lambda ramped
0→1 over epochs 11–40). This directly penalises non-certifiable behaviour and keeps weight
norms small enough for IBP to certify through 14 timesteps. Without IBP regularisation,
/255-trained models have larger weights (inputs are smaller: mean ≈ 0.13 vs 0 for Z-score)
and IBP interval blowup is 0% certifiable. With IBP regularisation, ~74% of correctly-
classified test images are certifiable at ε=1/255.

Instances are **pre-screened**: only test inputs where `ibp_certify(model_small, x, 1/255, true_cls)`
returns `True` are written to VNN-LIB. The IBP check does a full forward pass through all 14
timesteps with interval arithmetic, confirming the true class lower-bound beats every other
class upper-bound.

Why h8 specifically? The small model (8 hidden units) has narrow weight matrices — the
intervals spread less per timestep, so IBP remains tight enough to certify. The large h64
model has 64-dimensional hidden state; interval blowup from 64 × 14 = 896 sigmoid/tanh
applications makes IBP non-certifying for almost any test input.

### 2.2 ε ≈ 0.022 (effectively fixed) — SAT instances (h64 model)

The SAT epsilon is **effectively fixed** at `eps = 1.1 × delta = 0.022` (verified range 0.021996–0.022001 across all 25 instances; 9 distinct values; spread 4.25 × 10⁻⁵). It is computed per instance by `find_sat_instance()` at `generate.py:373`, but the boundary search converges to the same radius for all instances:

```
eps = 1.1 × L∞(x_test, boundary_point)
```

The geometry:
1. Find the nearest test image `x1` from a different class (in L∞ distance)
2. Binary-search 50 times between `x0` (correct class) and `x1` (wrong class) to locate the decision boundary
3. After convergence: `a` is just inside the correct class, `b` is just inside the wrong class
4. Step `delta=0.02` further into the correct class from `a` to get `x_test`
5. Set `eps = 1.1 × L∞(x_test, b)` — the ball spans the boundary and contains the witness `b`

The 10% margin (×1.1) ensures the wrong-class witness is comfortably inside the ball, not just touching it. After 50 bisections the boundary is located to precision ≈ L∞(x0, x1) / 2⁵⁰ ≈ 0, so `L∞(x_test, b) ≈ delta = 0.02` for all instances, giving `eps = 1.1 × 0.02 = 0.022`. The tight range [0.021996, 0.022001] was confirmed by VERIFY_v2.md Task 4.

All inputs are in [0,1] (/255 normalization). ε = 0.022 corresponds to ≈ 5.6 pixel levels of
L∞ perturbation (5.61 = 0.022 × 255).

**Why h64 for SAT?** The large model has a well-trained decision boundary with sharp transitions. The binary search converges to a geometrically close boundary point, yielding a reasonably small ε (0.02 range). The small h8 model's decision boundary is poorly calibrated (83% accuracy) and much harder to falsify at meaningful ε values.

---

## 3. Model Accuracy

| Model | Hidden size | Test accuracy | Role |
|---|---|---|---|
| `lstm_psMNIST_h8.onnx` | 8 | **79.7%** | UNSAT instances — certified by IBP-regularised model |
| `lstm_psMNIST_h64.onnx` | 64 | **92.5%** | SAT instances — falsification witnesses |

**h8 at 79.7%**: The accuracy is intentionally lower than a standard-trained h8. The IBP
regularisation introduces a conservative bias (penalising non-certifiable behaviour) which
trades some accuracy for certifiability. What matters is IBP certifiability at ε=1/255: with
IBP-regularised training, 74%+ of correctly-classified test images are certifiable vs 0% with
standard training.

**h64 at 92.5%**: 92.5% is a good accuracy for an LSTM on truncated psMNIST (14/28 timesteps).
High accuracy means the model has a well-defined decision boundary that binary search can
locate reliably, enabling tight SAT witnesses.

Both models are trained with `SEED=42` for reproducibility. Training is ~40 epochs on the full 60,000 MNIST training images with batch size 256.

---

## 4. Benchmark Format and Pipeline Status

### 4.1 Pipeline checklist

| Stage | Status | Evidence |
|---|---|---|
| Train h8 + h64 LSTM | **Done** | `models/lstm_psMNIST_h8.pt`, `models/lstm_psMNIST_h64.pt` |
| Export to ONNX (opset 14) | **Done** | `onnx/lstm_psMNIST_h8.onnx`, `onnx/lstm_psMNIST_h64.onnx` |
| Write VNN-LIB 2.0 properties | **Done** | `vnnlib/prop_000.vnnlib` … `prop_049.vnnlib` (50 files) |
| Label ≥50 instances | **Done** | `ground_truth.csv`: 25 UNSAT (h8) + 25 SAT (h64) |
| `instances.csv` (VNN-COMP format) | **Done** | 50 rows, no header, `onnx,vnnlib,timeout=120` |
| Validate labels (self) | **Done** | n2v Box/IBP for UNSAT, n2v PGD falsification for SAT |
| Validate labels (external) | **Done** | α,β-CROWN 0.7.0 full harness: 50/50 MATCH, 0 timeout (VALIDATION_ABCROWN.md) |
| Benchmark description document | **Not done** | VNN-COMP requires a short paper/description |

### 4.2 File structure

```
benchmarks/psMNIST_lstm/
├── generate.py          ← full pipeline script
├── instances.csv        ← 50 rows (no header), VNN-COMP format
├── ground_truth.csv     ← 51 lines (1 header + 50 rows)
├── onnx/
│   ├── lstm_psMNIST_h8.onnx
│   └── lstm_psMNIST_h64.onnx
└── vnnlib/
    ├── prop_000.vnnlib … prop_024.vnnlib  ← h8 UNSAT
    └── prop_025.vnnlib … prop_049.vnnlib  ← h64 SAT
```

### 4.3 VNN-COMP format compliance

- ONNX opset 14 ✓ (generated by `torch.onnx.export(..., opset_version=14)`)
- VNN-LIB 2.0 ✓ (multi-class disjunction semantics matching VNN-COMP convention)
- `instances.csv` has **no header** ✓ (VNN-COMP spec requires no header)
- `ground_truth.csv` has **a header** ✓ (`onnx,vnnlib,result,timeout`)
- 50 instances ✓ (VNN-COMP minimum is typically 50)
- Balanced label split: 25 UNSAT + 25 SAT ✓
- Timeout: 120 seconds per instance ✓

---

## 5. Verification Status

### 5.1 What has been verified

**UNSAT (25 instances, h8)**: Certified by IBP inside `generate.py`. The same IBP routine that certifies during generation is structurally identical to running n2v's Box reachability. The `run_n2v_dryrun()` function at the end of the pipeline confirms all 25 UNSAT labels are reproduced by n2v's Box method. This is **self-consistent verification** — the same tool that labeled the instances re-checks them.

**SAT (25 instances, h64)**: Each SAT label has an explicit witness: the boundary point `b` that the binary search found. This witness is a concrete input in `[x_test - eps, x_test + eps]` that the h64 model classifies into the wrong class. Existence of a concrete counterexample makes SAT sound by construction — no verifier is needed to validate SAT instances. The `run_n2v_dryrun()` function also confirms PGD (gradient-based falsification) recovers these witnesses.

### 5.2 External validation completed

**α,β-CROWN 0.7.0 full harness**: All 50 instances were run through `abcrown.py` (the official
VNN-COMP entry point) via CSV mode. Result: **50/50 MATCH, 0 timeout**. UNSAT instances
returned `safe-incomplete` (alpha-CROWN certified, 9.9–11.9 s each); SAT instances returned
`unsafe-pgd` (PGD counterexample, <0.15 s each). See `VALIDATION_ABCROWN.md`.

**auto_LiRPA 0.7.2 CROWN**: All 25 UNSAT instances independently certified with minimum margin
+3.188 logits (prop_024). See `VALIDATION_EXTERNAL.md`.

### 5.3 Current n2v verification capability on psMNIST

**Box/IBP reachability**: Works. The `run_n2v_dryrun()` call at the end of `generate.py` confirms all 50 labels are reproduced using n2v's Box method. This is fast (< 1s per instance) and self-consistent.

**Star approx reachability**: Does not yet complete in practical time. After the Minkowski sum fix (commit 55204f9) and LP fallback fix (commit 0db2786), the Star approx pipeline no longer crashes through the kinematics_lstm benchmark, but the per-neuron LP cost for sigmoid/tanh relaxation (∼1280 LP calls for 10-step h8) makes full Star verification of psMNIST (14-step, larger hidden) impractical without Step 2 (LP-free sigmoid bounds, see MINKOWSKI_SUM_REPORT.md §5).

---

## 6. Summary: Where Things Stand

### What you can claim on a slide

- "Train → Export ONNX → Write VNN-LIB → Label ≥50 instances": **all done**
- "50 labeled instances with balanced SAT/UNSAT split": **done**
- "Self-validated with n2v Box/IBP + falsification witnesses": **done**
- "Format-compliant (ONNX opset 14, VNN-LIB 2.0, instances.csv)": **done**

### What you cannot yet claim

- "Independently validated by α,β-CROWN/Marabou": **not done**
- "Packaged for official submission": **not done** (no benchmark description doc)
- "Star-level verification": **not done** (LP bottleneck blocks it; fix is Step 2)

### The one remaining technical gap to official VNN-COMP submission

Run α,β-CROWN or Marabou against all 50 instances and confirm all 25 UNSAT labels are reproduced (UNSAT) and all 25 SAT labels return SAT or TIMEOUT. If any UNSAT label is overturned, it means IBP was unsound (unlikely but possible due to floating-point or ONNX export issues) and that instance must be dropped or re-labeled.

---

## 7. Epsilon Values at a Glance

| Model | ε | How chosen | Verification method |
|---|---|---|---|
| h8 (UNSAT) | **0.005** | Fixed constant; tightest ball certifiable by IBP across 14 timesteps | IBP (`ibp_certify()`) → UNSAT |
| h64 (SAT) | **0.022 (effectively fixed)**; range 0.021996–0.022001 | `1.1 × delta` where delta=0.02; boundary search converges to same radius for all 25 instances | Concrete witness → SAT by construction |
