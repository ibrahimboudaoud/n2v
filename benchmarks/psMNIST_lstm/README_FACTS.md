# psMNIST LSTM Benchmark — Facts and Provenance

Single-source record of all factual details for `benchmarks/psMNIST_lstm/`.
Intended to be stable across re-runs and to serve as the canonical reference for the
submission writeup.

---

## 1. Task and Track

| Item | Value |
|---|---|
| **Benchmark name** | `psMNIST_lstm` |
| **Task** | Local robustness: ∀x' ∈ L∞-ball(center, ε): argmax f(x') = true_class (UNSAT half); ∃x' ∈ L∞-ball(center, ε): argmax f(x') ≠ true_class (SAT half) |
| **VNN-COMP track** | Extended track — recurrent / LSTM category |
| **Format** | ONNX opset 14, VNN-LIB 2.0, `instances.csv` (no header), `ground_truth.csv` (with header) |

---

## 2. Sequence Structure

| Item | Value |
|---|---|
| **Dataset** | MNIST (test split — 10,000 images, indices 0–9999) |
| **Permutation** | Fixed pixel permutation applied to each image before slicing |
| **Permutation seed** | `np.random.default_rng(42).permutation(392)` (SEED=42) |
| **Sequence length** | 14 timesteps |
| **Features per timestep** | 28 |
| **Total input dimension** | 392 (= 14 × 28) |
| **Classes** | 10 (digits 0–9) |
| **Normalization** | Z-score: mean=0.1307, std=0.3081 (standard MNIST normalization) |

Note: the canonical psMNIST task uses 28 timesteps × 28 features = 784-dim. This benchmark
uses a truncated 14-timestep sequence for faster LSTM verification benchmarking.

---

## 3. ONNX Models

### 3.1 `onnx/lstm_psMNIST_h8.onnx` — UNSAT half

| Item | Value |
|---|---|
| **Hidden size** | 8 |
| **Architecture** | Single-layer LSTM (14 timesteps, input_size=28 per step), nn.Linear(8, 10) output |
| **Trainable parameters** | ~1,306 (LSTM: W_ih 896 + W_hh 256 + b_ih 32 + b_hh 32 = 1,216; linear: 80 + 10 = 90) |
| **ONNX opset** | 14 |
| **Input shape** | (1, 392) — flattened; model slices internally into 14 × 28 |
| **Output shape** | (1, 10) — raw logits |
| **Test accuracy** | 83% on MNIST test set (14-step psMNIST, 10 classes) |
| **Training seed** | 42 |
| **Training config** | EPOCHS=40, BATCH_SIZE=256, LR=1e-3, Adam optimizer |
| **Key ONNX ops** | Gemm, Add, Sigmoid, Tanh, Mul, Slice, MatMul, Constant |
| **SHA-256** | `531d21dc9e93a27812b956e49ae9dbf713c220be3c832205b639db0af403fcb4` |

### 3.2 `onnx/lstm_psMNIST_h64.onnx` — SAT half

| Item | Value |
|---|---|
| **Hidden size** | 64 |
| **Architecture** | Single-layer LSTM (14 timesteps, input_size=28 per step), nn.Linear(64, 10) output |
| **Trainable parameters** | ~24,714 (LSTM: W_ih 7,168 + W_hh 16,384 + b_ih 256 + b_hh 256 = 24,064; linear: 640 + 10 = 650) |
| **ONNX opset** | 14 |
| **Input shape** | (1, 392) — flattened; model slices internally into 14 × 28 |
| **Output shape** | (1, 10) — raw logits |
| **Test accuracy** | 91% on MNIST test set (14-step psMNIST, 10 classes) |
| **Training seed** | 42 |
| **Training config** | EPOCHS=40, BATCH_SIZE=256, LR=1e-3, Adam optimizer |
| **Key ONNX ops** | Gemm, Add, Sigmoid, Tanh, Mul, Slice, MatMul, Constant |

---

## 4. Instance Counts and Split

| Split | Properties | ONNX model | Ground truth label | Timeout |
|---|---|---|---|---|
| UNSAT | prop_000–prop_024 (25) | `onnx/lstm_psMNIST_h8.onnx` | `unsat` | 120 s |
| SAT | prop_025–prop_049 (25) | `onnx/lstm_psMNIST_h64.onnx` | `sat` | 120 s |
| **Total** | **50** | — | — | — |

CSV format:
- `instances.csv`: no header, 3 columns (onnx_path, vnnlib_path, timeout), LF line endings
- `ground_truth.csv`: header row (`onnx,vnnlib,result,timeout`), 50 data rows, LF line endings

---

## 5. Epsilon Values

| Model | ε | Derivation |
|---|---|---|
| h8 (UNSAT) | **0.005** (fixed constant) | Tightest L∞ ball certifiable by IBP through 14 timesteps on h8; pre-screened at generation time |
| h64 (SAT) | **0.022 (effectively fixed)**; range 0.021996–0.022001 | `1.1 × L∞(x_test, boundary)` from 50-step bisection with delta=0.02; boundary search converges to same radius for all 25 instances |

Verified by VERIFY_v2.md Task 4: 9 distinct epsilon values in [0.021996, 0.022001]; spread 4.25 × 10⁻⁵.
The near-uniformity follows from `eps = 1.1 × delta = 1.1 × 0.02 = 0.022`; after 50 bisections
`L∞(x_test, boundary) ≈ delta` to precision L∞(x0, x1) / 2⁵⁰ ≈ 0.

---

## 6. Source Image Indices (shared pool)

25 MNIST test images form the shared pool. Each source image generates one UNSAT property
(prop_k on h8) and one SAT property (prop_{k+25} on h64). Pool order corresponds to k = 0…24.

```
k  → MNIST test index   k  → MNIST test index
 0 → 5761               13 → 7845
 1 → 8098               14 → 4474
 2 → 3182               15 → 4392
 3 →  665               16 → 5768
 4 → 6129               17 → 1131
 5 → 5669               18 → 2170
 6 → 8535               19 → 2833
 7 → 6121               20 → 9135
 8 → 8853               21 → 4593
 9 → 9080               22 → 4200
10 → 9087               23 → 5385
11 → 9040               24 → 5153
12 → 3221
```

All indices are from the MNIST **test** split (indices 0–9999). No training images are used.
No two properties share a source image. Verified by VERIFY_v2.md Task 1: 25/25 UNSAT centers
reconstruct to their source image with L∞ ≤ 8.5 × 10⁻⁸ (format-rounding residual only).

---

## 7. Validation Status

### UNSAT (25 instances — prop_000–024)

| Verifier | Method | Result | Min margin | Reference |
|---|---|---|---|---|
| n2v IBP | Interval bound propagation | 25/25 certified | +0.166 logits (prop_024) | generate.py at generation time |
| auto_LiRPA 0.7.2 | CROWN (linear relaxation) | **25/25 CONFIRMED** | +1.455 logits (prop_024) | VALIDATION_EXTERNAL.md |

CROWN is independent of n2v IBP: it applies linear relaxation (BoundMul uses McCormick
envelopes for bilinear terms), not interval arithmetic. The 2–4× tighter CROWN margins confirm
no systematic IBP bug. ONNX SHA-256 verified unchanged before and after all verification runs
(VALIDATION_AUDIT.md Check 6).

### SAT (25 instances — prop_025–049)

| Verifier | Method | Result | Reference |
|---|---|---|---|
| generate.py | Concrete witness inside L∞ ball | 25/25 witnesses confirmed | VERIFY_v2.md Task 2 |
| n2v dry-run | PGD falsification | 25/25 pass | generate.py `run_n2v_dryrun()` |

SAT is sound by construction: the boundary search produces a concrete input `b` inside the
L∞ ball that the h64 model classifies as the wrong class.

---

## 8. Regenerate Command

```bash
cd benchmarks/psMNIST_lstm
python generate.py
```

Requires the base conda environment with PyTorch, ONNX, and n2v installed. SEED=42 is
hardcoded; the run is deterministic given the same MNIST test data and PyTorch version.
All 50 instances (UNSAT + SAT) and both ONNX models are regenerated. Training variance is
suppressed by `torch.manual_seed(SEED)` at the start of `main()`.

---

## 9. Pinned Versions (abcrown env — used for CROWN external validation)

| Package | Version | Notes |
|---|---|---|
| Python | 3.11.15 | conda env `abcrown` |
| PyTorch | 2.12.0 | CPU build |
| onnx | 1.21.0 | |
| onnx2torch | 1.5.15 | |
| auto_LiRPA | 0.7.2 | installed from GitHub HEAD (PyPI 0.3 incompatible with NumPy 2.x) |
| macOS / hardware | 15.2, Apple Silicon arm64 | CPU only |

Versions confirmed via `conda run -n abcrown pip show <pkg>` on 2026-06-11.
No requirements file for the abcrown env is pinned in this repo.

---

## 10. Known Notes

**3 class-6 pairing exceptions (props 027, 041, 043)**: SAT properties for pool indices k=2,
16, 18 (MNIST test[3182], test[5768], test[2170] — all class 6) have source images whose h64
decision boundaries are geometrically distant (>17 L∞ units in normalized 392-d space). As a
result, `x_test` ends up far from the source image and closer in L∞ to a different class-6
test image. This is **correct-by-construction**: `true_cls=6` is preserved for all three, and
the shared-pool invariant (`unsat_indices == sat_indices`) is structurally enforced by
`generate.py`. See VERIFY_v2.md Task 3 for full analysis.

**No root-level LICENSE**: As of 2026-06-11 the n2v repository has no root-level LICENSE file.
This `benchmarks/psMNIST_lstm/LICENSE` (MIT, 2026 Ibrahim Boudaoud) covers the benchmark
directory for VNN-COMP packaging purposes. A root LICENSE should be added to the fork before
any upstream PR.
