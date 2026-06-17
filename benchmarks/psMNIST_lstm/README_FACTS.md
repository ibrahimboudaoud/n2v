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
| **Normalization** | `/255` only — inputs in [0,1]. ε = k/255 corresponds to exactly k pixel levels of perturbation (standard VNN-COMP image-benchmark convention). |

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
| **Test accuracy** | 79.7% on MNIST test set (14-step psMNIST, 10 classes; lower expected — tiny model, IBP-regularised training) |
| **Training seed** | 42 |
| **Training config** | EPOCHS=40, BATCH_SIZE=256, LR=1e-3, Adam; IBP-regularised (CROWN-IBP style, lambda ramp 0→1 over epochs 11–40) |
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
| **Test accuracy** | 92.5% on MNIST test set (14-step psMNIST, 10 classes) |
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

| Model | ε | Pixel interpretation | Derivation |
|---|---|---|---|
| h8 (UNSAT) | **1/255 ≈ 0.003922** (fixed constant) | Exactly **1 pixel level** of L∞ perturbation | IBP-certifiable at generation time; pre-screened |
| h64 (SAT) | **0.022 (effectively fixed)** | ≈ **5.6 pixel levels** | `1.1 × L∞(x_test, boundary)` from 50-step bisection with delta=0.02 |

Verified by VERIFY_v2.md Task 4: all 25 SAT instances have eps=0.022000 (uniform).
The near-uniformity follows from `eps = 1.1 × delta = 1.1 × 0.02 = 0.022`; after 50 bisections
`L∞(x_test, boundary) ≈ delta` to precision L∞(x0, x1) / 2⁵⁰ ≈ 0.

---

## 6. Source Image Indices (shared pool)

25 MNIST test images form the shared pool. Each source image generates one UNSAT property
(prop_k on h8) and one SAT property (prop_{k+25} on h64). Pool order corresponds to k = 0…24.

```
k  → MNIST test index  cls    k  → MNIST test index  cls
 0 → 1434               1    13 → 5056               5
 1 → 4984               1    14 → 7409               6
 2 → 4386               1    15 → 9052               0
 3 →  675               1    16 → 8312               4
 4 → 1884               1    17 → 8822               4
 5 → 6463               6    18 → 8395               4
 6 → 7752               5    19 → 9134               0
 7 → 8996               6    20 → 9866               4
 8 → 4583               5    21 →   67               4
 9 → 7155               5    22 →  440               0
10 → 7152               6    23 → 8458               0
11 → 9606               5    24 → 6858               7
12 → 7172               6
```

Class distribution: 0 (4), 1 (5), 4 (5), 5 (5), 6 (5), 7 (1). Pool capped at max 5 per class.

All indices are from the MNIST **test** split (indices 0–9999). No training images are used.
No two properties share a source image. Verified by VERIFY_v2.md Task 1: 25/25 UNSAT centers
reconstruct to their source image with L∞ ≤ 3 × 10⁻⁸ (format-rounding residual only).

---

## 7. Validation Status

### UNSAT (25 instances — prop_000–024)

| Verifier | Method | Result | Min margin | Reference |
|---|---|---|---|---|
| n2v IBP (CROWN-IBP regularised) | Interval bound propagation | 25/25 certified | — | generate.py at generation time |
| auto_LiRPA 0.7.2 | CROWN (linear relaxation) | **25/25 CONFIRMED** | +3.188 logits (prop_024) | VALIDATION_EXTERNAL.md |
| alpha-beta-CROWN 0.7.0 | Full harness (abcrown.py, CROWN mode) | **25/25 safe-incomplete** | — | VALIDATION_ABCROWN.md |

CROWN is independent of n2v IBP: it applies linear relaxation (BoundMul uses McCormick
envelopes for bilinear terms), not interval arithmetic. Positive margins confirm no systematic
IBP bug.

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

**IBP-regularised training (h8 only)**: The h8 model is trained with a CROWN-IBP style combined
loss (clean CE + IBP worst-case CE, lambda 0→1 over epochs 11–40). This is required because
plain /255 inputs (mean ≈ 0.13) lead to larger model weights than Z-score inputs, which blows
up IBP interval expansion through 14 LSTM timesteps. The IBP-regularised training directly
penalises non-certifiable behaviour and achieves 74%+ IBP certification rate on correctly-
classified test images at ε=1/255. The h64 model uses standard Adam training (no IBP loss needed
since it is only used for SAT instances via PGD falsification).

**Class coverage**: Classes 2, 3, 8, 9 are absent from the pool. IBP certification at ε=1/255
is harder for those digit geometries under the /255-trained h8 model. The pool cap (max 5 per
class) ensures 5 distinct classes appear; 6 classes are represented in practice (0, 1, 4, 5, 6, 7).
