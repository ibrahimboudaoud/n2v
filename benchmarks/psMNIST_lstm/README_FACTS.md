# psMNIST LSTM Benchmark — Facts and Provenance

Single-source record of all factual details for `benchmarks/psMNIST_lstm/`.
Intended to be stable across re-runs and to serve as the canonical reference for the
submission writeup.

**Version**: v3 random-392 (branch `experiment-random392`)

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
| **Permutation design** | random-392: 392 unique pixel indices drawn from ALL 784 positions |
| **Permutation seed** | `np.random.default_rng(42).choice(784, size=392, replace=False)` (SEED=42) |
| **Permutation range** | min index = 3, max index = 783 (confirmed from `norm_params.npz`) |
| **Sequence length** | 14 timesteps |
| **Features per timestep** | 28 |
| **Total input dimension** | 392 (= 14 × 28) |
| **Classes** | 10 (digits 0–9) |
| **Normalization** | `/255` only — inputs in [0,1]. ε = k/255 corresponds to exactly k pixel levels of perturbation (standard VNN-COMP image-benchmark convention). |

Note: the canonical psMNIST task uses 28 timesteps × 28 features = 784-dim. This benchmark
uses a truncated 14-timestep sequence for faster LSTM verification benchmarking.

**random-392 vs top-half (v2)**: The previous v2 design used `_PERM_RNG.permutation(392)`,
which only ever accessed pixel indices 0–391 (upper half of the 28×28 image). The v3 random-392
design draws from all 784 positions, giving better digit-discriminating features, higher clean
accuracy (+4% for h8, +3.6% for h64), stronger IBP certifiability, and all 10 digit classes
viable in the pool (vs 6 in top-half).

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
| **Test accuracy** | **83.7%** on MNIST test set (14-step random-392 psMNIST, 10 classes) |
| **Training seed** | 42 |
| **Training config** | EPOCHS=40, BATCH_SIZE=256, LR=1e-3, Adam; IBP-regularised (CROWN-IBP style, lambda ramp 0→1 over epochs 11–40) |
| **IBP certifiability** | ~87.2% of correctly-classified test images certifiable at ε=1/255 |
| **Key ONNX ops** | Add, Constant, Gemm, MatMul, Mul, Sigmoid, Slice, Tanh |
| **SHA-256** | `c98df1dd8dcea2a2be0d4b8412e764b127deda14732e2bef03546ba78714d4fd` |

### 3.2 `onnx/lstm_psMNIST_h64.onnx` — SAT half

| Item | Value |
|---|---|
| **Hidden size** | 64 |
| **Architecture** | Single-layer LSTM (14 timesteps, input_size=28 per step), nn.Linear(64, 10) output |
| **Trainable parameters** | ~24,714 (LSTM: W_ih 7,168 + W_hh 16,384 + b_ih 256 + b_hh 256 = 24,064; linear: 640 + 10 = 650) |
| **ONNX opset** | 14 |
| **Input shape** | (1, 392) — flattened; model slices internally into 14 × 28 |
| **Output shape** | (1, 10) — raw logits |
| **Test accuracy** | **96.1%** on MNIST test set (14-step random-392 psMNIST, 10 classes) |
| **Training seed** | 42 |
| **Training config** | EPOCHS=40, BATCH_SIZE=256, LR=1e-3, Adam optimizer (no IBP loss) |
| **Key ONNX ops** | Add, Constant, Gemm, MatMul, Mul, Sigmoid, Slice, Tanh |
| **SHA-256** | `1a8c7fb828ee2254552cf532cd6029b84b530bf65eab5e7161053e2afbb1d308` |

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

| Model | ε | k/255 form | Pixel interpretation | Derivation |
|---|---|---|---|---|
| h8 (UNSAT) | **1/255 ≈ 0.003922** | k=1 | Exactly **1 pixel level** of L∞ perturbation | IBP-certifiable; fixed constant `EPS_UNSAT` |
| h64 (SAT) | **6/255 ≈ 0.023529** | k=6 | Exactly **6 pixel levels** of L∞ perturbation | Snapped: boundary `d≈0.02 < 6/255`; fixed constant `EPS_SAT` |

Both epsilons are exact k/255 values stored as constants in `generate.py` (`EPS_UNSAT = 1/255`,
`EPS_SAT = 6/255`). The SAT eps is derived by: boundary search finds d ≈ 0.020 for all
instances (dominated by `SAT_DELTA = 0.02`); since `d < 6/255`, generate.py snaps to `EPS_SAT`.
Verified by VERIFY_v2.md Task 4: tightest margin = 6/255 − 0.020 = 0.003529.

---

## 6. Source Image Indices (shared pool)

25 MNIST test images form the shared pool. Each source image generates one UNSAT property
(prop_k on h8) and one SAT property (prop_{k+25} on h64). Pool order corresponds to k = 0…24.

```
k  → MNIST test index  cls    k  → MNIST test index  cls
 0 → 8501               0    13 → 8248               7
 1 → 2087               0    14 → 2442               7
 2 → 6191               0    15 → 7758               6
 3 → 8528               0    16 → 7753               6
 4 → 5452               0    17 → 9178               6
 5 → 3281               1    18 →  130               6
 6 → 6901               1    19 → 6122               6
 7 → 1193               1    20 → 3221               3
 8 → 4931               1    21 → 8082               5
 9 → 9291               1    22 → 1365               2
10 → 7744               7    23 → 4600               3
11 → 9586               7    24 → 6413               3
12 → 5297               7
```

Class distribution: 0 (5), 1 (5), 7 (5), 6 (5), 3 (3), 5 (1), 2 (1). **7 classes** represented
(0, 1, 2, 3, 5, 6, 7). Classes 4, 8, 9 absent — pool cap (max 5 per class) fills the
highest-margin classes first; those three digits have lower IBP margins under this permutation.

All indices are from the MNIST **test** split (indices 0–9999). No training images are used.
No two properties share a source image. Verified by VERIFY_v2.md Task 1: 25/25 UNSAT centers
reconstruct to their source image with L∞ ≤ 3 × 10⁻⁸ (format-rounding residual only).

---

## 7. Validation Status

### UNSAT (25 instances — prop_000–024)

| Verifier | Method | Result | Min margin | Reference |
|---|---|---|---|---|
| n2v IBP (CROWN-IBP regularised) | Interval bound propagation | 25/25 certified | — | generate.py at generation time |
| auto_LiRPA 0.7.2 | CROWN (linear relaxation) | **25/25 CONFIRMED** | +4.317 logits (prop_022) | VALIDATION_EXTERNAL.md (2026-06-23) |
| alpha-beta-CROWN 0.7.0 | Full harness (abcrown.py, CROWN mode) | **25/25 safe-incomplete** | — | VALIDATION_ABCROWN.md (2026-06-23) |

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

Versions confirmed via `conda run -n abcrown pip show <pkg>`. No requirements file for
the abcrown env is pinned in this repo.

---

## 10. Known Notes

**IBP-regularised training (h8 only)**: The h8 model is trained with a CROWN-IBP style combined
loss (clean CE + IBP worst-case CE, lambda 0→1 over epochs 11–40). This is required because
plain /255 inputs (mean ≈ 0.13) lead to larger model weights than Z-score inputs, which blows
up IBP interval expansion through 14 LSTM timesteps. The IBP-regularised training directly
penalises non-certifiable behaviour and achieves ~87% IBP certification rate on correctly-
classified test images at ε=1/255. The h64 model uses standard Adam training (no IBP loss needed
since it is only used for SAT instances via PGD falsification).

**Class coverage**: Classes 4, 8, 9 are absent from the 25-instance pool. The pool cap
(max 5 per class) fills the 5 highest-IBP-margin classes to their cap first (0, 1, 7, 6 × 5
each), leaving only 5 slots distributed among the remaining classes: 3 gets 3, 5 gets 1, 2
gets 1. Seven classes total (0, 1, 2, 3, 5, 6, 7) are represented — up from 6 in the
top-half v2 design. Reducing MAX_PER_CLASS from 5 to 3 would yield all 10 classes but would
require a larger source image search across the test set.

**Parser note (`validate_crown_random392.py`)**: VNN-LIB `(assert (>= X_i val))` means X_i ≥
val (lower bound); `(assert (<= X_i val))` means X_i ≤ val (upper bound). Earlier versions of
the parser had these reversed; fixed in the v3 random-392 rebuild.
