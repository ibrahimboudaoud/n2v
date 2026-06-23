# psMNIST LSTM Benchmark — Facts and Provenance

Single-source record of all factual details for `benchmarks/psMNIST_lstm/`.
Intended to be stable across re-runs and to serve as the canonical reference for the
submission writeup.

**Version**: v3 random-392, all-10 class composition (branch `experiment-all10`)

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

---

## 3. ONNX Models

Models are unchanged from `random392-7class`. Only instance selection differs.

### 3.1 `onnx/lstm_psMNIST_h8.onnx` — UNSAT half

| Item | Value |
|---|---|
| **Hidden size** | 8 |
| **Architecture** | Single-layer LSTM (14 timesteps, input_size=28 per step), nn.Linear(8, 10) output |
| **Trainable parameters** | ~1,306 |
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
| **Trainable parameters** | ~24,714 |
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
Verified by VERIFY_v2.md Task 4.

---

## 6. Source Image Indices (shared pool)

25 MNIST test images form the shared pool. Each source image generates one UNSAT property
(prop_k on h8) and one SAT property (prop_{k+25} on h64). Pool order corresponds to k = 0…24.

Pool is built by `regen_pool_all10.py` with MAX_PER_CLASS=3, all-10 guarantee: first 1 from
each class (mandatory, highest h8 margin), then fill remaining 15 slots by descending margin.

```
k  → MNIST test index  cls    k  → MNIST test index  cls
 0 → 8501               0    13 → 1193               1
 1 → 3281               1    14 → 9586               7
 2 → 1365               2    15 → 5297               7
 3 → 3221               3    16 → 7753               6
 4 → 1958               4    17 → 9178               6
 5 → 8082               5    18 → 4600               3
 6 → 7758               6    19 → 6413               3
 7 → 7744               7    20 → 9114               5
 8 → 9420               8    21 → 6270               5
 9 → 6512               9    22 → 7093               2
10 → 2087               0    23 → 3176               2
11 → 6191               0    24 → 8685               4
12 → 6901               1
```

Class distribution: 0 (3), 1 (3), 2 (3), 3 (3), 4 (2), 5 (3), 6 (3), 7 (3), 8 (1), 9 (1).
**All 10 classes represented.** Classes 4, 8, 9 have fewer slots because the fill phase exhausts
its 15 slots on higher-margin classes before reaching the cap for 4, 8, 9.

All indices are from the MNIST **test** split (indices 0–9999). No training images are used.
No two properties share a source image.

---

## 7. Validation Status

### UNSAT (25 instances — prop_000–024)

| Verifier | Method | Result | Min margin | Reference |
|---|---|---|---|---|
| n2v IBP (CROWN-IBP regularised) | Interval bound propagation | 25/25 certified | — | regen_pool_all10.py + n2v dry run |
| auto_LiRPA 0.7.2 | CROWN (linear relaxation) | **25/25 CONFIRMED** | +2.882 logits (prop_008, cls=8) | VALIDATION_EXTERNAL.md (2026-06-23) |
| alpha-beta-CROWN 0.7.0 | Full harness (abcrown.py, CROWN mode) | **25/25 safe-incomplete** | — | VALIDATION_ABCROWN.md (2026-06-23) |

### SAT (25 instances — prop_025–049)

| Verifier | Method | Result | Reference |
|---|---|---|---|
| regen_pool_all10.py | Concrete witness inside L∞ ball | 25/25 witnesses confirmed | VERIFY_v2.md Task 2 |
| n2v dry-run | PGD falsification | 25/25 pass | `run_n2v_dryrun()` |

---

## 8. Regenerate Command

```bash
# Full pipeline from scratch (trains models, exports ONNX, builds pool):
cd benchmarks/psMNIST_lstm
python generate.py

# Pool-only rebuild (uses pre-trained .pt checkpoints, no retraining):
cd benchmarks/psMNIST_lstm
python regen_pool_all10.py
```

SEED=42 is hardcoded; the run is deterministic given the same MNIST test data and PyTorch version.

---

## 9. Pinned Versions (abcrown env — used for CROWN external validation)

| Package | Version | Notes |
|---|---|---|
| Python | 3.11.15 | conda env `abcrown` |
| PyTorch | 2.12.0 | CPU build |
| onnx | 1.21.0 | |
| onnx2torch | 1.5.15 | |
| auto_LiRPA | 0.7.2 | installed from GitHub HEAD |
| macOS / hardware | 15.2, Apple Silicon arm64 | CPU only |

---

## 10. Known Notes

**All-10 class composition**: Pool algorithm ensures all 10 digit classes appear by taking
1 mandatory instance from each class first, then filling by margin. Classes 4, 8, 9 have fewer
instances (2, 1, 1) because they fall at the bottom of the h8 IBP margin ranking — those digits
are harder to certify under the random-392 permutation. Reducing MAX_PER_CLASS further or
increasing N_POOL would give more even distribution.

**IBP-regularised training (h8 only)**: The h8 model is trained with a CROWN-IBP style combined
loss (clean CE + IBP worst-case CE, lambda 0→1 over epochs 11–40). This achieves ~87% IBP
certification rate on correctly-classified test images at ε=1/255. The h64 model uses standard
Adam training.

**Parser note (`validate_crown_random392.py`)**: VNN-LIB `(assert (>= X_i val))` means X_i ≥
val (lower bound); `(assert (<= X_i val))` means X_i ≤ val (upper bound). Fixed in this version.
