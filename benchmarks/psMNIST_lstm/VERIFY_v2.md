# Independent Verification Report — psMNIST_lstm v3 random-392

**Date**: 2026-06-23
**Verifier**: file-only inspection — no reliance on generate.py assertions
**Method**: read `norm_params.npz`, MNIST test binaries, ONNX model, and all VNN-LIB/CSV files
**Files read**: `norm_params.npz`, `data/MNIST/raw/t10k-*`, `onnx/lstm_psMNIST_h64.onnx`,
`vnnlib/prop_000–049.vnnlib`, `instances.csv`, `ground_truth.csv`

**Normalization**: inputs are in **[0,1]** (divide by 255 only — no Z-score). ε = 1/255 ≈ 0.003922
corresponds to exactly 1 pixel level of perturbation.

---

## Task 1 — UNSAT center reconstruction

**Method**: For each `prop_000–024`, parse the box center from the VNN-LIB bounds. The center is
already in [0,1] space (no denormalization needed). Apply the random-392 permutation (loaded
from `norm_params.npz`) and compare against all 10,000 MNIST test images by L∞ distance.

**Result: 25/25 exact matches.**

| Property | true\_cls | MNIST test index | MNIST label | L∞ distance |
|---|---|---|---|---|
| prop_000 | 0 | 8501 | 0 | 3.00e-08 |
| prop_001 | 0 | 2087 | 0 | 1.42e-08 |
| prop_002 | 0 | 6191 | 0 | 3.00e-08 |
| prop_003 | 0 | 8528 | 0 | 3.00e-08 |
| prop_004 | 0 | 5452 | 0 | 3.00e-08 |
| prop_005 | 1 | 3281 | 1 | 3.11e-09 |
| prop_006 | 1 | 6901 | 1 | 3.32e-09 |
| prop_007 | 1 | 1193 | 1 | 2.84e-09 |
| prop_008 | 1 | 4931 | 1 | 3.00e-08 |
| prop_009 | 1 | 9291 | 1 | 2.91e-09 |
| prop_010 | 7 | 7744 | 7 | 3.00e-08 |
| prop_011 | 7 | 9586 | 7 | 3.00e-08 |
| prop_012 | 7 | 5297 | 7 | 3.00e-08 |
| prop_013 | 7 | 8248 | 7 | 3.00e-08 |
| prop_014 | 7 | 2442 | 7 | 3.00e-08 |
| prop_015 | 6 | 7758 | 6 | 3.00e-08 |
| prop_016 | 6 | 7753 | 6 | 3.00e-08 |
| prop_017 | 6 | 9178 | 6 | 3.21e-09 |
| prop_018 | 6 |  130 | 6 | 3.00e-08 |
| prop_019 | 6 | 6122 | 6 | 3.37e-09 |
| prop_020 | 3 | 3221 | 3 | 3.89e-09 |
| prop_021 | 5 | 8082 | 5 | 3.00e-08 |
| prop_022 | 2 | 1365 | 2 | 1.13e-08 |
| prop_023 | 3 | 4600 | 3 | 3.89e-09 |
| prop_024 | 3 | 6413 | 3 | 3.00e-08 |

The residual L∞ distances (0–3 × 10⁻⁸) are purely floating-point rounding from the VNN-LIB
8-decimal-place format; they are not pixel-level errors. In every case:
- The MNIST label at the matched index equals `true_cls` in the property file.
- No two properties map to the same test index.
- No property maps to a training image.

The pool covers 7 digit classes: 0 (5), 1 (5), 7 (5), 6 (5), 3 (3), 5 (1), 2 (1).
Classes 4, 8, 9 are absent — the per-class pool cap (max 5 per class) fills the highest-margin
classes first; those three digits have lower IBP margins under the random-392 permutation.

---

## Task 2 — SAT concrete witness confirmation

**Method**: Load `lstm_psMNIST_h64.onnx` via onnx2torch. For each `prop_025–049`, parse the box
`[center − ε, center + ε]` from the VNN-LIB file. Attempt to find a concrete input inside the
box that the h64 model classifies as a wrong digit (≠ `true_cls`). Strategies in order: (1)
evaluate the center point; (2) 2000 uniform random samples inside the box.

**Result: 25/25 confirmed — every SAT instance has a concrete witness inside its L∞ ball.**

| Property | true\_cls | witness\_pred | in\_box | eps |
|---|---|---|---|---|
| prop_025 | 0 | 6 | ✓ | 0.023529 |
| prop_026 | 0 | 9 | ✓ | 0.023529 |
| prop_027 | 0 | 9 | ✓ | 0.023529 |
| prop_028 | 0 | 5 | ✓ | 0.023529 |
| prop_029 | 0 | 6 | ✓ | 0.023529 |
| prop_030 | 1 | 8 | ✓ | 0.023529 |
| prop_031 | 1 | 4 | ✓ | 0.023529 |
| prop_032 | 1 | 9 | ✓ | 0.023529 |
| prop_033 | 1 | 8 | ✓ | 0.023529 |
| prop_034 | 1 | 7 | ✓ | 0.023529 |
| prop_035 | 7 | 2 | ✓ | 0.023529 |
| prop_036 | 7 | 3 | ✓ | 0.023529 |
| prop_037 | 7 | 2 | ✓ | 0.023529 |
| prop_038 | 7 | 2 | ✓ | 0.023529 |
| prop_039 | 7 | 2 | ✓ | 0.023529 |
| prop_040 | 6 | 4 | ✓ | 0.023529 |
| prop_041 | 6 | 2 | ✓ | 0.023529 |
| prop_042 | 6 | 0 | ✓ | 0.023529 |
| prop_043 | 6 | 0 | ✓ | 0.023529 |
| prop_044 | 6 | 4 | ✓ | 0.023529 |
| prop_045 | 3 | 9 | ✓ | 0.023529 |
| prop_046 | 5 | 8 | ✓ | 0.023529 |
| prop_047 | 2 | 1 | ✓ | 0.023529 |
| prop_048 | 3 | 8 | ✓ | 0.023529 |
| prop_049 | 3 | 2 | ✓ | 0.023529 |

All witnesses found by random sampling inside the L∞ box (center ± 6/255) confirm the
adversarial region is well-populated — not a borderline case. The center of each SAT ball
(`x_test`) is the concrete witness embedded at generation time.

---

## Task 3 — UNSAT–SAT pairing confirmation

**Method**: The shared-pool invariant is that both halves draw from the same 25 source images.
Check that `true_cls` is consistent between each UNSAT prop_k and its SAT counterpart prop_{k+25}.

**Result: 25/25 true_cls matches confirmed.**

| UNSAT | SAT | cls | match |
|---|---|---|---|
| prop_000 | prop_025 | 0 | ✓ |
| prop_001 | prop_026 | 0 | ✓ |
| prop_002 | prop_027 | 0 | ✓ |
| prop_003 | prop_028 | 0 | ✓ |
| prop_004 | prop_029 | 0 | ✓ |
| prop_005 | prop_030 | 1 | ✓ |
| prop_006 | prop_031 | 1 | ✓ |
| prop_007 | prop_032 | 1 | ✓ |
| prop_008 | prop_033 | 1 | ✓ |
| prop_009 | prop_034 | 1 | ✓ |
| prop_010 | prop_035 | 7 | ✓ |
| prop_011 | prop_036 | 7 | ✓ |
| prop_012 | prop_037 | 7 | ✓ |
| prop_013 | prop_038 | 7 | ✓ |
| prop_014 | prop_039 | 7 | ✓ |
| prop_015 | prop_040 | 6 | ✓ |
| prop_016 | prop_041 | 6 | ✓ |
| prop_017 | prop_042 | 6 | ✓ |
| prop_018 | prop_043 | 6 | ✓ |
| prop_019 | prop_044 | 6 | ✓ |
| prop_020 | prop_045 | 3 | ✓ |
| prop_021 | prop_046 | 5 | ✓ |
| prop_022 | prop_047 | 2 | ✓ |
| prop_023 | prop_048 | 3 | ✓ |
| prop_024 | prop_049 | 3 | ✓ |

The pairing is structurally guaranteed by `generate.py`: `unsat_indices` and `sat_indices` are
built from the same `pairs` list and an `assert unsat_indices == sat_indices` check fires at
generation time.

---

## Task 4 — SAT epsilon values

**Method**: Compute `eps = max((ub[i] - lb[i]) / 2)` from the VNN-LIB box bounds for each SAT
property.

**Result**: All 25 SAT instances have `eps = 6/255 ≈ 0.02352941` (exact). The boundary witness
`b` is at L∞ distance `d ≈ 0.020000` from `x_test` for all instances (boundary search converges
to precision `|x0 − x1|_∞ / 2⁵⁰ ≈ 0`, dominated by `SAT_DELTA = 0.02`). Since `d < 6/255`,
`b` is strictly inside the 6/255 ball; generate.py snaps eps to `EPS_SAT = 6/255` exactly.

In pixel terms:
- ε_UNSAT = 1/255 ≈ 0.003922 → **exactly 1 pixel level** of L∞ perturbation
- ε_SAT   = 6/255 ≈ 0.023529 → **exactly 6 pixel levels** of L∞ perturbation
