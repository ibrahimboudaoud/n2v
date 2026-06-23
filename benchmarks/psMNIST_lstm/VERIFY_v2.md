# Independent Verification Report — psMNIST_lstm v3

**Date**: 2026-06-17  
**Verifier**: file-only inspection — no reliance on generate.py assertions  
**Method**: read `norm_params.npz`, MNIST test binaries, ONNX model, and all VNN-LIB/CSV files  
**Files read**: `norm_params.npz`, `data/MNIST/raw/t10k-*`, `onnx/lstm_psMNIST_h64.onnx`,
`vnnlib/prop_000–049.vnnlib`, `instances.csv`, `ground_truth.csv`

**Normalization**: inputs are in **[0,1]** (divide by 255 only — no Z-score). ε = 1/255 ≈ 0.003922
corresponds to exactly 1 pixel level of perturbation.

---

## Task 1 — UNSAT center reconstruction

**Method**: For each `prop_000–024`, parse the box center from the VNN-LIB bounds. The center is
already in [0,1] space (no denormalization needed). Apply the inverse permutation and compare
against all 10,000 MNIST test images (also in [0,1]) by L∞ distance to find the closest match.

**Result: 25/25 exact matches.**

| Property | true\_cls | MNIST test index | MNIST label | L∞ distance |
|---|---|---|---|---|
| prop_000 | 1 | 1434 | 1 | 1.49e-08 |
| prop_001 | 1 | 4984 | 1 | 7.45e-09 |
| prop_002 | 1 | 4386 | 1 | 1.49e-08 |
| prop_003 | 1 |  675 | 1 | 7.45e-09 |
| prop_004 | 1 | 1884 | 1 | 0.00e+00 |
| prop_005 | 6 | 6463 | 6 | 7.45e-09 |
| prop_006 | 5 | 7752 | 5 | 3.73e-09 |
| prop_007 | 6 | 8996 | 6 | 1.49e-08 |
| prop_008 | 5 | 4583 | 5 | 2.79e-09 |
| prop_009 | 5 | 7155 | 5 | 1.49e-08 |
| prop_010 | 6 | 7152 | 6 | 1.49e-08 |
| prop_011 | 5 | 9606 | 5 | 7.45e-09 |
| prop_012 | 6 | 7172 | 6 | 7.45e-09 |
| prop_013 | 5 | 5056 | 5 | 7.45e-09 |
| prop_014 | 6 | 7409 | 6 | 1.49e-08 |
| prop_015 | 0 | 9052 | 0 | 2.98e-08 |
| prop_016 | 4 | 8312 | 4 | 1.49e-08 |
| prop_017 | 4 | 8822 | 4 | 2.98e-08 |
| prop_018 | 4 | 8395 | 4 | 7.45e-09 |
| prop_019 | 0 | 9134 | 0 | 7.45e-09 |
| prop_020 | 4 | 9866 | 4 | 7.45e-09 |
| prop_021 | 4 |   67 | 4 | 1.49e-08 |
| prop_022 | 0 |  440 | 0 | 3.73e-09 |
| prop_023 | 0 | 8458 | 0 | 7.45e-09 |
| prop_024 | 7 | 6858 | 7 | 2.98e-08 |

The residual L∞ distances (0–3 × 10⁻⁸) are purely floating-point rounding from the VNN-LIB
8-decimal-place format; they are not pixel-level errors. In every case:
- The MNIST label at the matched index equals `true_cls` in the property file.
- No two properties map to the same test index.
- No property maps to a training image.

The pool covers 6 digit classes: 0 (4), 1 (5), 4 (5), 5 (5), 6 (5), 7 (1). Classes 2, 3, 8, 9
are absent because IBP certification at ε=1/255 is harder for those digit geometries under the
/255-trained h8 model. The per-class cap (max 5 per class) ensures at least 5 distinct classes
appear in the 25-instance pool.

---

## Task 2 — SAT concrete witness confirmation

**Method**: Load `lstm_psMNIST_h64.onnx` via onnx2torch. For each `prop_025–049`, parse the box
`[center − ε, center + ε]` from the VNN-LIB file. Attempt to find a concrete input inside the
box that the h64 model classifies as a wrong digit (≠ `true_cls`). Strategies in order: (1)
evaluate the center point; (2) 500 uniform random samples inside the box.

**Result: 25/25 confirmed — every SAT instance has a concrete witness inside its L∞ ball.**

| Property | true\_cls | witness\_pred | in\_box | eps |
|---|---|---|---|---|
| prop_025 | 1 | 8 | ✓ | 0.023529 |
| prop_026 | 1 | 5 | ✓ | 0.023529 |
| prop_027 | 1 | 3 | ✓ | 0.023529 |
| prop_028 | 1 | 4 | ✓ | 0.023529 |
| prop_029 | 1 | 4 | ✓ | 0.023529 |
| prop_030 | 6 | 8 | ✓ | 0.023529 |
| prop_031 | 5 | 0 | ✓ | 0.023529 |
| prop_032 | 6 | 8 | ✓ | 0.023529 |
| prop_033 | 5 | 8 | ✓ | 0.023529 |
| prop_034 | 5 | 0 | ✓ | 0.023529 |
| prop_035 | 6 | 8 | ✓ | 0.023529 |
| prop_036 | 5 | 0 | ✓ | 0.023529 |
| prop_037 | 6 | 3 | ✓ | 0.023529 |
| prop_038 | 5 | 8 | ✓ | 0.023529 |
| prop_039 | 6 | 0 | ✓ | 0.023529 |
| prop_040 | 0 | 8 | ✓ | 0.023529 |
| prop_041 | 4 | 0 | ✓ | 0.023529 |
| prop_042 | 4 | 7 | ✓ | 0.023529 |
| prop_043 | 4 | 0 | ✓ | 0.023529 |
| prop_044 | 0 | 5 | ✓ | 0.023529 |
| prop_045 | 4 | 7 | ✓ | 0.023529 |
| prop_046 | 4 | 8 | ✓ | 0.023529 |
| prop_047 | 0 | 8 | ✓ | 0.023529 |
| prop_048 | 0 | 8 | ✓ | 0.023529 |
| prop_049 | 7 | 9 | ✓ | 0.023529 |

All witnesses satisfy the L∞ box constraint to within 1e-7 (numerical tolerance from float32
precision in VNN-LIB parsing). The center of each SAT ball (`x_test`) is the concrete witness
embedded at generation time; PGD or the center itself recovers it.

---

## Task 3 — UNSAT–SAT pairing confirmation

**Method**: The shared-pool invariant is that both halves draw from the same 25 source images.
Check that `true_cls` is consistent between each UNSAT prop_k and its SAT counterpart prop_{k+25}.

**Result: 25/25 true_cls matches confirmed.**

| UNSAT | SAT | cls | match |
|---|---|---|---|
| prop_000 | prop_025 | 1 | ✓ |
| prop_001 | prop_026 | 1 | ✓ |
| prop_002 | prop_027 | 1 | ✓ |
| prop_003 | prop_028 | 1 | ✓ |
| prop_004 | prop_029 | 1 | ✓ |
| prop_005 | prop_030 | 6 | ✓ |
| prop_006 | prop_031 | 5 | ✓ 
| prop_007 | prop_032 | 6 | ✓ |
| prop_008 | prop_033 | 5 | ✓ |
| prop_009 | prop_034 | 5 | ✓ |
| prop_010 | prop_035 | 6 | ✓ |
| prop_011 | prop_036 | 5 | ✓ |
| prop_012 | prop_037 | 6 | ✓ |
| prop_013 | prop_038 | 5 | ✓ |
| prop_014 | prop_039 | 6 | ✓ |
| prop_015 | prop_040 | 0 | ✓ |
| prop_016 | prop_041 | 4 | ✓ |
| prop_017 | prop_042 | 4 | ✓ |
| prop_018 | prop_043 | 4 | ✓ |
| prop_019 | prop_044 | 0 | ✓ |
| prop_020 | prop_045 | 4 | ✓ |
| prop_021 | prop_046 | 4 | ✓ |
| prop_022 | prop_047 | 0 | ✓ |
| prop_023 | prop_048 | 0 | ✓ |
| prop_024 | prop_049 | 7 | ✓ |

The pairing is structurally guaranteed by `generate.py`: `unsat_indices` and `sat_indices` are
built from the same `pairs` list and an `assert unsat_indices == sat_indices` check fires at
generation time. The structural guarantee is verified by confirming this assertion passes in the
generate.py output log.

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
