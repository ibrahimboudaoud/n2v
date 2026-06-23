# Independent Verification Report — psMNIST_lstm v3 all-10

**Date**: 2026-06-23
**Verifier**: file-only inspection — no reliance on generate.py assertions
**Method**: read `norm_params.npz`, MNIST test binaries, ONNX model, and all VNN-LIB/CSV files

**Normalization**: inputs are in **[0,1]** (divide by 255 only). ε = 1/255 ≈ 0.003922
corresponds to exactly 1 pixel level of perturbation.

---

## Task 1 — UNSAT center reconstruction

**Method**: Parse box center from each VNN-LIB. Apply the random-392 permutation (from
`norm_params.npz`) and find the closest MNIST test image by L∞ distance.

**Result: 25/25 exact matches.**

| Property | true\_cls | MNIST test index | MNIST label | L∞ distance |
|---|---|---|---|---|
| prop_000 | 0 |  8501 | 0 | 3.00e-08 |
| prop_001 | 1 |  3281 | 1 | 3.11e-09 |
| prop_002 | 2 |  1365 | 2 | 1.13e-08 |
| prop_003 | 3 |  3221 | 3 | 3.89e-09 |
| prop_004 | 4 |  1958 | 4 | 3.37e-09 |
| prop_005 | 5 |  8082 | 5 | 3.00e-08 |
| prop_006 | 6 |  7758 | 6 | 3.00e-08 |
| prop_007 | 7 |  7744 | 7 | 3.00e-08 |
| prop_008 | 8 |  9420 | 8 | 3.00e-08 |
| prop_009 | 9 |  6512 | 9 | 3.00e-08 |
| prop_010 | 0 |  2087 | 0 | 1.42e-08 |
| prop_011 | 0 |  6191 | 0 | 3.00e-08 |
| prop_012 | 1 |  6901 | 1 | 3.32e-09 |
| prop_013 | 1 |  1193 | 1 | 2.84e-09 |
| prop_014 | 7 |  9586 | 7 | 3.00e-08 |
| prop_015 | 7 |  5297 | 7 | 3.00e-08 |
| prop_016 | 6 |  7753 | 6 | 3.00e-08 |
| prop_017 | 6 |  9178 | 6 | 3.21e-09 |
| prop_018 | 3 |  4600 | 3 | 3.89e-09 |
| prop_019 | 3 |  6413 | 3 | 3.00e-08 |
| prop_020 | 5 |  9114 | 5 | 3.00e-08 |
| prop_021 | 5 |  6270 | 5 | 3.00e-08 |
| prop_022 | 2 |  7093 | 2 | 1.13e-08 |
| prop_023 | 2 |  3176 | 2 | 3.37e-09 |
| prop_024 | 4 |  8685 | 4 | 3.00e-08 |

The residual L∞ distances (≤ 3 × 10⁻⁸) are purely floating-point rounding from the VNN-LIB
8-decimal-place format. In every case the MNIST label equals `true_cls`. No two properties
map to the same test index. No property maps to a training image.

The pool covers all **10 digit classes**: 0 (3), 1 (3), 2 (3), 3 (3), 4 (2), 5 (3), 6 (3),
7 (3), 8 (1), 9 (1). The two-phase selection algorithm in `regen_pool_all10.py` guarantees
every class appears: 1 mandatory instance per class from Phase A, then 15 fill slots by
descending h8 logit margin. Classes 4, 8, 9 have fewer slots because they exhaust the fill
quota before higher-margin classes do.

---

## Task 2 — SAT concrete witness confirmation

**Method**: Load `lstm_psMNIST_h64.onnx` via onnx2torch. For each `prop_025–049`, parse
`[center − ε, center + ε]`. Find a concrete input inside the box that h64 classifies as a
wrong digit (≠ `true_cls`) by: (1) center point, (2) 2000 uniform random samples.

**Result: 25/25 confirmed — every SAT instance has a concrete witness inside its L∞ ball.**

| Property | true\_cls | witness\_pred | in\_box | eps |
|---|---|---|---|---|
| prop_025 | 0 | 6 | ✓ | 0.023529 |
| prop_026 | 1 | 8 | ✓ | 0.023529 |
| prop_027 | 2 | 1 | ✓ | 0.023529 |
| prop_028 | 3 | 9 | ✓ | 0.023529 |
| prop_029 | 4 | 9 | ✓ | 0.023529 |
| prop_030 | 5 | 8 | ✓ | 0.023529 |
| prop_031 | 6 | 4 | ✓ | 0.023529 |
| prop_032 | 7 | 2 | ✓ | 0.023529 |
| prop_033 | 8 | 2 | ✓ | 0.023529 |
| prop_034 | 9 | 0 | ✓ | 0.023529 |
| prop_035 | 0 | 9 | ✓ | 0.023529 |
| prop_036 | 0 | 9 | ✓ | 0.023529 |
| prop_037 | 1 | 4 | ✓ | 0.023529 |
| prop_038 | 1 | 9 | ✓ | 0.023529 |
| prop_039 | 7 | 3 | ✓ | 0.023529 |
| prop_040 | 7 | 2 | ✓ | 0.023529 |
| prop_041 | 6 | 2 | ✓ | 0.023529 |
| prop_042 | 6 | 0 | ✓ | 0.023529 |
| prop_043 | 3 | 8 | ✓ | 0.023529 |
| prop_044 | 3 | 2 | ✓ | 0.023529 |
| prop_045 | 5 | 8 | ✓ | 0.023529 |
| prop_046 | 5 | 9 | ✓ | 0.023529 |
| prop_047 | 2 | 3 | ✓ | 0.023529 |
| prop_048 | 2 | 3 | ✓ | 0.023529 |
| prop_049 | 4 | 8 | ✓ | 0.023529 |

All 10 classes appear in the SAT half (same source images as UNSAT half).

---

## Task 3 — UNSAT–SAT pairing confirmation

**Result: 25/25 true_cls matches confirmed.**

| UNSAT | SAT | cls | match |
|---|---|---|---|
| prop_000 | prop_025 | 0 | ✓ |
| prop_001 | prop_026 | 1 | ✓ |
| prop_002 | prop_027 | 2 | ✓ |
| prop_003 | prop_028 | 3 | ✓ |
| prop_004 | prop_029 | 4 | ✓ |
| prop_005 | prop_030 | 5 | ✓ |
| prop_006 | prop_031 | 6 | ✓ |
| prop_007 | prop_032 | 7 | ✓ |
| prop_008 | prop_033 | 8 | ✓ |
| prop_009 | prop_034 | 9 | ✓ |
| prop_010 | prop_035 | 0 | ✓ |
| prop_011 | prop_036 | 0 | ✓ |
| prop_012 | prop_037 | 1 | ✓ |
| prop_013 | prop_038 | 1 | ✓ |
| prop_014 | prop_039 | 7 | ✓ |
| prop_015 | prop_040 | 7 | ✓ |
| prop_016 | prop_041 | 6 | ✓ |
| prop_017 | prop_042 | 6 | ✓ |
| prop_018 | prop_043 | 3 | ✓ |
| prop_019 | prop_044 | 3 | ✓ |
| prop_020 | prop_045 | 5 | ✓ |
| prop_021 | prop_046 | 5 | ✓ |
| prop_022 | prop_047 | 2 | ✓ |
| prop_023 | prop_048 | 2 | ✓ |
| prop_024 | prop_049 | 4 | ✓ |

---

## Task 4 — SAT epsilon values

**Result**: All 25 SAT instances have `eps = 6/255 ≈ 0.02352941` (exact).

Boundary search converges to d ≈ 0.020000 (dominated by `SAT_DELTA = 0.02`).
Since d < 6/255 = 0.023529, `regen_pool_all10.py` snaps all instances to `EPS_SAT = 6/255`.
No instance had d > EPS_SAT (confirmed by the no-bad-eps check in Phase A).

In pixel terms:
- ε_UNSAT = 1/255 ≈ 0.003922 → **exactly 1 pixel level** of L∞ perturbation
- ε_SAT   = 6/255 ≈ 0.023529 → **exactly 6 pixel levels** of L∞ perturbation
