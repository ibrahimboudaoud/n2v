# psMNIST LSTM Benchmark (v3 — /255 normalization, k/255 epsilon)

VNN-COMP 2026 robustness benchmark for recurrent networks.

## Quick start

```bash
cd benchmarks/psMNIST_lstm
python generate.py
```

Requires: `torch torchvision onnx numpy`

## Design

### Shared image pool

50 instances drawn from **25 source images**. Each source image produces exactly two instances:

| Property index | Model | Result | Epsilon |
|---|---|---|---|
| prop_000 … prop_024 | `lstm_psMNIST_h8.onnx` | UNSAT | 1/255 ≈ 0.003922 (1 pixel level, fixed) |
| prop_025 … prop_049 | `lstm_psMNIST_h64.onnx` | SAT | 0.022 ≈ 5.6/255 (effectively fixed) |

Image `i` maps to `prop_i` (UNSAT) and `prop_{i+25}` (SAT). Only **(model, epsilon)** vary between the two halves — image identity is held constant. This removes the confound present in the v1 design (`benchmarks/psMNIST_lstm_v1_confounded/`) where model, epsilon, and image all changed together.

### Why two models are still necessary

IBP (Interval Bound Propagation) is the only sound UNSAT certifier n2v currently supports for LSTMs. IBP error compounds through the element-wise gate multiplications (`f⊙c`, `o⊙tanh(c)`) at each of the 14 timesteps. With `hidden=64`, the interval widths grow too fast for IBP to certify anything. `hidden=8` keeps weight norms small enough that IBP stays tight.

The h64 model is required for SAT because its well-trained decision boundary produces geometrically tight bisections. The h8 model's decision boundary is too coarse (83% accuracy) for reliable falsification at small epsilon values.

**In short**: a single model cannot satisfy both requirements. UNSAT requires IBP-certifiability (h8); SAT requires a sharp decision boundary (h64).

### Epsilon values and normalization

Inputs are normalized by dividing by 255 (plain `/255`, no Z-score). Inputs live in **[0,1]**.
This makes epsilon directly interpretable as pixel-level perturbation: ε = k/255 means exactly
k plus-or-minus pixel levels of perturbation (standard VNN-COMP image benchmark convention).

- **UNSAT ε = 1/255 ≈ 0.003922**: fixed constant (`EPS_UNSAT` in `generate.py`). Exactly
  **1 pixel level** of L∞ perturbation. The h8 model is trained with IBP regularisation
  (CROWN-IBP style) to be certifiable at this epsilon.
- **SAT ε = 0.022 (effectively fixed)**: ≈ 5.6 pixel levels. Computed as
  `1.1 × L∞(x_test, boundary_point)` via 50-step binary search (`SAT_BISECT_STEPS`,
  `SAT_DELTA` in `generate.py`). After 50 bisections `L∞(x_test, boundary) ≈ delta = 0.02`,
  so `eps = 1.1 × 0.02 = 0.022` for all 25 instances.

The permutation index is stored in `norm_params.npz` (only the permutation — no mean/std needed).

### Instance selection

Candidates are filtered jointly: a source image is accepted only if
1. both h8 and h64 correctly classify it,
2. `ibp_certify(h8, x, 1/255, true_class)` returns True, and
3. `find_sat_instance(h64, x, ...)` finds a valid boundary witness.

Candidates are sorted by h8 logit margin (descending) to maximise IBP certifiability. A per-class
cap (max 5 per digit class) ensures class diversity. The first `N_POOL=25` images passing all
filters form the shared pool (covering 6 digit classes: 0, 1, 4, 5, 6, 7).

## Reproducibility

Fully deterministic from seed 42. Pinned versions:

| Package | Version |
|---|---|
| torch | 2.12.0 |
| onnx | 1.21.0 |
| onnx2torch | 1.5.15 |
| auto_LiRPA | 0.7.2 |

Re-running `python generate.py` from a clean directory produces bit-identical ONNX models, VNN-LIB files, and CSV index files.

## File format

- `instances.csv` — no header; columns: `onnx_path, vnnlib_path, timeout`
- `ground_truth.csv` — header row; columns: `onnx, vnnlib, result, timeout`
- VNN-LIB 2.0 multi-class disjunction: SAT = some wrong digit beats the true digit
- ONNX opset 14, Unix LF line endings throughout

## Verification status

- **UNSAT (25)**: certified by n2v IBP-regularised training at generation time; independently confirmed 25/25 by auto_LiRPA 0.7.2 CROWN (min margin +3.188 logits, prop_024); end-to-end confirmed by alpha-beta-CROWN 0.7.0 harness (50/50 MATCH, 0 timeout). See `VALIDATION_EXTERNAL.md` and `VALIDATION_ABCROWN.md`.
- **SAT (25)**: concrete witnesses confirmed inside every L∞ ball by file-only inspection; confirmed by alpha-beta-CROWN PGD (<0.15 s each). See `VERIFY_v2.md`.

## Files

- `onnx/` — the two LSTM models
- `vnnlib/` — 50 property files (prop_000–024 UNSAT, prop_025–049 SAT)
- `instances.csv` — (onnx, vnnlib, timeout); no result column
- `ground_truth.csv` — labels for self-checking
- `generate.py` — deterministic generator
- `norm_params.npz` — normalization stats + fixed permutation
- `VALIDATION_EXTERNAL.md`, `VERIFY_v2.md`, `README_FACTS.md` — validation evidence

## License

MIT — see `LICENSE`.
