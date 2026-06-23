# External UNSAT Validation Report — psMNIST_lstm v3 all-10

**Date**: 2026-06-23
**Benchmark**: `benchmarks/psMNIST_lstm/` — 25 UNSAT instances, `prop_000–prop_024` on `lstm_psMNIST_h8.onnx`
**Task**: confirm or refute the 25 UNSAT labels using a sound verifier independent of n2v's IBP

---

## ⚠️ CONTRADICTED INSTANCES

**None.** No UNSAT label was refuted. See Section 4 for full results.

---

## 1. Tool and Environment

| Item | Value |
|---|---|
| Tool | auto_LiRPA |
| Version | 0.7.2 |
| Method | `CROWN` (linear relaxation — sound, not just IBP) |
| PyTorch | 2.12.0 |
| onnx2torch | 1.5.15 |
| Python | 3.11.15 |
| OS / Hardware | macOS 15.2, Apple Silicon arm64 (CPU only) |
| Conda env | `abcrown` |
| Date run | 2026-06-23 |
| Script | `validate_crown_random392.py` |

### Why CROWN is a sound, independent check

CROWN computes linear relaxation bounds (triangle enclosures for Sigmoid/Tanh, McCormick
envelopes for bilinear Mul). CROWN is sound: a positive margin certifies the property
with no counterexample in the L∞ ball. It is strictly stronger than IBP (n2v's method),
so a positive CROWN margin confirms the IBP certification is not a systematic false pass.

---

## 2. Results — 25-instance table

| Instance | true\_cls | CROWN verdict | Bucket | Min margin | Runtime |
|---|---|---|---|---|---|
| prop_000 | 0 | CERTIFIED | **CONFIRMED** | +6.8576 | 1.79 s |
| prop_001 | 1 | CERTIFIED | **CONFIRMED** | +6.3567 | 2.05 s |
| prop_002 | 2 | CERTIFIED | **CONFIRMED** | +4.3168 | 1.87 s |
| prop_003 | 3 | CERTIFIED | **CONFIRMED** | +5.0012 | 2.02 s |
| prop_004 | 4 | CERTIFIED | **CONFIRMED** | +3.5558 | 1.93 s |
| prop_005 | 5 | CERTIFIED | **CONFIRMED** | +4.3857 | 1.94 s |
| prop_006 | 6 | CERTIFIED | **CONFIRMED** | +5.1238 | 1.84 s |
| prop_007 | 7 | CERTIFIED | **CONFIRMED** | +5.5616 | 2.00 s |
| prop_008 | 8 | CERTIFIED | **CONFIRMED** | +2.8824 | 1.96 s |
| prop_009 | 9 | CERTIFIED | **CONFIRMED** | +3.1062 | 1.85 s |
| prop_010 | 0 | CERTIFIED | **CONFIRMED** | +6.9555 | 2.00 s |
| prop_011 | 0 | CERTIFIED | **CONFIRMED** | +6.8011 | 1.97 s |
| prop_012 | 1 | CERTIFIED | **CONFIRMED** | +6.3502 | 1.95 s |
| prop_013 | 1 | CERTIFIED | **CONFIRMED** | +6.1866 | 1.85 s |
| prop_014 | 7 | CERTIFIED | **CONFIRMED** | +5.3813 | 2.06 s |
| prop_015 | 7 | CERTIFIED | **CONFIRMED** | +5.7638 | 1.94 s |
| prop_016 | 6 | CERTIFIED | **CONFIRMED** | +5.2800 | 1.96 s |
| prop_017 | 6 | CERTIFIED | **CONFIRMED** | +5.3103 | 1.86 s |
| prop_018 | 3 | CERTIFIED | **CONFIRMED** | +4.3658 | 2.03 s |
| prop_019 | 3 | CERTIFIED | **CONFIRMED** | +4.4587 | 1.96 s |
| prop_020 | 5 | CERTIFIED | **CONFIRMED** | +4.1911 | 1.96 s |
| prop_021 | 5 | CERTIFIED | **CONFIRMED** | +4.0714 | 1.88 s |
| prop_022 | 2 | CERTIFIED | **CONFIRMED** | +4.1250 | 2.03 s |
| prop_023 | 2 | CERTIFIED | **CONFIRMED** | +3.6612 | 1.96 s |
| prop_024 | 4 | CERTIFIED | **CONFIRMED** | +3.6562 | 1.96 s |

---

## 3. Summary

| Bucket | Count |
|---|---|
| **CONFIRMED** | **25 / 25** |
| INCONCLUSIVE | 0 / 25 |
| CONTRADICTED | 0 / 25 |

| Statistic | Value |
|---|---|
| Min margin | +2.8824 (prop_008, cls=8) |
| Median margin | +5.0012 (prop_003, cls=3) |
| Max margin | +6.9555 (prop_010, cls=0) |
| Total wall time | ~57 s (~2.0 s/instance + ~7 s model init) |

The minimum margin (+2.882 at prop_008, cls=8) is lower than in the 7-class composition
(+4.317) because class 8 has lower IBP margins under this permutation. It is still strongly
positive — no instance is borderline.

---

## 4. Verdict

**The 25 UNSAT labels are independently validated.**

auto_LiRPA CROWN v0.7.2 certifies all 25 instances at ε = 1/255 ≈ 0.003922 on
`lstm_psMNIST_h8.onnx`. The minimum certification margin is +2.882 logit units (prop_008,
cls=8). All 10 digit classes are covered in the UNSAT half.

See `VALIDATION_ABCROWN.md` for the full alpha-beta-CROWN harness result (50/50 match).
See `VERIFY_v2.md` for SAT concrete witness confirmation.
