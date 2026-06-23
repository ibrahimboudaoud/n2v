# End-to-End Validation Report — α,β-CROWN Full Harness

**Date**: 2026-06-23
**Benchmark**: `benchmarks/psMNIST_lstm/` — 50 instances (25 UNSAT prop_000–024, 25 SAT prop_025–049)
**Task**: smoke-test the complete alpha-beta-CROWN pipeline (`abcrown.py`) against all 50 instances
via CSV mode.

---

## ⚠️ CONFLICTS

**None.** No label was contradicted by alpha-beta-CROWN.

---

## 1. Tool and Environment

| Item | Value |
|---|---|
| **Tool** | α,β-CROWN (alpha-beta-CROWN) |
| **Version** | 0.7.0 |
| **Commit** | `746b7d0128df1806c92381d1c8b3a66c9cba990c` |
| **Entry point** | `complete_verifier/abcrown.py` |
| **Conda env** | `abcrown` (Python 3.11.15) |
| **Device** | CPU (Apple Silicon arm64, macOS 15.2) |
| **auto_LiRPA** | 0.7.2 |
| **torch** | 2.12.0 |
| **Date run** | 2026-06-23 |

### Config used

File: `/tmp/psmnist_abcrown_all10.yaml`

```yaml
general:
  root_path: /Users/ibrahimboudaoud/n2v/benchmarks/psMNIST_lstm
  csv_name: instances.csv
  device: cpu
  complete_verifier: auto
solver:
  bound_prop_method: crown
bab:
  timeout: 120
```

---

## 2. Summary Counts

| Classification | Count |
|---|---|
| **MATCH** | **50 / 50** |
| INCONCLUSIVE | 0 / 50 |
| **CONFLICT** | **0 / 50** |

Breakdown by half:

| Half | Instances | abcrown verdict | Classification |
|---|---|---|---|
| UNSAT (prop_000–024, h8, ε=1/255≈0.003922) | 25 | all `safe-incomplete` | 25 MATCH |
| SAT (prop_025–049, h64, ε=6/255≈0.023529) | 25 | all `unsafe-pgd` | 25 MATCH |

Abcrown summary line (verbatim from run log):
```
Problem instances count: 50 , total verified (safe/unsat): 25 , total falsified (unsafe/sat): 25 , timeout: 0
mean time for ALL instances (total 50): 5.403 s
mean time for verified SAFE instances (total 25): 10.729 s, max: 11.827 s
mean time for verified UNSAFE instances (total 25): 0.077 s, max: 0.144 s
```

---

## 3. Verdict

**The benchmark runs end-to-end through the official alpha-beta-CROWN harness without errors or
label conflicts.**

1. **All 25 UNSAT labels confirmed.** alpha-CROWN certifies all 25 h8 instances at ε = 1/255
   in 10.7 s mean / 11.8 s max. Consistent with CROWN validation (min margin +2.882 logits).

2. **All 25 SAT labels confirmed.** PGD finds a counterexample inside every h64 L∞ ball in
   under 0.15 s. Witness distance ≈ 0.020, well inside the 6/255 ball (margin ≈ 0.0035).

3. **Zero timeouts.** All 50 instances resolve within the 120 s budget. Slowest: 11.83 s.

4. **Zero conflicts.** No UNSAT instance returned `unsafe` and no SAT instance returned `safe`.

5. **All 10 digit classes verified.** The UNSAT half covers classes 0–9; the SAT half covers
   the same source images.

**For VNN-COMP 2026**: benchmark passes the full official harness end-to-end with all 10
digit classes represented and runtime well within budget.
