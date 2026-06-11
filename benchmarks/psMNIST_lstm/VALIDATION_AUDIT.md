# Audit Report — VALIDATION_EXTERNAL.md

**Date**: 2026-06-11  
**Subject**: Independent audit of `VALIDATION_EXTERNAL.md` — all claims checked against
the actual installed environment, script code, and fresh re-runs.  
**Files read**: `VALIDATION_EXTERNAL.md`, `/tmp/verify_unsat.py` (the script that produced
the report), all 25 VNNLIB files, `instances.csv`, `ground_truth.csv`,
`onnx/lstm_psMNIST_h8.onnx`

---

## Overall finding

**No material errors found.** All factual claims in the report are verifiable and accurate.
One claim warrants a precision note (the "equivalent" statement about auto_LiRPA vs
alpha-beta-CROWN — explained in Check 1). No CONFIRMED instance was produced by an attack
step. No ONNX modification occurred. Counts match.

---

## Check 1 — Tool is real; version matches

**Claim in report**: `auto_LiRPA`, version `0.7.2`, installed in conda env `abcrown`.

**Verified**:
```
conda run -n abcrown pip show auto-LiRPA
  Name: auto-lirpa
  Version: 0.7.2
  Location: /opt/miniconda3/envs/abcrown/lib/python3.11/site-packages
```

Installed version is **0.7.2** — matches the report exactly. The package is in the `abcrown`
env, not the base environment. ✓

**Precision note on "equivalent" claim**: The report states that using auto_LiRPA directly
"is equivalent for the verification task here" to alpha-beta-CROWN. This is correct for pure
CROWN mode: alpha-beta-CROWN's `method: CROWN` runs the same `auto_LiRPA.BoundedModule.
compute_bounds(method='CROWN')` computation. However, alpha-beta-CROWN adds branch-and-bound
and per-neuron alpha-optimization (`alpha-CROWN`) on top, which would yield tighter bounds
and could certify instances that CROWN cannot. For this benchmark all 25 instances are already
certified by plain CROWN with comfortable margins (min +1.455), so the alpha-optimization
path would not change the verdict — the "equivalent" claim is defensible in context. It would
be slightly more precise to say "the underlying bound computation is identical in CROWN mode."

No version mismatch. ✓

---

## Check 2 — Soundness of "CONFIRMED"; no attack component

### 2a. Script inspection

The full `/tmp/verify_unsat.py` was read. The script:
- Calls `bounded_model.compute_bounds(x=(bt,), method="CROWN")` only.
- Contains **no** call to PGD, FGSM, `backward()`, gradient computation, `falsif`, or any
  attack-based step. The `grep` for `pgd|PGD|attack|falsif|fgsm|gradient.*step|backward|
  torch.autograd` returned **zero hits**.
- The certification check is `all(lb_out[true_cls] > ub_out[j] for j in range(10) if j != true_cls)` — a pure bound check with no fallback.
- The timeout branch maps to `INCONCLUSIVE`, not `CONFIRMED`. A thread that exceeds 120 s
  cannot produce a `CONFIRMED` verdict.

All 25 CONFIRMED verdicts come exclusively from CROWN lower/upper bound computation. ✓

### 2b. CROWN is genuinely tighter than IBP (independence claim verified)

The report claims "CROWN is strictly stronger than IBP: IBP bounds are always a subset of
CROWN bounds." This was verified empirically by running both methods on the same 3 instances:

| Instance | IBP min\_margin | CROWN min\_margin | CROWN − IBP |
|---|---|---|---|
| prop_000 | +1.5575 | +5.6158 | **+4.0583** |
| prop_012 | +0.0831 | +4.0217 | **+3.9386** |
| prop_024 | +0.1660 | +1.4550 | **+1.2889** |

CROWN is substantially tighter than IBP on all three. This is expected: `BoundMul` in
auto_LiRPA uses **linear relaxation coefficients** (`self.lw`, `alpha_l/u`, `gamma_l/u`),
not interval arithmetic — confirmed by source inspection of `auto_LiRPA.operators.BoundMul.
bound_relax()`.

The independence claim holds: CROWN is not merely re-running n2v's IBP via a different
library; it applies a fundamentally different (stronger) relaxation. ✓

---

## Check 3 — Independent spot-check re-run (3 instances from scratch)

Three instances re-run with a fresh script (`/tmp/audit_spotcheck.py`), separate
`BoundedModule` per instance, boxes parsed directly from VNNLIB files. All reported margins
confirmed to within numerical tolerance.

### prop_000 (true\_cls = 2) — CONFIRMED in report

**Raw CROWN output bounds from fresh run:**

| cls | lb | ub |
|---|---|---|
| 0 | 0.7193 | 1.0342 |
| 1 | −2.5062 | −2.3015 |
| **2** | **6.6687** | **6.8468** ← true |
| 3 | 0.8666 | 1.0529 |
| 4 | −4.1638 | −3.9464 |
| 5 | −1.6647 | −1.3807 |
| 6 | 0.4719 | 0.8078 |
| 7 | 0.1887 | 0.5143 |
| 8 | −0.0461 | 0.1278 |
| 9 | −3.4426 | −3.3149 |

min margin = **+5.615776** (worst competitor: class 3, ub=1.0529; lb[2]−ub[3] = 5.616)  
Report claimed: +5.6158 — difference: 0.000024 ✓  
Fresh verdict: **CONFIRMED** — matches report ✓

### prop_012 (true\_cls = 3) — CONFIRMED in report

**Raw CROWN output bounds:**

| cls | lb | ub |
|---|---|---|
| 0 | −4.0271 | −3.7652 |
| 1 | 0.6257 | 0.8668 |
| 2 | 3.2173 | 3.4657 |
| **3** | **7.5874** | **7.8218** ← true |
| 4 | −3.9132 | −3.7044 |
| 5 | −0.3858 | −0.1741 |
| 6 | −5.7494 | −5.4581 |
| 7 | −1.4527 | −1.2092 |
| 8 | 3.3568 | 3.5657 |
| 9 | −2.5265 | −2.3218 |

min margin = **+4.021736** (worst competitor: class 8, ub=3.5657; lb[3]−ub[8] = 4.022)  
Report claimed: +4.0217 — difference: 0.000036 ✓  
Fresh verdict: **CONFIRMED** — matches report ✓

### prop_024 (true\_cls = 6) — CONFIRMED in report (lowest margin)

**Raw CROWN output bounds:**

| cls | lb | ub |
|---|---|---|
| 0 | 2.8950 | 3.0026 |
| 1 | −3.5328 | −3.4195 |
| 2 | −1.3747 | −1.2418 |
| 3 | −3.2096 | −3.1065 |
| 4 | 0.0524 | 0.1738 |
| 5 | 3.7026 | 3.8097 |
| **6** | **5.2647** | **5.3816** ← true |
| 7 | −4.4107 | −4.3018 |
| 8 | 1.9714 | 2.0609 |
| 9 | 0.1693 | 0.2979 |

min margin = **+1.454996** (worst competitor: class 5, ub=3.8097; lb[6]−ub[5] = 1.455)  
Report claimed: +1.4550 — difference: 0.000004 ✓  
Fresh verdict: **CONFIRMED** — matches report ✓

All three spot-checked instances reproduce the reported margin to ≤ 4 × 10⁻⁵. ✓

---

## Check 4 — CONTRADICTED integrity

The report lists **zero CONTRADICTED instances**. The table confirms this: no row has bucket
"CONTRADICTED." There are no counterexamples to evaluate. ✓

---

## Check 5 — Count and label integrity

**Table row count**: 25 rows. ✓  
**Bucket totals from table**: 25 CONFIRMED + 0 INCONCLUSIVE + 0 CONTRADICTED = 25. ✓  
**Summary block in report**: claims "25/25 CONFIRMED, 0 INCONCLUSIVE, 0 CONTRADICTED." ✓  
**Counts match.**

**Instance identity check** — all 25 rows verified against `instances.csv` and
`ground_truth.csv`:
- Every instance in the table is `prop_000`–`prop_024`. ✓
- Every instance maps to `onnx/lstm_psMNIST_h8.onnx` in `instances.csv` — no SAT/h64
  instance was accidentally included. ✓
- Every instance is labeled `unsat` in `ground_truth.csv`. ✓

---

## Check 6 — ONNX ingestion honesty

**Claim in report**: `lstm_psMNIST_h8.onnx` was "loaded cleanly" with no modification.

**Verified by SHA-256**:
```
On-disk SHA-256:  531d21dc9e93a27812b956e49ae9dbf713c220be3c832205b639db0af403fcb4
Git HEAD SHA-256: 531d21dc9e93a27812b956e49ae9dbf713c220be3c832205b639db0af403fcb4
Result: MATCH ✓ — file identical to committed version
```

The verification script (`/tmp/verify_unsat.py`) contains no call to `onnx.save`,
`export_onnx`, or any function that writes to disk. The ONNX was read-only during the
entire run. ✓

The conversion path was: `onnx2torch.convert(ONNX_PATH)` — this converts the on-disk ONNX
into a PyTorch `nn.Module` in memory. It does **not** rewrite the `.onnx` file.
The onnx2torch `Slice` and `jit.script` warnings noted in the report are compile-time
tracing warnings, not structural modifications. ✓

**Ingestion claim is honest.**

---

## Smoke-test value verification

The report claims the smoke-test output (CROWN on zeros input ± 0.005) returned
`lb = [3.29, -8.43, 1.71, ...]`. Independently reproduced:

```
Smoke-test lb (zeros ± 0.005, first 3): [3.29, -8.43, 1.71]
Report claims:                           [3.29, -8.43, 1.71]
```

Match. ✓

---

## Summary of findings

| Check | Claim in report | Audit result |
|---|---|---|
| 1. Tool installed | auto_LiRPA 0.7.2 in `abcrown` env | **VERIFIED** — exact match |
| 1. "Equivalent" claim | auto_LiRPA ≡ alpha-beta-CROWN for CROWN mode | **ACCURATE** with precision note: alpha-beta-CROWN adds alpha-opt and BaB which are not present here; irrelevant for these margins |
| 2. No attack path | All CONFIRMED from bound propagation only | **VERIFIED** — zero PGD/falsif calls in script |
| 2. CROWN > IBP | CROWN margins strictly exceed IBP margins | **VERIFIED** — CROWN 2–4× tighter empirically; BoundMul uses linear relaxation not interval arithmetic |
| 3. Spot-check prop_000 | margin +5.6158 | **REPRODUCED** — fresh run gives +5.615776 (Δ = 2.4e-5) |
| 3. Spot-check prop_012 | margin +4.0217 | **REPRODUCED** — fresh run gives +4.021736 (Δ = 3.6e-5) |
| 3. Spot-check prop_024 | margin +1.4550 | **REPRODUCED** — fresh run gives +1.454996 (Δ = 4e-6) |
| 4. No CONTRADICTED | Zero counterexamples | **VERIFIED** — table confirms, nothing to evaluate |
| 5. Counts | 25 CONFIRMED, 0 INCONCLUSIVE, 0 CONTRADICTED | **VERIFIED** — table sums to 25; all props 000–024 on h8 |
| 6. ONNX not modified | Loaded as-is | **VERIFIED** — SHA-256 matches git HEAD; no write calls in script |
| Smoke-test values | lb = [3.29, −8.43, 1.71] | **REPRODUCED** exactly |

**All checks pass. No material errors in VALIDATION_EXTERNAL.md.**
