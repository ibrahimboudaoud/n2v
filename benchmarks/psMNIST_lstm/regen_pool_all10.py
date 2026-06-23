#!/usr/bin/env python3
"""
Regenerate the 25-instance pool for experiment-all10 (all 10 digit classes).

Uses pre-trained model checkpoints from models/ — does NOT retrain or
re-export ONNX.  Models stay exactly as in random392-7class (opset 14).

Pool algorithm (two-phase, all-10 guaranteed):
  Phase A: collect up to MAX_PER_CLASS=3 certified+SAT instances per class.
           Stop when all 10 classes have their full quota.
  Phase B: mandatory 1 from each class (highest h8 logit margin), then
           fill the remaining 15 slots by descending margin, cap 3/class.

Outputs (overwrite existing):
  vnnlib/prop_000–024.vnnlib  (25 UNSAT, h8, eps=1/255)
  vnnlib/prop_025–049.vnnlib  (25 SAT,   h64, eps=6/255)
  instances.csv
  ground_truth.csv

Does NOT touch:
  onnx/lstm_psMNIST_h8.onnx
  onnx/lstm_psMNIST_h64.onnx
  models/*.pt (read-only)

Run from: benchmarks/psMNIST_lstm/
"""

import sys, os, csv, warnings, time
import numpy as np
import torch
warnings.filterwarnings("ignore")

# ── Import constants and helpers from generate.py (no retraining) ────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate import (
    SEED, SEQ_LEN, INPUT_SIZE, INPUT_DIM, NUM_CLASSES,
    HIDDEN_SMALL, HIDDEN_LARGE, TIMEOUT,
    EPS_UNSAT, EPS_SAT,
    PERMUTATION,
    psMNIST_LSTM,
    load_mnist,
    ibp_certify, find_sat_instance,
    write_vnnlib,
    run_n2v_dryrun,
)

MAX_PER_CLASS = 3
N_POOL        = 25


def build_pool(model_small, model_large, X_te, y_te):
    """
    Two-phase all-10 pool selection.

    Returns a list of N_POOL (test_idx, x, x_test, eps_sat, true_cls) tuples,
    ordered so that: first 10 = 1 per class (highest margin), remaining 15 =
    descending margin, all subject to MAX_PER_CLASS=3 cap.

    Aborts via SystemExit if any class has zero certifiable+SAT instances.
    If any witness distance d > EPS_SAT, reports the instance and stops.
    """
    with torch.no_grad():
        Xt       = torch.tensor(X_te, dtype=torch.float32)
        logits_s = model_small(Xt).numpy()
        logits_l = model_large(Xt).numpy()

    preds_s = logits_s.argmax(1)
    preds_l = logits_l.argmax(1)
    acc_s = (preds_s == y_te).mean()
    acc_l = (preds_l == y_te).mean()
    print(f"  Loaded h8 acc={acc_s:.3f}  h64 acc={acc_l:.3f}")

    # Sort dual-correct candidates by h8 logit margin descending
    margin_s     = logits_s.max(1) - np.partition(logits_s, -2, axis=1)[:, -2]
    both_correct = (preds_s == y_te) & (preds_l == y_te)
    candidates   = np.where(both_correct)[0]
    candidates   = candidates[np.argsort(margin_s[candidates])[::-1]]
    correct_l    = preds_l == y_te

    per_class = {cls: [] for cls in range(NUM_CLASSES)}
    bad_eps   = []   # instances where d_witness > EPS_SAT

    print(f"\nPhase A — collecting up to {MAX_PER_CLASS} certified+SAT pairs per class …")
    t0 = time.time()

    for idx in candidates:
        if all(len(v) >= MAX_PER_CLASS for v in per_class.values()):
            break
        x  = X_te[idx]
        tc = int(y_te[idx])
        if len(per_class[tc]) >= MAX_PER_CLASS:
            continue

        if not ibp_certify(model_small, x, EPS_UNSAT, tc):
            continue

        x1_pool = X_te[np.where((y_te != tc) & correct_l)[0]]
        result  = find_sat_instance(model_large, x, tc, x1_pool)
        if result is None:
            continue

        x_test, eps_raw, true_cls = result
        d_witness = eps_raw / 1.1

        if d_witness > EPS_SAT:
            bad_eps.append((tc, int(idx), d_witness, eps_raw))

        eps_sat = EPS_SAT if d_witness <= EPS_SAT else eps_raw
        per_class[tc].append((float(margin_s[idx]), int(idx), x, x_test, eps_sat, tc))
        total = sum(len(v) for v in per_class.values())
        print(f"  cls={tc}  idx={idx:5d}  margin={margin_s[idx]:.3f}"
              f"  d={d_witness:.4f}  eps_sat={eps_sat:.6f}"
              f"  [{len(per_class[tc])}/{MAX_PER_CLASS}]  total={total}")

    print(f"  Phase A done in {time.time()-t0:.0f}s")

    # Report any witness distances > EPS_SAT and STOP
    if bad_eps:
        print(f"\nSTOP — {len(bad_eps)} instance(s) have d_witness > EPS_SAT={EPS_SAT:.6f} (6/255):")
        for tc, idx, d, eps_raw in bad_eps:
            # Find smallest clean k/255 that covers this witness
            k = int(np.ceil(d * 255)) + 1
            print(f"  cls={tc}  idx={idx}  d_witness={d:.6f}  eps_raw={eps_raw:.6f}"
                  f"  → smallest clean eps = {k}/255 = {k/255:.6f}")
        all_d = [e[2] for e in bad_eps]
        k_all = int(np.ceil(max(all_d) * 255)) + 1
        print(f"  Smallest clean k/255 covering ALL bad instances: {k_all}/255 = {k_all/255:.6f}")
        raise SystemExit("Halted — review SAT epsilon before proceeding.")

    # Check all 10 classes have at least 1 instance
    missing = [cls for cls in range(NUM_CLASSES) if len(per_class[cls]) == 0]
    if missing:
        for cls in missing:
            print(f"\nSTOP: class {cls} has zero certifiable+SAT instances.")
        raise SystemExit("Cannot build all-10 pool — see missing classes above.")

    # Phase B — build ordered pool
    print(f"\nPhase B — building ordered pool (1 mandatory per class, then fill by margin) …")
    pairs        = []
    class_counts = {cls: 0 for cls in range(NUM_CLASSES)}

    # Mandatory: 1 from each class (highest margin = index 0)
    for cls in range(NUM_CLASSES):
        _, idx, x, x_test, eps_sat, tc = per_class[cls][0]
        pairs.append((idx, x, x_test, eps_sat, tc))
        class_counts[cls] = 1
        print(f"  mandatory cls={cls}  idx={idx:5d}  eps_sat={eps_sat:.6f}")

    # Fill remaining 15 by descending margin
    remaining = []
    for cls in range(NUM_CLASSES):
        for entry in per_class[cls][1:]:
            remaining.append(entry)
    remaining.sort(key=lambda e: e[0], reverse=True)

    for entry in remaining:
        if len(pairs) >= N_POOL:
            break
        margin, idx, x, x_test, eps_sat, tc = entry
        if class_counts[tc] >= MAX_PER_CLASS:
            continue
        pairs.append((idx, x, x_test, eps_sat, tc))
        class_counts[tc] += 1
        print(f"  fill     cls={tc}  idx={idx:5d}  margin={margin:.3f}  eps_sat={eps_sat:.6f}")

    print(f"\nFinal pool ({len(pairs)} pairs):")
    for i, (idx, x, x_test, eps_sat, tc) in enumerate(pairs):
        print(f"  [{i:2d}] test_idx={idx:5d}  cls={tc}  eps_sat={eps_sat:.6f}")
    print(f"  Class distribution: {dict(sorted(class_counts.items()))}")

    if len(pairs) < N_POOL:
        raise SystemExit(f"Only {len(pairs)}/{N_POOL} pairs — cannot proceed.")

    return pairs


def write_artifacts(pairs):
    """Write all 50 VNN-LIB files, instances.csv, ground_truth.csv."""
    os.makedirs("vnnlib", exist_ok=True)
    onnx_small = f"lstm_psMNIST_h{HIDDEN_SMALL}.onnx"
    onnx_large = f"lstm_psMNIST_h{HIDDEN_LARGE}.onnx"
    records       = []
    unsat_indices = []
    sat_indices   = []
    prop_idx      = 0

    print(f"\nWriting UNSAT properties (h8, eps={EPS_UNSAT:.8f} = 1/255) …")
    for (test_idx, x, x_test, eps_sat, tc) in pairs:
        pname = f"prop_{prop_idx:03d}.vnnlib"
        write_vnnlib(f"vnnlib/{pname}", x, EPS_UNSAT, tc)
        records.append((f"onnx/{onnx_small}", f"vnnlib/{pname}", "unsat", TIMEOUT))
        unsat_indices.append(test_idx)
        prop_idx += 1

    print(f"Writing SAT properties (h64, eps={EPS_SAT:.8f} = 6/255) …")
    for (test_idx, x, x_test, eps_sat, tc) in pairs:
        pname = f"prop_{prop_idx:03d}.vnnlib"
        write_vnnlib(f"vnnlib/{pname}", x_test, eps_sat, tc)
        records.append((f"onnx/{onnx_large}", f"vnnlib/{pname}", "sat", TIMEOUT))
        sat_indices.append(test_idx)
        prop_idx += 1

    assert unsat_indices == sat_indices, "BUG: UNSAT and SAT pools differ"

    with open("instances.csv", "w", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(
            [(r[0], r[1], r[3]) for r in records]
        )
    with open("ground_truth.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["onnx", "vnnlib", "result", "timeout"])
        w.writerows(records)

    print(f"\nArtifacts written: {len(records)} instances, instances.csv, ground_truth.csv")
    return records


def main():
    t_start = time.time()
    print("=" * 60)
    print(f"regen_pool_all10  —  MAX_PER_CLASS={MAX_PER_CLASS}, N_POOL={N_POOL}")
    print(f"  ONNX files: unchanged (opset 14, random-392 weights)")
    print(f"  EPS_UNSAT = 1/255 = {EPS_UNSAT:.8f}")
    print(f"  EPS_SAT   = 6/255 = {EPS_SAT:.8f}")
    print("=" * 60)

    print("\nLoading MNIST …")
    X_tr, y_tr, X_te, y_te = load_mnist()
    print(f"  test set: {X_te.shape}")

    print("\nLoading pre-trained models from checkpoints …")
    model_small = psMNIST_LSTM(HIDDEN_SMALL)
    ckpt_s = f"models/lstm_psMNIST_h{HIDDEN_SMALL}_random392.pt"
    model_small.load_state_dict(torch.load(ckpt_s, map_location="cpu"))
    model_small.eval()

    model_large = psMNIST_LSTM(HIDDEN_LARGE)
    ckpt_l = f"models/lstm_psMNIST_h{HIDDEN_LARGE}_random392.pt"
    model_large.load_state_dict(torch.load(ckpt_l, map_location="cpu"))
    model_large.eval()
    print(f"  h8  ← {ckpt_s}")
    print(f"  h64 ← {ckpt_l}")

    pairs   = build_pool(model_small, model_large, X_te, y_te)
    records = write_artifacts(pairs)

    print("\n── n2v dry run ──────────────────────────────────────────")
    run_n2v_dryrun(records)

    print(f"\nTotal wall time: {time.time()-t_start:.0f}s")
    print("DONE. Review pool above, then run:")
    print("  conda run -n abcrown python validate_crown_random392.py")
    print("  conda run -n abcrown python abcrown.py --config /tmp/psmnist_abcrown_all10.yaml")


if __name__ == "__main__":
    main()
