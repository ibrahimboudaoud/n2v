#!/usr/bin/env python3
"""
Independent CROWN validation of the 25 UNSAT instances (random-392 benchmark).
Requires: conda env `abcrown` with auto_LiRPA 0.7.2 and onnx2torch 1.5.15.
Run from: benchmarks/psMNIST_lstm/
"""
import sys, time, warnings, numpy as np
warnings.filterwarnings("ignore")

try:
    import onnx2torch
    import torch
    from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
except ImportError as e:
    print(f"Import error: {e}")
    print("Run via: conda run -n abcrown python validate_crown_random392.py")
    sys.exit(1)

N_UNSAT = 25

def parse_vnnlib(path):
    lbs, ubs, true_cls = [], [], None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("; true_class="):
                true_cls = int(line.split("true_class=")[1].split()[0])
            elif line.startswith("(assert (>="):
                ubs.append(float(line.split()[-1].rstrip(")")))
            elif line.startswith("(assert (<="):
                v = line.split()[-1].rstrip(")")
                # first N_INPUT lower bounds, then N_INPUT upper bounds interleaved
                lbs.append(float(v))
    return lbs, ubs, true_cls

def parse_vnnlib_v2(path):
    """Parse box bounds. (assert (>= X_i val)) → lower bound; (<= X_i val)) → upper bound."""
    lbs, ubs = [], []
    true_cls = None
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("; true_class="):
                true_cls = int(s.split("true_class=")[1].split()[0])
            elif s.startswith("(assert (>= X_"):
                lbs.append(float(s.split()[-1].rstrip(")")))
            elif s.startswith("(assert (<= X_"):
                ubs.append(float(s.split()[-1].rstrip(")")))
    return np.array(lbs, dtype=np.float32), np.array(ubs, dtype=np.float32), true_cls

def main():
    t0 = time.time()
    print("=" * 60)
    print("CROWN VALIDATION — random-392 benchmark, 25 UNSAT instances")
    print("=" * 60)

    import onnx2torch, onnx
    raw = onnx.load("onnx/lstm_psMNIST_h8.onnx")
    model = onnx2torch.convert(raw)
    model.eval()

    dummy = torch.zeros(1, 392)
    bounded = BoundedModule(model, dummy, device="cpu")
    print(f"  h8 ONNX loaded and wrapped in BoundedModule ✓")

    results = []
    for k in range(N_UNSAT):
        t_inst = time.time()
        path = f"vnnlib/prop_{k:03d}.vnnlib"
        lb_np, ub_np, true_cls = parse_vnnlib_v2(path)

        x_lb  = torch.tensor(lb_np).unsqueeze(0)
        x_ub  = torch.tensor(ub_np).unsqueeze(0)
        x_ctr = (x_lb + x_ub) / 2.0

        ptb    = PerturbationLpNorm(norm=np.inf, x_L=x_lb, x_U=x_ub)
        bt     = BoundedTensor(x_ctr, ptb)
        lb_out, ub_out = bounded.compute_bounds(x=(bt,), method="CROWN")

        # margin: true-class lb minus max wrong-class ub
        ub_wrong = ub_out.clone()
        ub_wrong[0, true_cls] = -1e9
        margin = float((lb_out[0, true_cls] - ub_wrong[0].max()).item())
        certified = margin > 0
        inst_t = time.time() - t_inst

        results.append((k, true_cls, certified, margin, inst_t))
        status = "CERTIFIED" if certified else "INCONCLUSIVE"
        print(f"  prop_{k:03d}  cls={true_cls}  {status}  margin={margin:+.4f}  {inst_t:.2f}s")

    margins    = [r[3] for r in results]
    runtimes   = [r[4] for r in results]
    n_cert     = sum(1 for r in results if r[2])
    min_margin = min(margins)
    max_margin = max(margins)
    sorted_m   = sorted(margins)
    med_margin = sorted_m[len(sorted_m)//2]

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Certified     : {n_cert} / {N_UNSAT}")
    print(f"  Min margin    : {min_margin:+.4f}  (prop_{min(range(N_UNSAT), key=lambda i: margins[i]):03d})")
    print(f"  Median margin : {med_margin:+.4f}")
    print(f"  Max margin    : {max_margin:+.4f}  (prop_{max(range(N_UNSAT), key=lambda i: margins[i]):03d})")
    print(f"  Wall time     : {time.time()-t0:.1f}s")

    if n_cert == N_UNSAT:
        print(f"\n  VERDICT: ALL 25 UNSAT labels CONFIRMED by independent CROWN. ✓")
    else:
        failed = [r[0] for r in results if not r[2]]
        print(f"\n  VERDICT: {N_UNSAT - n_cert} instances INCONCLUSIVE: {failed}")
        print(f"  (CROWN is incomplete — inconclusive ≠ wrong, but investigate)")

if __name__ == "__main__":
    main()
