#!/usr/bin/env python3
"""
Random-392 probe — psMNIST LSTM, 14 timesteps × 28 features = 392 pixels
drawn from the FULL 0-783 pixel range (not just the top-half 0-391).

Experiment only.  Does NOT modify generate.py or any existing benchmark
artifacts (ONNX, VNNLIB, instances.csv, ground_truth.csv).

Comparison target (top-half, current benchmark):
  PERMUTATION = _PERM_RNG.permutation(392)      → indices drawn from [0, 391]
  Reported:  ~79.7% accuracy, ~74% IBP-certifiable, 6 viable classes

This probe:
  PERMUTATION = _PERM_RNG.choice(784, 392, replace=False) → indices from [0, 783]
  Everything else identical: same SEED, same IBP schedule, same ε = 1/255.

Run from: benchmarks/psMNIST_lstm/
"""
import time
import random
import numpy as np
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

# ── Constants — identical to generate.py except permutation source ────────────
SEED         = 42
SEQ_LEN      = 14          # unchanged: 14 timesteps × 28 features = 392 inputs
INPUT_SIZE   = 28
INPUT_DIM    = SEQ_LEN * INPUT_SIZE   # 392 — model shape unchanged
NUM_CLASSES  = 10
HIDDEN_SMALL = 8
EPOCHS       = 40
BATCH_SIZE   = 256
LR           = 1e-3
EPS_UNSAT    = 1 / 255

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

_PERM_RNG = np.random.default_rng(SEED)
# THE ONE CHANGE: draw 392 unique indices from the full 784-pixel range
PERMUTATION = _PERM_RNG.choice(784, size=392, replace=False)   # shape (392,)

print(f"  Permutation: 392 indices from [0, 783]  "
      f"(min={PERMUTATION.min()}, max={PERMUTATION.max()})")


# ── Data ──────────────────────────────────────────────────────────────────────

def load_mnist():
    from torchvision import datasets
    raw_train = datasets.MNIST("data", train=True,  download=True)
    raw_test  = datasets.MNIST("data", train=False, download=True)

    def _extract(d):
        imgs = d.data.numpy().astype(np.float32) / 255.0
        flat = imgs.reshape(len(imgs), -1)          # (N, 784)
        return flat[:, PERMUTATION], d.targets.numpy()   # (N, 392)

    X_tr, y_tr = _extract(raw_train)
    X_te, y_te = _extract(raw_test)
    return X_tr, y_tr, X_te, y_te


# ── Model — identical to generate.py ─────────────────────────────────────────

class _ManualLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.Wih = nn.Linear(input_size,  4 * hidden_size, bias=True)
        self.Whh = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

    def forward(self, x_t, h, c):
        gates = self.Wih(x_t) + self.Whh(h)
        H = self.hidden_size
        i = torch.sigmoid(gates[:, :H])
        f = torch.sigmoid(gates[:, H:2*H])
        g = torch.tanh(gates[:, 2*H:3*H])
        o = torch.sigmoid(gates[:, 3*H:])
        c_new = f * c + i * g
        return o * torch.tanh(c_new), c_new


class psMNIST_LSTM(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell        = _ManualLSTMCell(INPUT_SIZE, hidden_size)
        self.classifier  = nn.Linear(hidden_size, NUM_CLASSES)
        self.register_buffer("h0", torch.zeros(1, hidden_size))
        self.register_buffer("c0", torch.zeros(1, hidden_size))

    def forward(self, x_flat):
        h, c = self.h0, self.c0
        for t in range(SEQ_LEN):
            x_t = x_flat[:, t * INPUT_SIZE : (t + 1) * INPUT_SIZE]
            h, c = self.cell(x_t, h, c)
        return self.classifier(h)


# ── IBP helpers — identical to generate.py ───────────────────────────────────

def _affine_ibp(W, b, lb, ub):
    W_pos, W_neg = W.clamp(min=0), W.clamp(max=0)
    return (lb @ W_pos.T + ub @ W_neg.T + b,
            ub @ W_pos.T + lb @ W_neg.T + b)


def _mul_ibp(a_lb, a_ub, b_lb, b_ub):
    corners = torch.stack([a_lb*b_lb, a_lb*b_ub, a_ub*b_lb, a_ub*b_ub])
    return corners.min(0).values, corners.max(0).values


def ibp_certify_batch(model, X, y, eps, batch_size=512):
    """Vectorised IBP certification. Returns bool array (N,)."""
    H   = model.hidden_size
    Wih = model.cell.Wih.weight.data; bih = model.cell.Wih.bias.data
    Whh = model.cell.Whh.weight.data; z4H = torch.zeros(4 * H)
    Wc  = model.classifier.weight.data; bc = model.classifier.bias.data

    N = len(X)
    certified = np.zeros(N, dtype=bool)

    with torch.no_grad():
        for start in range(0, N, batch_size):
            xb = torch.tensor(X[start:start+batch_size], dtype=torch.float32)
            yb = torch.tensor(y[start:start+batch_size], dtype=torch.long)
            B  = xb.shape[0]

            lb = (xb - eps).clamp(0.0, 1.0)
            ub = (xb + eps).clamp(0.0, 1.0)

            h_lb = torch.zeros(B, H); h_ub = torch.zeros(B, H)
            c_lb = torch.zeros(B, H); c_ub = torch.zeros(B, H)

            for t in range(SEQ_LEN):
                xt_lb = lb[:, t*INPUT_SIZE:(t+1)*INPUT_SIZE]
                xt_ub = ub[:, t*INPUT_SIZE:(t+1)*INPUT_SIZE]
                gi_lb, gi_ub = _affine_ibp(Wih, bih, xt_lb, xt_ub)
                gh_lb, gh_ub = _affine_ibp(Whh, z4H, h_lb,  h_ub)
                g_lb = gi_lb + gh_lb; g_ub = gi_ub + gh_ub

                i_lb = torch.sigmoid(g_lb[:, :H]);     i_ub = torch.sigmoid(g_ub[:, :H])
                f_lb = torch.sigmoid(g_lb[:, H:2*H]);  f_ub = torch.sigmoid(g_ub[:, H:2*H])
                g2_lb = torch.tanh(g_lb[:, 2*H:3*H]);  g2_ub = torch.tanh(g_ub[:, 2*H:3*H])
                o_lb = torch.sigmoid(g_lb[:, 3*H:]);   o_ub = torch.sigmoid(g_ub[:, 3*H:])

                fc_lb, fc_ub = _mul_ibp(f_lb, f_ub, c_lb, c_ub)
                ig_lb, ig_ub = _mul_ibp(i_lb, i_ub, g2_lb, g2_ub)
                c_lb = fc_lb + ig_lb; c_ub = fc_ub + ig_ub

                tc_lb = torch.tanh(c_lb); tc_ub = torch.tanh(c_ub)
                h_lb, h_ub = _mul_ibp(o_lb, o_ub, tc_lb, tc_ub)

            logit_lb, logit_ub = _affine_ibp(Wc, bc, h_lb, h_ub)

            true_lb   = logit_lb[torch.arange(B), yb]
            ub_masked = logit_ub.clone()
            ub_masked[torch.arange(B), yb] = -1e9
            certified[start:start+B] = (true_lb > ub_masked.max(dim=1).values).numpy()

    return certified


# ── IBP-regularised training — identical lambda ramp to generate.py ───────────

def train_ibp(model, X, y, eps, warm=10):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ce  = nn.CrossEntropyLoss()
    Xt  = torch.tensor(X, dtype=torch.float32)
    yt  = torch.tensor(y, dtype=torch.long)
    n   = len(Xt)
    model.train()

    for ep in range(1, EPOCHS + 1):
        lam  = 0.0 if ep <= warm else (ep - warm) / (EPOCHS - warm)
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            xb, yb = Xt[idx], yt[idx]
            opt.zero_grad()
            clean_loss = ce(model(xb), yb)
            if lam > 0:
                lb_b = (xb - eps).clamp(0.0, 1.0)
                ub_b = (xb + eps).clamp(0.0, 1.0)
                l_lb, l_ub = _ibp_fwd_train(model, lb_b, ub_b)
                worst = l_ub.clone()
                worst[torch.arange(len(yb)), yb] = l_lb[torch.arange(len(yb)), yb]
                loss = clean_loss + lam * ce(worst, yb)
            else:
                loss = clean_loss
            loss.backward(); opt.step()

        if ep % 5 == 0 or ep == EPOCHS:
            with torch.no_grad():
                tr_acc = (model(Xt).argmax(1) == yt).float().mean().item()
            print(f"  ep {ep:3d}  lam={lam:.2f}  train_acc={tr_acc:.3f}", flush=True)

    model.eval()


def _ibp_fwd_train(model, lb, ub):
    H = model.hidden_size; B = lb.shape[0]
    Wih = model.cell.Wih.weight; bih = model.cell.Wih.bias
    Whh = model.cell.Whh.weight; z4H = torch.zeros(4 * H)
    Wc  = model.classifier.weight; bc = model.classifier.bias
    h_lb = torch.zeros(B, H); h_ub = torch.zeros(B, H)
    c_lb = torch.zeros(B, H); c_ub = torch.zeros(B, H)

    for t in range(SEQ_LEN):
        xt_lb = lb[:, t*INPUT_SIZE:(t+1)*INPUT_SIZE]
        xt_ub = ub[:, t*INPUT_SIZE:(t+1)*INPUT_SIZE]
        gi_lb, gi_ub = _affine_ibp(Wih, bih, xt_lb, xt_ub)
        gh_lb, gh_ub = _affine_ibp(Whh, z4H, h_lb,  h_ub)
        g_lb = gi_lb + gh_lb; g_ub = gi_ub + gh_ub

        i_lb = torch.sigmoid(g_lb[:, :H]);     i_ub = torch.sigmoid(g_ub[:, :H])
        f_lb = torch.sigmoid(g_lb[:, H:2*H]);  f_ub = torch.sigmoid(g_ub[:, H:2*H])
        g2_lb = torch.tanh(g_lb[:, 2*H:3*H]);  g2_ub = torch.tanh(g_ub[:, 2*H:3*H])
        o_lb = torch.sigmoid(g_lb[:, 3*H:]);   o_ub = torch.sigmoid(g_ub[:, 3*H:])

        fc_lb, fc_ub = _mul_ibp(f_lb, f_ub, c_lb, c_ub)
        ig_lb, ig_ub = _mul_ibp(i_lb, i_ub, g2_lb, g2_ub)
        c_lb = fc_lb + ig_lb; c_ub = fc_ub + ig_ub
        tc_lb = torch.tanh(c_lb); tc_ub = torch.tanh(c_ub)
        h_lb, h_ub = _mul_ibp(o_lb, o_ub, tc_lb, tc_ub)

    return _affine_ibp(Wc, bc, h_lb, h_ub)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 62)
    print(f"RANDOM-392 PROBE — 14 timesteps × 28 features = {INPUT_DIM} pixels")
    print(f"Pixels drawn from FULL 0-783 range (vs. top-half 0-391)")
    print(f"SEED={SEED}  EPS_UNSAT=1/255={EPS_UNSAT:.6f}  H={HIDDEN_SMALL}")
    print("=" * 62)

    print("\nLoading MNIST (/255, random-392 permutation from [0, 783])…")
    X_tr, y_tr, X_te, y_te = load_mnist()
    print(f"  train {X_tr.shape}  test {X_te.shape}")

    print(f"\nTraining h8 (same IBP schedule: warm=10, λ→1 over epochs 11–{EPOCHS})…")
    t_train = time.time()
    model = psMNIST_LSTM(HIDDEN_SMALL)
    train_ibp(model, X_tr, y_tr, EPS_UNSAT)
    print(f"  Training time: {time.time()-t_train:.0f}s")

    # ── Accuracy ──────────────────────────────────────────────────────────────
    with torch.no_grad():
        Xt    = torch.tensor(X_te, dtype=torch.float32)
        preds = model(Xt).argmax(1).numpy()
    correct   = preds == y_te
    n_correct = int(correct.sum())
    acc       = correct.mean()

    # ── IBP certifiability ────────────────────────────────────────────────────
    print(f"\nBatched IBP certify on all {len(y_te)} test images…", flush=True)
    t_cert   = time.time()
    cert_all = ibp_certify_batch(model, X_te, y_te, EPS_UNSAT, batch_size=512)
    print(f"  Certify time: {time.time()-t_cert:.0f}s")

    cert_and_correct = cert_all & correct
    n_cert    = int(cert_and_correct.sum())
    cert_rate = n_cert / n_correct if n_correct > 0 else 0.0

    per_class_cert  = {}
    per_class_total = {}
    for cls in range(NUM_CLASSES):
        mask = y_te == cls
        per_class_total[cls] = int((correct & mask).sum())
        per_class_cert[cls]  = int((cert_and_correct & mask).sum())

    viable = [c for c in range(NUM_CLASSES) if per_class_cert[c] >= 5]

    # ── Per-class breakdown ───────────────────────────────────────────────────
    print(f"\n  Per-class IBP certifiable (of correctly classified):")
    print(f"  {'Cls':>4}  {'Cert':>6}  {'Corr':>6}  {'Rate':>7}")
    print(f"  {'-'*30}")
    for cls in range(NUM_CLASSES):
        if per_class_total[cls] > 0:
            rate = per_class_cert[cls] / per_class_total[cls]
            flag = " ← ≥5" if per_class_cert[cls] >= 5 else ""
            print(f"  {cls:>4}  {per_class_cert[cls]:>6}  {per_class_total[cls]:>6}  {rate:>7.1%}{flag}")

    # ── Comparison table ──────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"COMPARISON TABLE")
    print(f"{'='*62}")
    print(f"  {'Metric':<30} | {'Top-half (current)':^20} | {'Random-392':^12}")
    print(f"  {'-'*70}")
    print(f"  {'Pixels / source':<30} | {'392 from [0, 391]':^20} | {'392 from [0, 783]':^12}")
    print(f"  {'Timesteps':<30} | {'14':^20} | {'14':^12}")
    print(f"  {'h8 clean accuracy':<30} | {'79.7%':^20} | {acc:.1%}{'':^5}")
    print(f"  {'IBP-certifiable %':<30} | {'~74%':^20} | {cert_rate:.1%}{'':^5}")
    print(f"  {'Viable classes (≥5 cert)':<30} | {'6 (0,1,4,5,6,7)':^20} | {len(viable)} {str(viable):<10}")
    print(f"\n  Total wall time: {time.time()-t0:.0f}s")

    # ── Plain read ────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"PLAIN READ")
    print(f"{'='*62}")

    acc_delta  = acc - 0.797
    cert_delta = cert_rate - 0.74

    if abs(acc_delta) < 0.02:
        acc_verdict = f"accuracy is essentially unchanged ({acc:.1%} vs 79.7%)"
    elif acc_delta > 0:
        acc_verdict = f"accuracy IMPROVED by {acc_delta:+.1%} ({acc:.1%} vs 79.7%)"
    else:
        acc_verdict = f"accuracy DROPPED by {acc_delta:+.1%} ({acc:.1%} vs 79.7%)"

    if cert_delta > 0.05:
        cert_verdict = f"certifiability IMPROVED to {cert_rate:.1%} (+{cert_delta:.1%})"
    elif cert_delta > -0.10:
        cert_verdict = f"certifiability roughly unchanged at {cert_rate:.1%}"
    else:
        cert_verdict = f"certifiability DROPPED to {cert_rate:.1%} ({cert_delta:+.1%})"

    viable_verdict = (
        f"class diversity is {'maintained' if len(viable) >= 5 else 'REDUCED'} "
        f"({len(viable)} viable classes vs 6)"
    )

    print(f"  {acc_verdict}.")
    print(f"  {cert_verdict}.")
    print(f"  {viable_verdict}.")

    is_net_improvement = (
        cert_rate >= 0.60 and
        len(viable) >= 5 and
        acc >= 0.70
    )
    print()
    if is_net_improvement:
        print(f"  NET VERDICT: Random-392 is a net improvement — better coverage of")
        print(f"  the full digit with no significant cost to certifiability or diversity.")
        print(f"  Viable for Phase 1 full rebuild.")
    else:
        reasons = []
        if cert_rate < 0.60:
            reasons.append(f"certifiability too low ({cert_rate:.1%} < 60% threshold)")
        if len(viable) < 5:
            reasons.append(f"only {len(viable)} viable classes (need ≥5 for a diverse pool)")
        if acc < 0.70:
            reasons.append(f"accuracy too low ({acc:.1%}) — model not learning well enough")
        print(f"  NET VERDICT: Random-392 does NOT clearly improve the benchmark.")
        for r in reasons:
            print(f"    — {r}")


if __name__ == "__main__":
    main()
