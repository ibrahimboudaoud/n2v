#!/usr/bin/env python3
"""
Phase 0 probe — psMNIST LSTM at 28 timesteps / 784 pixels.

Standalone script.  Does NOT modify generate.py or any existing artifacts.
Run from: benchmarks/psMNIST_lstm/

What it does:
  1. Loads MNIST with full 784-pixel permutation (/255 normalization, no Z-score)
  2. Trains h8 once with the existing IBP-regularised setup (warm=10, lambda→1
     over epochs 11–40, same as current benchmark)
  3. Computes IBP certifiability rate at ε=1/255 on correctly-classified test images
  4. Prints per-class counts and a go/no-go recommendation for Phase 1

Does NOT build the shared pool or write any ONNX / VNNLIB files.
"""
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

# ── Constants (28-step version) ───────────────────────────────────────────────
SEED         = 42
SEQ_LEN      = 28          # full image: 28 rows × 28 features
INPUT_SIZE   = 28
INPUT_DIM    = SEQ_LEN * INPUT_SIZE   # 784
NUM_CLASSES  = 10
HIDDEN_SMALL = 8
EPOCHS       = 40
BATCH_SIZE   = 256
LR           = 1e-3
EPS_UNSAT    = 1 / 255     # 1 pixel level — unchanged from current benchmark

import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

_PERM_RNG   = np.random.default_rng(SEED)
PERMUTATION = _PERM_RNG.permutation(INPUT_DIM)   # shape (784,), seeded at 42


# ── Data ──────────────────────────────────────────────────────────────────────

def load_mnist():
    from torchvision import datasets
    raw_train = datasets.MNIST("data", train=True,  download=True)
    raw_test  = datasets.MNIST("data", train=False, download=True)

    def _extract(d):
        imgs = d.data.numpy().astype(np.float32) / 255.0   # (N,28,28) → [0,1]
        flat = imgs.reshape(len(imgs), -1)                  # (N,784)
        return flat[:, PERMUTATION], d.targets.numpy()      # permute all 784

    X_tr, y_tr = _extract(raw_train)
    X_te, y_te = _extract(raw_test)
    return X_tr, y_tr, X_te, y_te


# ── Model — identical structure to generate.py, 28-step version ──────────────

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


# ── IBP helpers ───────────────────────────────────────────────────────────────

def _affine_ibp(W, b, lb, ub):
    W_pos, W_neg = W.clamp(min=0), W.clamp(max=0)
    return (lb @ W_pos.T + ub @ W_neg.T + b,
            ub @ W_pos.T + lb @ W_neg.T + b)


def _mul_ibp(a_lb, a_ub, b_lb, b_ub):
    corners = torch.stack([a_lb*b_lb, a_lb*b_ub, a_ub*b_lb, a_ub*b_ub])
    return corners.min(0).values, corners.max(0).values


def _ibp_fwd_batch(model, lb, ub):
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
        g_lb = gi_lb + gh_lb;  g_ub = gi_ub + gh_ub

        i_lb = torch.sigmoid(g_lb[:, :H]);      i_ub = torch.sigmoid(g_ub[:, :H])
        f_lb = torch.sigmoid(g_lb[:, H:2*H]);   f_ub = torch.sigmoid(g_ub[:, H:2*H])
        g2_lb = torch.tanh(g_lb[:, 2*H:3*H]);   g2_ub = torch.tanh(g_ub[:, 2*H:3*H])
        o_lb = torch.sigmoid(g_lb[:, 3*H:]);    o_ub = torch.sigmoid(g_ub[:, 3*H:])

        fc_lb, fc_ub = _mul_ibp(f_lb, f_ub, c_lb, c_ub)
        ig_lb, ig_ub = _mul_ibp(i_lb, i_ub, g2_lb, g2_ub)
        c_lb = fc_lb + ig_lb;  c_ub = fc_ub + ig_ub

        tc_lb = torch.tanh(c_lb);  tc_ub = torch.tanh(c_ub)
        h_lb, h_ub = _mul_ibp(o_lb, o_ub, tc_lb, tc_ub)

    return _affine_ibp(Wc, bc, h_lb, h_ub)


def ibp_certify_batch(model, X, y, eps, batch_size=512):
    """Vectorised IBP certification over the full test set. Returns bool array (N,)."""
    N = len(X)
    certified = np.zeros(N, dtype=bool)

    with torch.no_grad():
        for start in range(0, N, batch_size):
            xb = torch.tensor(X[start:start+batch_size], dtype=torch.float32)
            yb = torch.tensor(y[start:start+batch_size], dtype=torch.long)
            B  = xb.shape[0]

            lb = (xb - eps).clamp(0.0, 1.0)
            ub = (xb + eps).clamp(0.0, 1.0)

            logit_lb, logit_ub = _ibp_fwd_batch(model, lb, ub)

            # Certify: true-class lower-bound > max wrong-class upper-bound
            true_lb            = logit_lb[torch.arange(B), yb]
            logit_ub_m         = logit_ub.clone()
            logit_ub_m[torch.arange(B), yb] = -1e9          # mask true class
            max_wrong_ub       = logit_ub_m.max(dim=1).values
            certified[start:start+B] = (true_lb > max_wrong_ub).numpy()

    return certified


# ── IBP-regularised training (identical to generate.py's train_ibp) ───────────

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
                lb = (xb - eps).clamp(0.0, 1.0)
                ub = (xb + eps).clamp(0.0, 1.0)
                l_lb, l_ub = _ibp_fwd_batch(model, lb, ub)
                worst = l_ub.clone()
                worst[torch.arange(len(yb)), yb] = l_lb[torch.arange(len(yb)), yb]
                loss = clean_loss + lam * ce(worst, yb)
            else:
                loss = clean_loss
            loss.backward()
            opt.step()

        if ep % 5 == 0 or ep == EPOCHS:
            with torch.no_grad():
                tr_acc = (model(Xt).argmax(1) == yt).float().mean().item()
            print(f"  ep {ep:3d}  lam={lam:.2f}  train_acc={tr_acc:.3f}", flush=True)

    model.eval()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 62)
    print(f"PHASE 0 PROBE — 28 timesteps × 28 features = {INPUT_DIM} pixels")
    print(f"SEED={SEED}  EPS_UNSAT=1/255={EPS_UNSAT:.6f}  H={HIDDEN_SMALL}")
    print(f"Current benchmark: 14×28=392 pixels  |  This probe: 28×28={INPUT_DIM}")
    print("=" * 62)

    print("\nLoading MNIST (/255, full 784-pixel permutation)…")
    X_tr, y_tr, X_te, y_te = load_mnist()
    print(f"  train {X_tr.shape}  test {X_te.shape}")

    print(f"\nTraining h8 with IBP regularisation "
          f"(warm=10, λ→1 over epochs 11–{EPOCHS})…")
    t_train = time.time()
    model = psMNIST_LSTM(HIDDEN_SMALL)
    train_ibp(model, X_tr, y_tr, EPS_UNSAT)
    print(f"  Training wall time: {time.time()-t_train:.0f}s")

    # Save weights so Phase 1 can inspect without retraining
    torch.save(model.state_dict(), "probe_h8_28step.pt")
    print(f"  Weights saved → probe_h8_28step.pt")

    # ── Clean accuracy ────────────────────────────────────────────────────────
    with torch.no_grad():
        Xt    = torch.tensor(X_te, dtype=torch.float32)
        preds = model(Xt).argmax(1).numpy()
    correct   = preds == y_te
    n_correct = correct.sum()
    acc       = correct.mean()
    print(f"\nh8 clean test accuracy : {acc:.3f}  ({n_correct}/{len(y_te)})")

    # ── Batched IBP certifiability ────────────────────────────────────────────
    print(f"\nBatched IBP certify on all {len(y_te)} test images…", flush=True)
    t_cert   = time.time()
    cert_all = ibp_certify_batch(model, X_te, y_te, EPS_UNSAT, batch_size=512)
    print(f"  Certify wall time: {time.time()-t_cert:.0f}s")

    cert_and_correct = cert_all & correct
    n_cert    = cert_and_correct.sum()
    cert_rate = n_cert / n_correct   # denominator: correctly classified only

    per_class_cert  = {}
    per_class_total = {}
    for cls in range(NUM_CLASSES):
        mask = y_te == cls
        per_class_total[cls] = int((correct & mask).sum())
        per_class_cert[cls]  = int((cert_and_correct & mask).sum())

    viable = [c for c in range(NUM_CLASSES) if per_class_cert[c] >= 5]

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"PHASE 0 RESULTS")
    print(f"{'='*62}")
    print(f"  Input             : {SEQ_LEN} timesteps × {INPUT_SIZE} features = {INPUT_DIM} pixels")
    print(f"  h8 test accuracy  : {acc:.3f}")
    print(f"  IBP certifiable   : {n_cert} / {n_correct} correctly-classified = {cert_rate:.1%}")
    print(f"  (reference: ~74% at 14 timesteps)")
    print()
    print(f"  Per-class breakdown (of correctly classified in test set):")
    print(f"  {'Cls':>4}  {'Cert':>6}  {'Corr':>6}  {'Rate':>7}")
    print(f"  {'-'*30}")
    for cls in range(NUM_CLASSES):
        if per_class_total[cls] > 0:
            rate = per_class_cert[cls] / per_class_total[cls]
            flag = " ← ≥5" if per_class_cert[cls] >= 5 else ""
            print(f"  {cls:>4}  {per_class_cert[cls]:>6}  {per_class_total[cls]:>6}  {rate:>7.1%}{flag}")

    print(f"\n  Classes with ≥5 certifiable images (pool-viable): {viable}  ({len(viable)} total)")
    print(f"  Total wall time: {time.time()-t0:.0f}s")

    # ── Recommendation ────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"RECOMMENDATION")
    print(f"{'='*62}")

    if cert_rate >= 0.30 and len(viable) >= 5:
        print(f"  PROCEED TO PHASE 1.")
        print(f"  Rate {cert_rate:.1%} ≥ 30% with {len(viable)} viable classes →")
        print(f"  a 25-instance diverse pool is achievable with the current setup.")
    elif cert_rate >= 0.10 and len(viable) >= 4:
        print(f"  MARGINAL — confirm with user before proceeding.")
        print(f"  Rate {cert_rate:.1%} may yield a pool but class diversity could be tight.")
        print(f"  Suggestion: try warm=20 or EPOCHS=60 for stronger certifiability.")
    else:
        print(f"  STOP — DO NOT proceed to Phase 1 without tuning.")
        if cert_rate < 0.10:
            print(f"  Rate {cert_rate:.1%} < 10%: IBP blowup at 28 steps is too severe")
            print(f"  with the current warm=10 schedule.")
            print(f"  Options (in order of least disruption):")
            print(f"    A. Increase warm period: warm=20, keep EPOCHS=40")
            print(f"       (more clean training before IBP pressure is applied)")
            print(f"    B. Extend training: EPOCHS=60, warm=20")
            print(f"    C. Reduce hidden size to H=6 or H=4")
            print(f"       (smaller Whh → less IBP blowup per timestep)")
        if len(viable) < 4:
            print(f"  Only {len(viable)} viable class(es) → pool would lack diversity.")


if __name__ == "__main__":
    main()
