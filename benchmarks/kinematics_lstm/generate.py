#!/usr/bin/env python3
"""
kinematics_lstm — VNN-COMP 2026 benchmark generator.

Produces a family of small LSTM classifiers on synthetic 2-D point-mass
kinematic time-series and emits ONNX models + VNN-LIB 2.0 robustness
properties for ≥ 50 labeled SAT/UNSAT instances.

Architecture
------------
LSTMs are *unrolled* at ONNX export time: the Python loop over SEQ_LEN=10
timesteps is traced by torch.onnx.export into a flat Gemm/Add/Sigmoid/Tanh/Mul
graph with no ONNX LSTM operator.  Every VNN-COMP-capable feed-forward verifier
(α,β-CROWN, Marabou, n2v, …) can consume these models without modification.

Data domain
-----------
2-D point mass with (de)amplified velocity, 4 features [x, y, vx, vy] over
10 timesteps.  Label 1 = trajectory stays in [-3, 3]^2 (stable), 0 otherwise.

Robustness property (SAT semantics)
------------------------------------
For each test sample x*:
    ∃ x′ ∈ [x* - ε, x* + ε]  s.t.  Y_other ≥ Y_true
i.e. SAT = adversarial example exists (not robust); UNSAT = robust.

Labeling
--------
  UNSAT: certified via Interval Bound Propagation (IBP) through the LSTM.
  SAT:   confirmed via PGD-Linf attack (explicit counterexample found).

Usage
-----
    cd benchmarks/kinematics_lstm
    python generate.py

Dependencies
------------
    pip install torch onnx numpy
"""

import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
import onnx

# ── Constants ─────────────────────────────────────────────────────────────────

SEED         = 42
SEQ_LEN      = 10
INPUT_FEAT   = 4              # [x, y, vx, vy]
INPUT_DIM    = SEQ_LEN * INPUT_FEAT   # 40 — flattened LSTM input
OUTPUT_DIM   = 2              # stable (1) vs. unstable (0)
DT           = 0.1
SAFE_BOUND   = 3.0
HIDDEN_SIZES = [8, 16, 32]    # three model variants
EPS_UNSAT    = 0.005          # tight L-inf ball → IBP-certifiable UNSAT
EPS_SAT      = 0.05           # loose L-inf ball → PGD finds adversarial
TIMEOUT      = 120            # VNN-COMP per-instance timeout (seconds)
N_UNSAT_EACH = 12             # UNSAT instances to collect per model
N_SAT_EACH   = 8              # SAT instances to collect per model

# ── Reproducibility ───────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Data generation
# ─────────────────────────────────────────────────────────────────────────────

def _one_trajectory(a: float, rng: np.random.Generator):
    """
    Single 2-D kinematic trajectory with damped/amplified velocity.

        x(t+1)  = x(t)  + DT * vx(t)
        y(t+1)  = y(t)  + DT * vy(t)
        vx(t+1) = a * vx(t) + N(0, 0.05)
        vy(t+1) = a * vy(t) + N(0, 0.05)

    Returns (flat array of shape (INPUT_DIM,), int label).
    Label 1 (stable) iff all |x|, |y| ≤ SAFE_BOUND across SEQ_LEN steps.
    """
    x  = rng.uniform(-1.0, 1.0)
    y  = rng.uniform(-1.0, 1.0)
    vx = rng.uniform(-2.0, 2.0)
    vy = rng.uniform(-2.0, 2.0)

    traj = []
    for _ in range(SEQ_LEN):
        traj.append([x, y, vx, vy])
        x  = x  + DT * vx
        y  = y  + DT * vy
        vx = a * vx + rng.normal(0.0, 0.05)
        vy = a * vy + rng.normal(0.0, 0.05)

    traj = np.array(traj, dtype=np.float32)
    label = int(np.all(np.abs(traj[:, :2]) <= SAFE_BOUND))
    return traj.flatten(), label


def generate_dataset(n_per_class: int, rng: np.random.Generator):
    """Generate a balanced dataset with n_per_class samples per label."""
    seqs, labels = [], []

    # stable trajectories (a in [0.70, 0.95])
    while sum(l == 1 for l in labels) < n_per_class:
        s, l = _one_trajectory(rng.uniform(0.70, 0.95), rng)
        if l == 1:
            seqs.append(s); labels.append(l)

    # unstable trajectories (a in [1.05, 1.30])
    while sum(l == 0 for l in labels) < n_per_class:
        s, l = _one_trajectory(rng.uniform(1.05, 1.30), rng)
        if l == 0:
            seqs.append(s); labels.append(l)

    idx = np.arange(len(seqs))
    rng.shuffle(idx)
    X = np.stack([seqs[i] for i in idx])
    y = np.array([labels[i] for i in idx], dtype=np.int64)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Model — unrolled LSTM
# ─────────────────────────────────────────────────────────────────────────────

class _ManualLSTMCell(nn.Module):
    """
    LSTM cell with explicit gate ops (no nn.LSTMCell / nn.LSTM).

    When torch.onnx.export traces KinematicsLSTM, the Python loop is unrolled
    and each gate reduces to Gemm + Add + Sigmoid/Tanh — standard ONNX ops
    that any feed-forward verifier can handle.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # Combined [i, f, g, o] projection — matches PyTorch LSTM gate order
        self.Wih = nn.Linear(input_size,  4 * hidden_size, bias=True)
        self.Whh = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

    def forward(self, x_t, h, c):
        gates = self.Wih(x_t) + self.Whh(h)
        H = self.hidden_size
        # Explicit constant-index slices avoid Shape/Gather/Div ops at export time
        i = torch.sigmoid(gates[:, :H])
        f = torch.sigmoid(gates[:, H:2*H])
        g = torch.tanh(gates[:, 2*H:3*H])
        o = torch.sigmoid(gates[:, 3*H:4*H])
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class KinematicsLSTM(nn.Module):
    """
    Unrolled LSTM binary classifier.

    Input  : (1, INPUT_DIM=40) — flattened kinematic sequence
    Output : (1, 2)            — class logits [unstable, stable]
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell        = _ManualLSTMCell(INPUT_FEAT, hidden_size)
        self.classifier  = nn.Linear(hidden_size, OUTPUT_DIM)
        # Register zero initial states as buffers → ONNX initializers, not Shape/Gather ops
        self.register_buffer("h0", torch.zeros(1, hidden_size))
        self.register_buffer("c0", torch.zeros(1, hidden_size))

    def forward(self, x_flat):
        h, c = self.h0, self.c0
        for t in range(SEQ_LEN):
            x_t = x_flat[:, t * INPUT_FEAT : (t + 1) * INPUT_FEAT]
            h, c = self.cell(x_t, h, c)
        return self.classifier(h)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_model(model: nn.Module, X: np.ndarray, y: np.ndarray,
                epochs: int = 80, lr: float = 1e-3, batch: int = 64) -> None:
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    Xt      = torch.tensor(X, dtype=torch.float32)
    yt      = torch.tensor(y, dtype=torch.long)
    n       = len(Xt)
    model.train()

    for ep in range(1, epochs + 1):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            loss_fn(model(Xt[idx]), yt[idx]).backward()
            opt.step()
        if ep % 20 == 0 or ep == epochs:
            with torch.no_grad():
                acc = (model(Xt).argmax(1) == yt).float().mean().item()
            print(f"    ep {ep:3d}  train_acc={acc:.3f}")

    model.eval()


# ─────────────────────────────────────────────────────────────────────────────
# ONNX export
# ─────────────────────────────────────────────────────────────────────────────

def _fold_constant_matmul(path: str) -> None:
    """
    Fold MatMul(A, B) → constant initializer when both A and B are ONNX
    initializers (constants).

    At timestep 0 the LSTM runs Whh @ h0 where h0 = zeros.  n2v's
    _handle_onnx_matmul only handles MatMul(set, const); it returns None
    when the first operand is also a constant.  By folding these
    constant-only MatMuls into new initializers, the downstream Add sees
    Add(computed_set, const_zero), which Case-2 of _handle_onnx_binary_op
    already handles correctly.
    """
    from onnx import numpy_helper
    model = onnx.load(path)
    init_map = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}

    new_inits, keep_nodes = [], []
    for node in model.graph.node:
        if (node.op_type == "MatMul" and len(node.input) == 2
                and all(inp in init_map for inp in node.input)):
            result = np.matmul(init_map[node.input[0]],
                               init_map[node.input[1]]).astype(np.float32)
            init_map[node.output[0]] = result
            new_inits.append(numpy_helper.from_array(result, name=node.output[0]))
        else:
            keep_nodes.append(node)

    model.graph.initializer.extend(new_inits)
    del model.graph.node[:]
    model.graph.node.extend(keep_nodes)
    onnx.save(model, path)


def _remove_identity_ops(path: str) -> None:
    """
    Remove ONNX Identity ops by rewiring their connections in-place.

    Identity ops are ONNX no-ops (y = x).  They appear as OnnxCopyIdentity
    in onnx2torch's FX graph, which n2v's reach engine does not yet handle.
    Removing them here keeps the ONNX artifact clean and immediately verifier-
    compatible without modifying n2v.
    """
    model = onnx.load(path)

    # Build: identity_output → real_source (follow chains)
    id_map: dict = {}
    keep_nodes = []
    for node in model.graph.node:
        if node.op_type == "Identity":
            id_map[node.output[0]] = node.input[0]
        else:
            keep_nodes.append(node)

    def _resolve(name: str) -> str:
        seen = set()
        while name in id_map and name not in seen:
            seen.add(name)
            name = id_map[name]
        return name

    for node in keep_nodes:
        for i, inp in enumerate(node.input):
            node.input[i] = _resolve(inp)

    # Replace graph nodes
    del model.graph.node[:]
    model.graph.node.extend(keep_nodes)

    onnx.checker.check_model(model)
    onnx.save(model, path)


def export_onnx(model: nn.Module, path: str) -> None:
    """Export unrolled LSTM to ONNX (opset 14, static batch=1, no Identity ops)."""
    dummy = torch.zeros(1, INPUT_DIM, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, path,
        input_names=["model_in"],
        output_names=["model_out"],
        opset_version=14,
        dynamo=False,
    )
    _fold_constant_matmul(path)
    _remove_identity_ops(path)
    onnx.checker.check_model(onnx.load(path))
    print(f"    Exported → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Interval Bound Propagation — UNSAT certification
# ─────────────────────────────────────────────────────────────────────────────

def _affine_ibp(W, b, lb, ub):
    """
    Tight interval bounds for y = x @ W.T + b,  x ∈ [lb, ub].

    W : (out, in),  b : (out,),  lb/ub : (1, in) → y_lb/y_ub : (1, out)
    """
    W_pos = W.clamp(min=0)
    W_neg = W.clamp(max=0)
    y_lb = lb @ W_pos.T + ub @ W_neg.T + b
    y_ub = ub @ W_pos.T + lb @ W_neg.T + b
    return y_lb, y_ub


def _mul_ibp(a_lb, a_ub, b_lb, b_ub):
    """Tight interval bounds for c = a ⊙ b (element-wise)."""
    corners = torch.stack([a_lb * b_lb, a_lb * b_ub, a_ub * b_lb, a_ub * b_ub])
    return corners.min(0).values, corners.max(0).values


def ibp_certify(model: nn.Module, x: np.ndarray, eps: float, true_cls: int) -> bool:
    """
    Certify UNSAT (robustness) via IBP through the unrolled LSTM.

    Returns True iff IBP proves that for ALL x' ∈ [x-ε, x+ε]:
        logit[true_cls] > logit[other_cls]
    which means the property (SAT = adversarial flip exists) is UNSAT.
    """
    H    = model.hidden_size
    cell = model.cell
    clf  = model.classifier

    Wih      = cell.Wih.weight.data   # (4H, INPUT_FEAT)
    bih      = cell.Wih.bias.data     # (4H,)
    Whh      = cell.Whh.weight.data   # (4H, H)
    zero4H   = torch.zeros(4 * H)
    Wc       = clf.weight.data        # (2, H)
    bc       = clf.bias.data          # (2,)

    lb = torch.tensor(x - eps, dtype=torch.float32).unsqueeze(0)  # (1, 40)
    ub = torch.tensor(x + eps, dtype=torch.float32).unsqueeze(0)

    # Initial hidden/cell state is exactly zero (no uncertainty)
    h_lb = torch.zeros(1, H);  h_ub = torch.zeros(1, H)
    c_lb = torch.zeros(1, H);  c_ub = torch.zeros(1, H)

    for t in range(SEQ_LEN):
        xt_lb = lb[:, t * INPUT_FEAT : (t + 1) * INPUT_FEAT]
        xt_ub = ub[:, t * INPUT_FEAT : (t + 1) * INPUT_FEAT]

        # gates = Wih @ x_t + b_ih  +  Whh @ h  (no bias for Whh)
        gi_lb, gi_ub = _affine_ibp(Wih, bih,  xt_lb, xt_ub)
        gh_lb, gh_ub = _affine_ibp(Whh, zero4H, h_lb, h_ub)
        g_lb = gi_lb + gh_lb
        g_ub = gi_ub + gh_ub

        # Split into gates [i, f, g, o]
        def _sl(t, k): return t[:, k * H : (k + 1) * H]

        i_lb, i_ub = torch.sigmoid(_sl(g_lb, 0)), torch.sigmoid(_sl(g_ub, 0))
        f_lb, f_ub = torch.sigmoid(_sl(g_lb, 1)), torch.sigmoid(_sl(g_ub, 1))
        g2_lb, g2_ub = torch.tanh(_sl(g_lb, 2)),  torch.tanh(_sl(g_ub, 2))
        o_lb, o_ub  = torch.sigmoid(_sl(g_lb, 3)), torch.sigmoid(_sl(g_ub, 3))

        # c_new = f * c + i * g
        fc_lb, fc_ub = _mul_ibp(f_lb, f_ub, c_lb, c_ub)
        ig_lb, ig_ub = _mul_ibp(i_lb, i_ub, g2_lb, g2_ub)
        c_lb = fc_lb + ig_lb
        c_ub = fc_ub + ig_ub

        # h_new = o * tanh(c_new)
        tc_lb, tc_ub = torch.tanh(c_lb), torch.tanh(c_ub)
        h_lb, h_ub   = _mul_ibp(o_lb, o_ub, tc_lb, tc_ub)

    # Classifier output bounds
    logit_lb, logit_ub = _affine_ibp(Wc, bc, h_lb, h_ub)

    other = 1 - true_cls
    # UNSAT iff lower bound of true class > upper bound of adversarial class
    return bool(logit_lb[0, true_cls].item() > logit_ub[0, other].item())


# ─────────────────────────────────────────────────────────────────────────────
# SAT instance generation — binary-search boundary finder
# ─────────────────────────────────────────────────────────────────────────────

def find_sat_instance(
    model: nn.Module,
    x0: np.ndarray,
    c0: int,
    x1_pool: np.ndarray,
    n_bisect: int = 50,
    delta: float = 0.015,
) -> tuple:
    """
    Guarantee a SAT instance with small eps by using the decision boundary
    as the test center.

    Algorithm:
      1. Binary-search for the boundary between x0 (class c0) and the
         nearest x1 from x1_pool (class 1-c0).
      2. After convergence, a ≈ boundary (class c0 side), b ≈ boundary
         (class 1-c0 side).
      3. Step delta further into c0 to get x_test.
      4. eps = 1.1 × L-inf(x_test, b) ≈ delta → small, ~0.015.

    Returns (x_test, eps, c0) where b (the SAT witness) is inside the ball.
    """
    other = 1 - c0
    dists = np.abs(x1_pool - x0).max(axis=1)
    x1    = x1_pool[dists.argmin()]

    def _pred(z):
        with torch.no_grad():
            return int(model(torch.tensor(z, dtype=torch.float32).unsqueeze(0)).argmax(1))

    if _pred(x0) != c0 or _pred(x1) != other:
        return None

    # Binary search to the boundary
    a, b = x0.copy(), x1.copy()
    for _ in range(n_bisect):
        mid = (a + b) / 2.0
        if _pred(mid) == c0:
            a = mid
        else:
            b = mid

    # Unit L-inf direction from x1 toward x0 (into c0 territory)
    direction = x0 - x1
    linf_norm = float(np.abs(direction).max())
    if linf_norm < 1e-10:
        return None
    unit_dir = direction / linf_norm

    # x_test is delta steps deeper into c0 territory from the boundary
    x_test = a + delta * unit_dir
    if _pred(x_test) != c0:
        x_test = a   # fallback: use boundary directly (may still be c0)
        if _pred(x_test) != c0:
            return None

    # eps covers the SAT witness b (just past boundary in c1)
    eps = float(np.abs(x_test - b).max()) * 1.1
    eps = max(eps, 1e-5)

    if _pred(b) == c0:
        return None
    if float(np.abs(x_test - b).max()) > eps:
        return None

    return x_test, eps, c0


# ─────────────────────────────────────────────────────────────────────────────
# VNN-LIB 2.0 writer
# ─────────────────────────────────────────────────────────────────────────────

def write_vnnlib(path: str, x: np.ndarray, eps: float, true_cls: int) -> None:
    """
    Write a VNN-LIB 2.0 robustness property.

    SAT semantics: the property is satisfiable if there exists an input x' in
    the L-inf ball that causes the network to predict the wrong class.

        input:  X_0 … X_39  ∈ [x-ε, x+ε]
        output: (<= Y_{true_cls} Y_{other})   ← adversarial class wins or ties
    """
    other = 1 - true_cls
    lb    = x - eps
    ub    = x + eps

    with open(path, "w") as f:
        f.write("; kinematics_lstm robustness property\n")
        f.write(f"; true_class={true_cls}  eps={eps:.6f}\n")
        f.write(f"; SAT = adversarial example exists (predicted class flips to {other})\n\n")

        for i in range(INPUT_DIM):
            f.write(f"(declare-const X_{i} Real)\n")
        f.write("\n")
        for i in range(OUTPUT_DIM):
            f.write(f"(declare-const Y_{i} Real)\n")
        f.write("\n")

        f.write("; Input: L-inf ball of radius eps around test sample\n")
        for i in range(INPUT_DIM):
            f.write(f"(assert (>= X_{i} {lb[i]:.8f}))\n")
            f.write(f"(assert (<= X_{i} {ub[i]:.8f}))\n")
        f.write("\n")

        # Adversarial class wins: Y_true ≤ Y_other
        f.write("; Output: adversarial class logit >= true class logit\n")
        f.write(f"(assert (<= Y_{true_cls} Y_{other}))\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(SEED)

    os.makedirs("onnx",   exist_ok=True)
    os.makedirs("vnnlib", exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    print("Generating synthetic kinematic dataset …")
    X_tr, y_tr = generate_dataset(800, rng)
    X_te, y_te = generate_dataset(200, rng)   # 400 test samples (200 per class)
    print(f"  train {X_tr.shape}  test {X_te.shape}")

    # Z-score normalisation (computed from training set)
    mu  = X_tr.mean(axis=0, keepdims=True)
    sig = X_tr.std(axis=0,  keepdims=True) + 1e-6
    X_tr_n = (X_tr - mu) / sig
    X_te_n = (X_te - mu) / sig
    np.savez("norm_params.npz", mean=mu, std=sig)

    # ── Per-model loop ────────────────────────────────────────────────────────
    records  = []   # (onnx_rel, vnnlib_rel, label, timeout)
    prop_idx = 0

    for hs in HIDDEN_SIZES:
        print(f"\n── hidden_size={hs} {'─'*40}")
        model = KinematicsLSTM(hs)
        train_model(model, X_tr_n, y_tr)

        with torch.no_grad():
            Xt     = torch.tensor(X_te_n, dtype=torch.float32)
            logits = model(Xt).numpy()
            preds  = logits.argmax(axis=1)
        acc = (preds == y_te).mean()
        print(f"  test_acc = {acc:.3f}")

        onnx_name = f"lstm_kin_h{hs}.onnx"
        export_onnx(model, f"onnx/{onnx_name}")

        margins = np.abs(logits[:, 0] - logits[:, 1])
        correct = preds == y_te

        # ── UNSAT: high-margin correctly classified samples ────────────────
        pool = np.where(correct)[0]
        pool = pool[np.argsort(margins[pool])[::-1]]   # highest margin first

        n_unsat = 0
        for idx in pool:
            if n_unsat >= N_UNSAT_EACH:
                break
            x    = X_te_n[idx]
            tc   = int(y_te[idx])
            if not ibp_certify(model, x, EPS_UNSAT, tc):
                continue
            pname = f"prop_{prop_idx:03d}.vnnlib"
            write_vnnlib(f"vnnlib/{pname}", x, EPS_UNSAT, tc)
            records.append((f"onnx/{onnx_name}", f"vnnlib/{pname}", "unsat", TIMEOUT))
            prop_idx += 1
            n_unsat  += 1

        # ── SAT: binary-search to decision boundary ────────────────────────
        # Build per-class pools (correctly classified test samples)
        c0_pool = X_te_n[np.where((y_te == 0) & correct)[0]]
        c1_pool = X_te_n[np.where((y_te == 1) & correct)[0]]
        sat_sources = (
            [(x, 0, c1_pool) for x in c0_pool[:N_SAT_EACH * 4]] +
            [(x, 1, c0_pool) for x in c1_pool[:N_SAT_EACH * 4]]
        )

        n_sat = 0
        for x, tc, opp_pool in sat_sources:
            if n_sat >= N_SAT_EACH:
                break
            result = find_sat_instance(model, x, tc, opp_pool)
            if result is None:
                continue
            x_test, eps, true_cls = result
            pname = f"prop_{prop_idx:03d}.vnnlib"
            write_vnnlib(f"vnnlib/{pname}", x_test, eps, true_cls)
            records.append((f"onnx/{onnx_name}", f"vnnlib/{pname}", "sat", TIMEOUT))
            prop_idx += 1
            n_sat    += 1

        print(f"  instances: {n_unsat} UNSAT  {n_sat} SAT  "
              f"(targets: {N_UNSAT_EACH} / {N_SAT_EACH})")

    # ── Write instances.csv (VNN-COMP standard: no header) ───────────────────
    with open("instances.csv", "w", newline="") as f:
        csv.writer(f).writerows(
            [(r[0], r[1], r[3]) for r in records]
        )

    # ── Write ground_truth.csv (benchmark validation) ────────────────────────
    with open("ground_truth.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["onnx", "vnnlib", "result", "timeout"])
        w.writerows(records)

    # ── Summary ───────────────────────────────────────────────────────────────
    total   = len(records)
    n_sat   = sum(1 for r in records if r[2] == "sat")
    n_unsat = sum(1 for r in records if r[2] == "unsat")
    print(f"\n{'─'*50}")
    print(f"Total instances : {total}  ({n_unsat} UNSAT / {n_sat} SAT)")
    print(f"Models          : onnx/  ({len(HIDDEN_SIZES)} files)")
    print(f"Properties      : vnnlib/  ({total} files)")
    print(f"Instance list   : instances.csv")
    print(f"Ground truth    : ground_truth.csv")
    if total < 50:
        print(f"\nWARNING: {total} < 50 instances.  "
              "Try lowering EPS_UNSAT or increasing N_UNSAT_EACH / N_SAT_EACH.")
    else:
        print(f"\nBenchmark satisfies the ≥ 50 labeled instance requirement.")


if __name__ == "__main__":
    main()
