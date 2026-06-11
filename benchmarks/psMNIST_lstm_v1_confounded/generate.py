#!/usr/bin/env python3
"""
psMNIST_lstm — VNN-COMP 2026 benchmark generator.

Permuted Sequential MNIST (psMNIST) is the LSTM equivalent of the classic
MNIST feedforward benchmark.  Every verifier in VNN-COMP has an MNIST
feed-forward entry; this provides the recurrent counterpart.

How it works:
  - Take an MNIST image (28x28 = 784 pixels)
  - Apply a FIXED random permutation to the flattened pixels
  - Feed the permuted sequence row-by-row: 28 timesteps x 28 features
  - An LSTM classifies into 10 digit classes (0-9)

Why psMNIST instead of plain sMNIST?
  The permutation breaks all spatial locality.  The model cannot exploit
  the fact that adjacent pixels are related — it must learn long-range
  dependencies across the full 784-step sequence.  This is the hardest
  version of the task and is the standard recurrent network benchmark.

Verification property (same structure as ACAS Xu robustness):
  For a test image x* and epsilon ball of radius eps:
    SAT  = there exists x' in [x*-eps, x*+eps] where a wrong digit wins
    UNSAT = for every x' in the ball, the correct digit always wins

Pipeline stages this script exercises end-to-end:
  1. Real data loading (MNIST via torchvision)
  2. LSTM training
  3. ONNX export + cleanup (Identity removal, constant folding)
  4. VNN-LIB 2.0 property writing (multi-class disjunction)
  5. IBP-based UNSAT certification
  6. Decision-boundary SAT instance construction
  7. n2v Box reachability verification (dry run)

Usage:
    cd benchmarks/psMNIST_lstm
    python generate.py

Dependencies:
    pip install torch torchvision onnx numpy
"""

import csv
import os
import sys
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import onnx

warnings.filterwarnings("ignore")

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Constants ─────────────────────────────────────────────────────────────────
SEQ_LEN    = 14          # first 14 rows of the permuted image (392 pixels)
INPUT_SIZE = 28          # pixels per row
INPUT_DIM  = SEQ_LEN * INPUT_SIZE   # 392 — total flattened input to ONNX model
NUM_CLASSES = 10
HIDDEN_SMALL = 8     # used for UNSAT — tiny network, tight IBP bounds
HIDDEN_LARGE = 64   # used for SAT  — realistic accuracy, falsification works
EPOCHS       = 40
BATCH_SIZE   = 256
LR           = 1e-3

EPS_UNSAT    = 0.005     # L-inf ball radius for UNSAT (tight perturbation)
TIMEOUT      = 120       # VNN-COMP per-instance timeout (seconds)

N_UNSAT      = 25        # target UNSAT instances
N_SAT        = 25        # target SAT instances

# Fixed permutation — every run uses the same one (reproducible benchmark)
_PERM_RNG   = np.random.default_rng(SEED)
PERMUTATION = _PERM_RNG.permutation(INPUT_DIM)   # shape (784,)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_mnist():
    """
    Download MNIST and return normalised, permuted numpy arrays.

    Preprocessing:
      1. Flatten 28x28 image → 784-vector, scale to [0,1]
      2. Apply fixed permutation
      3. Z-score normalise using training-set statistics
      4. Return as (N, 784) float32 arrays with integer labels
    """
    from torchvision import datasets, transforms

    raw_train = datasets.MNIST("data", train=True,  download=True,
                               transform=transforms.ToTensor())
    raw_test  = datasets.MNIST("data", train=False, download=True,
                               transform=transforms.ToTensor())

    def _extract(dataset):
        imgs   = dataset.data.numpy().astype(np.float32) / 255.0  # (N,28,28)
        labels = dataset.targets.numpy()
        flat   = imgs.reshape(len(imgs), -1)                       # (N,784)
        perm   = flat[:, PERMUTATION]                              # permute
        return perm, labels

    X_tr, y_tr = _extract(raw_train)
    X_te, y_te = _extract(raw_test)

    # Z-score from training set only
    mu  = X_tr.mean(axis=0, keepdims=True)
    sig = X_tr.std(axis=0,  keepdims=True) + 1e-6
    X_tr = (X_tr - mu) / sig
    X_te = (X_te - mu) / sig

    return X_tr, y_tr, X_te, y_te, mu, sig


# ─────────────────────────────────────────────────────────────────────────────
# Model — unrolled LSTM
# ─────────────────────────────────────────────────────────────────────────────

class _ManualLSTMCell(nn.Module):
    """
    LSTM cell with explicit gate ops.
    Using constant-index slices (not torch.chunk) avoids Shape/Gather/Div
    ops in the exported ONNX graph.
    """
    def __init__(self, input_size: int, hidden_size: int):
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
        o = torch.sigmoid(gates[:, 3*H:4*H])
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class psMNIST_LSTM(nn.Module):
    """
    Unrolled LSTM classifier for permuted sequential MNIST.

    Input  : (1, 784)  — flattened, permuted, normalised pixel sequence
    Output : (1, 10)   — class logits for digits 0-9

    The Python loop over SEQ_LEN=28 is traced by torch.onnx.export into a
    flat Gemm/Sigmoid/Tanh/Mul/Slice graph.  No ONNX LSTM operator is emitted.
    """
    def __init__(self, hidden_size: int = HIDDEN_LARGE):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell        = _ManualLSTMCell(INPUT_SIZE, hidden_size)
        self.classifier  = nn.Linear(hidden_size, NUM_CLASSES)
        # Register zero initial states as buffers (become ONNX initializers,
        # not dynamic Shape ops)
        self.register_buffer("h0", torch.zeros(1, hidden_size))
        self.register_buffer("c0", torch.zeros(1, hidden_size))

    def forward(self, x_flat):          # x_flat: (1, 784)
        h, c = self.h0, self.c0
        for t in range(SEQ_LEN):
            x_t = x_flat[:, t * INPUT_SIZE : (t + 1) * INPUT_SIZE]
            h, c = self.cell(x_t, h, c)
        return self.classifier(h)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(model: nn.Module, X: np.ndarray, y: np.ndarray) -> None:
    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    Xt      = torch.tensor(X, dtype=torch.float32)
    yt      = torch.tensor(y, dtype=torch.long)
    n       = len(Xt)
    model.train()

    for ep in range(1, EPOCHS + 1):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            opt.zero_grad()
            loss_fn(model(Xt[idx]), yt[idx]).backward()
            opt.step()
        if ep % 10 == 0 or ep == EPOCHS:
            with torch.no_grad():
                acc = (model(Xt).argmax(1) == yt).float().mean().item()
            print(f"    ep {ep:3d}  train_acc={acc:.3f}")

    model.eval()


# ─────────────────────────────────────────────────────────────────────────────
# ONNX export + graph cleanup
# ─────────────────────────────────────────────────────────────────────────────

def _fold_constant_matmul(path: str) -> None:
    """
    Fold MatMul(const, const) into a new ONNX initializer.

    At timestep 0, Whh @ h0 = Whh @ zeros = zeros.  n2v's MatMul handler
    only handles MatMul(set, const); it fails on MatMul(const, const).
    Folding this into a constant initializer lets n2v see
    Add(computed_set, zeros) instead — which it already handles.
    """
    from onnx import numpy_helper
    model   = onnx.load(path)
    imap    = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    new_i, keep = [], []

    for node in model.graph.node:
        if (node.op_type == "MatMul" and len(node.input) == 2
                and all(x in imap for x in node.input)):
            result = np.matmul(imap[node.input[0]],
                               imap[node.input[1]]).astype(np.float32)
            imap[node.output[0]] = result
            new_i.append(numpy_helper.from_array(result, name=node.output[0]))
        else:
            keep.append(node)

    model.graph.initializer.extend(new_i)
    del model.graph.node[:]
    model.graph.node.extend(keep)
    onnx.save(model, path)


def _remove_identity_ops(path: str) -> None:
    """
    Remove Identity ops by rewiring their connections.

    PyTorch's ONNX exporter emits Identity nodes when the same tensor is
    used in multiple places (e.g., h0 and c0 are both zeros, so
    c0 = Identity(h0)).  onnx2torch converts these to OnnxCopyIdentity
    modules, which n2v's reach engine doesn't handle.  Removing them is
    mathematically equivalent — Identity(x) = x.
    """
    model = onnx.load(path)
    id_map, keep = {}, []

    for node in model.graph.node:
        if node.op_type == "Identity":
            id_map[node.output[0]] = node.input[0]
        else:
            keep.append(node)

    def _resolve(name):
        seen = set()
        while name in id_map and name not in seen:
            seen.add(name); name = id_map[name]
        return name

    for node in keep:
        for i, inp in enumerate(node.input):
            node.input[i] = _resolve(inp)

    del model.graph.node[:]
    model.graph.node.extend(keep)
    onnx.checker.check_model(model)
    onnx.save(model, path)


def export_onnx(model: nn.Module, path: str) -> None:
    """Export psMNIST LSTM to ONNX opset 14, then clean Identity and
    constant MatMul ops so n2v can process it."""
    dummy = torch.zeros(1, INPUT_DIM, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, path,
        input_names=["model_in"], output_names=["model_out"],
        opset_version=14, dynamo=False,
    )
    _fold_constant_matmul(path)
    _remove_identity_ops(path)
    onnx.checker.check_model(onnx.load(path))
    ops = sorted({n.op_type for n in onnx.load(path).graph.node})
    print(f"    Exported → {path}")
    print(f"    ONNX ops: {ops}")


# ─────────────────────────────────────────────────────────────────────────────
# Interval Bound Propagation — UNSAT certification
# ─────────────────────────────────────────────────────────────────────────────

def _affine_ibp(W, b, lb, ub):
    """Tight interval bounds for y = x @ W.T + b,  x in [lb, ub]."""
    W_pos, W_neg = W.clamp(min=0), W.clamp(max=0)
    return (lb @ W_pos.T + ub @ W_neg.T + b,
            ub @ W_pos.T + lb @ W_neg.T + b)


def _mul_ibp(a_lb, a_ub, b_lb, b_ub):
    """Tight interval bounds for c = a * b element-wise."""
    corners = torch.stack([a_lb*b_lb, a_lb*b_ub, a_ub*b_lb, a_ub*b_ub])
    return corners.min(0).values, corners.max(0).values


def ibp_certify(model: nn.Module, x: np.ndarray, eps: float,
                true_cls: int) -> bool:
    """
    Certify UNSAT via IBP through the unrolled LSTM.

    Returns True iff for ALL inputs in [x-eps, x+eps]:
        logit[true_cls] > logit[j]  for every j != true_cls

    This is STRONGER than the binary case — all 9 wrong classes must lose.
    """
    H    = model.hidden_size
    cell = model.cell
    clf  = model.classifier
    Wih  = cell.Wih.weight.data
    bih  = cell.Wih.bias.data
    Whh  = cell.Whh.weight.data
    zero4H = torch.zeros(4 * H)
    Wc   = clf.weight.data
    bc   = clf.bias.data

    lb = torch.tensor(x - eps, dtype=torch.float32).unsqueeze(0)
    ub = torch.tensor(x + eps, dtype=torch.float32).unsqueeze(0)

    h_lb = torch.zeros(1, H);  h_ub = torch.zeros(1, H)
    c_lb = torch.zeros(1, H);  c_ub = torch.zeros(1, H)

    for t in range(SEQ_LEN):
        xt_lb = lb[:, t*INPUT_SIZE:(t+1)*INPUT_SIZE]
        xt_ub = ub[:, t*INPUT_SIZE:(t+1)*INPUT_SIZE]

        gi_lb, gi_ub = _affine_ibp(Wih, bih,   xt_lb, xt_ub)
        gh_lb, gh_ub = _affine_ibp(Whh, zero4H, h_lb,  h_ub)
        g_lb = gi_lb + gh_lb;  g_ub = gi_ub + gh_ub

        def _sl(t, k): return t[:, k*H:(k+1)*H]

        i_lb, i_ub   = torch.sigmoid(_sl(g_lb,0)), torch.sigmoid(_sl(g_ub,0))
        f_lb, f_ub   = torch.sigmoid(_sl(g_lb,1)), torch.sigmoid(_sl(g_ub,1))
        g2_lb, g2_ub = torch.tanh(_sl(g_lb,2)),    torch.tanh(_sl(g_ub,2))
        o_lb, o_ub   = torch.sigmoid(_sl(g_lb,3)), torch.sigmoid(_sl(g_ub,3))

        fc_lb, fc_ub = _mul_ibp(f_lb, f_ub, c_lb, c_ub)
        ig_lb, ig_ub = _mul_ibp(i_lb, i_ub, g2_lb, g2_ub)
        c_lb = fc_lb + ig_lb;  c_ub = fc_ub + ig_ub

        tc_lb, tc_ub = torch.tanh(c_lb), torch.tanh(c_ub)
        h_lb, h_ub   = _mul_ibp(o_lb, o_ub, tc_lb, tc_ub)

    logit_lb, logit_ub = _affine_ibp(Wc, bc, h_lb, h_ub)

    # UNSAT iff true class lower-bound beats every other class upper-bound
    for j in range(NUM_CLASSES):
        if j == true_cls:
            continue
        if not (logit_lb[0, true_cls] > logit_ub[0, j]):
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# SAT instance construction — decision-boundary search
# ─────────────────────────────────────────────────────────────────────────────

def find_sat_instance(model: nn.Module, x0: np.ndarray, c0: int,
                      x1_pool: np.ndarray, n_bisect: int = 50,
                      delta: float = 0.02) -> tuple:
    """
    Guarantee a SAT instance by binary-searching for the decision boundary.

    1. Find nearest opposite-class sample x1 (in L-inf distance).
    2. Binary-search between x0 (class c0) and x1 (class c1 != c0).
    3. After convergence: a is just inside c0, b is just inside c1.
    4. Step delta further into c0 to get x_test.
    5. eps = 1.1 * L-inf(x_test, b) — the ball spans the boundary.

    The SAT witness is b: it is inside the ball and predicts a wrong class.
    """
    other = [j for j in range(NUM_CLASSES) if j != c0]

    def _pred(z):
        with torch.no_grad():
            return int(model(torch.tensor(z, dtype=torch.float32).unsqueeze(0))
                       .argmax(1))

    if _pred(x0) != c0:
        return None

    # Find nearest sample from any other class
    dists = np.abs(x1_pool - x0).max(axis=1)
    x1    = x1_pool[dists.argmin()]
    if _pred(x1) == c0:
        return None

    a, b = x0.copy(), x1.copy()
    for _ in range(n_bisect):
        mid = (a + b) / 2.0
        if _pred(mid) == c0:
            a = mid
        else:
            b = mid

    # Step deeper into c0 territory
    direction = x0 - x1
    linf_norm = float(np.abs(direction).max())
    if linf_norm < 1e-10:
        return None
    unit_dir = direction / linf_norm

    x_test = a + delta * unit_dir
    if _pred(x_test) != c0:
        x_test = a
        if _pred(x_test) != c0:
            return None

    eps = float(np.abs(x_test - b).max()) * 1.1
    eps = max(eps, 1e-5)

    if _pred(b) == c0:
        return None
    if float(np.abs(x_test - b).max()) > eps:
        return None

    return x_test, eps, c0


# ─────────────────────────────────────────────────────────────────────────────
# VNN-LIB 2.0 writer — multi-class disjunction
# ─────────────────────────────────────────────────────────────────────────────

def write_vnnlib(path: str, x: np.ndarray, eps: float, true_cls: int) -> None:
    """
    Write a VNN-LIB 2.0 robustness property for a 10-class classifier.

    SAT semantics (matches VNN-COMP convention):
        ∃ x' in [x-eps, x+eps] s.t. SOME wrong class beats the true class

    Output constraint is a disjunction over all 9 wrong classes:
        (assert (or
          (and (<= Y_true Y_0))   ; class 0 beats true class
          (and (<= Y_true Y_2))   ; class 2 beats true class
          ...                     ; (skip j == true_cls)
        ))
    """
    lb = x - eps
    ub = x + eps
    others = [j for j in range(NUM_CLASSES) if j != true_cls]

    with open(path, "w") as f:
        f.write(f"; psMNIST LSTM robustness property\n")
        f.write(f"; true_class={true_cls}  eps={eps:.6f}\n")
        f.write(f"; SAT = exists input in ball where a wrong digit wins\n\n")

        for i in range(INPUT_DIM):
            f.write(f"(declare-const X_{i} Real)\n")
        f.write("\n")
        for i in range(NUM_CLASSES):
            f.write(f"(declare-const Y_{i} Real)\n")
        f.write("\n")

        f.write("; Input: L-inf ball\n")
        for i in range(INPUT_DIM):
            f.write(f"(assert (>= X_{i} {lb[i]:.8f}))\n")
            f.write(f"(assert (<= X_{i} {ub[i]:.8f}))\n")
        f.write("\n")

        # Multi-class adversarial condition: any wrong class can win
        f.write("; Output: some wrong class beats the true class\n")
        f.write("(assert (or\n")
        for j in others:
            f.write(f"  (and (<= Y_{true_cls} Y_{j}))\n")
        f.write("))\n")


# ─────────────────────────────────────────────────────────────────────────────
# n2v dry run — Box reachability verification
# ─────────────────────────────────────────────────────────────────────────────

def run_n2v_dryrun(onnx_path, records: list, n_sample: int = 10) -> None:
    """
    Run a sample of instances through n2v's Box reachability to confirm labels.

    Box (interval arithmetic) is used because:
    - Star-set reachability for LSTMs is still an open problem in n2v
    - Box is sound: if the output Box certifies UNSAT, it is truly UNSAT
    - Box runs in seconds per instance

    For SAT instances, we use n2v's PGD falsification instead of reachability.
    """
    try:
        import sys
        sys.path.insert(0, "../..")
        sys.path.insert(0, "../../examples/VNN-COMP")
        from n2v.utils.model_loader import load_onnx
        from n2v.utils import load_vnnlib
        from n2v.utils.verify_specification import verify_specification
        from n2v.utils.falsify import falsify
        from n2v.nn.reach import reach_pytorch_model
        from n2v.sets import Box
    except ImportError as e:
        print(f"  [dryrun] n2v not importable from here: {e}")
        print(f"  [dryrun] Run manually: python ../../examples/VNN-COMP/run_instance.py <onnx> <vnnlib>")
        return

    print(f"\n── n2v dry run ({n_sample} instances) ──────────────────────────")

    loaded = {}
    def get_model(p):
        if p not in loaded:
            loaded[p] = load_onnx(p)
        return loaded[p]

    sample = records[:n_sample//2] + records[-n_sample//2:]
    ok = wrong = 0

    for onnx_r, vnnlib_r, expected, _ in sample:
        try:
            model = get_model(onnx_r)
            prop  = load_vnnlib(vnnlib_r)
            lb    = np.array(prop['lb'])
            ub    = np.array(prop['ub'])
            iset  = Box(lb.reshape(-1,1), ub.reshape(-1,1))
            out   = reach_pytorch_model(model, iset, method='approx')
            cert  = 'unsat' if verify_specification(out, prop['prop']) == 1 else 'open'

            if expected == 'sat':
                f = falsify(model, lb, ub, prop['prop'], method='pgd')
                final = 'sat' if f else cert
            else:
                final = cert

            status = '✓' if final == expected else '✗'
            print(f"  {status} {vnnlib_r}: expected={expected}  got={final}")
            if final == expected:
                ok += 1
            else:
                wrong += 1
        except Exception as e:
            print(f"  ? {vnnlib_r}: error — {e}")

    print(f"\n  dry run: {ok}/{len(sample)} correct  {wrong} wrong")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("onnx",   exist_ok=True)
    os.makedirs("vnnlib", exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("Loading psMNIST …")
    X_tr, y_tr, X_te, y_te, mu, sig = load_mnist()
    print(f"  train {X_tr.shape}  test {X_te.shape}")
    np.savez("norm_params.npz", mean=mu, std=sig, permutation=PERMUTATION)

    # ── Train two models ──────────────────────────────────────────────────────
    # Small model (hidden=8): compact enough for IBP to certify UNSAT.
    # Large model (hidden=64): realistic accuracy, used for SAT via falsification.
    #
    # Why two? IBP over-approximation in LSTMs compounds with every timestep
    # through the element-wise Mul gates (f*c, o*tanh(c)).  With hidden=64 and
    # 28 timesteps, the bounds widen too much to certify anything.  hidden=8
    # keeps the intermediate values small enough that IBP stays tight.

    print(f"\nTraining SMALL model (hidden={HIDDEN_SMALL}) for UNSAT instances …")
    model_small = psMNIST_LSTM(HIDDEN_SMALL)
    train(model_small, X_tr, y_tr)
    with torch.no_grad():
        Xt = torch.tensor(X_te, dtype=torch.float32)
        logits_s = model_small(Xt).numpy()
        preds_s  = logits_s.argmax(1)
    acc_s = (preds_s == y_te).mean()
    print(f"  test accuracy = {acc_s:.3f}  (lower is expected — tiny model)")

    print(f"\nTraining LARGE model (hidden={HIDDEN_LARGE}) for SAT instances …")
    model_large = psMNIST_LSTM(HIDDEN_LARGE)
    train(model_large, X_tr, y_tr)
    with torch.no_grad():
        logits_l = model_large(Xt).numpy()
        preds_l  = logits_l.argmax(1)
    acc_l = (preds_l == y_te).mean()
    print(f"  test accuracy = {acc_l:.3f}")

    # ── Export both models ────────────────────────────────────────────────────
    onnx_small = f"lstm_psMNIST_h{HIDDEN_SMALL}.onnx"
    onnx_large = f"lstm_psMNIST_h{HIDDEN_LARGE}.onnx"
    print(f"\nExporting ONNX …")
    export_onnx(model_small, f"onnx/{onnx_small}")
    export_onnx(model_large, f"onnx/{onnx_large}")

    # ── UNSAT: small model, high-confidence, IBP-certified ───────────────────
    margin_s = logits_s.max(1) - np.partition(logits_s, -2, axis=1)[:, -2]
    correct_s = preds_s == y_te
    pool_u    = np.where(correct_s)[0]
    pool_u    = pool_u[np.argsort(margin_s[pool_u])[::-1]]

    records  = []
    prop_idx = 0
    n_unsat  = 0

    print(f"\nGenerating UNSAT instances (small model, eps={EPS_UNSAT}) …")
    for idx in pool_u:
        if n_unsat >= N_UNSAT:
            break
        x  = X_te[idx]
        tc = int(y_te[idx])
        if not ibp_certify(model_small, x, EPS_UNSAT, tc):
            continue
        pname = f"prop_{prop_idx:03d}.vnnlib"
        write_vnnlib(f"vnnlib/{pname}", x, EPS_UNSAT, tc)
        records.append((f"onnx/{onnx_small}", f"vnnlib/{pname}", "unsat", TIMEOUT))
        prop_idx += 1
        n_unsat  += 1
        if n_unsat % 5 == 0:
            print(f"  {n_unsat}/{N_UNSAT} UNSAT certified …")

    print(f"  UNSAT: {n_unsat} instances generated")

    # ── SAT: large model, decision-boundary search ────────────────────────────
    correct_l = preds_l == y_te
    print(f"\nGenerating SAT instances (large model, boundary search) …")
    n_sat = 0

    for c0 in range(NUM_CLASSES):
        if n_sat >= N_SAT:
            break
        pool_c0        = X_te[np.where((y_te == c0) & correct_l)[0]]
        pool_all_other = X_te[np.where((y_te != c0) & correct_l)[0]]
        if len(pool_c0) == 0 or len(pool_all_other) == 0:
            continue
        for x in pool_c0[:N_SAT * 3]:
            if n_sat >= N_SAT:
                break
            result = find_sat_instance(model_large, x, c0, pool_all_other)
            if result is None:
                continue
            x_test, eps, true_cls = result
            pname = f"prop_{prop_idx:03d}.vnnlib"
            write_vnnlib(f"vnnlib/{pname}", x_test, eps, true_cls)
            records.append((f"onnx/{onnx_large}", f"vnnlib/{pname}", "sat", TIMEOUT))
            prop_idx += 1
            n_sat    += 1

    print(f"  SAT:   {n_sat} instances generated")

    # ── Write instance files ──────────────────────────────────────────────────
    with open("instances.csv", "w", newline="") as f:
        csv.writer(f).writerows([(r[0], r[1], r[3]) for r in records])

    with open("ground_truth.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["onnx", "vnnlib", "result", "timeout"])
        w.writerows(records)

    total = len(records)
    print(f"\n{'─'*55}")
    print(f"Total instances : {total}  ({n_unsat} UNSAT / {n_sat} SAT)")
    print(f"Models          : onnx/{onnx_small}  (UNSAT)  |  onnx/{onnx_large}  (SAT)")
    print(f"Properties      : vnnlib/  ({total} files)")
    print(f"Instance list   : instances.csv")
    print(f"Ground truth    : ground_truth.csv")

    if total < 50:
        print(f"\nWARNING: {total} < 50 — try increasing EPOCHS or reducing EPS_UNSAT.")
    else:
        print(f"\nBenchmark satisfies the ≥50 labeled instance requirement.")

    # ── n2v dry run ───────────────────────────────────────────────────────────
    if records:
        run_n2v_dryrun(None, records, n_sample=10)


if __name__ == "__main__":
    main()
