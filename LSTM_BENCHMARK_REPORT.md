# LSTM Verification Benchmarks for VNN-COMP 2026
**VeriVITAL Lab — Summer 2026**
**Contributor:** Ibrahim Boudaoud
**Supervisor:** Prof. Taylor Johnson
**Repository:** github.com/ibrahimboudaoud/n2v

---

## 1. Motivation

The VNN-COMP competition (2023–2025) defines the benchmark landscape for neural network
verification. Every major architecture class has scored benchmarks — feedforward networks,
CNNs, ResNets, transformers, and vision models. One class is entirely absent: **recurrent
networks**. No LSTM, GRU, or RNN benchmark has appeared in VNN-COMP 2023, 2024, or 2025.

This is not because recurrent networks are unimportant. LSTMs are deployed in safety-critical
settings including surgical robotics, autonomous vehicles, and medical monitoring. The gap
exists because of a format problem: LSTM computation involves element-wise gate multiplications
and recurrent state that do not map cleanly onto the ONNX + VNN-LIB 2.0 format that
VNN-COMP requires.

This project fills that gap. We designed, implemented, and verified two LSTM benchmarks in
full VNN-COMP format, and identified the open algorithmic problems that future work must solve
to enable tight formal certification of recurrent networks.

---

## 2. Technical Background

### 2.1 The LSTM Cell

An LSTM processes a sequence one timestep at a time, maintaining two state vectors: a hidden
state `h` (working memory) and a cell state `c` (long-term memory). At each timestep `t`,
given input `x_t` and previous states `h_prev`, `c_prev`:

```
gates  =  W_ih · x_t  +  b  +  W_hh · h_prev

i  =  σ(gates[0:H])          input gate  — how much new info to write
f  =  σ(gates[H:2H])         forget gate — how much old memory to keep
g  = tanh(gates[2H:3H])      cell gate   — what new info to write
o  =  σ(gates[3H:4H])        output gate — how much memory to expose

c_new  =  f ⊙ c_prev  +  i ⊙ g
h_new  =  o ⊙ tanh(c_new)
```

The cell state update `c = f·c + i·g` uses addition rather than multiplicative squashing,
allowing gradients to flow across long sequences without vanishing.

### 2.2 The Unrolling Strategy

Standard `nn.LSTM` exports to ONNX as a single LSTM operator that no VNN-COMP verifier
supports. Our approach: implement the LSTM cell using explicit gate operations and a Python
loop. When exported with `torch.onnx.export(dynamo=False)`, the loop is traced and unrolled
— the ONNX graph contains only `Gemm`, `Add`, `Sigmoid`, `Tanh`, `Mul`, and `Slice`
operators. Every feed-forward verifier (α,β-CROWN, Marabou, n2v) can consume these models
without modification.

### 2.3 Verification Property

Both benchmarks use **L-∞ robustness** as the verification property, matching the standard
used in ACAS Xu and MNIST feedforward benchmarks:

```
For test input x* and radius ε:
  UNSAT = for all x' ∈ [x*−ε, x*+ε], model(x') = model(x*)    (robust)
  SAT   = ∃ x' ∈ [x*−ε, x*+ε] such that model(x') ≠ model(x*) (not robust)
```

In VNN-LIB 2.0 format, this is encoded as: input constraints define the ε-ball; output
constraints define the adversarial condition (wrong class wins).

---

## 3. Benchmark 1 — kinematics_lstm

### 3.1 Data

Synthetic 2-D point-mass trajectories. State vector `[x, y, vx, vy]` at each timestep.
Dynamics: `x(t+1) = x(t) + 0.1·vx(t)`, `vx(t+1) = a·vx(t) + N(0, 0.05)`.

- Stable trajectories: damping factor `a ∈ [0.70, 0.95]` — label 1
- Unstable trajectories: amplification factor `a ∈ [1.05, 1.30]` — label 0
- Label rule: stable iff all `|x|, |y| ≤ 3.0` across all timesteps

Training set: 1,600 samples. Test set: 400 samples. Z-score normalised.

### 3.2 Architecture

Three LSTM models with hidden sizes 8, 16, and 32. All share:
- Input: `(1, 40)` — 10 timesteps × 4 features, flattened
- Output: `(1, 2)` — binary logits (stable / unstable)
- Sequence length: 10
- Training: Adam, lr=0.001, batch=64, 80 epochs, cross-entropy loss
- Test accuracy: ~99.8% across all three sizes

### 3.3 Instances

| Model | UNSAT instances | SAT instances |
|---|---|---|
| hidden=8 | 12 | 8 |
| hidden=16 | 12 | 8 |
| hidden=32 | 12 | 8 |
| **Total** | **36** | **24** |

**Total: 60 instances**

UNSAT instances: IBP-certified at ε=0.005. Selected from highest-margin correctly-classified
test samples.

SAT instances: constructed by binary-searching to the decision boundary (50 bisections)
between pairs of opposite-class test samples. The SAT witness `b` is placed just past the
boundary at distance δ=0.015; ε = 1.1 × ‖x_test − b‖_∞ ≈ 0.0165.

**Verification result: 60/60 labels confirmed by n2v** (Box reachability for UNSAT, PGD
falsification for SAT).

---

## 4. Benchmark 2 — psMNIST_lstm

### 4.1 Data

Permuted Sequential MNIST (psMNIST) — the standard recurrent network benchmark from
Bai et al. (2018), arXiv:1803.01271. Real MNIST data: 60,000 training images, 10,000 test
images. Preprocessing:

1. Flatten 28×28 image → 784 pixels, scale to [0, 1]
2. Apply fixed random permutation (seed=42, stored in `norm_params.npz`)
3. Truncate to first 392 values (14 rows × 28 pixels per row)
4. Z-score normalise using training-set per-feature statistics

The permutation destroys spatial locality, requiring the model to learn genuine long-range
dependencies. Sequence length was reduced from 28 to 14 rows after discovering that IBP
certification fails for any hidden size when the sequence exceeds ~14–16 steps — an empirical
finding that defines the open research problem (see Section 6).

### 4.2 Architecture

Two LSTM models, each with sequence length 14, input size 28, output 10 classes:

| Model | Hidden size | Test accuracy | Purpose |
|---|---|---|---|
| lstm_psMNIST_h8 | 8 | 83% | UNSAT instances (IBP-certifiable) |
| lstm_psMNIST_h64 | 64 | 91% | SAT instances (PGD-falsifiable) |

Training: Adam, lr=0.001, batch=256, 40 epochs, cross-entropy loss.

### 4.3 VNN-LIB Property (10-class)

Multi-class robustness requires a disjunction over 9 adversarial classes:

```
(assert (or
  (and (<= Y_true Y_0))
  (and (<= Y_true Y_1))
  ...                        ; skip true class
  (and (<= Y_true Y_9))
))
```

SAT = some wrong digit can beat the true digit within the ε-ball.

### 4.4 Instances

| Type | Count | Model | Method | ε |
|---|---|---|---|---|
| UNSAT | 25 | hidden=8 | IBP certification | 0.005 |
| SAT | 25 | hidden=64 | Boundary search | ~0.02 |
| **Total** | **50** | | | |

**Verification result: 50/50 labels confirmed by n2v** (Box reachability for UNSAT, PGD
falsification for SAT).

---

## 5. Contributions to n2v

Three issues were discovered and resolved during benchmark development:

### 5.1 OnnxFunction handler in reach.py (+19 lines)

**Problem:** onnx2torch converts the ONNX Tanh operator to `OnnxFunction(torch.tanh)` rather
than `nn.Tanh`. n2v's reach engine had no handler for `OnnxFunction`, causing silent failure
on any ONNX model containing Tanh — including all LSTM models.

**Fix:** Added a handler in `_handle_graphmodule` that inspects the wrapped function, maps it
to the equivalent `nn.Module` via a lookup table, and dispatches to the existing handler.
This fix benefits any future ONNX benchmark with Tanh or Sigmoid activations.

### 5.2 Constant MatMul folding in generate.py

**Problem:** At timestep 0, `W_hh @ h_0 = W_hh @ zeros = zeros`. This appears as a MatMul
node in the ONNX graph with two constant inputs. n2v's MatMul handler only handles
`MatMul(set, constant)` — it returns None when both inputs are constants.

**Fix:** Post-export preprocessing in `generate.py` evaluates any MatMul where both inputs
are already-known constants, stores the result as a new constant initializer, and removes the
node. Mathematically identical; verifier-compatible.

### 5.3 Identity op removal in generate.py

**Problem:** PyTorch's ONNX exporter deduplicates identical zero tensors with Identity ops
(`c0 = Identity(h0)`). onnx2torch converts these to `OnnxCopyIdentity`, which n2v's reach
engine does not handle. The initial cell state `c0` was never registered in `node_values`,
breaking all downstream gate computations.

**Fix:** Post-export preprocessing rewires all Identity connections and removes the nodes.

---

## 6. Open Problems Discovered

### 6.1 Star-set addition across independent predicate spaces

n2v's `_add_sets` for Star sets assumes both operands share the same predicate variables —
valid for residual connections but not for LSTM gate sums, where `W_ih(x_t)` and `W_hh(h)`
come from computation branches with independent predicate histories.

The correct operation is the **Minkowski sum**: concatenate generator matrices rather than
adding them. This is unimplemented in n2v.

### 6.2 Predicate explosion under Minkowski sums

Even with a correct Minkowski sum implementation, predicate count grows by `n_b` at every
gate summation. Across 14 timesteps with 3 sigmoid and 2 tanh activations each (each
introducing `hidden_size` new predicates), the predicate count reaches O(timesteps ×
hidden_size × activations_per_step). LP calls — one per neuron per activation — operate over
this growing variable space, making star-set certification computationally intractable for
realistic hidden sizes.

### 6.3 IBP compounding through gate multiplications

IBP certification fails for sequences longer than ~14 steps even with hidden=8. The element-
wise multiplications `f ⊙ c` and `o ⊙ tanh(c)` each grow the interval width through the
four-corner product bound. Over 28 steps, the output intervals span the full possible range
regardless of ε. This sets a hard limit on IBP-based certification for longer-horizon
recurrent models.

---

## 7. Future Goals

### Near term (remaining summer)

**7.1 Open the pull request**
Submit both benchmarks to `sammsaski/n2v` as a PR for lab review. The commit history,
ground_truth.csv files, and dry-run results provide complete documentation.

**7.2 Test against an external verifier**
Run the benchmark instances through α,β-CROWN or Marabou to confirm the ONNX models parse
correctly in an independent verifier. This is required for a VNN-COMP 2026 submission.
Contact Samuel Sasaki or Hongchao Zhang for access to lab machines with these tools installed.

**7.3 Adding problem benchmark (Benchmark 3)**
Following PI direction, formalize the adding problem from Bai et al. (2018) as a third
benchmark. This is a regression task (output = sum of two marked sequence values), requiring
a new VNN-LIB property type: **output range bounding** rather than classification robustness.
The property would be: for all inputs in ε-ball, the predicted sum stays within δ of the
true prediction. This property type does not exist in any current VNN-COMP benchmark.

### Medium term (fall / publication)

**7.4 Fix `_add_sets` for Minkowski sum**
Implement the correct Minkowski sum for Star sets in `n2v/nn/reach.py`. This removes the
shape mismatch error that currently prevents star-set reachability from running on LSTM
models at all. Verify that the implementation is sound against the IBP-certified UNSAT
instances.

**7.5 Address predicate explosion**
Investigate strategies to control predicate growth:
- *Abstraction-refinement*: start with Box, switch to Star only for the final timesteps
- *Predicate merging*: merge predicates from different branches using LP-based projection
- *CROWN-style propagation*: bound propagation without explicit predicate variables

This is the core open research problem for LSTM verification and is suitable for a workshop
paper at SAIV, ARCH, or FORMATS.

**7.6 Full 28-step psMNIST certification**
Once predicate explosion is addressed, revisit psMNIST with the full 28-step sequence and
hidden=64 model. The goal is to certify robustness at ε=0.005 for the realistic model — a
result that would be publishable as the first formal LSTM robustness certificate on a standard
ML benchmark.

**7.7 JIGSAWS kinematic data (P3 crossover)**
Apply the kinematics benchmark methodology to real da Vinci surgical robot data once the
JIGSAWS data use agreement is approved. The ε-ball radius would be calibrated to actual
encoder noise, giving the UNSAT certificates physical meaning: "the classifier is immune to
sensor noise of this magnitude."

### Long term (VNN-COMP 2026 submission)

**7.8 Package for official VNN-COMP submission**
Both benchmarks satisfy the format requirements: ONNX opset 14, VNN-LIB 2.0, ≥50 labeled
instances, instances.csv. The remaining step is a one-page benchmark description and a
GitHub repository structured per VNN-COMP 2026 submission guidelines. Contact the VNN-COMP
organizers (see Kaulen et al., arXiv:2512.19007) for submission deadlines and format
requirements.

---

## 8. Summary

| | kinematics_lstm | psMNIST_lstm |
|---|---|---|
| Data | Synthetic physics | Real MNIST |
| Task | Binary classification | 10-class digit recognition |
| Models | 3 (hidden 8/16/32) | 2 (hidden 8, 64) |
| Instances | 60 (36 UNSAT, 24 SAT) | 50 (25 UNSAT, 25 SAT) |
| UNSAT method | IBP, ε=0.005 | IBP, ε=0.005 |
| SAT method | Boundary search | Boundary search |
| Verified by n2v | 60/60 ✓ | 50/50 ✓ |
| ONNX ops | Gemm/Sigmoid/Tanh/Mul/Slice | Gemm/Sigmoid/Tanh/Mul/Slice |
| n2v fix required | Yes (OnnxFunction) | Yes (OnnxFunction) |

Both benchmarks are committed to `github.com/ibrahimboudaoud/n2v` and ready for PR submission
to the lab repository. Together they constitute the first LSTM benchmarks in VNN-COMP format,
closing a three-year gap in the competition's architecture coverage.

---

*Generated from work completed June 2026. All artifacts reproducible by running
`python generate.py` in the respective benchmark folder.*
