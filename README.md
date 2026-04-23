# CME202 Mathematics — Neural Network Learning Dynamics

**Minor Project | Mathematics II | Navrachana University, 2026**

> Deconstructing how a neural network learns using Linear Algebra, Differential Calculus, Fourier Analysis, and Numerical Optimisation — built from scratch with NumPy only.

---

## Group Members

| Name | Student ID |
|------|-----------|
| Harshal Vakhariya (Leader) | 25001144 |
| Meet Chauhan | 25001205 |
| Utsav Patel | 25001166 |
| Kathen Kale | 25000144 |
| Aum Nimkar | 25001057 |

**Faculty Guide:** Santosh Sadguru, Asst. Professor — School of Engineering and Technology

---

## What This Project Does

We train a neural network **from scratch** — NumPy only, no TensorFlow, no PyTorch — on the **XOR classification problem**, and show how every step maps to CME202 mathematics:

| Step | Operation | Maths Topic |
|------|-----------|-------------|
| Forward pass | `z = Wa + b`, `a = σ(z)` | **Linear Algebra** |
| Weight analysis | SVD, eigenvalues, condition number | **Linear Algebra** |
| Activation function | `σ(z) = 1/(1+e⁻ᶻ)` and its derivative | **Differential Calculus** |
| Backpropagation | Chain rule, layer by layer | **Differential Calculus** |
| Activation analysis | Fourier sine series of σ on [−π, π] | **Fourier Analysis** |
| Weight optimisation | Gradient descent / Newton–Raphson | **Numerical Methods** |

**Final result: 99.3% accuracy on XOR classification.**

---

## Project Structure

```
maths-project/
│
├── train.py          ← Main training script — generates training figures
├── math_plots.py     ← Mathematical analysis figures (sigmoid, SVD, Fourier, backprop)
├── dashboard.py      ← Live 5-panel animated training dashboard
│
├── assets/
│   └── navrachana_logo.png       ← University logo
│
├── results/          ← Auto-created when you run the scripts
│   ├── training_curves.png
│   ├── decision_boundary.png
│   ├── weight_gradient_dynamics.png
│   ├── nn_architecture.png
│   ├── sigmoid_and_derivative.png
│   ├── loss_landscape.png
│   ├── fourier_sigmoid.png
│   ├── svd_geometric.png
│   ├── backprop_graph.png
│   ├── dashboard_snapshot.png
│   └── nn_training_animation.mp4   (or .gif if FFmpeg not installed)
│
├── docs/
│   └── CME202_NeuralNetwork_Report.tex   ← Full LaTeX report
│
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate all figures in one go

```bash
python train.py        # Training figures (4 plots)
python math_plots.py   # Maths analysis figures (5 plots)
python dashboard.py --snapshot   # Dashboard PNG snapshot
```

### 3. Open the live training dashboard

```bash
python dashboard.py
```

Opens a real-time **5-panel dark dashboard** showing:
- Decision boundary evolving live
- Loss and accuracy curves
- Live bar charts of 20 sampled weight values
- Live bar charts of gradient magnitudes

### 4. Export the dashboard as a video

```bash
python dashboard.py --save
```

Saves `results/nn_training_animation.mp4` if FFmpeg is installed, or `results/nn_training_animation.gif` as a Pillow fallback.

### 5. Compile the LaTeX report

Copy figures to the `docs/` folder then:
```bash
cd docs
pdflatex CME202_NeuralNetwork_Report.tex
pdflatex CME202_NeuralNetwork_Report.tex   # run twice for TOC
```

Or just copy all `results/*.png` and `assets/navrachana_logo.png` into the same folder as the `.tex` file.

---

## Key Results

| Metric | Value |
|--------|-------|
| Network Architecture | 2 → 8 → 8 → 1 |
| Total Parameters | 105 |
| Training Epochs | 700 |
| Learning Rate η | 0.5 |
| Final Accuracy | **99.3%** |
| Loss at Convergence | ~0.046 |
| Condition number κ | ≈ 1.73 (well-conditioned) |
| Analytical minimum w* | ln(4) ≈ 1.386 (verified) |
| Dominant Fourier coefficient b₁ | ≈ 0.391 |

---

## Mathematics Summary

### Linear Algebra — Forward Pass + SVD

Each layer applies an affine transformation:
```
z[l] = W[l] * a[l-1] + b[l]        # matrix–vector product
a[l] = sigmoid(z[l])               # nonlinear activation
```

The weight matrix W is analysed via SVD: `W = U Σ Vᵀ`
- Singular values σ₁, σ₂ determine the condition number `κ = σ_max / σ_min`
- κ ≈ 1.73 → well-conditioned → stable training

### Differential Calculus — Backpropagation

Chain rule applied layer by layer:
```
δ[L]     = ŷ - y                                   # output error
δ[l]     = (W[l+1])ᵀ @ δ[l+1]  ⊙  σ'(z[l])       # propagate back
∂L/∂W[l] = δ[l] @ (a[l-1])ᵀ                       # weight gradient
W[l]    := W[l] - η * ∂L/∂W[l]                    # gradient descent update
```

### Fourier Analysis — Sigmoid as a Sine Series

Since σ(x) − 0.5 is odd, all cosine terms vanish:
```
σ(x) ≈ 0.5 + Σ bₙ sin(nx),   x ∈ [−π, π]
b₁ ≈ +0.391 (dominant)
b₂ ≈ −0.145
b₃ ≈ +0.098  ...
```
Rapid coefficient decay → sigmoid is a **low-pass function**.

### Numerical Methods — Gradient Descent Convergence

Analytical minimum of `L(w) = ½(σ(w) − 0.8)²`:
```
L'(w) = 0  →  σ(w*) = 0.8  →  w* = ln(4) ≈ 1.386
```
Gradient descent from w₀ = −2 converges to w* in ~20 iterations.
Newton–Raphson converges in ~5 iterations (quadratic convergence, higher cost).

---

## Dependencies

```
numpy >= 1.24
matplotlib >= 3.7
scipy >= 1.10
```

Standard scientific Python stack — **no ML frameworks required**.

Install with:
```bash
pip install -r requirements.txt
```

For MP4 export, also install [FFmpeg](https://ffmpeg.org/). Without it, `--save` falls back to GIF automatically.

---

## Credits

All coding, animation, mathematical modelling, and documentation (LaTeX report + README) were done by **Harshal Vakhariya**.

| Member | Contribution |
|--------|-------------|
| **Harshal Vakhariya** | Code, maths, dashboard, LaTeX report, GitHub |
| Utsav Patel | MOM (Memorandum of Meeting) maintainer |
| Meet Chauhan | PPT preparation |
| Kathen Kale | Poster design |
| Aum Nimkar | Research helper |

→ See [CREDITS.md](CREDITS.md) for full detailed credits.



📌 **AI Usage Note:** This project was built with my own effort, with AI used only for learning, debugging, and concept clarification. Full details: [PostScript (AI Usage Declaration)](./PostScript.md)
---

*CME202 — Mathematics II | School of Engineering and Technology | Navrachana University | 2026*
