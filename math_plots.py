"""
CME202 - Mathematics II | Minor Project
========================================
Mathematical Analysis Figures
Generates all maths-related plots used in the report:
  1. Sigmoid function and its derivative
  2. Loss landscape with gradient descent steps
  3. Fourier series approximation of sigmoid
  4. SVD geometric interpretation
  5. Backpropagation computation graph

Usage:
    python math_plots.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from scipy.integrate import quad
import os

os.makedirs("results", exist_ok=True)

# ── Shared colour palette ─────────────────────────────────────────────────────
C_RED    = "#E63946"
C_BLUE   = "#457B9D"
C_TEAL   = "#2A9D8F"
C_AMBER  = "#E9C46A"
C_PURPLE = "#9B5DE5"
C_BG     = "#F8F9FA"
C_DARK   = "#1D3557"

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.22,
    "grid.linestyle":     "--",
    "axes.facecolor":     C_BG,
    "figure.facecolor":   "white",
    "axes.labelcolor":    C_DARK,
    "xtick.color":        C_DARK,
    "ytick.color":        C_DARK,
    "axes.titleweight":   "bold",
    "axes.titlesize":     12,
})

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


# ─────────────────────────────────────────────────────────────
# Figure 1 — Sigmoid + Derivative (improved)
# ─────────────────────────────────────────────────────────────

z = np.linspace(-6, 6, 500)
sig   = sigmoid(z)
dsig  = sig * (1 - sig)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Sigmoid Activation Function and Its Derivative",
             fontsize=13, color=C_DARK, fontweight="bold")

# Left: sigmoid
ax = axes[0]
ax.plot(z, sig, color=C_BLUE, lw=2.8, label=r"$\sigma(z)$", zorder=4)
ax.axhline(0.5, color="gray", ls=":", lw=1.2, alpha=0.7)
ax.axhline(1.0, color="gray", ls=":", lw=1.0, alpha=0.5)
ax.axhline(0.0, color="gray", ls=":", lw=1.0, alpha=0.5)
ax.axvline(0,   color="gray", ls=":", lw=1.2, alpha=0.7)
ax.scatter([0], [0.5], color=C_RED, s=100, zorder=6, label=r"$\sigma(0)=0.5$")
ax.fill_between(z, 0, sig, alpha=0.08, color=C_BLUE)

# Saturation regions
ax.axvspan(-6, -3, alpha=0.06, color=C_RED, label="Saturation zone")
ax.axvspan( 3,  6, alpha=0.06, color=C_RED)
ax.text(-4.5, 0.08, "Saturation\n(gradient ≈ 0)", ha="center",
        fontsize=8, color=C_RED, style="italic")
ax.text( 4.5, 0.92, "Saturation\n(gradient ≈ 0)", ha="center",
        fontsize=8, color=C_RED, style="italic")

ax.set_title(r"$\sigma(z) = \dfrac{1}{1+e^{-z}}$", fontsize=12)
ax.set_xlabel("$z$"); ax.set_ylabel(r"$\sigma(z)$")
ax.set_ylim(-0.08, 1.12)
ax.legend(fontsize=9)

# Right: derivative
ax = axes[1]
ax.plot(z, dsig, color=C_TEAL, lw=2.8, label=r"$\sigma'(z)$", zorder=4)
ax.fill_between(z, 0, dsig, alpha=0.10, color=C_TEAL)
ax.axhline(0.25, color="gray", ls=":", lw=1.2, alpha=0.7)
ax.axvline(0,    color="gray", ls=":", lw=1.2, alpha=0.7)
ax.scatter([0], [0.25], color=C_RED, s=100, zorder=6,
           label=r"$\sigma'(0)=0.25$ (maximum)")

# Annotate vanishing gradient
ax.annotate("Vanishing gradient\nregion", xy=(-4.5, sigmoid(-4.5)*(1-sigmoid(-4.5))),
            xytext=(-4, 0.08),
            arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.4),
            fontsize=8, color=C_RED, style="italic")

ax.set_title(r"$\sigma'(z) = \sigma(z)(1-\sigma(z))$", fontsize=12)
ax.set_xlabel("$z$"); ax.set_ylabel(r"$\sigma'(z)$")
ax.set_ylim(-0.02, 0.32)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("results/sigmoid_and_derivative.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved: results/sigmoid_and_derivative.png")


# ─────────────────────────────────────────────────────────────
# Figure 2 — Loss Landscape + Gradient Descent Steps (improved)
# ─────────────────────────────────────────────────────────────

def loss_fn(w):
    return 0.5 * (sigmoid(w) - 0.8)**2

def loss_grad(w):
    s = sigmoid(w)
    return (s - 0.8) * s * (1 - s)

# Gradient descent trace
w = -2.0; lr = 0.5
gd_ws = [w]
for _ in range(25):
    w = w - lr * loss_grad(w)
    gd_ws.append(w)
gd_ws = np.array(gd_ws)

# Newton-Raphson trace (for comparison)
def loss_hess(w):
    s = sigmoid(w)
    sd = s*(1-s)
    return sd**2 + (s-0.8)*sd*(1-2*s)

nr_ws = [-2.0]
for _ in range(8):
    g = loss_grad(nr_ws[-1]); h = loss_hess(nr_ws[-1])
    if abs(h) < 1e-10: break
    nr_ws.append(nr_ws[-1] - g/h)
nr_ws = np.array(nr_ws)

ww = np.linspace(-4, 4, 500)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Loss Landscape and Gradient Descent Convergence",
             fontsize=13, color=C_DARK, fontweight="bold")

# Left: landscape + GD trajectory
ax = axes[0]
ax.plot(ww, loss_fn(ww), color=C_BLUE, lw=2.5,
        label=r"$\mathcal{L}(w) = \frac{1}{2}(\sigma(w)-0.8)^2$")
ax.fill_between(ww, loss_fn(ww), alpha=0.07, color=C_BLUE)
ax.axvline(np.log(4), color=C_TEAL, ls="--", lw=1.8,
           label=r"$w^* = \ln 4 \approx 1.386$")

# GD steps
ax.scatter(gd_ws, loss_fn(gd_ws), color=C_RED, s=50, zorder=5)
ax.plot(gd_ws, loss_fn(gd_ws), color=C_RED, lw=1.2, ls="--",
        alpha=0.7, label="Gradient descent steps")

ax.annotate("Start $w_0 = -2$",
            xy=(gd_ws[0], loss_fn(gd_ws[0])),
            xytext=(-3.0, 0.22),
            arrowprops=dict(arrowstyle="->", color=C_RED),
            fontsize=9, color=C_RED)
ax.annotate(f"$w^* \\approx {np.log(4):.3f}$",
            xy=(gd_ws[-1], loss_fn(gd_ws[-1])),
            xytext=(0.2, 0.19),
            arrowprops=dict(arrowstyle="->", color=C_TEAL),
            fontsize=9, color=C_TEAL)

ax.set_xlabel("$w$"); ax.set_ylabel("Loss $\\mathcal{L}(w)$")
ax.set_title("Loss Surface with GD Trajectory")
ax.legend(fontsize=9)

# Right: convergence speed comparison (GD vs Newton)
ax = axes[1]
gd_losses = loss_fn(gd_ws)
nr_losses  = loss_fn(nr_ws)

ax.semilogy(range(len(gd_losses)), gd_losses, "o-",
            color=C_RED, lw=2, ms=6, label=f"Gradient Descent ({len(gd_ws)-1} steps)")
ax.semilogy(range(len(nr_losses)), nr_losses,  "s-",
            color=C_TEAL, lw=2, ms=7, label=f"Newton–Raphson ({len(nr_ws)-1} steps)")
ax.axhline(1e-6, color="gray", ls=":", lw=1, label="Tolerance $10^{-6}$")

ax.set_xlabel("Iteration"); ax.set_ylabel("Loss (log scale)")
ax.set_title("Convergence Comparison: GD vs Newton–Raphson")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("results/loss_landscape.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved: results/loss_landscape.png")


# ─────────────────────────────────────────────────────────────
# Figure 3 — Fourier Series Approximation (improved)
# ─────────────────────────────────────────────────────────────

L = np.pi

def fourier_bn(n, L=np.pi):
    integrand = lambda x: sigmoid(x) * np.sin(n * np.pi * x / L)
    val, _ = quad(integrand, -L, L)
    return val / L

x = np.linspace(-np.pi, np.pi, 500)

fig = plt.figure(figsize=(14, 5))
gs  = fig.add_gridspec(1, 3, width_ratios=[2, 1, 1], wspace=0.40)
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])
ax2 = fig.add_subplot(gs[2])

fig.suptitle(r"Fourier Sine Series Approximation of $\sigma(x)$ on $[-\pi, \pi]$",
             fontsize=13, color=C_DARK, fontweight="bold")

# Panel A: approximation curves
ax0.plot(x, sigmoid(x), "k-", lw=3, label=r"Exact $\sigma(x)$", zorder=10)

colors_f = [C_RED, C_AMBER, C_TEAL, C_PURPLE]
Ns = [1, 3, 5, 10]
bns_all = [fourier_bn(n) for n in range(1, 11)]

for N, col in zip(Ns, colors_f):
    approx = 0.5 + sum(bns_all[n-1] * np.sin(n * x) for n in range(1, N+1))
    err = np.max(np.abs(approx - sigmoid(x)))
    ax0.plot(x, approx, "--", lw=1.9, color=col, alpha=0.9,
             label=f"N={N}  (max err={err:.3f})")

ax0.set_title("Approximation Quality vs Number of Terms")
ax0.set_xlabel("$x$"); ax0.set_ylabel(r"$\sigma(x)$")
ax0.legend(fontsize=8)

# Panel B: coefficient bar chart
ns   = list(range(1, 11))
babs = [abs(b) for b in bns_all]
bar_cols = [C_BLUE if b > 0 else C_RED for b in bns_all]
ax1.bar(ns, babs, color=[C_BLUE if b>0 else C_RED for b in bns_all],
        edgecolor="white", lw=0.8, width=0.7)
ax1.set_title("Coefficient Magnitudes $|b_n|$")
ax1.set_xlabel("Harmonic $n$"); ax1.set_ylabel("$|b_n|$")
ax1.set_xticks(ns)
ax1.annotate(f"$b_1 \\approx {babs[0]:.3f}$\n(dominant)",
             xy=(1, babs[0]), xytext=(3, babs[0]*0.85),
             arrowprops=dict(arrowstyle="->", color=C_DARK),
             fontsize=8, color=C_DARK)

# Panel C: cumulative error decay
errs = []
for N in range(1, 11):
    approx = 0.5 + sum(bns_all[n-1] * np.sin(n * x) for n in range(1, N+1))
    errs.append(np.max(np.abs(approx - sigmoid(x))))

ax2.semilogy(range(1, 11), errs, "o-", color=C_TEAL, lw=2, ms=7)
ax2.set_title("Max Approx Error vs $N$")
ax2.set_xlabel("Number of terms $N$"); ax2.set_ylabel("Max |Error| (log)")
ax2.set_xticks(range(1, 11))

plt.tight_layout()
plt.savefig("results/fourier_sigmoid.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved: results/fourier_sigmoid.png")

print("\nFourier Sine Coefficients b_n:")
print(f"{'n':>4}  {'b_n':>10}  {'|b_n|':>8}")
for n in range(1, 6):
    b = fourier_bn(n)
    print(f"{n:>4}  {b:>10.4f}  {abs(b):>8.4f}")


# ─────────────────────────────────────────────────────────────
# Figure 4 — SVD Geometric Interpretation (improved)
# ─────────────────────────────────────────────────────────────

W = np.array([[0.5, -0.3],
              [0.2,  0.8],
              [-0.1, 0.4]])

U, S, Vt = np.linalg.svd(W, full_matrices=False)
print(f"\nSVD of W^[1]:")
print(f"  Singular values: σ1={S[0]:.4f}, σ2={S[1]:.4f}")
print(f"  Condition number κ = {S[0]/S[1]:.4f}")

theta   = np.linspace(0, 2*np.pi, 400)
circle  = np.array([np.cos(theta), np.sin(theta)])
W2      = W[:2, :]
ellipse = W2 @ circle

# Color the circle/ellipse by angle
colors_angle = plt.cm.hsv(theta / (2*np.pi))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("SVD Geometric Interpretation — Weight Matrix $\\mathbf{W}^{[1]}$ Transformation",
             fontsize=13, color=C_DARK, fontweight="bold")

# Panel A: unit circle (input space)
ax = axes[0]
for i in range(len(theta)-1):
    ax.plot(circle[0, i:i+2], circle[1, i:i+2], color=colors_angle[i], lw=2.5)
ax.set_aspect("equal")
ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
ax.axhline(0, color="gray", lw=0.7); ax.axvline(0, color="gray", lw=0.7)
ax.set_title("Input Space\nUnit Circle $\\|\\mathbf{v}\\|=1$")
ax.set_xlabel("$v_1$"); ax.set_ylabel("$v_2$")

# Right singular vectors
for i, (color, label) in enumerate(zip([C_RED, C_BLUE],
                                        ["$\\mathbf{v}_1$", "$\\mathbf{v}_2$"])):
    ax.annotate("", xy=Vt[i], xytext=(0,0),
                arrowprops=dict(arrowstyle="->", color=color, lw=2.2))
    ax.text(Vt[i,0]*1.2, Vt[i,1]*1.2, label, color=color, fontsize=11, ha="center")

ax.text(0.05, -1.45, "Right singular vectors $\\mathbf{V}^T$",
        fontsize=8, color="gray", style="italic")

# Panel B: ellipse (output space)
ax = axes[1]
for i in range(len(theta)-1):
    ax.plot(ellipse[0, i:i+2], ellipse[1, i:i+2], color=colors_angle[i], lw=2.5)
ax.set_aspect("equal")
pad = 0.3
ax.set_xlim(ellipse[0].min()-pad, ellipse[0].max()+pad)
ax.set_ylim(ellipse[1].min()-pad, ellipse[1].max()+pad)
ax.axhline(0, color="gray", lw=0.7); ax.axvline(0, color="gray", lw=0.7)
ax.set_title(f"Output Space\nEllipse  $\\kappa = {S[0]/S[1]:.2f}$")
ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$")

# Semi-axes from U and S
for i, (color, label) in enumerate(zip([C_RED, C_BLUE],
                                        [f"$\\sigma_1={S[0]:.3f}$", f"$\\sigma_2={S[1]:.3f}$"])):
    end = S[i] * U[:2, i]
    ax.annotate("", xy=end, xytext=(0,0),
                arrowprops=dict(arrowstyle="->", color=color, lw=2.2))
    ax.text(end[0]*1.25, end[1]*1.25, label, color=color, fontsize=9, ha="center")

# Panel C: condition number interpretation bar
ax = axes[2]
ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 10)

info_text = [
    ("SVD Decomposition", 9.0, C_DARK, 11, True),
    (r"$\mathbf{W}^{[1]} = \mathbf{U}\,\boldsymbol{\Sigma}\,\mathbf{V}^T$", 8.0, C_DARK, 11, False),
    ("", 7.4, C_DARK, 9, False),
    (f"Singular value $\\sigma_1 = {S[0]:.4f}$", 7.0, C_RED, 10, False),
    (f"Singular value $\\sigma_2 = {S[1]:.4f}$", 6.3, C_BLUE, 10, False),
    ("", 5.8, C_DARK, 9, False),
    (f"Condition number $\\kappa = {S[0]/S[1]:.4f}$", 5.4, C_TEAL, 11, True),
    ("", 4.9, C_DARK, 9, False),
    ("$\\kappa \\approx 1.73$  →  Well-conditioned", 4.5, C_TEAL, 9, False),
    ("Gradients flow uniformly →", 4.0, C_DARK, 9, False),
    ("stable training confirmed!", 3.6, C_DARK, 9, False),
    ("", 3.0, C_DARK, 9, False),
    ("Rule of thumb:", 2.6, "gray", 9, True),
    ("$\\kappa < 10$  →  Stable", 2.1, C_TEAL, 9, False),
    ("$\\kappa > 100$  →  May diverge", 1.6, C_RED, 9, False),
]
for text, y, color, size, bold in info_text:
    ax.text(5, y, text, ha="center", va="center",
            fontsize=size, color=color,
            fontweight="bold" if bold else "normal")

ax.add_patch(mpatches.FancyBboxPatch((0.3, 0.4), 9.4, 9.2,
             boxstyle="round,pad=0.2",
             facecolor="#f0f4f8", edgecolor=C_BLUE, lw=1.5))
ax.set_title("Analysis Summary", pad=10)

plt.tight_layout()
plt.savefig("results/svd_geometric.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved: results/svd_geometric.png")


# ─────────────────────────────────────────────────────────────
# Figure 5 — Backpropagation Computation Graph (improved)
# ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(-0.2, 10.5); ax.set_ylim(-1.5, 5)
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.set_title("Backpropagation: Chain Rule Applied Layer by Layer",
             fontsize=13, fontweight="bold", color=C_DARK, pad=18)

nodes = {
    "x":    (0.8, 2.0),
    "W1":   (0.8, 3.5),
    "z1":   (3.0, 2.0),
    "a1":   (5.0, 2.0),
    "W2":   (5.0, 3.5),
    "z2":   (7.0, 2.0),
    "yhat": (8.8, 2.0),
    "L":    (8.8, 0.3),
}
labels = {
    "x":    "$\\mathbf{x}$",
    "W1":   "$\\mathbf{W}^{[1]}$",
    "z1":   "$\\mathbf{z}^{[1]}$",
    "a1":   "$\\mathbf{a}^{[1]}$",
    "W2":   "$\\mathbf{W}^{[2]}$",
    "z2":   "$z^{[2]}$",
    "yhat": "$\\hat{y}$",
    "L":    "$\\mathcal{L}$",
}
node_col = {
    "x":    "#D8F3DC", "W1": "#BDE0FE", "z1": "#FFF3BF",
    "a1":   "#FFF3BF", "W2": "#BDE0FE", "z2": "#FFF3BF",
    "yhat": "#FFD6A5", "L": "#FFADAD",
}
node_ec = {
    "x": C_TEAL, "W1": C_BLUE, "z1": C_AMBER, "a1": C_AMBER,
    "W2": C_BLUE, "z2": C_AMBER, "yhat": "#F4A261", "L": C_RED,
}
R = 0.42

for key, (px, py) in nodes.items():
    circ = plt.Circle((px, py), R, color=node_col[key],
                       ec=node_ec[key], lw=2.5, zorder=4)
    ax.add_patch(circ)
    ax.text(px, py, labels[key], ha="center", va="center",
            fontsize=11, zorder=5, color=C_DARK)

# Operation labels between nodes
def mid(a, b):
    return ((nodes[a][0]+nodes[b][0])/2, (nodes[a][1]+nodes[b][1])/2)

op_labels = {
    ("x","z1"):   ("z=Wx+b", 2.1, 2.45),
    ("W1","z1"):  ("", 0, 0),
    ("z1","a1"):  ("$\\sigma(\\cdot)$", 4.0, 2.35),
    ("a1","z2"):  ("z=Wa+b", 6.1, 2.45),
    ("W2","z2"):  ("", 0, 0),
    ("z2","yhat"):("$\\sigma(\\cdot)$", 7.9, 2.35),
    ("yhat","L"): ("CE Loss", 8.6, 1.15),
}

# Forward edges
fwd_edges = [("x","z1"), ("W1","z1"), ("z1","a1"),
             ("a1","z2"), ("W2","z2"), ("z2","yhat"), ("yhat","L")]
for src, dst in fwd_edges:
    x1,y1 = nodes[src]; x2,y2 = nodes[dst]
    dx = x2-x1; dy = y2-y1; dist = np.sqrt(dx**2+dy**2)
    sx = x1 + R*dx/dist; ex = x2 - R*dx/dist
    sy = y1 + R*dy/dist; ey = y2 - R*dy/dist
    ax.annotate("", xy=(ex,ey), xytext=(sx,sy),
                arrowprops=dict(arrowstyle="-|>", color=C_DARK, lw=1.6,
                                mutation_scale=14))

for (src,dst), (lbl, lx, ly) in op_labels.items():
    if lbl:
        ax.text(lx, ly, lbl, ha="center", va="bottom", fontsize=8.5,
                color=C_DARK,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="#cccccc", alpha=0.85))

# Backward gradient arrows (below)
bwd = [
    ("L", "yhat", "$\\frac{\\partial\\mathcal{L}}{\\partial\\hat{y}}$"),
    ("yhat","z2", "$\\delta^{[2]}$"),
    ("z2","a1",   "$\\delta^{[2]}(\\mathbf{W}^{[2]})^T$"),
    ("a1","z1",   "$\\delta^{[1]}$"),
    ("z1","W1",   "$\\frac{\\partial\\mathcal{L}}{\\partial\\mathbf{W}^{[1]}}$"),
]
Y_BACK = 0.3
for src, dst, lbl in bwd:
    x1,y1 = nodes[src]; x2,y2 = nodes[dst]
    ys = y1 - 1.35; yd = y2 - 1.35
    ax.annotate("", xy=(x2+R*0.9, yd), xytext=(x1-R*0.9, ys),
                arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=1.8,
                                mutation_scale=14,
                                connectionstyle="arc3,rad=0.0"))
    mx = (x1+x2)/2
    ax.text(mx, min(ys,yd)-0.30, lbl, ha="center", va="top",
            fontsize=8, color=C_RED,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#fff0f0",
                      edgecolor=C_RED, alpha=0.85, lw=0.8))

# Legend
fwd_patch = mpatches.Patch(color=C_DARK, label="Forward pass (data)")
bwd_patch = mpatches.Patch(color=C_RED,  label="Backward pass (gradients)")
ax.legend(handles=[fwd_patch, bwd_patch], loc="lower left",
          fontsize=9, framealpha=0.9)

# Layer labels at top
for name, lx, label in [("Input", 0.8, "Input Layer"),
                          ("Hidden 1", 3.0, "Layer 1"),
                          ("Hidden 2", 5.8, "Layer 2"),
                          ("Output", 8.8, "Output")]:
    ax.text(lx, 4.7, label, ha="center", va="bottom",
            fontsize=9, color="gray", style="italic")

plt.tight_layout()
plt.savefig("results/backprop_graph.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved: results/backprop_graph.png")

print("\nAll math figures saved to results/")
print("Run train.py to generate the training figures.")
