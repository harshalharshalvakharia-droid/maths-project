"""
CME202 - Mathematics II | Minor Project
========================================
XOR Neural Network — Full Training Script
Trains a [2 → 8 → 8 → 1] network from scratch using only NumPy.
Generates all training figures used in the report.

Usage:
    python train.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from scipy.integrate import quad
import os

np.random.seed(42)
os.makedirs("results", exist_ok=True)

# ── Colour palette (consistent across all figures) ───────────────────────────
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
    "grid.alpha":         0.25,
    "grid.linestyle":     "--",
    "axes.facecolor":     C_BG,
    "figure.facecolor":   "white",
    "axes.labelcolor":    C_DARK,
    "xtick.color":        C_DARK,
    "ytick.color":        C_DARK,
    "axes.titleweight":   "bold",
    "axes.titlesize":     12,
})

# ─────────────────────────────────────────────────────────────
# Core math
# ─────────────────────────────────────────────────────────────

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

def make_xor(n=300, noise=0.35):
    X = np.random.randn(n, 2) * noise
    y = np.zeros(n)
    X[:n//4]       += [ 1,  1];  y[:n//4]       = 1   # top-right:    class 1
    X[n//4:n//2]   += [-1, -1];  y[n//4:n//2]   = 1   # bottom-left:  class 1
    X[n//2:3*n//4] += [ 1, -1];  y[n//2:3*n//4] = 0   # bottom-right: class 0
    X[3*n//4:]     += [-1,  1];  y[3*n//4:]     = 0   # top-left:     class 0
    return X, y


class NeuralNet:
    def __init__(self, sizes):
        self.W = [np.random.randn(sizes[i+1], sizes[i]) * np.sqrt(2/sizes[i])
                  for i in range(len(sizes)-1)]
        self.b = [np.zeros((sizes[i+1], 1)) for i in range(len(sizes)-1)]

    def forward(self, X):
        self.zs = []; self.acts = [X.T]; a = X.T
        for w, b in zip(self.W, self.b):
            z = w @ a + b
            self.zs.append(z); a = sigmoid(z); self.acts.append(a)
        return a

    def loss(self, X, y):
        yh = self.forward(X).flatten(); eps = 1e-12
        return -np.mean(y * np.log(yh + eps) + (1-y) * np.log(1-yh + eps))

    def accuracy(self, X, y):
        return np.mean((self.forward(X).flatten() >= 0.5) == y) * 100

    def train_step(self, X, y, lr):
        m = X.shape[0]; yh = self.acts[-1]
        d = (yh - y.reshape(1,-1)) / m
        for i in reversed(range(len(self.W))):
            dW = d @ self.acts[i].T
            db = np.sum(d, axis=1, keepdims=True)
            if i > 0:
                d = (self.W[i].T @ d) * sigmoid_deriv(self.zs[i-1])
            self.W[i] -= lr * dW
            self.b[i]  -= lr * db


# ─────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────

X, y = make_xor(300)
net  = NeuralNet([2, 8, 8, 1])

loss_h, acc_h, snapshots = [], [], {}
EPOCHS = 700

for epoch in range(EPOCHS):
    loss_h.append(net.loss(X, y))
    acc_h.append(net.accuracy(X, y))
    if epoch in [0, 50, 200, 699]:
        snapshots[epoch] = ([w.copy() for w in net.W], [b.copy() for b in net.b])
    net.train_step(X, y, lr=0.5)

print(f"Final Accuracy : {acc_h[-1]:.1f}%")
print(f"Final Loss     : {loss_h[-1]:.4f}")


# ─────────────────────────────────────────────────────────────
# Figure 1 — Training Curves (dual-panel, styled)
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Neural Network Training Progress  ·  XOR Problem  ·  [2→8→8→1]",
             fontsize=13, color=C_DARK, fontweight="bold", y=1.02)

# Loss
ax = axes[0]
ax.plot(loss_h, color=C_RED, lw=2.2, label="Cross-Entropy Loss")
ax.fill_between(range(EPOCHS), loss_h, alpha=0.08, color=C_RED)
ax.set_title("Training Loss vs Epoch")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss  $\\mathcal{L}$")
ax.annotate(f"Final: {loss_h[-1]:.4f}",
            xy=(EPOCHS-1, loss_h[-1]),
            xytext=(500, loss_h[-1]+0.1),
            arrowprops=dict(arrowstyle="->", color=C_RED),
            color=C_RED, fontsize=9)

# Accuracy
ax = axes[1]
ax.plot(acc_h, color=C_BLUE, lw=2.2, label="Accuracy")
ax.fill_between(range(EPOCHS), acc_h, 40, alpha=0.08, color=C_BLUE)
ax.axhline(99, color=C_AMBER, ls="--", lw=1.4, label="99 % mark")
ax.set_title("Accuracy vs Epoch")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
ax.set_ylim(40, 105)
ax.legend(fontsize=9)
ax.annotate(f"{acc_h[-1]:.1f}%",
            xy=(EPOCHS-1, acc_h[-1]),
            xytext=(500, acc_h[-1]-8),
            arrowprops=dict(arrowstyle="->", color=C_BLUE),
            color=C_BLUE, fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("results/training_curves.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved: results/training_curves.png")


# ─────────────────────────────────────────────────────────────
# Figure 2 — Decision Boundary Evolution (4 snapshots, heatmap style)
# ─────────────────────────────────────────────────────────────

xx, yy = np.meshgrid(np.linspace(-2.5, 2.5, 250), np.linspace(-2.5, 2.5, 250))
grid   = np.c_[xx.ravel(), yy.ravel()]

# Custom diverging colormap: red→white→blue
cmap_db = LinearSegmentedColormap.from_list(
    "xor", ["#E63946", "#FFFFFF", "#457B9D"], N=256)

fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
fig.suptitle("Decision Boundary Evolution During Training",
             fontsize=13, color=C_DARK, fontweight="bold")

for ax, (ep, (Ws, bs)) in zip(axes, sorted(snapshots.items())):
    tmp = NeuralNet([2, 8, 8, 1])
    tmp.W = Ws; tmp.b = bs
    Z = tmp.forward(grid).flatten().reshape(xx.shape)
    acc_snap = np.mean((tmp.forward(X).flatten() >= 0.5) == y) * 100

    im = ax.contourf(xx, yy, Z, levels=60, cmap=cmap_db, alpha=0.85)
    ax.contour(xx, yy, Z, levels=[0.5], colors=C_DARK, linewidths=2.2)
    ax.scatter(X[y==0, 0], X[y==0, 1], c=C_RED,  s=15, edgecolors="white",
               lw=0.4, zorder=5, label="Class 0")
    ax.scatter(X[y==1, 0], X[y==1, 1], c=C_BLUE, s=15, edgecolors="white",
               lw=0.4, zorder=5, label="Class 1")

    ax.set_title(f"Epoch {ep}", fontsize=11)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")

    # Accuracy badge
    col = C_TEAL if acc_snap >= 90 else (C_AMBER if acc_snap >= 60 else C_RED)
    ax.text(0.97, 0.03, f"{acc_snap:.0f}%",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=12, fontweight="bold", color=col,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=col, alpha=0.85))

axes[0].legend(fontsize=8, loc="upper left",
               facecolor="white", edgecolor="#cccccc")
plt.tight_layout()
plt.savefig("results/decision_boundary.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved: results/decision_boundary.png")


# ─────────────────────────────────────────────────────────────
# Figure 3 — Weight & Gradient Dynamics
# ─────────────────────────────────────────────────────────────

net2 = NeuralNet([2, 8, 8, 1])
np.random.seed(42)
w_means, g_means, w_stds = [], [], []

for epoch in range(EPOCHS):
    net2.forward(X)
    all_w = np.concatenate([w.flatten() for w in net2.W])
    w_means.append(np.mean(np.abs(all_w)))
    w_stds.append(np.std(all_w))
    m = X.shape[0]; yh = net2.acts[-1]; d = (yh - y.reshape(1,-1)) / m
    gs = []
    for i in reversed(range(len(net2.W))):
        gs.append(np.mean(np.abs(d @ net2.acts[i].T)))
        if i > 0:
            d = (net2.W[i].T @ d) * sigmoid_deriv(net2.zs[i-1])
    g_means.append(np.mean(gs))
    net2.train_step(X, y, lr=0.5)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Weight and Gradient Dynamics During Training",
             fontsize=13, color=C_DARK, fontweight="bold")

ax = axes[0]
ax.plot(w_means, color=C_BLUE, lw=2.2, label="Mean |weight|")
ax.fill_between(range(EPOCHS),
                np.array(w_means) - np.array(w_stds),
                np.array(w_means) + np.array(w_stds),
                alpha=0.15, color=C_BLUE, label="±1 std dev")
ax.set_title("Weight Magnitude over Training")
ax.set_xlabel("Epoch"); ax.set_ylabel("Mean |W|")
ax.legend(fontsize=9)

ax = axes[1]
ax.semilogy(g_means, color=C_AMBER, lw=2.2)
ax.fill_between(range(EPOCHS), g_means, 1e-6, alpha=0.12, color=C_AMBER)
ax.set_title("Gradient Magnitude  (log scale)")
ax.set_xlabel("Epoch"); ax.set_ylabel("Mean |Gradient|")
ax.annotate("Gradients → 0\nat convergence",
            xy=(EPOCHS-50, g_means[-50]),
            xytext=(400, g_means[100]*2),
            arrowprops=dict(arrowstyle="->", color=C_AMBER),
            color=C_AMBER, fontsize=9)

plt.tight_layout()
plt.savefig("results/weight_gradient_dynamics.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved: results/weight_gradient_dynamics.png")


# ─────────────────────────────────────────────────────────────
# Figure 4 — Neural Network Architecture Diagram
# ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, 10); ax.set_ylim(-0.5, 9.5)
ax.axis("off")
fig.patch.set_facecolor("white")

layer_sizes  = [2, 8, 8, 1]
layer_labels = ["Input\nLayer", "Hidden\nLayer 1", "Hidden\nLayer 2", "Output\nLayer"]
layer_colors = [C_TEAL, C_BLUE, C_BLUE, C_RED]
xs = [1.5, 3.8, 6.2, 8.5]

node_positions = {}
for li, (n, lx) in enumerate(zip(layer_sizes, xs)):
    ys = np.linspace(9 - (9 - (9 - n)) / 2,
                     (9 - (9 - n)) / 2 + 0.2,
                     n) if n > 1 else [4.5]
    # evenly space
    if n > 1:
        ys = np.linspace(1, 8, n)
    else:
        ys = [4.5]
    node_positions[li] = list(zip([lx]*n, ys))

# Draw connections (only a subset for clarity — not all 8×2 + 8×8 lines)
for li in range(len(layer_sizes)-1):
    for (x1, y1) in node_positions[li]:
        for (x2, y2) in node_positions[li+1]:
            ax.plot([x1, x2], [y1, y2], color="#cccccc", lw=0.5, alpha=0.6, zorder=1)

# Draw nodes
for li, (color, label, lx) in enumerate(zip(layer_colors, layer_labels, xs)):
    for (nx, ny) in node_positions[li]:
        circ = plt.Circle((nx, ny), 0.32, color=color, zorder=3, ec="white", lw=1.5)
        ax.add_patch(circ)
        sym = "$x$" if li == 0 else ("$\\sigma$" if li < 3 else "$\\hat{y}$")
        ax.text(nx, ny, sym, ha="center", va="center",
                fontsize=8, color="white", fontweight="bold", zorder=4)
    # Layer label below
    ax.text(lx, -0.3, label, ha="center", va="top",
            fontsize=10, color=C_DARK, fontweight="bold")
    # Dimension annotation
    dims = [f"$\\mathbf{{x}} \\in \\mathbb{{R}}^2$",
            f"$\\mathbf{{z}}^{{[1]}} \\in \\mathbb{{R}}^8$",
            f"$\\mathbf{{z}}^{{[2]}} \\in \\mathbb{{R}}^8$",
            f"$\\hat{{y}} \\in [0,1]$"]
    ax.text(lx, 8.9, dims[li], ha="center", va="bottom",
            fontsize=8, color=color, style="italic")

# Weight matrix annotations between layers
for li, (lx, rx, wdim) in enumerate(zip(xs[:-1], xs[1:],
        ["$\\mathbf{W}^{[1]} \\in \\mathbb{R}^{8\\times2}$",
         "$\\mathbf{W}^{[2]} \\in \\mathbb{R}^{8\\times8}$",
         "$\\mathbf{W}^{[3]} \\in \\mathbb{R}^{1\\times8}$"])):
    mx = (lx + rx) / 2
    ax.text(mx, 9.3, wdim, ha="center", va="bottom",
            fontsize=9, color="#888888",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#f0f0f0",
                      edgecolor="#cccccc", alpha=0.9))

ax.set_title("Neural Network Architecture  ·  [2 → 8 → 8 → 1]  ·  Sigmoid Activation",
             fontsize=13, fontweight="bold", color=C_DARK, pad=18)

plt.tight_layout()
plt.savefig("results/nn_architecture.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved: results/nn_architecture.png")

print("\nAll training figures saved to results/")
print("Run math_plots.py to generate the mathematical analysis figures.")
