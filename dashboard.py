"""
CME202 - Mathematics II | Minor Project
========================================
Live Training Dashboard — Real-time 5-panel visualization
Run this to see the neural network learning live!

Usage:
    python dashboard.py              # shows live window
    python dashboard.py --save       # saves animation (MP4 if FFmpeg found, else GIF)
    python dashboard.py --snapshot   # saves a static PNG snapshot of the final state
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import sys
import os
import shutil

np.random.seed(42)
os.makedirs("results", exist_ok=True)

# ── Design tokens ─────────────────────────────────────────────────────────────
BG_MAIN   = "#0F1923"   # deep navy
BG_PANEL  = "#1A2736"   # panel bg
BG_CARD   = "#21303F"   # slightly lighter
ACCENT1   = "#00D4FF"   # cyan
ACCENT2   = "#FF6B6B"   # coral red
ACCENT3   = "#2ECC71"   # emerald green
ACCENT4   = "#F39C12"   # amber
ACCENT5   = "#9B59B6"   # purple
TEXT_PRI  = "#ECF0F1"   # almost white
TEXT_SEC  = "#95A5A6"   # muted
BORDER    = "#2C3E50"   # border colour

# Custom colourmap for decision boundary
cmap_db = LinearSegmentedColormap.from_list(
    "xor_dark", ["#FF6B6B", "#1A2736", "#00D4FF"], N=512)


# ─────────────────────────────────────────────────────────────
# Core math (same as train.py)
# ─────────────────────────────────────────────────────────────

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

def make_xor_dataset(n=200, noise=0.35):
    X = np.random.randn(n, 2) * noise
    y = np.zeros(n)
    X[:n//4]       += [ 1,  1];  y[:n//4]       = 1
    X[n//4:n//2]   += [-1, -1];  y[n//4:n//2]   = 1
    X[n//2:3*n//4] += [ 1, -1];  y[n//2:3*n//4] = 0
    X[3*n//4:]     += [-1,  1];  y[3*n//4:]     = 0
    return X, y


class NeuralNetwork:
    def __init__(self, sizes):
        self.W = [np.random.randn(sizes[i+1], sizes[i]) * np.sqrt(2/sizes[i])
                  for i in range(len(sizes)-1)]
        self.b = [np.zeros((sizes[i+1], 1)) for i in range(len(sizes)-1)]
        self.layer_sizes = sizes

    def forward(self, X):
        self.zs = []; self.acts = [X.T]; a = X.T
        for w, b in zip(self.W, self.b):
            z = w @ a + b; self.zs.append(z); a = sigmoid(z); self.acts.append(a)
        return a

    def loss(self, X, y):
        yh = self.forward(X).flatten(); e = 1e-12
        return -np.mean(y * np.log(yh+e) + (1-y) * np.log(1-yh+e))

    def acc(self, X, y):
        return np.mean((self.forward(X).flatten() >= 0.5) == y) * 100

    def backward(self, X, y, lr):
        m = X.shape[0]; yh = self.acts[-1]; d = (yh - y.reshape(1,-1)) / m
        for i in reversed(range(len(self.W))):
            dW = d @ self.acts[i].T; db = np.sum(d, axis=1, keepdims=True)
            if i > 0:
                d = (self.W[i].T @ d) * sigmoid_deriv(self.zs[i-1])
            self.W[i] -= lr * dW; self.b[i] -= lr * db

    def weights_flat(self):
        return np.concatenate([w.flatten() for w in self.W])

    def grads_flat(self, X, y):
        self.forward(X); m = X.shape[0]; yh = self.acts[-1]
        d = (yh - y.reshape(1,-1)) / m; gs = []
        for i in reversed(range(len(self.W))):
            gs.append((d @ self.acts[i].T).flatten())
            if i > 0:
                d = (self.W[i].T @ d) * sigmoid_deriv(self.zs[i-1])
        return np.abs(np.concatenate(gs[::-1]))


# ─────────────────────────────────────────────────────────────
# Pre-train and collect snapshots
# ─────────────────────────────────────────────────────────────

print("Pre-training network and collecting snapshots...")
X, y   = make_xor_dataset(200)
net    = NeuralNetwork([2, 8, 8, 1])
EPOCHS = 500
loss_h, acc_h, snaps = [], [], []

for ep in range(EPOCHS):
    loss_h.append(net.loss(X, y))
    acc_h.append(net.acc(X, y))
    if ep % 5 == 0:
        snaps.append({
            "ep"  : ep,
            "W"   : [w.copy() for w in net.W],
            "b"   : [b.copy() for b in net.b],
            "loss": loss_h[-1],
            "acc" : acc_h[-1],
            "wv"  : net.weights_flat().copy(),
            "gv"  : net.grads_flat(X, y).copy(),
        })
    net.backward(X, y, lr=0.5)

print(f"Done. {len(snaps)} frames. Final accuracy: {acc_h[-1]:.1f}%")

# Grid for decision boundary
xx, yy = np.meshgrid(np.linspace(-2.5, 2.5, 150),
                      np.linspace(-2.5, 2.5, 150))
grid = np.c_[xx.ravel(), yy.ravel()]


# ─────────────────────────────────────────────────────────────
# Build the figure (dark, premium dashboard)
# ─────────────────────────────────────────────────────────────

plt.rcParams.update({"font.family": "DejaVu Sans"})

fig = plt.figure(figsize=(16, 9), facecolor=BG_MAIN)
gs  = GridSpec(2, 3, figure=fig,
               hspace=0.48, wspace=0.40,
               left=0.05, right=0.97, top=0.88, bottom=0.08)

ax_db   = fig.add_subplot(gs[:, 0])   # decision boundary — full height
ax_loss = fig.add_subplot(gs[0, 1])   # loss curve
ax_acc  = fig.add_subplot(gs[1, 1])   # accuracy curve
ax_w    = fig.add_subplot(gs[0, 2])   # weight bars
ax_g    = fig.add_subplot(gs[1, 2])   # gradient bars

def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(BG_PANEL)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.tick_params(colors=TEXT_SEC, labelsize=8)
    ax.xaxis.label.set_color(TEXT_SEC)
    ax.yaxis.label.set_color(TEXT_SEC)
    ax.set_title(title, color=TEXT_PRI, fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, color=BORDER, alpha=0.6, linestyle="--", linewidth=0.7)

style_ax(ax_db,   "Decision Boundary",        "$x_1$", "$x_2$")
style_ax(ax_loss, "Training Loss",             "Epoch", "Cross-Entropy")
style_ax(ax_acc,  "Training Accuracy",         "Epoch", "Accuracy (%)")
style_ax(ax_w,    "Weight Values (live)",      "Index", "Value")
style_ax(ax_g,    "Gradient Magnitudes (live)","Index", "|Gradient|")

# Main title
fig.text(0.5, 0.95,
         "CME202  ·  Neural Network Training Dashboard  ·  XOR Classification",
         ha="center", color=TEXT_PRI, fontsize=14, fontweight="bold")
fig.text(0.5, 0.91,
         "Architecture: [2 → 8 → 8 → 1]  ·  Batch Gradient Descent  ·  η = 0.5",
         ha="center", color=TEXT_SEC, fontsize=9)

# ── Decision boundary panel ───────────────────────────────────────────────────
ax_db.scatter(X[y==0,0], X[y==0,1], c=ACCENT2, s=28,
              edgecolors="white", lw=0.5, zorder=5, label="Class 0")
ax_db.scatter(X[y==1,0], X[y==1,1], c=ACCENT1, s=28,
              edgecolors="white", lw=0.5, zorder=5, label="Class 1")
ax_db.set_xlim(-2.5, 2.5); ax_db.set_ylim(-2.5, 2.5)
ax_db.set_aspect("equal")
ax_db.legend(fontsize=8, loc="upper right",
             labelcolor=TEXT_PRI, facecolor=BG_CARD, edgecolor=BORDER)

ep_txt   = ax_db.text(0.04, 0.97, "", transform=ax_db.transAxes,
                      color=TEXT_PRI, fontsize=11, fontweight="bold", va="top")
acc_badge = ax_db.text(0.96, 0.97, "", transform=ax_db.transAxes,
                        color=ACCENT3, fontsize=11, fontweight="bold",
                        va="top", ha="right")
img_obj = ax_db.imshow(np.zeros((150, 150)),
                        extent=[-2.5, 2.5, -2.5, 2.5],
                        origin="lower", cmap=cmap_db,
                        vmin=0, vmax=1, alpha=0.75, zorder=1)
contour_store = [None]

# ── Loss curve ────────────────────────────────────────────────────────────────
ax_loss.set_xlim(0, EPOCHS); ax_loss.set_ylim(0, max(loss_h)*1.1)
loss_line, = ax_loss.plot([], [], color=ACCENT2, lw=2.4)
loss_fill  = ax_loss.fill_between([], [], alpha=0.12, color=ACCENT2)
loss_dot,  = ax_loss.plot([], [], "o", color=ACCENT2, ms=7, zorder=5)
loss_val   = ax_loss.text(0.97, 0.87, "", transform=ax_loss.transAxes,
                           color=ACCENT2, fontsize=9, ha="right",
                           fontweight="bold")

# ── Accuracy curve ────────────────────────────────────────────────────────────
ax_acc.set_xlim(0, EPOCHS); ax_acc.set_ylim(35, 105)
ax_acc.axhline(100, color=BORDER, lw=1, ls="--")
acc_line, = ax_acc.plot([], [], color=ACCENT3, lw=2.4)
acc_dot,  = ax_acc.plot([], [], "o", color=ACCENT3, ms=7, zorder=5)
acc_val   = ax_acc.text(0.97, 0.10, "", transform=ax_acc.transAxes,
                         color=ACCENT3, fontsize=10, ha="right",
                         fontweight="bold")

# ── Weight bars ───────────────────────────────────────────────────────────────
NB = 20
x_idx = range(NB)
bars_w = ax_w.bar(x_idx, [0]*NB, color=ACCENT1,
                   edgecolor=BG_MAIN, lw=0.7, width=0.75)
ax_w.set_ylim(-2.5, 2.5)
ax_w.axhline(0, color=TEXT_SEC, lw=0.8)
ax_w.set_xticks(range(0, NB, 4))

# ── Gradient bars ─────────────────────────────────────────────────────────────
bars_g = ax_g.bar(x_idx, [0]*NB, color=ACCENT4,
                   edgecolor=BG_MAIN, lw=0.7, width=0.75)
ax_g.set_ylim(0, 0.15)
ax_g.set_xticks(range(0, NB, 4))


# ─────────────────────────────────────────────────────────────
# Animation update
# ─────────────────────────────────────────────────────────────

loss_xs, loss_ys, acc_ys = [], [], []

def update(frame):
    global loss_fill
    snap = snaps[frame]
    ep = snap["ep"]; lv = snap["loss"]; av = snap["acc"]
    wv = snap["wv"]; gv = snap["gv"]

    # Decision boundary
    tmp = NeuralNetwork([2, 8, 8, 1])
    tmp.W = [w.copy() for w in snap["W"]]
    tmp.b = [b.copy() for b in snap["b"]]
    Z = tmp.forward(grid).flatten().reshape(150, 150)
    img_obj.set_data(Z)

    if contour_store[0] is not None:
        try:
            contour_store[0].remove()
        except Exception:
            try:
                for c in contour_store[0].collections:
                    c.remove()
            except Exception:
                pass
    cs = ax_db.contour(xx, yy, Z, levels=[0.5],
                        colors="white", linewidths=2.5, zorder=3)
    contour_store[0] = cs

    ep_txt.set_text(f"Epoch  {ep:>3d}")
    acc_badge.set_text(f"Acc  {av:.1f}%")
    col = ACCENT3 if av >= 90 else (ACCENT4 if av >= 70 else ACCENT2)
    acc_badge.set_color(col)

    # Loss + accuracy curves
    loss_xs.append(ep); loss_ys.append(lv); acc_ys.append(av)
    loss_line.set_data(loss_xs, loss_ys)
    loss_dot.set_data([ep], [lv])
    loss_val.set_text(f"Loss = {lv:.4f}")

    acc_line.set_data(loss_xs, acc_ys)
    acc_dot.set_data([ep], [av])
    acc_val.set_text(f"{av:.1f}%")

    # Weights
    ws = wv[:NB]
    ylim = max(2.5, float(np.abs(ws).max()) * 1.35)
    ax_w.set_ylim(-ylim, ylim)
    for bar, val in zip(bars_w, ws):
        bar.set_height(float(val))
        bar.set_color(ACCENT1 if val >= 0 else ACCENT2)

    # Gradients
    gs16 = gv[:NB]
    gmax = max(float(gs16.max()) * 1.35, 0.005)
    ax_g.set_ylim(0, gmax)
    for bar, val in zip(bars_g, gs16):
        bar.set_height(float(val))
        norm = float(val) / gmax
        # Colour interpolates amber → purple with magnitude
        r = int(243 * (1-norm) + 155 * norm)
        g = int(156 * (1-norm) +  89 * norm)
        b = int( 18 * (1-norm) + 182 * norm)
        bar.set_color(f"#{r:02x}{g:02x}{b:02x}")

    return (list(bars_w) + list(bars_g) +
            [loss_line, loss_dot, acc_line, acc_dot,
             ep_txt, acc_badge, loss_val, acc_val, img_obj])


# ─────────────────────────────────────────────────────────────
# Run, save, or snapshot
# ─────────────────────────────────────────────────────────────

ani = animation.FuncAnimation(
    fig, update,
    frames=len(snaps),
    interval=70,
    blit=False,
    repeat=True
)

SAVE_MODE     = "--save"     in sys.argv
SNAPSHOT_MODE = "--snapshot" in sys.argv

if SNAPSHOT_MODE:
    # Render final state as a static PNG — useful for the report / README
    print("Rendering final-state snapshot...")
    update(len(snaps) - 1)
    out = "results/dashboard_snapshot.png"
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=BG_MAIN)
    print(f"Saved snapshot: {out}")

elif SAVE_MODE:
    ffmpeg_available = shutil.which("ffmpeg") is not None

    if ffmpeg_available:
        out = "results/nn_training_animation.mp4"
        print(f"FFmpeg found — saving MP4 → {out}")
        writer = animation.FFMpegWriter(
            fps=14, bitrate=2400,
            extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
        ani.save(out, writer=writer, dpi=120, savefig_kwargs={"facecolor": BG_MAIN})
        print(f"Saved MP4: {out}")
    else:
        print("FFmpeg not found — falling back to GIF (Pillow)...")
        try:
            import PIL  # noqa
        except ImportError:
            print("Pillow not installed! Run: pip install pillow"); sys.exit(1)
        out = "results/nn_training_animation.gif"
        ani.save(out, writer="pillow", fps=14, dpi=80,
                 savefig_kwargs={"facecolor": BG_MAIN})
        print(f"Saved GIF: {out}")
        print("Tip: install FFmpeg for MP4 output.")

else:
    print("Showing live dashboard. Use --save to export video, --snapshot for PNG.")
    plt.show()
