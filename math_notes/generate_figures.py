#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate illustrative figures for diffusion math teaching notes."""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D

OUT = "/workspace/diffusion-notes/figures"
os.makedirs(OUT, exist_ok=True)

# Prefer Chinese-capable fonts
FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/workspace/fonts/wqy-microhei.ttf",
]
font_path = None
for p in FONT_CANDIDATES:
    if os.path.exists(p):
        font_path = p
        break

if font_path:
    font_manager.fontManager.addfont(font_path)
    # For TTC, prop name may vary
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False
    CN = prop
else:
    CN = None

def set_cn(ax_or_fig_artists=None):
    """Apply Chinese font to text objects if needed."""
    pass

# Color palette
C_PRIMARY = "#2563eb"
C_NOISE = "#64748b"
C_SIGNAL = "#059669"
C_PRED = "#dc2626"
C_BG = "#f8fafc"
C_ACCENT = "#7c3aed"

# --------------------------------------------------------------------------
# Figure 1: Forward noise schedule — signal vs noise strength vs t
# --------------------------------------------------------------------------
def fig1_forward_schedule():
    T = 1000
    t = np.arange(1, T + 1)
    # linear beta schedule (classic DDPM)
    beta_start, beta_end = 1e-4, 0.02
    beta = np.linspace(beta_start, beta_end, T)
    alpha = 1.0 - beta
    alphabar = np.cumprod(alpha)
    signal = np.sqrt(alphabar)
    noise = np.sqrt(1.0 - alphabar)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    fig.patch.set_facecolor("white")

    ax = axes[0]
    ax.plot(t, beta, color=C_PRIMARY, lw=2, label=r"$\beta_t$ (linear)")
    ax.set_xlabel("timestep $t$", fontsize=11)
    ax.set_ylabel(r"$\beta_t$", fontsize=11)
    ax.set_title("噪声方差日程 β schedule", fontproperties=CN, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    ax = axes[1]
    ax.plot(t, signal, color=C_SIGNAL, lw=2.2, label=r"信号系数 $\sqrt{\bar{\alpha}_t}$")
    ax.plot(t, noise, color=C_NOISE, lw=2.2, label=r"噪声系数 $\sqrt{1-\bar{\alpha}_t}$")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("timestep $t$", fontsize=11)
    ax.set_ylabel("系数大小", fontproperties=CN, fontsize=11)
    ax.set_title(r"$q(x_t|x_0)$ 中信号与噪声的相对强度", fontproperties=CN, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right", prop=CN if CN else None)

    # annotate key points
    for tt in [1, 200, 500, 999]:
        axes[1].axvline(tt, color="#cbd5e1", ls="--", lw=0.8)

    fig.suptitle("图1  前向过程：加噪强度随 t 变化", fontproperties=CN, fontsize=13, y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT, "fig1_forward_schedule.png")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path

# --------------------------------------------------------------------------
# Figure 2: 1D mixture gradually becoming Gaussian
# --------------------------------------------------------------------------
def fig2_1d_gaussianization():
    rng = np.random.default_rng(42)
    # mixture of two Gaussians as "data"
    n = 8000
    half = n // 2
    x0 = np.concatenate([
        rng.normal(-2.0, 0.35, half),
        rng.normal(2.0, 0.45, half),
    ])

    betas = [0.0, 0.05, 0.2, 0.5]  # cumulative-ish noise levels via alphabar
    # Use closed form: x_t = sqrt(abar) x0 + sqrt(1-abar) eps
    abars = [1.0, 0.85, 0.4, 0.05]
    titles = [
        r"$t=0$（数据）",
        r"$t$ 较小",
        r"$t$ 中等",
        r"$t \to T$（近标准正态）",
    ]

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2), sharey=True)
    fig.patch.set_facecolor("white")
    xs = np.linspace(-5, 5, 400)

    for ax, abar, title in zip(axes, abars, titles):
        eps = rng.normal(0, 1, n)
        xt = np.sqrt(abar) * x0 + np.sqrt(1 - abar) * eps
        ax.hist(xt, bins=60, density=True, color=C_PRIMARY, alpha=0.55, edgecolor="none")
        # overlay N(0,1) for reference when noisy
        
        pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * xs ** 2)
        ax.plot(xs, pdf, color=C_PRED, lw=1.5, ls="--", label=r"$\mathcal{N}(0,1)$")
        ax.set_xlim(-5, 5)
        ax.set_title(title, fontproperties=CN, fontsize=10)
        ax.set_xlabel("$x$", fontsize=10)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("密度", fontproperties=CN, fontsize=10)
    axes[-1].legend(fontsize=8, loc="upper right")
    fig.suptitle("图2  一维混合分布在前向加噪下逐渐变为高斯", fontproperties=CN, fontsize=13, y=1.05)
    fig.tight_layout()
    path = os.path.join(OUT, "fig2_1d_gaussianization.png")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path

# --------------------------------------------------------------------------
# Figure 3: Training schematic — network predicts noise
# --------------------------------------------------------------------------
def fig3_training_schematic():
    fig, ax = plt.subplots(figsize=(10, 4.2))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x, y, w, h, text, color, fs=10):
        r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
                           facecolor=color, edgecolor="#334155", lw=1.2, alpha=0.92)
        ax.add_patch(r)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontproperties=CN, color="#0f172a", wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#475569", lw=1.8))

    # Row of process
    box(0.3, 3.2, 1.8, 1.1, "干净数据\n$x_0 \\sim q(x_0)$", "#bbf7d0", 9)
    arrow(2.2, 3.75, 2.7, 3.75)
    box(2.7, 3.2, 2.4, 1.1, "闭式加噪\n$x_t=\\sqrt{\\bar{\\alpha}_t}x_0$\n$+\\sqrt{1-\\bar{\\alpha}_t}\\varepsilon$", "#bfdbfe", 8)
    arrow(5.2, 3.75, 5.7, 3.75)
    box(5.7, 3.0, 2.6, 1.5, "网络 $\\varepsilon_\\theta(x_t, t)$\n(U-Net 等)", "#ddd6fe", 9)
    arrow(8.4, 3.75, 8.9, 3.75)
    box(8.9, 3.2, 2.5, 1.1, "预测噪声\n$\\hat{\\varepsilon}=\\varepsilon_\\theta(x_t,t)$", "#fecaca", 9)

    # True noise branch
    box(2.7, 0.8, 2.4, 1.0, "真实噪声\n$\\varepsilon \\sim \\mathcal{N}(0,I)$", "#e2e8f0", 9)
    arrow(3.9, 1.85, 3.9, 3.15)
    # loss
    arrow(10.15, 3.15, 10.15, 2.2)
    arrow(5.1, 1.3, 9.0, 1.3)
    box(9.0, 0.6, 2.6, 1.2, "损失\n$\\|\\varepsilon - \\varepsilon_\\theta\\|^2$", "#fef08a", 10)

    ax.text(6, 4.85, "图3  训练示意：用网络预测加到 $x_0$ 上的噪声 $\\varepsilon$",
            ha="center", fontsize=12, fontproperties=CN)
    ax.text(6, 0.15,
            "采样 $t\\sim\\mathrm{Uniform}\\{1..T\\}$，$\\varepsilon\\sim\\mathcal{N}(0,I)$，对 MSE 做梯度下降",
            ha="center", fontsize=9, fontproperties=CN, color="#475569")

    path = os.path.join(OUT, "fig3_training_schematic.png")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path

# --------------------------------------------------------------------------
# Figure 4: Reverse denoising trajectory schematic
# --------------------------------------------------------------------------
def fig4_reverse_trajectory():
    rng = np.random.default_rng(7)
    # Simulate a simple 2D reverse path from noise to two modes
    T = 80
    # target: approach a point near (-1.5, 0.5) or (1.5, -0.3)
    # We'll show several trajectories starting from N(0,I)
    n_paths = 12
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.patch.set_facecolor("white")

    # Left: conceptual forward strip
    ax = axes[0]
    stages = 6
    # make synthetic "images" as 2D blobs fading to noise
    for i, abar in enumerate(np.linspace(1.0, 0.02, stages)):
        xx = np.linspace(-3, 3, 80)
        yy = np.linspace(-3, 3, 80)
        X, Y = np.meshgrid(xx, yy)
        # two modes
        g1 = np.exp(-((X + 1.5) ** 2 + (Y - 0.5) ** 2) / (2 * 0.35 ** 2))
        g2 = np.exp(-((X - 1.5) ** 2 + (Y + 0.4) ** 2) / (2 * 0.4 ** 2))
        field = np.sqrt(abar) * (g1 + g2) + np.sqrt(1 - abar) * 0.35 * rng.standard_normal(X.shape)
        # plot as small heatmap strip
        ax_i = ax.inset_axes([i / stages + 0.02, 0.15, 0.14, 0.7])
        ax_i.imshow(field, cmap="viridis", origin="lower", vmin=-0.5, vmax=1.5)
        ax_i.set_xticks([])
        ax_i.set_yticks([])
        label = ["$x_0$", "$x_{t_1}$", "$x_{t_2}$", "$x_{t_3}$", "$x_{t_4}$", "$x_T$"][i]
        ax_i.set_xlabel(label, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("前向：清晰 → 噪声", fontproperties=CN, fontsize=12)
    ax.annotate("", xy=(0.95, 0.05), xytext=(0.05, 0.05),
                arrowprops=dict(arrowstyle="->", color=C_NOISE, lw=2))
    ax.text(0.5, 0.02, "增加 $t$", ha="center", fontsize=9, fontproperties=CN, color=C_NOISE)

    # Right: reverse trajectories in 2D
    ax = axes[1]
    # modes
    ax.scatter([-1.5, 1.5], [0.5, -0.4], s=120, c=[C_SIGNAL, C_SIGNAL],
               marker="*", zorder=5, label="数据模态中心")
    for k in range(n_paths):
        # start from noise
        p = rng.normal(0, 1.8, size=2)
        # choose target mode
        target = np.array([-1.5, 0.5]) if (k % 2 == 0) else np.array([1.5, -0.4])
        path = [p.copy()]
        for s in range(T):
            # pull toward target + shrinking noise
            alpha_pull = 0.08
            p = p + alpha_pull * (target - p) + rng.normal(0, 0.15 * (1 - s / T), size=2)
            path.append(p.copy())
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], color=C_PRIMARY, alpha=0.35, lw=1.0)
        ax.scatter(path[0, 0], path[0, 1], c=C_NOISE, s=18, zorder=3)
        ax.scatter(path[-1, 0], path[-1, 1], c=C_PRED, s=28, zorder=4)

    ax.set_xlim(-4, 4)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("$x^{(1)}$")
    ax.set_ylabel("$x^{(2)}$")
    ax.set_title("反向：从噪声出发的去噪轨迹", fontproperties=CN, fontsize=12)
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_NOISE, markersize=8, label=r"$x_T$ 起点"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_PRED, markersize=8, label=r"$x_0$ 终点"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=C_SIGNAL, markersize=12, label="数据模态"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=8, prop=CN if CN else None)

    fig.suptitle("图4  反向去噪生成：从纯噪声逐步回到数据分布", fontproperties=CN, fontsize=13, y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT, "fig4_reverse_trajectory.png")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path

# --------------------------------------------------------------------------
# Figure 5 (bonus): Forward / reverse overview pipeline
# --------------------------------------------------------------------------
def fig5_overview_pipeline():
    fig, ax = plt.subplots(figsize=(10, 3.6))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")

    def node(x, y, label, color):
        circ = Circle((x, y), 0.55, facecolor=color, edgecolor="#1e293b", lw=1.3, zorder=3)
        ax.add_patch(circ)
        ax.text(x, y, label, ha="center", va="center", fontsize=9,
                fontproperties=CN, color="#0f172a", zorder=4)

    positions = [1.2, 3.5, 5.8, 8.1, 10.4, 12.7]
    labels_fwd = ["$x_0$", "$x_1$", "$\\cdots$", "$x_t$", "$\\cdots$", "$x_T$"]
    colors = ["#86efac", "#a5b4fc", "#cbd5e1", "#94a3b8", "#64748b", "#475569"]

    for x, lab, c in zip(positions, labels_fwd, colors):
        node(x, 2.6, lab, c)

    for i in range(len(positions) - 1):
        ax.annotate("", xy=(positions[i + 1] - 0.55, 2.6), xytext=(positions[i] + 0.55, 2.6),
                    arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.6))

    ax.text(7, 3.55, "前向过程 $q$：逐步加噪（马尔可夫链）", ha="center",
            fontsize=11, fontproperties=CN, color="#334155")

    # reverse arrows below
    for i in range(len(positions) - 1, 0, -1):
        ax.annotate("", xy=(positions[i - 1] + 0.55, 1.0), xytext=(positions[i] - 0.55, 1.0),
                    arrowprops=dict(arrowstyle="->", color=C_PRIMARY, lw=1.6))

    for x, lab, c in zip(positions, labels_fwd, colors):
        node(x, 1.0, lab, c)

    ax.text(7, 0.25, "反向过程 $p_\\theta$：学习去噪生成（从 $x_T\\sim\\mathcal{N}(0,I)$ 出发）",
            ha="center", fontsize=11, fontproperties=CN, color=C_PRIMARY)

    ax.text(7, 3.95, "图0  扩散模型总览：前向加噪与反向去噪",
            ha="center", fontsize=12, fontproperties=CN)

    path = os.path.join(OUT, "fig0_overview_pipeline.png")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    # Fix fig2 - remove invalid scipy line
    paths = []
    paths.append(fig5_overview_pipeline())
    paths.append(fig1_forward_schedule())
    # rewrite fig2 without bad line - call fixed version inline
    paths.append(fig2_1d_gaussianization())
    paths.append(fig3_training_schematic())
    paths.append(fig4_reverse_trajectory())
    print("Generated:")
    for p in paths:
        print(p)
