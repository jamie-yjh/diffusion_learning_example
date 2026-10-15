#!/usr/bin/env python3
"""教学消融：MSE 回归学的是 E[x0|xt]，多峰双月在高噪声下塌缩到均值附近；
DDPM 逐步采样则保持双月。产出 figs/mean_collapse.png 与 figs/mean_collapse_by_t.png。

用法：
  python mean_collapse_demo.py
  # 若已有 runs/moons_e400/ckpt.pt，右侧 DDPM 面板直接加载它；否则现训一个 eps 模型。
"""
from __future__ import annotations
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np, torch, torch.nn as nn
from train_2d_ddpm import Diffusion, EpsMLP, get_data, SinusoidalTimeEmb

for fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    try: font_manager.fontManager.addfont(fp)
    except Exception: pass
plt.rcParams["font.family"] = "Noto Sans CJK SC"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(8)
T, epochs, batch, n = 100, 120, 512, 6000
out = Path("figs"); out.mkdir(exist_ok=True)
data = get_data(n, 0.08, 0)
diff = Diffusion(T=T)

class X0MLP(nn.Module):
    def __init__(self, x_dim=2, t_dim=32, hidden=128):
        super().__init__()
        self.t_emb = SinusoidalTimeEmb(t_dim)
        self.net = nn.Sequential(
            nn.Linear(x_dim + t_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, x_dim),
        )
    def forward(self, x, t):
        return self.net(torch.cat([x, self.t_emb(t)], dim=-1))

x0_net = X0MLP()
opt = torch.optim.AdamW(x0_net.parameters(), lr=3e-3)
for ep in range(epochs):
    perm = torch.randperm(len(data))
    for i in range(0, len(data) - batch + 1, batch):
        x0 = data[perm[i:i + batch]]
        t = torch.randint(0, T, (x0.size(0),))
        xt, _ = diff.q_sample(x0, t)
        loss = ((x0_net(xt, t) - x0) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
print(f"x0-predictor last loss={float(loss):.4f}")

with torch.no_grad():
    collapsed = x0_net(torch.randn(2000, 2), torch.full((2000,), T - 1, dtype=torch.long)).numpy()
    blobs = {}
    x0_true = data[:1500]
    for tt in (10, 40, 70, 99):
        t = torch.full((len(x0_true),), tt, dtype=torch.long)
        xt, _ = diff.q_sample(x0_true, t)
        blobs[tt] = x0_net(xt, t).numpy()
print("collapsed mean", collapsed.mean(0), "std", collapsed.std(0))

ckpt = Path("runs/moons_e400/ckpt.pt")
eps = EpsMLP()
if ckpt.exists():
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    eps.load_state_dict(ck.get("model", ck)); print("loaded", ckpt)
else:
    opt2 = torch.optim.AdamW(eps.parameters(), lr=2e-3)
    for ep in range(400):
        perm = torch.randperm(len(data))
        for i in range(0, len(data) - batch + 1, batch):
            x0 = data[perm[i:i + batch]]
            t = torch.randint(0, T, (x0.size(0),))
            xt, noise = diff.q_sample(x0, t)
            loss = ((eps(xt, t) - noise) ** 2).mean()
            opt2.zero_grad(); loss.backward(); opt2.step()
    print("trained eps from scratch")
eps.eval()
ddpm, _ = diff.p_sample_loop(eps, 2000)

fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6))
lim = (-2.6, 2.6)
for ax, pts, title, c in [
    (axes[0], data[:2000].numpy(), "真实双月（标准化后）", "#1d4ed8"),
    (axes[1], collapsed, "MSE 回归预测 $E[x_0|x_T]$\n一次去噪 → 塌缩到均值附近", "#b91c1c"),
    (axes[2], ddpm, "DDPM 逐步采样（同容量 MLP）\n保持双峰", "#15803d"),
]:
    ax.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.45, c=c, edgecolors="none")
    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
    ax.set_title(title, fontsize=10); ax.grid(True, alpha=0.25)
    ax.axhline(0, color="gray", lw=0.4); ax.axvline(0, color="gray", lw=0.4)
fig.suptitle(r"为什么不能用「普通回归」生成双月：$E[x_0|x_t]$ 在多峰分布下塌缩", fontsize=11, y=1.02)
fig.tight_layout(); fig.savefig(out / "mean_collapse.png", dpi=150, bbox_inches="tight"); plt.close(fig)

fig, axes = plt.subplots(1, 4, figsize=(11, 2.8))
for ax, tt in zip(axes, (10, 40, 70, 99)):
    pts = blobs[tt]
    ax.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.4, c="#b91c1c", edgecolors="none")
    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
    ax.set_title(f"$t={tt}$,  std={pts.std():.2f}", fontsize=10); ax.grid(True, alpha=0.25)
fig.suptitle(r"同一 $x_0$-回归器：噪声越大，$E[x_0|x_t]$ 越向原点（两月之间）塌缩", fontsize=10, y=1.05)
fig.tight_layout(); fig.savefig(out / "mean_collapse_by_t.png", dpi=140, bbox_inches="tight"); plt.close(fig)
print("saved", out / "mean_collapse.png", out / "mean_collapse_by_t.png")
