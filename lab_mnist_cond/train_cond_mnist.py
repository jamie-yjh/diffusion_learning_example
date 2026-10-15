#!/usr/bin/env python3
"""Lightweight class-conditional DDPM on MNIST (CPU-friendly teaching lab).

Condition: digit class c ∈ {0..9}, with optional classifier-free guidance
(null label = 10). Network predicts ε given (x_t, t, c).
"""
from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# -------------------- schedule --------------------
def make_beta_schedule(T: int, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, T)


class Diffusion:
    def __init__(self, T=100, device="cpu"):
        self.T = T
        self.device = device
        betas = make_beta_schedule(T).to(device)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.abar = abar
        self.sqrt_ab = torch.sqrt(abar)
        self.sqrt_1mab = torch.sqrt(1.0 - abar)
        print(f"T={T}  beta: {betas[0].item():.2e} -> {betas[-1].item():.3f}  "
              f"abar_T={abar[-1].item():.2e}")

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sa = self.sqrt_ab[t][:, None, None, None]
        sm = self.sqrt_1mab[t][:, None, None, None]
        return sa * x0 + sm * noise, noise

    @torch.no_grad()
    def p_sample_loop(self, model, labels, guidance=1.0, null_label=10):
        """Sample one batch with class labels. guidance=1 → pure conditional."""
        n = labels.size(0)
        x = torch.randn(n, 1, 28, 28, device=self.device)
        for i in reversed(range(self.T)):
            t = torch.full((n,), i, device=self.device, dtype=torch.long)
            if guidance == 1.0:
                eps = model(x, t, labels)
            else:
                eps_c = model(x, t, labels)
                eps_u = model(x, t, torch.full_like(labels, null_label))
                eps = eps_u + guidance * (eps_c - eps_u)
            beta, alpha, ab = self.betas[i], self.alphas[i], self.abar[i]
            ab_prev = self.abar[i - 1] if i > 0 else torch.tensor(1.0, device=self.device)
            # 先估计 x0 并裁剪到数据范围 [-1,1]，再用后验 q(x_{t-1}|x_t,x0) 的均值
            # （与 eps 形式 mu=(x-beta/sqrt(1-ab)*eps)/sqrt(alpha) 在不裁剪时完全等价）
            x0_hat = ((x - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)).clamp(-1, 1)
            mean = (torch.sqrt(ab_prev) * beta / (1 - ab)) * x0_hat \
                 + (torch.sqrt(alpha) * (1 - ab_prev) / (1 - ab)) * x
            if i > 0:
                var = beta * (1 - ab_prev) / (1 - ab)      # 后验方差 beta_tilde
                x = mean + torch.sqrt(var) * torch.randn_like(x)
            else:
                x = mean                                   # 最后一步不加噪声
        return x.clamp(-1, 1)


# -------------------- tiny conditional U-Net --------------------
class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Attn(nn.Module):
    """单头自注意力（只在 7x7 分辨率使用，提供全局感受野）。"""

    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, C, H * W).unbind(1)
        w = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(C), dim=-1)  # (B,HW,HW)
        h = (v @ w.transpose(1, 2)).reshape(B, C, H, W)
        return x + self.proj(h)


class CondUNet(nn.Module):
    """U-Net 28→14→7→14→28，每级 2 个 ResBlock，7x7 处加自注意力。
    base=48 时约 2M 参数。条件 = 时间嵌入 + 类别嵌入（相加）。"""

    def __init__(self, num_classes=10, base=48, emb_dim=128, null_label=10):
        super().__init__()
        self.null_label = null_label  # CFG null token index
        self.t_emb = SinusoidalTimeEmb(emb_dim)
        self.t_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(),
                                   nn.Linear(emb_dim, emb_dim))
        # labels 0..9 + null=10 → Embedding(11, ·)
        self.c_emb = nn.Embedding(num_classes + 1, emb_dim)
        c1, c2, c3 = base, base * 2, base * 2
        self.in_conv = nn.Conv2d(1, c1, 3, padding=1)
        self.d1 = nn.ModuleList([ResBlock(c1, c1, emb_dim), ResBlock(c1, c1, emb_dim)])   # 28
        self.d2 = nn.ModuleList([ResBlock(c1, c2, emb_dim), ResBlock(c2, c2, emb_dim)])   # 14
        self.d3 = nn.ModuleList([ResBlock(c2, c3, emb_dim), ResBlock(c3, c3, emb_dim)])   # 7
        self.mid1, self.attn, self.mid2 = ResBlock(c3, c3, emb_dim), Attn(c3), ResBlock(c3, c3, emb_dim)
        self.u3 = nn.ModuleList([ResBlock(c3 * 2, c3, emb_dim), ResBlock(c3, c2, emb_dim)])  # 7
        self.u2 = nn.ModuleList([ResBlock(c2 * 2, c2, emb_dim), ResBlock(c2, c1, emb_dim)])  # 14
        self.u1 = nn.ModuleList([ResBlock(c1 * 2, c1, emb_dim), ResBlock(c1, c1, emb_dim)])  # 28
        self.out = nn.Sequential(nn.GroupNorm(8, c1), nn.SiLU(), nn.Conv2d(c1, 1, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias)
        self.pool = nn.AvgPool2d(2)

    @staticmethod
    def _run(blocks, h, emb):
        for b in blocks:
            h = b(h, emb)
        return h

    def forward(self, x, t, c):
        emb = self.t_mlp(self.t_emb(t)) + self.c_emb(c)   # (B, emb_dim)
        h1 = self._run(self.d1, self.in_conv(x), emb)       # 28, c1
        h2 = self._run(self.d2, self.pool(h1), emb)         # 14, c2
        h3 = self._run(self.d3, self.pool(h2), emb)         # 7,  c3
        m = self.mid2(self.attn(self.mid1(h3, emb)), emb)   # 7
        u = self._run(self.u3, torch.cat([m, h3], 1), emb)                          # 7  → c2
        u = self._run(self.u2, torch.cat([F.interpolate(u, scale_factor=2), h2], 1), emb)  # 14 → c1
        u = self._run(self.u1, torch.cat([F.interpolate(u, scale_factor=2), h1], 1), emb)  # 28
        return self.out(u)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def get_loader(batch, root, n_train=None, seed=0):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # → [-1, 1]
    ])
    ds = datasets.MNIST(root, train=True, download=True, transform=tf)
    if n_train is not None and n_train < len(ds):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(ds), n_train, replace=False)
        ds = Subset(ds, idx.tolist())
    return DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True,
                      num_workers=0)


@torch.no_grad()
def sample_grid(model, diff, out_path, guidance=1.0, n_per_class=8, seed=0):
    model.eval()
    device = diff.device
    torch.manual_seed(seed)
    labels = torch.arange(10, device=device).repeat_interleave(n_per_class)
    imgs = diff.p_sample_loop(model, labels, guidance=guidance)
    # denorm to [0,1]
    imgs = (imgs + 1) / 2
    grid = imgs.cpu().numpy()
    fig, axes = plt.subplots(10, n_per_class, figsize=(n_per_class * 1.1, 11))
    for i in range(10):
        for j in range(n_per_class):
            ax = axes[i, j]
            ax.imshow(grid[i * n_per_class + j, 0], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(str(i), fontsize=12, rotation=0, labelpad=12, va="center")
    title = f"Conditional samples (guidance={guidance:g})"
    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


@torch.no_grad()
def x0_recon_check(model, diff, loader, out_path, ts=(100, 300, 500, 700, 900)):
    """健全性检查：真实图加噪到 t，用预测噪声一步估计 x0_hat=(x_t-sqrt(1-abar)eps)/sqrt(abar)。"""
    model.eval()
    xb, yb = next(iter(loader))
    xb, yb = xb[:6].to(diff.device), yb[:6].to(diff.device)
    ts = [t for t in ts if t < diff.T] or [diff.T // 2]
    rows = []
    for tt in ts:
        t = torch.full((xb.size(0),), tt, device=diff.device, dtype=torch.long)
        xt, _ = diff.q_sample(xb, t)
        eps = model(xt, t, yb)
        x0h = (xt - diff.sqrt_1mab[tt] * eps) / diff.sqrt_ab[tt]
        rows.append((tt, xt.clamp(-1, 1), x0h.clamp(-1, 1)))
    fig, axes = plt.subplots(len(rows), 13, figsize=(13, 1.1 * len(rows)))
    for r, (tt, xt, x0h) in enumerate(rows):
        for j in range(6):
            axes[r, j].imshow(((xt[j, 0] + 1) / 2).cpu(), cmap="gray", vmin=0, vmax=1)
            axes[r, 7 + j].imshow(((x0h[j, 0] + 1) / 2).cpu(), cmap="gray", vmin=0, vmax=1)
        axes[r, 6].text(0.5, 0.5, f"t={tt}\n→", ha="center", va="center")
        for ax in axes[r]:
            ax.axis("off")
    fig.suptitle("x0 reconstruction check: left x_t, right x0_hat", fontsize=11)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)
    print(f"saved {out_path}", flush=True)


def train(args):
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    loader = get_loader(args.batch, args.data_root, args.n_train, args.seed)
    diff = Diffusion(T=args.T, device=device)
    model = CondUNet(base=args.base, emb_dim=args.emb_dim).to(device)
    ema = copy.deepcopy(model).eval()          # EMA 权重：采样用，样本更平滑
    for p_ in ema.parameters():
        p_.requires_grad_(False)
    print(f"params: {count_params(model):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * len(loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps))

    losses = []
    t0 = time.time()
    model.train()
    null_p = args.cfg_drop  # probability of dropping label → null
    for epoch in range(args.epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            # classifier-free: randomly replace label with null
            if null_p > 0:
                drop = torch.rand(yb.size(0), device=device) < null_p
                yb = torch.where(drop, torch.full_like(yb, 10), yb)
            t = torch.randint(0, diff.T, (xb.size(0),), device=device)
            xt, noise = diff.q_sample(xb, t)
            pred = model(xt, t, yb)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(args.ema).add_(pm.detach(), alpha=1 - args.ema)
            losses.append(loss.item())
        avg = float(np.mean(losses[-min(100, len(losses)):]))
        if (epoch + 1) % max(1, args.epochs // 10) == 0 or epoch == 0:
            print(f"epoch {epoch+1}/{args.epochs}  loss={avg:.4f}  "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    train_s = time.time() - t0
    final = float(np.mean(losses[-50:]))
    print(f"train time: {train_s:.1f}s  final loss (last 50 steps) = {final:.4f}", flush=True)

    torch.save({
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "T": args.T,
        "base": args.base,
        "emb_dim": args.emb_dim,
        "cfg_drop": args.cfg_drop,
        "final_loss": final,
        "train_seconds": train_s,
        "epochs": args.epochs,
        "n_train": args.n_train,
    }, out / "ckpt.pt")
    np.save(out / "losses.npy", np.array(losses))

    # loss curve
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(losses, lw=0.3, color="#cbd5e1", label="per step")
    w = 100
    if len(losses) > w:
        ma = np.convolve(losses, np.ones(w) / w, mode="valid")
        ax.plot(np.arange(w - 1, len(losses)), ma, lw=1.5, color="#1d4ed8",
                label=f"moving avg ({w})")
    ax.set_xlabel("step"); ax.set_ylabel("MSE(eps)"); ax.legend()
    ax.set_title(f"Cond. MNIST DDPM  T={args.T}  epochs={args.epochs}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "loss.png", dpi=140)
    plt.close(fig)

    # sample grids
    t_samp = time.time()
    x0_recon_check(ema, diff, loader, out / "x0_recon_check.png")
    sample_grid(ema, diff, out / "samples_g1.png", guidance=1.0,
                n_per_class=args.n_per_class, seed=args.seed + 1)
    if args.cfg_drop > 0:
        sample_grid(ema, diff, out / f"samples_g{args.guidance:g}.png",
                    guidance=args.guidance, n_per_class=args.n_per_class,
                    seed=args.seed + 1)
        sample_grid(ema, diff, out / "samples_g3.png",
                    guidance=3.0, n_per_class=args.n_per_class,
                    seed=args.seed + 1)
    print(f"sampling time: {time.time()-t_samp:.1f}s")
    print(f"saved under {out}/")


def parse_args():
    p = argparse.ArgumentParser(description="Conditional MNIST DDPM teaching lab")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--T", type=int, default=1000)  # 标准设置；T=100 时 abar_T≈0.36 是本实验原 bug
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--base", type=int, default=48, help="U-Net base channels")
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--cfg-drop", type=float, default=0.1,
                   help="classifier-free guidance: p(drop label → null)")
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--n-train", type=int, default=None,
                   help="subset size; None = full 60k")
    p.add_argument("--n-per-class", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--out", default="runs/demo")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
