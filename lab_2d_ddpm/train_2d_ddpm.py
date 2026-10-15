#!/usr/bin/env python3
"""Lightweight 2D DDPM teaching lab: sklearn moons, CPU, minutes to train."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, TensorDataset


# ---------- schedule ----------
def make_beta_schedule(T: int, beta_start=1e-4, beta_end=2e-2):
    # 线性日程。注意：T=100 时 abar_T≈0.36，x_T 并非严格 N(0,I)；
    # 但实测（见讲义）这比把 β 放大到 0.2 生成得更好——小步长让反向一步更接近高斯。
    return torch.linspace(beta_start, beta_end, T)


class Diffusion:
    def __init__(self, T=100, device="cpu", beta_end=2e-2):
        self.T = T
        self.device = device
        betas = make_beta_schedule(T, beta_end=beta_end).to(device)
        alphas = 1.0 - betas
        alphabar = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alphabar = alphabar
        self.sqrt_ab = torch.sqrt(alphabar)
        self.sqrt_1mab = torch.sqrt(1.0 - alphabar)
        print(f"T={T}  beta: {betas[0].item():.2e} -> {betas[-1].item():.3f}  abar_T={alphabar[-1].item():.2e}")

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        # x_t = sqrt(abar_t) x0 + sqrt(1-abar_t) eps
        sa = self.sqrt_ab[t].unsqueeze(-1)
        sm = self.sqrt_1mab[t].unsqueeze(-1)
        return sa * x0 + sm * noise, noise

    @torch.no_grad()
    def p_sample_loop(self, model, n, shape=(2,)):
        x = torch.randn(n, *shape, device=self.device)
        traj = [x.cpu().numpy()]
        for i in reversed(range(self.T)):
            t = torch.full((n,), i, device=self.device, dtype=torch.long)
            eps = model(x, t)
            beta = self.betas[i]
            alpha = self.alphas[i]
            abar = self.alphabar[i]
            # mean from epsilon prediction
            mean = (1.0 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1.0 - abar)) * eps
            )
            if i > 0:
                x = mean + torch.sqrt(beta) * torch.randn_like(x)
            else:
                x = mean
            if i % max(1, self.T // 10) == 0 or i == 0:
                traj.append(x.cpu().numpy())
        return x.cpu().numpy(), traj


# ---------- tiny MLP conditioned on t ----------
class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -np.log(10000) * torch.arange(half, device=t.device).float() / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class EpsMLP(nn.Module):
    def __init__(self, x_dim=2, t_dim=32, hidden=128):
        super().__init__()
        self.t_emb = SinusoidalTimeEmb(t_dim)
        self.net = nn.Sequential(
            nn.Linear(x_dim + t_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, x_dim),
        )

    def forward(self, x, t):
        h = torch.cat([x, self.t_emb(t)], dim=-1)
        return self.net(h)


def get_data(n=8000, noise=0.08, seed=0):
    x, _ = make_moons(n_samples=n, noise=noise, random_state=seed)
    x = x.astype(np.float32)
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    return torch.from_numpy(x)


def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)  # 可复现：固定 t、eps、采样噪声的随机种子
    np.random.seed(args.seed)
    t_start = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = get_data(args.n_samples, args.noise, args.seed)
    loader = DataLoader(
        TensorDataset(data), batch_size=args.batch, shuffle=True, drop_last=True
    )
    diff = Diffusion(T=args.T, device=device, beta_end=args.beta_end)
    model = EpsMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * len(loader)
    # 余弦学习率衰减：后期小步长，生成的月牙更“干净”
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    losses = []
    model.train()
    step = 0
    for epoch in range(args.epochs):
        for (xb,) in loader:
            xb = xb.to(device)
            t = torch.randint(0, diff.T, (xb.size(0),), device=device)
            xt, noise = diff.q_sample(xb, t)
            pred = model(xt, t)
            loss = nn.functional.mse_loss(pred, noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
            step += 1
        if (epoch + 1) % max(1, args.epochs // 5) == 0 or epoch == 0:
            print(f"epoch {epoch+1}/{args.epochs}  loss={np.mean(losses[-50:]):.4f}")

    t_train = time.time() - t_start
    print(f"train time: {t_train:.1f}s  final loss (mean of last 50 steps) = {np.mean(losses[-50:]):.4f}")
    torch.save({"model": model.state_dict(), "T": args.T}, out / "ckpt.pt")
    np.save(out / "losses.npy", np.array(losses))

    # sample + visualize
    model.eval()
    samples, traj = diff.p_sample_loop(model, n=2000)
    data_np = data.numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].scatter(data_np[:, 0], data_np[:, 1], s=3, alpha=0.4, c="#3b82f6")
    axes[0].set_title("Data (moons)")
    axes[0].set_aspect("equal")
    axes[1].plot(losses, lw=0.6, color="#64748b")
    axes[1].set_title("Train loss (eps MSE)")
    axes[1].set_xlabel("step")
    # reverse trajectory: early noise -> late samples
    mid = traj[len(traj) // 2]
    axes[2].scatter(traj[0][:, 0], traj[0][:, 1], s=2, alpha=0.25, c="#94a3b8", label="t=T")
    axes[2].scatter(mid[:, 0], mid[:, 1], s=2, alpha=0.35, c="#f59e0b", label="mid")
    axes[2].scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.45, c="#10b981", label="t=0")
    axes[2].set_title("Reverse sampling")
    axes[2].legend(markerscale=3, fontsize=8)
    axes[2].set_aspect("equal")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "result.png", dpi=140)
    plt.close(fig)

    # 额外教学图：数据 vs 生成；反向轨迹逐帧；平滑 loss 曲线
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), sharex=True, sharey=True)
    axes[0].scatter(data_np[:2000, 0], data_np[:2000, 1], s=3, alpha=0.4, c="#3b82f6")
    axes[0].set_title("Data (2000 pts)")
    axes[1].scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.4, c="#10b981")
    axes[1].set_title(f"Generated (T={args.T}, {args.epochs} epochs)")
    for ax in axes:
        ax.set_aspect("equal"); ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(out / "data_vs_generated.png", dpi=140); plt.close(fig)

    k = len(traj)
    idx = np.linspace(0, k - 1, 6).round().astype(int)
    fig, axes = plt.subplots(1, len(idx), figsize=(15, 2.9), sharex=True, sharey=True)
    for ax, j in zip(axes, idx):
        ax.scatter(traj[j][:, 0], traj[j][:, 1], s=2, alpha=0.4, c="#7c3aed")
        ax.set_title("x_T (noise)" if j == 0 else f"frame {j}/{k-1}" + (" = x_0" if j == k - 1 else ""))
        ax.set_aspect("equal"); ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5); ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(out / "reverse_frames.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(losses, lw=0.4, color="#cbd5e1", label="per step")
    w = 50
    if len(losses) > w:
        ma = np.convolve(losses, np.ones(w) / w, mode="valid")
        ax.plot(np.arange(w - 1, len(losses)), ma, lw=1.6, color="#1d4ed8", label=f"moving avg ({w})")
    ax.set_xlabel("step"); ax.set_ylabel("MSE(eps)"); ax.legend(); ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(out / "loss.png", dpi=140); plt.close(fig)
    print(f"saved {out}/result.png, data_vs_generated.png, reverse_frames.png, loss.png, ckpt.pt")


def parse_args():
    p = argparse.ArgumentParser(description="2D moons DDPM teaching lab")
    p.add_argument("--epochs", type=int, default=400)  # 40 轮只够看到 loss 下降，生成仍是一团；400 轮约 20 s（CPU）
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta-end", type=float, default=2e-2)
    p.add_argument("--n-samples", type=int, default=8000)
    p.add_argument("--noise", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="runs/moons")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
