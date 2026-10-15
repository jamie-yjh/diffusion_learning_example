#!/usr/bin/env python3
"""Show forward diffusion on 2D moons (no training). Good warm-up demo."""
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_moons


def main():
    torch.manual_seed(0)  # 可复现
    x, _ = make_moons(2000, noise=0.08, random_state=0)
    x = torch.from_numpy(((x - x.mean(0)) / x.std(0)).astype(np.float32))
    T, beta0, beta1 = 100, 1e-4, 2e-2
    betas = torch.linspace(beta0, beta1, T)  # 与 train_2d_ddpm.py 默认一致
    abar = torch.cumprod(1 - betas, 0)
    print(f"abar_T = {abar[-1].item():.2e}")
    ts = [0, 20, 50, 99]
    fig, axes = plt.subplots(1, len(ts), figsize=(12, 3))
    for ax, t in zip(axes, ts):
        if t == 0:
            xt = x
        else:
            eps = torch.randn_like(x)
            xt = torch.sqrt(abar[t]) * x + torch.sqrt(1 - abar[t]) * eps
        ax.scatter(xt[:, 0], xt[:, 1], s=3, alpha=0.4)
        ax.set_title("x_0 (data)" if t == 0 else f"t={t}  abar={abar[t].item():.2f}")
        ax.set_aspect("equal")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Forward process q(x_t|x_0) on moons")
    fig.tight_layout()
    fig.savefig("forward_demo.png", dpi=140)
    print("saved forward_demo.png")


if __name__ == "__main__":
    main()
