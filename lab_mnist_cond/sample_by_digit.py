#!/usr/bin/env python3
"""Sample MNIST digits conditioned on a specified class (or all 0–9)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from train_cond_mnist import CondUNet, Diffusion, sample_grid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/final/ckpt.pt")
    p.add_argument("--digit", type=int, default=None,
                   help="single digit 0–9; omit to make 0–9 grid")
    p.add_argument("--n", type=int, default=16, help="samples for --digit")
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    model = CondUNet(base=ckpt.get("base", 48), emb_dim=ckpt.get("emb_dim", 128)).to(device)
    # 采样必须用 EMA 权重（若存在），并切到 eval
    model.load_state_dict(ckpt.get("ema", ckpt["model"]))
    model.eval()
    diff = Diffusion(T=ckpt["T"], device=device)

    out = Path(args.out) if args.out else Path(args.ckpt).parent
    out.mkdir(parents=True, exist_ok=True)

    if args.digit is None:
        path = out / f"samples_g{args.guidance:g}.png"
        sample_grid(model, diff, path, guidance=args.guidance, n_per_class=10, seed=args.seed)
    else:
        assert 0 <= args.digit <= 9
        labels = torch.full((args.n,), args.digit, device=device, dtype=torch.long)
        imgs = diff.p_sample_loop(model, labels, guidance=args.guidance)
        imgs = (imgs + 1) / 2
        cols = min(8, args.n)
        rows = (args.n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
        axes = axes.flatten() if args.n > 1 else [axes]
        for i, ax in enumerate(axes):
            if i < args.n:
                ax.imshow(imgs[i, 0].cpu(), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
        fig.suptitle(f"digit={args.digit}  guidance={args.guidance:g}", fontsize=12)
        fig.tight_layout()
        path = out / f"digit_{args.digit}_g{args.guidance:g}.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
