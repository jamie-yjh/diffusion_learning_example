#!/usr/bin/env python3
"""用一个小 CNN 分类器给生成样本「判卷」：每类生成 N 张，统计被判为目标类的比例。"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from train_cond_mnist import CondUNet, Diffusion


def make_clf():
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(64 * 49, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 10))


def train_clf(root, path, epochs=2):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])  # 与扩散模型同为 [-1,1]
    tr = DataLoader(datasets.MNIST(root, True, download=True, transform=tf), 128, shuffle=True)
    te = DataLoader(datasets.MNIST(root, False, download=True, transform=tf), 512)
    clf = make_clf(); opt = torch.optim.Adam(clf.parameters(), 1e-3)
    for ep in range(epochs):
        clf.train()
        for x, y in tr:
            loss = F.cross_entropy(clf(x), y); opt.zero_grad(); loss.backward(); opt.step()
    clf.eval(); ok = n = 0
    with torch.no_grad():
        for x, y in te:
            ok += (clf(x).argmax(1) == y).sum().item(); n += len(y)
    torch.save(clf.state_dict(), path)
    return ok / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/final/ckpt.pt")
    p.add_argument("--n", type=int, default=30, help="每类样本数")
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--data-root", default="./data")
    args = p.parse_args()
    torch.set_num_threads(8); torch.manual_seed(123)
    out = Path(args.ckpt).parent
    cpath = out / "mnist_cnn.pt"
    if cpath.exists():
        clf = make_clf(); clf.load_state_dict(torch.load(cpath)); test_acc = None
    else:
        t0 = time.time(); test_acc = train_clf(args.data_root, cpath)
        print(f"classifier test acc = {test_acc:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        clf = make_clf(); clf.load_state_dict(torch.load(cpath))
    clf.eval()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    m = CondUNet(base=ck["base"], emb_dim=ck["emb_dim"]); m.load_state_dict(ck.get("ema", ck["model"])); m.eval()
    diff = Diffusion(T=ck["T"])
    labels = torch.arange(10).repeat_interleave(args.n)
    t0 = time.time()
    x = diff.p_sample_loop(m, labels, guidance=args.guidance)
    with torch.no_grad():
        pred = clf(x).argmax(1)
    per = {d: (pred[labels == d] == d).float().mean().item() for d in range(10)}
    overall = (pred == labels).float().mean().item()
    res = {"guidance": args.guidance, "n_per_class": args.n, "classifier_test_acc": test_acc,
           "per_class": per, "overall": overall, "sample_seconds": time.time() - t0}
    print(json.dumps(res, indent=1))
    (out / f"clf_eval_g{args.guidance:g}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
