# 课堂单文件版：条件 DDPM（MNIST，数字类别 c 作为条件）
# 与 train_cond_mnist.py 完全相同的模型（CondUNet，2,139,713 参数）、日程与采样器，
# 只是把代码放进一个文件、并针对课堂 GPU（A100 / H200）做了加速设置。
#   训练：   python classroom_cond_mnist.py --epochs 10
#   免训练： python classroom_cond_mnist.py --load runs/final/ckpt_ema.pt
import argparse, copy, math, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================= 1. 噪声日程（T=1000，线性 β）=======================
T = 1000
betas = torch.linspace(1e-4, 2e-2, T)
alphas = 1 - betas
abar = torch.cumprod(alphas, 0)          # abar_T ≈ 4e-5，x_T 才真正接近 N(0, I)

# ======================= 2. 网络：CondUNet（与完整版逐行一致）=======================
class SinusoidalTimeEmb(nn.Module):
    """正弦时间嵌入：把整数 t 变成 dim 维向量（不同频率的 sin/cos）"""
    def __init__(self, dim):
        super().__init__(); self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        a = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)

class ResBlock(nn.Module):
    """GN→SiLU→Conv →(+条件偏置)→ GN→SiLU→Conv，再加残差"""
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, out_ch)                 # 条件向量 → 逐通道偏置
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]       # ★ 条件在每个块注入
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)

class Attn(nn.Module):
    """单头自注意力，只在 7x7 使用（49 个 token），补全局感受野"""
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, C, H * W).unbind(1)
        w = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(C), dim=-1)   # (B, 49, 49)
        h = (v @ w.transpose(1, 2)).reshape(B, C, H, W)
        return x + self.proj(h)

class CondUNet(nn.Module):
    """ε_θ(x_t, t, c)：28→14→7→14→28，每级 2 个 ResBlock，7x7 处自注意力"""
    def __init__(self, num_classes=10, base=48, emb_dim=128, null_label=10):
        super().__init__()
        self.null_label = null_label
        self.t_emb = SinusoidalTimeEmb(emb_dim)
        self.t_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.c_emb = nn.Embedding(num_classes + 1, emb_dim)  # 0..9 数字 + 10 = null（CFG）
        c1, c2, c3 = base, base * 2, base * 2
        self.in_conv = nn.Conv2d(1, c1, 3, padding=1)
        self.d1 = nn.ModuleList([ResBlock(c1, c1, emb_dim), ResBlock(c1, c1, emb_dim)])   # 28x28
        self.d2 = nn.ModuleList([ResBlock(c1, c2, emb_dim), ResBlock(c2, c2, emb_dim)])   # 14x14
        self.d3 = nn.ModuleList([ResBlock(c2, c3, emb_dim), ResBlock(c3, c3, emb_dim)])   # 7x7
        self.mid1, self.attn, self.mid2 = ResBlock(c3, c3, emb_dim), Attn(c3), ResBlock(c3, c3, emb_dim)
        self.u3 = nn.ModuleList([ResBlock(c3 * 2, c3, emb_dim), ResBlock(c3, c2, emb_dim)])  # 7x7
        self.u2 = nn.ModuleList([ResBlock(c2 * 2, c2, emb_dim), ResBlock(c2, c1, emb_dim)])  # 14x14
        self.u1 = nn.ModuleList([ResBlock(c1 * 2, c1, emb_dim), ResBlock(c1, c1, emb_dim)])  # 28x28
        self.out = nn.Sequential(nn.GroupNorm(8, c1), nn.SiLU(), nn.Conv2d(c1, 1, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias)   # 初始输出 0
        self.pool = nn.AvgPool2d(2)
    @staticmethod
    def _run(blocks, h, emb):
        for b in blocks:
            h = b(h, emb)
        return h
    def forward(self, x, t, c):
        emb = self.t_mlp(self.t_emb(t)) + self.c_emb(c)          # ★ 条件入口：时间 + 类别
        h1 = self._run(self.d1, self.in_conv(x), emb)              # 48 x 28 x 28
        h2 = self._run(self.d2, self.pool(h1), emb)                # 96 x 14 x 14
        h3 = self._run(self.d3, self.pool(h2), emb)                # 96 x 7 x 7
        m = self.mid2(self.attn(self.mid1(h3, emb)), emb)
        u = self._run(self.u3, torch.cat([m, h3], 1), emb)                                  # 跳连 h3
        u = self._run(self.u2, torch.cat([F.interpolate(u, scale_factor=2), h2], 1), emb)  # 跳连 h2
        u = self._run(self.u1, torch.cat([F.interpolate(u, scale_factor=2), h1], 1), emb)  # 跳连 h1
        return self.out(u)

# ======================= 3. 采样：x0 裁剪 + 后验均值/方差 + CFG =======================
@torch.no_grad()
def sample(net, labels, guidance=1.5, amp_dtype=None):
    dev = labels.device
    b_, a_, ab_ = betas.to(dev), alphas.to(dev), abar.to(dev)
    n = labels.size(0)
    x = torch.randn(n, 1, 28, 28, device=dev)                      # x_T ~ N(0, I)
    null = torch.full_like(labels, 10)
    for i in reversed(range(T)):
        t = torch.full((n,), i, device=dev, dtype=torch.long)
        with torch.autocast(dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            if guidance == 1.0:
                e = net(x, t, labels)
            else:   # CFG：eps = eps_uncond + w * (eps_cond - eps_uncond)，一次前向算两份
                e2 = net(torch.cat([x, x]), torch.cat([t, t]), torch.cat([labels, null]))
                e_c, e_u = e2.float().chunk(2)
                e = e_u + guidance * (e_c - e_u)
        e = e.float()
        b, a, ab = b_[i], a_[i], ab_[i]
        ab_prev = ab_[i - 1] if i > 0 else torch.tensor(1.0, device=dev)
        x0 = ((x - torch.sqrt(1 - ab) * e) / torch.sqrt(ab)).clamp(-1, 1)   # 估计并裁剪 x0
        mean = torch.sqrt(ab_prev) * b / (1 - ab) * x0 + torch.sqrt(a) * (1 - ab_prev) / (1 - ab) * x
        var = b * (1 - ab_prev) / (1 - ab)                                    # 后验方差 β̃_t
        x = mean + torch.sqrt(var) * torch.randn_like(x) if i > 0 else mean  # 最后一步不加噪
    return ((x + 1) / 2).clamp(0, 1)                                          # [-1,1] → [0,1]

def save_grid(imgs, n_per, path, title):
    fig, axes = plt.subplots(10, n_per, figsize=(n_per * 1.0 + 0.6, 10.6))
    for i in range(10):
        for j in range(n_per):
            ax = axes[i, j]
            ax.imshow(imgs[i * n_per + j, 0].cpu(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])          # 不用 axis('off')，否则行标签也会被隐藏
            for s in ax.spines.values(): s.set_visible(False)
            if j == 0:
                ax.set_ylabel(str(i), rotation=0, fontsize=14, labelpad=14, va="center")
    fig.suptitle(title, y=0.995); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print("saved", path, flush=True)

# ======================= 4. 主程序：训练（GPU 友好）或直接加载 =======================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=None, help="默认 8e-4 * batch/128 的平方根缩放")
    p.add_argument("--device", default="auto")
    p.add_argument("--no-amp", action="store_true", help="关闭 bf16 autocast（仅 cuda 默认开启）")
    p.add_argument("--compile", action="store_true", help="torch.compile（默认关，稳妥）")
    p.add_argument("--max-steps", type=int, default=None, help="冒烟测试：训练到这么多步就停")
    p.add_argument("--load", default=None, help="跳过训练，直接加载 ckpt_ema.pt 采样")
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--n-per-class", type=int, default=10)
    p.add_argument("--out", default="classroom_out")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                       else ("cpu" if args.device == "auto" else args.device))
    torch.manual_seed(args.seed)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
        print("GPU:", torch.cuda.get_device_name(dev), flush=True)
    else:
        torch.set_num_threads(max(1, torch.get_num_threads()))
    amp = torch.bfloat16 if (dev.type == "cuda" and not args.no_amp) else None
    import pathlib; out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)

    net = CondUNet().to(dev)
    print(f"device={dev}  amp={'bf16' if amp else 'off'}  params={sum(q.numel() for q in net.parameters()):,}", flush=True)

    if args.load:                                   # ---- 免训练：直接用给定 EMA 权重 ----
        ck = torch.load(args.load, map_location=dev, weights_only=False)
        net.load_state_dict(ck.get("ema", ck["model"]))
        ema = net.eval()
    else:                                           # ---- 训练 ----
        from torchvision import datasets
        ds = datasets.MNIST("./data", train=True, download=True)
        # 整个 MNIST（60000x28x28，约 47MB float）一次性放到 GPU/内存，免去 DataLoader 瓶颈
        X = (ds.data.float().div(255).mul(2).sub(1)).unsqueeze(1).to(dev)   # [-1, 1]
        Y = ds.targets.to(dev)
        lr = args.lr or 8e-4 * math.sqrt(args.batch / 128)
        opt = torch.optim.AdamW(net.parameters(), lr=lr)
        steps_per_epoch = len(X) // args.batch
        total = args.epochs * steps_per_epoch if args.max_steps is None else min(args.max_steps, args.epochs * steps_per_epoch)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
        ema = copy.deepcopy(net).eval()
        for q in ema.parameters(): q.requires_grad_(False)
        fwd = torch.compile(net) if args.compile else net
        ab_d = abar.to(dev); sa, sm = ab_d.sqrt(), (1 - ab_d).sqrt()
        print(f"batch={args.batch} lr={lr:.2e} steps/epoch={steps_per_epoch} total_steps={total}", flush=True)
        losses, step, t0 = [], 0, time.time()
        for epoch in range(args.epochs):
            te = time.time(); perm = torch.randperm(len(X), device=dev)
            for k in range(steps_per_epoch):
                idx = perm[k * args.batch:(k + 1) * args.batch]
                xb, yb = X[idx], Y[idx]
                drop = torch.rand(yb.size(0), device=dev) < 0.1                 # CFG：10% 丢标签
                yb = torch.where(drop, torch.full_like(yb, 10), yb)
                t = torch.randint(0, T, (xb.size(0),), device=dev)
                eps = torch.randn_like(xb)
                xt = sa[t][:, None, None, None] * xb + sm[t][:, None, None, None] * eps   # 闭式加噪
                with torch.autocast(dev.type, dtype=amp, enabled=amp is not None):
                    loss = F.mse_loss(fwd(xt, t, yb).float(), eps)                       # 回归噪声
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
                with torch.no_grad():                                                      # EMA 0.999
                    for pe, pm in zip(ema.parameters(), net.parameters()):
                        pe.lerp_(pm, 1 - 0.999)
                losses.append(loss.item())
                step += 1
                if step >= total: break
            if dev.type == "cuda": torch.cuda.synchronize()
            print(f"epoch {epoch+1}/{args.epochs}  loss(avg last 100)={np.mean(losses[-100:]):.4f}  "
                  f"epoch_time={time.time()-te:.1f}s  total={time.time()-t0:.1f}s", flush=True)
            if step >= total: break
        train_s = time.time() - t0
        print(f"TRAIN DONE: {step} steps, {train_s:.1f}s, final loss(last 50)={np.mean(losses[-50:]):.4f}", flush=True)
        torch.save({"model": ema.state_dict(), "T": T, "base": 48, "emb_dim": 128,
                    "steps": step, "train_seconds": train_s}, out / "ckpt_ema.pt")
        fig, ax = plt.subplots(figsize=(6, 3.2)); ax.plot(losses, lw=0.4)
        ax.set_xlabel("step"); ax.set_ylabel("MSE(eps)"); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(out / "loss.png", dpi=130); plt.close(fig)

    # ---- 条件采样：每行一个数字 0–9 ----
    torch.manual_seed(args.seed + 1)
    labels = torch.arange(10, device=dev).repeat_interleave(args.n_per_class)
    ts = time.time()
    imgs = sample(ema, labels, guidance=args.guidance, amp_dtype=amp)
    if dev.type == "cuda": torch.cuda.synchronize()
    print(f"SAMPLING: {len(labels)} images x {T} steps in {time.time()-ts:.1f}s", flush=True)
    save_grid(imgs, args.n_per_class, out / f"samples_g{args.guidance:g}.png",
              f"classroom CondUNet, guidance={args.guidance:g}")

if __name__ == "__main__":
    main()
