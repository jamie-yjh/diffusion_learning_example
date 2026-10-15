# 课堂精简版 2D DDPM（双月数据，CPU 约 20 秒）——可现场敲 / 投影
import numpy as np, torch, torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, TensorDataset
torch.manual_seed(0)

# ---- 1. 噪声日程：beta_t、alpha_t、alpha-bar_t ----
T = 100
betas = torch.linspace(1e-4, 2e-2, T)          # beta_t，线性日程
alphas = 1 - betas                              # alpha_t = 1 - beta_t
abar = torch.cumprod(alphas, 0)                 # abar_t = prod_{s<=t} alpha_s
sa, sm = torch.sqrt(abar), torch.sqrt(1 - abar) # sqrt(abar_t), sqrt(1-abar_t)

# ---- 2. 网络 eps_theta(x_t, t)：正弦时间嵌入 + 小 MLP ----
class TimeEmb(nn.Module):
    def __init__(self, dim=32):
        super().__init__(); self.dim = dim
    def forward(self, t):                       # 整数 t -> 32 维向量
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half).float() / half)
        a = t.float().unsqueeze(1) * freqs
        return torch.cat([torch.sin(a), torch.cos(a)], -1)

class EpsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.te = TimeEmb(32)
        self.net = nn.Sequential(
            nn.Linear(2 + 32, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 2))                  # 输出与 x 同维：预测的噪声
    def forward(self, x, t):
        return self.net(torch.cat([x, self.te(t)], -1))

# ---- 3. 数据：双月，标准化到均值 0、方差 1 ----
x, _ = make_moons(8000, noise=0.08, random_state=0)
x = torch.tensor(((x - x.mean(0)) / x.std(0)).astype("float32"))
loader = DataLoader(TensorDataset(x), batch_size=256, shuffle=True, drop_last=True)
net = EpsNet()
opt = torch.optim.AdamW(net.parameters(), lr=1e-3)

# ---- 4. 训练：随机 t、随机 eps，一步构造 x_t，回归 eps ----
EPOCHS = 400                                    # 40 轮只够 loss 下降，生成仍是一团
for epoch in range(EPOCHS):
    for (xb,) in loader:
        t = torch.randint(0, T, (xb.size(0),))  # t ~ Uniform（代码下标 0..T-1）
        eps = torch.randn_like(xb)              # eps ~ N(0, I)
        xt = sa[t].unsqueeze(1) * xb + sm[t].unsqueeze(1) * eps   # 闭式加噪（重参数化）
        loss = nn.functional.mse_loss(net(xt, t), eps)            # ||eps - eps_theta||^2
        opt.zero_grad(); loss.backward(); opt.step()
    if (epoch + 1) % 50 == 0:
        print("epoch", epoch + 1, "loss", round(float(loss), 4))

# ---- 5. 采样：从 N(0, I) 出发，逐步去噪 ----
@torch.no_grad()
def sample(n=2000):
    x = torch.randn(n, 2)                       # x_T ~ N(0, I)
    for i in reversed(range(T)):
        t = torch.full((n,), i, dtype=torch.long)
        eps = net(x, t)                         # 预测噪声
        beta, alpha, ab = betas[i], alphas[i], abar[i]
        mean = (x - beta / torch.sqrt(1 - ab) * eps) / torch.sqrt(alpha)  # mu_theta
        if i > 0:
            x = mean + torch.sqrt(beta) * torch.randn_like(x)  # + sigma_t z
        else:
            x = mean                            # 最后一步不加噪声
    return x.numpy()

ys = sample()
fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
ax[0].scatter(x[:2000, 0], x[:2000, 1], s=3); ax[0].set_title("data")
ax[1].scatter(ys[:, 0], ys[:, 1], s=3, c="g"); ax[1].set_title("generated")
fig.savefig("classroom_result.png", dpi=120)
