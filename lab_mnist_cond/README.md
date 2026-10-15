# 条件 DDPM · MNIST 教案例子（进阶）

前置：完成 `../lab_2d_ddpm` 双月无条件实验。配套教案 PDF：`diffusion_cond_mnist_lesson_plan.pdf`。

## 环境
```bash
# 复用 lab_venv；torchvision 须与 torch 同为 CPU 轮
pip install torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 课堂（A100 / H200）
```bash
CUDA_VISIBLE_DEVICES=0 python classroom_cond_mnist.py --epochs 10            # 自动 cuda、bf16、整库放显存
python classroom_cond_mnist.py --epochs 16 --batch 512                       # 大 batch 时多给几个 epoch
python classroom_cond_mnist.py --load runs/final/ckpt_ema.pt                 # 兜底：免训练，直接采样
```
`classroom_cond_mnist.py` 是单文件版，模型与 `train_cond_mnist.py` 逐行相同（2,139,713 参数，可互载权重）。
GPU 耗时**未实测**：估计 10 epochs 约 2–5 分钟；脚本会打印每个 epoch 的真实耗时与采样耗时。
选项：`--no-amp` 关闭 bf16，`--compile` 开启 torch.compile，`--max-steps N` 冒烟测试。

## 推荐命令（默认参数即下表设定）
```bash
python train_cond_mnist.py --epochs 8 --T 1000 --base 48 --device cpu --out runs/final
python sample_by_digit.py --ckpt runs/final/ckpt_ema.pt --guidance 1.5        # zip 内附 EMA 权重(8.6MB)，免训练直接采样
python sample_by_digit.py --ckpt runs/final/ckpt_ema.pt --digit 3 --guidance 1.5 --n 16
python eval_classifier.py --ckpt runs/final/ckpt_ema.pt --guidance 1.5 --n 30     # 分类器判卷
```

## 实测（8 核 CPU，PyTorch 2.14.1+cpu）
| 项 | 值 |
|---|---|
| 模型 | U-Net（每级 2 ResBlock + 7x7 自注意力），2,139,713 参数，EMA 0.999 |
| 日程 | linear β 1e-4→0.02，T=1000，ᾱ_T≈4.0e-5 |
| 训练 | 8 epochs，1649.2 s（27.5 min），末 50 步 loss 0.0220 |
| 采样 | x̂0 检查 + 3 张 10x10 网格 共 424.9 s |
| 分类器判卷（CNN 测试 98.83%，每类 30 张） | w=1.5: 300/300 = 100%；w=1: 95.0%（最弱 5: 26/30） |

最终网格：`runs/final/samples_g1.5.png`。

## 第一版 bug（已修复）
旧版 `T=100` → ᾱ_T≈0.36：真实 x_T = 0.6·x0 + 0.8·ε，背景均值≈-0.6，而采样从 N(0,I) 出发，起点错位 →
loss 平台化（≈0.054）但样本认不出、部分行全黑。另有网络过小（0.26M）。修复：T=1000、2.1M U-Net+注意力、EMA、
x̂0 裁剪 + 后验方差采样。`runs/longer/` 保留旧版输出作对照（`figs/before_after.png`）。

## 文件
- `train_cond_mnist.py` — 完整训练 + x̂0 重建检查 + 多 guidance 采样
- `classroom_cond_mnist.py` — 课堂单文件版（同一 CondUNet，GPU 友好：bf16、数据常驻显存、cudnn.benchmark）
- `sample_by_digit.py` — 指定数字采样 / 重出网格
- `eval_classifier.py` — 小 CNN 判卷
