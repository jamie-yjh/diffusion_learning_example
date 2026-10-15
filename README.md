# 扩散模型（DDPM）教学示例

本仓库包含三份配套教学材料：一份数学讲义 + 两个由浅入深的动手实验。

| 材料 | 内容 | PDF |
|---|---|---|
| `math_notes/` | 扩散模型数学讲义（前向加噪、反向贝叶斯推导、小 β 高斯近似、重参数化、训练目标） | [diffusion_math_teaching_notes_latex.pdf](math_notes/diffusion_math_teaching_notes_latex.pdf) |
| `lab_2d_ddpm/` | 实验一：二维双月（two-moons）无条件 DDPM，CPU 几十秒即可训练 | [diffusion_lab_lesson_plan.pdf](lab_2d_ddpm/diffusion_lab_lesson_plan.pdf) |
| `lab_mnist_cond/` | 实验二：MNIST 条件扩散（U-Net + 自注意力 + Classifier-Free Guidance），附 EMA 权重 | [diffusion_cond_mnist_lesson_plan.pdf](lab_mnist_cond/diffusion_cond_mnist_lesson_plan.pdf) |

## 推荐学习顺序
1. **数学讲义**：先理解前向过程 q(x_t|x_0)、反向过程与 ε 预测目标。
2. **实验一 双月 DDPM**：在 2D 上直观看到加噪/去噪轨迹，以及 T、β、训练轮数的影响。
3. **实验二 MNIST 条件扩散**：扩展到图像、类别条件与 CFG 引导强度。

## 环境
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # torch, torchvision, numpy, matplotlib, scikit-learn
```
GPU（A100 / H200）：按集群 CUDA 版本安装对应 torch/torchvision 轮子（如 `--index-url https://download.pytorch.org/whl/cu124`）；仅 CPU 时可用 `.../whl/cpu`。torch 与 torchvision 必须来自同一渠道。

## 实验一：二维双月 DDPM
```bash
cd lab_2d_ddpm
python classroom_ddpm.py                             # 课堂精简版，CPU 约 20 秒
python forward_demo.py                               # 前向加噪演示
python train_2d_ddpm.py --epochs 400 --out runs/moons_e400
python train_2d_ddpm.py --T 20 --out runs/moons_T20  # 对比不同 T / β / 轮数
python mean_collapse_demo.py                         # 直接回归均值 vs DDPM（加载 runs/moons_e400/ckpt.pt）
```
2D 模型很小，CPU 即可；GPU 上可加 `--device cuda`，但提速不明显。

## 实验二：MNIST 条件扩散
```bash
cd lab_mnist_cond
# 课堂（A100 / H200）：自动 cuda + bf16，整个数据集放入显存
CUDA_VISIBLE_DEVICES=0 python classroom_cond_mnist.py --epochs 10
python classroom_cond_mnist.py --epochs 16 --batch 512
# 兜底：免训练，直接加载仓库内 EMA 权重采样
python classroom_cond_mnist.py --load runs/final/ckpt_ema.pt
# 完整版脚本
python train_cond_mnist.py --epochs 8 --T 1000 --base 48 --out runs/final
python sample_by_digit.py --ckpt runs/final/ckpt_ema.pt --digit 3 --guidance 1.5 --n 16
python eval_classifier.py --ckpt runs/final/ckpt_ema.pt --guidance 1.5 --n 30
```
- 首次运行会通过 torchvision 自动下载 MNIST 到 `data/`（已在 .gitignore 中忽略）。
- GPU 耗时未实测，估计 A100/H200 上 10 epochs 约 2–5 分钟；CPU（8 核）8 epochs 约 27 分钟。
- 可选：`--no-amp` 关闭 bf16，`--compile` 开启 torch.compile，`--max-steps N` 冒烟测试。
- 仓库仅保留 `runs/final/` 的 EMA 权重（`ckpt_ema.pt`，8.6MB）与判卷用 CNN（`mnist_cnn.pt`）；非 EMA 的 `ckpt.pt` 未上传，脚本默认 `--ckpt` 指向它，请显式传 `ckpt_ema.pt` 或自行训练。
详见 [lab_mnist_cond/README.md](lab_mnist_cond/README.md)。

## 重新编译 PDF
各目录下 `xelatex <name>.tex`（中文需 ctex/xeCJK）。讲义图片可由 `math_notes/generate_figures.py`、`make_fig5.py`、`make_fig6.py` 重新生成。
