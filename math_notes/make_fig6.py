# 生成图6：重参数化——计算图对比 + 两个独立高斯噪声的合并
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
fp="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(fp)
plt.rcParams["font.family"]=font_manager.FontProperties(fname=fp).get_name()
plt.rcParams["axes.unicode_minus"]=False
plt.rcParams["mathtext.fontset"]="cm"

def box(ax,xy,text,fc,w=0.26,h=0.13,ls="-"):
    x,y=xy
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle="round,pad=0.015",fc=fc,ec="#334155",lw=1.2,ls=ls))
    ax.text(x,y,text,ha="center",va="center",fontsize=10.5)
def arrow(ax,a,b,color="#334155",ls="-",lw=1.4,rad=0.0):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=13,color=color,ls=ls,lw=lw,
                                 connectionstyle=f"arc3,rad={rad}"))

fig,axs=plt.subplots(1,3,figsize=(14,4.3),gridspec_kw=dict(width_ratios=[1,1,1.05]))
# (a) 无重参数化
ax=axs[0]; ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
ax.set_title("(a) 直接采样：随机节点挡住梯度",fontsize=12)
box(ax,(0.18,0.80),r"参数 $\phi$","#e0e7ff")
box(ax,(0.18,0.52),r"$\mu_\phi,\ \sigma_\phi$","#e0e7ff")
box(ax,(0.55,0.52),"采样\n"+r"$z\sim\mathcal{N}(\mu_\phi,\sigma_\phi^2)$","#fde68a",w=0.36,h=0.2,ls="--")
box(ax,(0.55,0.17),r"损失 $f(z)$","#fecaca")
arrow(ax,(0.18,0.73),(0.18,0.59)); arrow(ax,(0.31,0.52),(0.37,0.52)); arrow(ax,(0.55,0.42),(0.55,0.24))
arrow(ax,(0.68,0.24),(0.68,0.42),color="#dc2626",ls="--")
ax.text(0.68,0.33,r"$\times$",color="#dc2626",fontsize=26,ha="center",va="center",fontweight="bold")
ax.text(0.76,0.33,r"$\partial z/\partial\phi$"+"\n无定义",color="#dc2626",ha="left",va="center",fontsize=10)
ax.text(0.5,0.0,"「从分布里抽一个数」不是 $\\phi$ 的可微函数",ha="center",fontsize=9.5,color="#475569")
# (b) 重参数化
ax=axs[1]; ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
ax.set_title("(b) 重参数化：随机性移到外部输入",fontsize=12)
box(ax,(0.18,0.80),r"参数 $\phi$","#e0e7ff")
box(ax,(0.18,0.52),r"$\mu_\phi,\ \sigma_\phi$","#e0e7ff")
box(ax,(0.82,0.80),r"$\epsilon\sim\mathcal{N}(0,1)$"+"\n（无参数）","#fde68a",w=0.3,h=0.17,ls="--")
box(ax,(0.55,0.52),r"$z=\mu_\phi+\sigma_\phi\,\epsilon$"+"\n（确定性、可微）","#bbf7d0",w=0.36,h=0.2)
box(ax,(0.55,0.17),r"损失 $f(z)$","#fecaca")
arrow(ax,(0.18,0.73),(0.18,0.59)); arrow(ax,(0.31,0.52),(0.37,0.52)); arrow(ax,(0.78,0.71),(0.66,0.62))
arrow(ax,(0.55,0.42),(0.55,0.24))
arrow(ax,(0.45,0.24),(0.45,0.42),color="#dc2626",ls="--"); arrow(ax,(0.37,0.47),(0.28,0.47),color="#dc2626",ls="--")
arrow(ax,(0.10,0.59),(0.10,0.73),color="#dc2626",ls="--")
ax.text(0.03,0.35,"梯度\n"+r"$\nabla_\phi$",color="#dc2626",fontsize=10.5)
ax.text(0.5,0.0,r"$\nabla_\phi\,\mathbb{E}_\epsilon[f(\mu_\phi+\sigma_\phi\epsilon)]=\mathbb{E}_\epsilon[\nabla_\phi f(\cdot)]$",ha="center",fontsize=10.5,color="#475569")
# (c) 噪声合并
ax=axs[2]
at,atm1=0.7,0.6
a=np.sqrt(at*(1-atm1)); b=np.sqrt(1-at); s=np.sqrt(a*a+b*b)
rng=np.random.default_rng(0); n=200000
samp=a*rng.standard_normal(n)+b*rng.standard_normal(n)
xs=np.linspace(-3,3,600)
pdf=lambda x,sd: np.exp(-x*x/(2*sd*sd))/np.sqrt(2*np.pi)/sd
ax.hist(samp,bins=120,density=True,color="#93c5fd",alpha=0.75,label=r"$a\epsilon_{t-1}+b\epsilon_t$ 的样本直方图")
ax.plot(xs,pdf(xs,a),color="#f59e0b",ls="--",lw=1.4,label=rf"$a\epsilon_{{t-1}}$：方差 $a^2={a*a:.2f}$")
ax.plot(xs,pdf(xs,b),color="#10b981",ls="--",lw=1.4,label=rf"$b\epsilon_t$：方差 $b^2={b*b:.2f}$")
ax.plot(xs,pdf(xs,s),color="#1d4ed8",lw=2.2,label=rf"$\sqrt{{a^2+b^2}}\,\bar\epsilon$：方差 ${s*s:.2f}=1-\alpha_t\alpha_{{t-1}}$")
ax.set_title(rf"(c) 两个独立噪声合并为一个（$\alpha_t={at},\ \alpha_{{t-1}}={atm1}$）",fontsize=12)
ax.set_yticks([]); ax.legend(fontsize=8.6,loc="upper left",frameon=False); ax.set_xlim(-3,3)
ax.set_ylim(0,pdf(0,min(a,b))*1.45)
ax.set_xlabel(r"$a=\sqrt{\alpha_t(1-\alpha_{t-1})},\ b=\sqrt{1-\alpha_t}$",fontsize=10)
fig.tight_layout()
fig.savefig("figures/fig6_reparameterization.png",dpi=160,facecolor="white")
print(a*a,b*b,s*s,samp.var())
