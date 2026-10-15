# 生成图5：小步 vs 大步时，反向后验 q(x_{t-1}|x_t) ∝ 似然 × q_{t-1} 的形状
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
fp="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(fp)
plt.rcParams["font.family"]=font_manager.FontProperties(fname=fp).get_name()
plt.rcParams["axes.unicode_minus"]=False
plt.rcParams["mathtext.fontset"]="cm"

mus=np.array([-2.0,2.0]); sd=0.6; w=np.array([0.5,0.5])
def q(x):  return sum(wi*np.exp(-(x-m)**2/(2*sd**2))/np.sqrt(2*np.pi*sd**2) for wi,m in zip(w,mus))
def dlogq(x):
    comps=[wi*np.exp(-(x-m)**2/(2*sd**2)) for wi,m in zip(w,mus)]
    num=sum(c*(-(x-m)/sd**2) for c,m in zip(comps,mus)); return num/sum(comps)
x=np.linspace(-5,9,6001); dx=x[1]-x[0]
def panel(ax,beta,xt,title,xlim=(-5,5),inset=None):
    a=1-beta; m=xt/np.sqrt(a); v=beta/a
    lik=np.exp(-(xt-np.sqrt(a)*x)**2/(2*beta))
    post=lik*q(x); post/=post.sum()*dx
    g=dlogq(m); approx=np.exp(-(x-(m+v*g))**2/(2*v))/np.sqrt(2*np.pi*v)
    pk=post.max()
    ax.fill_between(x,q(x)/q(x).max()*pk*0.55,color="#cbd5e1",alpha=0.7,label=r"边缘 $q_{t-1}(x_{t-1})$（缩放）")
    ax.plot(x,lik/lik.max()*pk*0.9,color="#f59e0b",lw=1.6,ls="--",label=r"似然 $q(x_t\,|\,x_{t-1})$（缩放）")
    ax.plot(x,post,color="#1d4ed8",lw=2.2,label=r"真实后验 $q(x_{t-1}\,|\,x_t)$")
    ax.plot(x,approx,color="#dc2626",lw=1.6,ls=":",label="一阶泰勒高斯近似")
    ax.axvline(m,color="#64748b",lw=0.8,ls="-.")
    ax.text(m,pk*1.03,r"$x_t/\sqrt{\alpha_t}$",ha="center",fontsize=10,color="#475569")
    ax.set_title(title,fontsize=12); ax.set_xlabel(r"$x_{t-1}$"); ax.set_yticks([])
    ax.set_ylim(0,pk*1.12); ax.set_xlim(*xlim)
    if inset:
        ia=ax.inset_axes([0.03,0.42,0.36,0.5])
        ia.plot(x,lik/(lik.sum()*dx),color="#f59e0b",lw=1.3,ls="--")
        ia.plot(x,post,color="#1d4ed8",lw=2); ia.plot(x,approx,color="#dc2626",lw=1.4,ls=":")
        ia.axvline(m,color="#64748b",lw=0.8,ls="-.")
        ia.set_xlim(*inset); ia.set_yticks([]); ia.tick_params(labelsize=7)
        ia.set_title("放大：后验相对似然向右侧高密度处微移",fontsize=7.5)
fig,axs=plt.subplots(1,2,figsize=(11,3.9))
panel(axs[0],0.01,0.6,inset=(0.2,1.1),title=r"小步 $\beta_t=0.01$：窗口很窄 → 后验≈高斯")
panel(axs[1],0.6,0.5,xlim=(-5,8),title=r"大步 $\beta_t=0.6$：窗口很宽 → 后验双峰、非高斯")
h,l=axs[0].get_legend_handles_labels()
fig.legend(h,l,loc="lower center",ncol=4,frameon=False,fontsize=9.5,bbox_to_anchor=(0.5,0.0))
fig.tight_layout(rect=(0,0.11,1,1))
fig.savefig("figures/fig5_reverse_gaussian_smallstep.png",dpi=160,facecolor="white")
