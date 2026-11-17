"""Gross/net FF5+UMD alphas with HAC confidence intervals, separated by sample."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
HERE=Path(__file__).resolve().parent
d=pd.read_csv(HERE/'results_summary.csv')
font_manager.fontManager.addfont('/usr/share/fonts/google-droid-sans-fonts/DroidSansFallbackFull.ttf')
plt.rcParams.update({'font.family':['DejaVu Sans','Droid Sans Fallback'],'axes.unicode_minus':False,'font.size':11})
fig,axes=plt.subplots(1,2,figsize=(12.5,5.3),gridspec_kw={'width_ratios':[1.3,1]})
groups=[('月频 · 全美合格普通股 · 2022–2025',['DoubleAdapt','TRA','AlphaSAGE','AlphaPROBE']),
        ('已有日频实验 · 不同大盘股样本与测试期',['LiMT','PRISM-VQ'])]
for ax,(title,names) in zip(axes,groups):
    for i,name in enumerate(names):
        for cost,offset,color,label in [(0,-.12,'#8796A5','毛 alpha'),(10,.12,'#0072B2','净 alpha（10bp）')]:
            r=d.loc[d.paper.eq(name)&d.cost_bps.eq(cost)]
            if r.empty:continue
            r=r.iloc[0];value=r.annual_alpha*100
            ax.errorbar(value,i+offset,xerr=[[value-r.ci_low*100],[r.ci_high*100-value]],fmt='o',color=color,capsize=3,
                        label=label if i==0 else None,markersize=6,lw=1.5)
    ax.set_yticks(range(len(names)),names);ax.invert_yaxis();ax.axvline(0,color='#A0A8AF',ls='--',lw=1)
    ax.set_title(title,fontsize=12,pad=16);ax.set_xlabel('年化 alpha (%)，误差线为 95% HAC 区间')
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.18)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.925),ncol=2,frameon=False,fontsize=10)
fig.suptitle('CZ202 市值加权 D10−D1：FF5+UMD alpha',fontsize=18,y=.98)
fig.text(.5,.025,'LiMT 测试期 2017–2025；PRISM-VQ 测试期 2022–2024。各实验均剔除上月末 NYSE20 以下股票。',ha='center',fontsize=10)
fig.subplots_adjust(left=.11,right=.98,bottom=.17,top=.76,wspace=.45)
fig.savefig(HERE/'alpha_comparison.png',dpi=180,facecolor='white')
fig.savefig(HERE/'alpha_comparison.svg',facecolor='white')
