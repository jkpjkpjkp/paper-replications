#!/usr/bin/env python3
"""Scientific diagnostic figures: predictors, future returns, calibration, P&L."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/factor-overlay-mpl')
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data import HERE,FACTORS

OUT=HERE/'diagnostics';PLOTS=HERE/'plots';PLOTS.mkdir(exist_ok=True)
COLORS=['#007f86','#d17b28']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'#ffffff','savefig.facecolor':'#ffffff'})

def finish(fig,name):
    fig.savefig(PLOTS/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(PLOTS/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)

def costs():
    f=pd.read_csv(OUT/'pre2018_paired_inference.csv')
    periods=['2005_2017','2018_2025']
    fig,axs=plt.subplots(2,3,figsize=(15,8),sharey=True)
    for factor,ax in zip(FACTORS,axs.flat):
        for i,p in enumerate(periods):
            q=f[(f.factor==factor)&(f.period==p)&(f.family=='h1')]
            gross=q[q.series=='gross_same_cost_holdings'].delta_mean.mean()*100
            net=q[q.series=='net'].delta_mean.mean()*100
            ax.plot([i-.16,i+.16],[gross,net],color=COLORS[i],lw=3)
            ax.scatter(i-.16,gross,marker='o',s=70,color=COLORS[i],label='Before fees' if i==0 else None)
            ax.scatter(i+.16,net,marker='s',s=60,color=COLORS[i],label='After fees' if i==0 else None)
            ax.annotate(f'{gross:+.2f}',(i-.16,gross),xytext=(-5,9),textcoords='offset points',ha='center',fontsize=10)
            ax.annotate(f'{net:+.2f}',(i+.16,net),xytext=(7,-16),textcoords='offset points',ha='center',fontsize=10)
        ax.axhline(0,color='#526068',lw=1);ax.set_xticks([0,1],['2005–2017','2018–2025']);ax.set_title(factor);ax.grid(axis='y',alpha=.15)
    axs[0,0].legend(frameon=False,fontsize=9,loc='lower left');axs[0,0].set_ylabel('Incremental return (percentage points/year)');axs[1,0].set_ylabel('Incremental return (percentage points/year)')
    fig.suptitle('Does the overlay earn enough before costs?\nCircle = gross improvement; square = net improvement; line = cost erosion',fontsize=17,y=1.02)
    fig.text(.5,-.035,'Same holdings before/after fee deductions. Mean across six fixed regression families; descriptive, not six independent studies.\nOriginal one-day-forecast overlay; 5 bp per dollar traded + 1% short borrow. Neither period produced a significant positive paired net gain after correction.',ha='center',fontsize=10)
    fig.tight_layout();finish(fig,'04_gross_to_net_before_after_2018')

def return_correlations():
    f=pd.read_csv(OUT/'weekly_timing_return_correlations.csv')
    q=f[(f.period=='2015_2025')&(f.series=='net')&(f.support=='all_dates')].set_index('factor').reindex(FACTORS)
    if q.correlation.isna().any():
        print('Return-correlation schema',f[['period','series','support']].drop_duplicates().to_string(index=False));return
    fig,ax=plt.subplots(figsize=(9,5));yy=np.arange(6)
    ax.errorbar(q.correlation,yy,xerr=[q.correlation-q.corr_ci_low,q.corr_ci_high-q.correlation],fmt='o',color=COLORS[0],capsize=4)
    ax.axvline(0,color='#526068');ax.set_yticks(yy,FACTORS);ax.invert_yaxis();ax.set_xlim(-.35,.25);ax.grid(axis='x',alpha=.15)
    ax.set_xlabel('Daily return correlation');ax.set_title('Actual weekly timing portfolio vs each plain factor baseline\n2015–2025, net returns; HAC 95% intervals')
    fig.text(.5,-.03,'All 2,766 days, including cash days. Same accounting/cost schedule; timing uses all eligible stocks,\nwhereas each factor needs its own characteristics. Low return correlation indicates diversification, not positive incremental expected return.',ha='center',fontsize=10)
    fig.tight_layout();finish(fig,'05_portfolio_return_correlations')

def calibration():
    p=OUT/'calibration_bins.csv'
    if not p.exists():return
    f=pd.read_csv(p);f['period']=f.period.str.replace('-','_');fig,axs=plt.subplots(2,3,figsize=(15,8))
    for factor,ax in zip(FACTORS,axs.flat):
        q=f[f.factor==factor]
        for period,color in zip(['2005_2017','2018_2025'],COLORS):
            g=q[q.period==period].sort_values('x_bps')
            assert len(g)==10,(factor,period,len(g))
            ax.errorbar(g.x_bps,g.y_bps,yerr=[g.y_bps-g.lo,g.hi-g.y_bps],fmt='o-',color=color,lw=1,capsize=2,markersize=4,label=period.replace('_','–'))
        limit=max(abs(q.x_bps.min()),abs(q.x_bps.max()))*1.05
        ax.plot([-limit,limit],[-limit,limit],ls='--',color='#91999d',lw=1,label='Perfect increment')
        ax.axhline(0,color='#526068',lw=1);ax.axvline(0,color='#bbb',lw=.7);ax.set_xlim(-limit,limit);ax.set_title(factor);ax.grid(alpha=.1)
        ax.set_xlabel('Added predicted five-day return (bp)')
    axs[0,0].set_ylabel('Realized return minus baseline forecast (bp)');axs[1,0].set_ylabel('Realized return minus baseline forecast (bp)');axs[0,0].legend(frameon=False,fontsize=9)
    fig.suptitle('Does a larger added forecast lead to a larger realized payoff?\nFixed ridge model, trained directly on five-day returns; ten forecast bins',fontsize=17,y=1.02)
    fig.text(.5,-.035,'Both axes demeaned within date and factor leg. Dots are realized held-share returns, not fitted returns.\nDate-level HAC 95% intervals; descriptive post-result diagnostics. These are sampled stock predictions, not portfolio returns.',ha='center',fontsize=10)
    fig.tight_layout();finish(fig,'03_added_forecast_vs_ground_truth')

def signal_matrices():
    path=OUT/'groundtruth_signal_correlations.csv'
    if not path.exists():return
    f=pd.read_csv(path);names=FACTORS+['clock_score'];labels=FACTORS+['Announcement']
    periods=['2005_2017','2018_2025']
    fig,axs=plt.subplots(1,2,figsize=(16,8))
    fig.subplots_adjust(wspace=.12,bottom=.16,top=.85)
    for panel,(period,ax) in enumerate(zip(periods,axs)):
        q=f[(f.period==period)&(f.metric=='spearman')]
        matrix=np.full((7,7),np.nan)
        for r in q.itertuples():
            if r.variable1 in names and r.variable2 in names:
                a=names.index(r.variable1);b=names.index(r.variable2);matrix[a,b]=matrix[b,a]=r.mean_correlation
        np.fill_diagonal(matrix,np.nan)
        im=ax.imshow(matrix,cmap='RdBu_r',vmin=-.35,vmax=.35)
        for i in range(7):
            for j in range(7):
                label='—' if i==j else 'N/A' if not np.isfinite(matrix[i,j]) else f'{matrix[i,j]:+.3f}'
                ax.text(j,i,label,ha='center',va='center',fontsize=10,color='white' if abs(matrix[i,j])>.25 else '#18303a')
        ax.set_xticks(range(7),labels,rotation=45,ha='right');ax.set_yticks(range(7),labels if panel==0 else ['']*7);ax.set_title(period.replace('_','–'));ax.spines[['top','right','left','bottom']].set_visible(False)
    fig.colorbar(im,ax=axs,shrink=.70,label='Mean within-date rank correlation',pad=.025)
    fig.suptitle('Do the stock signals agree or oppose each other?',fontsize=19,y=1.02)
    fig.text(.45,-.04,'Each number averages cross-sectional Spearman correlations on rebalance dates. Signed factor scores: higher = preferred long.\nAnnouncement score: +1 in forecast pre1–5, −1 in pre21–40, zero otherwise. Pairwise available stocks; availability differs by pair.\nN/A = insufficient variation in the overlap, not zero correlation. The announcement row is close to zero: largely different stock rankings.',ha='center',fontsize=10)
    finish(fig,'01_signal_correlation_matrices')
    fig,axs=plt.subplots(1,2,figsize=(15,6),sharey=True)
    for target,ax in zip(['return_1d','return_5d'],axs):
        for p,color,offset in zip(periods,COLORS,[-.13,.13]):
            q=f[(f.period==p)&(f.metric=='spearman')]
            vals=[];low=[];high=[]
            for name in names:
                r=q[((q.variable1==name)&(q.variable2==target))|((q.variable2==name)&(q.variable1==target))].iloc[0]
                vals.append(r.mean_correlation);low.append(r.ci_low);high.append(r.ci_high)
            vals=np.array(vals);yy=np.arange(7)+offset
            ax.errorbar(vals,yy,xerr=[vals-np.array(low),np.array(high)-vals],fmt='o',color=color,capsize=3,label=p.replace('_','–'))
        ax.axvline(0,color='#526068',lw=1);ax.set_yticks(range(7),labels);ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
        ax.set_title('Next-session realized return' if target=='return_1d' else 'Next-five-session realized return')
        ax.set_xlabel('Rank correlation with subsequent realized stock return');ax.legend(frameon=False)
    fig.suptitle('Ground truth: do higher scores actually precede higher returns?',fontsize=19,y=1.02)
    fig.text(.5,-.05,'Expanded scale to make small predictive relationships visible. Positive is the intended direction; horizontal bars are HAC12 95% intervals.\nAll eligible stocks with each characteristic, sampled every fifth session. Returns are realized after the score; five-day labels hold shares and retain dividends as cash.\nA predictor–predictor correlation near zero does not imply predictive correlation with returns. Intervals are descriptive, not multiplicity-adjusted.',ha='center',fontsize=10)
    fig.tight_layout();finish(fig,'02_signal_vs_ground_truth')

def main():
    signal_matrices();costs();return_correlations();calibration()

if __name__=='__main__':main()
