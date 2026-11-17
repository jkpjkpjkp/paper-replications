from pathlib import Path
import sys
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ocp import run,controller
EXT=ROOT/'extensions';OUT=EXT/'figures'
C={'UP-paper':'#cf3e35','P':'#7956a1','SFOGD':'#a37b13','KT':'#377ac1','DtACI':'#299362','PI-expanded':'#555555'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'savefig.dpi':180})

def save(fig,name):
 for ext in ['png','svg']:fig.savefig(OUT/f'{name}.{ext}',bbox_inches='tight')
 plt.close(fig)

def main():
 d=pd.read_csv(EXT/'scenario_screen/confirmation/fixed_summary.csv')
 primary='smooth_correlated_lognormal_sigma15_amplitude8'
 sub=d[d.scenario==primary].set_index('method')
 methods=['UP-paper','P','SFOGD','KT']
 names=['UP-OCP\nAlgorithm 1','P control','SF-OGD','KT']
 fig,axs=plt.subplots(1,3,figsize=(14,4.7),layout='constrained')
 for i,(m,n) in enumerate(zip(methods,names)):
  r=sub.loc[m]
  axs[0].bar(i,r.mean_width,color=C[m],width=.65)
  axs[0].text(i,r.mean_width+.1,f'{r.mean_width:.3f}',ha='center',fontsize=10)
  axs[1].errorbar(i,r.coverage,yerr=r.coverage_sem,fmt='o',capsize=4,color=C[m],ms=7)
 axs[0].set(xticks=range(4),xticklabels=names,ylabel='Mean interval width',title='Width on 30 unseen paths',ylim=(0,13))
 axs[1].set(xticks=range(4),xticklabels=names,ylabel='Realized coverage',title='Comparable coverage',ylim=(.945,.96))
 axs[1].axhline(.95,color='black',ls='--',lw=1)
 axs[1].yaxis.set_major_formatter(PercentFormatter(1))
 pairs=pd.read_csv(EXT/'scenario_screen/confirmation/paired_comparisons.csv')
 for i,m in enumerate(['P','SFOGD','KT']):
  r=pairs[(pairs.scenario==primary)&(pairs.baseline==m)].iloc[0]
  axs[2].errorbar(r.saving_percent,i,xerr=[[r.saving_percent-r.saving_ci_low],[r.saving_ci_high-r.saving_percent]],fmt='o',capsize=4,color=C[m])
  axs[2].text(r.saving_ci_high+1,i,f'{r.saving_percent:.1f}%',va='center',fontsize=10)
 axs[2].set(yticks=range(3),yticklabels=['versus P','versus SF-OGD','versus KT'],xlabel='UP width reduction (%)',title='Paired 95% bootstrap intervals',xlim=(0,56))
 axs[2].invert_yaxis()
 fig.suptitle('Confirmed scenario: smooth, clustered lognormal error magnitudes\nSettings selected on seeds 0–4; evaluated on seeds 100–129',fontsize=14)
 save(fig,'confirmed_scenario')

 z=np.load(EXT/'scenario_screen/confirmation/scores.npz')
 scores=z[primary][0]
 t=np.arange(1000,1500)
 fig,axs=plt.subplots(2,1,figsize=(12,6),layout='constrained',sharex=True)
 axs[0].plot(t,scores[t],color='black',lw=1,label='Observed score')
 for m,a,lr in [('UP-paper',.035,1),('P',.04,.5),('KT',.03,1)]:
  q=controller(scores,a,lr,100) if m=='P' else run(scores,m,a,lr)
  axs[0].plot(t,q[t],color=C[m],label=m,lw=1.3)
  axs[1].plot(t,pd.Series(scores<=q).rolling(100).mean().to_numpy()[t],color=C[m],lw=1.2)
 axs[0].set(ylabel='Radius / observed score',title='First confirmation path (seed 100): a fixed 500-step view')
 axs[0].legend(ncol=4,fontsize=9)
 axs[1].axhline(.95,color='black',ls='--',lw=1)
 axs[1].set(ylabel='100-step local coverage',xlabel='Time step')
 axs[1].yaxis.set_major_formatter(PercentFormatter(1))
 save(fig,'clustered_error_dynamics')

 d=pd.read_csv(EXT/'country_confirmation/confirmation_results.csv')
 countries=['Canada','United Kingdom','Germany','Japan']
 fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained',sharey=True)
 for ax,init in zip(axs,['cold','warm']):
  for j,m in enumerate(['UP-paper','P','SFOGD','KT','DtACI']):
   p=d[(d.initialization==init)&(d.method==m)].set_index('country').loc[countries]
   ax.plot(range(4),p.coverage,marker='o',label=m,color=C[m],lw=1.3)
  ax.axhline(.95,color='black',ls='--',lw=1)
  ax.set(xticks=range(4),xticklabels=['Canada','UK','Germany','Japan'],title='Common '+init+' initialization',ylim=(.68,1))
  ax.yaxis.set_major_formatter(PercentFormatter(1))
 axs[0].set_ylabel('Realized coverage')
 axs[0].legend(fontsize=8,ncol=2,loc='lower left')
 fig.suptitle('New real-data test: frozen US settings transferred to four countries\nAll country choices registered before download; DtACI mean width is infinite in every case',fontsize=13)
 save(fig,'country_transfer')
 print('Wrote 3 PNG/SVG pairs:',OUT)

if __name__=='__main__':main()
