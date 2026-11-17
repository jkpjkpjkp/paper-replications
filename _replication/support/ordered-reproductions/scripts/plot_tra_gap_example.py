"""Plot the verified stock-history gap using actual recorded dates."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/gap-resolution-20261113'
f=pd.read_pickle(ROOT/'data/TRA/kdd21/feature.pkl')
calendar=pd.DatetimeIndex(f.index.get_level_values('datetime').unique()).sort_values()
stock=f.xs('SZ002080',level='instrument').loc[:'2020-06-29'];old=stock.index[-60:]
assert (old.year==2008).sum()==51 and (old.year==2020).sum()==9
end=calendar.get_loc(pd.Timestamp('2020-06-29'));recent=calendar[end-59:end+1];present=recent.isin(stock.index)
assert present.sum()==9
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
fig,axes=plt.subplots(2,1,figsize=(11,4.9));fig.patch.set_facecolor('#fcfcfb')
for ax in axes:
 ax.set_facecolor('#fcfcfb');ax.set_xlim(-.7,59.7);ax.set_ylim(0,1.7);ax.set_yticks([]);ax.set_xticks([])
 for spine in ax.spines.values():spine.set_visible(False)
axes[0].bar(np.arange(60),.55,bottom=.55,width=.88,color=['#d48c41' if d.year==2008 else '#187c89' for d in old])
axes[0].set_title('Previous input: 60 recorded observations spanning more than 12 years',loc='left',fontweight='bold',pad=4)
axes[0].text(25,.28,'51 observations from April–July 2008',ha='center',color='#785022')
axes[0].text(55,.28,'9 from June 2020',ha='center',color='#12616c')
axes[0].annotate('12-year jump',xy=(50.5,1.12),xytext=(44,1.48),arrowprops={'arrowstyle':'->','color':'#5a6066'},ha='center',fontsize=10)
axes[1].bar(np.arange(60),.55,bottom=.55,width=.88,color=np.where(present,'#187c89','#d9dfe2'))
axes[1].set_title('Corrected input: 60 actual market sessions ending June 29, 2020',loc='left',fontweight='bold',pad=4)
axes[1].text(25,.28,'51 unavailable sessions: explicit zero padding',ha='center',color='#525e65')
axes[1].text(55,.28,'9 real observations',ha='center',color='#12616c')
fig.text(.04,.025,'Actual example: SZ002080. Padding denotes missing features, not zero stock returns. The target and forecast row are unchanged.',fontsize=9,color='#525e65')
fig.subplots_adjust(left=.04,right=.98,top=.9,bottom=.12,hspace=.4)
for suffix in ['png','pdf']:
 fig.savefig(OUT/f'tra-history-gap-example.{suffix}',dpi=180,facecolor=fig.get_facecolor())
