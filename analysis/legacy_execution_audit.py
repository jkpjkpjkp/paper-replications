"""Check original forecasts using CRSP tradability and total-return execution."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hermes.evaluation import evaluated_artifacts


def main():
    old=np.load(ROOT/'results/sp500-no_skip-seed0/test_predictions.npz')
    current=np.load(ROOT/'data/recent/frozen.npz')
    lookup={day:i for i,day in enumerate(current['dates'])}
    index=np.array([lookup[day] for day in old['dates']])
    metrics,_=evaluated_artifacts(old['prediction'],current['returns'][index],current['valid'][index],
                                 old['dates'],old['tickers'],5,current['pnl_returns'][index])
    violations=np.sum((old['portfolio_weights']!=0)&~current['valid'][index])
    result={'method':'same original predictions; separate CRSP total-return/tradability execution audit',
            'test_dates':[str(old['dates'][0]),str(old['dates'][-1])],
            'legacy_held_security_days_not_eligible_on_crsp':int(violations),'test':metrics}
    (ROOT/'results/legacy_execution_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Legacy execution audit:',metrics['mean_net_daily_return']*10000,'net bp/day')


if __name__=='__main__':main()
