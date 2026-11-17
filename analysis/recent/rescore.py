"""Recompute recent P&L/cost artifacts from fixed predictions, without refitting."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from hermes.evaluation import evaluated_artifacts
from hermes.metrics import PORTFOLIO_PROTOCOL,evaluate


def main():
    for path in sorted((ROOT/'results/recent').glob('*/metrics.json')):
        run=path.parent;result=json.loads(path.read_text())
        for split in ['validation','test']:
            array_path=run/f'{split}_predictions.npz'
            if not array_path.exists():continue
            with np.load(array_path) as source:d={key:source[key] for key in source.files}
            metrics,arrays=evaluated_artifacts(d['prediction'],d['target'],d['valid'],d['dates'],d['tickers'],5,d['pnl_target'])
            np.testing.assert_allclose(metrics['IC'],result[split]['IC'],atol=1e-12)
            np.savez_compressed(array_path,**arrays);result[split]=metrics
        result['evaluation_protocol']=PORTFOLIO_PROTOCOL
        if (run/'config.json').exists():
            config=json.loads((run/'config.json').read_text());config['evaluation_protocol']=PORTFOLIO_PROTOCOL
            (run/'config.json').write_text(json.dumps(config,indent=2)+'\n')
            data=np.load(config['data']);test_days=np.arange(config['val_end'],len(data['returns']))
            result['previous_day_return_control']=evaluate(data['returns'][test_days-1],d['target'],d['valid'],5,data['pnl_returns'][test_days])[0]
        path.write_text(json.dumps(result,indent=2)+'\n')
        print(run.name,'net SR',result['test']['net_SR_annualized'])


if __name__=='__main__':main()
