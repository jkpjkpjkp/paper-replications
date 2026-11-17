"""Recompute all eight Table 1 metrics without drawing new routing samples."""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / 'ordered-reproductions/data/TRA/kdd21'
CODE = ROOT / 'sources/examples/benchmarks/TRA'
TARGET = dict(MSE=.157, MAE=.318, IC=.059, ICIR=.460,
              AR=.124, AV=.140, SR=.885, MDD=.204)


def evaluate(path, output=None):
    pred = pd.read_pickle(path).loc['2018-09-21':'2020-06-30'].copy()
    assert pred.index.names == ['datetime', 'instrument']
    assert not pred.index.has_duplicates
    assert len(pred) == 340790
    assert np.isfinite(pred[['score', 'label']].values).all()
    returns = pd.read_pickle(DATA / 'ret.pkl').clip(-.1, .1)
    pred['ret'] = returns
    assert pred.ret.notna().all()

    # Use the actual released notebook function as a separate reference.
    nb = json.loads((CODE / 'Reports.ipynb').read_text())
    cell = ''.join(nb['cells'][1]['source'])
    parsed = ast.parse('\n'.join(s for s in cell.splitlines() if not s.startswith('%')))
    fn = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == 'func')
    scope = {'pd': pd}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<author notebook func>', 'exec'), scope)
    records = []
    max_error = 0.
    for date, group in pred.groupby(level='datetime', sort=True):
        n = len(group)
        a = rankdata(group.score.to_numpy(), method='average') / n
        b = rankdata(group.label.to_numpy(), method='average') / n
        # Match pandas nlargest/nsmallest's original-row tie policy explicitly.
        long_order = np.argsort(-a, kind='stable')[:80]
        short_order = np.argsort(a, kind='stable')[:80]
        ret = group.ret.to_numpy()
        row = dict(MSE=np.mean((a-b)**2), MAE=np.mean(np.abs(a-b)),
                   IC=np.corrcoef(a, b)[0, 1],
                   R=np.mean(ret[long_order])-np.mean(ret[short_order]))
        native = scope['func'](group)
        for key, value in row.items():
            error = abs(value - native[key])
            max_error = max(max_error, error)
            assert error < 1e-10, (date, key, value, native[key])
        records.append(dict(datetime=date, **row))
    daily = pd.DataFrame(records).set_index('datetime')
    assert len(daily) == 426
    r = daily.R.reindex(pd.date_range(daily.index[0], daily.index[-1])).fillna(0.)
    metrics = {k: float(daily[k].mean()) for k in ['MSE', 'MAE', 'IC']}
    metrics.update(ICIR=float(daily.IC.mean()/daily.IC.std(ddof=1)),
                   AR=float(r.mean()*365), AV=float(r.std(ddof=1)*np.sqrt(365)),
                   SR=float(r.mean()/r.std(ddof=1)*np.sqrt(365)),
                   MDD=float((r.cumsum().cummax()-r.cumsum()).max()))
    info = Path(path).parent / 'info.json'
    if info.exists():
        reported = json.loads(info.read_text())['metric']
        for key in ['MSE', 'MAE', 'IC', 'ICIR']:
            assert np.isclose(metrics[key], reported[key], atol=1e-10, rtol=0), key
    result = dict(metrics=metrics, rows=len(pred), trading_days=len(daily),
                  calendar_days=len(r), native_notebook_max_absolute_error=max_error,
                  definitions='Author notebook: ranks, top/bottom 80, clipped released ret, calendar zeros, ddof=1, arithmetic MDD',
                  comparison={k: dict(paper=v, reproduced=metrics[k], delta=metrics[k]-v)
                              for k, v in TARGET.items()})
    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        daily.to_csv(output/'daily.csv')
        r.to_csv(output/'calendar-returns.csv', header=['return'])
        (output/'evaluation.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('prediction')
    parser.add_argument('--output')
    args = parser.parse_args()
    print(json.dumps(evaluate(args.prediction, args.output), indent=2))
