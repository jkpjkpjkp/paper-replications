"""Regression checks for adjustments, exit labels, cache identity and universe timing."""
import json
from pathlib import Path
import tempfile

import common
from common import LocalStockData, adjusted_values, fresh_output, normalize_members
from integration import ForwardReturn, LocalAlphaPoolGFN
import numpy as np
import pandas as pd
import torch
from alphagen.data.expression import Feature, Rank, TsRank, Sign, Pow
from alphagen_qlib.stock_data import FeatureType


def fixture(root):
    dates = pd.bdate_range('2009-01-01', periods=250)
    (root/'calendars').mkdir(parents=True)
    (root/'instruments').mkdir()
    (root/'calendars/day.txt').write_text('\n'.join(dates.strftime('%Y-%m-%d'))+'\n')
    intervals = [('AAA', 0, 110), ('BBB', 0, 249), ('FUT', 150, 249)]
    (root/'instruments/csi300.txt').write_text(''.join(
        f'{stock}\t{dates[low].date()}\t{dates[high].date()}\n' for stock, low, high in intervals))
    for number, (stock, _, _) in enumerate(intervals):
        folder = root/'features'/stock.lower()
        folder.mkdir(parents=True)
        factor = np.where(np.arange(250) >= 108, 2., 1.)
        adjusted = (100 + 20*number) * (1.001 + .0001*number)**np.arange(250)
        for feature in ['open', 'close', 'high', 'low', 'vwap', 'volume', 'factor']:
            values = (factor if feature == 'factor' else
                      100*factor if feature == 'volume' else adjusted/factor)
            np.r_[0, values].astype('<f4').tofile(folder/(feature+'.day.bin'))
    return dates


def main():
    torch.set_num_threads(2)
    frame = pd.DataFrame({'$close':[100., 50.], '$open':[100., 50.],
                          '$volume':[100., 200.], '$factor':[1., 2.]})
    values, close = adjusted_values(frame, [FeatureType.CLOSE, FeatureType.VOLUME])
    np.testing.assert_array_equal(values, [[100., 10000.], [100., 10000.]])
    invalid = frame.copy()
    invalid.loc[1, '$factor'] = 0
    try:
        adjusted_values(invalid, [FeatureType.CLOSE])
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid observed adjustment factor was accepted')
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        common.DATA, common.CACHE = root/'provider', root/'cache'
        dates = fixture(common.DATA)
        common.initialize()
        short = LocalStockData('csi300', str(dates[100].date()), str(dates[115].date()))
        long = LocalStockData('csi300', str(dates[100].date()), str(dates[175].date()))
        assert short.n_stocks == 2 and long.n_stocks == 3
        index = list(short._stock_ids).index('AAA')
        # Day 100's endpoint is after the day-110 exit, but the outcome exists.
        returns = short.forward_returns()
        np.testing.assert_allclose(returns[0,index], 1.001**20-1, atol=2e-7)
        assert torch.isnan(returns[11:,index]).all(), 'Post-exit formations must be excluded'
        assert torch.isnan(short.data[111:,1,index]).all()
        assert torch.isfinite(short.full_close[120,index])
        expressions = [Feature(FeatureType.CLOSE), TsRank(Feature(FeatureType.HIGH), 20),
                       Rank(TsRank(Feature(FeatureType.HIGH), 20)),
                       Rank(Sign(Feature(FeatureType.CLOSE))),
                       Rank(Pow(Feature(FeatureType.CLOSE), 0))]
        maximum = 0.
        for expression in expressions:
            a = short.normalize(expression.evaluate(short))
            b = long.normalize(expression.evaluate(long))[:short.n_days,:short.n_stocks]
            torch.testing.assert_close(a,b,atol=2e-5,rtol=1e-5,equal_nan=True)
            difference = torch.abs(a-b)
            if torch.isfinite(difference).any():
                maximum = max(maximum, float(difference[torch.isfinite(difference)].max()))
        tied = Rank(Feature(FeatureType.CLOSE))._apply(torch.tensor([[1.,1.,2.,float('nan')]]))
        torch.testing.assert_close(tied, torch.tensor([[1/6,1/6,2/3,float('nan')]]), equal_nan=True)
        pool = LocalAlphaPoolGFN(2, short, ForwardReturn())
        assert torch.isnan(pool.target[11:,index]).all()
        # Missing labels and future-only columns cannot dilute the SSL MSE.
        a = torch.tensor([[1.,2.], [3.,4.]])
        b = a+1
        pool.values[0] = b
        first = pool._compute_consistency_loss(a,[0],torch.tensor([1.]))
        pool.values[0] = torch.cat([b,torch.full((2,1),torch.nan)],dim=1)
        second = pool._compute_consistency_loss(torch.cat([a,torch.full((2,1),torch.nan)],dim=1),[0],torch.tensor([1.]))
        assert first == second == 1.
        # A content change must change cache identity even at the same provider path.
        before = short.provenance['cache_key']
        file = common.DATA/'features/aaa/close.day.bin'
        content = np.fromfile(file,dtype='<f4')
        content[102] *= 1.01
        content.tofile(file)
        common.initialize()
        changed = LocalStockData('csi300', str(dates[100].date()), str(dates[115].date()))
        assert changed.provenance['cache_key'] != before
        assert changed.data[101,1,index] != short.data[101,1,index]
        # Existing partial artifacts and linked output roots must be protected.
        used = fresh_output(root/'output')
        (used/'progress.json').write_text('{}')
        for path in [used]:
            try:
                fresh_output(path)
            except ValueError:
                pass
            else:
                raise AssertionError('Existing run was accepted for overwrite')
        (root/'linked').symlink_to(used, target_is_directory=True)
        try:
            fresh_output(root/'linked'/'new')
        except ValueError:
            pass
        else:
            raise AssertionError('Output through archive/input link was accepted')
    print(json.dumps(dict(status='passed', corporate_action_return_and_volume_identity=True,
        invalid_factors_rejected=True, exit_labels_retained=True, post_exit_formations_excluded=True,
        future_universe_normalization_max_error=maximum, deterministic_average_rank_ties=True,
        missing_training_labels_preserved=True, ssl_unaffected_by_absent_columns=True,
        input_content_invalidates_cache=True, existing_outputs_protected=True), indent=2))


if __name__ == '__main__':
    main()
