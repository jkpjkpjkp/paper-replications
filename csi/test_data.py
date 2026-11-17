"""Data invariants for the daily CSI adaptation, using synthetic fixtures."""
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT', '1')

import numpy as np
import pandas as pd

from data import (alpha158_config, active_members, init_worker, extract_instrument,
                  label_plan, parse_membership, vwap_labels)


class DataInvariants(unittest.TestCase):
    def test_membership_is_historical_and_inclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'members.txt'
            p.write_text('000001.SZ\t2020-01-01\t2020-01-03\n000002.SZ\t2020-01-03\t2020-01-05\n')
            m = parse_membership(p)
            self.assertEqual(active_members(m, '2020-01-02'), ['000001.SZ'])
            self.assertEqual(active_members(m, '2020-01-03'), ['000001.SZ', '000002.SZ'])
            self.assertEqual(active_members(m, '2020-01-04'), ['000002.SZ'])

    def test_vwap_offsets_and_factor_adjustment(self):
        vwap = np.array([[999., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20.],
                         [999., 100., 51., 52., 53., 54., 55., 56., 57., 58., 59., 60.]])
        factor = np.ones_like(vwap)
        factor[1, 2:] = 2
        y = vwap_labels(vwap, factor, 0)
        np.testing.assert_allclose(y[0], np.arange(1, 11) / 10, rtol=1e-6)
        np.testing.assert_allclose(y[1, 0], 0.02, rtol=1e-6)
        changed = vwap.copy()
        changed[:, 0] = 1
        np.testing.assert_array_equal(vwap_labels(changed, factor, 0), y)

    def test_invalid_label_observations_remain_missing(self):
        v = np.ones((3, 12), dtype=float)
        f = np.ones_like(v)
        v[0, 4] = np.nan
        f[1, 1] = 0
        f[2, 2] = -1
        y = vwap_labels(v, f, 0)
        self.assertTrue(np.isnan(y[0, 2]))
        self.assertTrue(np.isnan(y[1]).all())
        self.assertTrue(np.isnan(y[2, 0]))
        self.assertTrue(np.isfinite(y[0, -1]))

    def test_all_horizons_purged_at_split_boundary(self):
        dates = pd.bdate_range('2023-06-01', '2023-07-31').strftime('%Y-%m-%d').to_numpy()
        plan, reason = label_plan(dates, 0)
        self.assertIsNone(reason)
        self.assertEqual(plan['entry_date'], dates[1])
        self.assertEqual(plan['label_exit_dates'][0], dates[2])
        self.assertEqual(plan['label_exit_dates'][-1], dates[11])
        index = int(np.flatnonzero(dates == '2023-06-23')[0])
        self.assertEqual(label_plan(dates, index), (None, 'split_boundary_purge'))

    def test_test_label_extension_is_allowed(self):
        dates = pd.bdate_range('2025-07-01', '2025-08-29').strftime('%Y-%m-%d').to_numpy()
        index = int(np.flatnonzero(dates == '2025-07-31')[0])
        plan, reason = label_plan(dates, index)
        self.assertIsNone(reason)
        self.assertEqual(plan['split'], 'test')
        self.assertGreater(plan['label_exit_dates'][-1], '2025-07-31')
        self.assertEqual(label_plan(dates, len(dates) - 1), (None, 'outside_decision_period'))

    def test_alpha158_names_and_adjusted_price_expressions(self):
        expressions, names = alpha158_config()
        self.assertEqual(len(names), 158)
        self.assertEqual(len(set(names)), 158)
        self.assertTrue(all('$factor' in expression for expression in expressions[:13]))
        self.assertFalse(any('Ref(' in e and ', -' in e for e in expressions))
        self.assertIn('Mean(($close*$factor), 5)/($close*$factor)', expressions)

    def test_real_qlib_features_are_asof_only(self):
        # Two providers agree through t and radically differ after t; all
        # feature values at t must agree. The synthetic raw factor jumps also
        # exercise adjusted feature construction independently of labels.
        import qlib
        from qlib.config import REG_CN
        from qlib.data import D
        from qlib.data.cache import H
        dates = pd.bdate_range('2020-01-01', periods=90)
        expressions, _ = alpha158_config()
        outputs = []
        rng = np.random.default_rng(17)
        base = 10 + np.cumsum(rng.normal(scale=0.05, size=90))
        with tempfile.TemporaryDirectory() as tmp:
            for variant in range(2):
                source = Path(tmp) / str(variant)
                (source / 'calendars').mkdir(parents=True)
                (source / 'calendars/day.txt').write_text('\n'.join(dates.strftime('%Y-%m-%d')) + '\n')
                (source / 'instruments').mkdir()
                (source / 'instruments/all.txt').write_text('SYNTH\t2020-01-01\t2021-01-01\n')
                folder = source / 'features/synth'
                folder.mkdir(parents=True)
                for field in ['open', 'high', 'low', 'close', 'vwap', 'volume', 'factor']:
                    if field == 'volume':
                        values = np.arange(100, 190, dtype=np.float32)
                    elif field == 'factor':
                        values = np.ones(90, dtype=np.float32)
                    else:
                        values = base.copy().astype(np.float32)
                        if field == 'high': values += 0.1
                        if field == 'low': values -= 0.1
                    if variant:
                        values[76:] *= 100
                    np.concatenate([np.array([0], dtype=np.float32), values]).tofile(folder / (field + '.day.bin'))
                H.clear()
                qlib.init(provider_uri=str(source), region=REG_CN, kernels=1, logging_level=50)
                frame = D.features(['SYNTH'], expressions, start_time=str(dates[75].date()), end_time=str(dates[75].date()))
                outputs.append(frame.to_numpy())
            np.testing.assert_allclose(outputs[0], outputs[1], equal_nan=True)
            self.assertTrue(np.isfinite(outputs[0]).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
