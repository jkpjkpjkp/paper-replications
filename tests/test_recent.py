from pathlib import Path
import numpy as np
import pytest
from analysis.recent.prepare import prior_eligibility


def test_recent_tradability_depends_only_on_preceding_history():
    observed=np.ones((30,4),dtype=bool);delisted=np.zeros_like(observed)
    original=prior_eligibility(observed,delisted)
    observed[20:]=False;delisted[20:]=True
    altered=prior_eligibility(observed,delisted)
    np.testing.assert_array_equal(original[:21],altered[:21])
    assert original[16].all() and not original[15].any()
    assert not altered[21].any()


def test_recent_fold_normalizers_and_chronological_boundaries():
    root=Path(__file__).resolve().parents[1]
    if not (root/'data/recent/fold-2025.npz').exists():pytest.skip('private recent-period CRSP inputs not prepared')
    for year in [2023,2024,2025]:
        d=np.load(root/f'data/recent/fold-{year}.npz')
        train=d['dates']<f'{year-1}-01-01'
        val=(d['dates']>=f'{year-1}-01-01')&(d['dates']<f'{year}-01-01')
        test=d['dates']>=f'{year}-01-01'
        assert train.any() and val.any() and test.any()
        assert d['dates'][train][-1]<d['dates'][val][0]<d['dates'][test][0]
        np.testing.assert_allclose(d['features'][train].mean(0),0,atol=2e-5)
        assert d['dates'][test][-1].startswith(str(year))
        assert np.isfinite(d['pnl_returns'][d['valid']]).all()
