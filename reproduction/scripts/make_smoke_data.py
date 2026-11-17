"""Synthetic fixture only: no claims about financial performance."""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(521)
rows = []
for firm in range(20):
    for t, date in enumerate(pd.date_range('1985-01-01', '2004-12-01', freq='MS')):
        scale = 100 + 10 * firm + t
        rows.append((date.strftime('%Y%m'), f'{firm:06d}', 1,
                     scale * (.6 + .03 * np.sin(t / 12)) + rng.normal(),
                     scale * (.08 + .01 * np.sin(t / 12)) + rng.normal(),
                     scale * 1.2 + rng.normal(), rng.uniform(), scale))
pd.DataFrame(rows, columns=['date', 'gvkey', 'active', 'saleq_ttm', 'oiadpq_ttm', 'atq_mrq', 'rel_mom1m', 'mrkcap']).to_csv(ROOT / 'results/synthetic.dat', sep=' ', index=False)
base = {
 'data_dir': str(ROOT / 'results'), 'datafile': 'synthetic.dat',
 'experiments_dir': str(ROOT / 'results'), 'financial_fields': 'saleq_ttm-atq_mrq',
 'aux_fields': 'rel_mom1m', 'target_field': 'oiadpq_ttm',
 'start_date': 199401, 'end_date': 199912, 'min_unrollings': 5, 'max_unrollings': 5,
 'stride': 12, 'forecast_n': 12, 'num_hidden': 16, 'num_layers': 1,
 'batch_size': 128, 'max_epoch': 2, 'early_stop': 25, 'learning_rate': .01,
 'recurrent_dropout': .25, 'dropout': 0, 'max_grad_norm': 1,
 'data_scaler': 'StandardScaler', 'log_squasher': False, 'use_custom_init': False,
 'initializer': 'GlorotUniform', 'seed': 521, 'train': True,
}
for mode, model in [('point', 'RNNPointEstimate'), ('uq', 'RNNUqRangeEstimate')]:
    config = dict(base, nn_type=model, UQ=mode=='uq', model_dir='smoke-'+mode, name='smoke-'+mode)
    (ROOT / 'configs' / ('smoke-'+mode+'.conf')).write_text(''.join(f'--{k} {v}\n' for k,v in config.items()))
print('Wrote synthetic fixture and point/UQ smoke configurations')
