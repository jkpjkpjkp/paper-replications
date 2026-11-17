"""Copy the pinned release, applying narrow, reviewable local repairs."""
from pathlib import Path
import shutil, subprocess, difflib
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'Lookahead'
REV = 'ac6f47c9a36f681920314423e2502f3654c5b592'
assert subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() == REV
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=ROOT / 'runtime')
TARGET = parser.parse_args().output
if TARGET.exists():
    raise SystemExit('runtime exists; preserve it rather than overwrite local changes')
shutil.copytree(SOURCE, TARGET, ignore=shutil.ignore_patterns('.git', '__pycache__'))

def edit(name, old, new):
    p = TARGET / name
    s = p.read_text()
    assert old in s, (name, old)
    p.write_text(s.replace(old, new))

edit('scripts/lfm_quant.py', 'from runtime.cdrs_data import GenerateCDRSData\n', '')
edit('scripts/lfm_quant.py', '        GenerateCDRSData(config)()', '        from runtime.cdrs_data import GenerateCDRSData\n        GenerateCDRSData(config)()')
edit('scripts/lfm_quant.py', '    config = get_configs()\n', '    config = get_configs()\n    import random\n    import numpy as np\n    import tensorflow as tf\n    random.seed(config.seed)\n    np.random.seed(config.seed)\n    tf.random.set_seed(config.seed)\n')
# Data uses positional monthly offsets. Reject unsorted, duplicated, or gapped
# panels instead of silently interpreting 12 rows as 12 calendar months.
edit('scripts/data_processing.py', '        # split gvkeys into train, validation', '''        if self.data.duplicated(['gvkey', 'date']).any():
            raise ValueError('Duplicate gvkey/month rows are not supported')
        self.data = self.data.sort_values(['gvkey', 'date']).reset_index(drop=True)
        months = self.data['date'].dt.year * 12 + self.data['date'].dt.month
        gaps = months.groupby(self.data['gvkey']).diff().dropna()
        if not gaps.eq(1).all():
            raise ValueError('Dataset requires a contiguous monthly panel for each gvkey')
        if self.config.forecast_steps != 1:
            raise ValueError('Released dataset only implements one forecast step')

        # split gvkeys into train, validation''')
edit('scripts/data_processing.py', 'self._date_offset_from_end = pd.DateOffset(months=self.config.stride)', 'self._date_offset_from_end = pd.DateOffset(months=self.config.forecast_n)')
edit('scripts/data_processing.py', 'self.end_date - pd.DateOffset(months=self.config.stride)', 'self.end_date - pd.DateOffset(months=self.config.forecast_n)')
# Metadata must follow exactly the same entity split as sequence indices.
p = TARGET / 'scripts/data_processing.py'
s = p.read_text()
for keys in ('self._train_gvkeys', 'self._valid_gvkeys', 'self._test_gvkeys'):
    s = s.replace('return tf.data.Dataset.zip((X, Y, self.pre_metadata))',
        f'keys = {keys}\n'
        "        metadata = self._dataset['pre_metadata']\n"
        '        metadata = metadata[np.isin(metadata[:, 1], keys)]\n'
        '        return tf.data.Dataset.zip((X, Y, tf.data.Dataset.from_tensor_slices(metadata)))', 1)
p.write_text(s)
# Public API equivalent, avoiding TensorFlow internal compatibility issues.
edit('scripts/model_utils/losses.py', 'ops.convert_to_tensor(', 'tf.convert_to_tensor(')
edit('scripts/configs.py', 'help=docstring,', "help=docstring.replace('%', '%%'),")
edit('scripts/predict.py', '        self.config = config\n', "        self.config = config\n        if config.UQ and config.log_squasher:\n            raise ValueError('Raw UQ variance under signed-log transformation is not implemented; use log_squasher=False as in the paper preprocessing')\n")
edit('scripts/train.py', 'preds = self.model(inp)', 'preds = self.model(inp, training=True)')
edit('scripts/train.py', 'valid_mse.shape[0] - np.argmin(valid_mse) + 1 >= self.config.early_stop', 'valid_mse.shape[0] - 1 - np.argmin(valid_mse) >= self.config.early_stop')
edit('scripts/predict.py', "        # reverse transform to get raw variance\n        var_keys = [x for x in df.columns if 'norm_variance_' in x]\n        for var_key in var_keys:\n            var_col_name = 'variance_' + var_key.split('_')[-1]\n            df[var_col_name] = df[var_key]\n\n            # reverse center and scale using scaling params\n            df.loc[:, var_col_name] = np.multiply(df.loc[:, var_col_name],\n                                                  self.dataset.scaling_params['scale'][self.target_index]) + \\\n                                      self.dataset.scaling_params['center'][self.target_index]\n\n            # Reverse log squasher\n            if self.config.log_squasher:\n                df.loc[:, var_col_name] = self.dataset.reverse_log_squasher(df.loc[:, var_col_name])\n\n            # un-normalize by seq_norm\n            df[var_col_name] = df[var_col_name].multiply(df['seq_norm'], axis='index')\n\n", "        # Variance is translation invariant and scales by the SQUARE of the\n        # affine factor. Point-estimate variance remains identically zero.\n        var_keys = [x for x in df.columns if 'norm_variance_' in x]\n        for var_key in var_keys:\n            var_col_name = 'variance_' + var_key.split('_')[-1]\n            factor = self.dataset.scaling_params['scale'][self.target_index] * df['seq_norm']\n            df[var_col_name] = df[var_key] * np.square(factor)\n\n")
edit('scripts/utils/post_processing.py', '        self.df = preds[0]', "        for frame in preds[1:]:\n            if not frame[['date', 'gvkey']].equals(preds[0][['date', 'gvkey']]):\n                raise ValueError('Ensemble prediction rows must have identical date/gvkey ordering')\n        self.df = preds[0].copy()")
edit('scripts/utils/post_processing.py', 'model_var = np.var(var_values, axis=0)', 'model_var = np.var([x[pred_field].values for x in self.preds], axis=0)')
patch = []
for p in sorted(TARGET.rglob('*.py')):
    src = SOURCE / p.relative_to(TARGET)
    patch.extend(difflib.unified_diff(src.read_text().splitlines(True), p.read_text().splitlines(True), fromfile='a/' + str(p.relative_to(TARGET)), tofile='b/' + str(p.relative_to(TARGET))))
(ROOT / 'evidence').mkdir(exist_ok=True)
(ROOT / 'evidence/runtime.patch').write_text(''.join(patch))
print('Prepared runtime and evidence/runtime.patch')
