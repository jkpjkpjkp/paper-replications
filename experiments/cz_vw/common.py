"""Paths and frozen settings for the requested CZ/value-weighted adaptation."""
from pathlib import Path
import json
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_data import sha, save_json

DATA = HERE / 'data'
BASE = ROOT / 'data/sp500'
CZ = ROOT.parent / 'Chen_Zimmermann_Factor_Zoo_2026_10'
SIGNALS = CZ / 'Firm Level Characteristics/Predictors.parquet'
FACTORS = ROOT.parent / 'factors-daily/ff5_momentum_daily.parquet'
PROTOCOL = ROOT.parent / 'protocol.txt'


def settings():
    return json.loads((HERE / 'protocol.json').read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def model_config(universe='sp500', seed=0, epochs=50, stage1=None):
    from runtime import config
    cfg = config('sp500', seed, epochs, stage1)
    cfg['vqvae']['num_features'] = cfg['predictor']['num_features'] = 202
    return cfg


def training_sources():
    files = [HERE / n for n in ['common.py', 'build.py', 'panel.py', 'training.py', 'protocol.json',
        'portfolio.py', 'analyze.py', 'verify.py', 'checks.py', 'run.py']]
    files += [ROOT / 'scripts' / n for n in ['runtime.py', 'train.py', 'build_data.py']]
    files += sorted((ROOT / 'upstream/module').rglob('*.py'))
    files += sorted((ROOT / 'upstream/trainer').rglob('*.py'))
    files += sorted((ROOT / 'upstream/utils').rglob('*.py'))
    files += [ROOT / 'upstream/configs/config.yaml', PROTOCOL]
    return {str(p): sha(p) for p in files}


def check_lock():
    lock = json.loads((HERE / 'locked_before_training.json').read_text())
    assert lock['training_sources_sha256'] == training_sources(), 'Training source/config changed'
    assert lock['data_manifest_sha256'] == sha(DATA / 'manifest.json')
    assert lock['factor_sha256'] == sha(FACTORS)
    return lock
