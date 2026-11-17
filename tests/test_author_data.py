"""Software-only checks: original-loop equivalence and interruption recovery."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import train_author_data as worker
import run_author_suite as suite


def synthetic_sampler():
    _, utils, _ = worker.upstream_modules()
    dates = pd.date_range('2008-01-02', periods=3, freq='B')
    index = pd.MultiIndex.from_product([dates, [f's{i}' for i in range(8)]],
                                      names=['datetime', 'instrument'])
    values = np.random.RandomState(10).normal(size=(len(index), 161)).astype('float32')
    frame = pd.DataFrame(values, index=index)
    return utils.TSDataSampler(frame, str(dates[0].date()), str(dates[-1].date()),
                               step_len=21, fillna_type='ffill+zero')


def compare_states(a, b):
    assert a.keys() == b.keys()
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_released_fit_wrapper_and_resume_are_equal(tmp_path, monkeypatch):
    config = suite.config_for('csi300', 0, [.65, .3, .05], tmp_path/'unused.json', 1)
    config['epochs'] = 2
    data = synthetic_sampler()
    def load_split(config, split):
        assert split in ('train', 'valid'), 'Training must never load test data'
        return data, {'software_test': 'synthetic; not empirical evidence'}
    monkeypatch.setattr(worker, 'load_split', load_split)

    original_dir = tmp_path/'original'
    original_dir.mkdir()
    original = worker.make_runner(config)
    original.save_path = str(original_dir)
    original.save_prefix = 'model'
    original.fit(data, data, None)

    full = tmp_path/'full'
    full.mkdir()
    worker.train(config, full)
    full_best = torch.load(full/'best.pt', map_location='cpu')
    compare_states(original.model.state_dict(), full_best)

    interrupted = tmp_path/'interrupted'
    interrupted.mkdir()
    base, _, _ = worker.upstream_modules()
    predict = base.SequenceModel.predict
    calls = 0
    def fail_second_validation(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('simulated interruption')
        return predict(self, *args, **kwargs)
    monkeypatch.setattr(base.SequenceModel, 'predict', fail_second_validation)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        worker.train(config, interrupted)
    monkeypatch.setattr(base.SequenceModel, 'predict', predict)
    worker.train(config, interrupted)
    compare_states(torch.load(full/'latest.pt')['model'],
                   torch.load(interrupted/'latest.pt')['model'])
    compare_states(full_best, torch.load(interrupted/'best.pt'))
    a = json.loads((full/'status.json').read_text())
    b = json.loads((interrupted/'status.json').read_text())
    assert a['best_validation_ic'] == b['best_validation_ic']
    assert a['best_epoch'] == b['best_epoch']


def test_validation_selection_rejects_incomplete_trials(tmp_path):
    plan = suite.prepare(tmp_path/'suite', Path(__file__), 1)
    assert plan['trials'] == 216
    assert len(set(suite.weight_grid())) == 36
    with pytest.raises(FileNotFoundError):
        suite.selected_jobs(plan['jobs'])
    with pytest.raises(FloatingPointError):
        worker.choose_checkpoint(.1, float('nan'))
    assert not worker.choose_checkpoint(.1, .1)


def test_validation_selection_ignores_test_metrics(tmp_path):
    plan = suite.prepare(tmp_path/'suite', Path(__file__), 1)
    winner = [.68, .33, .08]
    for job in plan['jobs']:
        output = Path(job['output'])
        worker.write_json(output/'status.json', dict(
            phase='trained', best_validation_ic=.1 if job['weights'] == winner else .01,
            test_IC=-100 if job['weights'] == winner else 100))
    selected = suite.selected_jobs(plan['jobs'])
    assert all(r['winner']['weights'] == winner for r in selected.values())


def test_selected_prediction_export_matches_released_metrics(tmp_path, monkeypatch):
    config = suite.config_for('csi300', 0, [.65, .3, .05], tmp_path/'unused.json', 1)
    config['epochs'] = 1
    data = synthetic_sampler()
    splits = []
    def load_split(config, split):
        splits.append(split)
        return data, {'software_test': True}
    monkeypatch.setattr(worker, 'load_split', load_split)
    worker.train(config, tmp_path)
    assert splits == ['train', 'valid']
    worker.evaluate_selected(config, tmp_path)
    assert splits == ['train', 'valid', 'valid', 'test']
    result = json.loads((tmp_path/'metrics.json').read_text())
    assert result['test']['metrics']['IC'] == pytest.approx(result['test']['independent']['IC'])
    frame = pd.read_parquet(tmp_path/'test_predictions.parquet')
    assert frame.shape == (24, 6)


def test_auto_device_requires_working_cuda_runtime(tmp_path, monkeypatch):
    import subprocess
    def failed_probe(command, **kwargs):
        if command[0] == 'nvidia-smi':
            return subprocess.CompletedProcess(command, 0, stdout='3000\n')
        return subprocess.CompletedProcess(command, 1)
    monkeypatch.setattr(suite.subprocess, 'run', failed_probe)
    job = {'output': str(tmp_path)}
    assert suite.device_for(job, 'auto', tmp_path/'missing-python') == 'cpu'
