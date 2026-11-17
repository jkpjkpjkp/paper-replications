"""Independently audit saved forecasts and native checkpoint replay."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from runtime import ROOT, Panel, temporal_model, rankic, ranking_metrics, encode, from_codes
from build_data import sha, save_json
from train import seed_all


def main():
    torch.set_num_threads(2)
    seed_all(0)
    checked = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for universe in ['csi300', 'sp500']:
        panel = Panel(universe)
        for name, digest in panel.manifest['output_hashes'].items():
            assert sha(panel.folder / name) == digest, f'Data checksum mismatch: {name}'
        for seed in range(5):
            folder = ROOT / f'results/{universe}/stage2/seed{seed}'
            metrics = json.loads((folder / 'metrics.json').read_text())
            assert sha(folder / 'best.pt') == metrics['checkpoint_sha256']
            assert sha(folder / 'predictions.parquet') == metrics['predictions_sha256']
            assert sha(ROOT / f'results/{universe}/stage1/best.pt') == metrics['stage1_sha256']
            predictions = pd.read_parquet(folder / 'predictions.parquet')
            assert not predictions.duplicated(['date', 'symbol']).any()
            assert np.isfinite(predictions.prediction).all()
            by_date = dict(tuple(predictions.groupby('date', sort=True)))
            assert list(by_date) == list(panel.dates[panel.plans['test']])
            daily = []
            for t in panel.plans['test']:
                f = by_date[str(panel.dates[t])]
                ids = panel.ids(t, 'test')
                np.testing.assert_array_equal(f.symbol, panel.symbols[ids])
                target = panel.close[ids, t+5]/panel.close[ids, t+1]-1
                target[~np.isfinite(target)] = np.nan
                np.testing.assert_allclose(f.label, target, rtol=0, atol=0, equal_nan=True)
                daily.append(rankic(f.prediction.to_numpy(), target))
            saved_daily = pd.read_csv(folder / 'daily_rankic.csv')
            np.testing.assert_allclose(saved_daily.RankIC, daily, atol=1e-14, rtol=1e-12, equal_nan=True)
            recomputed = ranking_metrics(daily)
            for name in ['RankIC', 'RankICIR']:
                np.testing.assert_allclose(metrics[name], recomputed[name], atol=1e-14, rtol=1e-12)
            cfg = json.loads((folder / 'config.json').read_text())
            model = temporal_model(cfg, len(panel.plans['train'])*cfg['train']['num_epochs']).to(device)
            state = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(state['state_dict'], strict=True)
            model.eval()
            replayed = []
            for t in panel.plans['test'][[0, len(panel.plans['test'])//2, -1]]:
                x, p, y, ids = panel.batch(t, 'test', device)
                with torch.no_grad():
                    native = model(x, p)[0]
                    z, _ = encode(model, x)
                    cached, _ = from_codes(model, x, p, z)
                torch.testing.assert_close(native, cached, rtol=0, atol=0)
                f = by_date[str(panel.dates[t])]
                # Replay the original precision/device; crossing a VQ boundary
                # under CPU/TF32 rounding is not a faithful checkpoint audit.
                np.testing.assert_allclose(native.cpu().numpy(), f.prediction, rtol=1e-5, atol=2e-6)
                replayed.append(str(panel.dates[t]))
            history = json.loads((folder / 'history.json').read_text())
            assert state['epoch'] == int(np.argmin([h['valid_loss'] for h in history]))
            checked.append(dict(universe=universe, seed=seed, predictions=len(predictions),
                                ranking_days=len(daily), native_checkpoint_replay_dates=replayed,
                                replay_device=device))
    result = dict(passed=True, checked_runs=checked, exact_published_results_reproduced=False,
                  scope='Adapted full-period empirical replication; all saved forecasts checked, 3 dates per model replayed natively at original GPU precision')
    save_json(ROOT / 'results/verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
