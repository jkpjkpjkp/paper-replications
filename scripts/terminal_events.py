"""Attach dated CRSP delisting events for portfolio settlement only."""
import numpy as np
import pandas as pd
from build_data import ROOT, CRSP, sha, save_json
import json


def main():
    folder = ROOT / 'data/sp500'
    dates = np.load(folder / 'dates.npy')
    symbols = np.load(folder / 'symbols.npy')
    positions = {int(s[1:]): i for i, s in enumerate(symbols)}
    offsets = {str(d): i for i, d in enumerate(dates)}
    events = np.zeros((len(symbols), len(dates)), dtype=bool)
    for year in range(2008, 2026):
        f = pd.read_parquet(CRSP / f'{year}.parquet', columns=['permno', 'dlycaldt', 'dlydelflg'],
                            filters=[('permno', 'in', list(positions)), ('dlydelflg', '==', 'Y')])
        for row in f.itertuples():
            if str(row.dlycaldt) in offsets:
                events[positions[row.permno], offsets[str(row.dlycaldt)]] = True
    np.save(folder / 'terminal_events.npy', events)
    manifest = json.loads((folder / 'manifest.json').read_text())
    manifest['output_hashes']['terminal_events.npy'] = sha(folder / 'terminal_events.npy')
    manifest['portfolio_events'] = 'CRSP dlydelflg=Y; dated terminal-return positions are settled to cash; known terminal names cannot be newly purchased'
    save_json(folder / 'manifest.json', manifest)
    print('Dated terminal events:', int(events.sum()))


if __name__ == '__main__':
    main()
