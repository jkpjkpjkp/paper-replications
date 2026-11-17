"""Read historical S&P 500 intervals through the existing WRDS connection."""
import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, '/home/jkp/data/ibes')
from bridge import Bridge
from download_ibes import remote, download, save_json, verified_file


def main():
    target = ROOT / 'sources/sp500_membership.parquet'
    receipt = target.with_suffix('.receipt.json')
    if receipt.exists() and verified_file(target, json.loads(receipt.read_text())):
        print('Historical membership checksum verified', flush=True)
        return
    # Reuse only the data-export implementation; do not execute its old queries.
    tree = ast.parse(Path('/home/jkp/data/song26l-us/download_metadata.py').read_text())
    worker = next(ast.literal_eval(n.value) for n in tree.body
                  if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'WORKER' for t in n.targets))
    statement = "SELECT * FROM crsp_a_indexes.dsp500list_v2 WHERE mbrstartdt<=DATE '2004-01-02' AND mbrenddt>=DATE '2004-01-02' ORDER BY permno,mbrstartdt"
    with Bridge(Path('/home/jkp/data/wrds-crsp-download/wrds-jupyter.token')) as b:
        b.execute('WRDS_USERNAME=' + repr(b.username) + '\n' + worker, timeout=180)
        try:
            info = remote(b, 'export_query', statement, 'sp500_membership')
            download(b, info, target)
            save_json(receipt, info)
            print(f"Downloaded {info['rows']} historical intervals", flush=True)
        finally:
            remote(b, 'cleanup')


if __name__ == '__main__':
    main()
