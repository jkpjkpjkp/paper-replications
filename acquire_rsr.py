"""Recreate the sparse checkout used by prepare_reference.py; pins source commit."""
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'reference-rsr'
COMMIT='cfbb01bdf194b81bc5893a1b37aff1c0d0d2a82d'


def git(*args):
    subprocess.run(['git','-C',str(DEST),*args],check=True)


if __name__=='__main__':
    if not DEST.exists():
        subprocess.run(['git','clone','--filter=blob:none','--no-checkout',
            'https://github.com/fulifeng/Temporal_Relational_Stock_Ranking.git',str(DEST)],check=True)
    git('sparse-checkout','set','--no-cone','/README.md','/LICENSE','/preprocess/',
        '/training/load_data.py','/data/*tickers*','/data/*aver_line_dates.csv','/data/relation.tar.gz')
    git('checkout',COMMIT)
    path=DEST/'.git/info/sparse-checkout'
    patterns=path.read_text()
    for market in ['NASDAQ','NYSE']:
        for ticker in (DEST/f'data/{market}_tickers_qualify_dr-0.98_min-5_smooth.csv').read_text().split():
            patterns+=f'/data/google_finance/{market}_{ticker}_30Y.csv\n'
    path.write_text(patterns)
    git('read-tree','-mu','HEAD')
