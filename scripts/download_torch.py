"""Fetch the official historical wheel in restartable, checked byte ranges."""
import concurrent.futures
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://download.pytorch.org/whl/cu110/torch-1.7.0%2Bcu110-cp38-cp38-linux_x86_64.whl'
SIZE = 1137106828
MD5 = '9e756d014c8d256a1bb2bccc02b6a3c7'  # Official S3 single-part ETag.
DEST = ROOT/'downloads'
DEST.mkdir(exist_ok=True)
WHEEL = DEST/'torch-1.7.0+cu110-cp38-cp38-linux_x86_64.whl'


def fetch(index):
    start = SIZE*index//16
    end = SIZE*(index+1)//16-1
    path = DEST/('torch.part{:02d}'.format(index))
    if path.exists() and path.stat().st_size == end-start+1:
        return path
    subprocess.run(['curl', '-fsSL', '--noproxy', '*', '--retry', '8', '--retry-all-errors',
                    '--connect-timeout', '30', '--max-time', '1800',
                    '--range', '{}-{}'.format(start,end), URL,
                    '-o', str(path)], check=True)
    assert path.stat().st_size == end-start+1, path
    print('Downloaded part {}/16'.format(index+1), flush=True)
    return path


if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        parts = list(pool.map(fetch, range(16)))
    digest = hashlib.md5()
    with WHEEL.open('wb') as target:
        for path in parts:
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1<<20), b''):
                    digest.update(chunk)
                    target.write(chunk)
    assert WHEEL.stat().st_size == SIZE
    assert digest.hexdigest() == MD5, 'Wheel does not match official ETag'
    (DEST/'torch.sha256').write_text(hashlib.sha256(WHEEL.read_bytes()).hexdigest()+'  '+WHEEL.name+'\n')
    for path in parts:
        path.unlink()
    print('Verified official wheel:', WHEEL, flush=True)
