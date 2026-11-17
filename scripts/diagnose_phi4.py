#!/usr/bin/env python3
"""Independent first-chaos capacity diagnostic, NOT the paper's F-SPDENO."""
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from chaos.basis import discrete_haar, haar_values
from chaos.data import phi4_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '.agent/diagnostics'

def relative(y, p):
    e = np.linalg.norm((p-y).reshape(len(y), -1), axis=1) / np.linalg.norm(y.reshape(len(y), -1), axis=1)
    return {'mean': float(e.mean()), 'path_sd': float(e.std(ddof=1)), 'path_se': float(e.std(ddof=1)/np.sqrt(len(e)))}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    b = discrete_haar(50)
    h = haar_values(50, 64)
    projection = np.linalg.pinv(h) @ h
    rows = []
    for seed in (0, 1, 2):
        data = phi4_dataset(1200, seed + 2000)
        noise, truth = data['dw'].astype(np.float64), data['u'][:, 1:].astype(np.float64)
        # First-order Gaussian coordinates are a lossless orthogonal transform
        # of all 50 noise increments. Spatial FFT diagonalizes linear diffusion.
        z = np.einsum('kt,btx->bkx', b, noise)
        xf = np.fft.rfft(z, axis=-1)
        yf = np.fft.rfft(truth[:1000], axis=-1)
        pf = np.empty((200, 50, 65), dtype=np.complex128)
        weights = []
        for k in range(65):
            a = xf[:1000, :, k]
            gram = a.conj().T @ a
            # Fixed numerical regularizer, set before evaluating test results.
            ridge = 1e-8 * np.trace(gram).real / 50
            w = np.linalg.solve(gram + ridge*np.eye(50), a.conj().T @ yf[:, :, k])
            weights.append(w)
            pf[:, :, k] = xf[1000:, :, k] @ w
        pred = np.fft.irfft(pf, n=128, axis=-1)
        pred = np.einsum('st,btx->bsx', projection, pred)
        target = truth[1000:]
        row = {'seed': seed, 'test_relative_l2': relative(target, pred),
               'temporal_projection_relative_l2': relative(target, np.einsum('st,btx->bsx', projection, target)),
               'noise_roundtrip_max_abs': float(np.max(np.abs(noise-np.einsum('kt,bkx->btx', b,z)))),
               'train_noise_sha256': hashlib.sha256(data['dw'][:1000].tobytes()).hexdigest(),
               'test_noise_sha256': hashlib.sha256(data['dw'][1000:].tobytes()).hexdigest()}
        np.savez_compressed(OUT/f'first_chaos_model_seed{seed}.npz', weights=np.stack(weights), noise_basis=b, temporal_projection=projection)
        np.savez_compressed(OUT/f'first_chaos_seed{seed}.npz', truth=target.astype(np.float32), prediction=pred.astype(np.float32))
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = {'scope': 'Fresh supervised linear first-chaos capacity diagnostic; not F-SPDENO or a published baseline',
              'train_paths': 1000, 'test_paths': 200, 'selection': 'No validation or test tuning; fixed ridge 1e-8 times mean Gram diagonal',
              'temporal_basis_rank': int(np.linalg.matrix_rank(h)), 'seeds': rows,
              'mean_relative_l2': float(np.mean([r['test_relative_l2']['mean'] for r in rows])),
              'seed_sd': float(np.std([r['test_relative_l2']['mean'] for r in rows], ddof=1)),
              'limitations': ['Independent local simulator, not author data', 'Fixed zero initial condition only',
                              'Linear Fourier regression replaces FNO; cannot establish paper replication',
                              'Overparameterized frequency-wise linear map; not parameter-matched to paper'],
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'first_chaos.json').write_text(json.dumps(result, indent=2)+'\n')
if __name__ == '__main__':
    main()
