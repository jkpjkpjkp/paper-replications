"""Fetch commit-pinned author artifacts and audit them in the staged workspace.

Pickles are restricted to numpy arrays and CPU torch tensors. Tensor storage
deserialization uses torch's weights_only loader rather than arbitrary pickle.
"""
import collections
import datetime
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import pickle
import pickletools
import resource
import urllib.request

ROOT = Path(__file__).resolve().parent
COMMIT = '3c620a2e6a41757ddc768e3db1a61ca3afacb25c'


def write_json(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n')
    os.replace(temp, path)


def fetch():
    tree = json.loads((ROOT / 'sources/repository-tree.json').read_text())
    files = []
    for info in tree['tree']:
        if info['type'] != 'blob' or not info['path'].endswith('.dat'):
            continue
        target = ROOT / 'official' / info['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://raw.githubusercontent.com/yhzhang1309/ConFVG/{COMMIT}/{info["path"]}'
        if not target.exists():
            temp = target.with_suffix('.part')
            with urllib.request.urlopen(url, timeout=120) as src, temp.open('wb') as dst:
                while chunk := src.read(1024 * 1024):
                    dst.write(chunk)
            temp.replace(target)
        raw = target.read_bytes()
        git_sha = hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()
        if len(raw) != info['size'] or git_sha != info['sha']:
            raise ValueError(f'Git blob mismatch: {target}')
        globals_ = set()
        with target.open('rb') as stream:
            for op, arg, pos in pickletools.genops(stream):
                if op.name == 'GLOBAL':
                    globals_.add(arg)
        files.append(dict(path=str(target), url=url, bytes=len(raw),
                          sha256=hashlib.sha256(raw).hexdigest(), git_blob_sha1=git_sha,
                          pickle_globals=sorted(globals_)))
        del raw
        print('Verified artifact', target.name, info['size'], flush=True)
    write_json(ROOT / 'sources/data-manifest.json', dict(
        commit=COMMIT, retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        files=files, license='ConFVG repository has no LICENSE; upstream Kaggle data card states CC0'))
    return files


def main():
    (ROOT / 'results').mkdir(exist_ok=True)
    files = fetch()
    import numpy as np
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)

    def cpu_storage(data):
        return torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)

    allowed = {
        ('torch._utils', '_rebuild_tensor_v2'): torch._utils._rebuild_tensor_v2,
        ('torch.storage', '_load_from_bytes'): cpu_storage,
        ('collections', 'OrderedDict'): collections.OrderedDict,
        ('numpy.core.multiarray', '_reconstruct'): np.core.multiarray._reconstruct,
        ('numpy', 'ndarray'): np.ndarray,
        ('numpy', 'dtype'): np.dtype,
        ('numpy.core.multiarray', 'scalar'): np.core.multiarray.scalar,
    }

    class RestrictedUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) not in allowed:
                raise pickle.UnpicklingError(f'Unapproved global {module}.{name}')
            return allowed[(module, name)]

    reports = []
    for artifact in files:
        with Path(artifact['path']).open('rb') as stream:
            data = RestrictedUnpickler(stream).load()
        print('Loaded', artifact['path'], type(data).__name__, len(data), flush=True)
        graphs = []
        for i, graph in enumerate(data):
            row = {'index': i, 'parts': len(graph)}
            arrays = [np.asarray(a) for a in graph]
            row['shapes'] = [list(a.shape) for a in arrays]
            row['dtypes'] = [str(a.dtype) for a in arrays]
            x, edge = arrays[:2]
            row.update(nodes=len(x), edges=edge.shape[1], finite_features=bool(np.isfinite(x).all()),
                       feature_min=float(np.nanmin(x)), feature_max=float(np.nanmax(x)),
                       valid_edges=bool(edge.size == 0 or (edge.min() >= 0 and edge.max() < len(x))))
            if len(arrays) == 4:
                y, reduced = arrays[2:]
                row['labels'] = {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
                row['reduced_labels'] = {str(k): int(v) for k, v in zip(*np.unique(reduced, return_counts=True))}
                row['label_fraction'] = float(np.mean(reduced >= 0))
                row['label_agreement'] = bool(np.array_equal(y[reduced >= 0], reduced[reduced >= 0]))
                row['feature_exact_label_columns'] = [j for j in range(x.shape[1]) if np.array_equal(x[:, j], y)]
            graphs.append(row)
        reports.append(dict(path=artifact['path'], graphs=len(graphs), total_nodes=sum(g['nodes'] for g in graphs),
                            total_directed_edges=sum(g['edges'] for g in graphs),
                            max_nodes=max(g['nodes'] for g in graphs), graph_details=graphs))
        del data
    env = {k: importlib.metadata.version(k) for k in ['torch', 'torch-geometric', 'numpy', 'scipy', 'scikit-learn', 'matplotlib', 'networkx']}
    write_json(ROOT / 'results/data-audit.json', dict(
        status='completed', artifacts=reports, environment=env,
        cpu_threads=torch.get_num_threads(), max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        caveats=['Graph artifacts have no raw beneficiary/provider IDs or timestamp metadata; chronology depends on preserved list order.',
                 'Medical source authors describe real-world claims. No independent CMS lineage or original labels proven by normalized artifact.',
                 'Exact paper has 517737 claims in Section 4.1 and 554407 nodes in Table 5; compare against release.']))
    print(json.dumps(dict(environment=env, summaries=[{k:v for k,v in r.items() if k!='graph_details'} for r in reports]), indent=2), flush=True)


if __name__ == '__main__':
    main()
