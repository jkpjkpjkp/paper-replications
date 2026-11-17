"""Run the unchanged official entrypoint and retain its predictions and failures.

CPU/library differences are recorded; no seed, losses, hyperparameters, training
schedule, graph, masks, or model definition are changed. Metrics are intercepted
only to save the exact scored labels, probabilities and predictions.
"""
import argparse
import contextlib
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent


class Tee:
    def __init__(self, output, file):
        self.output, self.file = output, file

    def write(self, value):
        self.output.write(value)
        self.file.write(value)
        self.file.flush()

    def flush(self):
        self.output.flush()
        self.file.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='released-seed10-cpu')
    args = parser.parse_args()
    out = ROOT / 'results' / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    import torch
    import numpy as np
    import sklearn.metrics as metrics

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    records = []
    auc_original, f1_original = metrics.roc_auc_score, metrics.f1_score

    def auc(y, probs, *a, **kw):
        labels = np.asarray(y).copy()
        probabilities = np.asarray(probs).copy()
        value = auc_original(y, probs, *a, **kw)
        records.append(dict(y=labels, probabilities=probabilities, auc=value))
        return value

    def f1(y, preds, *a, **kw):
        value = f1_original(y, preds, *a, **kw)
        records[-1].update(predictions=np.asarray(preds).copy(), f1=value)
        return value

    metrics.roc_auc_score, metrics.f1_score = auc, f1
    start = time.time()
    namespace = {'__name__': '__main__', '__file__': str(ROOT / 'official/main.py')}
    os.chdir(ROOT / 'official')
    sys.path.insert(0, str(ROOT / 'official'))
    source = (ROOT / 'official/main.py').read_bytes()
    status, error = 'complete_released_entrypoint', None
    with (out / 'stdout.log').open('w') as log, (out / 'stderr.log').open('w') as errors:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, errors)):
            try:
                exec(compile(source, str(ROOT / 'official/main.py'), 'exec'), namespace)
            except Exception:
                status = 'failed_released_entrypoint'
                error = traceback.format_exc()
                print(error, file=sys.stderr, flush=True)
    metrics.roc_auc_score, metrics.f1_score = auc_original, f1_original
    summary = dict(status=status, error=error, seed=10, elapsed_seconds=time.time() - start,
                   cpu_threads=torch.get_num_threads(), gpu=False,
                   source_sha256=hashlib.sha256(source).hexdigest(),
                   environment={k:importlib.metadata.version(k) for k in ['torch','torch-geometric','numpy','scipy','scikit-learn']},
                   adaptations=['CPU instead of paper NVIDIA 4090; reused read-only environment with recorded library versions',
                                'Metric wrappers save exactly the arrays passed by unchanged official source'],
                   reproduction_level='released_code_execution; partial relative to full paper',
                   max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   evaluated_online_tasks=len(records),
                   updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if records:
        arrays = {}
        for j, row in enumerate(records):
            for key in ['y', 'probabilities', 'predictions']:
                if key in row:
                    arrays[f'{j:03d}_{key}'] = row[key]
        np.savez_compressed(out / 'predictions.npz', **arrays)
        summary['daily_metrics'] = [dict(task_index=j+14, auc=float(r['auc']), f1=float(r['f1']),
                                              accuracy=float(np.mean(r['y']==r['predictions'])), nodes=len(r['y']))
                                    for j,r in enumerate(records) if 'predictions' in r]
        summary['mean_metrics'] = {k:float(np.mean([r[k] for r in summary['daily_metrics']])) for k in ['auc','f1','accuracy']}
    model_info = {}
    for name in ['student_pretrain_model','student_classifier','teacher_pretrain_model','teacher_classifier']:
        model = namespace.get(name)
        if model is not None:
            model_info[name] = dict(parameters=sum(p.numel() for p in model.parameters()),
                                    all_parameters_finite=all(bool(torch.isfinite(p).all()) for p in model.parameters()))
            torch.save(model.state_dict(), out / f'{name}.pt')
    summary['models'] = model_info
    if all(name in namespace for name in ['student_pretrain_model','student_classifier','online_dataset']):
        model, classifier = namespace['student_pretrain_model'], namespace['student_classifier']
        model.eval(); classifier.eval()
        graph = namespace['online_dataset'][14]
        with torch.no_grad():
            x, edge = torch.tensor(graph[0]), torch.tensor(graph[1])
            a = classifier(model(x,edge,use_fusion=True)[0],edge)
            b = classifier(model(x,edge,use_fusion=True)[0],edge)
        difference = float((a-b).abs().max())
        summary['eval_repeat_max_abs_logit_difference'] = difference if np.isfinite(difference) else None
        summary['eval_is_deterministic_without_reseeding'] = bool(torch.equal(a,b))
    temp = out / 'result.json.tmp'
    temp.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    temp.replace(out / 'result.json')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['daily_metrics','error']}, indent=2), flush=True)
    return 0 if status == 'complete_released_entrypoint' else 1


if __name__ == '__main__':
    raise SystemExit(main())
