"""Independent score/accounting checks and source-supported model diagnostics.

Run from the staged workspace described in README.md. Does not train or modify author source, artifacts,
predictions or checkpoints. Uses actual released medical graphs.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='released-seed10-cpu')
    args = parser.parse_args()
    out = ROOT / 'results' / args.run_id
    result = json.loads((out / 'result.json').read_text())
    import numpy as np
    import scipy.linalg
    from scipy.stats import rankdata
    from scipy.sparse.csgraph import connected_components
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    artifact = ROOT / 'official/Dataset/0.1/dataset.dat'
    expected = json.loads((ROOT / 'sources/data-manifest.json').read_text())['files'][0]['sha256']
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
        raise ValueError('Data checksum changed')
    # This exact artifact was previously read with the restricted unpickler.
    with artifact.open('rb') as f:
        dataset = pickle.load(f)
    checks = dict(data_checksum=True, released_graphs=len(dataset),
                  expected_online_graphs=len(dataset)-14,
                  evaluated_online_graphs=result['evaluated_online_tasks'])
    independently_computed = []
    if (out / 'predictions.npz').exists():
        predictions = np.load(out / 'predictions.npz', allow_pickle=False)
        for j in range(result['evaluated_online_tasks']):
            y = predictions[f'{j:03d}_y'].astype(bool)
            p = predictions[f'{j:03d}_probabilities']
            pred = predictions[f'{j:03d}_predictions'].astype(bool)
            assert np.array_equal(y, dataset[j+14][2])
            assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
            assert np.array_equal(pred, p > 0.5)
            positive = int(y.sum())
            negative = len(y)-positive
            auc = (rankdata(p)[y].sum()-positive*(positive+1)/2)/(positive*negative)
            tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
            f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
            independently_computed.append(dict(task_index=j+14, nodes=len(y), auc=float(auc),
                f1=float(f1), accuracy=float(np.mean(y==pred))))
        for independent, saved in zip(independently_computed, result['daily_metrics']):
            for metric in ['auc', 'f1', 'accuracy']:
                assert abs(independent[metric]-saved[metric]) < 1e-12
        checks.update(labels_match_author_graphs=True, probabilities_finite_and_bounded=True,
                      predictions_match_binary_probability_rule=True,
                      independent_metrics_match_to_1e12=True,
                      daily_metric_aggregation='equal weight per graph/day; not pooled per node')
    loss_lines = [line for line in (out / 'stdout.log').read_text().splitlines() if 'Loss:' in line]
    losses = [float(line.rsplit('Loss:',1)[1]) for line in loss_lines]
    checks['pretrain_loss_lines'] = len(losses)
    checks['all_pretrain_losses_finite'] = bool(np.isfinite(losses).all())
    checks['all_model_parameters_finite'] = all(m['all_parameters_finite'] for m in result['models'].values())

    # The paper's uniform complete-graph perturbation leaves the zero-eigenspace
    # (except its constant vector) degenerate when the original graph disconnects.
    x, edge = map(np.asarray, dataset[0][:2])
    n = len(x)
    adj = np.zeros((n,n), dtype=np.float64)
    adj[edge[0],edge[1]] = 1
    adj[edge[1],edge[0]] = 1
    components, _ = connected_components(adj, directed=False)
    epsilon = 1e-5
    laplacian = np.diag(adj.sum(1))-adj
    perturbation = epsilon*(n*np.eye(n)-np.ones((n,n)))
    eigenvalues = scipy.linalg.eigh(laplacian+perturbation, subset_by_index=[0,7], eigvals_only=True)
    spectral = dict(graph_index=0, nodes=n, connected_components=int(components), epsilon=epsilon,
                    smallest_eigenvalues=eigenvalues.tolist(),
                    predicted_repeated_eigenvalue=epsilon*n,
                    predicted_multiplicity=int(components-1),
                    repeated_eigenvalue_confirmed=bool(np.allclose(eigenvalues[1:],epsilon*n,rtol=1e-7,atol=1e-10)),
                    implication='Uniform complete-graph perturbation connects the graph but does not identify a unique Fiedler vector when original component count exceeds 2; eigensolver initialization may select different vectors in that subspace.')
    checks['spectral_assumption_checked_on_released_data'] = True
    if 'student_pretrain_model' in result['models']:
        sys.path.insert(0, str(ROOT / 'official'))
        from models import EnhancedADDer, Classifier
        model, classifier = EnhancedADDer(32,32), Classifier(32,2)
        model.load_state_dict(torch.load(out/'student_pretrain_model.pt', map_location='cpu',weights_only=True))
        classifier.load_state_dict(torch.load(out/'student_classifier.pt', map_location='cpu',weights_only=True))
        model.eval(); classifier.eval()
        tx, te = torch.tensor(x), torch.tensor(edge)
        with torch.no_grad():
            torch.manual_seed(8273)
            a = classifier(model(tx,te,use_fusion=True)[0],te)
            torch.manual_seed(8273)
            b = classifier(model(tx,te,use_fusion=True)[0],te)
            c = classifier(model(tx,te,use_fusion=True)[0],te)
            fused = model(tx,te,use_fusion=True)
        dropout = dict(same_seed_logits_identical=bool(torch.equal(a,b)),
                       without_reseeding_logits_identical=bool(torch.equal(b,c)),
                       max_logit_change_without_reseeding=float((b-c).abs().max()) if torch.isfinite(b-c).all() else None,
                       selected_attention_elements=int(fused[3].sum()),
                       unselected_attention_elements=int((~fused[3]).sum()))
    else:
        dropout = None
    final = dict(run_id=args.run_id, status='audited', checks=checks,
                 mean_metrics={k:float(np.mean([r[k] for r in independently_computed])) for k in ['auc','f1','accuracy']} if independently_computed else None,
                 independently_computed_daily_metrics=independently_computed,
                 spectral_assumption=spectral, dropout_and_attention=dropout,
                 no_test_tuning='No hyperparameters, thresholds or checkpoints selected from these scores.',
                 updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    temp = out/'validation.json.tmp'
    temp.write_text(json.dumps(final,indent=2,allow_nan=False)+'\n')
    temp.replace(out/'validation.json')
    print(json.dumps({k:v for k,v in final.items() if k!='independently_computed_daily_metrics'},indent=2),flush=True)


if __name__ == '__main__':
    main()
