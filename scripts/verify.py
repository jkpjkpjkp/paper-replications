#!/usr/bin/env python3
"""Fail-closed audit of run counts, configurations, references and error bounds."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
from threadpoolctl import threadpool_limits
from krrdp.market import Market
from krrdp.pricing import Training, train

ROOT=Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(test, message):
    if not test:
        raise AssertionError(message)


def main():
    checks={}
    expected={'main':100,'ten_steps':40,'independent_mc':40}
    for directory,count in expected.items():
        target=ROOT/'results'/directory
        complete=json.loads((target/'COMPLETE.json').read_text())
        manifest=json.loads((target/'manifest.json').read_text())
        fingerprint=hashlib.sha256(json.dumps(manifest['config'],sort_keys=True).encode()).hexdigest()
        require(fingerprint==manifest['config_sha256']==complete['config_sha256'],'Config hash mismatch')
        paths=sorted((target/'runs').glob('*.json'))
        require(len(paths)==count==complete['runs'],f'Wrong run count in {directory}')
        runs=[json.loads(p.read_text()) for p in paths]
        require(len(set(r['seed'] for r in runs))==count,'Repeated seeds within scenario')
        for row in runs:
            require(row['config_sha256']==fingerprint,'Mixed configurations')
            require(len(row['stages'])==row['market']['steps'],'Wrong stage count')
            require(np.isfinite(row['price']) and row['seconds']>0,'Invalid result')
            require(row['training']['regularization']==1e-6,'Wrong regularization')
            require(row['price']>=0,'Negative initial price')
            for stage in row['stages']:
                require(stage['lengthscale'] in [40,80],'Wrong lengthscale')
                clip=100 if row['kind']=='geometric_put' else 10000
                require(stage['target_max']<=clip,'Observed target exceeds chosen clipping level')
        references=json.loads((target/'references.json').read_text())
        for dim,ref in references.items():
            require(ref['geometric_tree_difference']<.001,'Reference tree not converged enough')
            require(ref['geometric_european']<ref['geometric_put'],'Put early-exercise reference error')
            require(ref['max_call_quadrature_error']<1e-7,'Quadrature uncertainty too high')
        checks[directory]={'runs':count,'config_sha256':fingerprint,'valid':True}
    convergence=[json.loads(p.read_text()) for p in (ROOT/'results/convergence').glob('d*.json')]
    require(len(convergence)==50,'Convergence sweep incomplete')
    for r in convergence:
        require(abs(r['squared_error']-(r['price']-r['reference'])**2)<1e-12,'Wrong squared error')
        require(r['training']['max_centers']==256,'Wrong convergence center budget')
    checks['convergence']={'runs':50,'valid':True}
    diagnostic=json.loads((ROOT/'results/diagnostics.json').read_text())
    require(len(diagnostic['decomposition'])==240,'Wrong diagnostic replication count')
    stages=[s for r in diagnostic['decomposition'] for s in r['stages']]
    require(len(stages)==2160,'Wrong diagnostic stage count')
    for s in stages:
        require(s['identity_residual']<1e-12,'Error residual identity failed')
        require(s['total_mse']<=s['triangle_bound']+1e-13,'Squared triangle bound failed')
        require(s['propagation_mse']<=s['contraction_bound']+1e-13,'Bellman contraction failed')
    checks['diagnostics']={'runs':240,'stage_checks':2160,'valid':True}
    for kind in ['geometric_put','max_call']:
        a=np.load(ROOT/'results/surfaces'/f'{kind}.npz')
        for stage in [0,2,4,6,8,9]:
            require(a[f'value_{stage}'].shape==(51,51),'Wrong surface shape')
            require(np.all(np.isfinite(a[f'value_{stage}'])),'Invalid surface')
        require((ROOT/'results/surfaces'/f'{kind}_weights.npz').exists(),'Missing surface weights')
    checks['surfaces']={'runs':2,'surfaces':12,'valid':True}
    # Replay two saved primary runs using the final source code. Runtime is not compared.
    with threadpool_limits(2):
        replays=[]
        for kind in ['geometric_put','max_call']:
            saved=json.loads((ROOT/'results/main/runs'/f'{kind}_d2_r00.json').read_text())
            actual=train(Market(**saved['market']),Training(**saved['training']),kind,saved['seed'])
            difference=abs(actual.price-saved['price'])
            require(difference<1e-10,'Final code differs from saved main result')
            replays.append({'kind':kind,'absolute_price_difference':difference})
        checks['deterministic_replay']=replays
    tests=subprocess.run([sys.executable,str(ROOT/'scripts/check_numerics.py')],cwd=ROOT,
                          check=True,capture_output=True,text=True)
    checks['tests']={'returncode':tests.returncode,'output':tests.stdout.strip()}
    require((ROOT/'report.html').exists(),'Missing report')
    class LocalLinks(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key,value in attrs:
                if key in ('src','href') and value and not value.startswith(('https:','http:','#','mailto:')):
                    target=ROOT/value.split('#')[0]
                    # The manifest is emitted by this audit after the report check.
                    require(target.exists() or target==ROOT/'results/artifact_manifest.json',
                            f'Broken local report link: {value}')
            if tag=='img':
                require(bool(dict(attrs).get('alt')),'Report figure has no alt text')
    LocalLinks().feed((ROOT/'report.html').read_text())
    checks['report_local_links_and_alt_text']=True
    for name in ['table_comparison','convergence','error_diagnostics','geometric_put_surfaces','max_call_surfaces']:
        for extension in ['png','svg']:
            require((ROOT/'figures'/f'{name}.{extension}').stat().st_size>1000,'Missing figure')
    source_manifest=json.loads((ROOT/'sources/source_manifest.json').read_text())
    for source in source_manifest['sources']:
        require(sha256(ROOT/source['local_path'])==source['sha256'],'Source hash mismatch')
    checks['source_integrity']=True
    output=dict(status='passed',checks=checks,python=platform.python_version(),
                note='Numerical and artifact verification does not certify paper result replication')
    (ROOT/'results/verification.json').write_text(json.dumps(output,indent=2)+'\n')
    files=[]
    allowed={'.py','.txt','.json','.csv','.html','.toml','.png','.svg','.npz','.pdf'}
    for folder in ['krrdp','scripts','tests','configs','sources','results','figures']:
        for path in sorted((ROOT/folder).rglob('*')):
            if (not path.is_file() or '__pycache__' in path.parts
                    or path.suffix not in allowed or path.name=='artifact_manifest.json'):
                continue
            files.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=sha256(path)))
    for name in ['README.md','report.html','requirements.lock.txt','pyproject.toml','Makefile']:
        p=ROOT/name
        files.append(dict(path=name,bytes=p.stat().st_size,sha256=sha256(p)))
    (ROOT/'results/artifact_manifest.json').write_text(json.dumps(dict(algorithm='sha256',files=files),indent=2)+'\n')
    require((ROOT/'results/artifact_manifest.json').exists(),'Manifest generation failed')
    print(json.dumps(dict(status='passed',main_runs=100,sensitivity_runs=80,convergence_runs=50,
                         finite_state_runs=240,stage_checks=2160,surface_runs=2,
                         artifact_files=len(files),tests=tests.stdout.strip()),indent=2))


if __name__=='__main__':
    main()
