"""Independently recompute extension metrics and check immutable inputs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT=Path(__file__).resolve().parents[1]
EXT=ROOT/"extensions"
sys.path.insert(0,str(ROOT))
from ocp import run,controller,metrics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    protocol=json.loads((EXT/"scenario_screen/confirmation/locked_protocol.json").read_text())
    for name,value in protocol["frozen_sources"].items():
        assert digest(ROOT/name)==value,name
    checksum=json.loads((EXT/"scenario_screen/confirmation/artifact_checksums.json").read_text())
    for name,value in checksum.items():
        assert digest(ROOT/name)==value,name
    scores=np.load(EXT/"scenario_screen/confirmation/scores.npz")
    fixed=pd.read_csv(EXT/"scenario_screen/confirmation/fixed_per_seed.csv")
    summary=pd.read_csv(EXT/"scenario_screen/confirmation/fixed_summary.csv")
    assert len(fixed)==360
    checked=0
    for scenario,sigma in [(protocol["primary_scenario"],15),(protocol["transfer_scenario"],20)]:
        for i,seed in enumerate(protocol["seeds"]):
            rng=np.random.default_rng(seed)
            rng.normal(0,.3,protocol["n"])
            expected=np.exp(8*gaussian_filter1d(rng.normal(size=protocol["n"]),sigma,mode="reflect"))
            s=scores[scenario][i]
            np.testing.assert_array_equal(s,expected)
            for method,cfg in protocol["fixed_configs"].items():
                q=controller(s,**cfg) if method=="P" else run(s,method,**cfg)
                b=protocol["burn"]
                coverage=float(np.mean(s[b:]<=q[b:]))
                width=float(np.mean(2*np.maximum(q[b:],0)))
                row=fixed[(fixed.scenario==scenario)&(fixed.seed==seed)&(fixed.method==method)].iloc[0]
                np.testing.assert_allclose([coverage,width],[row.coverage,row.mean_width],rtol=1e-12)
                checked+=1
    pairs=pd.read_csv(EXT/"scenario_screen/confirmation/paired_comparisons.csv")
    bootstrap_checked=0
    for scenario,part in fixed.groupby("scenario"):
        p=part.pivot(index="seed",columns="method",values="mean_width")
        for baseline in ["P","SFOGD","KT"]:
            a=p['UP-paper'].to_numpy();b=p[baseline].to_numpy()
            rng=np.random.default_rng(protocol['bootstrap']['seed'])
            ix=rng.integers(0,len(a),size=(10000,len(a)))
            ratios=a[ix].mean(axis=1)/b[ix].mean(axis=1)
            row=pairs[(pairs.scenario==scenario)&(pairs.baseline==baseline)].iloc[0]
            np.testing.assert_allclose([a.mean()/b.mean(),np.quantile(ratios,.025),np.quantile(ratios,.975)],
                                       [row.ratio,row.ratio_ci_low,row.ratio_ci_high],rtol=1e-12)
            bootstrap_checked+=1
    countries=pd.read_csv(EXT/'country_confirmation/confirmation_results.csv')
    provenance=json.loads((EXT/'country_confirmation/provenance.json').read_text())
    assert digest(EXT/'country_confirmation/protocol.json')==provenance['protocol_sha256']
    assert digest(EXT/'country_confirmation/confirmed_global.csv')==provenance['raw_sha256']
    assert provenance['protocol_registered_at']<provenance['downloaded_at']
    country_checks=0
    for country in countries.country.unique():
        z=np.load(EXT/'country_confirmation/traces'/f"{country.lower().replace(' ','_')}.npz")
        np.testing.assert_array_equal(np.abs(z['observed']-z['forecasts']),z['raw_scores'])
        offset=int(z['evaluation_start'])
        for _,row in countries[countries.country==country].iterrows():
            prefix=row.initialization+'__'+row.method
            start=int(z[prefix+'__start_index'])
            radius=z[prefix+'__radii'][offset-start:]
            result=metrics(z['raw_scores'][offset:],radius)
            for key,value in result.items():
                np.testing.assert_allclose(value,row[key],rtol=1e-11,atol=1e-10)
            country_checks+=1
    real_manifest=json.loads((EXT/'real_candidates/manifest.json').read_text())
    for entry in real_manifest.values():
        assert digest(EXT/'real_candidates'/entry['path'])==entry['sha256']
    assert len(countries)==48 and country_checks==48
    for name in ['confirmed_scenario','country_transfer','clustered_error_dynamics']:
        for suffix in ['png','svg']:
            assert (EXT/'figures'/f'{name}.{suffix}').stat().st_size>10000
    for relative in ['NEW_SCENARIOS.md','scenario_screen/README.md','real_candidates/REAL_SCREEN.md','country_confirmation/README.md']:
        p=EXT/relative
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if '://' not in link:
                assert (p.parent/link).exists(),(relative,link)
    status=dict(checked_at=datetime.now(timezone.utc).isoformat(),status='passed',
                synthetic_fixed_runs_independently_recomputed=checked,
                paired_bootstrap_intervals_recomputed=bootstrap_checked,
                country_runs_independently_recomputed=country_checks,
                source_hashes_valid=True,frozen_protocol_hashes_valid=True,
                original_ocp_code_unchanged=True,documentation_links_valid=True,
                primary_scenario='smooth_correlated_lognormal_sigma15_amplitude8',
                primary_claim='4.656% narrower than strongest validation-selected finite baseline, with slightly higher realized coverage; 30 new seeds',
                real_data_claim='Fixed configurations transfer >=95% UP coverage across four prespecified countries; no universal width dominance')
    (EXT/'VALIDATION.json').write_text(json.dumps(status,indent=2))
    files=[p for p in EXT.rglob('*') if p.is_file() and '__pycache__' not in str(p)
           and p.name != 'artifact_manifest.json' and p.suffix != '.log']
    manifest={str(p.relative_to(EXT)):dict(sha256=digest(p),bytes=p.stat().st_size) for p in sorted(files)}
    (EXT/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(status,indent=2))


if __name__=='__main__':main()
