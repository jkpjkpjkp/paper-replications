"""Resumable original-setting release replay; resource policy is set by systemd."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def atomic(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False,default=str));temp.replace(path)


def resources():
    relative=Path('/proc/self/cgroup').read_text().strip().split('::',1)[1]
    cg=Path('/sys/fs/cgroup')/relative.lstrip('/')
    values={name:(cg/name).read_text().strip()for name in ['memory.peak','memory.max','memory.high','memory.swap.max','memory.events']}
    values['peak_gib']=int(values['memory.peak'])/2**30
    history_path=ROOT/'runtime/resource_history.json'
    history=json.loads(history_path.read_text())if history_path.exists()else[]
    if not history and(ROOT/'runtime/resources.json').exists():history.append({'cgroup_id':'earlier_service_record',**json.loads((ROOT/'runtime/resources.json').read_text())})
    cgroup_id=str(cg.stat().st_ino)
    history=[record for record in history if record.get('cgroup_id')!=cgroup_id]
    history.append({'cgroup_id':cgroup_id,**values});atomic(history_path,history)
    values['study_peak_gib']=max(record['peak_gib']for record in history)
    atomic(ROOT/'runtime/resources.json',values)


def make_jobs(protocol):
    jobs=[]
    def add(market,mode,seed=0,alpha=.5,selectors=None,phase=0):
        selectors=selectors or ['oscar']
        name=f'{phase}-{market}-{mode}-a{alpha:g}-seed{seed}-'+','.join(selectors)
        command=[sys.executable,'-u','reproduction/run_native.py','--mode',mode,'--dataset',market,'--seed',str(seed),
            '--alpha',str(alpha),'--hidden_dims','512','256','--batchsize','64','--epochs','300','--cuda','-1',
            '--splits','test','--save_dir','artifacts_released','--selector',*selectors,'--print_every','100']
        jobs.append({'id':name,'market':market,'mode':mode,'seed':seed,'alpha':alpha,'selectors':selectors,'phase':phase,'command':command,'state':'pending'})
    for m in protocol['markets']:add(m,'historic',selectors=['oscar','ssrm','oscar_hard'],phase=0)
    for seed in protocol['seeds']:
        for m in protocol['markets']:add(m,'pfl',seed,selectors=['oscar','ssrm','oscar_hard'],phase=1)
    for seed in protocol['seeds']:
        for m in protocol['markets']:add(m,'dfl',seed,phase=2)
    for alpha in [0.,.1,.9,1.]:
        for seed in protocol['seeds']:add('euro50','dfl',seed,alpha,phase=3)
    for m in protocol['markets']:add(m,'historic',selectors=['sdr'],phase=4)
    for seed in protocol['seeds']:
        for m in protocol['markets']:add(m,'pfl',seed,selectors=['sdr'],phase=4)
    return jobs


def main():
    (ROOT/'runtime/logs').mkdir(parents=True,exist_ok=True)
    protocol=json.loads((ROOT/'reproduction/protocol.json').read_text())
    assert json.loads((ROOT/'runtime/equivalence.json').read_text())['passed']
    assert subprocess.run(['git','diff','--quiet','--','src'],cwd=ROOT/'upstream').returncode==0
    path=ROOT/'runtime/jobs.json';jobs=json.loads(path.read_text())if path.exists()else make_jobs(protocol)
    for job in jobs:
        if job['state']=='running':job['state']='pending'
    def priority(job):
        if job['phase']==0:return(0,0,protocol['markets'].index(job['market']),0)
        if job['phase']in[1,2]:return(1,protocol['markets'].index(job['market']),protocol['seeds'].index(job['seed']),job['phase'])
        return(job['phase'],0,0,0)
    jobs.sort(key=priority)
    atomic(path,jobs)
    while True:
        for job in jobs:
            if job['state']=='blocked_data'and(ROOT/'data/daily'/f"{job['market']}_160101_251231.csv").exists():job['state']='pending'
        pending=next((job for job in jobs if job['state']=='pending'),None)
        if pending is None:break
        job=pending
        if not(ROOT/'data/daily'/f"{job['market']}_160101_251231.csv").exists():
            job.update(state='blocked_data',reason='Original-market input panel has not been acquired; no substitute market or synthetic data used')
            atomic(path,jobs);continue
        source=ROOT/'data/daily'/f"{job['market']}_160101_251231.csv"
        receipt=json.loads((ROOT/'data'/f"{job['market']}_receipt.json").read_text())
        assert hashlib.sha256(source.read_bytes()).hexdigest()==receipt['sha256'],'Frozen financial input changed; no run may mix data vintages'
        if job['phase']==4 and not(ROOT/'runtime/sdr_equivalence.json').exists():
            with(ROOT/'runtime/sdr-equivalence.log').open('a')as stream:
                subprocess.run([sys.executable,'reproduction/verify_sdr_runtime.py'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        job.update(state='running',started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        atomic(path,jobs);atomic(ROOT/'runtime/status.json',{'stage':'running','current_job':job['id'],'done':sum(j['state']=='complete'for j in jobs),'total':len(jobs)})
        print('START',job['id'],flush=True)
        with(ROOT/'runtime/logs'/f"{job['id']}.log").open('a')as log:
            result=subprocess.run(job['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        job.update(state='complete'if result.returncode==0 else'failed',exit_code=result.returncode,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        atomic(path,jobs);resources()
        print(job['state'].upper(),job['id'],flush=True)
        compare=ROOT/'reproduction/compare_published.py'
        if compare.exists()and any((ROOT/'artifacts_released').glob('*/summary_metrics.csv')):
            with(ROOT/'runtime/comparison.log').open('a')as stream:
                subprocess.run([sys.executable,str(compare),'--artifacts','artifacts_released','--baseline-oscar','oscar_hard'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    resources();atomic(ROOT/'runtime/status.json',{'stage':'complete'if all(j['state']=='complete'for j in jobs)else'partial','counts':{state:sum(j['state']==state for j in jobs)for state in ['complete','failed','blocked_data']},'total':len(jobs),'exact_published_results_replicated':False})


if __name__=='__main__':main()
