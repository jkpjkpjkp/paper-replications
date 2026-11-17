"""Reuse paper-aligned AlphaPROBE graph/retrieval/generation on CZ terminals."""
import argparse, importlib.util, json, os, sys, types, time, signal
import numpy as np
import pandas as pd
import torch
import requests
from symbolic_common import HERE,ROOT,NAMES,CZData,prepare,target_class,protocol,write_json,evaluate_pool

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args()
    out=HERE/'results/AlphaPROBE';out.mkdir(parents=True,exist_ok=True)
    runtime=prepare('alphaprobe-reproduction');protocol('alphaprobe-reproduction',out)
    if (out/'evaluation/report.json').exists():return
    original=ROOT/'alphaprobe-reproduction'
    common=types.ModuleType('common');common.ROOT=original;common.write_json=write_json
    common.LocalStockData=CZData;common.ForwardReturn=target_class();common.initialize=lambda:None
    sys.modules['common']=common
    sys.path.insert(0,str(original/'scripts'))
    import run_paper as runner
    import paper_protocol as pp
    from alphagen.data.expression_knowledge_graph import ExpressionKnowledgeGraph
    from alphagen.data.tree import ExpressionParser
    from alphagen.data.expression import Feature,FeatureType
    # Reuse already installed 4B Q8 embedding weights on CPU, avoiding competing GPU allocations.
    class Embedding(pp.OllamaEmbedding):
        def encode(self,texts):
            import hashlib
            vectors=[]
            for text in texts:
                p=self.cache_dir/(hashlib.sha256(text.encode()).hexdigest()+'.npy')
                if p.exists():v=np.load(p)
                else:
                    r=requests.post('http://127.0.0.1:11434/api/embed',json=dict(model=pp.EMBED_MODEL,input=text,truncate=False,
                        keep_alive='10m',options=dict(num_gpu=0,num_ctx=2048,num_batch=128,num_thread=2)),timeout=300)
                    r.raise_for_status();v=np.asarray(r.json()['embeddings'][0],dtype=np.float32)
                    assert v.shape==(2560,) and np.isfinite(v).all();v/=np.linalg.norm(v);np.save(p,v)
                vectors.append(v)
            return np.stack(vectors)
    def pool():
        data=CZData(start_time='2009-01-01',end_time='2019-12-31')
        p=pp.PaperPool(data,common.ForwardReturn(),Embedding(out/'embeddings'))
        g=ExpressionKnowledgeGraph(depth_limit=7,expression_size_limit=40);p.attach_knowledge_graph(g)
        return p,g,ExpressionParser()
    docs=pd.read_csv(ROOT/'Chen_Zimmermann_Factor_Zoo_2026_10/SignalDoc.csv').set_index('Acronym')
    descriptions={n:str(docs.loc[n,'LongDescription']) if n in docs.index else n for n in NAMES}
    # 29 named economic seed characteristics, fixed before mining; no performance sorting.
    seed_names=['AM','BM','EP','CF','SP','GP','OperProf','RoE','Investment','AssetGrowth','Accruals',
        'Mom12m','Mom6m','LRreversal','Beta','IdioVol3F','Illiquidity','DolVol','ShareVol','BidAskSpread',
        'NetPayoutYield','ShareIss1Y','EarningsSurprise','AnalystRevision','ShortInterest','OrgCap','RD','Cash','Leverage']
    assert len(seed_names)==29 and set(seed_names)<=set(NAMES)
    def seeds(p,g,parser):
        records=[]
        for name in seed_names:
            e=Feature(FeatureType[name.upper()]);desc=descriptions[name]
            node=g.build_expression_node(str(e),'CZ stock characteristics',desc,40)
            assert node is not None
            rec=p.admit(e,node,is_seed=True)
            if rec['admitted']:g.insert(node,None)
            records.append(rec)
        write_json(out/'seeds.json',records);assert p.size>0
    prompt=runner.PROMPT_FEATURES_AND_OPERATORS
    a=prompt.index('1. You can use');b=prompt.index('2. You can use')
    feature_text='1. Only these 202 CZ characteristics are available, already cross-sectional ranks in [-1,1], missing values imputed zero.\n'
    feature_text+='\n'.join('$'+n.lower()+': '+descriptions[n] for n in NAMES)+'\n'
    prompt=prompt[:a]+feature_text+prompt[b:]
    prompt=prompt.replace('days','months').replace('daily','monthly').replace("Today's",'Current month')
    prompt=prompt[:prompt.index('Examples of valid alpha expressions:')]+'Examples of valid alpha expressions:\nAdd($bm, $gp)\nSub($mom12m, $investment)\n'
    prompt+='Use positive rolling windows from 1,3,6,12,24 months. No future references. Predict next-month stock return.\n'
    runner.PROMPT_FEATURES_AND_OPERATORS=prompt
    runner.OUTPUT=out;runner.make_pool=pool;runner.seeds=seeds
    runner.status=lambda **kw:write_json(out/'status.json',dict(**kw,pid=os.getpid(),updated=time.time()))
    # Replace daily window-search values in a private function copy, preserving all admission logic.
    import inspect
    def bounded_post(*a,**kw):
        # requests read timeouts reset on incoming bytes; enforce a total wall deadline too.
        def expired(signum,frame):raise requests.Timeout('Generation request exceeded 180-second total wall deadline')
        previous=signal.signal(signal.SIGALRM,expired)
        signal.setitimer(signal.ITIMER_REAL,180)
        try:return requests.post(*a,**kw)
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
    runner.bounded_post=bounded_post
    request_source=inspect.getsource(runner.request).replace('requests.post(', 'bounded_post(').replace('timeout=300','timeout=(10,90)')
    exec(compile(request_source,str(out/'request_adapter.py'),'exec'),runner.__dict__)
    (out/'request_adapter.py').write_text(request_source)
    src=inspect.getsource(runner.candidate).replace('[5,10,20,30]','[1,3,6,12]')
    exec(compile(src,str(out/'candidate_adapter.py'),'exec'),runner.__dict__)
    (out/'candidate_adapter.py').write_text(src);(out/'feature_prompt.txt').write_text(prompt)
    write_json(out/'adaptation.json',dict(seed_features=seed_names,features=202,rounds=20,capacity=50,
        generator='deepseek-flash',embedding=pp.EMBED_MODEL,embedding_device='CPU',
        note='Original OHLCV seeds replaced by 29 prespecified CZ characteristics; all 202 terminals available to generation. No native-price expressions relabeled as CZ.'))
    if not (out/'mining-summary.json').exists():runner.run(preflight_only=args.preflight)
    if not args.preflight:
        evaluate_pool('alphaprobe-reproduction',out,15)
        write_json(out/'status.json',dict(stage='complete'))

if __name__=='__main__':
    try:main()
    except Exception as e:
        write_json(HERE/'results/AlphaPROBE/status.json',dict(stage='failed',error=repr(e)));raise
