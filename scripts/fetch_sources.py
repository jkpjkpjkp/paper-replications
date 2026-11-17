"""Download public research sources; no local research data are transmitted."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib, json, requests, subprocess, zipfile, io

ROOT = Path(__file__).resolve().parents[1]
URLS = {
    'hou2020': 'https://global-q.org/uploads/1/2/2/6/122679606/houxuezhang2020rfs.pdf',
    'gu2020': 'https://dachxiu.chicagobooth.edu/download/ML.pdf',
    'chen2022': 'https://www.federalreserve.gov/econres/feds/files/2021-037pap.pdf',
    'harvey2016': 'https://people.duke.edu/~charvey/Research/Published_Papers/P118_and_the_cross.PDF',
    'novymarx2016': 'https://www.nber.org/system/files/working_papers/w20721/w20721.pdf',
    'azevedo': 'https://afajof.org/management/viewp.php?n=75544',
}
def fetch(item):
    key, url = item
    try:
        r=requests.get(url, timeout=45); r.raise_for_status()
        if not r.content.startswith(b'%PDF'): raise ValueError('Not a PDF')
        p=ROOT/'sources'/f'{key}.pdf'; p.write_bytes(r.content)
        subprocess.run(['pdftotext','-layout',str(p),str(p.with_suffix('.txt'))],check=True)
        return {'id':key,'url':url,'bytes':len(r.content),'sha256':hashlib.sha256(r.content).hexdigest(),'status':'retrieved'}
    except Exception as e: return {'id':key,'url':url,'status':'failed','error':str(e)}

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=6) as pool: records=list(pool.map(fetch,URLS.items()))
    (ROOT/'sources'/'download_log.json').write_text(json.dumps(records,indent=2))
    for r in records: print(r['id'],r['status'],r.get('bytes',r.get('error')))
    url='https://ijcai.org/sites/default/files/Guidelines/FormattingGuidelines-IJCAI-27.zip'
    r=requests.get(url,timeout=40);r.raise_for_status()
    (ROOT/'sources'/'FormattingGuidelines-IJCAI-27.zip').write_bytes(r.content)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for n in z.namelist():
            if not n.startswith('__MACOSX/') and not '..' in Path(n).parts:z.extract(n,ROOT/'template')
    print('template downloaded',len(r.content))
