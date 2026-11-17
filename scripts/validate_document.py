"""Final source bindings, citation coverage, PDF and layout checks."""
from pathlib import Path
import csv,hashlib,json,re,subprocess,sys,xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'evidence'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
tex=(ROOT/'paper.tex').read_text();lines=tex.splitlines()
claims=list(csv.DictReader((OUT/'claim_evidence.csv').open()))
sources=json.loads((OUT/'source_manifest.json').read_text())['sources']
ids={s['evidence_id'] for s in sources}
for c in claims:
    line=lines[int(c['tex_line'])-1]
    assert hashlib.sha256(' '.join(line.split()).encode()).hexdigest()==c['claim_text_sha256'],c['claim_id']
    assert set(c['evidence_ids'].split(','))<=ids
    assert c['verification_status']=='human_verification_pending'
for s in sources:
    if 'relative_path' in s:
        p=(ROOT.parent/s['relative_path']) if s['source_type']=='local_artifact' else (ROOT/s['relative_path'])
        assert sha(p)==s['sha256'],s['evidence_id']
keys=set(re.findall(r'@\w+\{([^,]+),',(ROOT/'references.bib').read_text()))
cites=set(k for match in re.findall(r'\\cite\{([^}]+)\}',tex) for k in match.split(','))
assert cites==keys,(cites-keys,keys-cites)
log=(ROOT/'paper.log').read_text(errors='replace')
assert 'Overfull' not in log
assert not re.search(r'(Citation .+ undefined|Reference .+ undefined|Undefined control sequence)',log)
fonts=subprocess.check_output(['pdffonts',str(ROOT/'paper.pdf')],text=True)
assert 'Type 3' not in fonts
data=subprocess.check_output(['pdftotext','-layout',str(ROOT/'paper.pdf'),'-'],text=True)
pages=[p for p in data.split('\f') if p.strip()]
assert len(pages)==6
assert 'References' in pages[5] and all('References' not in p for p in pages[:5])
assert 'Research draft' in pages[0] and '?' not in pages[0]
assert not re.search(r'/home/|jkp|@gmail|\[\[TODO',data)
bbox=subprocess.check_output(['pdftotext','-bbox',str(ROOT/'paper.pdf'),'-'],text=True)
doc=ET.fromstring(bbox)
words=doc.findall('.//{http://www.w3.org/1999/xhtml}word')
assert words
for w in words:
    assert 0<=float(w.attrib['xMin'])<float(w.attrib['xMax'])<=612
    assert 0<=float(w.attrib['yMin'])<float(w.attrib['yMax'])<=792
assert sha(ROOT/'ijcai27.sty')==sha(ROOT/'template/ijcai27.sty')
assert sha(ROOT/'named.bst')==sha(ROOT/'template/named.bst')
validation=dict(status='passed',pdf_pages=6,body_pages=5,reference_pages=1,
    bibliography_entries=len(keys),bound_paragraphs=len(claims),sources=len(sources),
    undefined_citations=False,overfull_boxes=False,type3_fonts=False,
    text_outside_page=False,official_style_unmodified=True,
    visual_review=dict(by='Codex',pages=[1,2,3,4,5,6],result='no_clipping_or_overlapping_elements_observed'),
    human_source_verification='pending',submission_ready=False,
    paper_pdf_sha256=sha(ROOT/'paper.pdf'),paper_tex_sha256=sha(ROOT/'paper.tex'),
    bibliography_sha256=sha(ROOT/'references.bib'))
(OUT/'document_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
print(json.dumps(validation,indent=2))
