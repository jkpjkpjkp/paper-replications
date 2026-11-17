"""Read the user's source workbook without changing it and freeze the work order."""
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
ORDER = ['AlphaSAGE', 'DoubleAdapt', 'AlphaGen', 'TRA', 'AlphaQCM', 'Lookahead factors']


def main():
    with zipfile.ZipFile(BASE / 'reference.xlsx') as z:
        strings = [''.join(s.itertext()) for s in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si', NS)] if 'xl/sharedStrings.xml' in z.namelist() else []
        sheets = {}
        for name in z.namelist():
            if re.fullmatch(r'xl/worksheets/sheet\d+\.xml', name):
                rows = []
                for row in ET.fromstring(z.read(name)).findall('.//s:row', NS):
                    values = {}
                    for c in row.findall('s:c', NS):
                        v, inline = c.find('s:v', NS), c.find('s:is', NS)
                        val = v.text if v is not None else ''.join(inline.itertext()) if inline is not None else ''
                        if c.get('t') == 's':
                            val = strings[int(val)]
                        values[re.sub(r'\d', '', c.get('r'))] = val
                    if values:
                        rows.append({'row': int(row.get('r')), **values})
                sheets[name] = rows
    ref = {r.get('A'): r for r in sheets['xl/worksheets/sheet3.xml']}
    papers = []
    for line in (BASE / 'paperlist.txt').read_text().splitlines():
        name = line.split('\t')[0]
        r = ref[name]
        papers.append(dict(paper=name, priority=ORDER.index(name)+1 if name in ORDER else None,
                           stage='not_started' if name in ORDER else 'deferred_by_user',
                           original_experiment=r.get('K'), original_gate='not_assessed',
                           full_cz_stage='waiting_for_original_experiment',
                           compute_tflops=[r.get('E'), r.get('F')], runtime=r.get('J'),
                           known_requirements=r.get('L'), data=r.get('M'),
                           source_url=r.get('P'), code_url=r.get('Q'), commit=r.get('S'),
                           reference_location=f'Reproduction!A{r["row"]}:S{r["row"]}'))
    papers.sort(key=lambda p: p['priority'] or 999)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  scope='User clarified: factor models first; allocation and trading-policy methods deferred.',
                  ordering='Ascending geometric midpoint of workbook TFLOP ranges for the six factor models. Estimates overlap; not measured runtime.',
                  original_gate='One complete own-method experiment, correct data/preprocessing/settings, independent metric verification and comparison to the paper. Completion alone is not validation. No tuning to test results.',
                  protocol=(BASE/'protocol.txt').read_text(),
                  source_sha256={p: hashlib.sha256((BASE/p).read_bytes()).hexdigest() for p in ['reproduce.txt','paperlist.txt','reference.xlsx','protocol.txt']},
                  papers=papers)
    (ROOT/'reference-extract.json').write_text(json.dumps(sheets, indent=2, ensure_ascii=False))
    target = ROOT/'queue.json'
    if target.exists():
        raise RuntimeError('Refusing to overwrite a queue with live progress')
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps([{'priority':p['priority'],'paper':p['paper'],'stage':p['stage']} for p in papers], indent=2))


if __name__ == '__main__':
    main()
