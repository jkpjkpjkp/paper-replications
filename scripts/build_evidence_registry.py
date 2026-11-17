"""Bind draft paragraphs to inspected sources without claiming human approval."""
from pathlib import Path
import csv, hashlib, json, re

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence'
BASE=ROOT.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

local=[
 ('E001','User evaluation protocol','protocol.txt','entire file'),
 ('E002','Completed LiMT CZ202 experiment','limt-us-reproduction/experiments/standard_characteristics/protocol.json','entire specification; companion status.json and REPORT.txt'),
 ('E003','Independent recalculation of LiMT CZ202 results','ijcai-paper/evidence/independent_hac.csv','all 36 rows; primary_results.csv and audit_validation.json'),
 ('E004','LiMT CZ202 accounting and performance','limt-us-reproduction/experiments/standard_characteristics/analysis/performance.csv','full-period and subperiod rows; four ledgers checked by audit_results.py'),
 ('E005','LiMT CZ202 adverse missing-payoff diagnostic','limt-us-reproduction/experiments/standard_characteristics/analysis/missing_stress.csv','value and volume rows'),
 ('E006','LiMT CZ202 universe and feature audit','limt-us-reproduction/experiments/standard_characteristics/data/manifest.json','feature_names; universe_audit; data/universe_audit.csv'),
 ('E007','Legacy programme coverage snapshot','ijcai-paper/evidence/coverage_snapshot.json','source timestamp, categories, counts; catalogue_437.csv'),
 ('E008','Other saved LiMT US transfer','limt-us-reproduction/REPORT.txt','full-period equal-dollar ensemble results and limitations'),
 ('E009','Other saved zero-cost experiments','no-cost-neutral-audit/REPORT.txt','LiMT CSI300/CSI500 results and design limitations'),
 ('E010','Unfinished CS-GAT protocol transfer','financial-graph-reproduction/protocol202/protocol.json','prepared protocol and data manifest; no completed return-inference artifacts observed at initial audit'),
 ('E011','Post hoc paired block bootstrap','ijcai-paper/evidence/posthoc_bootstrap.csv','all four rows; scripts/audit_results.py'),
]
public=[
 ('E101','gu2020','Gu, Kelly and Xiu (2020)','https://dachxiu.chicagobooth.edu/download/ML.pdf','published article abstract; portfolios and Table8; p2267 microcap sensitivity','sources/gu2020.pdf'),
 ('E102','chen2022','Chen and Zimmermann (2022)','https://www.nowpublishers.com/article/Details/CFR-0112','published abstract and metadata; earlier FEDS text introduction distinguishes replication definitions','sources/chen2022.pdf'),
 ('E103','hou2017','Hou, Xue and Zhang (2017)','https://www.nber.org/papers/w23394','working-paper abstract; not the inaccessible 2020 full article',None),
 ('E104','novymarx2016','Novy-Marx and Velikov (2016)','https://www.nber.org/papers/w20721','earlier full text abstract; final citation confirmed by author publication list and Crossref','sources/novymarx2016.pdf'),
 ('E105','avramov2023','Avramov, Cheng and Metzker (2023)','https://cris.huji.ac.il/en/publications/machine-learning-vs-economic-restrictions-evidence-from-stock-ret/','institutional abstract and publication metadata',None),
 ('E106','azevedo','Azevedo, Hoegner and Velikov, archived working paper','https://afajof.org/management/viewp.php?n=75544','archived undated manuscript abstract and introduction; not claimed to be latest2026 version','sources/azevedo.pdf'),
 ('E107','harvey2016','Harvey, Liu and Zhu (2016)','https://people.duke.edu/~charvey/Research/Published_Papers/P118_and_the_cross.PDF','published abstract and multiple-testing framework','sources/harvey2016.pdf'),
 ('E108','white2000','White (2000)','https://www.econometricsociety.org/publications/econometrica/2000/09/01/reality-check-data-snooping','publisher abstract, metadata; full text retrieved from university','sources/white2000.pdf'),
 ('E109','newey1987','Newey and West (1987)','https://www.ssc.wisc.edu/~kwest/publications/1980/A%20Simple%20PSD%20HAC%20Covariance%20Matrix.pdf','published PDF, p703 inspected visually (scanned source); author order from PDF overrides society metadata error','sources/newey1987.pdf'),
 ('E110','yang2026','Yang et al. (2026), LiMT','https://arxiv.org/abs/2609.25617','arXiv title/authors/date/abstract; local original paper and code retained in source project',None),
 ('E111','yan2026','Yan et al. (2026), Arboreal Neural Network','https://proceedings.mlr.press/v306/yan26o.html','official PMLR abstract, author list, citation',None),
 ('E112','french','Kenneth French official factor definitions','https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html','FF5 2x3 and daily Mom definition pages opened',None),
 ('E113','czdata','CZ October2026 release','https://www.openassetpricing.com/data/','current release, revisions, option-signal early end dates',None),
 ('E114','ijcai2027','IJCAI2027 main-track instructions','https://2027.ijcai.org/ijcai-2027-call-for-papers-main-track/','page limits, anonymity, LLM policy, deadlines; official template link and archive',None),
 ('E115','workflow','Scientific Agent Skills workflow attribution','https://arxiv.org/abs/2609.00065','current arXiv authors, year, v2 revision and title opened; attribution in research_notes.txt only',None),
]
sources=[]
for eid,title,rel,loc in local:
    p=BASE/rel;assert p.is_file(),p
    sources.append(dict(evidence_id=eid,title=title,source_type='local_artifact',
        confidentiality='local_research_artifact_not_cleared_for_redistribution',
        authors=[],year=None,identifiers={'url':''},locator=loc,relative_path=rel,
        sha256=sha(p),verification=dict(source_opened=True,status='unverified',
        verified_by='',verified_on=''),machine_review=dict(by='Codex',on='2026-11-10',
        status='inspected_or_recomputed',human_approval='pending')))
for eid,key,title,url,loc,rel in public:
    s=dict(evidence_id=eid,title=title,bibtex_key=key,source_type='public_primary_source',
        confidentiality='public',authors=[],year=None,identifiers={'url':url},locator=loc,
        verification=dict(source_opened=True,status='unverified',verified_by='',verified_on=''),
        machine_review=dict(by='Codex',on='2026-11-10',status='source_content_inspected',human_approval='pending'))
    if rel:s.update(relative_path=rel,sha256=sha(ROOT/rel))
    sources.append(s)
(OUT/'source_manifest.json').write_text(json.dumps(dict(schema_version='1.0',sources=sources,
    note='Unverified means accountable human verification is pending; machine inspection is recorded separately.'),indent=2)+'\n')

bykey={s['bibtex_key']:s['evidence_id'] for s in sources if 'bibtex_key' in s}
text=(ROOT/'paper.tex').read_text();lines=text.splitlines();rows=[];section='preamble'
# Paragraph-level binding: exact text hashes support change detection. Proposals
# and mathematical interpretation are labelled separately from empirical findings.
for i,line in enumerate(lines,1):
    m=re.match(r'\\(?:sub)?section\*?\{(.+)\}',line)
    if m:section=m.group(1)
    if not line or line.startswith(('%','\\')) or len(line)<120:continue
    evidence=set()
    for keys in re.findall(r'\\cite\{([^}]+)\}',line):
        for k in keys.split(','):evidence.add(bykey[k])
    kind='interpretation';intent='descriptive_audit'
    if section=='preamble':evidence.update(['E001','E002','E003','E004','E007'])
    if section in ['Introduction','Related Evidence and Scope']:evidence.update(['E001','E002','E003','E007','E008','E009'])
    if section in ['Investment Universe and Portfolio','Risk Adjustment and Uncertainty']:evidence.update(['E001','E002']);kind='methodological'
    if section=='Coverage Is Not a Failure Rate':evidence.update(['E007','E010']);kind='factual'
    if section=='LiMT Transfer Design':evidence.update(['E002','E006']);kind='methodological'
    if section=='Independent Recalculation':evidence.update(['E003','E004','E011']);kind='methodological'
    if section=='Positive Returns Do Not Establish Positive Alpha':evidence.update(['E003','E004']);kind='numeric_result'
    if section=='Robustness and Precision':evidence.update(['E003','E005','E006','E011']);kind='numeric_result'
    if section=='Other Saved Designs Prevent a Universal Claim':evidence.update(['E008','E009']);kind='numeric_result'
    if section in ['Three Distinct Claims','The Dummy Baseline Must Be Defined']:evidence.update(['E001','E002','E003']);kind='mathematical_or_methodological_argument';intent='not_an_empirical_baseline_result'
    if section=='A Publication Standard for Return Claims':evidence.update(['E001','E003','E007']);kind='proposal';intent='recommendation'
    if section=='Limitations and Conclusion':evidence.update(['E002','E003','E004','E007']);kind='interpretation'
    if section=='Ethical Considerations':evidence.update(['E002']);kind='draft_declaration';intent='human_release_permissions_pending'
    assert evidence,(i,line)
    rows.append(dict(claim_id=f'C{len(rows)+1:03d}',section=section,claim_kind=kind,
        claim_text_sha256=hashlib.sha256(' '.join(line.split()).encode()).hexdigest(),
        evidence_ids=','.join(sorted(evidence)),verification_status='human_verification_pending',
        uncertainty='see_named_intervals_and_limitations',analysis_intent=intent,
        tex_line=i,binding_unit='exact_tex_paragraph',source_opened_by_machine=True))
with (OUT/'claim_evidence.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
manifest=dict(schema_version='1.0',document_id='ijcai-finance-ml-audit',
    draft_status='draft',study_design='retrospective_artifact_audit_with_targeted_narrative_review',
    submission_ready=False,human_verification=dict(completed=False,verified_by='',verified_on=''),
    confidentiality_review=dict(completed=False,external_services_authorized=False,policy_checked=False,
        verified_by='',verified_on='',note='Local research processed locally; only public research queries and URLs used externally.'),
    required_statements=dict(ai_disclosure='documented_in_research_notes_pending_final_author_review',
        author_contributions='unresolved',code_availability='local_only_public_release_unresolved',
        conflicts='unresolved',consent='not_assumed_applicable',data_availability='licensed_sources_local',
        ethics='draft_in_manuscript',funding='unresolved'),
    reporting_guidelines=['IJCAI2027_official_main_track_instructions'],
    registries=dict(source_manifest='source_manifest.json',claim_evidence='claim_evidence.csv',
        consistency_manifest='audit_validation.json',reporting_coverage='reporting_coverage.json',
        authorship_manifest='authorship.json'))
(OUT/'manuscript_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(OUT/'authorship.json').write_text(json.dumps(dict(authors=[],order_approved=False,
    guarantor=None,AI_is_not_author=True,human_approval='pending'),indent=2)+'\n')
topics=[('study_scope','addressed','Sections1,2,4,8'),('protocol_and_timing','addressed','Sections3,4'),
    ('estimates_and_uncertainty','addressed','Section5; evidence CSVs'),
    ('matched_predictive_baselines','missing','No matching CZ202 baseline ledger; Section6 states limitation'),
    ('multiplicity','addressed','Holm36 only; unrecorded prior search unresolved'),
    ('external_validity','addressed','Sections4,8'),('review_completeness','not_applicable','Targeted narrative review, not systematic'),
    ('authorship_and_declarations','missing','Accountable authors must finalize'),
    ('release_permissions','missing','Local derived outputs do not establish redistribution rights')]
(OUT/'reporting_coverage.json').write_text(json.dumps(dict(schema_version='1.0',
    disclaimer='Non-scoring audit aid, not a compliance certificate.',
    guideline_id='custom_financial_ml_evidence_audit',
    items=[dict(topic=a,status=b,locator_or_reason=c) for a,b,c in topics]),indent=2)+'\n')
print(f'Registered {len(sources)} sources and {len(rows)} exact paragraph bindings; all human approvals pending.')
