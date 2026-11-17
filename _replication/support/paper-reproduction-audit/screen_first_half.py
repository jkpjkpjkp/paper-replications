"""Provisional title/topic screening; this does not verify source contents."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAPERS = json.loads((ROOT / "papers.json").read_text())
REASONS = """
1|uncertain|Real disclosure and banking workflow data are plausible; verify released task files and a feasible small-model or stored-prediction evaluation.
2|uncertain|Finance spreadsheet tasks fit the data criterion if authentic; agent/model requirements and released files remain unverified.
3|uncertain|Financial charts and reports can be real data; multimodal model sizes and dataset authenticity need checking.
4|uncertain|Financial reports may be authentic; verify accessible evaluation data, judging dependence, and local model requirements.
5|uncertain|Real options trading data are indicated; intraday options data access and LLM-agent cost could prevent full reproduction.
6|uncertain|Real economic embeddings are indicated, but multiple historical language-model training runs may exceed workstation capacity; check released embeddings/checkpoints.
7|candidate|Prediction-market comments and transactions are explicitly indicated; likely manageable evaluation if released artifacts exist.
8|uncertain|Real stock alpha mining is indicated and local work exists; verify LLM dependency, data release, and what remains reproducible.
9|uncertain|Real stock alpha mining is indicated and local work exists; source/data and retrieval/evolution resource requirements need auditing.
10|candidate|Money-laundering graph classification is a plausible moderate-size real-finance experiment; verify graph release and scale.
11|candidate|Stock time-series forecasting is directly empirical and local Hermes artifacts exist; audit completion and exact data protocol.
12|screen_excluded|Supplied description concerns sandwich-equilibrium thresholds and theory; no real-data experiment is indicated.
13|uncertain|The main contribution is theoretical online portfolio selection; check whether the existing project or paper includes real-market experiments.
14|screen_excluded|Supplied description is a frontier-model corporate-governance decision experiment; authentic financial observations are not indicated.
15|uncertain|Real cryptocurrency portfolio data are plausible, but multiple LLM experts and released outputs/model sizes require checking.
16|candidate|Ethereum account classification has an explicit real blockchain setting and a potentially tractable graph method; verify dataset scale.
17|uncertain|Financial graphs with aligned news/regulatory text appear empirical; data availability, graph size, and text encoding need verification.
18|candidate|Empirical SDF portfolio modeling is indicated; moderate diffusion models may fit the workstation, subject to released data and model sizes.
19|candidate|Real stock asset pricing is indicated and local Stable Alpha artifacts exist; audit real-data coverage and training budget.
20|uncertain|Financial tail-risk generation may use real returns or synthetic processes; establish real-data experiments and diffusion training size.
21|uncertain|Market-1T second-level data are explicitly large and may exceed disk/compute; check released small benchmark splits rather than assuming full feasibility.
22|candidate|Prediction-market empirical reproduction already has local artifacts; audit data provenance, completed experiments, and missing baselines.
23|candidate|Small-sample financial series generation with random convolutional features looks resource-compatible; verify real input series and code.
24|uncertain|SEC disclosures are authentic data, but reconstructing the full pretraining corpus may be excessive; check bounded released evaluation tasks.
25|screen_excluded|Risk-attitude allocation tasks are described as experimental prompts rather than observations of real financial data.
26|screen_excluded|Supplied description constructs textbook-derived questions; real financial observations are not indicated.
27|uncertain|Real legal cases are indicated; check which cases contain financial data and whether released evaluations work with feasible models.
28|uncertain|Smart-contract bytecode can be real blockchain data; confirm actual contracts, benchmark scope, and feasible decompiler model evaluation.
29|candidate|Training-free graph anomaly scoring may be cheap and includes financial fraud networks; verify that at least one released real graph fits memory.
30|uncertain|Financial time-series tasks may qualify; agent training/model sizes and real-finance benchmark release need checking.
31|uncertain|News-driven stock and banking tasks may use real observations; establish authentic datasets and manageable test-generation/model costs.
32|candidate|Ethereum phishing graph experiments explicitly concern real finance; verify released graph size and method implementation.
33|candidate|S&P 500 option calibration explicitly supplies a real financial application; check option data availability and numerical training budget.
34|uncertain|Authentic financial reports/disclosures are explicitly indicated; verify files and feasible spreadsheet-agent inference/evaluation.
35|uncertain|Banking account/tool tasks may be simulated; establish whether authentic financial data are actually used and whether models fit locally.
36|candidate|Sparse portfolio optimization is computationally modest and existing local work is present; verify real market inputs and completed comparisons.
37|uncertain|Real-user financial QA is indicated; released data, accessible evidence, and small-model evaluation need verification.
38|candidate|Decision-focused sparse portfolio optimization has local artifacts and a real-market setting; audit exact runs and missing experiments.
39|uncertain|American-option dynamic-programming experiments may use only simulated price processes; verify real-data eligibility before extending local work.
40|uncertain|Digital-twin trading may be calibrated to real markets or synthetic-only; check original data and self-play training size.
41|uncertain|Real merger-arbitrage outcomes and documents likely qualify; long-context LLM size, proprietary event labels, and released outputs need checking.
42|candidate|U.S. equity state learning clearly targets real market observations; verify source panel, network size, and released implementation.
43|uncertain|ESG/news/returns are authentic if released; proprietary ESG labels and domain language-model requirements could block reproduction.
44|uncertain|Large-scale generative-agent simulation may exceed budget; check whether real-data comparisons and released simulation traces can be reproduced locally.
45|candidate|Signature-based transformer asset allocation is a plausible workstation experiment on market data; verify universe, lengths, and code.
46|candidate|Supervision-horizon comparisons on financial forecasting look tractable; verify authentic panels and the exact models and split protocol.
47|uncertain|Professional financial-agent tools may use real filings; benchmark release and inference/judging requirements remain unknown.
48|screen_excluded|Position paper on evaluation bias; supplied record does not identify a reproducible empirical method using real financial data.
49|candidate|Workbook notes explicitly identify real multi-period portfolio experiments; optimization method should be feasible subject to implementation audit.
50|candidate|Portfolio risk-control experiments are explicitly indicated; confirm real price data rather than simulation and exact online protocol.
51|candidate|Real loan applications explicitly underpin TabCredit and existing local work; large row count warrants streaming/memory audit.
52|uncertain|Loan-approval prompts may be generated scenarios rather than authentic loan data; verify data and feasible model evaluations.
53|uncertain|Finance/crypto social-belief tasks may qualify; verify authentic financial records and model/training scale.
54|uncertain|Financial MCP sandbox may be synthetic; establish authentic data use and feasible supported agent evaluation.
55|candidate|Workbook notes explicitly say real stock-market data; modular conformal prediction is likely tractable.
56|candidate|Real portfolio allocation is explicitly indicated; constrained layers and training dimensions need source verification.
57|uncertain|Fraud experiments are indicated without a confirmed financial dataset; verify data domain, authenticity, and algorithm runtime.
58|uncertain|Corporate-law scenarios may derive from real cases or synthetic prompts; real financial data and feasible model evaluation need confirmation.
59|uncertain|Financial extrapolation could involve only synthetic SDE/PDE paths; local existing work needs a real-data eligibility audit.
60|uncertain|Financial agent examples alone do not establish authentic data experiments; verify the financial dataset and manageable attack/model evaluation.
61|uncertain|A financial transfer experiment is mentioned but the task is mainly clinical; verify actual real-finance data and compressor/model scale.
62|candidate|Workbook notes explicitly identify financial-economic data and local FlowMSM artifacts; audit real-data cases separately from synthetic validation.
63|uncertain|Lending with selective labels can be simulation or real-loan based; verify the empirical dataset and long-term evaluation protocol.
64|candidate|Portfolio optimization is explicitly empirical and an upstream checkout exists locally; verify market panel and computational cost.
65|screen_excluded|Supplied notes explicitly call this a fairness simulation benchmark with a lending environment; no real financial observations are indicated.
66|uncertain|Finance domain adaptation may use authentic corpora; training base-model size and released finance evaluation data need checking.
67|uncertain|Supplied notes describe a theoretical portfolio-to-conformal reduction, not investing; check if any real-financial-series experiment exists.
68|candidate|Stock-series benchmark is explicitly indicated and analytic generation may be inexpensive; verify real inputs and released implementation.
69|candidate|Five single-stock and Dow-30 trading environments explicitly indicate real-market tasks; confirm data provenance and evaluation budget.
70|screen_excluded|Finance association comes from scam-link motivation; no authentic financial-data empirical task is indicated in supplied notes.
71|uncertain|Financial spreadsheet assessments may use real inputs; verify authentic finance files and manageable generation/judging setup.
72|uncertain|Although a position paper, finance is an explicit empirical comparison domain; inspect whether released benchmark runs use authentic data and feasible models.
73|screen_excluded|Online bilateral-trade theory/benchmark; supplied record does not indicate real financial observations.
74|screen_excluded|General data-acquisition/pricing theory rather than real financial-data experiments.
75|screen_excluded|General matching-market bandit theory; no real financial-data application is indicated.
76|screen_excluded|Advertising creative/auction application; real financial data as required by the user are not indicated.
77|screen_excluded|Procurement-auction mechanism theory; no authentic financial dataset is indicated.
78|screen_excluded|AI-model market arbitrage concerns computational service markets; no real financial dataset is indicated.
79|screen_excluded|General dynamic matching-market bandits; no real financial dataset is indicated.
80|screen_excluded|General contextual pricing; supplied record does not establish a finance dataset.
81|screen_excluded|Advertising bidding application; no real financial dataset is indicated.
82|screen_excluded|Auction payment/mechanism theory; no real financial observations are indicated.
83|screen_excluded|Data-market equilibrium/pricing theory; no real financial dataset is indicated.
84|screen_excluded|General data pricing and market equilibrium; no real financial dataset is indicated.
85|screen_excluded|LLM token pricing/incentives are adjacent economics; no real financial-data experiment is indicated.
86|screen_excluded|Advertising auction monotonicity theory; no real financial dataset is indicated.
87|screen_excluded|AutoML marketplace pricing; no real financial dataset is indicated.
88|screen_excluded|First-price auction revenue theory; no real financial observations are indicated.
89|screen_excluded|Advertising auction incentive compatibility; no real financial dataset is indicated.
90|screen_excluded|Digital-data sales mechanism theory; no real financial dataset is indicated.
91|screen_excluded|Search-ad auction causal estimation concerns advertising data, without an indicated finance dataset.
92|screen_excluded|Welfare/accuracy auction mechanism; real financial observations are not indicated.
93|screen_excluded|AI labour-market agent competition; real financial observations are not indicated.
94|screen_excluded|Position paper on agent collusion certification; no real financial-data reproduction is indicated.
95|screen_excluded|Position paper on market selection and learning; no real financial-data empirical task is indicated.
96|screen_excluded|Position paper on token taxation; no real financial-data empirical task is indicated.
97|uncertain|Stock factor discovery plausibly uses real market data; self-evolving LLM reinforcement learning may be too large unless released models/outputs suffice.
98|uncertain|Alpha evaluation likely uses real market panels; verify public tasks and small-model/stored-generation reproduction paths.
99|candidate|GFlowNet factor mining is a plausible moderate workstation task and existing local AlphaSAGE artifacts are present.
100|candidate|Crypto series generation can use modest real market histories; verify source datasets and benchmark model budgets.
101|uncertain|Japanese financial statements are authentic if released; verify accessible labels, parsing resources, and feasible LLM evaluation.
102|uncertain|Expert financial search may use live/authentic material; dataset access and model/judge costs remain unknown.
103|candidate|Healthcare insurance fraud detection is an empirical financial claims task; verify authentic graph release and memory requirements.
104|uncertain|Financial technical-analysis series likely qualify; reasoning-model training scale and released checkpoints require verification.
105|candidate|Black-Litterman allocation with conditional diffusion is plausibly modest on market data; verify portfolio universe and model size.
106|screen_excluded|Supplied topic explicitly describes the InvestESG multi-agent simulation; no authentic financial observations are indicated.
107|uncertain|Stock/crypto minute data may qualify; multi-agent LLM and data scale/access need checking.
108|screen_excluded|Supplied description is a simulated social-platform fraud experiment; real financial observations are not indicated.
109|uncertain|Financial PDE benchmark may be synthetic-only; check for authentic observed prices before considering local reproduction.
110|uncertain|Portfolio optimization is an application, but input authenticity and optimization/training sizes require checking.
111|uncertain|Finance analysis is one MCP task domain; verify authentic released financial data and feasible model evaluation.
112|screen_excluded|MBA professional knowledge questions do not by themselves establish use of real financial data.
113|uncertain|Real-domain financial time-series reasoning is indicated; confirm released authentic series and feasible agent/model setup.
114|screen_excluded|Generative-service market equilibria and strategic entry are adjacent theory; no real financial dataset is indicated.
115|screen_excluded|General demand/contextual-pricing method; real financial data are not established by the supplied record.
116|uncertain|Crypto prices and financial news likely meet the data criterion; released demo artifacts and summarizer model size need verification.
117|uncertain|Portfolio development platform can backtest authentic prices; verify whether published experiments and agent models are reproducible locally.
118|uncertain|Drug-investment due diligence is finance-adjacent but may use biological rather than financial outcomes; verify authentic investment data and accessible small-model workflow.
119|uncertain|Real retail lending is explicit, but production preference-rate data may be proprietary; verify released experiments and matching data availability.
120|uncertain|Interactive trading simulation may be fitted to real market data; verify authentic calibration/evaluation observations and training size.
121|uncertain|Sensitive financial information protection may use generated templates; verify authentic public financial examples and feasible model evaluation.
122|candidate|Event-driven stock predictions use financial news/prices and perturbations can be modest; verify released models/data and attack budget.
123|uncertain|Related-party transactions are financial observations, but graph release and production-data scale are unknown.
124|uncertain|Indian stock experiments may qualify and be small, but the student abstract may not provide enough method/data detail for faithful reproduction.
125|screen_excluded|Student abstract describes synthetic dialogue generation; authentic financial observations are not indicated.
126|uncertain|Order-flow generation may use genuine market calibration data; verify accessible intraday datasets and diffusion/agent training budget.
127|uncertain|Equity reports likely contain real finance data; full multi-agent generation/judge setup and data release need checking.
128|uncertain|Alpha discovery likely uses real stock prices; LLM MCTS budget and accessible factors/data require verification.
129|uncertain|Finance terminology corpora may qualify; confirm corpus provenance and contrastive fine-tuning base-model size.
130|candidate|Financial transaction motif networks are plausible workstation experiments; verify released authentic graph and motif preprocessing size.
131|uncertain|Real fraud graph is plausible, but LLM guidance dependency, graph sizes, and public labels need checking.
132|candidate|Risk-aware portfolio RL on daily prices is plausibly manageable; verify actual network sizes, seeds, and market data.
133|uncertain|Real financial documents may qualify; multimodal evaluator sizes and released benchmark materials need verification.
134|uncertain|Real market K-lines qualify and upstream code exists locally; training from scratch may be excessive, so inspect released checkpoint-based evaluation.
135|candidate|Offline RL trading can be feasible on a bounded real-market panel; verify algorithm size, original data, and full training budget.
136|candidate|Crypto RL with strategy archetypes appears workstation-compatible; verify genuine exchange data and published run budget.
137|uncertain|High-frequency Heston estimation may use simulation or real observations; establish real-data portion, data licensing, and sample scale.
138|uncertain|Real financial document tasks may qualify, but cross-multimodal-model reflection and judge sizes could exceed resources.
139|screen_excluded|Formula-generated financial mathematics questions are not established as authentic financial observations.
140|uncertain|Fedspeak documents are real economic data; verify labels, market links, small-model feasibility, and released predictions.
141|uncertain|FinOps concerns authentic enterprise expenses if public; production data and agent workflow release may be unavailable.
142|candidate|Workbook explicitly reports four loan-approval datasets; causal/fairness method is likely tractable subject to real-data provenance.
143|uncertain|Chronology questions could be synthetic; student abstract may include authentic financial backtests, which need checking.
144|screen_excluded|Core data/task are crop imagery and yields; insurance motivation alone does not establish real financial observations.
145|uncertain|Public and industry fraud graphs are indicated but financial identity and graph-enhanced LLM requirements need verification.
146|uncertain|A finance QA domain is included; authentic data, search reproducibility, and feasible local model size remain unknown.
147|uncertain|Stock trading is an explicit application, but multi-agent LLM prompt optimization can be costly and needs released data/outputs.
148|uncertain|Finance-domain knowledge extraction may evaluate authentic corpora or generated questions; verify data and queried-model requirements.
149|screen_excluded|Supplied notes explicitly describe theoretical auditing with credit-fraud motivation; no real financial experiment is indicated.
150|uncertain|Offline bids may be actual financial-market observations or simulated auctions; verify market domain and authentic source data.
151|candidate|Credit-card fraud GNN/RL is a likely tractable real financial classification task; verify graph release and actual experiment size.
152|candidate|Blockchain anomaly detection explicitly targets real crypto transaction graphs; verify manageable released datasets and adaptation protocol.
153|uncertain|Real startup financing outcomes likely qualify; roleplay LLM model sizes, data release, and label availability need checking.
154|uncertain|Spoofing may be studied using proprietary order data or synthetic generators; verify authenticity, public access, and graph size.
155|uncertain|Limit-order-book diffusion uses potentially genuine finance data; full data size, licensing, and training budget require verification.
156|candidate|Money-laundering representation methods plausibly fit a workstation on released transaction graphs; verify authenticity and graph scope.
157|candidate|Dynamic financial factor/portfolio modeling is empirical and plausibly moderate; verify original features, universe, and training budget.
158|uncertain|Financial reasoning model may use synthetic reasoning trajectories; verify authentic financial data and model size/checkpoint evaluation path.
159|uncertain|Local TransAlpha work exists, but workbook explicitly flags 400 proprietary intraday factors; audit faithful data access rather than substitute daily features.
160|candidate|PRISM-VQ stock ranking is empirical and local runs exist; audit completion and full published evaluation coverage.
161|screen_excluded|Survey has no single empirical method to comprehensively reproduce under the user's criteria.
162|uncertain|Fraud call conversations may be genuine or generated; verify authentic financial cases, release terms, and feasible audio/text baselines.
163|uncertain|Financial event-sequence experiments are indicated but dataset identity, authenticity, and sequence lengths require checking.
164|uncertain|Supplied notes say social/e-commerce graphs; verify whether any evaluated dataset is authentically financial before including.
165|uncertain|Financial disclosure is an authentic data source; verify released cross-jurisdiction data and manageable agent/judge models.
166|uncertain|Trading toolchain can use real prices; faithful evaluation, live dependencies, and feasible agent model sizes remain unverified.
167|candidate|Financial document retrieval is plausibly reproducible with released modest embeddings/models; verify real PDFs and visual-model sizes.
168|uncertain|Quantitative code/reasoning benchmark may use real market data or synthetic tasks; verify authentic data and feasible model evaluation.
169|uncertain|Real stock trading is indicated, but end-to-end agentic RL may exceed local resources; inspect released checkpoints and actual run budget.
170|uncertain|Financial document error detection can use authentic reports; verify corruption method, release, and feasible evaluator models.
171|screen_excluded|Supplied topic explicitly studies trading-style switching in stock-market simulation; authentic financial observations are not indicated.
172|uncertain|Business/finance knowledge prompts may be synthetic or based on real data; verify benchmark origin and feasible evaluation.
173|uncertain|Financial report generation may use authentic filings; dynamic thought-graph agents and evaluation costs need verification.
174|candidate|Financial social-media sentiment is real financial text if sourced authentically; modest text/embedding evaluation seems plausible.
175|uncertain|Behavioral bias prompts may be synthetic; establish real-financial-data use and feasible LLM evaluation.
176|candidate|Financial retrieval plausibly uses authentic documents and tractable retrievers; verify dataset release and any required generator.
177|uncertain|Financial QA reranking likely uses real reports, but analyst-card generation and model/judge dependencies require checking.
178|uncertain|Finance/accounting spreadsheet tasks may use authentic workbooks; verify released data and manageable agent evaluation.
179|uncertain|Indic financial multimodal QA may qualify; authenticity, benchmark availability, and local evaluator sizes remain unknown.
180|uncertain|Full financial-management benchmark may use simulated businesses; verify authentic financial data and model/runtime needs.
181|uncertain|Financial-document RAG likely uses authentic reports; large document collections and multimodal model requirements need checking.
182|uncertain|Financial safety scenarios may be generated prompts; authentic financial data and feasible model evaluation need confirmation.
183|screen_excluded|Supplied topic is financial tool-use dialogue synthesis; authentic financial observations are not indicated.
184|uncertain|Finance multi-agent risk evaluation may involve real prices/documents or synthetic tasks; verify data and feasible agent/model budget.
185|candidate|Numerically aware scoring can be computationally light on real financial text pairs; verify underlying benchmark authenticity and dependencies.
186|uncertain|Financial database documents suggest authentic data but large collection size and analytical-QA models need checking.
187|uncertain|Incomplete-user financial prompts may be synthetic; establish authentic data and feasible inference/judging models.
188|candidate|Earnings-surprise labels with sparse autoencoder features indicate an empirical finance task; verify available feature model and released data.
189|uncertain|Financial misinformation benchmark may contain generated claims; verify authentic financial source evidence and feasible model evaluation.
190|uncertain|SEC tables provide an authentic base, but generated inconsistency tasks and large-model evaluation requirements need verification.
191|candidate|Embedding bias/default prediction is a plausible modest real financial experiment; verify authentic loan/default labels and released embeddings.
192|uncertain|Finance QA may use real reports or generated questions; verify datasets and whether small-model or stored-output analyses suffice.
193|uncertain|Industrial KYC scans are authentic financial records, but privacy/access and OCR/model requirements may block reproduction.
194|uncertain|Heterogeneous finance datasets may be authentic; dual-granularity model size and actual released benchmarks need checking.
195|uncertain|Production credit-risk observations qualify, but proprietary refresh data and multi-agent models may be unavailable.
196|candidate|Earnings-call KPI extraction uses authentic financial text; likely feasible with bounded models and released annotations.
197|uncertain|Financial jailbreak tasks may be generated; verify authentic financial data and safe-tuning base-model size/released artifacts.
198|uncertain|Production banking data may be proprietary and a 12B model may strain current GPU/memory; inspect quantized/released evaluation artifacts.
199|uncertain|Real banking conversations qualify but may be private; verify accessible labels and taxonomy/model requirements.
200|uncertain|Prediction-market lead-lag prices are authentic finance data; LLM filtering outputs and manageable model access need checking.
201|uncertain|Numerical reasoning may use authentic reports; metric-graph construction and feasible LLM evaluation remain unknown.
202|uncertain|Counterfactual misinformation tasks may derive from real finance evidence; verify source authenticity and feasible evaluation.
203|uncertain|Chinese financial multimodal benchmark may use real documents/charts; model sizes and data release need checking.
204|uncertain|Stock factor mining likely uses genuine market data and local upstream code; code-evolution LLM requirements and run budget need verification.
205|uncertain|Debt-collection dialogues may be genuine or generated; verify authentic finance interactions and locally feasible model training.
206|uncertain|Earnings-call audio/text and surprise labels are authentic if released; large-scale audio volume and model requirements need checking.
207|screen_excluded|Supplied description is a symbolic financial reasoning benchmark; authentic financial observations are not indicated.
208|uncertain|Financial charts may come from genuine prices or synthetic plots; verify benchmark provenance and feasible vision-language models.
209|uncertain|Financial event/knowledge-graph construction may use real news; verify corpus release, graph scale, and extraction model requirements.
210|uncertain|Real-world financial deep research likely uses authentic documents, but agent models, live evidence, and judging costs need checking.
211|uncertain|Insurance-claim cases may be authentic or synthetic; verify public case data and feasible reasoning model evaluation.
212|uncertain|Insurance regulatory tables are authentic if released, but privacy and LLM requirements may block full reproduction.
213|uncertain|Credit contracts may be real financial text; verify access, document lengths, and feasible extraction/evaluation model sizes.
214|uncertain|Financial red-team prompts may be generated; authentic-data use and controllable-training model scale need confirmation.
215|uncertain|Multilingual financial multimodal tasks may use real data; verify released sources, total size, and feasible model evaluation.
216|uncertain|Fraud audio/text may be genuine finance interactions, but provenance and RL/audio model scale need verification.
217|uncertain|Arabic finance/Sharia knowledge questions may be synthetic; establish authentic financial data and feasible model evaluation.
218|uncertain|DeFi contracts can be authentic on-chain data; verify real contracts, Lean toolchain availability, and manageable agent model setup.
219|uncertain|Structured financial fraud data may be genuine or simulated; verify released provenance and feasible LLM inference/training.
220|uncertain|Production financial workflow data may be private and unified-model training could be large; verify released artifacts and authentic finance evaluation.
221|uncertain|Banking legacy code is finance-domain code rather than necessarily financial data; verify released authentic financial rules/cases and feasible models.
222|uncertain|Professional finance rubrics may be based on authentic cases or synthetic prompts; verify actual data and manageable model evaluation.
223|screen_excluded|Supplied topic explicitly describes opinion-dynamics agent simulation in economic environments; authentic financial observations are not indicated.
224|screen_excluded|Supplied topic is agentic economic modeling/experiments without an indicated authentic financial dataset.
225|screen_excluded|Economic/trade competition benchmark does not establish use of authentic financial observations.
""".strip()

classifications = {}
for line in REASONS.splitlines():
    number, triage, reason = line.split("|", 2)
    classifications[int(number)] = (triage, reason)

existing = {
    8: "alphapareto-reproduction", 9: "alphaprobe-reproduction",
    11: "hermes-reproduction", 13: "learning-augmented-ops-reproduction",
    19: "stable-alpha-reproduction", 22: "prophets-profit-reproduction",
    36: "proximal-portfolio-reproduction", 38: "sparse-tangent-reproduction",
    39: "error-propagation-reproduction", 51: "arboreal-reproduction",
    59: "expanding-chaos-reproduction", 62: "flowmsm-reproduction",
    64: "ohnemus26a-reproduction", 99: "alphasage-reproduction",
    134: "alpha-research-reproduction/upstream/kronos",
    159: "transalpha-reproduction", 160: "prism-vq-reproduction",
    204: "alpha-research-reproduction/upstream/cogalpha",
}
assert set(classifications) == set(range(1, 226))
result = []
for paper in PAPERS:
    number = paper["id"]
    if number > 225:
        continue
    triage, reason = classifications[number]
    result.append({
        "id": number, "title": paper["title"], "triage": triage,
        "reason": reason, "existing_directory": existing.get(number),
        "assessment_basis": "Workbook title/topic/notes only; provisional and not source-verified",
        "source_verified": False,
    })
(ROOT / "screen-001-225.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(dict(Counter(p["triage"] for p in result)), sort_keys=True))
