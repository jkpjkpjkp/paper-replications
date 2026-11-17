"""Transcribe published reference values, never experimental results.

Source: arXiv:2505.11243v2, PDF pages 8, 9, 22, 25, 26.
Tables checked against PDF text; figures checked against rendered PDF pages.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rows = []


def add(location, model, metric, value, period, unit="dimensionless", **extra):
    rows.append(dict(location=location, model=model, metric=metric,
                     published=value, period=period, unit=unit,
                     initial_state=extra.get("initial_state", ""),
                     end_state=extra.get("end_state", ""),
                     reproduced="", status="not_run"))


table4 = {
    "LongConv": [3.64, .14, 12.8, 3.51, .033, .97, .48],
    "S4": [3.94, .29, 13.5, 3.43, .028, .90, .48],
    "H3": [3.46, .28, 10.6, 3.08, .026, 1.12, .48],
    "Transformer": [3.65, .52, 12.9, 3.62, .035, .83, .47],
    "Hyena": [2.91, .33, 8.7, 3.02, .032, 1.28, .47],
    "Set-Sequence": [4.82, .12, 13.0, 2.69, .028, .91, .48],
}
for model, values in table4.items():
    for metric, value in zip(["sharpe", "sharpe_seed_std", "annual_return_pct",
                             "annual_volatility_pct", "beta", "daily_turnover",
                             "short_fraction"], values):
        add("Table 4", model, metric, value, "2002-01/2021-12",
            "percent" if metric.endswith("pct") else "dimensionless")

table5 = {
    "CNN+Trans K=5": [4.16, 8.7, 2.1], "CNN+Trans K=8": [3.95, 8.2, 2.1],
    "CNN+Trans K=10": [3.97, 8.0, 2.0], "CNN+Trans K=15": [4.17, 8.4, 2.0],
    "Fourier+FNN K=10": [1.93, 7.6, 3.9], "Fourier+FNN K=15": [2.06, 7.9, 3.8],
    "OU+Thresh K=10": [.86, 3.1, 3.6], "OU+Thresh K=15": [.93, 3.2, 3.5],
    "Set-Sequence": [5.91, 14.7, 2.5],
}
table9 = {
    "CNN+Trans K=0 (IPCA)": [.52, 8.5, 16.3],
    "CNN+Trans K=1 (IPCA)": [.85, 5.9, 6.9],
    "CNN+Trans K=3 (IPCA)": [1.24, 6.6, 5.4],
    "CNN+Trans K=5 (IPCA)": [1.11, 5.5, 5.0],
    "CNN+Trans K=10 (IPCA)": [.98, 5.1, 5.2],
    "CNN+Trans K=15 (IPCA)": [.94, 4.8, 5.1],
    "LongConv": [1.64, 7.41, 4.52], "S4": [2.17, 8.31, 3.82],
    "H3": [.76, 3.71, 4.89], "Transformer": [1.59, 7.40, 4.66],
    "Hyena": [.60, 3.59, 6.00], "Set-Sequence": [2.46, 8.4, 3.4],
}
for location, table in [("Table 5", table5), ("Table 9", table9)]:
    for model, values in table.items():
        for metric, value in zip(["sharpe", "annual_return_pct", "annual_volatility_pct"], values):
            add(location, model, metric, value, "2002-01/2016-12",
                "percent" if metric.endswith("pct") else "dimensionless")
add("Table 5", "Set-Sequence", "sharpe_seed_std", .21, "2002-01/2016-12")

table6 = {"5-Layer NN": [.205, .642], "Log. regression": [.225, .622],
          "LongConv": [.216, .669], "S4": [.226, .681], "H3": [.222, .583],
          "Transformer": [.227, .666], "Hyena": [.213, .674],
          "Set-Sequence": [.200, .683]}
for model, values in table6.items():
    for metric, value in zip(["cross_entropy", "average_transition_auc"], values):
        add("Table 6", model, metric, value, "2010-01/2022-12 (Section 7)")

ends = ["Current", "30dd", "60dd", "90dd", "F", "REO", "Paid Off"]
figure3 = {
    "F": [(.039,.033), (.0096,.014), None, (-.002,.025), (.012,.026), (.17,.041), (.036,.02)],
    "90dd": [(.0066,.022), (.004,.019), (.0088,.013), (.01,.029), (.044,.016), (.22,.02), (.026,.013)],
    "60dd": [(.019,.021), (.029,.011), (.061,.0059), (.095,.023), (-.034,.012), None, (.023,.028)],
    "30dd": [(.031,.025), (.017,.018), (.048,.026), None, None, None, (-.0046,.024)],
    "Current": [(.018,.0084), (.0002,.012), None, None, None, None, (.015,.013)],
}
figure13 = {
    "F": [(.771,.025), (.762,.068), None, (.532,.014), (.529,.021), (.550,.009), (.582,.031)],
    "90dd": [(.630,.019), (.793,.021), (.912,.008), (.588,.014), (.557,.015), (.690,.052), (.575,.023)],
    "60dd": [(.622,.013), (.670,.015), (.742,.010), (.741,.014), (.696,.025), None, (.690,.016)],
    "30dd": [(.707,.016), (.750,.012), (.669,.014), None, None, None, (.699,.012)],
    "Current": [(.613,.016), (.847,.010), None, None, None, None, (.574,.011)],
}
for location, table, name in [("Figure 3", figure3, "auc_gain"), ("Figure 13", figure13, "auc")]:
    for start, values in table.items():
        for end, pair in zip(ends, values):
            if pair is None:
                continue  # blank means unreported, never zero
            for metric, value in zip([name, name + "_subset_std"], pair):
                add(location, "Set-Sequence", metric, value, "test period ambiguous",
                    initial_state=start, end_state=end)

figure9 = {
    "F": [2.10,.40,.10,9.20,81.40,2.80,4.00],
    "90dd": [3.50,.80,1.50,81.20,9.70,.20,3.20],
    "60dd": [9.50,13.40,34.90,37.70,2.30,.01,2.30],
    "30dd": [34.10,43.10,19.00,0,.05,0,3.70],
    "Current": [96.90,.90,0,0,0,0,2.10],
}
figure10 = {
    "F": [165,17,1,1433,10743,395,552],
    "90dd": [1104,125,297,22702,1982,50,1092],
    "60dd": [596,744,3013,2312,73,0,92],
    "30dd": [4926,8466,3314,0,0,0,342],
    "Current": [955508,7361,0,0,0,0,21639],
}
for location, table, metric, period, unit in [
    ("Figure 9",figure9,"transition_probability_pct","train+validation+test","percent"),
    ("Figure 10",figure10,"transition_count","2010-01/2023-12 (caption)","count")]:
    for start, values in table.items():
        for end, value in zip(ends, values):
            add(location, "empirical data", metric, value, period, unit,
                initial_state=start, end_state=end)
add("Figure 12 / G.3.1", "Set-Sequence", "summary_foreclosure_pearson_r", .67, "unspecified")

(ROOT / "targets").mkdir(exist_ok=True)
with (ROOT / "targets/published_values.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
audit = {
    "paper_version": "2505.11243v2", "source": "https://arxiv.org/pdf/2505.11243v2",
    "reference_values": len(rows), "experimental_results": 0,
    "table_values": sum(r["location"].startswith("Table") for r in rows),
    "figure3_annotated_transitions": sum(v is not None for a in figure3.values() for v in a),
    "figure3_positive_annotated_transitions": sum(v is not None and v[0] > 0 for a in figure3.values() for v in a),
    "main_text_claim": "22 of 25 transitions improve; Figure 3 annotations show 23 of 26",
    "not_numerically_available": ["Figure 8 cumulative return series", "Figure 11 monthly series",
        "Figure 12 underlying summary and foreclosure series", "Figure 14 refitting curves",
        "Figure 15 yearly aggregate metrics", "Figure 16 yearly transition AUCs",
        "Figure 17 gating matrices"],
    "scope": "Real-data finance experiments. Synthetic contagion (Tables 1-3, Figures 2,4-7) is separate.",
}
(ROOT / "targets/inventory.json").write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps(audit, indent=2))
