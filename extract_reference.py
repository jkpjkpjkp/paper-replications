"""Extract the authors' saved numerical outputs, never used by the optimizer."""
from hashlib import sha256
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
COMMIT = "448ec4897ff588c8bed318afa13b8858d7e383fa"


def main():
    path = ROOT / "upstream/notebooks/portfolio.ipynb"
    notebook = json.loads(path.read_text())
    methods = {"kl_bdro": "BDRO", "kl_dro_bas": "BASPE", "kl_pp": "BASPP"}
    aggregate, cumulative = [], []
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if source.startswith("agg_df = "):
            for output in cell.get("outputs", []):
                content = "".join(output.get("data", {}).get("text/plain", []))
                first_block = content.split("sum_of_in_group_var")[0]
                method = None
                for line in first_block.splitlines():
                    match = re.match(r"\s*(?:(kl_bdro|kl_dro_bas|kl_pp)\s+)?"
                                     r"([0-9.]+)\s+bayes\s+([0-9.]+)\s+([0-9.]+)", line)
                    if match:
                        method = match[1] or method
                        aggregate.append({"method": methods[method], "radius_above_offset": float(match[2]),
                                          "mean_weekly_return": float(match[3]),
                                          "variance_weekly_return": float(match[4])})
        if "print(cum_return_df.unstack" in source:
            for output in cell.get("outputs", []):
                content = "".join(output.get("text", []))
                for match in re.finditer(r"([0-9.]+) & ([0-9.]+) & ([0-9.]+) & ([0-9.]+)", content):
                    for method, val in zip(("BDRO", "BASPE", "BASPP"), match.groups()[1:]):
                        cumulative.append({"method": method, "radius_above_offset": float(match[1]),
                                           "sum_weekly_returns": float(val)})
    assert len(aggregate) == 48 and len(cumulative) == 9
    reference = {"source": f"https://github.com/PatrickOHara/mis-dro-code/blob/{COMMIT}/notebooks/portfolio.ipynb",
                 "upstream_commit": COMMIT, "notebook_sha256": sha256(path.read_bytes()).hexdigest(),
                 "precision": "Saved notebook outputs rounded to six decimal places",
                 "aggregate": aggregate, "cumulative": cumulative}
    (ROOT / "data/authors_reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    print("Extracted 48 mean/variance pairs and 9 cumulative-return endpoints")


if __name__ == "__main__":
    main()
