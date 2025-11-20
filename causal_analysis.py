import os, json
import pandas as pd
import matplotlib.pyplot as plt
from dowhy import CausalModel

OUTPUT = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUTPUT, exist_ok=True)

def build_graph():
    return """digraph {
        X1 -> treatment;
        X2 -> treatment;
        X3 -> treatment;
        X4 -> treatment;
        X5 -> treatment;
        X1 -> outcome;
        X2 -> outcome;
        X3 -> outcome;
        X4 -> outcome;
        X5 -> outcome;
        treatment -> outcome;
    }"""

def run_analysis(data_path=None):
    if data_path is None:
        data_path = os.path.join(OUTPUT, "synthetic_data.csv")
    df = pd.read_csv(data_path)

    model = CausalModel(data=df, treatment="treatment", outcome="outcome", graph=build_graph())
    estimand = model.identify_effect()
    ipw = model.estimate_effect(estimand, method_name="backdoor.propensity_score_weighting",
                               method_params={"weighting_scheme": "ips_weight"}, target_units="ate")

    results = {"ipw_estimate": float(ipw.value)}
    df.groupby("treatment")["outcome"].mean().plot(kind="bar")
    plt.savefig(os.path.join(OUTPUT, "mean_outcome.png"))
    plt.close()

    with open(os.path.join(OUTPUT, "ipw_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results
