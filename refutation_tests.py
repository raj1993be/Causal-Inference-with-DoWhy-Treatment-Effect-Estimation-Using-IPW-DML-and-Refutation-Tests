import os, json
import pandas as pd
from dowhy import CausalModel

OUTPUT = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUTPUT, exist_ok=True)

def run_refuters(data_path=None):
    if data_path is None:
        data_path = os.path.join(OUTPUT, "synthetic_data.csv")

    df = pd.read_csv(data_path)
    graph = """digraph {
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

    model = CausalModel(data=df, treatment="treatment", outcome="outcome", graph=graph)
    est = model.identify_effect()
    ipw = model.estimate_effect(est, method_name="backdoor.propensity_score_weighting",
                                method_params={"weighting_scheme": "ips_weight"})

    r1 = model.refute_estimate(est, ipw, method_name="add_unobserved_common_cause")
    r2 = model.refute_estimate(est, ipw, method_name="placebo_treatment_refuter")
    r3 = model.refute_estimate(est, ipw, method_name="random_common_cause")

    res = {
        "unobserved_common_cause": r1.new_effect,
        "placebo": r2.new_effect,
        "random_common_cause": r3.new_effect
    }

    with open(os.path.join(OUTPUT, "refuter_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res
