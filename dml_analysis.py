import os, json
import pandas as pd
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor

OUTPUT = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUTPUT, exist_ok=True)

def run_dml(data_path=None):
    if data_path is None:
        data_path = os.path.join(OUTPUT, "synthetic_data.csv")

    df = pd.read_csv(data_path)
    X = df[["X1","X2","X3","X4","X5"]]
    T = df["treatment"]
    Y = df["outcome"]

    model = LinearDML(model_y=RandomForestRegressor(),
                      model_t=RandomForestRegressor(),
                      discrete_treatment=True)
    model.fit(Y, T, X=X)
    ate = model.ate(X)
    ci = model.ate_interval(X)

    results = {"dml_ate": float(ate),
               "ci": [float(ci[0]), float(ci[1])]}

    with open(os.path.join(OUTPUT, "dml_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results
