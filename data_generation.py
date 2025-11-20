import os, numpy as np, pandas as pd
OUTPUT = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUTPUT, exist_ok=True)

def generate_synthetic_data(n=2000, seed=42, true_ate=2.0):
    np.random.seed(seed)
    X1 = np.random.normal(0,1,n)
    X2 = np.random.normal(0,1,n)
    X3 = np.random.binomial(1,0.4,n)
    X4 = np.random.normal(2,1.5,n)
    X5 = np.random.uniform(-1,1,n)
    U = np.random.normal(0,1,n)
    logits = 0.5*X1 -0.25*X2 +0.7*X3 +0.1*X4 +0.5*U -0.2*X5
    prob = 1/(1+np.exp(-logits))
    treatment = np.random.binomial(1, prob, n)
    noise = np.random.normal(0,1,n)
    y = true_ate*treatment + 1.2*X1 -0.8*X2 +0.5*X3 +0.3*X4 -0.4*X5 +0.8*U +0.5*noise
    df = pd.DataFrame({"treatment":treatment,"outcome":y,"X1":X1,"X2":X2,"X3":X3,"X4":X4,"X5":X5})
    path = os.path.join(OUTPUT, "synthetic_data.csv")
    df.to_csv(path, index=False)
    return df
