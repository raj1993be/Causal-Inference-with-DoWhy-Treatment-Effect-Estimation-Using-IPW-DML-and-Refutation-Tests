from src.data_generation import generate_synthetic_data
from src.causal_analysis import run_analysis
from src.dml_analysis import run_dml
from src.refutation_tests import run_refuters

if __name__ == "__main__":
    generate_synthetic_data()
    print(run_analysis())
    print(run_dml())
    print(run_refuters())
