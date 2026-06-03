from pathlib import Path
import pandas as pd

from ounmr.data import load_relaxation_csv
from ounmr.fitting import fit_inversion_recovery, fit_spin_echo
from ounmr.plotting import plot_inversion_recovery_fit, plot_spin_echo_fit

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    {
        "sample": "heavy_oil",
        "label": "Heavy mineral oil",
        "kind": "T1",
        "path": DATA_DIR / "heavy_oil_T1.csv",
    },
    {
        "sample": "heavy_oil",
        "label": "Heavy mineral oil",
        "kind": "T2",
        "path": DATA_DIR / "heavy_oil_T2.csv",
    },
    {
        "sample": "light_oil",
        "label": "Light mineral oil",
        "kind": "T1",
        "path": DATA_DIR / "light_oil_T1.csv",
    },
    {
        "sample": "light_oil",
        "label": "Light mineral oil",
        "kind": "T2",
        "path": DATA_DIR / "light_oil_T2.csv",
    },
]

def main():
    rows = []

    for dataset in DATASETS:
        tau, voltage = load_relaxation_csv(dataset["path"])

        if dataset["kind"] == "T1":
            fit = fit_inversion_recovery(tau=tau, V=voltage)

            fig, ax = plot_inversion_recovery_fit(tau=tau, V=voltage, fit_result=fit, label=dataset["label"])

            figure_path = FIGURE_DIR / f"{dataset['sample']}_T1_fit.png"

            rows.append(
                {
                    "sample": dataset["sample"],
                    "experiment": "T1",
                    "A": fit["params"]["A"],
                    "A_err": fit["errors"]["A"],
                    "T1_s": fit["params"]["T1"],
                    "T1_err_s": fit["errors"]["T1"],
                    "C": fit["params"]["C"],
                    "C_err": fit["errors"]["C"],
                    "r_squared": fit["r_squared"]
                }
            )

        elif dataset["kind"] == "T2":
            fit = fit_spin_echo(tau=tau, A_echo=voltage)

            fig, ax = plot_spin_echo_fit(tau=tau, A_echo=voltage, fit_result=fit, label=dataset["label"])

            figure_path = FIGURE_DIR / f"{dataset['sample']}_T2_fit.png"

            rows.append(
                {
                    "sample": dataset["sample"],
                    "experiment": "T2",
                    "A": fit["params"]["A0"],
                    "A_err": fit["errors"]["A0"],
                    "T2_s": fit["params"]["T2"],
                    "T2_err_s": fit["errors"]["T2"],
                    "C": fit["params"]["C"],
                    "C_err": fit["errors"]["C"],
                    "r_squared": fit["r_squared"]
                }
            )
            
        else:
            raise ValueError(f"Unknown experiment kind: {dataset['kind']}")
        
        fig.savefig(figure_path, dpi=300, bbox_inches="tight")
        print("Saved figure to:", figure_path)

    df = pd.DataFrame(rows)
    output_path = TABLE_DIR / "relaxation_fits.csv"
    df.to_csv(output_path, index=False)

    print("Saved fit results to:", output_path)

if __name__ == "__main__":
    main()