from pathlib import Path

import pandas as pd
import numpy as np

from ounmr.constants import GAMMA_PROTON, F_REC_LIGHT_OIL, F_REC_HEAVY_OIL
from ounmr.ou import ou_params

ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "results" / "tables"

RELAXATION_FITS_PATH = TABLE_DIR / "relaxation_fits.csv"
OUTPUT_PATH = TABLE_DIR / "ou_parameters.csv"

FREQUENCIES_HZ = {
    "light_oil": F_REC_LIGHT_OIL,
    "heavy_oil": F_REC_HEAVY_OIL
}

def get_relaxation_time(df, sample, experiment):
    row = df[(df["sample"] == sample) & (df["experiment"] == experiment)]

    if len(row) != 1:
        raise ValueError(f"Expected exactly one row for sample {sample} and experiment {experiment}, but got {len(row)}")
    
    row = row.iloc[0]

    if experiment == "T1":
        return float(row["T1_s"]), float(row["T1_err_s"])
    elif experiment == "T2":
        return float(row["T2_s"]), float(row["T2_err_s"])
    else:
        raise ValueError(f"Unknown experiment type: {experiment}")
    
def main():
    df = pd.read_csv(RELAXATION_FITS_PATH)

    rows = []

    for sample, f_rec in FREQUENCIES_HZ.items():
        T1, T1_err = get_relaxation_time(df, sample, "T1")
        T2, T2_err = get_relaxation_time(df, sample, "T2")

        omega0 = 2 * np.pi * f_rec

        tau_c, Delta2, sigma = ou_params(
            T1=T1, 
            T2=T2, 
            omega0=omega0, 
            gamma=GAMMA_PROTON
        )

        rows.append(
            {"sample": sample,
             "f_rec_Hz": f_rec,
             "omega0_rad_s": omega0,
             "T1_s": T1,
             "T1_err_s": T1_err,
             "T2_s": T2,
             "T2_err_s": T2_err,
             "tau_c_s": tau_c,
             "Delta2_T2": Delta2,
             "Delta_T": np.sqrt(Delta2),
             "Delta_G": np.sqrt(Delta2) * 1e4,
             "sigma_T_s_minus_3_half": sigma
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_PATH, index=False)

    print(out)
    print(f"Saved OU parameters to {OUTPUT_PATH}")

if __name__ == "__main__":
    main() 