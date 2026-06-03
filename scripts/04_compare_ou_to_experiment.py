from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from ounmr.constants import GAMMA_PROTON
from ounmr.data import load_relaxation_csv
from ounmr.ou import ou_hahn_echo_analytic

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"

RELAXATION_FITS_PATH = TABLE_DIR / "relaxation_fits.csv"
OU_PARAMETERS_PATH = TABLE_DIR / "ou_parameters.csv"
OUTPUT_PATH = FIGURE_DIR / "OU_vs_exp.png"

def get_fit_value(df, sample, experiment, column):
    row = df[(df["sample"] == sample) & (df["experiment"] == experiment)]

    if len(row) != 1:
        raise ValueError(f"Expected one row for {sample}, {experiment}")

    return float(row.iloc[0][column])


def prepare_echo_data(path, offset):
    tau, voltage = load_relaxation_csv(path)
    echo = voltage - offset
    echo_norm = echo / np.max(echo)
    return tau, echo_norm


def main():
    fits = pd.read_csv(RELAXATION_FITS_PATH)
    ou_params_table = pd.read_csv(OU_PARAMETERS_PATH).set_index("sample")

    T1_light = get_fit_value(fits, "light_oil", "T1", "T1_s")
    T2_light = get_fit_value(fits, "light_oil", "T2", "T2_s")
    C_light = get_fit_value(fits, "light_oil", "T2", "C")

    T1_heavy = get_fit_value(fits, "heavy_oil", "T1", "T1_s")
    T2_heavy = get_fit_value(fits, "heavy_oil", "T2", "T2_s")
    C_heavy = get_fit_value(fits, "heavy_oil", "T2", "C")

    omega0_light = float(ou_params_table.loc["light_oil", "omega0_rad_s"])
    omega0_heavy = float(ou_params_table.loc["heavy_oil", "omega0_rad_s"])

    tau_exp_light, echo_exp_norm_light = prepare_echo_data(
        DATA_DIR / "light_oil_T2.csv",
        offset=C_light,
    )

    tau_exp_heavy, echo_exp_norm_heavy = prepare_echo_data(
        DATA_DIR / "heavy_oil_T2.csv",
        offset=C_heavy,
    )

    tau_grid_light = np.linspace(np.min(tau_exp_light), np.max(tau_exp_light), 400)
    tau_grid_heavy = np.linspace(np.min(tau_exp_heavy), np.max(tau_exp_heavy), 400)

    echo_pred_light, _ = ou_hahn_echo_analytic(
        tau_grid_light,
        T1=T1_light,
        T2=T2_light,
        omega0=omega0_light,
        gamma=GAMMA_PROTON,
        T1_correction=False,
    )

    echo_pred_heavy, _ = ou_hahn_echo_analytic(
        tau_grid_heavy,
        T1=T1_heavy,
        T2=T2_heavy,
        omega0=omega0_heavy,
        gamma=GAMMA_PROTON,
        T1_correction=False,
    )

    echo_pred_light_corr, _ = ou_hahn_echo_analytic(
        tau_grid_light,
        T1=T1_light,
        T2=T2_light,
        omega0=omega0_light,
        gamma=GAMMA_PROTON,
        T1_correction=True,
    )

    echo_pred_heavy_corr, _ = ou_hahn_echo_analytic(
        tau_grid_heavy,
        T1=T1_heavy,
        T2=T2_heavy,
        omega0=omega0_heavy,
        gamma=GAMMA_PROTON,
        T1_correction=True,
    )

    echo_pred_norm_light = echo_pred_light / np.max(echo_pred_light)
    echo_pred_norm_heavy = echo_pred_heavy / np.max(echo_pred_heavy)

    echo_pred_norm_light_corr = echo_pred_light_corr / np.max(echo_pred_light_corr)
    echo_pred_norm_heavy_corr = echo_pred_heavy_corr / np.max(echo_pred_heavy_corr)

    fig, ax = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5]},
    )

    ax[0].scatter(
        2 * tau_exp_light * 1e3,
        echo_exp_norm_light,
        marker="x",
        color="k",
        label="Light oil experiment",
    )

    ax[0].plot(
        2 * tau_grid_light * 1e3,
        echo_pred_norm_light,
        linestyle="-",
        label="Light oil OU pure dephasing",
        alpha=0.7,
    )

    ax[0].plot(
        2 * tau_grid_light * 1e3,
        echo_pred_norm_light_corr,
        linestyle="--",
        label="Light oil OU relaxation-corrected",
        alpha=0.7,
    )

    ax[0].scatter(
        2 * tau_exp_heavy * 1e3,
        echo_exp_norm_heavy,
        marker="^",
        color="k",
        facecolors="none",
        label="Heavy oil experiment",
    )

    ax[0].plot(
        2 * tau_grid_heavy * 1e3,
        echo_pred_norm_heavy,
        linestyle="-",
        label="Heavy oil OU pure dephasing",
        alpha=0.7,
    )

    ax[0].plot(
        2 * tau_grid_heavy * 1e3,
        echo_pred_norm_heavy_corr,
        linestyle="--",
        label="Heavy oil OU relaxation-corrected",
        alpha=0.7,
    )

    ax[0].set_ylabel("Normalized echo amplitude")
    ax[0].legend(loc="upper right")

    f_light = interp1d(
        2 * tau_grid_light * 1e3,
        echo_pred_norm_light,
        bounds_error=False,
        fill_value="extrapolate",
    )

    f_heavy = interp1d(
        2 * tau_grid_heavy * 1e3,
        echo_pred_norm_heavy,
        bounds_error=False,
        fill_value="extrapolate",
    )

    x_light = 2 * tau_exp_light * 1e3
    x_heavy = 2 * tau_exp_heavy * 1e3

    res_light = echo_exp_norm_light - f_light(x_light)
    res_heavy = echo_exp_norm_heavy - f_heavy(x_heavy)

    ax[1].scatter(
        x_light,
        res_light,
        marker="x",
        label="Light oil residuals",
    )

    ax[1].scatter(
        x_heavy,
        res_heavy,
        facecolors="none",
        marker="^",
        label="Heavy oil residuals",
    )

    ax[1].axhline(0, linestyle="--", linewidth=1)
    ax[1].set_xlabel(r"$2\tau\;(\mathrm{ms})$")
    ax[1].set_ylabel("Residual")
    ax[1].legend(loc="upper right")

    plt.tight_layout()

    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()