from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ounmr.simulation import simulate_ou_field, autocorr_ensemble

ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"

OU_PARAMETERS_PATH = TABLE_DIR / "ou_parameters.csv"
OUTPUT_PATH = FIGURE_DIR / "OU_field_autocorr.png"

def main():
    params = pd.read_csv(OU_PARAMETERS_PATH).set_index("sample")

    tau_c_light = float(params.loc["light_oil", "tau_c_s"])
    sigma_light = float(params.loc["light_oil", "sigma_T_s_minus_3_half"])

    tau_c_heavy = float(params.loc["heavy_oil", "tau_c_s"])
    sigma_heavy = float(params.loc["heavy_oil", "sigma_T_s_minus_3_half"])

    tau_c_min = min(tau_c_light, tau_c_heavy)
    tau_c_max = max(tau_c_light, tau_c_heavy)

    dt = tau_c_min / 10
    t_max = 10 * tau_c_max
    t = np.arange(0, t_max, dt)

    rng = np.random.default_rng(seed=42)

    B_light_all = simulate_ou_field(
        n_traj=2000,
        t=t,
        tau_c=tau_c_light,
        sigma=sigma_light,
        rng=rng
    )

    B_heavy_all = simulate_ou_field(
        n_traj=2000,
        t=t,
        tau_c=tau_c_heavy,
        sigma=sigma_heavy,
        rng=rng
    )

    B_light = B_light_all[0]
    B_heavy = B_heavy_all[0]

    C_light = autocorr_ensemble(B_light_all)
    C_heavy = autocorr_ensemble(B_heavy_all)

    t_lag_light = t[: len(C_light)]
    t_lag_heavy = t[: len(C_heavy)]

    mask_light = t_lag_light < 4 * tau_c_light
    mask_heavy = t_lag_heavy < 4 * tau_c_heavy

    fig, ax = plt.subplots(2, 1, figsize=(10, 5))

    ax[0].plot(t * 1e9, B_light * 1e4, label="Light oil", alpha=0.8)
    ax[0].plot(t * 1e9, B_heavy * 1e4, label="Heavy oil", alpha=0.8)
    ax[0].set_xlabel(r"$t\;(\mathrm{ns})$")
    ax[0].set_ylabel(r"$\mathfrak{B}_z(t)\;(\mathrm{G})$")
    ax[0].legend(frameon=False, loc="lower left")

    ax[1].plot(
        t_lag_light[mask_light] * 1e9,
        C_light[mask_light],
        label="Light oil",
        alpha=0.8,
    )
    ax[1].plot(
        t_lag_heavy[mask_heavy] * 1e9,
        C_heavy[mask_heavy],
        label="Heavy oil",
        alpha=0.8,
    )
    ax[1].set_xlabel(r"$t_{\mathrm{lag}}\;(\mathrm{ns})$")
    ax[1].set_ylabel(r"$C(t_{\mathrm{lag}})$")
    ax[1].legend(frameon=False, loc="upper right")

    plt.subplots_adjust(hspace=0.5)

    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved {OUTPUT_PATH}")

if __name__ == "__main__":
    main()