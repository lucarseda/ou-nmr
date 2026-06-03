from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ounmr.constants import GAMMA_PROTON
from ounmr.simulation import (
    simulate_ou_field,
    hahn_toggling_function,
    stochastic_phase,
    ensemble_average,
)


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"

OU_PARAMETERS_PATH = TABLE_DIR / "ou_parameters.csv"
OUTPUT_PATH = FIGURE_DIR / "phase_distribution.png"


def main():
    params = pd.read_csv(OU_PARAMETERS_PATH).set_index("sample")

    tau_c_light = float(params.loc["light_oil", "tau_c_s"])
    sigma_light = float(params.loc["light_oil", "sigma_T_s_minus_3_half"])

    tau_c_heavy = float(params.loc["heavy_oil", "tau_c_s"])
    sigma_heavy = float(params.loc["heavy_oil", "sigma_T_s_minus_3_half"])

    tau_c_min = min(tau_c_light, tau_c_heavy)
    tau_c_max = max(tau_c_light, tau_c_heavy)

    dt = tau_c_min / 20
    t_max = 10 * tau_c_max
    t = np.arange(0, t_max, dt)

    rng = np.random.default_rng(493)

    B_light = simulate_ou_field(
        n_traj=2000,
        t=t,
        tau_c=tau_c_light,
        sigma=sigma_light,
        rng=rng,
    )

    B_heavy = simulate_ou_field(
        n_traj=2000,
        t=t,
        tau_c=tau_c_heavy,
        sigma=sigma_heavy,
        rng=rng,
    )

    tau = 1.5 * tau_c_max
    n_show = 25

    t_cut = 2 * tau * 1.1
    mask = t <= t_cut

    y = hahn_toggling_function(t, tau)

    phi_light = stochastic_phase(B_light, t, GAMMA_PROTON, y=y)
    phi_heavy = stochastic_phase(B_heavy, t, GAMMA_PROTON, y=y)

    S_light = ensemble_average(phi_light)
    S_heavy = ensemble_average(phi_heavy)

    echo_idx = np.argmin(np.abs(t - 2 * tau))

    phi_final_light = phi_light[:, echo_idx]
    phi_final_heavy = phi_heavy[:, echo_idx]

    echo_amp_light = np.abs(S_light[echo_idx])
    echo_amp_heavy = np.abs(S_heavy[echo_idx])

    fig, ax = plt.subplots(2, 1, figsize=(10, 6))

    for k in range(n_show):
        ax[0].plot(t[mask] * 1e9, phi_light[k, mask], alpha=0.35, linewidth=1)

    ax[0].axvline(
        tau * 1e9,
        linestyle="--",
        color="k",
        linewidth=1,
        label=r"$\pi$ pulse",
    )

    ax[0].axvline(
        2 * tau * 1e9,
        linestyle=":",
        color="k",
        linewidth=1,
        label=r"echo time $2\tau$",
    )

    ax[0].set_xlabel(r"$t\;(\mathrm{ns})$")
    ax[0].set_ylabel(r"$\phi_k(t)$")
    ax[0].set_title(r"Light oil phase trajectories")
    ax[0].legend(frameon=False, loc="upper left")

    ax[1].hist(
        phi_final_light,
        bins=50,
        density=True,
        alpha=0.55,
        label=rf"Light oil, $|S|={echo_amp_light:.7f}$",
    )

    ax[1].hist(
        phi_final_heavy,
        bins=50,
        density=True,
        alpha=0.55,
        label=rf"Heavy oil, $|S|={echo_amp_heavy:.7f}$",
    )

    ax[1].set_xlabel(r"$\phi_k(2\tau)$")
    ax[1].set_ylabel("Density at echo")
    ax[1].set_title("Phase distributions at the echo time")
    ax[1].legend(frameon=False)

    plt.subplots_adjust(hspace=0.7)

    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()