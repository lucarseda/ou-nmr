import numpy as np
import matplotlib.pyplot as plt

from ounmr.models import inversion_recovery_model, spin_echo_model

def plot_inversion_recovery_fit(tau, V, fit_result, label=None):
    """
    Plot inversion-recovery data and fitted model.

    Parameters
    ----------
    tau : array-like
        Delay times in seconds.
    V : array-like
        Measured voltages.
    fit_result : dict
        Output from fit_inversion_recovery.
    label : str, optional
        Human-readable dataset label, e.g. "Heavy mineral oil".

    Returns
    -------
    fig, ax
        Matplotlib figure and axes objects.
    """
    tau = np.asarray(tau, dtype=float)
    V = np.asarray(V, dtype=float)

    if tau.shape != V.shape:
        raise ValueError("tau and V must have the same shape")

    order = np.argsort(tau)
    tau = tau[order]
    V = V[order]

    A_fit = fit_result["params"]["A"]
    T1_fit = fit_result["params"]["T1"]
    C_fit = fit_result["params"]["C"]
    tau0_fit = fit_result["tau0"]

    tau_fit = np.linspace(np.min(tau), np.max(tau), 1000)
    V_fit = inversion_recovery_model(tau_fit, A_fit, T1_fit, C_fit)

    fig, ax = plt.subplots(figsize=(8, 5))

    data_label = f"{label} data" if label else "data"

    ax.scatter(
        tau,
        V,
        label=data_label,
        color="k",
        marker="x",
    )

    ax.plot(
        tau_fit,
        V_fit,
        color="r",
        label=(
            rf"Fit: {A_fit:.2f}$|1 - 2e^{{-\tau/{T1_fit:.4g}}}|$"
            rf" + {C_fit:.2f}"
        ),
    )

    ax.axvline(
        tau0_fit,
        linestyle="--",
        label=rf"$\tau_0 = T_1 \ln 2 = {tau0_fit:.4g}$ s",
    )

    ax.set_xlabel(r"$\tau$ (s)")
    ax.set_ylabel("Pulse B max FID amplitude (V)")
    ax.legend()
    fig.tight_layout()

    return fig, ax

def plot_spin_echo_fit(tau, A_echo, fit_result, label=None):
    """
    Plot spin-echo data and fitted exponential decay.

    Parameters
    ----------
    tau : array-like
        Pulse delay times in seconds.
    A_echo : array-like
        Measured spin-echo amplitudes.
    fit_result : dict
        Output from fit_spin_echo.
    label : str, optional
        Human-readable dataset label, e.g. "Heavy mineral oil".

    Returns
    -------
    fig, ax
        Matplotlib figure and axes objects.
    """
    tau = np.asarray(tau, dtype=float)
    A_echo = np.asarray(A_echo, dtype=float)

    if tau.shape != A_echo.shape:
        raise ValueError("tau and A_echo must have the same shape.")

    order = np.argsort(tau)
    tau = tau[order]
    A_echo = A_echo[order]

    A0_fit = fit_result["params"]["A0"]
    T2_fit = fit_result["params"]["T2"]
    C_fit = fit_result["params"]["C"]

    tau_fit = np.linspace(np.min(tau), np.max(tau), 1000)
    A_fit = spin_echo_model(tau_fit, A0_fit, T2_fit, C_fit)

    fig, ax = plt.subplots(figsize=(8, 5))

    data_label = f"{label} data" if label else "data"

    ax.scatter(
        tau,
        A_echo,
        color="black",
        marker="x",
        label=data_label,
    )

    ax.plot(
        tau_fit,
        A_fit,
        color="red",
        label=(
            rf"Fit: ${A0_fit:.3g}\exp"
            rf"\!\left(-\frac{{2\tau}}{{{T2_fit:.3g}}}\right)"
            rf"+{C_fit:.3g}$"
        ),
    )

    ax.set_xlabel(r"$\tau$ [s]")
    ax.set_ylabel("Spin echo amplitude [V]")
    ax.legend()
    fig.tight_layout()

    return fig, ax

