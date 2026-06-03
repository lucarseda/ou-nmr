import numpy as np
from scipy.optimize import curve_fit

from ounmr.models import inversion_recovery_model, spin_echo_model

def fit_inversion_recovery(tau, V):
    """
    Fit inversion-recovery data to

        S(tau) = A * |1 - 2 exp(-tau / T1)| + C

    Parameters
    ----------
    tau : array-like
        Delay times in seconds.
    V : array-like
        Measured voltages.

    Returns
    -------
    result : dict
        Dictionary containing best-fit parameters, uncertainties,
        covariance matrix, sorted data, residuals, and model information.
    """
    tau = np.asarray(tau, dtype=float)
    V = np.asarray(V, dtype=float)

    if tau.shape != V.shape:
        raise ValueError("tau and V must have the same shape")
    if tau.ndim != 1:
        raise ValueError("tau and V must be 1D arrays")
    if np.any(tau <= 0):
        raise ValueError("All tau values must be positive")

    order = np.argsort(tau)
    tau = tau[order]
    V = V[order]

    # Initial guesses
    C0 = np.min(V)
    A0 = np.max(V) - C0

    # The inversion-recovery cusp occurs near tau = T1 ln(2)
    tau0_guess = tau[np.argmin(V)]
    T10 = tau0_guess / np.log(2) if tau0_guess > 0 else np.median(tau)

    p0 = [A0, T10, C0]
    bounds = ([0, 0, -np.inf], [np.inf, np.inf, np.inf])

    popt, pcov = curve_fit(
        inversion_recovery_model,
        tau,
        V,
        p0=p0,
        bounds=bounds,
        maxfev=10000,
    )

    perr = np.sqrt(np.diag(pcov))

    A_fit, T1_fit, C_fit = popt
    V_pred = inversion_recovery_model(tau, *popt)
    residuals = V - V_pred

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((V - np.mean(V))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    tau0_fit = T1_fit * np.log(2)

    result = {
        "model_name": "inversion_recovery",
        "popt": popt,
        "pcov": pcov,
        "perr": perr,
        "params": {
            "A": A_fit,
            "T1": T1_fit,
            "C": C_fit,
        },
        "errors": {
            "A": perr[0],
            "T1": perr[1],
            "C": perr[2],
        },
        "tau_data": tau,
        "V_data": V,
        "V_pred": V_pred,
        "residuals": residuals,
        "r_squared": r_squared,
        "tau0": tau0_fit,
        "model": inversion_recovery_model,
    }

    return result

def fit_spin_echo(tau, A_echo):
    """
    Fit spin-echo amplitudes to

        A_echo(tau) = A0 exp(-2 tau / T2) + C

    Parameters
    ----------
    tau : array-like
        Pulse delay times in seconds.
    A_echo : array-like
        Measured spin-echo amplitudes.

    Returns
    -------
    result : dict
        Dictionary containing best-fit parameters, uncertainties,
        covariance matrix, sorted data, residuals, and model information.
    """
    tau = np.asarray(tau, dtype=float)
    A_echo = np.asarray(A_echo, dtype=float)

    if tau.shape != A_echo.shape:
        raise ValueError("tau and A_echo must have the same shape.")
    if tau.ndim != 1:
        raise ValueError("tau and A_echo must be 1D arrays.")
    if len(tau) < 3:
        raise ValueError("Need at least 3 data points.")
    if np.any(tau < 0):
        raise ValueError("tau values must be nonnegative.")

    order = np.argsort(tau)
    tau = tau[order]
    A_echo = A_echo[order]

    # Initial guesses
    C0 = np.min(A_echo)
    A0_guess = np.max(A_echo) - C0

    # Since A - C = A0 exp(-2 tau / T2),
    # the 1/e point occurs near tau = T2 / 2.
    target = C0 + A0_guess / np.e
    idx = np.argmin(np.abs(A_echo - target))
    T2_guess = 2 * tau[idx] if tau[idx] > 0 else max(np.median(tau), 1e-3)

    p0 = [A0_guess, T2_guess, C0]
    bounds = ([0, 0, -np.inf], [np.inf, np.inf, np.inf])

    popt, pcov = curve_fit(
        spin_echo_model,
        tau,
        A_echo,
        p0=p0,
        bounds=bounds,
        maxfev=10000,
    )

    perr = np.sqrt(np.diag(pcov))

    A0_fit, T2_fit, C_fit = popt
    A_pred = spin_echo_model(tau, *popt)
    residuals = A_echo - A_pred

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((A_echo - np.mean(A_echo))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    result = {
        "model_name": "spin_echo",
        "popt": popt,
        "pcov": pcov,
        "perr": perr,
        "params": {
            "A0": A0_fit,
            "T2": T2_fit,
            "C": C_fit,
        },
        "errors": {
            "A0": perr[0],
            "T2": perr[1],
            "C": perr[2],
        },
        "tau_data": tau,
        "A_data": A_echo,
        "A_pred": A_pred,
        "residuals": residuals,
        "r_squared": r_squared,
        "model": spin_echo_model,
    }

    return result
