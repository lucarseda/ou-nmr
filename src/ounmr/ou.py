import numpy as np

def ou_params(T1, T2, omega0, gamma):
    """
    Extract tau_c, Delta^2, and sigma from T1 and T2.

    Convention:
        dB = -(1/tau_c) B dt + sigma dW

    Then:
        Var(B) = Delta^2 = sigma^2 tau_c / 2
    """
    tau_c = (1 / omega0) * np.sqrt(2 * (T1 / T2 - 1))

    Delta2 = (1 + (omega0 * tau_c)**2) / (4 * gamma**2 * tau_c * T1)

    sigma = np.sqrt(2 * Delta2 / tau_c)

    return tau_c, Delta2, sigma

def ou_hahn_echo_analytic(tau_values, T1, T2, omega0, gamma=2.675e8, T1_correction=False):
    """
    Analytic Hahn echo envelope for OU field fluctuations.

    tau_values are pulse spacings tau.
    Echo occurs at 2tau.
    """
    tau_c, Delta2, sigma = ou_params(T1, T2, omega0, gamma)

    tau = np.asarray(tau_values, dtype=float)

    I = (
        4 * tau_c * tau
        - 6 * tau_c**2
        + 8 * tau_c**2 * np.exp(-tau / tau_c)
        - 2 * tau_c**2 * np.exp(-2 * tau / tau_c)
    )

    if T1_correction:
        S = np.exp(-0.5 * gamma**2 * Delta2 * I) * np.exp(-tau / T1)
    else:
        S = np.exp(-0.5 * gamma**2 * Delta2 * I)

    params = {
        "tau_c": tau_c,
        "Delta2": Delta2,
        "Delta": np.sqrt(Delta2),
        "sigma": sigma,
    }

    return S, params