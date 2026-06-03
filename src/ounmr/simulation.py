import numpy as np

def simulate_ou_field(n_traj, t, tau_c, sigma, rng=None):
    """
    Simulate OU field fluctuations using:

        dB = -(1/tau_c) B dt + sigma dW

    Parameters
    ----------
    sigma : float
        Noise strength in the SDE, not the RMS field amplitude.
    """
    if rng is None:
        rng = np.random.default_rng()

    t = np.asarray(t)
    dt = t[1] - t[0]
    n_t = len(t)

    B = np.zeros((n_traj, n_t))

    # Stationary variance for this SDE convention
    Delta2 = sigma**2 * tau_c / 2
    Delta = np.sqrt(Delta2)

    B[:, 0] = rng.normal(0.0, Delta, size=n_traj)

    # Exact OU update
    a = np.exp(-dt / tau_c)
    noise_std = Delta * np.sqrt(1 - a**2)

    for k in range(1, n_t):
        B[:, k] = a * B[:, k - 1] + noise_std * rng.normal(size=n_traj)

    return B

def autocorr_ensemble(B):
    B = B - B.mean(axis=1, keepdims=True)
    C = np.mean(
        [np.correlate(b, b, mode="full")[len(b)-1:] for b in B],
        axis=0
    )
    return C / C[0]

def hahn_toggling_function(t, tau):
    """
    Hahn echo toggling function:
        +1 before the pi pulse
        -1 after the pi pulse
    """
    y = np.ones_like(t)
    y[t >= tau] = -1
    return y

def stochastic_phase(B, t, gamma, y=None):
    """
    Compute phase accumulation:

        phi(t) = gamma integral y(t') B(t') dt'

    For FID, use y = 1.
    For Hahn echo, use y = +1 before pi pulse and -1 after.
    """
    t = np.asarray(t)
    dt = t[1] - t[0]

    if y is None:
        y = np.ones_like(t)

    phi = gamma * np.cumsum(B * y[None, :], axis=1) * dt

    return phi

def ensemble_average(phi):
    """
    Compute the ensemble-averaged transverse coherence:

        S(t) = < exp(i phi(t)) >
    """
    return np.mean(np.exp(1j * phi), axis=0)