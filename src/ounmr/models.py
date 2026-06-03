import numpy as np

def inversion_recovery_model(tau, A, T1, C):
    return A * np.abs(1 - 2 * np.exp(-tau / T1)) + C

def spin_echo_model(tau, A, T2, C):
    return A * np.exp(-2 * tau / T2) + C