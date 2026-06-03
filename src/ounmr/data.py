from pathlib import Path

import numpy as np
import pandas as pd

def load_relaxation_csv(path):
    """Load a CSV file containing relaxation data.

    The CSV file should have two columns: "time" and "signal".

    Parameters
    ----------
    path : str or Path
        The path to the CSV file.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the relaxation data.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    df = pd.read_csv(path)

    required_columns = {"tau_s", "voltage_V"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{path} is missing required columns: {missing}. "
            f"Expected columns: {required_columns}"
        )

    tau = df["tau_s"].to_numpy(dtype=float)
    voltage = df["voltage_V"].to_numpy(dtype=float)

    if len(tau) != len(voltage):
        raise ValueError(f"{path} has mismatched lengths: {len(tau)} tau values, "
                         f"but {len(voltage)} voltage values.")
    
    if len(tau) == 0:
        raise ValueError(f"{path} contains no data.")
    
    if np.any(~np.isfinite(tau)) or np.any(~np.isfinite(voltage)):
        raise ValueError(f"{path} contains NaN or infinite values.")
    
    return tau, voltage
    