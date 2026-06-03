import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = [
    "01_fit_relaxation.py",
    "02_infer_ou_parameters.py",
    "03_simulate_ou_fields.py",
    "04_compare_ou_to_experiment.py",
    "05_phase_distribution.py",
]


def main():
    for script in SCRIPTS:
        script_path = ROOT / "scripts" / script
        print(f"\nRunning {script_path.name}...")
        subprocess.run([sys.executable, str(script_path)], check=True)

    print("\nPipeline completed.")


if __name__ == "__main__":
    main()