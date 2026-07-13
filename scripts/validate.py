"""
scripts/validate.py
Python-side validation of a trained model against Thermopack Python.

NOTE: This script requires Thermopack Python (pip install thermopack).
For the current ph2_qcc pipeline, Thermopack 3.0 is only available as
a compiled Fortran library. Use the Fortran test driver instead:

    cd nn_fortran/predict
    make clean && make && make run
    python plot_validation.py

This script is kept for future use when:
  - A new EOS is added without a Fortran test driver
  - Thermopack Python is available for comparison
  - Quick Python-side MAPE check is needed during development

Usage (when Thermopack Python is available):
    python scripts/validate.py --property rho --phase liquid
    python scripts/validate.py --property cs2 --phase vapor
"""

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.helpers import load_model, load_scaling_params, get_project_root
import numpy as np
import yaml


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--property', required=True)
    parser.add_argument('--phase',    required=True, choices=['liquid', 'vapor'])
    args = parser.parse_args()

    print("\nNOTE: validate.py requires Thermopack Python.")
    print("For ph2_qcc, use Fortran test driver instead:")
    print("  cd nn_fortran/predict && make run")
    print("\nTo implement Python validation for a new EOS, edit this file.")
    print("The model and scaling params are loaded below as a starting point.\n")

    project_root = get_project_root()
    config_key   = f"{args.property}_{args.phase}"

    with open(project_root / 'config_train.yaml') as f:
        cfg = yaml.safe_load(f)[config_key]

    model          = load_model(config_key, project_root)
    scaling_params = load_scaling_params(cfg['eos'], cfg['phase'], project_root)

    print(f"Model loaded    : models/{config_key}/")
    print(f"Scaling loaded  : data/scaled/{cfg['phase']}_{cfg['eos']}_scaling.csv")
    print(f"\nAdd Thermopack Python calls here to compute reference values.")
    print("Then compare against model predictions and compute MAPE.")
