"""
src/scaling.py
Scale the raw dataset CSV for a given EOS and phase.
Reads config_scale.yaml for scaling methods per column.
Writes scaled CSV and scaling params CSV to data/scaled/.

Usage:
    python src/scaling.py --eos ph2_qcc --phase liquid
    python src/scaling.py --eos ph2_qcc --phase vapor

Outputs:
    data/scaled/{phase}_{eos}_scaled.csv     - all columns scaled
    data/scaled/{phase}_{eos}_scaling.csv    - scaling params (col_name, method, p1, p2)

Tunable:
    Edit config_scale.yaml to change scaling method per column.
    Available methods: minmax, standard, maxval
"""

import sys
import argparse
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))


def compute_scale_params(series, method):
    """
    Compute scaling parameters for a column.

    Args:
        series : pandas Series
        method : 'minmax', 'standard', or 'maxval'

    Returns:
        p1, p2 : scaling parameters
                 minmax  : p1=min, p2=max
                 standard: p1=mean, p2=std
                 maxval  : p1=max, p2=0.0 (unused)
    """
    if method == 'minmax':
        return float(series.min()), float(series.max())
    elif method == 'standard':
        return float(series.mean()), float(series.std())
    elif method == 'maxval':
        return float(series.abs().max()), 0.0
    else:
        raise ValueError(f"Unknown scaling method: {method}")


def apply_scaling(series, method, p1, p2):
    """Apply scaling to a pandas Series."""
    if method == 'minmax':
        return (series - p1) / (p2 - p1)
    elif method == 'standard':
        return (series - p1) / p2
    elif method == 'maxval':
        return series / p1
    else:
        raise ValueError(f"Unknown scaling method: {method}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--eos',   required=True, help='EOS identifier e.g. ph2_qcc')
    parser.add_argument('--phase', required=True, choices=['liquid', 'vapor'])
    args = parser.parse_args()

    eos   = args.eos
    phase = args.phase

    project_root = Path(__file__).resolve().parent.parent

    # ── load config ───────────────────────────────────────────────────
    config_path = project_root / 'config_scale.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if eos not in config:
        raise KeyError(
            f"EOS '{eos}' not found in config_scale.yaml.\n"
            f"Available: {list(config.keys())}"
        )
    scale_cfg = config[eos]

    # ── load raw CSV ──────────────────────────────────────────────────
    raw_path = project_root / 'data' / 'raw' / f'{phase}_{eos}.csv'
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw CSV not found: {raw_path}\n"
            f"Run Fortran dataset generator first."
        )
    df = pd.read_csv(raw_path)
    print(f"\nLoaded: {raw_path}  ({len(df):,} rows)")
    print(f"  Columns: {list(df.columns)}")

    # ── rename P, T columns to T_K, P_Pa if needed ───────────────────
    rename_map = {}
    if 'T' in df.columns and 'T_K' not in df.columns:
        rename_map['T'] = 'T_K'
    if 'P' in df.columns and 'P_Pa' not in df.columns:
        rename_map['P'] = 'P_Pa'
    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"  Renamed columns: {rename_map}")

    # ── compute and apply scaling ─────────────────────────────────────
    scaled_dir = project_root / 'data' / 'scaled'
    scaled_dir.mkdir(parents=True, exist_ok=True)

    df_scaled = pd.DataFrame()
    params_rows = []

    print(f"\nScaling columns:")
    for col, cfg in scale_cfg.items():
        if col not in df.columns:
            print(f"  SKIP {col} (not in CSV)")
            continue
        method = cfg['method']
        p1, p2 = compute_scale_params(df[col], method)
        df_scaled[col] = apply_scaling(df[col], method, p1, p2)
        params_rows.append({'col_name': col, 'method': method, 'p1': p1, 'p2': p2})
        print(f"  {col:12s}  method={method:8s}  p1={p1:.6g}  p2={p2:.6g}")

    # ── write scaled CSV ──────────────────────────────────────────────
    scaled_csv = scaled_dir / f'{phase}_{eos}_scaled.csv'
    df_scaled.to_csv(scaled_csv, index=False, float_format='%.8g')
    print(f"\nSaved scaled CSV : {scaled_csv}")

    # ── write scaling params CSV ──────────────────────────────────────
    params_csv = scaled_dir / f'{phase}_{eos}_scaling.csv'
    pd.DataFrame(params_rows).to_csv(params_csv, index=False, float_format='%.10g')
    print(f"Saved scaling params: {params_csv}")

    # ── sanity check ──────────────────────────────────────────────────
    print(f"\nScaled ranges:")
    for col in df_scaled.columns:
        print(f"  {col:12s}  [{df_scaled[col].min():.4f}, {df_scaled[col].max():.4f}]")
