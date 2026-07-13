"""
scripts/export_weights.py
Export trained model weights to plain text files for Fortran inference.

Usage:
    python scripts/export_weights.py --property rho --phase liquid
    python scripts/export_weights.py --property rho --phase vapor
    python scripts/export_weights.py --property cs2 --phase liquid

Reads eos, inputs, activation from config_train.yaml (flat config).
Output_col is always the property name.

Output (unchanged format, read by nn_read_predict.f90):
    weights/{property}_{phase}/
        W1.txt, b1.txt, W2.txt, b2.txt, ...
        architecture.txt
        scaling.txt
"""

import sys
import argparse
import yaml
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.helpers import load_model, load_scaling_params, get_project_root
from src.architecture import get_layer_activations_simple, SReLU, SReLU3


def write_architecture(model, weights_dir, input_cols):
    layers = [l for l in model.layers if hasattr(l, 'kernel')]
    n_layers = len(layers)
    act_list = get_layer_activations_simple(model)  # list of (act_name, eps)
    with open(weights_dir / 'architecture.txt', 'w') as f:
        f.write(f"n_layers  {n_layers}\n")
        f.write(f"input_dim {len(input_cols)}\n")
        for i, layer in enumerate(layers):
            n_in, n_out = layer.kernel.shape
            if i < n_layers - 1:
                act_name, eps = act_list[i] if i < len(act_list) else ('linear', None)
            else:
                act_name, eps = 'linear', None
            f.write(f"layer_{i+1}_in  {n_in}\n")
            f.write(f"layer_{i+1}_out {n_out}\n")
            f.write(f"layer_{i+1}_act {act_name}\n")
            if act_name in ('srelu', 'srelu3') and eps is not None:
                f.write(f"layer_{i+1}_eps {eps:.6f}\n")
    print(f"  Written: architecture.txt  ({n_layers} layers)")


def write_weights(model, weights_dir):
    layers = [l for l in model.layers if hasattr(l, 'kernel')]
    for i, layer in enumerate(layers):
        W = layer.kernel.numpy()
        b = layer.bias.numpy()
        np.savetxt(weights_dir / f'W{i+1}.txt', W, fmt='%.10e')
        np.savetxt(weights_dir / f'b{i+1}.txt', b.reshape(1, -1), fmt='%.10e')
        print(f"  Written: W{i+1}.txt {W.shape}  b{i+1}.txt {b.shape}")


def write_scaling(scaling_params, weights_dir, input_cols, output_col):
    def write_block(f, col_name, params):
        method = params['method']
        p1     = params['p1']
        p2     = params['p2']
        f.write(f"col    {col_name}\n")
        f.write(f"method {method}\n")
        if method == 'minmax':
            f.write(f"min    {p1:.10g}\n")
            f.write(f"max    {p2:.10g}\n")
        elif method == 'standard':
            f.write(f"mean   {p1:.10g}\n")
            f.write(f"std    {p2:.10g}\n")
        elif method == 'maxval':
            f.write(f"max    {p1:.10g}\n")
        f.write("\n")

    with open(weights_dir / 'scaling.txt', 'w') as f:
        for col in input_cols:
            if col not in scaling_params:
                raise KeyError(f"Column '{col}' not in scaling params")
            write_block(f, col, scaling_params[col])
        if output_col not in scaling_params:
            raise KeyError(f"Output column '{output_col}' not in scaling params")
        write_block(f, output_col, scaling_params[output_col])
    print(f"  Written: scaling.txt  (T, P, {output_col})")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--property', required=True,
                        help='Property name e.g. rho, cs2')
    parser.add_argument('--phase', required=True, choices=['liquid', 'vapor'])
    args = parser.parse_args()

    project_root = get_project_root()
    model_name   = f"{args.property}_{args.phase}"
    print(f"\nExporting weights: {model_name}")

    # ── read flat config (eos, inputs, activation) ────────────────────
    with open(project_root / 'config_train.yaml') as f:
        cfg = yaml.safe_load(f)

    eos        = cfg['eos']
    input_cols = cfg['inputs']
    output_col = args.property      # always same as property name

    # ── load model ────────────────────────────────────────────────────
    model = load_model(model_name, project_root,
                       custom_objects={'SReLU': SReLU, 'SReLU3': SReLU3})
    print(f"  Model loaded: models/{model_name}/")

    # ── load scaling params ───────────────────────────────────────────
    scaling_params = load_scaling_params(eos, args.phase, project_root)

    # ── create weights directory ──────────────────────────────────────
    weights_dir = project_root / 'weights' / model_name
    weights_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output dir: {weights_dir}")

    # ── write files ───────────────────────────────────────────────────
    write_weights(model, weights_dir)
    write_architecture(model, weights_dir, input_cols)
    write_scaling(scaling_params, weights_dir, input_cols, output_col)

    print(f"\nExport complete: {weights_dir}")
