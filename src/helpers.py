"""
src/helpers.py
Shared utility functions for the nn_eos pipeline.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path


def get_project_root():
    """Return absolute path to nn_eos project root."""
    return Path(__file__).resolve().parent.parent


def load_scaled_csv(eos, phase, project_root=None):
    """
    Load scaled dataset CSV for a given EOS and phase.

    Args:
        eos          : EOS identifier e.g. 'ph2_qcc'
        phase        : 'liquid' or 'vapor'
        project_root : optional path override

    Returns:
        df : DataFrame with all scaled columns
    """
    if project_root is None:
        project_root = get_project_root()
    path = project_root / 'data' / 'scaled' / f'{phase}_{eos}_scaled.csv'
    if not path.exists():
        raise FileNotFoundError(
            f"Scaled CSV not found: {path}\n"
            f"Run: python src/scaling.py --eos {eos} --phase {phase}"
        )
    return pd.read_csv(path)


def load_scaling_params(eos, phase, project_root=None):
    """
    Load scaling parameters CSV for a given EOS and phase.

    Args:
        eos          : EOS identifier e.g. 'ph2_qcc'
        phase        : 'liquid' or 'vapor'
        project_root : optional path override

    Returns:
        params : dict mapping col_name -> {method, p1, p2}
    """
    if project_root is None:
        project_root = get_project_root()
    path = project_root / 'data' / 'scaled' / f'{phase}_{eos}_scaling.csv'
    if not path.exists():
        raise FileNotFoundError(
            f"Scaling params not found: {path}\n"
            f"Run: python src/scaling.py --eos {eos} --phase {phase}"
        )
    df = pd.read_csv(path)
    params = {}
    for _, row in df.iterrows():
        params[row['col_name']] = {
            'method': row['method'],
            'p1':     float(row['p1']),
            'p2':     float(row['p2']),
        }
    return params


def compute_deriv_scale_factors(scaling_params, input_cols, output_col):
    """
    Compute chain rule scale factors for derivative loss.

    The network maps scaled inputs to scaled output:
        y_s = f(x1_s, x2_s)

    GradientTape gives df/dx1_s and df/dx2_s.
    Physical derivative: dy/dx1 = (df/dx1_s) * (dx1_s/dx1) * (dy/dy_s)^-1 ... wait,
    more precisely:
        dy/dx1 = (df/dx1_s) / (dx1_s/dx1)^-1 * (dy_s/dy)
    which simplifies to:
        dy/dx1 = (df/dx1_s) * (dy/dy_s) / (dx1_s/dx1)^-1

    For each input col i and output col:
        physical_deriv_i = (df/dxi_s) * unscale_out_factor / scale_in_factor_i

    where:
        scale_in_factor_i  = d(xi_s)/d(xi)  -- derivative of input scaling
        unscale_out_factor = d(y)/d(y_s)    -- derivative of output unscaling

    Args:
        scaling_params : dict from load_scaling_params()
        input_cols     : list of input column names e.g. ['T_K', 'P_Pa']
        output_col     : output column name e.g. 'rho'

    Returns:
        factors : list of float, one per input column
                  physical_deriv_i = (df/dxi_s) * factors[i]
    """
    out_p = scaling_params[output_col]
    unscale_out = _unscale_deriv(out_p['method'], out_p['p1'], out_p['p2'])

    factors = []
    for col in input_cols:
        in_p = scaling_params[col]
        scale_in = _scale_deriv(in_p['method'], in_p['p1'], in_p['p2'])
        factors.append(unscale_out * scale_in)

    return factors


def _scale_deriv(method, p1, p2):
    """d(x_scaled)/d(x) for given scaling method."""
    if method == 'minmax':
        return 1.0 / (p2 - p1)
    elif method == 'standard':
        return 1.0 / p2
    elif method == 'maxval':
        return 1.0 / p1
    else:
        raise ValueError(f"Unknown scaling method: {method}")


def _unscale_deriv(method, p1, p2):
    """d(x)/d(x_scaled) for given scaling method."""
    if method == 'minmax':
        return p2 - p1
    elif method == 'standard':
        return p2
    elif method == 'maxval':
        return p1
    else:
        raise ValueError(f"Unknown scaling method: {method}")


def scale_value(x, method, p1, p2):
    """Scale a single value or array."""
    if method == 'minmax':
        return (x - p1) / (p2 - p1)
    elif method == 'standard':
        return (x - p1) / p2
    elif method == 'maxval':
        return x / p1
    else:
        raise ValueError(f"Unknown scaling method: {method}")


def unscale_value(xs, method, p1, p2):
    """Unscale a single value or array."""
    if method == 'minmax':
        return xs * (p2 - p1) + p1
    elif method == 'standard':
        return xs * p2 + p1
    elif method == 'maxval':
        return xs * p1
    else:
        raise ValueError(f"Unknown scaling method: {method}")


def save_model(model, model_name, project_root=None):
    """Save TensorFlow model to models/{model_name}/."""
    if project_root is None:
        project_root = get_project_root()
    path = project_root / 'models' / model_name
    path.mkdir(parents=True, exist_ok=True)
    model.save(str(path / 'saved_model'))
    print(f"Model saved: {path / 'saved_model'}")


def load_model(model_name, project_root=None, custom_objects=None):
    """Load TensorFlow model from models/{model_name}/."""
    if project_root is None:
        project_root = get_project_root()
    path = project_root / 'models' / model_name / 'saved_model'
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            f"Run: python scripts/train.py --property ... --phase ..."
        )
    return tf.keras.models.load_model(str(path), custom_objects=custom_objects)


def load_csv(path):
    """Load a CSV file, raise clear error if missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)
