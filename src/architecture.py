"""
src/architecture.py
Build Keras models for nn_eos.
Supports single-output models with uniform or per-layer activations.

activation in config_train.yaml can be:
  - a string : 'relu', 'tanh', 'srelu'  -> same for all hidden layers
  - a list   : ['relu', 'relu', 'srelu'] -> per-layer

SReLU (C2 smooth ReLU approximation):
  f(x) = 0                                  x <= -eps
  f(x) = (x+eps)^3/(4eps^2) - (x+eps)^4/(8eps^3)   -eps < x < eps
  f(x) = x - eps/2                          x >= eps
  Default eps=0.5. Set via srelu_eps in config_train.yaml.
"""

import tensorflow as tf
import numpy as np


# ── SReLU C2 — piecewise cubic (degree-4 blend) ──────────────────────────────
class SReLU(tf.keras.layers.Layer):
    """
    C2-smooth piecewise activation. f, f', f'' continuous at x=±eps.
    No transcendental functions — only polynomial operations.
    """
    def __init__(self, eps=0.5, **kwargs):
        super().__init__(**kwargs)
        self.eps = float(eps)

    def call(self, x):
        eps = self.eps
        t      = x + eps
        cubic  = t**3 / (4.0 * eps**2) - t**4 / (8.0 * eps**3)
        linear = x - eps / 2.0
        return tf.where(x <= -eps, tf.zeros_like(x),
               tf.where(x >=  eps, linear, cubic))

    def get_config(self):
        config = super().get_config()
        config.update({'eps': self.eps})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ── SReLU C3 — piecewise degree-7 smoothstep blend ───────────────────────────
class SReLU3(tf.keras.layers.Layer):
    """
    C3-smooth piecewise activation. f, f', f'', f''' continuous at x=±eps.
    Uses degree-7 smoothstep polynomial P(t) = 35t^4 - 84t^5 + 70t^6 - 20t^7.
    No transcendental functions — only polynomial operations.
    Eliminates second-derivative discontinuity present in C2 SReLU.
    """
    def __init__(self, eps=0.5, **kwargs):
        super().__init__(**kwargs)
        self.eps = float(eps)

    def call(self, x):
        eps = self.eps
        t      = (x + eps) / (2.0 * eps)                          # t in [0,1]
        P      = 35*t**4 - 84*t**5 + 70*t**6 - 20*t**7           # smoothstep7
        blend  = (eps / 2.0) * P
        linear = x - eps / 2.0
        return tf.where(x <= -eps, tf.zeros_like(x),
               tf.where(x >=  eps, linear, blend))

    def get_config(self):
        config = super().get_config()
        config.update({'eps': self.eps})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def _make_activation(act_str, srelu_eps=0.5):
    """Convert activation string to Keras activation or layer."""
    if act_str == 'srelu':
        return SReLU(eps=srelu_eps)
    if act_str == 'srelu3':
        return SReLU3(eps=srelu_eps)
    return act_str  # 'relu', 'tanh', 'linear' etc — Keras handles these


def build_model(input_dim, n_hidden, activation, lr, output_dim=1, srelu_eps=0.5):
    """
    Build and compile a feedforward neural network.

    Args:
        input_dim  : number of input features (typically 2: T, P)
        n_hidden   : list of hidden layer sizes e.g. [32, 16, 8]
        activation : string (uniform) or list of strings (per-layer)
        lr         : learning rate for Adam optimizer
        output_dim : number of outputs
        srelu_eps  : epsilon for SReLU activation (default 0.5)

    Returns:
        model : compiled Keras model
    """
    if isinstance(activation, str):
        act_list = [activation] * len(n_hidden)
    else:
        act_list = list(activation)
        if len(act_list) != len(n_hidden):
            raise ValueError(
                f"activation list length ({len(act_list)}) must match "
                f"n_hidden length ({len(n_hidden)})"
            )

    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    for units, act_str in zip(n_hidden, act_list):
        act = _make_activation(act_str, srelu_eps)
        if isinstance(act, str):
            x = tf.keras.layers.Dense(units, activation=act)(x)
        else:
            # SReLU: Dense with linear activation + separate activation layer
            x = tf.keras.layers.Dense(units, activation='linear')(x)
            x = act(x)
    outputs = tf.keras.layers.Dense(output_dim, activation='linear')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.MeanAbsoluteError(),
        metrics=['mape']
    )
    return model


def get_layer_activations(model):
    """
    Extract per-layer activation info from a compiled model.
    Returns list of (act_name, eps) tuples for hidden layers only.
    eps is None for non-SReLU activations.
    Handles models with SReLU as separate activation layers.
    """
    result = []
    layers = model.layers[1:]  # skip Input layer

    i = 0
    while i < len(layers):
        layer = layers[i]
        if hasattr(layer, 'kernel'):
            # Dense layer — check if followed by SReLU
            cfg = layer.get_config()
            act = cfg.get('activation', 'linear')
            if isinstance(act, dict):
                act = act.get('class_name', 'linear').lower()

            # check if next layer is SReLU
            if (i + 1 < len(layers) and
                    isinstance(layers[i+1], SReLU)):
                srelu_layer = layers[i+1]
                # only append if this is not the output layer
                # output dense has no SReLU after it
                result.append(('srelu', srelu_layer.eps))
                i += 2
                continue
            else:
                if act != 'linear':
                    result.append((act, None))
                # skip output linear layer (last Dense with linear act)
            i += 1
        else:
            i += 1

    # remove the last entry if it came from the output layer
    # output layer is always linear — we only want hidden layer activations
    return result


def get_layer_activations_simple(model):
    """
    Simpler version: returns list of (act_name, eps) for hidden layers.
    Used by export_weights.py.
    """
    acts = []
    layers = model.layers

    for i, layer in enumerate(layers):
        if isinstance(layer, SReLU3):
            acts.append(('srelu3', layer.eps))
        elif isinstance(layer, SReLU):
            acts.append(('srelu', layer.eps))
        elif hasattr(layer, 'kernel'):
            cfg = layer.get_config()
            act = cfg.get('activation', 'linear')
            if isinstance(act, dict):
                act = act.get('class_name', 'linear').lower()
            if act != 'linear':
                acts.append((act, None))

    return acts
