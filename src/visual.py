"""
src/visual.py
Visualization utilities for nn_eos.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_contour(x, xlabel, y, ylabel, z, zlabel, save_path=None):
    """
    Plot a scatter/contour of z values in x-y space.
    Uses seismic colormap symmetric at zero.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    vmax = np.percentile(np.abs(z), 95)
    vmin = -vmax
    if vmax == 0:
        vmax = 1.0; vmin = -1.0

    sc = ax.scatter(x, y, c=z, cmap='seismic', vmin=vmin, vmax=vmax, s=2)
    fig.colorbar(sc, ax=ax, label=zlabel)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_1d_comparison(x, xlabel, y_ref, label_ref, y_pred, label_pred,
                        save_path=None):
    """
    1D comparison plot: reference vs prediction.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y_ref,  'k-',  lw=2,   label=label_ref)
    ax.plot(x, y_pred, 'r--', lw=1.5, label=label_pred)
    ax.set_xlabel(xlabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
