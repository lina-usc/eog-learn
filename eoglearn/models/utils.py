import numpy as np


def optimal_alpha(x_sim, x_raw, axis=None):
    """Compute the optimal scaling factor between a simulated and raw signal.

    Finds the scalar ``alpha`` that minimises
    ``mean((x_sim * alpha - x_raw) ** 2)`` using the analytical
    least-squares solution.

    Parameters
    ----------
    x_sim : np.ndarray
        Simulated signal array.
    x_raw : np.ndarray
        Raw (target) signal array, same shape as ``x_sim``.
    axis : int | None
        Axis along which to sum when computing the optimal alpha. If
        ``None`` (default), a single global scalar is returned. Pass an
        integer to compute per-slice alphas (e.g. ``axis=1`` for
        per-channel scaling).

    Returns
    -------
    alpha : float | np.ndarray
        Optimal scaling factor(s). A scalar when ``axis=None``, otherwise
        an array with the summed axis removed.
    """
    numerator = np.sum(x_sim * x_raw, axis=axis)
    denominator = np.sum(x_sim**2, axis=axis)
    return numerator / denominator
