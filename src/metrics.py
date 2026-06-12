##--------------------------------------------------------------------\
#   kalman_filter
#   './src/metrics.py'
#   Filter consistency metrics: NEES, NIS, chi-squared bounds.
#
#   Notes:
#     * The quadratic forms use true MATRIX multiplication (@), never
#       element-wise '*'.
#     * NEES compares the estimate against GROUND TRUTH, so it is a
#       simulation-only metric. NIS needs no ground truth and is the
#       one you can monitor on a real system.
#     * Chi-squared bounds are derived from scipy.stats.chi2.ppf with
#       explicit degrees of freedom rather than hard-coded table
#       values (e.g. chi2.ppf([0.025, 0.975], 4) = [0.484, 11.143]).
##--------------------------------------------------------------------\

import numpy as np
from scipy.stats import chi2


def nees(x_true, x_est, P):
    """Normalized Estimation Error Squared:
        eps = (x_true - x_est)' P^-1 (x_true - x_est)
    For a consistent filter, eps ~ chi-squared with df = state dimension.
    Requires ground truth, so it's a SIMULATION-ONLY metric."""
    e = np.asarray(x_true, dtype=float).reshape(-1, 1) \
        - np.asarray(x_est, dtype=float).reshape(-1, 1)
    return (e.T @ np.linalg.solve(P, e)).item()


def nis(innovation, S):
    """Normalized Innovation Squared:
        eps = y' S^-1 y
    For a consistent filter, eps ~ chi-squared with df = measurement
    dimension. Computable WITHOUT ground truth - this is the one you
    can monitor on a real system."""
    y = np.asarray(innovation, dtype=float).reshape(-1, 1)
    return (y.T @ np.linalg.solve(S, y)).item()


def chi2_interval(dof, confidence=0.95):
    """(lower, upper) bounds containing `confidence` of the chi-squared
    distribution with `dof` degrees of freedom."""
    alpha = (1.0 - confidence) / 2.0
    return float(chi2.ppf(alpha, dof)), float(chi2.ppf(1.0 - alpha, dof))


def mean_nees_interval(dof, n_runs, confidence=0.95):
    """Bounds for the AVERAGE NEES over n_runs Monte Carlo runs.
    The sum of n_runs chi2(dof) variables is chi2(n_runs*dof), so the
    average has tighter bounds - this is the proper consistency test."""
    lo, hi = chi2_interval(dof * n_runs, confidence)
    return lo / n_runs, hi / n_runs
