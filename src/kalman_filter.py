##--------------------------------------------------------------------\
#   kalman_filter
#   './src/kalman_filter.py'
#   A clean, generic linear Kalman filter.
#
#   Design notes:
#     * H is a proper (m x n) measurement matrix, so the innovation
#       covariance S is a well-conditioned (m x m) matrix that inverts
#       without tricks.
#     * Covariance update uses the JOSEPH FORM:
#           P = (I-KH) P (I-KH)' + K R K'
#       The short form P = (I-KH)P is algebraically equal but can lose
#       symmetry / positive-definiteness over many steps.
#     * No randomness is injected inside the filter. The filter's job
#       is to MODEL process noise via Q; the real noise lives in the
#       simulated world (ball.py).
#     * predict() and update() are separate calls, and update() may be
#       skipped (that's how you handle a missed measurement).
##--------------------------------------------------------------------\

import numpy as np


class KalmanFilter:
    """Linear Kalman filter:  x' = F x + B u + w,   z = H x + v
    with w ~ N(0, Q) and v ~ N(0, R)."""

    def __init__(self, F, H, Q, R, B=None, x0=None, P0=None):
        self.F = np.atleast_2d(np.asarray(F, dtype=float))
        self.H = np.atleast_2d(np.asarray(H, dtype=float))
        self.Q = np.atleast_2d(np.asarray(Q, dtype=float))
        self.R = np.atleast_2d(np.asarray(R, dtype=float))
        self.B = None if B is None else np.atleast_2d(np.asarray(B, dtype=float))

        n = self.F.shape[0]
        self.x = np.zeros((n, 1)) if x0 is None else \
            np.asarray(x0, dtype=float).reshape(n, 1)
        self.P = np.eye(n) * 500.0 if P0 is None else \
            np.atleast_2d(np.asarray(P0, dtype=float))

        # diagnostics from the most recent update()
        self.innovation = None   # y = z - H x_pred   (m x 1)
        self.S = None            # innovation covariance (m x m)

    def predict(self, u=None):
        """Time update: propagate state and covariance through the model."""
        self.x = self.F @ self.x
        if self.B is not None and u is not None:
            self.x = self.x + self.B @ np.asarray(u, dtype=float).reshape(-1, 1)
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        """Measurement update: fold in observation z (length-m vector)."""
        z = np.asarray(z, dtype=float).reshape(-1, 1)

        self.innovation = z - self.H @ self.x
        self.S = self.H @ self.P @ self.H.T + self.R
        # solve instead of explicit inverse: K = P H' S^-1
        K = np.linalg.solve(self.S.T, (self.P @ self.H.T).T).T

        self.x = self.x + K @ self.innovation

        # Joseph form covariance update (numerically stable)
        I_KH = np.eye(self.P.shape[0]) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return self.x
