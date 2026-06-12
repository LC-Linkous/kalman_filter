##--------------------------------------------------------------------\
#   kalman_filter
#   './src/models.py'
#   State-space model for a ball under gravity, observed by a noisy
#   position sensor. State: x = [x, y, vx, vy]'.
#
#   The dynamics here match ball.py's kinematic step EXACTLY, which is
#   what makes the consistency analysis (NEES) meaningful: when the
#   filter's model truly describes the world, NEES should live inside
#   its chi-squared bounds.
#
#   Notes:
#     * B (control) applies gravity to the Y components with the
#       kinematic coefficients [0, -dt^2/2, 0, -dt]'; the control
#       input is the gravity MAGNITUDE, B carries sign and geometry.
#     * Q is the standard discrete white-noise-acceleration covariance
#       scaled by the acceleration variance q.
#     * H is 2x4: we measure position only. Velocity is INFERRED by
#       the filter - that's the magic trick worth seeing.
##--------------------------------------------------------------------\

import numpy as np


def ball_tracking_model(dt=1.0, q=0.05, r_std=4.0):
    """Returns (F, B, H, Q, R) for the [x, y, vx, vy] ball model.

    dt    - time step (1.0 = one simulation tick)
    q     - variance of the random acceleration (process noise)
    r_std - standard deviation of the position measurement noise (px)
    """
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=float)

    # gravity enters as a known control input u = [g] (magnitude,
    # downward); B carries the sign and kinematic coefficients
    B = np.array([[0.0],
                  [-0.5 * dt * dt],
                  [0.0],
                  [-dt]])

    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]], dtype=float)

    Q = q * np.array([
        [dt**4 / 4, 0,         dt**3 / 2, 0],
        [0,         dt**4 / 4, 0,         dt**3 / 2],
        [dt**3 / 2, 0,         dt**2,     0],
        [0,         dt**3 / 2, 0,         dt**2]])

    R = np.eye(2) * (r_std ** 2)

    return F, B, H, Q, R


def initial_covariance(r_std, max_speed=20.0):
    """P0 for a filter initialized from the FIRST MEASUREMENT:
    position uncertainty = sensor noise; velocity unknown, so give it
    a variance generous enough to cover any plausible launch speed.
    Initializing from the first measurement (instead of an arbitrary
    zero state) removes the startup transient honestly."""
    return np.diag([r_std**2, r_std**2, max_speed**2, max_speed**2])
