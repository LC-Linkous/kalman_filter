##--------------------------------------------------------------------\
#   kalman_filter
#   './src/tracked_ball.py'
#   Couples one Ball + one noisy position sensor + one KalmanFilter,
#   and records the histories needed for plotting and analysis.
#
#   This is the seam between "the world" (ball.py - truth + real noise)
#   and "the estimator" (kalman_filter.py - which only ever sees the
#   noisy measurements). Keeping them strictly separate is the whole
#   point of the exercise.
#
#   History alignment: every per-step list grows by exactly one entry
#   per step(). The initialization step has no innovation, so its NIS
#   entry is NaN (matplotlib simply skips NaNs when plotting).
##--------------------------------------------------------------------\

import numpy as np

from ball import Ball
from kalman_filter import KalmanFilter
from models import ball_tracking_model, initial_covariance
from metrics import nees, nis


class TrackedBall:
    def __init__(self, ball: Ball, dt=1.0, q=0.05, r_std=4.0,
                 q_filter_scale=1.0, rng=None):
        self.ball = ball
        self.dt = dt
        self.q = q                       # REAL process noise (world)
        self.r_std = r_std               # REAL measurement noise (sensor)
        self.q_filter_scale = q_filter_scale  # filter mistuning knob
        self.rng = rng if rng is not None else np.random.default_rng()

        self.kf = None                   # created on first measurement
        self._rebuild_model()

        # histories - all the same length, one entry per step()
        self.truth_states = []           # [x, y, vx, vy] ground truth
        self.measurements = []           # [zx, zy]
        self.estimates = []              # [x, y, vx, vy] filter estimate
        self.nees_history = []
        self.nis_history = []            # NaN on the initialization step
        self.bounce_steps = []           # step indices where a bounce broke the model

    # ---------------- model / tuning ----------------
    def _rebuild_model(self):
        self.F, self.B, self.H, Q, self.R = ball_tracking_model(
            self.dt, self.q, self.r_std)
        self.Q = Q * self.q_filter_scale
        if self.kf is not None:          # apply tuning changes live
            self.kf.F, self.kf.B, self.kf.H = self.F, self.B, self.H
            self.kf.Q, self.kf.R = self.Q, self.R

    def set_noise(self, q=None, r_std=None, q_filter_scale=None):
        if q is not None:
            self.q = q
        if r_std is not None:
            self.r_std = r_std
        if q_filter_scale is not None:
            self.q_filter_scale = q_filter_scale
        self._rebuild_model()

    # ---------------- simulation step ----------------
    def step(self, gravity, restitution, width, height,
             tell_filter_about_bounces=False):
        """Advance the world one tick, take a measurement, run the filter."""
        bounced_x, bounced_y = self.ball.step(
            self.dt, gravity, restitution, width, height, process_q=self.q)
        truth = self.ball.state()

        z = np.array([
            self.ball.x + self.rng.normal(0.0, self.r_std),
            self.ball.y + self.rng.normal(0.0, self.r_std)])

        if self.kf is None:
            # initialize from the first measurement: position from z,
            # velocity unknown (large variance)
            x0 = np.array([z[0], z[1], 0.0, 0.0])
            self.kf = KalmanFilter(self.F, self.H, self.Q, self.R, B=self.B,
                                   x0=x0,
                                   P0=initial_covariance(self.r_std))
            step_nis = float('nan')      # no innovation on init step
        else:
            if tell_filter_about_bounces and (bounced_x or bounced_y):
                # optional cheat: reflect the filter's velocity belief on
                # the axis that actually bounced, and de-weight it (a
                # taste of interacting-multiple-model ideas)
                if bounced_x:
                    self.kf.x[2, 0] *= -1
                    self.kf.P[2, 2] += 25.0
                if bounced_y:
                    self.kf.x[3, 0] *= -1
                    self.kf.P[3, 3] += 25.0
            self.kf.predict(u=[gravity])
            self.kf.update(z)
            step_nis = nis(self.kf.innovation, self.kf.S)

        if bounced_x or bounced_y:
            self.bounce_steps.append(len(self.truth_states))

        # record (all histories stay index-aligned)
        self.truth_states.append(truth.ravel().copy())
        self.measurements.append(z.copy())
        self.estimates.append(self.kf.x.ravel().copy())
        self.nees_history.append(nees(truth, self.kf.x, self.kf.P))
        self.nis_history.append(step_nis)
