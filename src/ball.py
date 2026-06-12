##--------------------------------------------------------------------\
#   kalman_filter
#   './src/ball.py'
#   Ball physics: projectile motion with wall bounces.
#
#   Conventions:
#     * Physics lives in a y-UP world: gravity is negative-y.
#       The GUI flips y exactly ONCE, at draw time.
#     * Standard kinematic step (matches the Kalman model in models.py
#       exactly, which is what makes NEES consistency tests honest):
#           x' = x + vx*dt + 0.5*ax*dt^2
#           v' = v + a*dt
#     * Process noise is REAL here: each step draws a random
#       acceleration ~ N(0, q). The filter's Q describes this same
#       randomness, so "matched filter" actually means matched.
#     * Bounces reflect velocity (scaled by restitution) and mirror the
#       position overshoot, so the ball never tunnels into a wall.
##--------------------------------------------------------------------\

import numpy as np


class Ball:
    def __init__(self, x, y, vx, vy, radius=10.0, color=(220, 60, 60),
                 rng=None):
        self.x, self.y = float(x), float(y)
        self.vx, self.vy = float(vx), float(vy)
        self.radius = float(radius)
        self.color = color
        self.rng = rng if rng is not None else np.random.default_rng()

    def state(self):
        """Ground-truth state vector [x, y, vx, vy] (column)."""
        return np.array([[self.x], [self.y], [self.vx], [self.vy]])

    def step(self, dt, gravity, restitution, width, height, process_q=0.0):
        """Advance one time step.

        Returns (bounced_x, bounced_y): which axes reflected this step.
        Callers that only care whether ANY bounce happened can use
        `any(ball.step(...))`.

        gravity      - magnitude of downward acceleration (px/step^2)
        restitution  - 0..1, fraction of speed kept after a bounce
        process_q    - variance of the random acceleration (the 'real'
                       process noise the Kalman Q is modeling)
        """
        ax = 0.0
        ay = -gravity
        if process_q > 0.0:
            std = np.sqrt(process_q)
            ax += self.rng.normal(0.0, std)
            ay += self.rng.normal(0.0, std)

        # kinematic update (matches F, B in models.py)
        self.x += self.vx * dt + 0.5 * ax * dt * dt
        self.y += self.vy * dt + 0.5 * ay * dt * dt
        self.vx += ax * dt
        self.vy += ay * dt

        return self._bounce(restitution, width, height)

    def _bounce(self, restitution, width, height):
        """Reflect off field boundaries (mirror overshoot, damp speed).
        Returns (bounced_x, bounced_y) so the tracker knows WHICH axis
        broke the linear model - no position-based guessing needed."""
        r = self.radius
        bounced_x = bounced_y = False

        if self.x < r:                      # left wall
            self.x = r + (r - self.x)
            self.vx = -self.vx * restitution
            bounced_x = True
        elif self.x > width - r:            # right wall
            self.x = (width - r) - (self.x - (width - r))
            self.vx = -self.vx * restitution
            bounced_x = True

        if self.y < r:                      # floor (y-up world)
            self.y = r + (r - self.y)
            self.vy = -self.vy * restitution
            bounced_y = True
        elif self.y > height - r:           # ceiling
            self.y = (height - r) - (self.y - (height - r))
            self.vy = -self.vy * restitution
            bounced_y = True

        return bounced_x, bounced_y
