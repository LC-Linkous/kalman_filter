##--------------------------------------------------------------------\
#   kalman_filter
#   './src/run_headless.py'
#   Headless Monte Carlo consistency study - no GUI required.
#
#   A single run's NEES is noisy: any one step can poke outside the
#   chi-squared bounds by chance. The textbook consistency test
#   AVERAGES NEES over many independent runs; the average has much
#   tighter chi-squared bounds (chi2(N*dof)/N). This script shows:
#     Fig 1: one run - trajectory, truth vs measurements vs estimate
#     Fig 2: single-run NEES vs the 50-run average, with both bound
#            sets, on a matched filter (no bounces - clean model)
#     Fig 3: the same averaged NEES when the filter's Q is mistuned
#            (overconfident x0.1 and pessimistic x10)
#
#   Run:  python run_headless.py [n_runs]
##--------------------------------------------------------------------\

import sys, types, os
import numpy as np

wx = types.ModuleType("wx")
class _Any:
    def __init__(self, *a, **k): pass
    def __getattr__(self, n): return _Any()
    def __call__(self, *a, **k): return _Any()
for name in ["Panel","Frame","Colour","Pen","Brush","Font","Timer","Slider",
             "Button","StaticText","BoxSizer","CollapsiblePane","App","Size",
             "GraphicsContext","AutoBufferedPaintDC"]:
    setattr(wx, name, _Any)
for const in ["FULL_REPAINT_ON_RESIZE","BG_STYLE_PAINT","EVT_PAINT","EVT_TIMER",
              "EVT_BUTTON","EVT_COLLAPSIBLEPANE_CHANGED","HORIZONTAL","VERTICAL",
              "EXPAND","ALL","LEFT","RIGHT","BOTTOM","TOP","TRANSPARENT_PEN",
              "PENSTYLE_SHORT_DASH","FONTFAMILY_TELETYPE","FONTSTYLE_NORMAL",
              "FONTWEIGHT_NORMAL","CP_DEFAULT_STYLE"]:
    setattr(wx, const, 0)
sys.modules["wx"] = wx

sys.path.insert(0, os.path.dirname(__file__))
import run_gui_dark as g

rng = np.random.default_rng(0)
world = g.BallWorld(5.0, rng)
F, B, Q, H, R = g.make_matrices(5.0, 8.0)
from kalman_filter import KalmanFilter
from metrics import nees, nis, chi2_bounds

x0 = np.array([world.x[0], world.x[1], 0.0, 0.0])
P0 = np.diag([30.0**2, 30.0**2, 150.0**2, 150.0**2])
kf = KalmanFilter(F, H, Q, R, x0, P0, B=B)

nis_vals, nees_vals = [], []
for k in range(2000):
    truth = world.step()
    z = world.measure(8.0)
    kf.predict(u=np.array([g.GRAVITY]))
    if g.reflect(kf.x):
        kf.P[2, 2] += (0.35 * abs(kf.x[2]) + 20.0) ** 2
        kf.P[3, 3] += (0.35 * abs(kf.x[3]) + 20.0) ** 2
    y = z - H @ kf.x
    S = H @ kf.P @ H.T + R
    kf.update(z)
    nis_vals.append(nis(y, S))
    nees_vals.append(nees(truth, kf.x, kf.P))
    # the ellipse math the canvas runs every frame
    vals, vecs = np.linalg.eigh(kf.P[:2, :2])
    assert np.all(vals > 0), f"P not PD at step {k}"
    assert np.isfinite(kf.x).all()
    # ball stays in the box
    assert 0 <= truth[0] <= g.WORLD_W and 0 <= truth[1] <= g.WORLD_H

lo, hi = chi2_bounds(2)
inside = np.mean([(lo <= v <= hi) for v in nis_vals])
print(f"2000 ticks OK.  mean NIS {np.mean(nis_vals):.2f} (theory 2), "
      f"{inside*100:.0f}% in band; mean NEES {np.mean(nees_vals):.2f}")