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

import sys

import numpy as np
import matplotlib.pyplot as plt

from ball import Ball
from tracked_ball import TrackedBall
from metrics import chi2_interval, mean_nees_interval

STEPS = 150
GRAVITY = 0.3
Q_TRUE = 0.05
R_STD = 4.0
# huge field: the ball never bounces, so the linear model is exact
FIELD = 1_000_000.0


def simulate_run(seed, q_filter_scale=1.0):
    rng = np.random.default_rng(seed)
    ball = Ball(FIELD / 2, FIELD / 2 + 5000,
                rng.uniform(-6, 6), rng.uniform(0, 6),
                rng=np.random.default_rng(seed + 10_000))
    tb = TrackedBall(ball, q=Q_TRUE, r_std=R_STD,
                     q_filter_scale=q_filter_scale,
                     rng=np.random.default_rng(seed + 20_000))
    for _ in range(STEPS):
        tb.step(GRAVITY, 1.0, FIELD, FIELD)
    return tb


def averaged_nees(n_runs, q_filter_scale=1.0, seed0=0):
    runs = [simulate_run(seed0 + i, q_filter_scale) for i in range(n_runs)]
    return np.mean([r.nees_history[1:] for r in runs], axis=0), runs


def main():
    n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    print(f"Running {n_runs} Monte Carlo runs of {STEPS} steps...")
    avg_nees, runs = averaged_nees(n_runs)
    one = runs[0]

    # ---- Fig 1: trajectory of one run ----
    truth = np.array(one.truth_states)
    meas = np.array(one.measurements)
    est = np.array(one.estimates)
    plt.figure(figsize=(8, 5))
    plt.plot(truth[:, 0], truth[:, 1], '-', lw=2, label='truth $x(k)$')
    plt.plot(meas[:, 0], meas[:, 1], '.', ms=4, alpha=0.5,
             label='measured $z(k)$')
    plt.plot(est[:, 0], est[:, 1], '-', lw=1.5,
             label=r'estimate $\hat{x}(k)$')
    plt.xlabel('x (px)'); plt.ylabel('y (px)')
    plt.title('One run: projectile under gravity, noisy position sensor')
    plt.legend()

    # ---- Fig 2: single vs averaged NEES, matched filter ----
    lo1, hi1 = chi2_interval(4)
    loN, hiN = mean_nees_interval(4, n_runs)
    single = np.array(one.nees_history[1:])
    inside = np.mean((avg_nees >= loN) & (avg_nees <= hiN)) * 100

    plt.figure(figsize=(8, 5))
    plt.semilogy(single, lw=0.7, alpha=0.5, label='single-run NEES')
    plt.semilogy(avg_nees, lw=1.5, label=f'average NEES ({n_runs} runs)')
    plt.axhline(lo1, color='gray', ls=':', lw=1)
    plt.axhline(hi1, color='gray', ls=':', lw=1,
                label=r'single-run $\chi^2_4$ 95% bounds')
    plt.axhline(loN, color='k', ls='--', lw=1)
    plt.axhline(hiN, color='k', ls='--', lw=1,
                label=f'averaged bounds ({inside:.0f}% of steps inside)')
    plt.xlabel('step $k$'); plt.ylabel('NEES')
    plt.title('Matched filter: NEES is chi-squared consistent')
    plt.legend(fontsize=8)
    print(f"  matched filter: mean NEES = {avg_nees.mean():.3f} "
          f"(theory: 4.0), {inside:.0f}% of averaged steps in bounds")

    # ---- Fig 3: mistuned filters ----
    over, _ = averaged_nees(n_runs, q_filter_scale=0.1, seed0=50_000)
    under, _ = averaged_nees(n_runs, q_filter_scale=10.0, seed0=90_000)
    plt.figure(figsize=(8, 5))
    plt.semilogy(avg_nees, lw=1.2, label='matched (Q x1)')
    plt.semilogy(over, lw=1.2, label='overconfident (Q x0.1) - NEES too HIGH')
    plt.semilogy(under, lw=1.2, label='pessimistic (Q x10) - NEES too LOW')
    plt.axhline(loN, color='k', ls='--', lw=1)
    plt.axhline(hiN, color='k', ls='--', lw=1, label='averaged 95% bounds')
    plt.xlabel('step $k$'); plt.ylabel('average NEES')
    plt.title('Filter mistuning is visible in NEES')
    plt.legend(fontsize=8)
    print(f"  overconfident:  mean NEES = {over.mean():.2f}")
    print(f"  pessimistic:    mean NEES = {under.mean():.2f}")

    plt.show()


if __name__ == '__main__':
    main()
