##--------------------------------------------------------------------\
#   kalman_filter
#   './src/run_gui.py'
#   Entry point: wxPython GUI for the bouncing-ball Kalman demo.
#   Multiple balls, each tracked by its own Kalman filter, with live
#   controls and a matplotlib analysis tab.
#
#   Layout:
#     [ Notebook: Live Simulation | Analysis ]  [ control sidebar ]
#     [           Start / Pause / Reset buttons + status            ]
#
#   On the canvas, per ball:
#     filled circle         = latest noisy MEASUREMENT (what the
#                             filter sees - this is "the sensor")
#     small gray dots       = measurement trail
#     colored outline circle = GROUND TRUTH (toggleable; no fill)
#     black ring + crosshair = Kalman estimate
#   Watch the estimate ring hug the truth outline more tightly than
#   the filled measurement ball jitters: that's the filter earning
#   its keep.
#
#   Run:  python run_gui.py
##--------------------------------------------------------------------\

from collections import deque

import numpy as np
import wx
import matplotlib
matplotlib.use('WXAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

from ball import Ball
from tracked_ball import TrackedBall
from metrics import chi2_interval

BALL_COLORS = [(220, 60, 60), (60, 140, 220), (60, 180, 90), (230, 160, 40),
               (160, 80, 200), (50, 180, 180), (200, 90, 150), (130, 130, 60),
               (90, 90, 220), (200, 120, 80)]
TRAIL_LEN = 120
TIMER_MS = 33


class SimPanel(wx.Panel):
    """White canvas. Physics is y-up; the flip to screen coordinates
    happens HERE and only here."""

    def __init__(self, parent, frame):
        super().__init__(parent)
        self.frame = frame
        self.SetBackgroundColour(wx.WHITE)
        self.SetDoubleBuffered(True)
        self.Bind(wx.EVT_PAINT, self.on_paint)

    def flip(self, y):
        return self.GetClientSize().height - y

    def on_paint(self, event):
        # GCDC: wx.PaintDC ignores alpha on most platforms; the
        # graphics-context wrapper actually blends the pale trail.
        dc = wx.GCDC(wx.PaintDC(self))
        dc.Clear()
        s = self.frame.settings

        for tb in self.frame.tracked_balls:
            r, g, b = tb.ball.color
            color = wx.Colour(r, g, b)
            pale = wx.Colour(r, g, b, 70)

            # truth trail
            if s['show_truth_trail']:
                dc.SetPen(wx.Pen(pale, 1))
                dc.SetBrush(wx.Brush(pale))
                for st in list(tb.truth_states)[-TRAIL_LEN:]:
                    dc.DrawCircle(int(st[0]), int(self.flip(st[1])), 2)

            # measurement trail
            if s['show_measurements']:
                gray = wx.Colour(120, 120, 120, 130)
                dc.SetPen(wx.Pen(gray, 1))
                dc.SetBrush(wx.Brush(gray))
                for z in list(tb.measurements)[-TRAIL_LEN:]:
                    dc.DrawCircle(int(z[0]), int(self.flip(z[1])), 2)

            # the "ball" you watch = the latest noisy measurement
            # (filled circle). This is what the filter actually sees.
            if tb.measurements:
                zx, zy = tb.measurements[-1]
                dc.SetPen(wx.Pen(color, 2))
                dc.SetBrush(wx.Brush(color))
                dc.DrawCircle(int(zx), int(self.flip(zy)),
                              int(tb.ball.radius))

            # ground truth: same color, OUTLINE ONLY, no fill
            if s['show_truth']:
                dc.SetPen(wx.Pen(color, 2))
                dc.SetBrush(wx.TRANSPARENT_BRUSH)
                dc.DrawCircle(int(tb.ball.x), int(self.flip(tb.ball.y)),
                              int(tb.ball.radius))

            # Kalman estimate: black ring + crosshair
            if s['show_estimates'] and tb.kf is not None:
                ex, ey = tb.kf.x[0, 0], self.flip(tb.kf.x[1, 0])
                dc.SetBrush(wx.TRANSPARENT_BRUSH)
                dc.SetPen(wx.Pen(wx.Colour(20, 20, 20), 2))
                rr = int(tb.ball.radius + 4)
                dc.DrawCircle(int(ex), int(ey), rr)
                dc.DrawLine(int(ex - rr), int(ey), int(ex + rr), int(ey))
                dc.DrawLine(int(ex), int(ey - rr), int(ex), int(ey + rr))


class KalmanFrame(wx.Frame):
    def __init__(self, parent, title):
        super().__init__(parent, title=title, size=(1250, 700))

        self.settings = {
            'n_balls': 3,
            'gravity': 0.50,
            'restitution': 0.85,
            'radius': 12,
            'speed': 8.0,
            'r_std': 4.0,          # measurement noise std (px)
            'q': 0.05,             # real process noise variance
            'q_filter_scale': 1.0, # filter mistuning multiplier
            'tell_bounces': False,
            'same_start': False,   # launch all balls identically (on Reset)
            'show_truth': True,
            'show_truth_trail': True,
            'show_measurements': True,
            'show_estimates': True,
        }
        self.tracked_balls = []
        self.is_running = False

        panel = wx.Panel(self)

        # ---------------- notebook (left) ----------------
        self.notebook = wx.Notebook(panel)
        self.sim_panel = SimPanel(self.notebook, self)
        self.notebook.AddPage(self.sim_panel, "Live Simulation")

        analysis_page = wx.Panel(self.notebook)
        self.fig = plt.figure(figsize=(6, 6))
        self.canvas = FigureCanvas(analysis_page, -1, self.fig)
        self.analysis_button = wx.Button(analysis_page,
                                         label="Update Analysis (ball 1)")
        self.analysis_button.Bind(wx.EVT_BUTTON, self.update_analysis)
        a_sizer = wx.BoxSizer(wx.VERTICAL)
        a_sizer.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 5)
        a_sizer.Add(self.analysis_button, 0, wx.ALL, 5)
        analysis_page.SetSizer(a_sizer)
        self.notebook.AddPage(analysis_page, "Analysis")

        # ---------------- control sidebar (right) ----------------
        sidebar = wx.Panel(panel)
        sb = wx.BoxSizer(wx.VERTICAL)

        def add_slider(label_fmt, key, lo, hi, value, scale=1.0):
            """Slider storing value*scale -> settings[key]."""
            label = wx.StaticText(sidebar, label=label_fmt.format(value))
            slider = wx.Slider(sidebar, value=int(round(value / scale)),
                               minValue=lo, maxValue=hi)

            def on_slide(evt, k=key, lbl=label, fmt=label_fmt, sc=scale,
                         sld=slider):
                val = sld.GetValue() * sc
                self.settings[k] = val
                lbl.SetLabel(fmt.format(val))
                self.apply_tuning()
            slider.Bind(wx.EVT_SLIDER, on_slide)
            sb.Add(label, 0, wx.LEFT | wx.TOP, 8)
            sb.Add(slider, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)
            return slider

        sb.Add(wx.StaticText(sidebar, label="-- World --"), 0, wx.ALL, 8)
        lbl = wx.StaticText(sidebar, label="Number of balls (on Reset)")
        self.n_spin = wx.SpinCtrl(sidebar, value="3", min=1, max=10)
        sb.Add(lbl, 0, wx.LEFT, 8)
        sb.Add(self.n_spin, 0, wx.LEFT, 8)

        self.same_start_check = wx.CheckBox(
            sidebar, label="Same start point && launch (on Reset)")
        self.same_start_check.Bind(
            wx.EVT_CHECKBOX,
            lambda e: self.settings.update(
                same_start=self.same_start_check.GetValue()))
        sb.Add(self.same_start_check, 0, wx.ALL, 8)

        add_slider("Gravity: {:.2f} px/step\u00b2", 'gravity', 0, 100,
                   self.settings['gravity'], scale=0.01)
        add_slider("Bounciness: {:.2f}", 'restitution', 50, 100,
                   self.settings['restitution'], scale=0.01)
        add_slider("Ball radius: {:.0f} px (on Reset)", 'radius', 5, 30,
                   self.settings['radius'])
        add_slider("Launch speed: {:.0f} px/step (on Reset)", 'speed', 1, 20,
                   self.settings['speed'])

        sb.Add(wx.StaticText(sidebar, label="-- Noise --"), 0, wx.ALL, 8)
        add_slider("Measurement noise \u03c3: {:.1f} px", 'r_std', 0, 200,
                   self.settings['r_std'], scale=0.1)
        add_slider("Process noise q: {:.2f}", 'q', 0, 200,
                   self.settings['q'], scale=0.01)

        sb.Add(wx.StaticText(sidebar, label="-- Filter --"), 0, wx.ALL, 8)
        add_slider("Filter Q multiplier: {:.2f}x", 'q_filter_scale', 5, 500,
                   self.settings['q_filter_scale'], scale=0.01)

        self.bounce_check = wx.CheckBox(sidebar,
                                        label="Tell filter about bounces")
        self.bounce_check.Bind(
            wx.EVT_CHECKBOX,
            lambda e: self.settings.update(
                tell_bounces=self.bounce_check.GetValue()))
        sb.Add(self.bounce_check, 0, wx.ALL, 8)

        sb.Add(wx.StaticText(sidebar, label="-- Display --"), 0, wx.ALL, 8)
        for key, text in [('show_truth', "Show ground truth (outline)"),
                          ('show_truth_trail', "Show truth trail"),
                          ('show_measurements', "Show measurement trail"),
                          ('show_estimates', "Show Kalman estimates")]:
            cb = wx.CheckBox(sidebar, label=text)
            cb.SetValue(self.settings[key])
            cb.Bind(wx.EVT_CHECKBOX,
                    lambda e, k=key, c=cb: self.settings.update({k: c.GetValue()}))
            sb.Add(cb, 0, wx.LEFT | wx.TOP, 8)

        sidebar.SetSizer(sb)

        # ---------------- buttons (bottom) ----------------
        btn_panel = wx.Panel(panel)
        start_btn = wx.Button(btn_panel, label="Start")
        pause_btn = wx.Button(btn_panel, label="Pause")
        reset_btn = wx.Button(btn_panel, label="Reset")
        start_btn.Bind(wx.EVT_BUTTON, self.on_start)
        pause_btn.Bind(wx.EVT_BUTTON, self.on_pause)
        reset_btn.Bind(wx.EVT_BUTTON, self.on_reset)
        self.status = wx.StaticText(btn_panel, label="Press Start")
        b_sizer = wx.BoxSizer(wx.HORIZONTAL)
        for b in (start_btn, pause_btn, reset_btn):
            b_sizer.Add(b, 0, wx.ALL, 5)
        b_sizer.Add(self.status, 1, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        btn_panel.SetSizer(b_sizer)

        # ---------------- main layout ----------------
        top = wx.BoxSizer(wx.HORIZONTAL)
        top.Add(self.notebook, 1, wx.EXPAND | wx.ALL, 5)
        top.Add(sidebar, 0, wx.EXPAND | wx.ALL, 5)
        main = wx.BoxSizer(wx.VERTICAL)
        main.Add(top, 1, wx.EXPAND)
        main.Add(btn_panel, 0, wx.EXPAND)
        panel.SetSizer(main)

        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_tick, self.timer)
        self.Bind(wx.EVT_CLOSE, self.on_close)

    # ---------------- ball management ----------------
    def make_balls(self):
        w, h = self.sim_panel.GetClientSize()
        w, h = max(w, 200), max(h, 200)
        s = self.settings
        rng = np.random.default_rng()
        self.tracked_balls = []

        # identical launch state for every ball when comparing filters:
        # same position, same velocity - only the noise realizations
        # (process + measurement) differ between balls.
        fixed_pos = (w * 0.25, h * 0.35)
        fixed_angle = np.deg2rad(60.0)

        for i in range(int(self.n_spin.GetValue())):
            speed = s['speed']
            if s['same_start']:
                x0, y0 = fixed_pos
                angle = fixed_angle
            else:
                x0 = rng.uniform(w * 0.2, w * 0.8)
                y0 = rng.uniform(h * 0.4, h * 0.8)
                angle = rng.uniform(0.25 * np.pi, 0.75 * np.pi)
            ball = Ball(x0, y0,
                        speed * np.cos(angle), speed * np.sin(angle),
                        radius=s['radius'],
                        color=BALL_COLORS[i % len(BALL_COLORS)], rng=rng)
            self.tracked_balls.append(
                TrackedBall(ball, q=s['q'], r_std=s['r_std'],
                            q_filter_scale=s['q_filter_scale'], rng=rng))

    def apply_tuning(self):
        s = self.settings
        for tb in self.tracked_balls:
            tb.set_noise(q=s['q'], r_std=s['r_std'],
                         q_filter_scale=s['q_filter_scale'])

    # ---------------- run control ----------------
    def on_start(self, event):
        if not self.tracked_balls:
            self.make_balls()
        self.is_running = True
        self.timer.Start(TIMER_MS)
        self.status.SetLabel("Running")

    def on_pause(self, event):
        self.is_running = False
        self.timer.Stop()
        self.status.SetLabel("Paused")

    def on_reset(self, event):
        self.is_running = False
        self.timer.Stop()
        self.make_balls()
        self.sim_panel.Refresh()
        self.status.SetLabel("Reset - press Start")

    def on_tick(self, event):
        if not self.is_running:
            return
        w, h = self.sim_panel.GetClientSize()
        s = self.settings
        for tb in self.tracked_balls:
            tb.step(s['gravity'], s['restitution'], w, h,
                    tell_filter_about_bounces=s['tell_bounces'])
        self.sim_panel.Refresh()
        steps = len(self.tracked_balls[0].truth_states)
        self.status.SetLabel(f"Running - step {steps}")

    # ---------------- analysis ----------------
    def update_analysis(self, event):
        if not self.tracked_balls:
            return
        tb = self.tracked_balls[0]
        if len(tb.truth_states) < 5:
            return
        truth = np.array(tb.truth_states)
        meas = np.array(tb.measurements)
        est = np.array(tb.estimates)
        nees_arr = np.array(tb.nees_history)
        nis_arr = np.array(tb.nis_history)   # NaN at init step; skipped in plot

        self.fig.clear()
        ax1 = self.fig.add_subplot(311)
        ax1.plot(truth[:, 0], truth[:, 1], '-', lw=1.5, label='truth $x(k)$')
        ax1.plot(meas[:, 0], meas[:, 1], '.', ms=3, alpha=0.4,
                 label='measured $z(k)$')
        ax1.plot(est[:, 0], est[:, 1], '-', lw=1.2,
                 label=r'estimate $\hat{x}(k)$')
        ax1.set_xlabel('x (px)'); ax1.set_ylabel('y (px)')
        ax1.legend(fontsize=7); ax1.set_title('Trajectory (ball 1)', fontsize=9)

        lo4, hi4 = chi2_interval(4)
        ax2 = self.fig.add_subplot(312)
        ax2.plot(nees_arr, lw=0.8, label=r'NEES $\epsilon(k)$')
        ax2.axhline(lo4, color='k', ls='--', lw=0.8)
        ax2.axhline(hi4, color='k', ls='--', lw=0.8,
                    label=r'$\chi^2_4$ 95% bounds')
        for b in tb.bounce_steps:
            ax2.axvline(b, color='r', alpha=0.15, lw=1)
        ax2.set_yscale('log')
        ax2.set_ylabel('NEES'); ax2.legend(fontsize=7)
        ax2.set_title('NEES (red bands = bounces; the model breaks there)',
                      fontsize=9)

        lo2, hi2 = chi2_interval(2)
        ax3 = self.fig.add_subplot(313)
        ax3.plot(nis_arr, lw=0.8, label=r'NIS')
        ax3.axhline(lo2, color='k', ls='--', lw=0.8)
        ax3.axhline(hi2, color='k', ls='--', lw=0.8,
                    label=r'$\chi^2_2$ 95% bounds')
        ax3.set_yscale('log')
        ax3.set_xlabel('step $k$'); ax3.set_ylabel('NIS'); ax3.legend(fontsize=7)

        self.fig.tight_layout()
        self.canvas.draw()

    def on_close(self, event):
        self.timer.Stop()
        self.Destroy()


if __name__ == '__main__':
    app = wx.App()
    frame = KalmanFrame(None, 'Kalman Examples: Bouncing Ball Tracker')
    frame.Show()
    app.MainLoop()
