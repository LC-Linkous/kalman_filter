##--------------------------------------------------------------------\
#   kalman_filter
#   './src/run_gui.py'
#   Entry point: wxPython GUI for the bouncing-ball Kalman demo.
#   Dark canvas-first restyle: the simulation canvas dominates, the
#   covariance is drawn as a translucent uncertainty cloud, and a live
#   NIS strip sits under the canvas so consistency is something you
#   watch, not a tab you visit. Sidebar controls are grouped into
#   collapsible panes; gravity is a fixed constant (in a bounded box it
#   changes bounce rhythm, not estimation behavior).
#
#   Layout:
#     [ Notebook: Live Simulation | Analysis ]  [ sidebar ]
#     [        NIS strip (live, chi-squared band)          ]
#     [   Start / Pause / Reset buttons + status           ]
#
#   On the canvas, per ball:
#     small colored dots      = noisy measurements (what the filter sees)
#     solid colored line      = ground-truth trail (toggleable)
#     dashed white line       = Kalman estimate trail
#     translucent ellipse     = 2-sigma position covariance
#     short white arrow       = estimated velocity (never measured!)
#
#   Physics is y-up; the flip to screen coordinates happens once, at
#   draw time, exactly as before.
#
#   Run:  python run_gui.py
##--------------------------------------------------------------------\

import numpy as np
import wx
import matplotlib
matplotlib.use('WXAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

from ball import Ball
from tracked_ball import TrackedBall
from metrics import chi2_interval

# saturated palette tuned for the dark canvas
BALL_COLORS = [(93, 202, 165), (240, 153, 123), (133, 183, 235),
               (237, 147, 177), (239, 159, 39), (151, 196, 89),
               (175, 169, 236), (240, 149, 149), (94, 202, 202),
               (211, 209, 199)]
TRAIL_LEN = 120
NIS_LEN = 200
TIMER_MS = 33
GRAVITY = 0.50            # fixed; was a slider, but in a bounded box it
                          # only changes bounce rhythm, not estimation

COL_BG = wx.Colour(16, 20, 24)
COL_GRID = wx.Colour(34, 40, 46)
COL_EST = wx.Colour(235, 235, 240)
COL_OK = wx.Colour(29, 158, 117)
COL_BAD = wx.Colour(226, 75, 74)


class SimPanel(wx.Panel):
    """Dark canvas. Physics is y-up; the flip to screen coordinates
    happens HERE and only here."""

    def __init__(self, parent, frame):
        super().__init__(parent)
        self.frame = frame
        self.SetBackgroundColour(COL_BG)
        self.SetDoubleBuffered(True)
        self.Bind(wx.EVT_PAINT, self.on_paint)

    def flip(self, y):
        return self.GetClientSize().height - y

    def on_paint(self, event):
        dc = wx.GCDC(wx.PaintDC(self))
        dc.SetBackground(wx.Brush(COL_BG))
        dc.Clear()
        w, h = self.GetClientSize()
        s = self.frame.settings

        # faint grid so motion reads against something
        dc.SetPen(wx.Pen(COL_GRID, 1))
        for gx in range(0, w, 80):
            dc.DrawLine(gx, 0, gx, h)
        for gy in range(0, h, 80):
            dc.DrawLine(0, gy, w, gy)

        gc = dc.GetGraphicsContext()

        for tb in self.frame.tracked_balls:
            r, g, b = tb.ball.color
            color = wx.Colour(r, g, b)

            # measurement dots (what the filter sees)
            if s['show_measurements']:
                meas_col = wx.Colour(r, g, b, 150)
                dc.SetPen(wx.Pen(meas_col, 1))
                dc.SetBrush(wx.Brush(meas_col))
                for z in tb.measurements[-TRAIL_LEN::3]:
                    dc.DrawCircle(int(z[0]), int(self.flip(z[1])), 2)

            # truth trail as a solid colored line
            if s['show_truth_trail'] and len(tb.truth_states) > 1:
                pts = tb.truth_states[-TRAIL_LEN:]
                dc.SetPen(wx.Pen(color, 2))
                dc.DrawLines([(int(p[0]), int(self.flip(p[1])))
                              for p in pts])

            # estimate trail as a dashed light line
            if s['show_estimates'] and len(tb.estimates) > 1:
                pts = tb.estimates[-TRAIL_LEN:]
                pen = wx.Pen(COL_EST, 2)
                pen.SetStyle(wx.PENSTYLE_SHORT_DASH)
                dc.SetPen(pen)
                dc.DrawLines([(int(p[0]), int(self.flip(p[1])))
                              for p in pts])

            # truth marker: outline circle, as before
            if s['show_truth']:
                dc.SetPen(wx.Pen(color, 2))
                dc.SetBrush(wx.TRANSPARENT_BRUSH)
                dc.DrawCircle(int(tb.ball.x), int(self.flip(tb.ball.y)),
                              int(tb.ball.radius))

            if s['show_estimates'] and tb.kf is not None:
                ex = tb.kf.x[0, 0]
                ey = self.flip(tb.kf.x[1, 0])

                # 2-sigma covariance ellipse as a translucent FILL:
                # an uncertainty cloud, not another outline
                Ppos = tb.kf.P[:2, :2]
                vals, vecs = np.linalg.eigh(Ppos)
                vals = np.maximum(vals, 1e-9)
                # eigh: ascending; major axis is column 1
                ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
                rx = 2.0 * np.sqrt(vals[1])
                ry = 2.0 * np.sqrt(vals[0])
                gc.PushState()
                gc.Translate(ex, ey)
                # y-flip means screen rotation is the negative angle
                gc.Rotate(np.radians(-ang))
                gc.SetPen(wx.TRANSPARENT_PEN)
                gc.SetBrush(wx.Brush(wx.Colour(r, g, b, 55)))
                gc.DrawEllipse(-rx, -ry, 2 * rx, 2 * ry)
                gc.PopState()

                # velocity arrow: the state the filter never measures.
                # velocities are px/step, so scale up to be visible.
                vx = tb.kf.x[2, 0]
                vy = tb.kf.x[3, 0]
                scale = 6.0
                dc.SetPen(wx.Pen(COL_EST, 2))
                dc.DrawLine(int(ex), int(ey),
                            int(ex + vx * scale), int(ey - vy * scale))

                # estimate marker: ring + crosshair (light, for dark bg)
                dc.SetBrush(wx.TRANSPARENT_BRUSH)
                dc.SetPen(wx.Pen(COL_EST, 2))
                rr = int(tb.ball.radius + 4)
                dc.DrawCircle(int(ex), int(ey), rr)
                dc.DrawLine(int(ex - rr), int(ey), int(ex + rr), int(ey))
                dc.DrawLine(int(ex), int(ey - rr), int(ex), int(ey + rr))


class NisStrip(wx.Panel):
    """Slim live NIS plot for ball 1 with the 95% chi-squared band
    shaded. Green while consistent, red when the filter's model is
    being violated (crank the Q multiplier or bounce a lot to see it)."""

    def __init__(self, parent, frame):
        super().__init__(parent)
        self.frame = frame
        self.SetBackgroundColour(COL_BG)
        self.SetDoubleBuffered(True)
        self.SetMinSize(wx.Size(-1, 56))
        self.lo, self.hi = chi2_interval(2)
        self.Bind(wx.EVT_PAINT, self.on_paint)

    def on_paint(self, event):
        dc = wx.GCDC(wx.PaintDC(self))
        dc.SetBackground(wx.Brush(COL_BG))
        dc.Clear()
        w, h = self.GetClientSize()

        tbs = self.frame.tracked_balls
        vals = []
        if tbs:
            vals = [v for v in tbs[0].nis_history[-NIS_LEN:]
                    if np.isfinite(v)]

        vmax = max(self.hi * 1.6, max(vals) if vals else 1.0)

        def ymap(v):
            return int(h - (min(v, vmax) / vmax) * (h - 10) - 5)

        # shaded 95% chi-squared band
        dc.SetPen(wx.TRANSPARENT_PEN)
        dc.SetBrush(wx.Brush(wx.Colour(COL_OK.Red(), COL_OK.Green(),
                                       COL_OK.Blue(), 40)))
        dc.DrawRectangle(0, ymap(self.hi), w, ymap(self.lo) - ymap(self.hi))

        if len(vals) > 1:
            cur_ok = self.lo <= vals[-1] <= self.hi
            dc.SetPen(wx.Pen(COL_OK if cur_ok else COL_BAD, 2))
            step = w / (NIS_LEN - 1)
            x0 = w - step * (len(vals) - 1)
            dc.DrawLines([(int(x0 + i * step), ymap(v))
                          for i, v in enumerate(vals)])

            dc.SetTextForeground(COL_OK if cur_ok else COL_BAD)
            dc.SetFont(wx.Font(9, wx.FONTFAMILY_TELETYPE,
                               wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
            state = "consistent" if cur_ok else "out of bounds"
            dc.DrawText(f"NIS {vals[-1]:5.1f}  {state}  (ball 1)", 8, 4)


class KalmanFrame(wx.Frame):
    def __init__(self, parent, title):
        super().__init__(parent, title=title, size=(1250, 720))

        self.settings = {
            'n_balls': 3,
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
        self.sidebar = wx.Panel(panel)
        self.sidebar.SetMinSize(wx.Size(240, -1))
        sb = wx.BoxSizer(wx.VERTICAL)

        def make_group(title, collapsed=False):
            pane = wx.CollapsiblePane(self.sidebar, label=title,
                                      style=wx.CP_DEFAULT_STYLE)
            pane.Collapse(collapsed)
            inner = wx.BoxSizer(wx.VERTICAL)
            pane.GetPane().SetSizer(inner)
            pane.Bind(wx.EVT_COLLAPSIBLEPANE_CHANGED, self.on_pane_toggle)
            sb.Add(pane, 0, wx.EXPAND | wx.ALL, 4)
            return pane.GetPane(), inner

        def add_slider(parent, sizer, label_fmt, key, lo, hi, value,
                       scale=1.0):
            """Slider storing value*scale -> settings[key] (live)."""
            label = wx.StaticText(parent, label=label_fmt.format(value))
            slider = wx.Slider(parent, value=int(round(value / scale)),
                               minValue=lo, maxValue=hi)

            def on_slide(evt, k=key, lbl=label, fmt=label_fmt, sc=scale,
                         sld=slider):
                val = sld.GetValue() * sc
                self.settings[k] = val
                lbl.SetLabel(fmt.format(val))
                self.apply_tuning()
            slider.Bind(wx.EVT_SLIDER, on_slide)
            sizer.Add(label, 0, wx.LEFT | wx.TOP, 4)
            sizer.Add(slider, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 4)
            return slider

        # -- World (collapsible) --
        world, wsz = make_group("World")
        wsz.Add(wx.StaticText(world, label="Number of balls (on Reset)"),
                0, wx.LEFT | wx.TOP, 4)
        self.n_spin = wx.SpinCtrl(world, value="3", min=1, max=10)
        wsz.Add(self.n_spin, 0, wx.LEFT, 4)

        self.same_start_check = wx.CheckBox(
            world, label="Same start point && launch (on Reset)")
        self.same_start_check.Bind(
            wx.EVT_CHECKBOX,
            lambda e: self.settings.update(
                same_start=self.same_start_check.GetValue()))
        wsz.Add(self.same_start_check, 0, wx.ALL, 4)

        add_slider(world, wsz, "Bounciness: {:.2f}", 'restitution',
                   50, 100, self.settings['restitution'], scale=0.01)
        add_slider(world, wsz, "Ball radius: {:.0f} px (on Reset)",
                   'radius', 5, 30, self.settings['radius'])
        add_slider(world, wsz, "Launch speed: {:.0f} px/step (on Reset)",
                   'speed', 1, 20, self.settings['speed'])

        # -- Noise (collapsible) --
        noise, nsz = make_group("Noise")
        add_slider(noise, nsz, "Measurement noise \u03c3: {:.1f} px",
                   'r_std', 0, 200, self.settings['r_std'], scale=0.1)
        add_slider(noise, nsz, "Process noise q: {:.2f}", 'q',
                   0, 200, self.settings['q'], scale=0.01)

        # -- Filter (collapsed by default) --
        filt, fsz = make_group("Filter", collapsed=True)
        add_slider(filt, fsz, "Filter Q multiplier: {:.2f}x",
                   'q_filter_scale', 5, 500,
                   self.settings['q_filter_scale'], scale=0.01)
        self.bounce_check = wx.CheckBox(filt,
                                        label="Tell filter about bounces")
        self.bounce_check.Bind(
            wx.EVT_CHECKBOX,
            lambda e: self.settings.update(
                tell_bounces=self.bounce_check.GetValue()))
        fsz.Add(self.bounce_check, 0, wx.ALL, 4)
        fsz.Add(wx.StaticText(
            filt, label="Mistune Q to watch the NIS\nstrip leave its band."),
            0, wx.LEFT | wx.BOTTOM, 4)

        # -- Display (collapsed by default) --
        disp, dsz = make_group("Display", collapsed=True)
        for key, text in [('show_truth', "Show ground truth (outline)"),
                          ('show_truth_trail', "Show truth trail"),
                          ('show_measurements', "Show measurement trail"),
                          ('show_estimates', "Show Kalman estimates")]:
            cb = wx.CheckBox(disp, label=text)
            cb.SetValue(self.settings[key])
            cb.Bind(wx.EVT_CHECKBOX,
                    lambda e, k=key, c=cb: self.settings.update(
                        {k: c.GetValue()}))
            dsz.Add(cb, 0, wx.LEFT | wx.TOP, 4)

        self.sidebar.SetSizer(sb)

        # ---------------- NIS strip + buttons (bottom) ----------------
        self.nis_strip = NisStrip(panel, self)

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
        top.Add(self.sidebar, 0, wx.EXPAND | wx.ALL, 5)
        main = wx.BoxSizer(wx.VERTICAL)
        main.Add(top, 1, wx.EXPAND)
        main.Add(self.nis_strip, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 10)
        main.Add(btn_panel, 0, wx.EXPAND)
        panel.SetSizer(main)
        self.panel = panel

        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_tick, self.timer)
        self.Bind(wx.EVT_CLOSE, self.on_close)

    def on_pane_toggle(self, event):
        self.panel.Layout()

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
        self.nis_strip.Refresh()
        self.status.SetLabel("Reset - press Start")

    def on_tick(self, event):
        if not self.is_running:
            return
        w, h = self.sim_panel.GetClientSize()
        s = self.settings
        for tb in self.tracked_balls:
            tb.step(GRAVITY, s['restitution'], w, h,
                    tell_filter_about_bounces=s['tell_bounces'])
        self.sim_panel.Refresh()
        self.nis_strip.Refresh()
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