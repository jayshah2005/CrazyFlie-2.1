"""
Poster figures for AI Day: the learned controller on the Crazyflie 2.1 Brushless.

Every figure is drawn at its final print size for an A0 portrait poster
(841 x 1189 mm, three columns), so font sizes in pt are the printed sizes:
labels >= 20 pt, box titles 26 pt. Each figure is written as SVG (vector,
editable text, for print) and PNG (300 dpi) under poster/figures/.

Facts and numbers come from the repo: rl_trigger.py (MissionProfile, safety
logic), l2f/README.md (flight results, model, fixes) and
l2f/firmware/build_firmware.sh (rates, inputs).

Usage: python poster/make_figures.py [--flight logs/rl_trigger_flight_<stamp>.csv]
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import run_log  # noqa: E402

log = run_log.setup(__file__)
OUT = Path(__file__).resolve().parent / 'figures'
MM = 1 / 25.4  # inches per mm

# ---- Design tokens (validated: blue/orange/aqua pass all-pairs CVD checks on white)
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS, BAND = '#e1e0d9', '#c3c2b7', '#f3f2ef'
SIM, REAL, THIRD = '#2a78d6', '#eb6834', '#1baf7a'         # simulation / real drone / 3rd series
SIM_TINT, REAL_TINT, NEUTRAL_TINT = '#eaf2fc', '#fdeee7', '#f6f5f2'
GOOD, WARN, CRIT = '#0ca30c', '#fab219', '#d03b3b'          # status: only with a text label
GOOD_TINT, WARN_TINT, CRIT_TINT = '#e7f6e7', '#fff4dc', '#fbe9e9'

LW = 2.5          # pt, box outlines and arrows
T_TITLE, T_BODY, T_LABEL = 26, 22, 21  # pt

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': T_LABEL, 'text.color': INK,
    'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'axes.linewidth': 1.5,
    'xtick.color': INK2, 'ytick.color': INK2, 'xtick.major.width': 1.5,
    'ytick.major.width': 1.5, 'xtick.major.size': 6, 'ytick.major.size': 6,
    'svg.fonttype': 'none', 'savefig.facecolor': 'white',
})


# ---- Helpers ---------------------------------------------------------------

def canvas(w, h):
    """Figure of w x h mm with one invisible axes whose data units are mm."""
    fig = plt.figure(figsize=(w * MM, h * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis('off')
    return fig, ax


def axes_mm(fig, w, h, left, bottom, width, height):
    return fig.add_axes([left / w, bottom / h, width / w, height / h])


def pt(v):
    return v * 25.4 / 72  # pt -> mm


def box(ax, cx, cy, w, h, title, body=None, edge=INK2, fill='white', lw=LW,
        title_size=T_TITLE, body_size=T_BODY):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, linewidth=lw,
                                boxstyle='round,pad=0,rounding_size=4',
                                edgecolor=edge, facecolor=fill, zorder=2))
    if body is None:
        ax.text(cx, cy, title, ha='center', va='center', fontsize=title_size,
                weight='bold', color=INK, zorder=3)
        return
    n = body.count('\n') + 1
    th = pt(title_size) * 1.2 * (title.count('\n') + 1)
    bh, gap = pt(body_size) * 1.3 * n, 2.5
    top = cy + (th + gap + bh) / 2
    ax.text(cx, top, title, ha='center', va='top', fontsize=title_size, weight='bold',
            color=INK, linespacing=1.2, zorder=3)
    ax.text(cx, top - th - gap, body, ha='center', va='top', fontsize=body_size,
            color=INK2, linespacing=1.3, zorder=3)


def arrow(ax, p, q, color=INK2, lw=LW, rad=0.0, scale=26):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=scale, lw=lw,
                                 color=color, connectionstyle=f'arc3,rad={rad}',
                                 shrinkA=0, shrinkB=0, zorder=1))


def swatch(ax, x, y, edge, fill, label):
    ax.add_patch(FancyBboxPatch((x, y - 4), 12, 8, boxstyle='round,pad=0,rounding_size=1.5',
                                linewidth=LW, edgecolor=edge, facecolor=fill))
    ax.text(x + 16, y, label, va='center', fontsize=T_LABEL, color=INK2)


def icon(ax, x, y, kind, color, r=5.5):
    """Status icon drawn as shapes (Arial has no check or cross glyphs)."""
    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor='none', zorder=4))
    k = dict(color=INK if kind == 'bang' else 'white', lw=3, solid_capstyle='round', zorder=5)
    if kind == 'check':
        ax.plot([x - 0.5 * r, x - 0.1 * r, x + 0.5 * r], [y, y - 0.4 * r, y + 0.4 * r], **k)
    elif kind == 'cross':
        d = 0.38 * r
        ax.plot([x - d, x + d], [y - d, y + d], **k)
        ax.plot([x - d, x + d], [y + d, y - d], **k)
    else:
        ax.plot([x, x], [y - 0.05 * r, y + 0.5 * r], **k)
        ax.plot([x], [y - 0.45 * r], 'o', ms=3.5, color=k['color'], zorder=5)


def clean_axes(ax, grid_axis='y'):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1.2)
    ax.set_axisbelow(True)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kw in (('svg', {}), ('png', {'dpi': 300})):
        fig.savefig(OUT / f'{name}.{ext}', **kw)
    w, h = (v / MM for v in fig.get_size_inches())
    log.info('%s: %.0f x %.0f mm (svg, png)', name, w, h)
    plt.close(fig)


# ---- Figure 1: pipeline ----------------------------------------------------

def fig1_pipeline():
    W, H = 780, 185
    fig, ax = canvas(W, H)
    stages = [
        ('1  Measure', 'Fly the stock PID,\nlog motor speeds', REAL, REAL_TINT),
        ('2  Model', 'Brushless simulator:\nmass, thrust,\nmotor lag', SIM, SIM_TINT),
        ('3  Train', 'Reinforcement\nlearning (TD3),\n2 million steps', SIM, SIM_TINT),
        ('4  Select', 'Score every 100k\ncheckpoint in a\nsimulated mission', SIM, SIM_TINT),
        ('5  Deploy', 'Compile the network\ninto the firmware,\nflash the drone', REAL, REAL_TINT),
        ('6  Fly', 'Policy runs onboard\nat 100 Hz, no PID', REAL, REAL_TINT),
    ]
    bw, gap, cy, bh = 112, 19.2, 110, 100
    xs = [6 + bw / 2 + i * (bw + gap) for i in range(len(stages))]
    for cx, (title, body, edge, fill) in zip(xs, stages):
        box(ax, cx, cy, bw, bh, title, body, edge=edge, fill=fill)
    for a, b in zip(xs, xs[1:]):
        arrow(ax, (a + bw / 2 + 2, cy), (b - bw / 2 - 2, cy))
    # feedback loop: flight back to the model
    y_loop, x6, x2 = 36, xs[5], xs[1]
    ax.plot([x6, x6, x2], [cy - bh / 2 - 2, y_loop, y_loop], color=INK2, lw=LW,
            solid_joinstyle='round', zorder=1)
    arrow(ax, (x2, y_loop), (x2, cy - bh / 2 - 2))
    ax.text((x2 + x6) / 2, y_loop - 7, 'Real flights reveal sim-to-real gaps  →  '
            'improve the simulator and retrain', ha='center', va='top',
            fontsize=T_BODY, color=INK2)
    swatch(ax, 8, 175, SIM, SIM_TINT, 'In simulation')
    swatch(ax, 150, 175, REAL, REAL_TINT, 'On the real drone')
    save(fig, 'fig1_pipeline')


# ---- Figure 2: onboard control loop ----------------------------------------

def fig2_control_loop():
    W, H = 515, 330
    fig, ax = canvas(W, H)
    ax.add_patch(FancyBboxPatch((5, 5), W - 10, 222, boxstyle='round,pad=0,rounding_size=6',
                                linewidth=LW, edgecolor=REAL, facecolor=REAL_TINT, zorder=0))
    ax.text(16, 218, 'Onboard the Crazyflie 2.1 Brushless (41 g)', va='top',
            fontsize=T_BODY, weight='bold', color=INK)
    bw, bh = 140, 64
    c1, c2, c3 = 90, 257, 425
    top, bot = 145, 52
    box(ax, c1, top, bw, bh, 'Sensors', 'Lighthouse, IMU,\nflow deck')
    box(ax, c2, top, bw, bh, 'State estimate', 'Kalman filter')
    box(ax, c3, top, bw, bh, 'Neural network', 'policy · 100 Hz', edge=SIM, lw=4)
    box(ax, c3, bot, bw, bh, 'RPM → motor', 'battery-\ncompensated')
    box(ax, c2, bot, bw, bh, 'Motors', 'DShot · 1 kHz')
    box(ax, c1, bot, bw, bh, 'Drone moves', 'physics')
    s = bw / 2 + 3
    arrow(ax, (c1 + s, top), (c2 - s, top))
    arrow(ax, (c2 + s, top), (c3 - s, top))
    arrow(ax, (c3, top - bh / 2 - 3), (c3, bot + bh / 2 + 3))
    arrow(ax, (c3 - s, bot), (c2 + s, bot))
    arrow(ax, (c2 - s, bot), (c1 + s, bot))
    arrow(ax, (c1, bot + bh / 2 + 3), (c1, top - bh / 2 - 3))
    ax.text(c3 - 6, (top + bot) / 2, '4 motor\nspeeds', ha='right', va='center',
            fontsize=T_LABEL, color=INK2, linespacing=1.2)
    ax.text((c2 + c3) / 2, top + 6, '149\ninputs', ha='center', va='bottom',
            fontsize=T_LABEL, color=INK2, linespacing=1.2)
    # laptop
    box(ax, c3, 296, 170, 58, 'Laptop', 'mission script, safety', fill=NEUTRAL_TINT)
    arrow(ax, (c3, 296 - 29 - 3), (c3, top + bh / 2 + 3))
    ax.text(c3 - 10, 248, 'radio · 50 Hz\nmoving target:\nposition + velocity',
            ha='right', va='center', fontsize=T_LABEL, color=INK2, linespacing=1.2)
    save(fig, 'fig3_control_loop')


# ---- Control levels: cascade vs end-to-end ---------------------------------

def fig_control_levels():
    W, H = 250, 302
    fig, ax = canvas(W, H)
    lx, rx, bw, bh = 61, 189, 112, 30
    ax.text(lx, H - 8, 'Stock firmware:\ntuned PID cascade', ha='center', va='top',
            fontsize=T_BODY, weight='bold', linespacing=1.15)
    ax.text(rx, H - 8, 'Ours: one\nneural network', ha='center', va='top',
            fontsize=T_BODY, weight='bold', linespacing=1.15)
    y_target, y_motor = 255, 28
    stages = ['Position', 'Velocity', 'Attitude', 'Body rate', 'Motor mixer']
    ys = [215 - i * 37 for i in range(len(stages))]
    for cx in (lx, rx):
        box(ax, cx, y_target, bw, bh, 'Target', title_size=T_LABEL,
            fill=NEUTRAL_TINT)
        box(ax, cx, y_motor, bw, bh, '4 motors', title_size=T_LABEL, edge=REAL,
            fill=REAL_TINT)
    prev = y_target
    for y, name in zip(ys, stages):
        arrow(ax, (lx, prev - bh / 2 - 2), (lx, y + bh / 2 + 2))
        box(ax, lx, y, bw, bh, name, title_size=T_LABEL)
        prev = y
    arrow(ax, (lx, prev - bh / 2 - 2), (lx, y_motor + bh / 2 + 2))
    top, bot = ys[0] + bh / 2, ys[-1] - bh / 2
    arrow(ax, (rx, y_target - bh / 2 - 2), (rx, top + 2))
    box(ax, rx, (top + bot) / 2, bw, top - bot, 'Neural\nnetwork',
        'outputs motor\nspeeds (RPM)\ndirectly', edge=SIM, fill=SIM_TINT, lw=4)
    arrow(ax, (rx, bot - 2), (rx, y_motor + bh / 2 + 2))
    ax.plot([W / 2, W / 2], [y_motor - 10, H - 50], color=GRID, lw=1.5)
    save(fig, 'fig2_control_levels')


# ---- The network ------------------------------------------------------------

def fig_network():
    W, H = 515, 250
    fig, ax = canvas(W, H)
    inputs = [('Past 32 motor commands', 128), ('Orientation (3\u00d73)', 9),
              ('Position error', 3), ('Velocity error', 3), ('Angular velocity', 3),
              ('Position-error integral', 3)]
    ib_w, ib_h, gap, top = 178, 25, 6, 200
    in_ys = [top - ib_h / 2 - i * (ib_h + gap) for i in range(len(inputs))]
    x_h1, x_h2, x_out, r = 290, 365, 440, 4.5
    h_ys = np.linspace(top - 12, in_ys[-1] - 2, 9)
    o_ys = np.linspace(in_ys[1], in_ys[4], 4)
    edge = dict(color='#d9d7cf', lw=0.7, zorder=0)
    for y in in_ys:
        for hy in h_ys:
            ax.plot([4 + ib_w, x_h1], [y, hy], **edge)
    for y1 in h_ys:
        for y2 in h_ys:
            ax.plot([x_h1, x_h2], [y1, y2], **edge)
        for yo in o_ys:
            ax.plot([x_h2, x_out], [y1, yo], **edge)
    for (label, n), y in zip(inputs, in_ys):
        ax.add_patch(FancyBboxPatch((4, y - ib_h / 2), ib_w, ib_h, linewidth=2,
                                    edgecolor=AXIS, facecolor=NEUTRAL_TINT, zorder=2,
                                    boxstyle='round,pad=0,rounding_size=3'))
        ax.text(10, y, label, va='center', fontsize=T_LABEL - 2, zorder=3)
        ax.text(ib_w - 2, y, str(n), ha='right', va='center', fontsize=T_LABEL - 2,
                weight='bold', zorder=3)
    for x in (x_h1, x_h2):
        for y in h_ys:
            ax.add_patch(Circle((x, y), r, facecolor='white', edgecolor=SIM, lw=2.5,
                                zorder=3))
    for i, y in enumerate(o_ys):
        ax.add_patch(Circle((x_out, y), r + 1.5, facecolor=SIM, edgecolor='white', lw=2,
                            zorder=3))
        ax.text(x_out + 11, y, f'Motor {i + 1}', va='center', fontsize=T_LABEL - 2)
    hy = top + 22
    for x, label in ((4 + ib_w / 2, '149 inputs'), (x_h1, '64'), (x_h2, '64'),
                     (x_out + 20, '4 outputs')):
        ax.text(x, hy, label, ha='center', va='bottom', fontsize=T_LABEL, weight='bold')
    ax.text((x_h1 + x_h2) / 2, hy + 12, 'hidden neurons (tanh)', ha='center',
            va='bottom', fontsize=T_LABEL - 2, color=INK2)
    ax.text(W / 2, 8, 'Motor speeds (RPM) every 10 ms  \u00b7  one pass takes ~0.7 ms on '
            'the drone\u2019s microcontroller', ha='center', va='bottom',
            fontsize=T_LABEL - 2, color=INK2)
    save(fig, 'fig4_network')


# ---- Figure 4: the mission --------------------------------------------------

class Mission:
    """Same profile as rl_trigger.MissionProfile with the flown 1 m settings."""
    H, R, P, v_up, v_down = 1.0, 0.6, 10.0, 0.25, 0.25
    phases = [('sit', 0.3), ('Take-off', 1.0 / 0.25), ('Hover', 2.0), ('Out', 2.0),
              ('Circle', 10.0), ('Back', 2.0), ('hover2', 1.0), ('Land', 1.02 / 0.25),
              ('ground', 1.5)]

    def at(self, t):
        H, R = self.H, self.R
        for name, d in self.phases:
            if t < d:
                break
            t -= d
        else:
            return -0.02, 0.0, 0.0
        if name == 'sit':
            return 0.0, 0.0, 0.0
        if name == 'Take-off':
            return self.v_up * t, 0.0, 0.0
        if name in ('Hover', 'hover2'):
            return H, 0.0, 0.0
        if name == 'Out':
            return H, R * t / 2.0, 0.0
        if name == 'Circle':
            a = 2 * math.pi * t / self.P
            return H, R * math.cos(a), R * math.sin(a)
        if name == 'Back':
            return H, R * (1 - t / 2.0), 0.0
        if name == 'Land':
            return H - self.v_down * t, 0.0, 0.0
        return -0.02, 0.0, 0.0


def fig4_mission():
    W, H = 515, 205
    m = Mission()
    total = sum(d for _, d in m.phases)
    t = np.linspace(0, total, 3000)
    z, x, y = np.array([m.at(v) for v in t]).T
    z = np.maximum(z, 0)
    dist = np.hypot(x, y)

    fig = plt.figure(figsize=(W * MM, H * MM))
    ax = axes_mm(fig, W, H, 32, 30, 300, 140)
    t0 = 0.0
    for i, (name, d) in enumerate(m.phases):
        if name in ('Take-off', 'Out', 'Back', 'Land'):
            ax.axvspan(t0, t0 + d, color=BAND, lw=0, zorder=0)
        if name[0].isupper():
            ax.text(t0 + d / 2, 1.33, name, ha='center', va='center', fontsize=T_LABEL - 1,
                    color=INK2)
        t0 += d
    ax.plot(t, z, color=SIM, lw=3.5, solid_capstyle='round', label='Height')
    ax.plot(t, dist, color=REAL, lw=3.5, solid_capstyle='round',
            label='Distance from take-off point')
    ax.set_xlim(0, total)
    ax.set_ylim(0, 1.42)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Target (m)')
    clean_axes(ax)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
              fontsize=T_LABEL, handlelength=1.6, borderaxespad=0.6)

    # top view
    ax2 = axes_mm(fig, W, H, 375, 30, 130, 140)
    a = np.linspace(0, 2 * math.pi, 400)
    ax2.plot([0, m.R], [0, 0], color=REAL, lw=3.5, solid_capstyle='round')
    ax2.plot(m.R * np.cos(a), m.R * np.sin(a), color=REAL, lw=3.5)
    for ang in (1.2, 1.2 + math.pi):
        ax2.add_patch(FancyArrowPatch((m.R * math.cos(ang), m.R * math.sin(ang)),
                                      (m.R * math.cos(ang + 0.35), m.R * math.sin(ang + 0.35)),
                                      arrowstyle='-|>', mutation_scale=42, color=REAL, lw=0,
                                      zorder=4))
    ax2.plot(0, 0, 'o', ms=13, color=INK, mec='white', mew=2.5, zorder=5)
    ax2.text(0, -0.1, 'take-off\n& landing', ha='center', va='top', fontsize=T_LABEL - 1,
             color=INK2, linespacing=1.15)
    ax2.text(-m.R * 0.45, m.R * 0.5, 'r = 0.6 m', ha='center', fontsize=T_LABEL - 1,
             color=INK2)
    ax2.set_aspect('equal')
    ax2.set_xlim(-0.75, 0.75)
    ax2.set_ylim(-0.75, 0.75)
    ax2.set_xticks([-0.6, 0, 0.6])
    ax2.set_yticks([-0.6, 0, 0.6])
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)', labelpad=2)
    clean_axes(ax2, grid_axis='both')
    fig.text(440 / W, (30 + 140 + 9) / H, 'Top view', ha='center', va='center',
             fontsize=T_LABEL, color=INK2)
    save(fig, 'fig5_mission')


# ---- Figure 5: real-flight results -----------------------------------------

def fig5_results():
    W, H = 250, 205
    # l2f/README.md, flown policy I1, --ki 0. 0.6 m row: range over repeated flights.
    groups = ['0.6 m', '0.75 m', '1.0 m', '1.5 m']
    radii = ['r 0.3 m', 'r 0.6 m', 'r 0.6 m', 'r 0.6 m']
    series = [('Hover', SIM, [(3.8, 4.0), 4.4, 5.0, 3.2]),
              ('Circle', REAL, [(5.7, 6.4), 8.4, 6.7, 10.9]),
              ('Landing', THIRD, [(3.4, 3.8), 3.9, 4.1, 5.8])]
    fig = plt.figure(figsize=(W * MM, H * MM))
    ax = axes_mm(fig, W, H, 30, 38, 212, 140)
    bw = 0.27
    xg = np.arange(len(groups))
    for k, (name, color, vals) in enumerate(series):
        xs = xg + (k - 1) * bw
        mids = [(v[0] + v[1]) / 2 if isinstance(v, tuple) else v for v in vals]
        ax.bar(xs, mids, width=bw, color=color, edgecolor='white', linewidth=2, label=name)
        for xi, v in zip(xs, vals):
            if isinstance(v, tuple):
                ax.plot([xi, xi], v, color=INK, lw=2)
                ax.plot([xi - 0.06, xi + 0.06], [v[0]] * 2, color=INK, lw=2)
                ax.plot([xi - 0.06, xi + 0.06], [v[1]] * 2, color=INK, lw=2)
    ax.text(3, 11.3, '36 cm Lighthouse\njump during\nthis circle', ha='right', va='bottom',
            fontsize=T_LABEL - 3, color=INK2, linespacing=1.15)
    ax.set_xticks(xg, [f'{g}\n{r}' for g, r in zip(groups, radii)], fontsize=T_LABEL)
    ax.set_xlabel('Flight height and circle radius', labelpad=6)
    ax.set_ylabel('Tracking error (cm)')
    ax.set_ylim(0, 14)
    ax.set_yticks([0, 4, 8, 12])
    ax.tick_params(axis='x', length=0)
    clean_axes(ax)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
              fontsize=T_LABEL, handlelength=1.0, columnspacing=1.2, borderaxespad=0.5)
    save(fig, 'fig6_results')


# ---- Real flight vs target (from an rl_trigger.py log) ----------------------

def load_flight(path):
    """Policy-active samples of an rl_trigger.py flight CSV: t, position, target."""
    with open(path) as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith('#')))
    rows = [r for r in rows if r['policy_active'] == '1' and r['err_x'] not in ('', 'None')]
    if len(rows) < 50:
        raise SystemExit(f'{path}: only {len(rows)} samples with the policy in control')
    t = np.array([float(r['drone_ms']) for r in rows]) / 1000.0
    pos = np.array([[float(r[k]) for k in 'xyz'] for r in rows])
    err = np.array([[float(r[f'err_{k}']) for k in 'xyz'] for r in rows])
    origin = pos[0].copy()
    origin[2] = pos[:, 2].min()
    return t - t[0], pos - origin, pos + err - origin  # err = target - position


def fig_flight(path):
    t, pos, tgt = load_flight(path)
    err_cm = 100 * np.linalg.norm(pos - tgt, axis=1)
    W, H = 515, 205
    fig = plt.figure(figsize=(W * MM, H * MM))
    ax = axes_mm(fig, W, H, 32, 30, 300, 140)
    ax.plot(t, pos[:, 2], color=REAL, lw=3, label='Flown (real drone)', zorder=2)
    ax.plot(t, tgt[:, 2], color=INK, lw=2, label='Target', zorder=3)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Height (m)')
    clean_axes(ax)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
              fontsize=T_LABEL, handlelength=1.6, borderaxespad=0.6)
    ax2 = axes_mm(fig, W, H, 375, 30, 130, 140)
    ax2.plot(pos[:, 0], pos[:, 1], color=REAL, lw=3, zorder=2)
    ax2.plot(tgt[:, 0], tgt[:, 1], color=INK, lw=2, zorder=3)
    ax2.set_aspect('equal', adjustable='datalim')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)', labelpad=2)
    clean_axes(ax2, grid_axis='both')
    fig.text(440 / W, (30 + 140 + 9) / H, f'Top view \u00b7 mean error {err_cm.mean():.1f} cm',
             ha='center', va='center', fontsize=T_LABEL, color=INK2)
    log.info('flight %s: %.1f s under the policy, mean error %.1f cm, max %.1f cm',
             path, t[-1], err_cm.mean(), err_cm.max())
    save(fig, 'fig7_flight')


# ---- Figure 6: comparison with published controllers -----------------------

def fig6_comparison():
    W, H = 250, 150
    # Ours: l2f/README.md table (hover phase over the four missions).
    # Published: brushless-modelling paper (Busetto et al., ICRA 2026), hover.
    rows = ['Ours: learned,\nfull mission', 'Learned policy', 'PID', 'Mellinger']
    err = [(3.2, 5.0), 4.3, 2.8, 5.5]
    jit = [(0.047, 0.072), 0.020, 0.023, 0.039]
    fig = plt.figure(figsize=(W * MM, H * MM))
    y = np.array([3.0, 1.8, 0.9, 0.0])
    panels = [(axes_mm(fig, W, H, 72, 30, 80, 110), err, 'Hover error (cm)', (0, 7), '%.1f'),
              (axes_mm(fig, W, H, 165, 30, 80, 110), jit, 'Jitter (paper metric)',
               (0, 0.09), '%.3f')]
    for i, (ax, vals, xlabel, xlim, fmt) in enumerate(panels):
        for yi, v in zip(y, vals):
            if isinstance(v, tuple):
                ax.plot(v, [yi, yi], color=REAL, lw=6, solid_capstyle='round', zorder=3)
                ax.plot(v, [yi, yi], 'o', ms=12, color=REAL, mec='white', mew=2, zorder=4)
                ax.text(v[1], yi + 0.38, (fmt + '–' + fmt) % v, ha='right', va='bottom',
                        fontsize=T_LABEL - 3, color=INK)
            else:
                ax.plot([v], [yi], 'o', ms=12, color=INK2, mec='white', mew=2, zorder=4)
                ax.text(v, yi + 0.3, fmt % v, ha='center', va='bottom',
                        fontsize=T_LABEL - 3, color=INK2)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.5, 3.8)
        ax.set_xlabel(xlabel, fontsize=T_LABEL - 1)
        ax.set_yticks(y, rows if i == 0 else [])
        ax.tick_params(axis='y', length=0, labelsize=T_LABEL - 1)
        ax.tick_params(axis='x', labelsize=T_LABEL - 3)
        ax.locator_params(axis='x', nbins=3)
        clean_axes(ax, grid_axis='x')
        ax.spines['left'].set_visible(False)
    fig.text(4 / W, (30 + 110 * (1.8 + 0.5) / 4.3 + 12) / H, 'Same drone, published:',
             fontsize=T_LABEL - 3, color=MUTED, va='bottom')
    save(fig, 'fig8_comparison')


# ---- Figure 7: closing the sim-to-real gap ---------------------------------

def fig7_sim_to_real():
    rows = [
        ('Motors chattered between 0 and 100 %:\nreal gyro noise is 35× the paper’s',
         'Train with the measured noise and\npenalise angular acceleration', 'sim'),
        ('Policies collapsed when the position\nestimate arrived > 5 ms late',
         'Randomise a 0–12 ms estimate\ndelay during training', 'sim'),
        ('Lighthouse drop-outs make the position\njump by 10–37 cm',
         'Inject random position jumps\nduring training', 'sim'),
        ('Policy held a steady position offset',
         'Feed a position-error integral\nas 3 extra inputs', 'both'),
        ('No PID: the policy must take off\nand land by itself',
         'Simulate floor contact and ground\neffect; curriculum of missions', 'sim'),
        ('Radio task starved with the\npolicy at 500 Hz',
         'Run the policy at 100 Hz,\nthe simulator’s step', 'fw'),
        ('Two motors stopped in flight\n(shared DMA in the DShot driver)',
         'Write all four motor commands\nevery 1 kHz tick', 'fw'),
    ]
    W, row_h, head = 515, 37, 20
    H = head + row_h * len(rows) + 6
    fig, ax = canvas(W, H)
    xp, xf, xt = 4, 262, 470
    ax.text(xp, H - 6, 'Seen on the real drone', va='top', fontsize=T_BODY, weight='bold')
    ax.text(xf, H - 6, 'Fix', va='top', fontsize=T_BODY, weight='bold')
    tags = {'sim': [('Sim', SIM, SIM_TINT)], 'fw': [('Firmware', REAL, REAL_TINT)],
            'both': [('Sim', SIM, SIM_TINT), ('Firmware', REAL, REAL_TINT)]}
    for i, (problem, fix, tag) in enumerate(rows):
        yc = H - head - row_h * (i + 0.5)
        ax.plot([xp, W - 4], [yc + row_h / 2] * 2, color=GRID, lw=1.5)
        ax.text(xp, yc, problem, va='center', fontsize=T_LABEL - 1, color=INK,
                linespacing=1.2)
        arrow(ax, (xf - 26, yc), (xf - 8, yc), color=MUTED)
        ax.text(xf, yc, fix, va='center', fontsize=T_LABEL - 1, color=INK2, linespacing=1.2)
        tl = tags[tag]
        for j, (label, edge, fill) in enumerate(tl):
            ty = yc + (len(tl) - 1) * 6 - j * 12
            ax.add_patch(FancyBboxPatch((xt, ty - 5), 41, 10, linewidth=2, edgecolor=edge,
                                        facecolor=fill, boxstyle='round,pad=0,rounding_size=5'))
            ax.text(xt + 20.5, ty, label, ha='center', va='center', fontsize=T_LABEL - 5,
                    color=INK)
    save(fig, 'fig9_sim_to_real')


# ---- Figure 8: safety in mission mode --------------------------------------

def fig8_safety():
    W, H = 515, 238
    fig, ax = canvas(W, H)
    ty, ly = 182, 88
    box(ax, 72, ty, 130, 66, 'On the floor', 'armed, ready', fill=NEUTRAL_TINT)
    box(ax, 258, ty, 140, 66, 'Mission', 'policy follows\nthe moving target', edge=SIM,
        fill=SIM_TINT)
    box(ax, 450, ty, 120, 66, 'Landed', 'motors off', edge=GOOD, fill=GOOD_TINT)
    icon(ax, 450 - 60 + 10, ty + 33 - 10, 'check', GOOD)
    box(ax, 345, ly, 160, 66, 'Automatic landing', 'descend at 0.3 m/s', edge=WARN,
        fill=WARN_TINT)
    icon(ax, 345 - 80 + 10, ly + 33 - 10, 'bang', WARN)
    arrow(ax, (72 + 65 + 3, ty), (258 - 70 - 3, ty))
    ax.text(163, ty + 6, 'hold\nEnter', ha='center', va='bottom', fontsize=T_LABEL,
            color=INK2, linespacing=1.15)
    arrow(ax, (258 + 70 + 3, ty), (450 - 60 - 3, ty))
    ax.text(359, ty + 6, 'mission\ncomplete', ha='center', va='bottom', fontsize=T_LABEL,
            color=INK2, linespacing=1.15)
    arrow(ax, (258, ty - 33 - 3), (285, ly + 33 + 3))
    ax.text(250, 100, 'In the air, if any of:\n\u2022 Enter released\n\u2022 battery < 3.45 V '
            'for 0.5 s\n\u2022 no telemetry for 0.5 s', ha='right', va='center',
            fontsize=T_LABEL, color=INK2, linespacing=1.3)
    arrow(ax, (405, ly + 33 + 3), (440, ty - 33 - 3))
    ax.text(437, 133, 'near the\nfloor', ha='left', va='center', fontsize=T_LABEL,
            color=INK2, linespacing=1.15)
    # emergency bar
    ax.add_patch(FancyBboxPatch((5, 8), W - 10, 34, linewidth=LW, edgecolor=CRIT,
                                facecolor=CRIT_TINT, boxstyle='round,pad=0,rounding_size=4'))
    icon(ax, 22, 25, 'cross', CRIT)
    ax.text(W / 2 + 8, 25, 'From any state: Ctrl+C, or 1 s without keep-alive packets '
            '\u2192 motors cut', ha='center', va='center', fontsize=T_LABEL, color=INK)
    save(fig, 'fig10_safety')


def main(args):
    global OUT
    if args.out:
        OUT = Path(args.out)
    for make in (fig1_pipeline, fig_control_levels, fig2_control_loop, fig_network,
                 fig4_mission, fig5_results, fig6_comparison, fig7_sim_to_real, fig8_safety):
        make()
    if args.flight:
        fig_flight(args.flight)
    else:
        log.info('No --flight log given: fig7_flight (flown vs target) skipped.')
    log.info('Figures written to %s', OUT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--flight', help='rl_trigger.py mission CSV for fig7_flight '
                        '(logs/rl_trigger_flight_*.csv)')
    parser.add_argument('--out', help='output folder (default poster/figures)')
    main(parser.parse_args())
