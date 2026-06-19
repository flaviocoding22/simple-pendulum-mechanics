# -*- coding: utf-8 -*-
"""
Pendulo Simples - Simulacao Interativa v4
==========================================
Requer: pip install numpy matplotlib PyQt5
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.widgets import Slider, Button
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe

# ── Paleta ────────────────────────────────────────────────────────────────────
BG      = '#07080D'
SURFACE = '#0C0E18'
PANEL   = '#0F1120'
BORDER  = '#1A1F35'
BORDER2 = '#252B45'
AMBER   = '#F5A623'
TEAL    = '#00C9A7'
GOLD    = '#FFD166'
BLUE    = '#7EB8FF'
VIOLET  = '#B48AFF'
WHITE   = '#DDE2F0'
MUTED   = '#3D4460'
MUTED2  = '#5A638A'
ACCENT  = '#0E1220'
RED     = '#FF6B6B'

plt.rcParams.update({
    'font.family':      'monospace',
    'text.color':       WHITE,
    'axes.facecolor':   PANEL,
    'figure.facecolor': BG,
    'axes.edgecolor':   BORDER2,
    'axes.linewidth':   0.6,
    'xtick.color':      MUTED2,
    'ytick.color':      MUTED2,
    'xtick.labelsize':  7.5,
    'ytick.labelsize':  7.5,
    'grid.color':       '#0B0D1A',
    'grid.linewidth':   0.5,
})

# ── Parâmetros ────────────────────────────────────────────────────────────────
DEFAULTS = dict(L=1.6, m=1.0, theta0_deg=55.0, g=9.81)
DT       = 0.016
T_SIM    = 14.0
TRAIL    = 80
N_SEGS   = 18

state = dict(
    paused=False, frame=0,
    L=DEFAULTS['L'], m=DEFAULTS['m'],
    theta0_deg=DEFAULTS['theta0_deg'],
    theta0=np.radians(DEFAULTS['theta0_deg']),
    g=DEFAULTS['g']
)

# ── RK4 ───────────────────────────────────────────────────────────────────────
def compute(g, L, m, theta0):
    t  = np.arange(0, T_SIM, DT)
    Ns = len(t)
    th = np.zeros(Ns); om = np.zeros(Ns)
    th[0] = theta0
    for i in range(1, Ns):
        def d(a, b): return b, -(g / L) * np.sin(a)
        k1 = d(th[i-1],             om[i-1])
        k2 = d(th[i-1]+DT/2*k1[0], om[i-1]+DT/2*k1[1])
        k3 = d(th[i-1]+DT/2*k2[0], om[i-1]+DT/2*k2[1])
        k4 = d(th[i-1]+DT  *k3[0], om[i-1]+DT  *k3[1])
        th[i] = th[i-1] + DT/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
        om[i] = om[i-1] + DT/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
    xb = L*np.sin(th); yb = -L*np.cos(th)
    hp = L*(1-np.cos(th)); vp = L*np.abs(om)
    Ek = 0.5*m*vp**2; Ep = m*g*hp; Et = Ek+Ep
    Tp = 2*np.pi*np.sqrt(L/g)
    return t, Ns, th, om, xb, yb, hp, vp, Ek, Ep, Et, Tp

t_arr, N, theta, omega, xb, yb, h_p, v_p, Ek, Ep, Et, Tp = \
    compute(state['g'], state['L'], state['m'], state['theta0'])
Et0 = Et[0]

# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
#  [  simulação (esq)  ] [ energia×tempo (dir-cima)  ]
#  [                   ] [ espaço de fases (dir-meio) ]
#  [                   ] [─────────── painel ─────────]
#  [                   ] [ slider L   ████░░  1.6 m   ]
#  [                   ] [ slider m   ██░░░░  1.0 kg  ]
#  [                   ] [ slider θ₀  ████░░  55 °    ]
#  [                   ] [ slider g   ███░░░  9.8 m/s²]
#  [                   ] [ [Pausar]  [Aplicar] [Reset] ]
# ═══════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(17, 9.5), facecolor=BG)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

# Faixa de cabeçalho: y de 0.88 a 1.00  (12% do topo)
# Simulação:          y de 0.04 a 0.87  (coluna esquerda, abaixo do header)
# Gráficos direita:   y de 0.275 a 0.87 (acima do painel de controles)
# Painel controles:   y de 0.04 a 0.262 (direita-baixo)

HEADER_BOT = 0.845   # onde o cabeçalho termina — mais espaço pro header

ax_d = fig.add_axes([0.01,  0.04, 0.505, HEADER_BOT - 0.04])  # simulação
ax_e = fig.add_axes([0.555, 0.540, 0.430, HEADER_BOT - 0.550])  # energia × tempo
ax_f = fig.add_axes([0.555, 0.275, 0.430, 0.265])               # espaço de fases

# ── Painel de controle (direita-baixo) ───────────────────────────────────────
# 4 sliders com label + barra integrada + valor numérico
# Cada linha: [LABEL] [══slider══] [valor + unidade]

CTRL_L   = 0.558   # left da área de controle
CTRL_R   = 0.985   # right
CTRL_W   = CTRL_R - CTRL_L
CTRL_BOT = 0.040   # bottom
CTRL_TOP = 0.255   # top  (= base do ax_f)

# Alturas internas (em coord figura)
LINE_H  = 0.028   # altura de cada linha de slider
LINE_G  = 0.016   # gap entre linhas
BTN_H   = 0.038   # altura dos botões
BTN_G   = 0.012   # gap botões↔sliders

# y positions (bottom-up)
y_btn  = CTRL_BOT
y_sl   = [y_btn + BTN_H + BTN_G + i*(LINE_H + LINE_G) for i in range(4)]

# Largura do slider em si (deixa espaço pro label e pro valor)
LBL_W  = 0.055   # largura do label (L, m, θ₀, g)
VAL_W  = 0.072   # largura da área de valor numérico
SL_W   = CTRL_W - LBL_W - VAL_W - 0.006

SL_X   = CTRL_L + LBL_W + 0.003

params = [
    # (label, unidade, vmin, vmax, valinit, color)
    ('L',  'm',      0.3,  3.0,  state['L'],          TEAL),
    ('m',  'kg',     0.1,  5.0,  state['m'],          AMBER),
    (u'\u03b8\u2080', u'\u00b0', 5.0, 85.0, state['theta0_deg'], BLUE),
    ('g',  'm/s\u00b2', 1.0, 25.0, state['g'],        VIOLET),
]

slider_axes   = []
sliders       = []
val_texts     = []   # referências para atualizar valores em tempo real

for i, (lbl, unit, vmin, vmax, vinit, col) in enumerate(params):
    y = y_sl[i]

    # Label do parâmetro
    fig.text(CTRL_L + 0.002, y + LINE_H/2, lbl,
             color=col, fontsize=9, fontweight='bold', va='center')

    # Eixo do slider
    ax_s = fig.add_axes([SL_X, y, SL_W, LINE_H], facecolor=ACCENT)
    for sp in ax_s.spines.values():
        sp.set_edgecolor(BORDER2); sp.set_linewidth(0.7)
    sl = Slider(ax_s, '', vmin, vmax, valinit=vinit, color=col)
    sl.valtext.set_visible(False)   # ocultamos o valor padrão do slider
    sl.label.set_visible(False)
    slider_axes.append(ax_s)
    sliders.append(sl)

    # Valor numérico à direita (atualizado dinamicamente)
    vt = fig.text(SL_X + SL_W + 0.005, y + LINE_H/2,
                  f'{vinit:.1f} {unit}',
                  color=WHITE, fontsize=8.5, va='center',
                  fontfamily='monospace')
    val_texts.append((vt, unit))

sl_L, sl_m, sl_th, sl_g = sliders

# Atualiza textos de valor quando slider move
def _upd_val(i, sl, fmt='.1f'):
    unit = val_texts[i][1]
    val_texts[i][0].set_text(f'{sl.val:{fmt}} {unit}')
    fig.canvas.draw_idle()

sl_L.on_changed( lambda v: _upd_val(0, sl_L))
sl_m.on_changed( lambda v: _upd_val(1, sl_m))
sl_th.on_changed(lambda v: _upd_val(2, sl_th, '.0f'))
sl_g.on_changed( lambda v: _upd_val(3, sl_g))

# ── Botões ────────────────────────────────────────────────────────────────────
N_BTN  = 3
BTN_W  = (CTRL_W - 0.012) / N_BTN

ax_pp  = fig.add_axes([CTRL_L,                    y_btn, BTN_W, BTN_H], facecolor=ACCENT)
ax_ap  = fig.add_axes([CTRL_L + BTN_W + 0.006,   y_btn, BTN_W, BTN_H], facecolor='#0A2015')
ax_rst = fig.add_axes([CTRL_L + 2*(BTN_W+0.006), y_btn, BTN_W, BTN_H], facecolor=ACCENT)

for ax_b in [ax_pp, ax_ap, ax_rst]:
    for sp in ax_b.spines.values():
        sp.set_edgecolor(BORDER2); sp.set_linewidth(0.7)

btn_pp  = Button(ax_pp,  u'\u25b6  Pausar',  color=ACCENT,    hovercolor='#1E2550')
btn_ap  = Button(ax_ap,  u'\u2713  Aplicar', color='#0A2015', hovercolor='#143320')
btn_rst = Button(ax_rst, u'\u21ba  Reset',   color=ACCENT,    hovercolor='#1E2550')

btn_pp.label.set_color(MUTED2);  btn_pp.label.set_fontsize(9)
btn_ap.label.set_color(TEAL);    btn_ap.label.set_fontsize(9)
btn_rst.label.set_color(MUTED2); btn_rst.label.set_fontsize(9)

# ── Cabeçalho (faixa dedicada no topo, acima de tudo) ────────────────────────
# Título centralizado na metade esquerda (acima da simulação)
fig.text(0.265, 0.994, u'P\u00caNDULO SIMPLES',
         ha='center', va='top', fontsize=17, fontweight='bold', color=AMBER,
         path_effects=[pe.withStroke(linewidth=10, foreground=BG)])
fig.text(0.265, 0.962,
         u'simulação interativa  \u00b7  runge-kutta 4\u00aa ordem',
         ha='center', va='top', fontsize=8, color=MUTED2)

# Sub-linha de parâmetros — também no cabeçalho, centralizada
sub_txt = fig.text(0.265, 0.934, '', ha='center', va='top', fontsize=8, color=MUTED)

# Status (direita-baixo, acima dos botões)
status_txt = fig.text(CTRL_L, CTRL_BOT + BTN_H + 0.005, '',
                      va='bottom', fontsize=7.5, color=MUTED2)

# Separadores
# Linha horizontal abaixo do cabeçalho (separa header da simulação/gráficos)
fig.add_artist(plt.Line2D([0.01, 0.99], [HEADER_BOT, HEADER_BOT],
    transform=fig.transFigure, color=BORDER, lw=0.7))
# Linha vertical separando coluna esquerda (simulação) da direita (gráficos)
fig.add_artist(plt.Line2D([0.538, 0.538], [0.03, HEADER_BOT],
    transform=fig.transFigure, color=BORDER, lw=0.7))
# Linha horizontal separando gráficos do painel de controle
fig.add_artist(plt.Line2D([0.555, 0.990], [0.262, 0.262],
    transform=fig.transFigure, color=BORDER2, lw=0.6))

# Label de seção dos controles
fig.text(CTRL_L, CTRL_TOP - 0.002,
         u'\u25c8  PAR\u00c2METROS', color=MUTED2, fontsize=7.5, fontweight='bold',
         va='top')

def _update_sub():
    sub_txt.set_text(
        u'L = %.2f m   \u00b7   m = %.1f kg   \u00b7   '
        u'\u03b8\u2080 = %.0f\u00b0   \u00b7   g = %.2f m/s\u00b2   \u00b7   T \u2248 %.3f s'
        % (state['L'], state['m'], state['theta0_deg'], state['g'], Tp)
    )

_update_sub()

def _style(ax, title, xl='', yl=''):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=MUTED2, fontsize=8, fontweight='bold',
                 pad=5, loc='left')
    ax.set_xlabel(xl, color=MUTED, fontsize=7.5, labelpad=3)
    ax.set_ylabel(yl, color=MUTED, fontsize=7.5, labelpad=3)
    ax.grid(True, alpha=0.6)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER2); sp.set_linewidth(0.6)

aobj = {}

def rebuild():
    global t_arr, N, theta, omega, xb, yb, h_p, v_p, Ek, Ep, Et, Tp, Et0

    t_arr, N, theta, omega, xb, yb, h_p, v_p, Ek, Ep, Et, Tp = \
        compute(state['g'], state['L'], state['m'], state['theta0'])
    Et0 = Et[0]
    state['frame'] = 0
    _update_sub()

    L     = state['L']
    bob_r = max(0.08, L * 0.074)
    margin= max(0.32, L * 0.21)

    # ── Simulação ─────────────────────────────────────────────────────────────
    ax_d.clear()
    ax_d.set_facecolor(BG)
    xlim = L + margin
    ax_d.set_xlim(-xlim, xlim)
    ax_d.set_ylim(-L - margin, 0.80)
    ax_d.set_aspect('equal', adjustable='datalim')
    ax_d.axis('off')

    # Grade de fundo
    for gx in np.arange(-xlim, xlim+0.01, 0.4):
        ax_d.axvline(gx, color='#0A0C18', lw=0.4, alpha=0.8)
    for gy in np.arange(-L-margin, 0.81, 0.4):
        ax_d.axhline(gy, color='#0A0C18', lw=0.4, alpha=0.8)

    # Suporte
    ax_d.fill_between([-xlim, xlim], [0.48, 0.48], [0.76, 0.76],
                      color='#080A12', alpha=0.95, zorder=1)
    for x in np.linspace(-xlim+0.06, xlim-0.06, 32):
        ax_d.plot([x, x+0.09], [0.48, 0.70],
                  color='#181E32', lw=0.7, alpha=0.55, zorder=2)
    ax_d.fill_between([-xlim, xlim], [0.46, 0.46], [0.50, 0.50],
                      color=MUTED, alpha=0.4, zorder=3)
    ax_d.plot([-xlim, xlim], [0.48, 0.48], color=MUTED, lw=1.4, zorder=4)

    # Pivô
    ax_d.plot(0, 0, 'o', color='#3A4060', ms=14, zorder=10)
    ax_d.plot(0, 0, 'o', color='#5A6080', ms=10, zorder=11)
    ax_d.plot(0, 0, 'o', color=BG,        ms=5,  zorder=12)

    # h=0
    ax_d.axhline(-L, color=TEAL, lw=0.7, ls=(0,(8,5)), alpha=0.30, zorder=2)
    ax_d.text(xlim-0.06, -L+0.06, 'h = 0',
              color=TEAL, fontsize=7, alpha=0.45, ha='right')

    # Arco fantasma
    arc_a = np.linspace(-state['theta0'], state['theta0'], 400)
    ax_d.plot(L*np.sin(arc_a), -L*np.cos(arc_a),
              color=BORDER2, lw=0.9, ls='--', alpha=0.35, zorder=2)
    for side in [-1,1]:
        ax_d.plot([0, side*L*np.sin(state['theta0'])],
                  [0,   -L*np.cos(state['theta0'])],
                  color=BORDER2, lw=0.6, ls=':', alpha=0.28, zorder=2)

    # Rastro
    segs = []
    for k in range(N_SEGS):
        frac = k / N_SEGS
        seg, = ax_d.plot([], [], color=AMBER,
                         lw=0.3 + 3.5*frac**2.2,
                         alpha=0.02 + 0.45*frac**1.6,
                         solid_capstyle='round', zorder=5)
        segs.append(seg)

    # Fio
    fio,  = ax_d.plot([], [], color='#6878A0', lw=2.0,
                      solid_capstyle='round', zorder=7)
    fio2, = ax_d.plot([], [], color='#8898C0', lw=0.5,
                      solid_capstyle='round', zorder=8, alpha=0.45)

    # Bob
    bob_p  = Circle((xb[0], yb[0]), bob_r,        color=AMBER,    zorder=10)
    bob_gl = Circle((xb[0], yb[0]), bob_r*1.55,   color=AMBER,    alpha=0.06, zorder=9)
    bob_hi = Circle((xb[0]-bob_r*0.28, yb[0]+bob_r*0.35), bob_r*0.26,
                    color='#FFFFFF', alpha=0.30, zorder=11)
    for p in [bob_p, bob_gl, bob_hi]:
        ax_d.add_patch(p)

    # Vetor velocidade
    varr = ax_d.quiver(xb[0], yb[0], 0, 0,
                       color=AMBER, scale=1, scale_units='xy', angles='xy',
                       width=0.010, headwidth=5, headlength=6,
                       headaxislength=5, zorder=12, alpha=0.80)

    # Caixas de energia (canto superior esquerdo da simulação)
    box_cfg = dict(boxstyle='round,pad=0.42', lw=0.7, alpha=0.88)
    ek_lbl = ax_d.text(0.03, 0.97, '', transform=ax_d.transAxes,
        fontsize=8.5, color=AMBER, fontweight='bold', ha='left', va='top',
        bbox=dict(**box_cfg, fc='#160A00', ec=AMBER))
    ep_lbl = ax_d.text(0.03, 0.88, '', transform=ax_d.transAxes,
        fontsize=8.5, color=TEAL,  fontweight='bold', ha='left', va='top',
        bbox=dict(**box_cfg, fc='#001610', ec=TEAL))
    et_lbl = ax_d.text(0.03, 0.79, '', transform=ax_d.transAxes,
        fontsize=8.5, color=GOLD,  fontweight='bold', ha='left', va='top',
        bbox=dict(**box_cfg, fc='#141000', ec=GOLD))

    # Info de rodapé dentro da simulação
    t_lbl = ax_d.text(0.50, 0.015, '', transform=ax_d.transAxes,
        fontsize=7.5, color=MUTED2, ha='center', va='bottom')

    # ── Energia × Tempo ───────────────────────────────────────────────────────
    ax_e.clear()
    _style(ax_e, u'\u25c6  ENERGIA \u00d7 TEMPO', 't (s)', 'E (J)')
    ax_e.set_xlim(0, T_SIM)
    ax_e.set_ylim(-0.03, Et0 * 1.28)
    ax_e.fill_between(t_arr, 0, Ek, color=AMBER, alpha=0.04)
    ax_e.fill_between(t_arr, 0, Ep, color=TEAL,  alpha=0.04)
    ax_e.plot(t_arr, Et, color=GOLD, lw=0.5, ls='--', alpha=0.10)

    ln_ek, = ax_e.plot([], [], color=AMBER, lw=2.0, label='Ek  cinética',  alpha=0.9)
    ln_ep, = ax_e.plot([], [], color=TEAL,  lw=2.0, label='Ep  potencial', alpha=0.9)
    ln_et, = ax_e.plot([], [], color=GOLD,  lw=1.2, ls='--', label='E  total', alpha=0.75)
    dk  = ax_e.plot([], [], 'o', color=AMBER, ms=5, zorder=8)[0]
    dp  = ax_e.plot([], [], 'o', color=TEAL,  ms=5, zorder=8)[0]
    dtt = ax_e.plot([], [], 'o', color=GOLD,  ms=4, zorder=8)[0]
    ax_e.legend(facecolor='#060810', edgecolor=BORDER2, labelcolor=WHITE,
                fontsize=7.5, loc='upper right', framealpha=0.94,
                handlelength=1.5, borderpad=0.6, labelspacing=0.45)

    # ── Espaço de Fases ───────────────────────────────────────────────────────
    ax_f.clear()
    _style(ax_f, u'\u25c6  ESPA\u00c7O DE FASES', u'\u00e2ngulo (\u00b0)', 'vel. angular (rad/s)')
    th_d = np.degrees(theta)
    pad_x = max(2, (th_d.max()-th_d.min()) * 0.14)
    pad_y = max(0.1, (omega.max()-omega.min()) * 0.14)
    ax_f.set_xlim(th_d.min()-pad_x, th_d.max()+pad_x)
    ax_f.set_ylim(omega.min()-pad_y, omega.max()+pad_y)
    ax_f.axhline(0, color=BORDER2, lw=0.7, alpha=0.7)
    ax_f.axvline(0, color=BORDER2, lw=0.7, alpha=0.7)
    ax_f.plot(th_d, omega, color='#111525', lw=1.2)

    ph_trail, = ax_f.plot([], [], color=AMBER, lw=2.0, alpha=0.75,
                          solid_capstyle='round')
    ph_dot    = ax_f.plot([], [], 'o', color=AMBER, ms=8,  zorder=9)[0]
    ph_ring   = ax_f.plot([], [], 'o', color=AMBER, ms=18, zorder=8,
                          markerfacecolor='none', markeredgewidth=1.0,
                          alpha=0.22)[0]

    aobj.update(dict(
        segs=segs, fio=fio, fio2=fio2,
        bob=bob_p, bob_gl=bob_gl, bob_hi=bob_hi,
        varr=varr,
        t_lbl=t_lbl, ek_lbl=ek_lbl, ep_lbl=ep_lbl, et_lbl=et_lbl,
        ln_ek=ln_ek, ln_ep=ln_ep, ln_et=ln_et,
        dk=dk, dp=dp, dtt=dtt,
        ph_trail=ph_trail, ph_dot=ph_dot, ph_ring=ph_ring,
    ))

rebuild()

def update(_frame):
    if state['paused']:
        return []

    state['frame'] = (state['frame'] + 1) % N
    i  = state['frame']
    xi = xb[i]; yi = yb[i]
    ti = t_arr[i]
    L  = state['L']
    br = aobj['bob'].radius

    aobj['fio'].set_data([0, xi],  [0, yi])
    aobj['fio2'].set_data([0, xi], [0, yi])

    for patch, off in [
        (aobj['bob'],    (0, 0)),
        (aobj['bob_gl'], (0, 0)),
        (aobj['bob_hi'], (-br*0.28, br*0.35)),
    ]:
        patch.center = (xi + off[0], yi + off[1])

    spd = np.clip(v_p[i] / (v_p.max() + 1e-9), 0, 1)
    r0,g0,b0 = mcolors.to_rgb(AMBER)
    rt,gt,bt = mcolors.to_rgb('#FF4040')
    aobj['bob'].set_color((r0+(rt-r0)*spd**0.7, g0+(gt-g0)*spd**0.7, b0+(bt-b0)*spd**0.7))
    aobj['bob_gl'].set_alpha(0.04 + 0.10*spd)

    i0 = max(0, i - TRAIL); sl_len = i - i0 + 1
    for k, seg in enumerate(aobj['segs']):
        a = i0 + int(k/N_SEGS*sl_len)
        b = i0 + int((k+1)/N_SEGS*sl_len)
        seg.set_data(xb[a:b+1], yb[a:b+1])

    vx = omega[i]*L*np.cos(theta[i])*0.28
    vy = omega[i]*L*np.sin(theta[i])*0.28
    aobj['varr'].set_offsets([[xi, yi]])
    aobj['varr'].set_UVC(vx, vy)

    aobj['ek_lbl'].set_text('  Ek = %6.3f J  ' % Ek[i])
    aobj['ep_lbl'].set_text('  Ep = %6.3f J  ' % Ep[i])
    aobj['et_lbl'].set_text('   E = %6.3f J  ' % Et[i])

    aobj['t_lbl'].set_text(
        u't = %.2f s   \u00b7   \u03b8 = %+.1f\u00b0   \u00b7   '
        u'\u03c9 = %+.2f rad/s   \u00b7   v = %.2f m/s'
        % (ti, np.degrees(theta[i]), omega[i], v_p[i])
    )

    aobj['ln_ek'].set_data(t_arr[:i+1], Ek[:i+1])
    aobj['ln_ep'].set_data(t_arr[:i+1], Ep[:i+1])
    aobj['ln_et'].set_data(t_arr[:i+1], Et[:i+1])
    aobj['dk'].set_data( [ti], [Ek[i]])
    aobj['dp'].set_data( [ti], [Ep[i]])
    aobj['dtt'].set_data([ti], [Et[i]])

    p0 = max(0, i - 110)
    aobj['ph_trail'].set_data(np.degrees(theta[p0:i+1]), omega[p0:i+1])
    aobj['ph_dot'].set_data(  [np.degrees(theta[i])], [omega[i]])
    aobj['ph_ring'].set_data( [np.degrees(theta[i])], [omega[i]])
    col = AMBER if Ek[i] > Et0*0.5 else TEAL
    aobj['ph_dot'].set_color(col)
    aobj['ph_ring'].set_color(col)

    return []

def apply_params(event=None):
    state['L']          = sl_L.val
    state['m']          = sl_m.val
    state['theta0_deg'] = sl_th.val
    state['theta0']     = np.radians(sl_th.val)
    state['g']          = sl_g.val
    state['paused']     = False
    btn_pp.label.set_text(u'\u25b6  Pausar')
    btn_pp.label.set_color(MUTED2)
    status_txt.set_text(u'\u27f3  recalculando...')
    fig.canvas.draw_idle()
    rebuild()
    status_txt.set_text(u'\u2713  T = %.3f s   \u00b7   E\u2080 = %.3f J' % (Tp, Et0))
    fig.canvas.draw_idle()

def toggle_pause(event):
    state['paused'] = not state['paused']
    if state['paused']:
        btn_pp.label.set_text(u'\u25b6  Continuar')
        btn_pp.label.set_color(TEAL)
        status_txt.set_text(u'\u23f8  pausado')
    else:
        btn_pp.label.set_text(u'\u25b6  Pausar')
        btn_pp.label.set_color(MUTED2)
        status_txt.set_text('')
    fig.canvas.draw_idle()

def do_reset(event):
    sl_L.reset(); sl_m.reset(); sl_th.reset(); sl_g.reset()
    # Atualiza textos de valor
    for idx, (sl, fmt) in enumerate(zip(sliders, ['.1f','.1f','.0f','.1f'])):
        unit = val_texts[idx][1]
        val_texts[idx][0].set_text(f'{sl.val:{fmt}} {unit}')
    apply_params()

btn_pp.on_clicked(toggle_pause)
btn_ap.on_clicked(apply_params)
btn_rst.on_clicked(do_reset)

ani = animation.FuncAnimation(
    fig, update,
    interval=14,
    blit=False,
    cache_frame_data=False,
)

plt.show()