"""
Pêndulo Simples — Conservação de Energia Mecânica
=================================================
Design: laboratório científico premium, paleta âmbar/carvão
Requer: pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.patches import Circle, Arc, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors

# ═══════════════════════════════════════════════════════════════════
#  PALETA — laboratório âmbar sobre carvão profundo
# ═══════════════════════════════════════════════════════════════════
BG        = '#0A0B0F'      # fundo absoluto
PANEL     = '#0F1118'      # painel
BORDER    = '#1C2030'      # borda sutil
GRID_C    = '#131825'      # grade

AMBER     = '#F5A623'      # energia cinética
AMBER_LO  = '#7A4800'      # fill suave
TEAL      = '#00C9A7'      # energia potencial
TEAL_LO   = '#003D32'      # fill suave
GOLD      = '#FFD166'      # energia total
WHITE     = '#E8EBF4'      # texto primário
MUTED     = '#4A5068'      # texto secundário
ACCENT    = '#2A3050'      # detalhe estrutural

# ═══════════════════════════════════════════════════════════════════
#  PARÂMETROS FÍSICOS
# ═══════════════════════════════════════════════════════════════════
g      = 9.81
L      = 1.6
m      = 1.0
theta0 = np.radians(55)
dt     = 0.018
T      = 10.0
TRAIL  = 55

# ═══════════════════════════════════════════════════════════════════
#  SIMULAÇÃO RK4
# ═══════════════════════════════════════════════════════════════════
t_arr = np.arange(0, T, dt)
N     = len(t_arr)
theta = np.zeros(N)
omega = np.zeros(N)
theta[0] = theta0

for i in range(1, N):
    def d(th, om): return om, -(g/L)*np.sin(th)
    k1t,k1o = d(theta[i-1], omega[i-1])
    k2t,k2o = d(theta[i-1]+dt/2*k1t, omega[i-1]+dt/2*k1o)
    k3t,k3o = d(theta[i-1]+dt/2*k2t, omega[i-1]+dt/2*k2o)
    k4t,k4o = d(theta[i-1]+dt*k3t,   omega[i-1]+dt*k3o)
    theta[i] = theta[i-1] + dt/6*(k1t+2*k2t+2*k3t+k4t)
    omega[i] = omega[i-1] + dt/6*(k1o+2*k2o+2*k3o+k4o)

xb   = L * np.sin(theta)
yb   = -L * np.cos(theta)
h_p  = L * (1 - np.cos(theta))
v_p  = L * np.abs(omega)
Ek   = 0.5 * m * v_p**2
Ep   = m * g * h_p
Et   = Ek + Ep
Et0  = Et[0]

# período analítico (pequenas oscilações)
T_periodo = 2 * np.pi * np.sqrt(L/g)

# ═══════════════════════════════════════════════════════════════════
#  FIGURA
# ═══════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family':     'monospace',
    'axes.facecolor':  PANEL,
    'figure.facecolor':BG,
    'text.color':      WHITE,
    'axes.edgecolor':  BORDER,
    'axes.linewidth':  0.7,
    'xtick.color':     MUTED,
    'ytick.color':     MUTED,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'grid.color':      GRID_C,
    'grid.linewidth':  0.7,
})

fig = plt.figure(figsize=(16, 9), facecolor=BG)

# ── layout: coluna esquerda larga (diagrama) + 3 linhas à direita
gs = gridspec.GridSpec(
    3, 3,
    figure=fig,
    left=0.04, right=0.97,
    top=0.91, bottom=0.07,
    hspace=0.55, wspace=0.38,
    width_ratios=[1.05, 1.3, 1.15],
    height_ratios=[1, 1, 0.85],
)

# ── cabeçalho ────────────────────────────────────────────────────────
fig.text(0.5, 0.975,
         'PÊNDULO SIMPLES',
         ha='center', va='top',
         fontsize=19, fontweight='bold',
         color=AMBER,
         path_effects=[pe.withStroke(linewidth=10, foreground=BG)])

fig.text(0.5, 0.952,
         f'L = {L} m   ·   m = {m} kg   ·   θ₀ = {np.degrees(theta0):.0f}°'
         f'   ·   T ≈ {T_periodo:.2f} s   ·   '
         r'$E_{mec} = \frac{1}{2}mv^2 + mgh = \mathrm{cte}$',
         ha='center', va='top',
         fontsize=9.5, color=MUTED)

# linha decorativa
fig.add_artist(plt.Line2D(
    [0.04, 0.96], [0.935, 0.935],
    transform=fig.transFigure,
    color=BORDER, lw=0.8
))

# ═══════════════════════════════════════════════════════════════════
#  PAINEL A — Diagrama animado do pêndulo  (coluna 0, linhas 0-2)
# ═══════════════════════════════════════════════════════════════════
ax_d = fig.add_subplot(gs[:, 0])
ax_d.set_facecolor(PANEL)
ax_d.set_xlim(-2.1, 2.1)
ax_d.set_ylim(-2.0, 0.55)
ax_d.set_aspect('equal')
ax_d.axis('off')

# moldura
for (x0,y0,x1,y1) in [(-2.1,-2.0,-2.1,0.55),(-2.1,0.55,2.1,0.55),
                       (2.1,0.55,2.1,-2.0),(2.1,-2.0,-2.1,-2.0)]:
    ax_d.plot([x0,x1],[y0,y1], color=BORDER, lw=0.8)

# suporte
ax_d.fill_between([-2.1,2.1],[0.38,0.38],[0.55,0.55], color='#111520')
for x in np.linspace(-2.0, 2.0, 22):
    ax_d.plot([x, x+0.12],[0.38,0.55], color=ACCENT, lw=0.6, alpha=0.6)
ax_d.plot([-2.1,2.1],[0.38,0.38], color=MUTED, lw=1.5)

# pivô
ax_d.plot(0,0,'o', color=MUTED, ms=9, zorder=10)
ax_d.plot(0,0,'o', color=BG,    ms=4, zorder=11)

# linha de referência (h=0)
ax_d.axhline(-L, color=TEAL, lw=0.7, ls=(0,(6,4)), alpha=0.45)
ax_d.text(-2.0, -L-0.1, 'h = 0', color=TEAL, fontsize=7.5, alpha=0.6)

# arco de trajetória fantasma
arc_th = np.linspace(-theta0, theta0, 300)
ax_d.plot(L*np.sin(arc_th), -L*np.cos(arc_th),
          color=ACCENT, lw=1.2, ls='--', alpha=0.5, zorder=2)

# indicação ângulo máx
for side in [-1, 1]:
    ax_d.plot([0, side*L*np.sin(theta0)],
              [0,    -L*np.cos(theta0)],
              color=ACCENT, lw=0.8, alpha=0.4)

# ── objetos animados ────────────────────────────────────────────────
trail_x = np.full(TRAIL, xb[0])
trail_y = np.full(TRAIL, yb[0])

# rastro gradiente simulado com várias linhas
trail_segs = []
n_segs = 12
for k in range(n_segs):
    alpha = 0.04 + 0.25 * (k / n_segs)**1.8
    lw    = 0.6 + 2.4 * (k / n_segs)**2
    seg,  = ax_d.plot([], [], color=AMBER, lw=lw, alpha=alpha,
                      solid_capstyle='round', zorder=4)
    trail_segs.append(seg)

fio_line, = ax_d.plot([], [], color='#8090B0', lw=2.2,
                       solid_capstyle='round', zorder=6)

bob = Circle((xb[0], yb[0]), 0.155, color=AMBER, zorder=9)
ax_d.add_patch(bob)
# reflexo no bob
bob_hi = Circle((xb[0]-0.04, yb[0]+0.06), 0.04,
                color='#FFFFFF', alpha=0.35, zorder=10)
ax_d.add_patch(bob_hi)

# seta velocidade
vel_arrow = ax_d.annotate('', xy=(0,0), xytext=(0,0),
    arrowprops=dict(arrowstyle='->', color=AMBER,
                    lw=2.0, mutation_scale=14))

# indicador de altura (barra lateral)
h_bar_bg = ax_d.fill_betweenx([0,0],[-2.05,-1.80],
                               color=ACCENT, alpha=0.4)
h_bar = ax_d.fill_betweenx([0,0],[-2.05,-1.80],
                             color=TEAL, alpha=0.7)
ax_d.text(-1.93, -2.15, 'h', color=TEAL, fontsize=8, ha='center')

# painéis de energia no diagrama
ek_box = ax_d.text(0.62, 0.44, '',
    transform=ax_d.transAxes, ha='center', va='top',
    fontsize=8.5, color=AMBER, fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.45', fc='#1A0F00',
              ec=AMBER, lw=0.8, alpha=0.85))
ep_box = ax_d.text(0.62, 0.29, '',
    transform=ax_d.transAxes, ha='center', va='top',
    fontsize=8.5, color=TEAL, fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.45', fc='#001A15',
              ec=TEAL, lw=0.8, alpha=0.85))
et_box = ax_d.text(0.62, 0.14, '',
    transform=ax_d.transAxes, ha='center', va='top',
    fontsize=8.5, color=GOLD, fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.45', fc='#1A1200',
              ec=GOLD, lw=0.8, alpha=0.85))

# label tempo + ângulo
time_lbl = ax_d.text(0.5, -0.04, '',
    transform=ax_d.transAxes, ha='center',
    fontsize=8, color=MUTED)

# ═══════════════════════════════════════════════════════════════════
#  helper eixo
# ═══════════════════════════════════════════════════════════════════
def sax(ax, title, xlabel='', ylabel=''):
    ax.set_title(title, color=WHITE, fontsize=9, fontweight='bold',
                 loc='left', pad=7)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=8)
    ax.grid(True, axis='both')
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
        sp.set_linewidth(0.7)

# ═══════════════════════════════════════════════════════════════════
#  PAINEL B — Energia × Tempo  (col 1, row 0-1)
# ═══════════════════════════════════════════════════════════════════
ax_e = fig.add_subplot(gs[0:2, 1])
sax(ax_e, 'ENERGIA  ×  TEMPO', 'tempo  (s)', 'energia  (J)')
ax_e.set_xlim(0, T)
ax_e.set_ylim(-0.15, Et0 * 1.18)

# faixas de fill estáticas (guia)
ax_e.fill_between(t_arr, 0, Ek, color=AMBER,  alpha=0.04)
ax_e.fill_between(t_arr, 0, Ep, color=TEAL,   alpha=0.04)
# linha total estática (guia)
ax_e.plot(t_arr, Et, color=GOLD, lw=0.8, ls='--', alpha=0.18)

cursor_e = ax_e.axvline(0, color=WHITE, lw=0.7, alpha=0.35, ls=':')
ln_ek,  = ax_e.plot([], [], color=AMBER, lw=2.4, label='$E_k$  cinética')
ln_ep,  = ax_e.plot([], [], color=TEAL,  lw=2.4, label='$E_p$  potencial')
ln_et,  = ax_e.plot([], [], color=GOLD,  lw=1.8, ls='--', label='$E_{mec}$  total')

dot_ek = ax_e.plot([], [], 'o', color=AMBER, ms=7, zorder=8)[0]
dot_ep = ax_e.plot([], [], 'o', color=TEAL,  ms=7, zorder=8)[0]
dot_et = ax_e.plot([], [], 'o', color=GOLD,  ms=6, zorder=8)[0]

leg = ax_e.legend(
    facecolor='#090C14', edgecolor=BORDER,
    labelcolor=WHITE, fontsize=8.5,
    loc='upper right', framealpha=0.9,
    handlelength=1.6
)

# ═══════════════════════════════════════════════════════════════════
#  PAINEL C — Espaço de fases  (col 2, row 0-1)
# ═══════════════════════════════════════════════════════════════════
ax_f = fig.add_subplot(gs[0:2, 2])
sax(ax_f, 'ESPAÇO DE FASES', 'ângulo  θ  (°)', 'vel. angular  ω  (rad/s)')
ax_f.set_xlim(np.degrees(theta).min()*1.15, np.degrees(theta).max()*1.15)
ax_f.set_ylim(omega.min()*1.15, omega.max()*1.15)
ax_f.axhline(0, color=BORDER, lw=0.8)
ax_f.axvline(0, color=BORDER, lw=0.8)

# órbita completa estática
ax_f.plot(np.degrees(theta), omega, color=ACCENT, lw=1.2, alpha=0.4)

phase_trail, = ax_f.plot([], [], color=AMBER, lw=2.0, alpha=0.7)
phase_dot    = ax_f.plot([], [], 'o', color=AMBER, ms=9, zorder=8)[0]
phase_dot_ring = ax_f.plot([], [], 'o', color=AMBER, ms=16,
                            markerfacecolor='none',
                            markeredgewidth=1.2, alpha=0.35, zorder=7)[0]

# ═══════════════════════════════════════════════════════════════════
#  PAINEL D — Ângulo × Tempo  (col 1, row 2)
# ═══════════════════════════════════════════════════════════════════
ax_a = fig.add_subplot(gs[2, 1])
sax(ax_a, 'ÂNGULO  ×  TEMPO', 'tempo  (s)', 'θ  (°)')
ax_a.set_xlim(0, T)
ylim_a = np.degrees(theta0) * 1.25
ax_a.set_ylim(-ylim_a, ylim_a)
ax_a.axhline(0, color=BORDER, lw=0.8)

ax_a.fill_between(t_arr, 0, np.degrees(theta), color=AMBER, alpha=0.05)

cursor_a = ax_a.axvline(0, color=WHITE, lw=0.7, alpha=0.35, ls=':')
ln_ang,  = ax_a.plot([], [], color='#A8C8FF', lw=2.2)
dot_ang  = ax_a.plot([], [], 'o', color='#A8C8FF', ms=7, zorder=8)[0]

# ═══════════════════════════════════════════════════════════════════
#  PAINEL E — Barra de proporção de energia  (col 2, row 2)
# ═══════════════════════════════════════════════════════════════════
ax_b = fig.add_subplot(gs[2, 2])
sax(ax_b, 'PROPORÇÃO DE ENERGIA')
ax_b.set_xlim(0, 1)
ax_b.set_ylim(0, 1)
ax_b.axis('off')
ax_b.set_facecolor(PANEL)
for sp in ax_b.spines.values():
    sp.set_edgecolor(BORDER)
    sp.set_linewidth(0.7)

# moldura manual
for (x0,y0,x1,y1) in [(0,0,0,1),(0,1,1,1),(1,1,1,0),(1,0,0,0)]:
    ax_b.plot([x0,x1],[y0,y1], color=BORDER, lw=0.7,
              transform=ax_b.transAxes)

# barras de energia
BAR_Y  = 0.62
BAR_H  = 0.22
BAR_X0 = 0.08
BAR_W  = 0.84

# fundo da barra
ax_b.add_patch(Rectangle((BAR_X0, BAR_Y), BAR_W, BAR_H,
    transform=ax_b.transAxes, fc=ACCENT, ec=BORDER, lw=0.8))

bar_ek = ax_b.add_patch(Rectangle((BAR_X0, BAR_Y), 0, BAR_H,
    transform=ax_b.transAxes, fc=AMBER, ec='none'))
bar_ep = ax_b.add_patch(Rectangle((BAR_X0, BAR_Y), 0, BAR_H,
    transform=ax_b.transAxes, fc=TEAL, ec='none'))

lbl_bar_ek = ax_b.text(0.5, 0.92, '',
    transform=ax_b.transAxes, ha='center',
    fontsize=9.5, color=AMBER, fontweight='bold')
lbl_bar_ep = ax_b.text(0.5, 0.52, '',
    transform=ax_b.transAxes, ha='center',
    fontsize=9.5, color=TEAL, fontweight='bold')

ax_b.text(0.5, 0.38, '─────────────────',
    transform=ax_b.transAxes, ha='center', color=BORDER, fontsize=8)

lbl_et_b = ax_b.text(0.5, 0.22, '',
    transform=ax_b.transAxes, ha='center',
    fontsize=9, color=GOLD, fontweight='bold')
lbl_om   = ax_b.text(0.5, 0.08, '',
    transform=ax_b.transAxes, ha='center',
    fontsize=8, color=MUTED)

# ═══════════════════════════════════════════════════════════════════
#  ANIMAÇÃO
# ═══════════════════════════════════════════════════════════════════
_vel_arrow = [vel_arrow]
_h_bar     = [h_bar]

def update(frame):
    i  = frame % N
    xi, yi = xb[i], yb[i]
    ti     = t_arr[i]
    hi     = h_p[i]
    eki    = Ek[i]
    epi    = Ep[i]
    eti    = Et[i]

    # ── fio + bob ─────────────────────────────────────────────────
    fio_line.set_data([0, xi], [0, yi])
    bob.center    = (xi, yi)
    bob_hi.center = (xi-0.04, yi+0.06)

    # cor do bob varia com velocidade (âmbar → branco qdo rápido)
    spd_frac = min(v_p[i] / v_p.max(), 1.0)
    r1,g1,b1 = mcolors.to_rgb(AMBER)
    r2,g2,b2 = 1.0, 1.0, 0.95
    bob.set_color((r1+(r2-r1)*spd_frac**0.6,
                   g1+(g2-g1)*spd_frac**0.6,
                   b1+(b2-b1)*spd_frac**0.6))

    # ── rastro gradiente ─────────────────────────────────────────
    i0 = max(0, i-TRAIL)
    seg_len = i - i0 + 1
    for k, seg in enumerate(trail_segs):
        a = i0 + int(k/n_segs * seg_len)
        b_ = i0 + int((k+1)/n_segs * seg_len)
        seg.set_data(xb[a:b_+1], yb[a:b_+1])

    # ── seta velocidade ───────────────────────────────────────────
    _vel_arrow[0].remove()
    vx = omega[i] * L * np.cos(theta[i]) * 0.22
    vy = omega[i] * L * np.sin(theta[i]) * 0.22
    _vel_arrow[0] = ax_d.annotate(
        '', xy=(xi+vx, yi+vy), xytext=(xi, yi),
        arrowprops=dict(arrowstyle='->', color=AMBER,
                        lw=2.0, mutation_scale=14))

    # ── barra lateral de altura ───────────────────────────────────
    _h_bar[0].remove()
    frac_h = hi / (L*(1-np.cos(theta0)) + 1e-9)
    frac_h = min(max(frac_h, 0), 1)
    y_ref  = -L
    y_top  = y_ref + hi
    _h_bar[0] = ax_d.fill_betweenx(
        [y_ref, y_top], [-2.05, -2.05], [-1.80, -1.80],
        color=TEAL, alpha=0.75)

    # ── labels energia no diagrama ────────────────────────────────
    ek_box.set_text(f'Eₖ = {eki:.3f} J')
    ep_box.set_text(f'Eₚ = {epi:.3f} J')
    et_box.set_text(f'E  = {eti:.3f} J')
    time_lbl.set_text(f't = {ti:.2f} s   θ = {np.degrees(theta[i]):+.1f}°'
                      f'   ω = {omega[i]:+.2f} rad/s')

    # ── gráfico energia ───────────────────────────────────────────
    cursor_e.set_xdata([ti, ti])
    ln_ek.set_data(t_arr[:i+1], Ek[:i+1])
    ln_ep.set_data(t_arr[:i+1], Ep[:i+1])
    ln_et.set_data(t_arr[:i+1], Et[:i+1])
    dot_ek.set_data([ti], [eki])
    dot_ep.set_data([ti], [epi])
    dot_et.set_data([ti], [eti])

    # ── espaço de fases ───────────────────────────────────────────
    p0 = max(0, i-80)
    phase_trail.set_data(np.degrees(theta[p0:i+1]), omega[p0:i+1])
    phase_dot.set_data([np.degrees(theta[i])], [omega[i]])
    phase_dot_ring.set_data([np.degrees(theta[i])], [omega[i]])
    # cor do ponto por energia cinética
    phase_dot.set_color(AMBER if eki > Et0*0.5 else TEAL)

    # ── ângulo × tempo ────────────────────────────────────────────
    cursor_a.set_xdata([ti, ti])
    ln_ang.set_data(t_arr[:i+1], np.degrees(theta[:i+1]))
    dot_ang.set_data([ti], [np.degrees(theta[i])])

    # ── barra de proporção ────────────────────────────────────────
    fek = eki / Et0
    fep = epi / Et0
    bar_ek.set_width(BAR_W * fek)
    bar_ep.set_x(BAR_X0 + BAR_W * fek)
    bar_ep.set_width(BAR_W * fep)

    lbl_bar_ek.set_text(f'Eₖ  {fek*100:.1f}%')
    lbl_bar_ep.set_text(f'Eₚ  {fep*100:.1f}%')
    lbl_et_b.set_text(f'E_total = {eti:.3f} J')
    lbl_om.set_text(f'v = {v_p[i]:.2f} m/s   h = {hi:.3f} m')

    return []

ani = animation.FuncAnimation(
    fig, update,
    frames=N,
    interval=18,
    blit=False,
    repeat=True
)

plt.show()
