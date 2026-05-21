"""
P�ndulo Simples — Simulação Interativa
========================================
Requer: pip install numpy matplotlib PyQt5

Controles:
  ⏸  Pausar / Continuar
  ▶  Aplicar parâmetros dos sliders
  ↺  Reiniciar para valores padrão
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.widgets import Slider, Button
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe

# ── Paleta ──────────────────────────────────────────────────────────
BG     = '#0A0B0F'
PANEL  = '#0F1118'
BORDER = '#1C2030'
AMBER  = '#F5A623'
TEAL   = '#00C9A7'
GOLD   = '#FFD166'
BLUE   = '#A8C8FF'
WHITE  = '#E8EBF4'
MUTED  = '#4A5068'
ACCENT = '#1E2440'

plt.rcParams.update({
    'font.family':      'monospace',
    'text.color':       WHITE,
    'axes.facecolor':   PANEL,
    'figure.facecolor': BG,
    'axes.edgecolor':   BORDER,
    'axes.linewidth':   0.7,
    'xtick.color':      MUTED,
    'ytick.color':      MUTED,
    'xtick.labelsize':  8,
    'ytick.labelsize':  8,
    'grid.color':       '#131825',
    'grid.linewidth':   0.6,
})

# ── Parâmetros padrão ────────────────────────────────────────────────
DEFAULTS = dict(L=1.6, m=1.0, theta0_deg=55.0, g=9.81)
DT       = 0.018
T_SIM    = 12.0
TRAIL    = 60
N_SEGS   = 14

# ── Estado global ────────────────────────────────────────────────────
state = dict(paused=False, frame=0,
             L=DEFAULTS['L'], m=DEFAULTS['m'],
             theta0_deg=DEFAULTS['theta0_deg'],
             theta0=np.radians(DEFAULTS['theta0_deg']),
             g=DEFAULTS['g'])

# ── Integração RK4 ───────────────────────────────────────────────────
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

# Cálculo inicial
t_arr, N, theta, omega, xb, yb, h_p, v_p, Ek, Ep, Et, Tp = \
    compute(state['g'], state['L'], state['m'], state['theta0'])
Et0 = Et[0]

# ── Figura ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 9), facecolor=BG)

# ┌─────────────────────────────────────────────────────────────────┐
# │  ZONAS (em coordenadas de figura, y cresce de baixo para cima) │
# │  Título:    0.93 – 1.00                                        │
# │  Painéis:   0.26 – 0.91                                        │
# │  Separador: 0.25                                               │
# │  Controles: 0.03 – 0.24                                        │
# └─────────────────────────────────────────────────────────────────┘

# Eixo do pêndulo  (coluna esquerda, ~55% da largura)
ax_d = fig.add_axes([0.02, 0.26, 0.52, 0.65])   # left=2%, bottom=26%, w=52%, h=65%

# Gráficos à direita  (coluna direita, ~38% da largura)
ax_e = fig.add_axes([0.59, 0.53, 0.38, 0.36])   # Energia × Tempo  (top=0.89)
ax_f = fig.add_axes([0.59, 0.26, 0.38, 0.23])   # Espaço de Fases  (top=0.49)

# ── Sliders: deixamos espaço generoso à esquerda para o label ───────
#    left=0.24 (label ocupa 0.03–0.23), width=0.26
SL_LEFT = 0.25
SL_W    = 0.27
SL_H    = 0.022
ax_sl_L  = fig.add_axes([SL_LEFT, 0.186, SL_W, SL_H], facecolor='#181C2C')
ax_sl_m  = fig.add_axes([SL_LEFT, 0.148, SL_W, SL_H], facecolor='#181C2C')
ax_sl_th = fig.add_axes([SL_LEFT, 0.110, SL_W, SL_H], facecolor='#181C2C')
ax_sl_g  = fig.add_axes([SL_LEFT, 0.072, SL_W, SL_H], facecolor='#181C2C')

# Labels dos sliders como texto da figura (fora dos eixos, sem corte)
SL_LX = 0.03
fig.text(SL_LX, 0.197, 'Comprimento', color=TEAL,  fontsize=8.5, va='center', fontweight='bold')
fig.text(SL_LX, 0.185, '  L  (m)',    color=MUTED, fontsize=8,   va='center')
fig.text(SL_LX, 0.159, 'Massa',       color=AMBER, fontsize=8.5, va='center', fontweight='bold')
fig.text(SL_LX, 0.147, '  m  (kg)',   color=MUTED, fontsize=8,   va='center')
fig.text(SL_LX, 0.121, 'Ângulo ini.', color=BLUE,  fontsize=8.5, va='center', fontweight='bold')
fig.text(SL_LX, 0.109, '  θ₀  (°)',  color=MUTED, fontsize=8,   va='center')
fig.text(SL_LX, 0.083, 'Gravidade',  color=GOLD,  fontsize=8.5, va='center', fontweight='bold')
fig.text(SL_LX, 0.071, '  g  (m/s²)',color=MUTED, fontsize=8,   va='center')

# ── Sliders (sem label embutido) ─────────────────────────────────────
sl_L  = Slider(ax_sl_L,  '', 0.3,  3.0,  valinit=state['L'],          color=TEAL)
sl_m  = Slider(ax_sl_m,  '', 0.1,  5.0,  valinit=state['m'],          color=AMBER)
sl_th = Slider(ax_sl_th, '', 5.0,  85.0, valinit=state['theta0_deg'], color=BLUE)
sl_g  = Slider(ax_sl_g,  '', 1.0,  25.0, valinit=state['g'],          color=GOLD)

for sl in [sl_L, sl_m, sl_th, sl_g]:
    sl.valtext.set_color(WHITE)
    sl.valtext.set_fontsize(8.5)

# ── Botões ───────────────────────────────────────────────────────────
ax_pp  = fig.add_axes([0.57, 0.155, 0.12, 0.044], facecolor=ACCENT)
ax_ap  = fig.add_axes([0.72, 0.155, 0.12, 0.044], facecolor='#0A2010')
ax_rst = fig.add_axes([0.87, 0.155, 0.12, 0.044], facecolor=ACCENT)

btn_pp  = Button(ax_pp,  '⏸  Pausar',    color=ACCENT,    hovercolor='#303870')
btn_ap  = Button(ax_ap,  '▶  Aplicar',   color='#0A2010', hovercolor='#1A4020')
btn_rst = Button(ax_rst, '↺  Reiniciar', color=ACCENT,    hovercolor='#303870')

btn_pp.label.set_color(WHITE);  btn_pp.label.set_fontsize(9)
btn_ap.label.set_color(TEAL);   btn_ap.label.set_fontsize(9)
btn_rst.label.set_color(WHITE); btn_rst.label.set_fontsize(9)

# ── Textos estáticos ─────────────────────────────────────────────────
# Título: y=0.968 (bem acima do separador em 0.93)
fig.text(0.5, 0.970, 'PÊNDULO SIMPLES — SIMULAÇÃO INTERATIVA',
         ha='center', va='top', fontsize=15, fontweight='bold', color=AMBER,
         path_effects=[pe.withStroke(linewidth=8, foreground=BG)])

# Subtítulo: y=0.943 (logo abaixo do título, acima do separador)
sub_txt = fig.text(0.5, 0.943, '',
                   ha='center', va='top', fontsize=8.5, color=MUTED)

# Separadores horizontais
fig.add_artist(plt.Line2D([0.02, 0.98], [0.930, 0.930],
    transform=fig.transFigure, color=BORDER, lw=0.8))
fig.add_artist(plt.Line2D([0.02, 0.98], [0.248, 0.248],
    transform=fig.transFigure, color=BORDER, lw=0.8))

# Separador vertical entre sliders e botões
fig.add_artist(plt.Line2D([0.555, 0.555], [0.05, 0.235],
    transform=fig.transFigure, color=BORDER, lw=0.8))

# Rótulos das seções de controle
fig.text(0.03, 0.228, 'PARÂMETROS', color=AMBER, fontsize=8.5, fontweight='bold')
fig.text(0.57, 0.228, 'CONTROLES',  color=AMBER, fontsize=8.5, fontweight='bold')

# Texto de status (abaixo dos botões)
status_txt = fig.text(0.57, 0.105, '', va='top', fontsize=8, color=MUTED)

def _update_sub():
    sub_txt.set_text(
        f'L = {state["L"]:.1f} m   ·   m = {state["m"]:.1f} kg   ·   '
        f'θ₀ = {state["theta0_deg"]:.0f}°   ·   '
        f'g = {state["g"]:.1f} m/s²   ·   T ≈ {Tp:.2f} s'
    )

_update_sub()

# ── Utilitário: estilo dos gráficos ──────────────────────────────────
def _style(ax, title, xl='', yl=''):
    ax.set_facecolor(PANEL)
    # Título com padding interno para não vazar
    ax.set_title(title, color=WHITE, fontsize=8.5, fontweight='bold', pad=5, loc='left')
    ax.set_xlabel(xl, color=MUTED, fontsize=7.5, labelpad=3)
    ax.set_ylabel(yl, color=MUTED, fontsize=7.5, labelpad=3)
    ax.grid(True)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER); sp.set_linewidth(0.7)

# ── Dicionário de objetos animados ───────────────────────────────────
aobj = {}

def rebuild():
    """Reconstrói todos os objetos com os parâmetros do estado atual."""
    global t_arr, N, theta, omega, xb, yb, h_p, v_p, Ek, Ep, Et, Tp, Et0

    t_arr, N, theta, omega, xb, yb, h_p, v_p, Ek, Ep, Et, Tp = \
        compute(state['g'], state['L'], state['m'], state['theta0'])
    Et0 = Et[0]
    state['frame'] = 0
    _update_sub()

    L      = state['L']
    bob_r  = max(0.09, L * 0.082)
    margin = max(0.38, L * 0.24)

    # ── Painel do pêndulo ────────────────────────────────────────────
    ax_d.clear()
    ax_d.set_facecolor(PANEL)
    xlim = L + margin
    ax_d.set_xlim(-xlim, xlim)
    ax_d.set_ylim(-L - margin, 0.72)
    ax_d.set_aspect('equal', adjustable='datalim')
    ax_d.axis('off')

    # Estrutura de suporte (teto)
    ax_d.fill_between([-xlim, xlim], [0.44, 0.44], [0.68, 0.68], color='#0D1020')
    for x in np.linspace(-xlim + 0.08, xlim - 0.08, 24):
        ax_d.plot([x, x + 0.10], [0.44, 0.62], color=ACCENT, lw=0.6, alpha=0.55)
    ax_d.plot([-xlim, xlim], [0.44, 0.44], color=MUTED, lw=1.8)

    # Pivô
    ax_d.plot(0, 0, 'o', color=MUTED, ms=11, zorder=10)
    ax_d.plot(0, 0, 'o', color=BG,    ms=5,  zorder=11)

    # Linha de referência h=0
    ax_d.axhline(-L, color=TEAL, lw=0.8, ls=(0, (6, 4)), alpha=0.4)
    ax_d.text(-xlim + 0.06, -L - 0.14,
              'h = 0', color=TEAL, fontsize=7.5, alpha=0.6)

    # Arco fantasma + linhas dos extremos
    arc = np.linspace(-state['theta0'], state['theta0'], 300)
    ax_d.plot(L*np.sin(arc), -L*np.cos(arc),
              color='#2A3050', lw=1.2, ls='--', alpha=0.5, zorder=2)
    for side in [-1, 1]:
        ax_d.plot([0, side*L*np.sin(state['theta0'])],
                  [0,   -L*np.cos(state['theta0'])],
                  color='#252A40', lw=0.9, alpha=0.5)

    # Rastro (fade gradiente)
    segs = []
    for k in range(N_SEGS):
        a  = 0.04 + 0.32*(k/N_SEGS)**1.8
        lw = 0.4  + 2.8 *(k/N_SEGS)**2
        seg, = ax_d.plot([], [], color=AMBER, lw=lw, alpha=a,
                         solid_capstyle='round', zorder=4)
        segs.append(seg)

    # Fio e bob
    fio, = ax_d.plot([], [], color='#7080A0', lw=2.6,
                     solid_capstyle='round', zorder=6)
    bob_p  = Circle((xb[0], yb[0]), bob_r, color=AMBER, zorder=9)
    ax_d.add_patch(bob_p)
    bob_hi = Circle((xb[0]-bob_r*0.30, yb[0]+bob_r*0.38),
                    bob_r*0.28, color='#FFFFFF', alpha=0.28, zorder=10)
    ax_d.add_patch(bob_hi)

    # Seta de velocidade
    varr = ax_d.annotate('', xy=(0,0), xytext=(0,0),
        arrowprops=dict(arrowstyle='->', color=AMBER, lw=2.2, mutation_scale=15))

    # ── Labels de energia: canto SUPERIOR ESQUERDO do painel ─────────
    #    (longe do texto de tempo que fica no rodapé)
    ek_lbl = ax_d.text(
        0.03, 0.97, '', transform=ax_d.transAxes,
        fontsize=9.0, color=AMBER, fontweight='bold', ha='left', va='top',
        bbox=dict(boxstyle='round,pad=0.42', fc='#190E00', ec=AMBER, lw=0.9, alpha=0.90))

    ep_lbl = ax_d.text(
        0.03, 0.88, '', transform=ax_d.transAxes,
        fontsize=9.0, color=TEAL, fontweight='bold', ha='left', va='top',
        bbox=dict(boxstyle='round,pad=0.42', fc='#001810', ec=TEAL, lw=0.9, alpha=0.90))

    et_lbl = ax_d.text(
        0.03, 0.79, '', transform=ax_d.transAxes,
        fontsize=9.0, color=GOLD, fontweight='bold', ha='left', va='top',
        bbox=dict(boxstyle='round,pad=0.42', fc='#181100', ec=GOLD, lw=0.9, alpha=0.90))

    # ── Info de tempo/ângulo: rodapé central (sem sobreposição) ──────
    t_lbl = ax_d.text(
        0.50, 0.022, '', transform=ax_d.transAxes,
        fontsize=8.5, color=MUTED, ha='center', va='bottom')

    # ── Gráfico Energia × Tempo ──────────────────────────────────────
    ax_e.clear()
    _style(ax_e, 'ENERGIA × TEMPO', 'tempo  (s)', 'energia  (J)')
    ax_e.set_xlim(0, T_SIM)
    ax_e.set_ylim(-0.05, Et0 * 1.30)
    ax_e.plot(t_arr, Et, color=GOLD, lw=0.8, ls='--', alpha=0.14)
    ax_e.fill_between(t_arr, 0, Ek, color=AMBER, alpha=0.05)
    ax_e.fill_between(t_arr, 0, Ep, color=TEAL,  alpha=0.05)

    ln_ek, = ax_e.plot([], [], color=AMBER, lw=2.2, label='Eₖ  cinética')
    ln_ep, = ax_e.plot([], [], color=TEAL,  lw=2.2, label='Eₚ  potencial')
    ln_et, = ax_e.plot([], [], color=GOLD,  lw=1.5, ls='--', label='E  total')
    dk  = ax_e.plot([], [], 'o', color=AMBER, ms=6, zorder=8)[0]
    dp  = ax_e.plot([], [], 'o', color=TEAL,  ms=6, zorder=8)[0]
    dtt = ax_e.plot([], [], 'o', color=GOLD,  ms=5, zorder=8)[0]
    ax_e.legend(facecolor='#08090F', edgecolor=BORDER,
                labelcolor=WHITE, fontsize=7.5, loc='upper right',
                framealpha=0.92, handlelength=1.5)

    # ── Espaço de fases ──────────────────────────────────────────────
    ax_f.clear()
    _style(ax_f, 'ESPAÇO DE FASES', 'θ  (°)', 'ω  (rad/s)')
    th_d = np.degrees(theta)
    ax_f.set_xlim(th_d.min()*1.22, th_d.max()*1.22)
    ax_f.set_ylim(omega.min()*1.22, omega.max()*1.22)
    ax_f.axhline(0, color=BORDER, lw=0.8)
    ax_f.axvline(0, color=BORDER, lw=0.8)
    ax_f.plot(th_d, omega, color='#202840', lw=1.0, alpha=0.5)

    ph_trail, = ax_f.plot([], [], color=AMBER, lw=1.8, alpha=0.70)
    ph_dot    = ax_f.plot([], [], 'o', color=AMBER, ms=8, zorder=8)[0]
    ph_ring   = ax_f.plot([], [], 'o', color=AMBER, ms=16, zorder=7,
                          markerfacecolor='none', markeredgewidth=1.2, alpha=0.30)[0]

    aobj.update(dict(
        segs=segs, fio=fio, bob=bob_p, bob_hi=bob_hi, varr=[varr],
        t_lbl=t_lbl, ek_lbl=ek_lbl, ep_lbl=ep_lbl, et_lbl=et_lbl,
        ln_ek=ln_ek, ln_ep=ln_ep, ln_et=ln_et,
        dk=dk, dp=dp, dtt=dtt,
        ph_trail=ph_trail, ph_dot=ph_dot, ph_ring=ph_ring,
    ))

rebuild()

# ── Loop de animação ─────────────────────────────────────────────────
def update(_frame):
    if state['paused']:
        return []

    state['frame'] = (state['frame'] + 1) % N
    i  = state['frame']
    xi = xb[i]; yi = yb[i]
    ti = t_arr[i]
    L  = state['L']
    br = aobj['bob'].radius

    # Fio e bob
    aobj['fio'].set_data([0, xi], [0, yi])
    aobj['bob'].center    = (xi, yi)
    aobj['bob_hi'].center = (xi - br*0.30, yi + br*0.38)

    # Cor do bob por velocidade (âmbar → branco no ponto mais rápido)
    spd = min(v_p[i] / (v_p.max() + 1e-9), 1.0)
    r0, g0, b0 = mcolors.to_rgb(AMBER)
    aobj['bob'].set_color((
        r0 + (1.00-r0)*spd**0.55,
        g0 + (1.00-g0)*spd**0.55,
        b0 + (0.92-b0)*spd**0.55,
    ))

    # Rastro com fade
    i0 = max(0, i - TRAIL); sl = i - i0 + 1
    for k, seg in enumerate(aobj['segs']):
        a = i0 + int(k/N_SEGS*sl)
        b = i0 + int((k+1)/N_SEGS*sl)
        seg.set_data(xb[a:b+1], yb[a:b+1])

    # Seta de velocidade
    aobj['varr'][0].remove()
    vx = omega[i]*L*np.cos(theta[i])*0.30
    vy = omega[i]*L*np.sin(theta[i])*0.30
    aobj['varr'][0] = ax_d.annotate(
        '', xy=(xi+vx, yi+vy), xytext=(xi, yi),
        arrowprops=dict(arrowstyle='->', color=AMBER, lw=2.2, mutation_scale=15))

    # Labels de energia (canto superior esquerdo)
    aobj['ek_lbl'].set_text(f'  Eₖ = {Ek[i]:.3f} J  ')
    aobj['ep_lbl'].set_text(f'  Eₚ = {Ep[i]:.3f} J  ')
    aobj['et_lbl'].set_text(f'   E = {Et[i]:.3f} J  ')

    # Info de tempo/ângulo (rodapé, sem sobreposição)
    aobj['t_lbl'].set_text(
        f't = {ti:.2f} s   |   θ = {np.degrees(theta[i]):+.1f}°'
        f'   |   ω = {omega[i]:+.2f} rad/s   |   v = {v_p[i]:.2f} m/s'
    )

    # Energia × tempo
    aobj['ln_ek'].set_data(t_arr[:i+1], Ek[:i+1])
    aobj['ln_ep'].set_data(t_arr[:i+1], Ep[:i+1])
    aobj['ln_et'].set_data(t_arr[:i+1], Et[:i+1])
    aobj['dk'].set_data([ti], [Ek[i]])
    aobj['dp'].set_data([ti], [Ep[i]])
    aobj['dtt'].set_data([ti], [Et[i]])

    # Espaço de fases
    p0 = max(0, i-90)
    aobj['ph_trail'].set_data(np.degrees(theta[p0:i+1]), omega[p0:i+1])
    aobj['ph_dot'].set_data([np.degrees(theta[i])], [omega[i]])
    aobj['ph_ring'].set_data([np.degrees(theta[i])], [omega[i]])
    aobj['ph_dot'].set_color(AMBER if Ek[i] > Et0*0.5 else TEAL)

    return []

# ── Callbacks ────────────────────────────────────────────────────────
def apply_params(event=None):
    state['L']          = sl_L.val
    state['m']          = sl_m.val
    state['theta0_deg'] = sl_th.val
    state['theta0']     = np.radians(sl_th.val)
    state['g']          = sl_g.val
    state['paused']     = False
    btn_pp.label.set_text('⏸  Pausar')
    status_txt.set_text('Recalculando...')
    fig.canvas.draw_idle()
    rebuild()
    status_txt.set_text(f'Aplicado  ·  T ≈ {Tp:.3f} s   ·   E₀ = {Et0:.3f} J')
    fig.canvas.draw_idle()

def toggle_pause(event):
    state['paused'] = not state['paused']
    if state['paused']:
        btn_pp.label.set_text('▶  Continuar')
        status_txt.set_text('Simulação pausada')
    else:
        btn_pp.label.set_text('⏸  Pausar')
        status_txt.set_text('')
    fig.canvas.draw_idle()

def do_reset(event):
    sl_L.reset(); sl_m.reset(); sl_th.reset(); sl_g.reset()
    apply_params()

btn_pp.on_clicked(toggle_pause)
btn_ap.on_clicked(apply_params)
btn_rst.on_clicked(do_reset)

# ── Inicia animação ──────────────────────────────────────────────────
ani = animation.FuncAnimation(
    fig, update,
    interval=18,
    blit=False,
    cache_frame_data=False,
)

plt.show()