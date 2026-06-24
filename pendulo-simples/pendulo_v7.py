#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pêndulo Simples — Simulação Interativa (v7)
============================================

Evolução do v6. Mantém a interface minimalista e os painéis minimizáveis,
mas com uma física mais completa e mais comodidades de uso:

  • Atrito viscoso opcional (γ): a equação ganha um termo -γ·θ'. Com γ = 0
    é o pêndulo ideal de sempre; com γ > 0 a energia decai e o espaço de
    fases vira uma espiral que converge para o repouso.
  • Período exato para grandes amplitudes (integral elíptica via AGM):
    o período de pequenas oscilações T₀ = 2π√(L/g) erra cada vez mais
    conforme θ₀ cresce — aqui mostramos T₀ (linear) e T (amplitude real)
    lado a lado, com o erro percentual.
  • Leitura de energia dissipada (ΔE) no painel da simulação, evidenciando
    a conservação de energia (γ = 0) ou a dissipação (γ > 0).
  • Atalhos de teclado: [espaço] pausa, [r] reset, [e]/[f] minimizam os
    painéis, [←]/[→] avançam/voltam quadro a quadro (pausado).

Requisitos:
    pip install numpy matplotlib PyQt5

Uso:
    python pendulo_v7.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.widgets import Slider, Button
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe

# ── Paleta de cores (base escura + acentos neon vibrantes) ───────────────────
BG      = '#06070C'   # quase preto: faz o neon "saltar"
SURFACE = '#0C0E18'
PANEL   = '#0E1020'
BORDER  = '#1A1F35'
BORDER2 = '#2A3150'
AMBER   = '#FFAA22'   # âmbar vivo  (bob, m, Ek, rastro)
TEAL    = '#13ECC6'   # aqua neon   (L, Ep, h=0)
GOLD    = '#FFD93D'   # ouro vivo   (energia total)
BLUE    = '#37B6FF'   # azul elétrico (θ₀)
VIOLET  = '#BE6BFF'   # violeta neon (g, presets)
WHITE   = '#E8EDFF'   # texto
MUTED   = '#3D4460'
MUTED2  = '#5A638A'
ACCENT  = '#0E1220'
RED     = '#FF466E'   # coral neon  (γ)
HOTRED  = '#FF335F'   # bob em alta velocidade

plt.rcParams.update({
    'font.family':      'sans-serif',
    'font.sans-serif':  ['Segoe UI', 'DejaVu Sans'],
    'font.monospace':   ['Consolas', 'DejaVu Sans Mono'],
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

# Liberamos as teclas que o matplotlib usa por padrão para que nossos
# atalhos ([f], [←], [→]) não disparem fullscreen/navegação de histórico.
for _km in ('keymap.fullscreen', 'keymap.back', 'keymap.forward'):
    plt.rcParams[_km] = []

# ── Parâmetros ────────────────────────────────────────────────────────────────
DEFAULTS = dict(L=1.6, m=1.0, theta0_deg=55.0, g=9.81, gamma=0.0)
DT       = 0.016
T_SIM    = 16.0
TRAIL    = 80
N_SEGS   = 18

# Presets de gravidade, em ordem crescente de g (m/s²).
PLANETS = [
    ('Lua',     1.62),
    ('Marte',   3.71),
    ('Terra',   9.81),
    ('Júpiter', 24.79),
]

state = dict(
    paused=False, frame=0,
    L=DEFAULTS['L'], m=DEFAULTS['m'],
    theta0_deg=DEFAULTS['theta0_deg'],
    theta0=np.radians(DEFAULTS['theta0_deg']),
    g=DEFAULTS['g'], gamma=DEFAULTS['gamma'],
    min_e=False, min_f=False,
)

# ── Período exato (integral elíptica completa via média aritmético-geométrica) ─
def _agm(a, b, iters=24):
    """Média aritmético-geométrica de a e b — converge quadraticamente."""
    for _ in range(iters):
        a, b = 0.5 * (a + b), np.sqrt(a * b)
    return a

def exact_period(g, L, theta0):
    """Período exato do pêndulo (sem atrito) para amplitude θ₀ qualquer.

    T = 4√(L/g)·K(sin(θ₀/2)), e K(k) = π / (2·AGM(1, cos(θ₀/2))), logo
    T = 2π√(L/g) / AGM(1, cos(θ₀/2)). Para θ₀ → 0 recai em 2π√(L/g).
    """
    return 2 * np.pi * np.sqrt(L / g) / _agm(1.0, np.cos(theta0 / 2.0))

# ── Integração numérica (RK4) ─────────────────────────────────────────────────
def compute(g: float, L: float, m: float, theta0: float, gamma: float):
    """Integra o pêndulo (com atrito viscoso opcional) via Runge-Kutta de 4ª ordem.

    Equação do movimento: θ'' = -(g/L)·sin(θ) - γ·θ'

    Args:
        g: aceleração da gravidade (m/s²).
        L: comprimento do fio (m).
        m: massa do bob (kg).
        theta0: ângulo inicial, em radianos.
        gamma: coeficiente de atrito viscoso (1/s); 0 = pêndulo ideal.

    Returns:
        Tupla (t, Ns, theta, omega, xb, yb, h, v, Ek, Ep, Et, T_lin, T_amp),
        onde xb/yb são as coordenadas do bob, h é a altura acima do ponto
        mais baixo, v é a velocidade linear, Ek/Ep/Et são as energias
        cinética/potencial/total, T_lin é o período de pequenas oscilações
        e T_amp é o período exato corrigido pela amplitude (sem atrito).
    """
    t  = np.arange(0, T_SIM, DT)
    Ns = len(t)
    th = np.zeros(Ns); om = np.zeros(Ns)
    th[0] = theta0
    for i in range(1, Ns):
        def d(a, w): return w, -(g / L) * np.sin(a) - gamma * w
        k1 = d(th[i-1],             om[i-1])
        k2 = d(th[i-1]+DT/2*k1[0], om[i-1]+DT/2*k1[1])
        k3 = d(th[i-1]+DT/2*k2[0], om[i-1]+DT/2*k2[1])
        k4 = d(th[i-1]+DT  *k3[0], om[i-1]+DT  *k3[1])
        th[i] = th[i-1] + DT/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
        om[i] = om[i-1] + DT/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
    xb = L*np.sin(th); yb = -L*np.cos(th)
    hp = L*(1-np.cos(th)); vp = L*np.abs(om)
    Ek = 0.5*m*vp**2; Ep = m*g*hp; Et = Ek+Ep
    T_lin = 2*np.pi*np.sqrt(L/g)
    T_amp = exact_period(g, L, theta0)
    return t, Ns, th, om, xb, yb, hp, vp, Ek, Ep, Et, T_lin, T_amp

t_arr, N, theta, omega, xb, yb, h_p, v_p, Ek, Ep, Et, Tp_lin, Tp_amp = \
    compute(state['g'], state['L'], state['m'], state['theta0'], state['gamma'])
Et0 = Et[0]

# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
#  [  simulação (esq)  ] [ ▾ energia×tempo (toggle)    ]
#  [                   ] [   energia×tempo (gráfico)       ]
#  [                   ] [ ▾ espaço de fases (toggle)  ]
#  [                   ] [   espaço de fases (gráfico)      ]
#  [                   ] [─────────── painel ─────────]
#  [                   ] [ slider L   ████░░  1.6 m   ]
#  [                   ] [ slider m   ██░░░░  1.0 kg  ]
#  [                   ] [ slider θ₀  ████░░  55 °    ]
#  [                   ] [ slider g   ███░░░  9.8 m/s²]
#  [                   ] [ slider γ   ░░░░░░  0.0 /s  ]
#  [                   ] [ Lua  Marte  Terra  Júpiter ]
#  [                   ] [ [Pausar]  [Aplicar] [Reset] ]
# ═══════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(17, 9.5), facecolor=BG)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
try:
    fig.canvas.manager.set_window_title('Pêndulo Simples — v7')
except Exception:
    pass

HEADER_BOT = 0.845   # onde o cabeçalho termina

# Margens externas simétricas (esquerda/direita) para a figura toda
MARGIN_L = 0.012
MARGIN_R = 0.012

# Coluna esquerda (simulação)
COL_L_X    = MARGIN_L
COL_L_W    = 0.505

# Coluna direita (gráficos + painel)
COL_R_X    = 0.555
COL_R_R    = 1.0 - MARGIN_R     # right da coluna direita
COL_R_W    = COL_R_R - COL_R_X

# ── Painel de controle: alturas das linhas (calculadas de baixo pra cima) ────
CTRL_BOT = 0.040
BTN_H    = 0.038
BTN_G    = 0.022   # gap maior: dá respiro pro status_txt entre os botões
                   # principais e a linha de presets de gravidade
y_btn    = CTRL_BOT                              # linha de botões principais

PLANET_ROW_H = 0.030
PLANET_GAP   = 0.013
y_planets    = y_btn + BTN_H + BTN_G             # linha de presets de gravidade

LINE_H = 0.027
LINE_G = 0.013
N_SLIDERS = 5
y_sl   = [y_planets + PLANET_ROW_H + PLANET_GAP + i*(LINE_H + LINE_G)
          for i in range(N_SLIDERS)]

CTRL_TOP = y_sl[-1] + LINE_H + 0.018   # top do painel de controles (= base do ax_f)

# ── Gráficos da coluna direita: cada um tem uma barra de título/toggle ───────
# As posições reais de ax_e/ax_f e das barras de toggle são calculadas
# dinamicamente em relayout() — ao minimizar um painel, o outro cresce pra
# ocupar o espaço liberado, em vez de deixar um vão vazio na tela.
TOGGLE_H   = 0.028
GAP_F_CTRL = 0.055   # espaço entre ax_f e CTRL_TOP, pro xlabel "ângulo (°)"
GAP_TOGGLE = 0.022   # espaço entre o gráfico e a barra de toggle acima dele —
                      # precisa caber o rótulo de unidade (ex.: "E (J)") que
                      # fica no canto superior do gráfico, senão colide com a barra
GAP_PANELS = 0.032   # espaço entre a barra de toggle de baixo e o eixo X
                      # (tick labels) do gráfico de cima — precisa ser maior
                      # que GAP_TOGGLE pra não colidir com os números do eixo
AX_F_H_DEFAULT = 0.180   # altura do espaço de fases quando os dois estão abertos

ax_d = fig.add_axes([COL_L_X, 0.04, COL_L_W, HEADER_BOT - 0.04])   # simulação
ax_e = fig.add_axes([COL_R_X, 0.5, COL_R_W, 0.1])   # energia × tempo (posição real via relayout())
ax_f = fig.add_axes([COL_R_X, 0.5, COL_R_W, 0.1])   # espaço de fases  (idem)

# ── Barras de título/toggle (minimizam cada gráfico) ─────────────────────────
def make_panel_toggle(title):
    bax = fig.add_axes([COL_R_X, 0.5, COL_R_W, TOGGLE_H], facecolor='#0B0D1A')
    for sp in bax.spines.values():
        sp.set_edgecolor(BORDER2); sp.set_linewidth(0.6)
    btn = Button(bax, f'▼  {title}', color='#0B0D1A', hovercolor='#141A30')
    btn.label.set_color(MUTED2)
    btn.label.set_fontsize(8)
    btn.label.set_fontweight('bold')
    return bax, btn

toggle_e_ax, toggle_e_btn = make_panel_toggle('ENERGIA × TEMPO')
toggle_f_ax, toggle_f_btn = make_panel_toggle('ESPAÇO DE FASES')

def relayout():
    """Recalcula a altura/posição de ax_e e ax_f conforme o estado de
    minimização: o painel aberto ocupa o espaço que o outro liberou ao
    minimizar, em vez de deixar um vão vazio no meio da tela."""
    min_e, min_f = state['min_e'], state['min_f']
    # Os gaps extras (GAP_F_CTRL/GAP_PANELS) só existem pra dar espaço aos
    # tick labels/xlabel de cada gráfico quando ele está aberto — minimizado,
    # encolhem pro tamanho mínimo (GAP_TOGGLE) e o vão vazio desaparece.
    gap_bottom = GAP_F_CTRL if not min_f else GAP_TOGGLE
    gap_panels = GAP_PANELS if not min_e else GAP_TOGGLE
    total_span = HEADER_BOT - CTRL_TOP - gap_bottom
    overhead   = 2*TOGGLE_H + 2*GAP_TOGGLE + gap_panels
    avail      = total_span - overhead

    if min_e and min_f:
        f_h, e_h = 0.0, 0.0
    elif min_f:
        f_h, e_h = 0.0, avail
    elif min_e:
        f_h, e_h = avail, 0.0
    else:
        f_h = AX_F_H_DEFAULT
        e_h = avail - f_h

    # Construído de cima (HEADER_BOT) para baixo: o espaço de um painel
    # minimizado some daqui, empurrando tudo pra cima — o vão sobra encostado
    # no painel de parâmetros (embaixo), não no topo da tela.
    cursor = HEADER_BOT
    cursor -= TOGGLE_H
    toggle_e_ax.set_position([COL_R_X, cursor, COL_R_W, TOGGLE_H])
    cursor -= GAP_TOGGLE + e_h
    ax_e.set_position([COL_R_X, cursor, COL_R_W, max(e_h, 1e-4)])
    ax_e.set_visible(not min_e)
    cursor -= gap_panels + TOGGLE_H
    toggle_f_ax.set_position([COL_R_X, cursor, COL_R_W, TOGGLE_H])
    cursor -= GAP_TOGGLE + f_h
    ax_f.set_position([COL_R_X, cursor, COL_R_W, max(f_h, 1e-4)])
    ax_f.set_visible(not min_f)

    # `cursor` agora é a borda de baixo do que está visível nos gráficos.
    # gap_bottom já é o respiro mínimo que ax_f precisa ali (cheio se aberto,
    # mínimo se minimizado) — o que sobrar disso até CTRL_BOT (rodapé fixo)
    # define o novo "topo" do painel; escalamos os gaps internos pra ocupar
    # esse espaço por completo, sem deixar vão vazio em lugar nenhum.
    new_ctrl_top = cursor - gap_bottom
    scale = (new_ctrl_top - CTRL_BOT) / (CTRL_TOP_BASE - CTRL_BOT)
    reflow_controls(max(1.0, scale))

def toggle_panel(key, btn, title):
    """Minimiza/restaura um painel de gráfico (usado por clique e por teclado)."""
    state[key] = not state[key]
    sym = '►' if state[key] else '▼'
    btn.label.set_text(f'{sym}  {title}')
    relayout()
    fig.canvas.draw_idle()

toggle_e_btn.on_clicked(lambda e: toggle_panel('min_e', toggle_e_btn, 'ENERGIA × TEMPO'))
toggle_f_btn.on_clicked(lambda e: toggle_panel('min_f', toggle_f_btn, 'ESPAÇO DE FASES'))

# ── Painel de controle: estica pra preencher o espaço liberado, sem deixar
# vão vazio em lugar nenhum ─────────────────────────────────────────────────
# Cada elemento abaixo (sliders, botões, labels...) é registrado aqui com sua
# posição "base" (a posição original, de quando os dois gráficos estão
# abertos). reflow_controls() escala tudo a partir do rodapé fixo (CTRL_BOT):
# os gaps entre as linhas crescem proporcionalmente pra ocupar exatamente o
# espaço entre a última barra de toggle e o rodapé — nunca sobra vão vazio.
CTRL_TOP_BASE = CTRL_TOP   # topo original (gráficos totalmente abertos)
CTRL_ARTISTS  = []

def register_ctrl(artist, kind, base, anchor=None, offset=0.0):
    """anchor é a posição (escalável) de referência da linha/elemento a que
    este artista pertence (ex.: o y0 do slider da mesma linha); offset é a
    distância FIXA até essa referência (ex.: meia altura do slider, pro
    label ficar centralizado) — offset nunca é escalado, só o anchor."""
    if anchor is None:
        anchor = base[1] if kind != 'line' else base
    CTRL_ARTISTS.append((artist, kind, base, anchor, offset))
    return artist

def reflow_controls(scale):
    for artist, kind, base, anchor, offset in CTRL_ARTISTS:
        new_anchor = CTRL_BOT + (anchor - CTRL_BOT) * scale + offset
        if kind == 'axes':
            x0, _, w, h = base
            artist.set_position([x0, new_anchor, w, h])
        elif kind == 'text':
            x0, _ = base
            artist.set_position((x0, new_anchor))
        elif kind == 'line':
            artist.set_ydata([new_anchor, new_anchor])

# ── Painel de controle (direita-baixo) ───────────────────────────────────────
# 5 sliders com label + barra integrada + valor numérico
# Cada linha: [LABEL] [══slider══] [valor + unidade]

CTRL_L = COL_R_X + 0.003   # left da área de controle
CTRL_R = COL_R_R           # right
CTRL_W = CTRL_R - CTRL_L

# Largura do slider em si (deixa espaço pro label e pro valor)
LBL_W  = 0.055   # largura do label (L, m, θ₀, g, γ)
VAL_W  = 0.072   # largura da área de valor numérico
SL_W   = CTRL_W - LBL_W - VAL_W - 0.006

SL_X   = CTRL_L + LBL_W + 0.003

params = [
    # (label, unidade, vmin, vmax, valinit, cor, formato)
    ('L',  'm',    0.3,  3.0,  state['L'],          TEAL,   '.2f'),
    ('m',  'kg',   0.1,  5.0,  state['m'],          AMBER,  '.1f'),
    ('θ₀', '°',    5.0,  85.0, state['theta0_deg'], BLUE,   '.0f'),
    ('g',  'm/s²', 1.0,  25.0, state['g'],          VIOLET, '.2f'),
    ('γ',  '/s',   0.0,  2.0,  state['gamma'],      RED,    '.2f'),
]

slider_axes = []
sliders     = []
val_texts   = []   # (text_artist, unidade, formato) — atualizados em tempo real

for i, (lbl, unit, vmin, vmax, vinit, col, fmt) in enumerate(params):
    # y_sl é de baixo pra cima; invertendo o índice, params fica na ordem
    # de leitura (L no topo → γ na base), igual ao cabeçalho.
    y = y_sl[N_SLIDERS - 1 - i]

    # Label do parâmetro
    lbl_txt = fig.text(CTRL_L + 0.002, y + LINE_H/2, lbl,
             color=col, fontsize=9, fontweight='bold', va='center')
    register_ctrl(lbl_txt, 'text', (CTRL_L + 0.002, y), anchor=y, offset=LINE_H/2)

    # Eixo do slider
    ax_s = fig.add_axes([SL_X, y, SL_W, LINE_H], facecolor=ACCENT)
    register_ctrl(ax_s, 'axes', (SL_X, y, SL_W, LINE_H), anchor=y)
    for sp in ax_s.spines.values():
        sp.set_edgecolor(BORDER2); sp.set_linewidth(0.7)
    sl = Slider(ax_s, '', vmin, vmax, valinit=vinit, color=col, initcolor='none')
    sl.valtext.set_visible(False)   # ocultamos o valor padrão do slider
    sl.label.set_visible(False)
    # Trilha escura integrada ao tema (em vez do cinza-claro padrão) e knob
    # vazado no tom do próprio parâmetro — a barra colorida passa a destacar.
    sl.track.set(facecolor='#161B2E', edgecolor='none')
    sl.poly.set_edgecolor('none')
    sl._handle.set(markerfacecolor=SURFACE, markeredgecolor=col,
                   markeredgewidth=1.6, markersize=10)
    slider_axes.append(ax_s)
    sliders.append(sl)

    # Valor numérico à direita (atualizado dinamicamente)
    vt = fig.text(SL_X + SL_W + 0.005, y + LINE_H/2,
                  f'{vinit:{fmt}} {unit}',
                  color=WHITE, fontsize=8.5, va='center',
                  fontfamily='monospace')
    register_ctrl(vt, 'text', (SL_X + SL_W + 0.005, y), anchor=y, offset=LINE_H/2)
    val_texts.append((vt, unit, fmt))

sl_L, sl_m, sl_th, sl_g, sl_gam = sliders

# Atualiza textos de valor quando slider move
def _upd_val(i):
    vt, unit, fmt = val_texts[i]
    vt.set_text(f'{sliders[i].val:{fmt}} {unit}')
    fig.canvas.draw_idle()

for i, sl in enumerate(sliders):
    sl.on_changed(lambda v, i=i: _upd_val(i))

# ── Presets de gravidade (Lua / Marte / Terra / Júpiter) ─────────────────────
N_PLANETS = len(PLANETS)
PL_GAP = 0.008
PL_W   = (CTRL_W - (N_PLANETS-1)*PL_GAP) / N_PLANETS

planet_axes = []
planet_btns = []

def _make_planet_cb(g_val):
    def _cb(event):
        sl_g.set_val(g_val)
        apply_params()
    return _cb

for i, (pname, gval) in enumerate(PLANETS):
    px = CTRL_L + i*(PL_W + PL_GAP)
    pax = fig.add_axes([px, y_planets, PL_W, PLANET_ROW_H], facecolor=ACCENT)
    register_ctrl(pax, 'axes', (px, y_planets, PL_W, PLANET_ROW_H))
    for sp in pax.spines.values():
        sp.set_edgecolor(BORDER2); sp.set_linewidth(0.6)
    pbtn = Button(pax, pname, color=ACCENT, hovercolor='#1E2550')
    pbtn.label.set_color(VIOLET)
    pbtn.label.set_fontsize(7.6)
    pbtn.on_clicked(_make_planet_cb(gval))
    planet_axes.append(pax)
    planet_btns.append(pbtn)

def _refresh_planet_highlight():
    """Realça o preset cujo g coincide com o g atual — afordância de qual
    'planeta' está selecionado."""
    for (pname, gval), pax, pbtn in zip(PLANETS, planet_axes, planet_btns):
        active = abs(gval - state['g']) < 1e-6
        pax.set_facecolor('#1C1633' if active else ACCENT)
        pbtn.color = '#1C1633' if active else ACCENT
        pbtn.label.set_color(GOLD if active else VIOLET)
        pbtn.label.set_fontweight('bold' if active else 'normal')
        for sp in pax.spines.values():
            sp.set_edgecolor(VIOLET if active else BORDER2)
            sp.set_linewidth(1.2 if active else 0.6)

_refresh_planet_highlight()

# ── Botões principais ─────────────────────────────────────────────────────────
N_BTN  = 3
BTN_W  = (CTRL_W - 0.012) / N_BTN

ax_pp  = fig.add_axes([CTRL_L,                    y_btn, BTN_W, BTN_H], facecolor=ACCENT)
ax_ap  = fig.add_axes([CTRL_L + BTN_W + 0.006,   y_btn, BTN_W, BTN_H], facecolor='#0A2015')
ax_rst = fig.add_axes([CTRL_L + 2*(BTN_W+0.006), y_btn, BTN_W, BTN_H], facecolor=ACCENT)

register_ctrl(ax_pp,  'axes', (CTRL_L,                  y_btn, BTN_W, BTN_H))
register_ctrl(ax_ap,  'axes', (CTRL_L + BTN_W + 0.006,  y_btn, BTN_W, BTN_H))
register_ctrl(ax_rst, 'axes', (CTRL_L + 2*(BTN_W+0.006), y_btn, BTN_W, BTN_H))

for ax_b in [ax_pp, ax_ap, ax_rst]:
    for sp in ax_b.spines.values():
        sp.set_edgecolor(BORDER2); sp.set_linewidth(0.7)

btn_pp  = Button(ax_pp,  '‖  Pausar', color=ACCENT,    hovercolor='#1E2550')
btn_ap  = Button(ax_ap,  'Aplicar',   color='#0A2015', hovercolor='#143320')
btn_rst = Button(ax_rst, 'Reset',     color=ACCENT,    hovercolor='#1E2550')

btn_pp.label.set_color(MUTED2);  btn_pp.label.set_fontsize(9)
btn_ap.label.set_color(TEAL);    btn_ap.label.set_fontsize(9)
btn_rst.label.set_color(MUTED2); btn_rst.label.set_fontsize(9)

# ── Cabeçalho (faixa dedicada no topo, acima de tudo) ────────────────────────
TITLE_X = 0.5
fig.text(TITLE_X, 0.994, 'PÊNDULO SIMPLES',
         ha='center', va='top', fontsize=17, fontweight='bold', color=AMBER,
         path_effects=[pe.withStroke(linewidth=10, foreground=BG)])
fig.text(TITLE_X, 0.962,
         'simulação interativa  ·  runge-kutta 4ª ordem  ·  atrito viscoso opcional',
         ha='center', va='top', fontsize=8, color=MUTED2)

sub_txt = fig.text(TITLE_X, 0.934, '', ha='center', va='top', fontsize=8, color='#7E88B0')

status_txt = fig.text(CTRL_L, CTRL_BOT + BTN_H + 0.005, '',
                      va='bottom', fontsize=7.5, color=MUTED2)
register_ctrl(status_txt, 'text', (CTRL_L, y_btn), anchor=y_btn, offset=BTN_H + 0.005)

# Dica de atalhos de teclado (rodapé da coluna esquerda, bem discreto)
fig.text(MARGIN_L, 0.013,
         'atalhos:   [espaço] pausar   ·   [r] reset   ·   '
         '[e]/[f] painéis   ·   [←]/[→] passo a passo (pausado)',
         fontsize=7, color=MUTED, va='bottom', ha='left')

# Separadores
fig.add_artist(plt.Line2D([MARGIN_L, 1.0 - MARGIN_R], [HEADER_BOT, HEADER_BOT],
    transform=fig.transFigure, color=BORDER, lw=0.7))
SEP_X = (COL_L_X + COL_L_W + COL_R_X) / 2
fig.add_artist(plt.Line2D([SEP_X, SEP_X], [0.03, HEADER_BOT],
    transform=fig.transFigure, color=BORDER, lw=0.7))
ctrl_sep_line = plt.Line2D([COL_R_X, COL_R_R], [CTRL_TOP, CTRL_TOP],
    transform=fig.transFigure, color=BORDER2, lw=0.6)
fig.add_artist(ctrl_sep_line)
register_ctrl(ctrl_sep_line, 'line', CTRL_TOP)

params_lbl = fig.text(CTRL_L, CTRL_TOP - 0.002,
         '▪  PARÂMETROS', color=MUTED2, fontsize=7.5, fontweight='bold',
         va='top')
register_ctrl(params_lbl, 'text', (CTRL_L, CTRL_TOP), anchor=CTRL_TOP, offset=-0.002)

relayout()

def _update_sub():
    dpct = (Tp_amp / Tp_lin - 1.0) * 100.0
    sub_txt.set_text(
        'L = %.2f m   ·   m = %.1f kg   ·   θ₀ = %.0f°   ·   '
        'g = %.2f m/s²   ·   γ = %.2f /s        '
        'T₀ = %.3f s (linear)   ·   T = %.3f s (amplitude, %+.1f%%)'
        % (state['L'], state['m'], state['theta0_deg'], state['g'],
           state['gamma'], Tp_lin, Tp_amp, dpct)
    )

_update_sub()

def _style(ax, xl='', yl=''):
    """Estilo minimalista: sem título embutido (a barra de toggle já cumpre
    esse papel) e rótulo Y horizontal no canto, evitando colisão com o spine."""
    ax.set_facecolor(PANEL)
    ax.set_xlabel(xl, color=MUTED, fontsize=7.5, labelpad=3)
    if yl:
        ax.text(1.0, 1.03, yl, transform=ax.transAxes,
                color=MUTED, fontsize=7, ha='right', va='bottom')
    ax.grid(True, alpha=0.5)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER2); sp.set_linewidth(0.6)

aobj = {}

def rebuild():
    global t_arr, N, theta, omega, xb, yb, h_p, v_p, Ek, Ep, Et, Tp_lin, Tp_amp, Et0

    t_arr, N, theta, omega, xb, yb, h_p, v_p, Ek, Ep, Et, Tp_lin, Tp_amp = \
        compute(state['g'], state['L'], state['m'], state['theta0'], state['gamma'])
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

    # Bob (com halo neon em duas camadas — reage à velocidade no update)
    bob_p   = Circle((xb[0], yb[0]), bob_r,       color=AMBER, zorder=10)
    bob_gl  = Circle((xb[0], yb[0]), bob_r*1.9,   color=AMBER, alpha=0.14, zorder=9)
    bob_gl2 = Circle((xb[0], yb[0]), bob_r*3.0,   color=AMBER, alpha=0.06, zorder=8.5)
    bob_hi  = Circle((xb[0]-bob_r*0.28, yb[0]+bob_r*0.35), bob_r*0.26,
                    color='#FFFFFF', alpha=0.32, zorder=11)
    for p in [bob_gl2, bob_gl, bob_p, bob_hi]:
        ax_d.add_patch(p)

    # Vetor velocidade
    varr = ax_d.quiver(xb[0], yb[0], 0, 0,
                       color=AMBER, scale=1, scale_units='xy', angles='xy',
                       width=0.010, headwidth=5, headlength=6,
                       headaxislength=5, zorder=12, alpha=0.80)

    # Painel de energia (canto superior direito da simulação): cabeçalho +
    # 4 leituras com marcadores que ligam Ek/Ep/E às linhas do gráfico.
    PANEL_W, PANEL_H = 0.240, 0.205
    PANEL_X, PANEL_Y = 1 - 0.025 - PANEL_W, 0.773
    e_panel = FancyBboxPatch((PANEL_X, PANEL_Y), PANEL_W, PANEL_H,
        transform=ax_d.transAxes, boxstyle='round,pad=0.010,rounding_size=0.016',
        facecolor='#0A0C16', edgecolor=BORDER2, linewidth=0.8, alpha=0.92,
        zorder=14)
    ax_d.add_patch(e_panel)

    DOT_X = PANEL_X + 0.018
    TXT_X = PANEL_X + 0.034
    VAL_X = PANEL_X + PANEL_W - 0.020
    ax_d.text(DOT_X, PANEL_Y + PANEL_H*0.88, 'ENERGIA', transform=ax_d.transAxes,
        fontsize=6.8, color=MUTED2, fontweight='bold', ha='left', va='center', zorder=15)
    ax_d.text(VAL_X, PANEL_Y + PANEL_H*0.88, 'joules', transform=ax_d.transAxes,
        fontsize=6.5, color=MUTED, ha='right', va='center', zorder=15)
    ax_d.plot([DOT_X, VAL_X], [PANEL_Y + PANEL_H*0.76]*2, transform=ax_d.transAxes,
        color=BORDER2, lw=0.6, zorder=15)

    ROW_Y = [PANEL_Y + PANEL_H*yy for yy in (0.60, 0.43, 0.26, 0.09)]
    e_rows = [('Ek', AMBER, True), ('Ep', TEAL, True),
              ('E', GOLD, True),  ('ΔE', MUTED2, False)]
    e_vals = []
    for (name, col, dot), ry in zip(e_rows, ROW_Y):
        if dot:
            ax_d.plot([DOT_X], [ry], marker='o', ms=4.5, color=col,
                      transform=ax_d.transAxes, zorder=15, clip_on=False)
        ax_d.text(TXT_X, ry, name, transform=ax_d.transAxes,
            fontsize=7.4, color=col, fontweight='bold', ha='left', va='center', zorder=15)
        vt_e = ax_d.text(VAL_X, ry, '', transform=ax_d.transAxes,
            fontsize=7.6, color=col, fontweight='bold', ha='right', va='center',
            zorder=15, fontfamily='monospace')
        e_vals.append(vt_e)
    ek_val, ep_val, et_val, ed_val = e_vals

    # Info de rodapé dentro da simulação
    t_lbl = ax_d.text(0.50, 0.015, '', transform=ax_d.transAxes,
        fontsize=7.5, color=MUTED2, ha='center', va='bottom')

    # ── Energia × Tempo (minimalista: 2 linhas + total tracejado + ref. E₀) ────
    ax_e.clear()
    _style(ax_e, 't (s)', 'E (J)')
    ax_e.set_xlim(0, T_SIM)
    ax_e.set_ylim(-0.03, Et0 * 1.18)
    ax_e.axhline(Et0, color=MUTED, lw=0.6, ls=':', alpha=0.45, zorder=2)

    ln_ek, = ax_e.plot([], [], color=AMBER, lw=1.8, alpha=0.9)
    ln_ep, = ax_e.plot([], [], color=TEAL,  lw=1.8, alpha=0.9)
    ln_et, = ax_e.plot([], [], color=GOLD,  lw=1.0, ls='--', alpha=0.5)

    # ── Espaço de Fases (minimalista: órbita de fundo + ponto + equilíbrio) ────
    ax_f.clear()
    _style(ax_f, 'ângulo (°)', 'vel. angular (rad/s)')
    th_d = np.degrees(theta)
    pad_x = max(2, (th_d.max()-th_d.min()) * 0.14)
    pad_y = max(0.1, (omega.max()-omega.min()) * 0.14)
    ax_f.set_xlim(th_d.min()-pad_x, th_d.max()+pad_x)
    ax_f.set_ylim(omega.min()-pad_y, omega.max()+pad_y)
    ax_f.axhline(0, color=BORDER2, lw=0.8, alpha=0.6, zorder=2)
    ax_f.axvline(0, color=BORDER2, lw=0.8, alpha=0.6, zorder=2)
    ax_f.plot(th_d, omega, color=MUTED, lw=1.2, alpha=0.55, zorder=3)
    # Ponto de equilíbrio (atrator quando há atrito)
    ax_f.plot(0, 0, marker='o', mfc='none', mec=MUTED2, ms=6, mew=0.9, zorder=4)

    ph_trail, = ax_f.plot([], [], color=AMBER, lw=1.8, alpha=0.85,
                          solid_capstyle='round', zorder=8)
    ph_dot    = ax_f.plot([], [], 'o', color=AMBER, ms=6, zorder=9,
                          markeredgecolor=WHITE, markeredgewidth=0.5)[0]

    aobj.update(dict(
        segs=segs, fio=fio, fio2=fio2,
        bob=bob_p, bob_gl=bob_gl, bob_gl2=bob_gl2, bob_hi=bob_hi,
        varr=varr,
        t_lbl=t_lbl, ek_val=ek_val, ep_val=ep_val, et_val=et_val, ed_val=ed_val,
        ln_ek=ln_ek, ln_ep=ln_ep, ln_et=ln_et,
        ph_trail=ph_trail, ph_dot=ph_dot,
    ))

rebuild()

def _draw_frame(i):
    """Desenha o quadro i (usado tanto pela animação quanto pelo passo manual)."""
    xi = xb[i]; yi = yb[i]
    ti = t_arr[i]
    L  = state['L']
    br = aobj['bob'].radius

    aobj['fio'].set_data([0, xi],  [0, yi])
    aobj['fio2'].set_data([0, xi], [0, yi])

    for patch, off in [
        (aobj['bob'],     (0, 0)),
        (aobj['bob_gl'],  (0, 0)),
        (aobj['bob_gl2'], (0, 0)),
        (aobj['bob_hi'],  (-br*0.28, br*0.35)),
    ]:
        patch.center = (xi + off[0], yi + off[1])

    spd = np.clip(v_p[i] / (v_p.max() + 1e-9), 0, 1)
    r0,g0,b0 = mcolors.to_rgb(AMBER)
    rt,gt,bt = mcolors.to_rgb(HOTRED)
    bcol = (r0+(rt-r0)*spd**0.7, g0+(gt-g0)*spd**0.7, b0+(bt-b0)*spd**0.7)
    aobj['bob'].set_color(bcol)
    # halo neon acompanha a cor e ganha intensidade com a velocidade
    aobj['bob_gl'].set_color(bcol);  aobj['bob_gl'].set_alpha(0.12 + 0.20*spd)
    aobj['bob_gl2'].set_color(bcol); aobj['bob_gl2'].set_alpha(0.05 + 0.11*spd)

    i0 = max(0, i - TRAIL); sl_len = i - i0 + 1
    for k, seg in enumerate(aobj['segs']):
        a = i0 + int(k/N_SEGS*sl_len)
        b = i0 + int((k+1)/N_SEGS*sl_len)
        seg.set_data(xb[a:b+1], yb[a:b+1])

    vx = omega[i]*L*np.cos(theta[i])*0.28
    vy = omega[i]*L*np.sin(theta[i])*0.28
    aobj['varr'].set_offsets([[xi, yi]])
    aobj['varr'].set_UVC(vx, vy)

    aobj['ek_val'].set_text(f'{Ek[i]:.3f}')
    aobj['ep_val'].set_text(f'{Ep[i]:.3f}')
    aobj['et_val'].set_text(f'{Et[i]:.3f}')
    aobj['ed_val'].set_text(f'{Et0-Et[i]:.3f}')

    aobj['t_lbl'].set_text(
        't = %.2f s   ·   θ = %+.1f°   ·   '
        'ω = %+.2f rad/s   ·   v = %.2f m/s'
        % (ti, np.degrees(theta[i]), omega[i], v_p[i])
    )

    aobj['ln_ek'].set_data(t_arr[:i+1], Ek[:i+1])
    aobj['ln_ep'].set_data(t_arr[:i+1], Ep[:i+1])
    aobj['ln_et'].set_data(t_arr[:i+1], Et[:i+1])

    p0 = max(0, i - 110)
    th_now = np.degrees(theta[i])
    aobj['ph_trail'].set_data(np.degrees(theta[p0:i+1]), omega[p0:i+1])
    aobj['ph_dot'].set_data([th_now], [omega[i]])
    col = AMBER if Ek[i] > Et0*0.5 else TEAL
    aobj['ph_dot'].set_color(col)

def update(_frame):
    if state['paused']:
        return []
    state['frame'] = (state['frame'] + 1) % N
    _draw_frame(state['frame'])
    return []

def apply_params(event=None):
    state['L']          = sl_L.val
    state['m']          = sl_m.val
    state['theta0_deg'] = sl_th.val
    state['theta0']     = np.radians(sl_th.val)
    state['g']          = sl_g.val
    state['gamma']      = sl_gam.val
    state['paused']     = False
    btn_pp.label.set_text('‖  Pausar')
    btn_pp.label.set_color(MUTED2)
    status_txt.set_text('recalculando…')
    fig.canvas.draw_idle()
    rebuild()
    _refresh_planet_highlight()
    status_txt.set_text('•  T = %.3f s   ·   E₀ = %.3f J' % (Tp_amp, Et0))
    fig.canvas.draw_idle()

def toggle_pause(event=None):
    state['paused'] = not state['paused']
    if state['paused']:
        btn_pp.label.set_text('►  Continuar')
        btn_pp.label.set_color(TEAL)
        status_txt.set_text('‖  pausado')
    else:
        btn_pp.label.set_text('‖  Pausar')
        btn_pp.label.set_color(MUTED2)
        status_txt.set_text('')
    fig.canvas.draw_idle()

def do_reset(event=None):
    for sl in sliders:
        sl.reset()
    for idx in range(len(sliders)):
        vt, unit, fmt = val_texts[idx]
        vt.set_text(f'{sliders[idx].val:{fmt}} {unit}')
    apply_params()

def _step(delta):
    """Avança/volta um quadro (com a animação pausada)."""
    state['frame'] = (state['frame'] + delta) % N
    _draw_frame(state['frame'])
    fig.canvas.draw_idle()

def on_key(event):
    k = event.key
    if k in (' ', 'p'):
        toggle_pause()
    elif k == 'r':
        do_reset()
    elif k == 'e':
        toggle_panel('min_e', toggle_e_btn, 'ENERGIA × TEMPO')
    elif k == 'f':
        toggle_panel('min_f', toggle_f_btn, 'ESPAÇO DE FASES')
    elif k in ('right', 'left'):
        if not state['paused']:
            toggle_pause()          # pausa antes de andar quadro a quadro
        _step(+1 if k == 'right' else -1)

btn_pp.on_clicked(toggle_pause)
btn_ap.on_clicked(apply_params)
btn_rst.on_clicked(do_reset)
fig.canvas.mpl_connect('key_press_event', on_key)

if __name__ == '__main__':
    ani = animation.FuncAnimation(
        fig, update,
        interval=14,
        blit=False,
        cache_frame_data=False,
    )
    plt.show()
