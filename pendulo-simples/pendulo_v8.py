#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pêndulo Simples — Simulação Interativa (v8)
============================================

Evolução do v7 (mesma física RK4, atrito viscoso, período exato e painéis
minimizáveis), agora num projeto mais completo e didático:

  • Comparação linear × não-linear: sobreposição "fantasma" do pêndulo de
    pequenas oscilações (solução linearizada θ''=-(g/L)θ-γθ') na simulação
    e no espaço de fases — deixa visível por que a aproximação de pequeno
    ângulo erra cada vez mais conforme θ₀ cresce.
  • Três períodos lado a lado: T₀ (linear), T (amplitude exata via AGM) e
    T medido diretamente da trajetória integrada — validação do integrador.
  • Força externa / ressonância: torque forçante A·cos(ω_d·t) opcional. Com
    ω_d ≈ ω₀ = √(g/L) a amplitude (e a energia) cresce — ressonância; o
    cabeçalho mostra ω₀ e a razão Ω/ω₀, acende um aviso "RESSONÂNCIA", e a
    tecla [b] abre a curva de ressonância (amplitude × Ω) comparando a
    previsão linear com a resposta não-linear real (com o pico "encurvado"
    e o salto/histerese do pêndulo de verdade).
  • Análise de caos (pêndulo forçado): seção de Poincaré [k] (revela o atrator
    estranho) e diagrama de bifurcação [j] (cascata de duplicação de período
    rumo ao caos, varrendo a força A).
  • Curva período × amplitude [a]: a reta linear T₀, a curva exata T(θ₀) via AGM
    (que dispara a 180°) e pontos medidos por RK4 — o "porquê" dos três períodos.
  • Arco de ângulo ao vivo na simulação.
  • Controle de velocidade da animação (0,5× / 1× / 2×); exportar PNG e CSV.
  • Atrito viscoso (γ), energia dissipada (ΔE), presets de gravidade e
    realce do preset ativo (como no v7).
  • Animação por blitting (blit=True): só os artistas que se movem são
    redesenhados a cada quadro (o fundo estático fica cacheado), o que dá
    um ganho de FPS de cerca de 15× frente ao redesenho da figura inteira.

Atalhos de teclado:
    [h] ajuda · [espaço] pausa · [r] reset · [e]/[f] minimiza energia/fases ·
    [c] compara linear · [v] velocidade · [b] curva de ressonância · [k] seção de
    Poincaré · [j] diagrama de bifurcação · [a] período × amplitude · [m] música
    de fundo · [s] salva PNG · [d] salva CSV · [←]/[→] passo

Requisitos:
    pip install numpy matplotlib PyQt5

Uso:
    python pendulo_v8.py
"""

import os
import sys
import math
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyBboxPatch, Wedge
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

# Liberamos as teclas que o matplotlib usa por padrão para que nossos atalhos
# ([f], [s], [k], [←], [→]...) não disparem fullscreen/salvar/escala-log/etc.
for _km in ('keymap.fullscreen', 'keymap.back', 'keymap.forward', 'keymap.save',
            'keymap.xscale', 'keymap.yscale'):
    plt.rcParams[_km] = []

# Face de "instrumento" (Bahnschrift = grotesca técnica DIN, a mesma família de
# painéis/sinalização de engenharia; vem com o Windows 10/11). Usada com
# restrição: só título, rótulos de seção e legendas de leitura. Os números
# continuam em Consolas (mono) e o corpo em Segoe UI. Cai para Segoe UI se a
# fonte não existir na máquina. (▼ ► ₀ não existem nela — esses ficam fora.)
DISPLAY = ['Bahnschrift', 'Segoe UI']

# ── Parâmetros ────────────────────────────────────────────────────────────────
# A = amplitude do torque forçante (rad/s²); wd = frequência forçante (rad/s).
# A = 0 → pêndulo livre de sempre.
DEFAULTS = dict(L=1.6, m=1.0, theta0_deg=55.0, g=9.81, gamma=0.0, A=0.0, wd=2.5)
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
    paused=False, frame=0, fpos=0.0,
    L=DEFAULTS['L'], m=DEFAULTS['m'],
    theta0_deg=DEFAULTS['theta0_deg'],
    theta0=np.radians(DEFAULTS['theta0_deg']),
    g=DEFAULTS['g'], gamma=DEFAULTS['gamma'],
    A=DEFAULTS['A'], wd=DEFAULTS['wd'],
    min_e=False, min_f=False,
    show_linear=False, speed=1.0,
)

SPEEDS = [0.5, 1.0, 2.0]   # multiplicadores de velocidade da animação

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

def _rk4(deriv, theta0, t):
    """Integra θ'' = deriv(t, θ, ω) (na forma de 1ª ordem) por RK4, com θ(0)=θ0,
    ω(0)=0. O tempo é passado a deriv por causa do termo forçante cos(ω_d·t).
    Retorna (theta, omega)."""
    Ns = len(t); th = np.zeros(Ns); om = np.zeros(Ns); th[0] = theta0
    for i in range(1, Ns):
        ti = t[i-1]
        k1 = deriv(ti,        th[i-1],            om[i-1])
        k2 = deriv(ti+DT/2,   th[i-1]+DT/2*k1[0], om[i-1]+DT/2*k1[1])
        k3 = deriv(ti+DT/2,   th[i-1]+DT/2*k2[0], om[i-1]+DT/2*k2[1])
        k4 = deriv(ti+DT,     th[i-1]+DT  *k3[0], om[i-1]+DT  *k3[1])
        th[i] = th[i-1] + DT/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
        om[i] = om[i-1] + DT/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
    return th, om

def measured_period(t, th):
    """Mede o período direto da trajetória: como θ(0)=θ₀ é um máximo, o
    primeiro máximo interior de θ acontece após uma oscilação completa.
    Retorna None se não houver oscilação (caso super-amortecido)."""
    for i in range(1, len(th) - 1):
        if th[i] > th[i-1] and th[i] >= th[i+1] and th[i] > 0.0:
            return float(t[i])
    return None

def driven_steady_amp(g, L, gam, A, wd, dt=0.02, t_total=42.0):
    """Amplitude estacionária (graus) do pêndulo NÃO-LINEAR forçado, partindo do
    repouso (θ=0): integra por RK4 e mede max|θ| na parte final (já sem o
    transitório). Satura em 180° (acima disso o pêndulo gira). Escalar + math
    para ser rápido o bastante para varrer dezenas de Ω num clique."""
    n = int(t_total / dt); gL = g / L; th = 0.0; om = 0.0; mx = 0.0
    t0 = 0.55 * t_total
    sin = math.sin; cos = math.cos
    for i in range(n):
        ti = i * dt
        F1 = A * cos(wd * ti); F2 = A * cos(wd * (ti + dt/2)); F4 = A * cos(wd * (ti + dt))
        k1t = om;               k1w = -gL*sin(th) - gam*om + F1
        k2t = om + dt/2*k1w;    k2w = -gL*sin(th + dt/2*k1t) - gam*(om + dt/2*k1w) + F2
        k3t = om + dt/2*k2w;    k3w = -gL*sin(th + dt/2*k2t) - gam*(om + dt/2*k2w) + F2
        k4t = om + dt*k3w;      k4w = -gL*sin(th + dt*k3t)   - gam*(om + dt*k3w)   + F4
        th += dt/6*(k1t + 2*k2t + 2*k3t + k4t)
        om += dt/6*(k1w + 2*k2w + 2*k3w + k4w)
        if ti >= t0:
            a = abs(th)
            if a > mx: mx = a
        if mx > 7.0:   # já passou de ~400°: claramente girando, corta cedo
            break
    return min(math.degrees(mx), 180.0)

def poincare_samples(g, L, gam, A, wd, n_periods, n_discard, theta0=0.2, om0=0.0):
    """Seção de Poincaré estroboscópica do pêndulo forçado: integra por RK4 e
    amostra (θ em (-π,π], ω) uma vez por período forçante T_d = 2π/ω_d,
    descartando os primeiros n_discard períodos (transitório). Para movimento
    periódico resultam poucos pontos distintos; no caos, uma nuvem fractal."""
    Td = 2.0 * math.pi / wd
    steps = max(40, int(round(Td / 0.02)))
    dt = Td / steps
    gL = g / L; th = theta0; om = om0
    sin = math.sin; cos = math.cos; twopi = 2.0 * math.pi
    th_s = []; om_s = []
    for p in range(n_periods):
        tb = p * Td
        for i in range(steps):
            ti = tb + i * dt
            F1 = A*cos(wd*ti); F2 = A*cos(wd*(ti+dt*0.5)); F4 = A*cos(wd*(ti+dt))
            k1t = om;             k1w = -gL*sin(th) - gam*om + F1
            k2t = om+dt*0.5*k1w;  k2w = -gL*sin(th+dt*0.5*k1t) - gam*(om+dt*0.5*k1w) + F2
            k3t = om+dt*0.5*k2w;  k3w = -gL*sin(th+dt*0.5*k2t) - gam*(om+dt*0.5*k2w) + F2
            k4t = om+dt*k3w;      k4w = -gL*sin(th+dt*k3t)     - gam*(om+dt*k3w)     + F4
            th += dt/6.0*(k1t + 2*k2t + 2*k3t + k4t)
            om += dt/6.0*(k1w + 2*k2w + 2*k3w + k4w)
        if p >= n_discard:
            th_s.append((th + math.pi) % twopi - math.pi)
            om_s.append(om)
    return th_s, om_s

def bifurcation_data(g, L, gam, wd, A_array, n_periods, n_discard, dt_target=0.02):
    """Dados do diagrama de bifurcação, VETORIZADO: integra o pêndulo forçado
    para todos os valores de A de uma vez (cada passo RK4 opera em arrays numpy),
    o que é muito mais rápido que um laço em Python por valor de A. Retorna
    (A_pts, omega_pts) com as amostras de ω na seção de Poincaré (regime
    permanente, após descartar n_discard períodos)."""
    Td = 2.0*np.pi/wd
    steps = max(20, int(round(Td/dt_target)))
    dt = Td/steps; gL = g/L
    A = np.asarray(A_array, dtype=float)
    th = np.full_like(A, 0.2); om = np.zeros_like(A)
    xs = []; ys = []
    for p in range(n_periods):
        tb = p*Td
        for i in range(steps):
            ti = tb + i*dt
            F1 = A*np.cos(wd*ti); F2 = A*np.cos(wd*(ti+dt*0.5)); F4 = A*np.cos(wd*(ti+dt))
            k1t = om;             k1w = -gL*np.sin(th) - gam*om + F1
            k2t = om+dt*0.5*k1w;  k2w = -gL*np.sin(th+dt*0.5*k1t) - gam*(om+dt*0.5*k1w) + F2
            k3t = om+dt*0.5*k2w;  k3w = -gL*np.sin(th+dt*0.5*k2t) - gam*(om+dt*0.5*k2w) + F2
            k4t = om+dt*k3w;      k4w = -gL*np.sin(th+dt*k3t)     - gam*(om+dt*k3w)     + F4
            th = th + dt/6.0*(k1t + 2*k2t + 2*k3t + k4t)
            om = om + dt/6.0*(k1w + 2*k2w + 2*k3w + k4w)
        if p >= n_discard:
            xs.append(A.copy()); ys.append(om.copy())
    return np.concatenate(xs), np.concatenate(ys)

def period_amplitude_data(g, L, n=260, deg_max=178.0):
    """Período do pêndulo livre em função da amplitude θ₀.

    Devolve (graus, T0, T_exato): a reta linear T₀ = 2π√(L/g) (constante) e a
    curva exata T(θ₀) = 2π√(L/g)/AGM(1, cos(θ₀/2)) — vetorizada via _agm sobre
    o array todo. T diverge quando θ₀ → 180° (o pêndulo para no alto)."""
    deg = np.linspace(2.0, deg_max, n)
    base = 2 * np.pi * np.sqrt(L / g)
    T_exact = base / _agm(1.0, np.cos(np.radians(deg) / 2.0))
    return deg, base, T_exact

def measured_periods_at(g, L, degs, t_total=26.0):
    """Período MEDIDO (1º máximo interior de θ) de pêndulos livres integrados por
    RK4, em cada amplitude de `degs` — pontos pra validar a curva exata. Pula as
    amplitudes que não completam uma oscilação dentro da janela."""
    t = np.arange(0, t_total, DT)
    out = []
    for d in degs:
        th, _ = _rk4(lambda ti, a, w: (w, -(g / L) * np.sin(a)), np.radians(d), t)
        Tm = measured_period(t, th)
        if Tm is not None:
            out.append((d, Tm))
    return out

# ── Integração numérica (RK4) ─────────────────────────────────────────────────
def compute(g: float, L: float, m: float, theta0: float, gamma: float,
            A: float, wd: float):
    """Integra o pêndulo (atrito + força externa opcionais) via RK4, junto com o
    pêndulo linearizado (pequenas oscilações) para comparação.

    Não-linear:  θ'' = -(g/L)·sin(θ) - γ·θ' + A·cos(ω_d·t)
    Linearizado: θ'' = -(g/L)·θ      - γ·θ' + A·cos(ω_d·t)   (só para θ pequeno)

    Com A = 0 é o pêndulo livre; com A > 0 vira o pêndulo forçado (ressonância
    quando ω_d ≈ ω₀ = √(g/L), e caos para forçamento forte em grande amplitude).

    Returns:
        (t, Ns, theta, omega, xb, yb, h, v, Ek, Ep, Et,
         T_lin, T_amp, T_med, theta_lin, omega_lin)
    """
    t  = np.arange(0, T_SIM, DT)
    Ns = len(t)
    th, om = _rk4(lambda ti, a, w: (w, -(g/L)*np.sin(a) - gamma*w + A*np.cos(wd*ti)),
                  theta0, t)
    th_l, om_l = _rk4(lambda ti, a, w: (w, -(g/L)*a - gamma*w + A*np.cos(wd*ti)),
                      theta0, t)
    xb = L*np.sin(th); yb = -L*np.cos(th)
    hp = L*(1-np.cos(th)); vp = L*np.abs(om)
    Ek = 0.5*m*vp**2; Ep = m*g*hp; Et = Ek+Ep
    T_lin = 2*np.pi*np.sqrt(L/g)
    T_amp = exact_period(g, L, theta0)
    T_med = measured_period(t, th)
    return t, Ns, th, om, xb, yb, hp, vp, Ek, Ep, Et, T_lin, T_amp, T_med, th_l, om_l

t_arr, N, theta, omega, xb, yb, h_p, v_p, Ek, Ep, Et, Tp_lin, Tp_amp, Tp_med, theta_lin, omega_lin = \
    compute(state['g'], state['L'], state['m'], state['theta0'], state['gamma'],
            state['A'], state['wd'])
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
    fig.canvas.manager.set_window_title('Pêndulo Simples — v8')
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
BTN_H    = 0.035
BTN_G    = 0.022   # gap maior: dá respiro pro status_txt entre os botões
                   # principais e a linha de presets de gravidade
y_btn    = CTRL_BOT                              # linha de botões principais

PLANET_ROW_H = 0.027
PLANET_GAP   = 0.011
y_planets    = y_btn + BTN_H + BTN_G             # linha de presets de gravidade

# 7 sliders (L, m, θ₀, g, γ, A, ω_d) — linhas mais compactas que nas versões
# anteriores para caber a força externa sem espremer os gráficos.
LINE_H = 0.024
LINE_G = 0.010
N_SLIDERS = 7
y_sl   = [y_planets + PLANET_ROW_H + PLANET_GAP + i*(LINE_H + LINE_G)
          for i in range(N_SLIDERS)]

CTRL_TOP = y_sl[-1] + LINE_H + 0.014   # top do painel de controles (= base do ax_f)

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
AX_F_H_DEFAULT = 0.165   # altura do espaço de fases quando os dois estão abertos

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
    # draw() síncrono (não draw_idle): refaz o layout E o fundo cacheado do blit
    # imediatamente, evitando o quadro "fantasma" com o cache antigo ao minimizar.
    fig.canvas.draw()

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

MINT = '#46E8B0'   # cor da frequência forçante (ω_d)

params = [
    # (label, unidade, vmin, vmax, valinit, cor, formato)
    ('L',  'm',     0.3,  3.0,  state['L'],          TEAL,   '.2f'),
    ('m',  'kg',    0.1,  5.0,  state['m'],          AMBER,  '.1f'),
    ('θ₀', '°',     5.0,  85.0, state['theta0_deg'], BLUE,   '.0f'),
    ('g',  'm/s²',  1.0,  25.0, state['g'],          VIOLET, '.2f'),
    ('γ',  '/s',    0.0,  2.0,  state['gamma'],      RED,    '.2f'),
    ('A',  'rad/s²',0.0,  6.0,  state['A'],          GOLD,   '.2f'),   # força externa
    ('Ω',  'rad/s', 0.1,  6.0,  state['wd'],         MINT,   '.2f'),   # freq. forçante
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

sl_L, sl_m, sl_th, sl_g, sl_gam, sl_A, sl_wd = sliders

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
fig.text(TITLE_X, 0.984, 'PÊNDULO SIMPLES',
         ha='center', va='top', fontsize=18, fontweight='bold', color=AMBER,
         fontfamily=DISPLAY,
         path_effects=[pe.withStroke(linewidth=8, foreground=BG)])
fig.text(TITLE_X, 0.960,
         'simulação interativa  ·  runge-kutta 4ª ordem  ·  atrito viscoso opcional',
         ha='center', va='top', fontsize=7.5, color=MUTED2, fontfamily=DISPLAY)

# ── Faixa de leitura dos períodos (mostrador de instrumento) ─────────────────
# O produto intelectual da simulação são os três períodos (linear, exato e
# medido) + a frequência natural — antes ficavam numa única linha cinza minúscula.
# Aqui viram um mostrador segmentado: RÓTULO (Bahnschrift, caixa-alta) em cima,
# VALOR (Consolas, com símbolo) embaixo, com o período EXATO em destaque (herói)
# e um "chip" de desvio em relação ao linear. Os valores são preenchidos em
# _update_sub(); só o conteúdo muda, não o cálculo.
_READ_CX = [0.32, 0.44, 0.56, 0.68]
read_artists = {}   # key -> (rótulo, valor, cor_ativa_rótulo, cor_ativa_valor)
def _readout(key, cx, label, lab_color, val_color, hero=False):
    la = fig.text(cx, 0.938, label, ha='center', va='top', fontfamily=DISPLAY,
                  fontsize=7.4 if hero else 6.6, fontweight='bold', color=lab_color)
    va = fig.text(cx, 0.921, '', ha='center', va='top', fontfamily='monospace',
                  fontsize=13 if hero else 9.5, fontweight='bold', color=val_color)
    read_artists[key] = (la, va, lab_color, val_color)
    return va
read_lin = _readout('lin', _READ_CX[0], 'LINEAR',  MUTED2, '#8E99CC')
read_ex  = _readout('ex',  _READ_CX[1], 'EXATO',   GOLD,   GOLD, hero=True)
read_med = _readout('med', _READ_CX[2], 'MEDIDO',  MUTED2, TEAL)
read_w0  = _readout('w0',  _READ_CX[3], 'NATURAL', MUTED2, VIOLET)
# chip de desvio do período exato vs. linear, sob o valor-herói (ou aviso de que
# os períodos só valem p/ pêndulo livre, quando há atrito/força).
read_dpct = fig.text(_READ_CX[1], 0.898, '', ha='center', va='top',
                     fontfamily='monospace', fontsize=7.2, fontweight='bold', color=MINT)
# divisórias finas entre os segmentos (linguagem de painel de instrumento)
for _dx in (0.38, 0.50, 0.62):
    fig.add_artist(plt.Line2D([_dx, _dx], [0.900, 0.940], transform=fig.transFigure,
                   color=BORDER2, lw=0.7, alpha=0.7))

# Leitura da força/ressonância (canto sup. direito): razão Ω/ω₀ e aviso quando
# Ω ≈ ω₀ com força ligada.
reson_txt = fig.text(1.0 - MARGIN_R, 0.921, '', ha='right', va='top',
                     fontsize=9.5, fontweight='bold', color=GOLD)

# ── Barra de ferramentas (topo-esquerda do cabeçalho; posição fixa) ──────────
def _make_tool_btn(x, w, label, color=MUTED2):
    bax = fig.add_axes([x, 0.858, w, 0.026], facecolor=ACCENT)
    for sp in bax.spines.values():
        sp.set_edgecolor(BORDER2); sp.set_linewidth(0.7)
    b = Button(bax, label, color=ACCENT, hovercolor='#1E2550')
    b.label.set_color(color); b.label.set_fontsize(8)
    return b

btn_lin   = _make_tool_btn(0.018, 0.070, 'Linear: off')
btn_speed = _make_tool_btn(0.093, 0.048, 'Vel: 1×')
btn_save  = _make_tool_btn(0.146, 0.074, 'Salvar PNG', color=MUTED)
btn_csv   = _make_tool_btn(0.225, 0.074, 'Salvar CSV', color=MUTED)
btn_reson = _make_tool_btn(0.304, 0.090, 'Ressonância', color=GOLD)
# Hierarquia: "Ressonância" é a ação primária de análise — recebe um leve
# preenchimento e contorno dourados pra sair da fileira de chips iguais; as
# utilidades de salvar ficam recuadas (cinza mais apagado, acima).
btn_reson.color = '#1A1407'
btn_reson.ax.set_facecolor('#1A1407')
for _sp in btn_reson.ax.spines.values():
    _sp.set_edgecolor(GOLD); _sp.set_linewidth(1.0)

status_txt = fig.text(CTRL_L, CTRL_BOT + BTN_H + 0.005, '',
                      va='bottom', fontsize=7.5, color=MUTED2)
register_ctrl(status_txt, 'text', (CTRL_L, y_btn), anchor=y_btn, offset=BTN_H + 0.005)

# Dica de atalhos de teclado (rodapé da coluna esquerda, bem discreto)
fig.text(MARGIN_L, 0.013,
         'atalhos:  [h] ajuda · [espaço] pausar · [r] reset · [e/f] painéis · [c] comparar · '
         '[v] velocidade · [b] ressonância · [k] Poincaré · [j] bifurcação · [a] período×amplitude · '
         '[m] música · [s] PNG · [d] CSV · [←/→] passo',
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
         'PARÂMETROS', color=MUTED2, fontsize=8, fontweight='bold',
         va='top', fontfamily=DISPLAY)
register_ctrl(params_lbl, 'text', (CTRL_L, CTRL_TOP), anchor=CTRL_TOP, offset=-0.002)

relayout()

def _update_sub():
    # Preenche o mostrador de períodos (grandezas derivadas — os parâmetros já
    # aparecem ao lado dos sliders). Só formata; o cálculo está em compute().
    dpct = (Tp_amp / Tp_lin - 1.0) * 100.0
    w0   = np.sqrt(state['g'] / state['L'])
    # T₀/T exato/T medido são definidos para o pêndulo LIVRE; com atrito (γ>0) ou
    # força (A>0) viram só referência — esmaecemos e zeramos o "medido" (não há
    # período limpo). ω₀ = √(g/L) continua valendo sempre.
    free = (state['gamma'] == 0.0 and state['A'] == 0.0)
    for key in ('lin', 'ex', 'med'):
        la, va, lc, vc = read_artists[key]
        la.set_color(lc if free else MUTED); va.set_color(vc if free else MUTED)
    read_lin.set_text('T₀ %.3f s' % Tp_lin)
    read_ex.set_text('T %.3f s' % Tp_amp)
    read_med.set_text(('T %.3f s' % Tp_med) if (free and Tp_med is not None) else 'T —')
    read_w0.set_text('ω₀ %.2f' % w0)
    if free:
        arrow = '▲' if dpct >= 0 else '▼'
        read_dpct.set_text('%s %+.1f%%  vs. linear' % (arrow, dpct))
        read_dpct.set_color(MINT if dpct >= 0 else RED)
    else:
        read_dpct.set_text('referência: pêndulo livre')
        read_dpct.set_color(MUTED2)
    # Leitura da força (canto sup. direito): razão Ω/ω₀ e aviso de ressonância
    # quando Ω está a menos de 8% de ω₀ com a força ligada.
    ratio = state['wd'] / w0
    if state['A'] > 0:
        near = abs(ratio - 1.0) < 0.08
        reson_txt.set_text(('Ω/ω₀ = %.2f' % ratio) + ('    ●  RESSONÂNCIA' if near else ''))
        reson_txt.set_color(GOLD if near else MUTED2)
    else:
        reson_txt.set_text('')

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
ANIM = []   # artistas redesenhados a cada quadro (blitting)

def rebuild():
    global t_arr, N, theta, omega, xb, yb, h_p, v_p, Ek, Ep, Et
    global Tp_lin, Tp_amp, Tp_med, theta_lin, omega_lin, Et0, ANIM

    t_arr, N, theta, omega, xb, yb, h_p, v_p, Ek, Ep, Et, \
        Tp_lin, Tp_amp, Tp_med, theta_lin, omega_lin = \
        compute(state['g'], state['L'], state['m'], state['theta0'], state['gamma'],
                state['A'], state['wd'])
    Et0 = Et[0]
    state['frame'] = 0
    state['fpos']  = 0.0
    _update_sub()

    L     = state['L']
    bob_r = max(0.08, L * 0.074)
    margin= max(0.32, L * 0.21)
    # coordenadas do bob "fantasma" (pêndulo linearizado)
    xb_lin = L*np.sin(theta_lin); yb_lin = -L*np.cos(theta_lin)

    # ── Simulação ─────────────────────────────────────────────────────────────
    ax_d.clear()
    ax_d.set_facecolor(BG)
    xlim = L + margin
    ax_d.set_xlim(-xlim, xlim)
    ax_d.set_ylim(-L - margin, 0.80)
    # adjustable='box' + anchor='N': respeita os limites de dados (sem expandir
    # céu vazio acima do suporte) e ancora a cena no TOPO do eixo, jogando a folga
    # pra baixo — elimina o vão morto entre a barra de ferramentas e o suporte.
    ax_d.set_aspect('equal', adjustable='box', anchor='N')
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

    # Transferidor graduado no pivô (escala de ângulo, 0° na vertical): torna a
    # leitura de θ tangível e reforça a estética de instrumento. Tudo estático
    # (fundo cacheado pelo blit) — não entra em ANIM, custo zero por quadro.
    prot_r = max(0.46, L*0.50)
    pa = np.radians(np.linspace(-90, 90, 160))
    ax_d.plot(prot_r*np.sin(pa), -prot_r*np.cos(pa),
              color=BORDER2, lw=0.8, alpha=0.55, zorder=2)
    for ang in range(-90, 91, 15):
        a = np.radians(ang); major = (ang % 30 == 0)
        r2 = prot_r + (0.05*L if major else 0.025*L)
        ax_d.plot([prot_r*np.sin(a), r2*np.sin(a)],
                  [-prot_r*np.cos(a), -r2*np.cos(a)],
                  color=MUTED2 if major else BORDER2,
                  lw=0.9 if major else 0.6,
                  alpha=0.6 if major else 0.4, zorder=2)
        if major and ang != 0:        # 0° fica sob o fio: dispensa rótulo
            rl = prot_r + 0.11*L
            ax_d.text(rl*np.sin(a), -rl*np.cos(a), '%d°' % abs(ang),
                      color=MUTED2, fontsize=6.0, alpha=0.6,
                      ha='center', va='center', zorder=2)

    # Rastro
    segs = []
    for k in range(N_SEGS):
        frac = k / N_SEGS
        seg, = ax_d.plot([], [], color=AMBER,
                         lw=0.3 + 3.5*frac**2.2,
                         alpha=0.02 + 0.45*frac**1.6,
                         solid_capstyle='round', zorder=5)
        segs.append(seg)

    # Pêndulo "fantasma" (aproximação linear) — visível só no modo comparar [c]
    gl_vis = state['show_linear']
    fio_lin, = ax_d.plot([], [], color=BLUE, lw=1.3, ls=(0, (5, 4)),
                         alpha=0.55, zorder=6, visible=gl_vis)
    bob_lin  = Circle((xb_lin[0], yb_lin[0]), bob_r*0.92, facecolor='none',
                      edgecolor=BLUE, lw=1.6, alpha=0.85, zorder=6, visible=gl_vis)
    ax_d.add_patch(bob_lin)

    # Arco do ângulo atual no pivô (+ vertical de referência tracejada)
    arc_r = max(0.22, L*0.30)
    ax_d.plot([0, 0], [0, -arc_r*1.18], color=MUTED2, lw=0.8, ls=(0, (3, 3)),
              alpha=0.35, zorder=3)
    arc_ln,  = ax_d.plot([], [], color=GOLD, lw=1.4, alpha=0.6, zorder=6)
    arc_lbl  = ax_d.text(0, 0, '', color=GOLD, fontsize=8, alpha=0.9,
                         ha='center', va='center', zorder=13)

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
    # Caixa retangular de canto vivo (mesma linguagem dos gráficos/sliders/botões
    # — era a única caixa arredondada da tela; agora todos os contêineres falam
    # a mesma língua de painel de instrumento).
    e_panel = FancyBboxPatch((PANEL_X, PANEL_Y), PANEL_W, PANEL_H,
        transform=ax_d.transAxes, boxstyle='square,pad=0.012',
        facecolor='#0A0C16', edgecolor=BORDER2, linewidth=0.8, alpha=0.92,
        zorder=14)
    ax_d.add_patch(e_panel)

    DOT_X = PANEL_X + 0.018
    TXT_X = PANEL_X + 0.034
    VAL_X = PANEL_X + PANEL_W - 0.020
    ax_d.text(DOT_X, PANEL_Y + PANEL_H*0.88, 'ENERGIA', transform=ax_d.transAxes,
        fontsize=7.0, color=MUTED2, fontweight='bold', ha='left', va='center',
        zorder=15, fontfamily=DISPLAY)
    ax_d.text(VAL_X, PANEL_Y + PANEL_H*0.88, 'joules', transform=ax_d.transAxes,
        fontsize=6.8, color=MUTED, ha='right', va='center', zorder=15,
        fontfamily=DISPLAY)
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
    # o topo acompanha o máximo de energia (com força externa a energia cresce)
    ax_e.set_ylim(-0.03, max(Et0, Et.max()) * 1.12 + 1e-6)
    ax_e.axhline(Et0, color=MUTED, lw=0.6, ls=':', alpha=0.45, zorder=2)

    ln_ek, = ax_e.plot([], [], color=AMBER, lw=1.8, alpha=0.9)
    ln_ep, = ax_e.plot([], [], color=TEAL,  lw=1.8, alpha=0.9)
    ln_et, = ax_e.plot([], [], color=GOLD,  lw=1.0, ls='--', alpha=0.5)

    # ── Espaço de Fases (minimalista: órbita de fundo + ponto + equilíbrio) ────
    ax_f.clear()
    _style(ax_f, 'ângulo (°)', 'vel. angular (rad/s)')
    th_d  = np.degrees(theta)
    thl_d = np.degrees(theta_lin)
    # limites englobam também a órbita linear (fantasma), pra nunca cortá-la
    allx = np.concatenate([th_d, thl_d]); ally = np.concatenate([omega, omega_lin])
    # com força forte o pêndulo pode girar (θ cresce sem limite) — limita a faixa
    # de ângulo exibida pra não espremer o gráfico num eixo gigante.
    xlo = max(allx.min(), -270.0); xhi = min(allx.max(), 270.0)
    pad_x = max(2, (xhi - xlo) * 0.14)
    pad_y = max(0.1, (ally.max()-ally.min()) * 0.14)
    ax_f.set_xlim(xlo-pad_x, xhi+pad_x)
    ax_f.set_ylim(ally.min()-pad_y, ally.max()+pad_y)
    ax_f.axhline(0, color=BORDER2, lw=0.8, alpha=0.6, zorder=2)
    ax_f.axvline(0, color=BORDER2, lw=0.8, alpha=0.6, zorder=2)
    ax_f.plot(th_d, omega, color=MUTED, lw=1.2, alpha=0.55, zorder=3)
    # Ponto de equilíbrio (atrator quando há atrito)
    ax_f.plot(0, 0, marker='o', mfc='none', mec=MUTED2, ms=6, mew=0.9, zorder=4)

    # Órbita "fantasma" da aproximação linear (visível só no modo comparar)
    gl_vis = state['show_linear']
    ph_orbit_lin, = ax_f.plot(thl_d, omega_lin, color=BLUE, lw=1.0, ls=(0, (5, 4)),
                              alpha=0.5, zorder=5, visible=gl_vis)
    ph_dot_lin    = ax_f.plot([], [], 'o', mfc='none', mec=BLUE, mew=1.4, ms=7,
                              zorder=7, visible=gl_vis)[0]

    ph_trail, = ax_f.plot([], [], color=AMBER, lw=1.8, alpha=0.85,
                          solid_capstyle='round', zorder=8)
    ph_dot    = ax_f.plot([], [], 'o', color=AMBER, ms=6, zorder=9,
                          markeredgecolor=WHITE, markeredgewidth=0.5)[0]

    aobj.update(dict(
        segs=segs, fio=fio, fio2=fio2,
        bob=bob_p, bob_gl=bob_gl, bob_gl2=bob_gl2, bob_hi=bob_hi,
        varr=varr,
        fio_lin=fio_lin, bob_lin=bob_lin, arc_ln=arc_ln, arc_lbl=arc_lbl, arc_r=arc_r,
        ph_orbit_lin=ph_orbit_lin, ph_dot_lin=ph_dot_lin,
        t_lbl=t_lbl, ek_val=ek_val, ep_val=ep_val, et_val=et_val, ed_val=ed_val,
        ln_ek=ln_ek, ln_ep=ln_ep, ln_et=ln_et,
        ph_trail=ph_trail, ph_dot=ph_dot,
    ))

    # ── Lista de artistas animados (redesenhados por blit a cada quadro),
    # em ordem de zorder dentro de cada eixo. Só eles são redesenhados; todo o
    # resto (grade, suporte, painéis, sliders, botões) fica no fundo cacheado.
    ANIM = list(segs) + [
        fio_lin, bob_lin, arc_ln,
        fio, fio2,
        bob_gl2, bob_gl, bob_p, bob_hi,
        varr, arc_lbl,
        ek_val, ep_val, et_val, ed_val, t_lbl,
        ln_ek, ln_ep, ln_et,
        ph_dot_lin, ph_trail, ph_dot,
    ]
    for a in ANIM:
        a.set_animated(True)

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

    # Pêndulo "fantasma" (aproximação linear) e ponto fantasma de fase
    xl = L*np.sin(theta_lin[i]); yl = -L*np.cos(theta_lin[i])
    aobj['fio_lin'].set_data([0, xl], [0, yl])
    aobj['bob_lin'].center = (xl, yl)
    aobj['ph_dot_lin'].set_data([np.degrees(theta_lin[i])], [omega_lin[i]])

    # Arco do ângulo atual no pivô
    ar  = aobj['arc_r']; phi = np.arctan2(yi, xi)
    aa  = np.linspace(-np.pi/2, phi, 24)
    aobj['arc_ln'].set_data(ar*np.cos(aa), ar*np.sin(aa))
    amid = 0.5*(-np.pi/2 + phi)
    aobj['arc_lbl'].set_position((ar*1.34*np.cos(amid), ar*1.34*np.sin(amid)))
    aobj['arc_lbl'].set_text(f'{np.degrees(theta[i]):+.0f}°')

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

def _anim_artists():
    # Só blitamos artistas em eixos visíveis: se um gráfico está minimizado, seus
    # artistas (linhas de energia / fase) são pulados — senão a FuncAnimation
    # tentaria blitar o eixo colapsado e deixaria um "glitch" na tela.
    return [a for a in ANIM if a.axes.get_visible()]

def _anim_init():
    _draw_frame(state['frame'])
    return _anim_artists()

def update(_frame):
    # Com blit, só os artistas de ANIM são redesenhados (não a figura inteira).
    # Mesmo pausado retornamos os artistas para que continuem visíveis no fundo.
    if not state['paused']:
        state['fpos']  = (state['fpos'] + state['speed']) % N   # avanço fracionário
        state['frame'] = int(state['fpos'])
        _draw_frame(state['frame'])
    return _anim_artists()

def apply_params(event=None):
    state['L']          = sl_L.val
    state['m']          = sl_m.val
    state['theta0_deg'] = sl_th.val
    state['theta0']     = np.radians(sl_th.val)
    state['g']          = sl_g.val
    state['gamma']      = sl_gam.val
    state['A']          = sl_A.val
    state['wd']         = sl_wd.val
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
    """Avança/volta um quadro (com a animação pausada). Com blit, o próximo
    tick da animação (que continua ocorrendo) reexibe os artistas atualizados."""
    state['frame'] = (state['frame'] + delta) % N
    state['fpos']  = float(state['frame'])
    _draw_frame(state['frame'])

def toggle_linear(event=None):
    """Mostra/esconde a sobreposição do pêndulo linearizado (comparação)."""
    state['show_linear'] = not state['show_linear']
    vis = state['show_linear']
    for key in ('fio_lin', 'bob_lin', 'ph_orbit_lin', 'ph_dot_lin'):
        aobj[key].set_visible(vis)
    btn_lin.label.set_text('Linear: on' if vis else 'Linear: off')
    btn_lin.label.set_color(BLUE if vis else MUTED2)
    fig.canvas.draw_idle()

def cycle_speed(event=None):
    """Alterna 0,5× → 1× → 2× na velocidade da animação."""
    state['speed'] = SPEEDS[(SPEEDS.index(state['speed']) + 1) % len(SPEEDS)]
    btn_speed.label.set_text(f'Vel: {state["speed"]:g}×')
    fig.canvas.draw_idle()

def _output_dir():
    """Pasta para salvar PNG/CSV: ao lado do .exe quando empacotado (PyInstaller),
    senão ao lado do script. (Empacotado, __file__ aponta para um temp que some.)"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def _out_path(ext):
    fname = datetime.datetime.now().strftime(f'pendulo_v8_%Y%m%d_%H%M%S.{ext}')
    return fname, os.path.join(_output_dir(), fname)

def save_png(event=None):
    """Salva um PNG da janela na pasta do script, com data/hora no nome."""
    fname, path = _out_path('png')
    # com blit os artistas são animated=True (excluídos do draw normal); para o
    # savefig sair completo, desligamos o flag temporariamente.
    for a in ANIM: a.set_animated(False)
    fig.savefig(path, dpi=160, facecolor=BG)
    for a in ANIM: a.set_animated(True)
    status_txt.set_text('salvo: ' + fname)
    fig.canvas.draw_idle()

def save_csv(event=None):
    """Exporta a trajetória integrada (t, θ, ω, energias) para CSV."""
    fname, path = _out_path('csv')
    data = np.column_stack([t_arr, np.degrees(theta), omega, Ek, Ep, Et])
    np.savetxt(path, data, delimiter=',', comments='', fmt='%.6g',
               header='t_s,theta_deg,omega_rad_s,Ek_J,Ep_J,Et_J')
    status_txt.set_text('salvo: ' + fname)
    fig.canvas.draw_idle()

def show_resonance(event=None):
    """Abre uma janela com a curva de ressonância (amplitude estacionária × Ω):
      - previsão LINEAR (forma fechada A / √((ω₀²-Ω²)² + (γΩ)²)), sempre;
      - resposta NÃO-LINEAR real (varrendo Ω e integrando o pêndulo forçado),
        quando há atrito e força — mostra o pico "encurvado" do pêndulo real.
    Marca a frequência natural ω₀ e a frequência forçante Ω atual."""
    g, L, gam, A = state['g'], state['L'], state['gamma'], state['A']
    w0 = np.sqrt(g / L)
    A_use = A if A > 0 else 1.0          # se A=0, mostra só a forma da curva
    W = np.linspace(0.05, max(6.0, 1.7*w0), 600)
    denom = np.sqrt((w0**2 - W**2)**2 + (gam*W)**2)
    amp_lin = np.clip(np.degrees(A_use / np.maximum(denom, 1e-9)), 0, 360)

    # Curva não-linear real (só faz sentido com atrito + força): varre Ω e mede
    # a amplitude estacionária integrando o pêndulo forçado em cada Ω.
    nl_W = nl_amp = None
    if gam > 0 and A > 0:
        status_txt.set_text('calculando curva de ressonância…'); fig.canvas.draw()
        nl_W = np.linspace(0.35*w0, 1.7*w0, 34)
        nl_amp = np.array([driven_steady_amp(g, L, gam, A, w) for w in nl_W])
        status_txt.set_text('')

    rfig = plt.figure(figsize=(7.4, 4.7), facecolor=BG)
    try:
        rfig.canvas.manager.set_window_title('Curva de ressonância — Pêndulo v8')
    except Exception:
        pass
    ax = rfig.add_axes([0.12, 0.16, 0.84, 0.72], facecolor=PANEL)
    ax.plot(W, amp_lin, color=GOLD, lw=2.0, zorder=5,
            label='linear (pequenas oscilações)')
    ax.fill_between(W, amp_lin, color=GOLD, alpha=0.10, zorder=3)
    if nl_amp is not None:
        ax.plot(nl_W, nl_amp, color=MINT, lw=2.0, marker='o', ms=3.5, zorder=6,
                label='não-linear (pêndulo real)')
    ax.axvline(w0, color=TEAL, ls='--', lw=1.3, alpha=0.85,
               label='ω₀ = %.2f rad/s (natural)' % w0)
    ax.axvline(state['wd'], color=WHITE, ls=':', lw=1.4, alpha=0.8,
               label='Ω = %.2f rad/s (atual)' % state['wd'])
    top = max(amp_lin.max(), (nl_amp.max() if nl_amp is not None else 0))
    ax.set_xlim(W[0], W[-1]); ax.set_ylim(0, top*1.12 + 1e-6)
    ax.set_xlabel('frequência forçante Ω (rad/s)', color=MUTED2, fontsize=9)
    ax.set_ylabel('amplitude estacionária (°)', color=MUTED2, fontsize=9)
    ax.set_title('Curva de ressonância', color=WHITE, fontsize=11, pad=10)
    ax.grid(True, color='#241B3E', alpha=0.6)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER2)
    ax.tick_params(colors=MUTED2, labelsize=8)
    ax.legend(facecolor=PANEL, edgecolor=BORDER2, fontsize=8.5, labelcolor=WHITE, loc='upper right')
    sub = 'A = %.2f rad/s²   ·   γ = %.2f /s' % (A_use, gam)
    if A == 0:
        sub += '   (A = 0: curva linear mostra só a forma — ajuste A)'
    elif gam == 0:
        sub += '   (sem atrito: curva não-linear omitida — não há regime estacionário)'
    else:
        sub += '   ·   não-linear satura em 180° quando o pêndulo passa a girar'
    rfig.text(0.12, 0.035, sub, color=MUTED, fontsize=8, ha='left')
    try:
        rfig.show()
    except Exception:
        pass
    status_txt.set_text('')          # limpa o "calculando…" da janela principal
    fig.canvas.draw_idle()

def _style_popup(ax, xl, yl, title):
    ax.set_xlabel(xl, color=MUTED2, fontsize=9)
    ax.set_ylabel(yl, color=MUTED2, fontsize=9)
    ax.set_title(title, color=WHITE, fontsize=10.5, pad=10)
    ax.grid(True, color='#241B3E', alpha=0.6)
    ax.tick_params(colors=MUTED2, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER2)

def show_poincare(event=None):
    """Janela com a seção de Poincaré (θ, ω amostrados 1×/período forçante) dos
    parâmetros atuais — revela órbitas periódicas (poucos pontos) ou o atrator
    estranho do caos (nuvem). Requer força ligada (A > 0)."""
    g, L, gam, A, wd = state['g'], state['L'], state['gamma'], state['A'], state['wd']
    if A <= 0 or wd <= 0:
        status_txt.set_text('Poincaré/bifurcação: ligue a força (A > 0)')
        fig.canvas.draw_idle(); return
    status_txt.set_text('calculando seção de Poincaré…'); fig.canvas.draw()
    th_s, om_s = poincare_samples(g, L, gam, A, wd, 520, 160, theta0=state['theta0'])
    rfig = plt.figure(figsize=(6.8, 5.6), facecolor=BG)
    try:
        rfig.canvas.manager.set_window_title('Seção de Poincaré — Pêndulo v8')
    except Exception:
        pass
    ax = rfig.add_axes([0.13, 0.12, 0.83, 0.79], facecolor=PANEL)
    ax.scatter(np.degrees(th_s), om_s, s=6, color=MINT, alpha=0.55, edgecolors='none')
    _style_popup(ax, 'ângulo θ (°)', 'velocidade angular ω (rad/s)',
                 'Seção de Poincaré  ·  1 amostra por período forçante')
    rfig.text(0.13, 0.025,
              'g=%.2f  L=%.2f  γ=%.2f  A=%.2f  Ω=%.2f      poucos pontos = periódico · nuvem = caótico'
              % (g, L, gam, A, wd), color=MUTED, fontsize=7.5, ha='left')
    try:
        rfig.show()
    except Exception:
        pass
    status_txt.set_text(''); fig.canvas.draw_idle()

def show_bifurcation(event=None):
    """Janela com o diagrama de bifurcação: varre a amplitude da força A e, para
    cada valor, plota as amostras de ω na seção de Poincaré (regime permanente).
    Revela a cascata de duplicação de período rumo ao caos. (Leva alguns segundos.)"""
    g, L, gam, wd = state['g'], state['L'], state['gamma'], state['wd']
    if wd <= 0:
        return
    status_txt.set_text('calculando diagrama de bifurcação…')
    fig.canvas.draw()
    A_max = max(12.0, state['A'] * 1.5)
    As = np.linspace(0.0, A_max, 260)
    xs, ys = bifurcation_data(g, L, gam, wd, As, n_periods=80, n_discard=48)
    rfig = plt.figure(figsize=(7.8, 5.2), facecolor=BG)
    try:
        rfig.canvas.manager.set_window_title('Diagrama de bifurcação — Pêndulo v8')
    except Exception:
        pass
    ax = rfig.add_axes([0.10, 0.13, 0.87, 0.78], facecolor=PANEL)
    ax.scatter(xs, ys, s=1.3, color=GOLD, alpha=0.5, edgecolors='none')
    ax.axvline(state['A'], color=MINT, ls=':', lw=1.5, alpha=0.9,
               label='A atual = %.2f' % state['A'])
    _style_popup(ax, 'amplitude da força A (rad/s²)', 'ω na seção de Poincaré (rad/s)',
                 'Diagrama de bifurcação  ·  varrendo a força A')
    ax.set_xlim(0, A_max)
    ax.legend(facecolor=PANEL, edgecolor=BORDER2, fontsize=8.5, labelcolor=WHITE, loc='upper left')
    rfig.text(0.10, 0.025, 'g=%.2f  L=%.2f  γ=%.2f  Ω=%.2f      ramos que se duplicam → rota para o caos'
              % (g, L, gam, wd), color=MUTED, fontsize=7.5, ha='left')
    try:
        rfig.show()
    except Exception:
        pass
    status_txt.set_text(''); fig.canvas.draw_idle()

def show_period_amplitude(event=None):
    """[a] Janela com a curva período × amplitude: por que os três períodos do
    cabeçalho divergem. Mostra a reta linear T₀ (válida só pra pequenas
    oscilações), a curva exata T(θ₀) via AGM (que cresce e dispara a 180°) e
    alguns pontos medidos por RK4 confirmando o exato. A amplitude atual fica
    marcada com o desvio % em relação ao linear."""
    g, L = state['g'], state['L']
    deg, T0, T_exact = period_amplitude_data(g, L)
    pts = measured_periods_at(g, L, np.arange(20.0, 161.0, 20.0))
    rfig = plt.figure(figsize=(7.4, 4.8), facecolor=BG)
    try:
        rfig.canvas.manager.set_window_title('Período × amplitude — Pêndulo v8')
    except Exception:
        pass
    ax = rfig.add_axes([0.12, 0.16, 0.84, 0.72], facecolor=PANEL)
    ax.axhline(T0, color=GOLD, lw=1.6, ls='--', zorder=4,
               label='T₀ linear = %.3f s (pequenas osc.)' % T0)
    ax.plot(deg, T_exact, color=MINT, lw=2.2, zorder=5, label='T exato (AGM)')
    if pts:
        md = [d for d, _ in pts]; mt = [t for _, t in pts]
        ax.plot(md, mt, 'o', mfc='none', mec=AMBER, mew=1.5, ms=7, zorder=6,
                label='T medido (RK4)')
    # amplitude atual + desvio %
    cur = state['theta0_deg']; curT = exact_period(g, L, np.radians(cur))
    ax.axvline(cur, color=WHITE, ls=':', lw=1.3, alpha=0.8, zorder=4)
    ax.plot([cur], [curT], 'o', color=BLUE, ms=8, zorder=7,
            markeredgecolor=WHITE, markeredgewidth=0.6)
    ax.annotate('θ₀ = %.0f°\nT = %.3f s  (+%.1f%%)' % (cur, curT, (curT/T0 - 1)*100),
                (cur, curT), textcoords='offset points', xytext=(10, -28),
                color=BLUE, fontsize=8.5, fontweight='bold')
    ax.set_xlim(0, deg.max())
    ax.set_ylim(0, min(T_exact.max(), T0 * 4.0) * 1.05)   # corta a explosão a 180°
    _style_popup(ax, 'amplitude θ₀ (°)', 'período T (s)',
                 'Período × amplitude  ·  exato vs. linear')
    ax.legend(facecolor=PANEL, edgecolor=BORDER2, fontsize=8.5, labelcolor=WHITE, loc='upper left')
    rfig.text(0.12, 0.035,
              'pequena amplitude: T ≈ T₀;  amplitude grande: o período cresce e dispara a 180°'
              '   ·   g=%.2f  L=%.2f' % (g, L), color=MUTED, fontsize=8, ha='left')
    try:
        rfig.show()
    except Exception:
        pass
    fig.canvas.draw_idle()

# ── Ajuda / controles ([h]) ──────────────────────────────────────────────────
# Janela separada (como as de análise) em vez de sobreposição no canvas: assim
# não interfere no blitting da animação principal. Guardamos o nº da figura pra
# não abrir várias cópias ao apertar [h] repetido.
_help = {'num': None}
_HELP_ROWS = [
    ('SIMULAÇÃO', None),
    ('espaço / p', 'pausar / continuar'),
    ('r',          'reset (volta aos padrões)'),
    ('← / →',      'avançar / voltar quadro a quadro (pausado)'),
    ('v',          'velocidade da animação (0,5× / 1× / 2×)'),
    ('PAINÉIS & COMPARAÇÃO', None),
    ('e / f',      'minimizar energia×tempo / espaço de fases'),
    ('c',          'comparar com a aproximação linear (fantasma)'),
    ('ANÁLISE (abre janela)', None),
    ('b',          'curva de ressonância'),
    ('k',          'seção de Poincaré'),
    ('j',          'diagrama de bifurcação'),
    ('a',          'período × amplitude'),
    ('SAÍDA & EXTRAS', None),
    ('s / d',      'salvar PNG / exportar CSV'),
    ('m',          'música de fundo (escolhe um arquivo)'),
    ('h',          'esta ajuda'),
]

def show_help(event=None):
    """[h] Janela com todos os controles — mais legível que a dica do rodapé."""
    if _help['num'] is not None and plt.fignum_exists(_help['num']):
        return                              # já está aberta
    hfig = plt.figure(figsize=(6.6, 6.4), facecolor=BG)
    _help['num'] = hfig.number
    try:
        hfig.canvas.manager.set_window_title('Controles — Pêndulo v8')
    except Exception:
        pass
    hfig.text(0.07, 0.95, 'CONTROLES', color=AMBER, fontsize=15, fontweight='bold',
              fontfamily=DISPLAY, va='top')
    hfig.text(0.07, 0.915, 'atalhos de teclado do Pêndulo Simples',
              color=MUTED2, fontsize=8.5, fontfamily=DISPLAY, va='top')
    y = 0.855
    for key, desc in _HELP_ROWS:
        if desc is None:                    # cabeçalho de seção
            hfig.text(0.07, y, key, color=TEAL, fontsize=8.5, fontweight='bold',
                      fontfamily=DISPLAY, va='center')
            y -= 0.046
        else:
            hfig.text(0.10, y, key, color=GOLD, fontsize=9.5, fontfamily='monospace',
                      va='center')
            hfig.text(0.38, y, desc, color=WHITE, fontsize=9, va='center')
            y -= 0.042
    hfig.text(0.07, 0.055,
              'Sliders ajustam L, m, θ₀, g, γ (atrito), A (força) e Ω (freq. forçante).\n'
              'Botões Lua / Marte / Terra / Júpiter trocam a gravidade.',
              color=MUTED, fontsize=8.5, va='bottom')
    try:
        hfig.show()
    except Exception:
        pass
    fig.canvas.draw_idle()

# ── Música de fundo (PyQt5.QtMultimedia; toca em loop) ───────────────────────
_music = {'player': None, 'playlist': None, 'on': False, 'name': ''}

def toggle_music(event=None):
    """[m] Música de fundo. Na 1ª vez abre um seletor para escolher um MP3/WAV;
    toca em loop. Os toques seguintes pausam/retomam. Não bloqueia a animação
    (usa o mesmo laço de eventos do Qt). Qualquer falha vira só uma mensagem."""
    try:
        from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent, QMediaPlaylist
        from PyQt5.QtCore import QUrl
        from PyQt5.QtWidgets import QFileDialog, QApplication
        if QApplication.instance() is None:   # sem janela Qt: evita abort do Qt
            status_txt.set_text('música: disponível só na janela do app')
            fig.canvas.draw_idle(); return
        if _music['player'] is None:
            path, _ = QFileDialog.getOpenFileName(
                None, 'Escolha a música de fundo', _output_dir(),
                'Áudio (*.mp3 *.wav *.ogg *.m4a);;Todos os arquivos (*.*)')
            if not path:
                status_txt.set_text('música: nenhum arquivo escolhido')
            else:
                pl = QMediaPlaylist()
                pl.addMedia(QMediaContent(QUrl.fromLocalFile(path)))
                pl.setPlaybackMode(QMediaPlaylist.Loop)
                mp = QMediaPlayer(); mp.setPlaylist(pl); mp.setVolume(35)
                _music.update(player=mp, playlist=pl, name=os.path.basename(path))
                mp.play(); _music['on'] = True
                status_txt.set_text('música: tocando ' + _music['name'])
        elif _music['on']:
            _music['player'].pause(); _music['on'] = False
            status_txt.set_text('música: pausada')
        else:
            _music['player'].play(); _music['on'] = True
            status_txt.set_text('música: tocando ' + _music['name'])
    except Exception as exc:
        status_txt.set_text('música: indisponível (%s)' % type(exc).__name__)
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
    elif k == 'c':
        toggle_linear()
    elif k == 'v':
        cycle_speed()
    elif k == 's':
        save_png()
    elif k == 'd':
        save_csv()
    elif k == 'b':
        show_resonance()
    elif k == 'k':
        show_poincare()
    elif k == 'j':
        show_bifurcation()
    elif k == 'a':
        show_period_amplitude()
    elif k == 'h':
        show_help()
    elif k == 'm':
        toggle_music()
    elif k in ('right', 'left'):
        if not state['paused']:
            toggle_pause()          # pausa antes de andar quadro a quadro
        _step(+1 if k == 'right' else -1)

btn_pp.on_clicked(toggle_pause)
btn_ap.on_clicked(apply_params)
btn_rst.on_clicked(do_reset)
btn_lin.on_clicked(toggle_linear)
btn_speed.on_clicked(cycle_speed)
btn_save.on_clicked(save_png)
btn_csv.on_clicked(save_csv)
btn_reson.on_clicked(show_resonance)
fig.canvas.mpl_connect('key_press_event', on_key)

if __name__ == '__main__':
    # blit=True: só os artistas animados são redesenhados a cada quadro (o fundo
    # estático fica cacheado), o que dá um ganho grande de FPS frente a blit=False.
    ani = animation.FuncAnimation(
        fig, update,
        init_func=_anim_init,
        interval=14,
        blit=True,
        cache_frame_data=False,
    )
    plt.show()
