# -*- coding: utf-8 -*-
"""
Pendulo Simples — Streamlit App
Requer: pip install streamlit plotly numpy
Rodar:  streamlit run pendulo_streamlit.py
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Paleta ────────────────────────────────────────────────────────────────────
BG     = '#07080D'
PANEL  = '#0F1120'
AMBER  = '#F5A623'
TEAL   = '#00C9A7'
GOLD   = '#FFD166'
MUTED  = '#5A638A'
WHITE  = '#DDE2F0'

# ── Página ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Pêndulo Simples',
    page_icon='🔵',
    layout='wide',
)

st.markdown(f"""
<style>
  /* Fundo geral */
  .stApp {{ background-color: {BG}; }}
  section[data-testid="stSidebar"] {{ background-color: #0C0E18; }}

  /* Título customizado */
  .titulo {{
      text-align: center;
      font-family: monospace;
      font-size: 2.2rem;
      font-weight: bold;
      color: {AMBER};
      letter-spacing: 4px;
      margin-bottom: 0;
  }}
  .subtitulo {{
      text-align: center;
      font-family: monospace;
      font-size: 0.78rem;
      color: {MUTED};
      margin-top: 4px;
      margin-bottom: 20px;
  }}
  /* Métricas */
  [data-testid="metric-container"] {{
      background-color: #0F1120;
      border: 1px solid #252B45;
      border-radius: 8px;
      padding: 10px 16px;
  }}
  [data-testid="metric-container"] label {{
      color: {MUTED} !important;
      font-family: monospace;
      font-size: 0.75rem;
  }}
  [data-testid="metric-container"] [data-testid="stMetricValue"] {{
      color: {WHITE} !important;
      font-family: monospace;
      font-size: 1.3rem;
  }}
  /* Slider labels */
  .stSlider label {{ color: {WHITE} !important; font-family: monospace; }}
  /* Sidebar texto */
  .stSidebar * {{ color: {WHITE}; font-family: monospace; }}
  hr {{ border-color: #1A1F35; }}
</style>
""", unsafe_allow_html=True)

# ── Cabeçalho ─────────────────────────────────────────────────────────────────
st.markdown('<p class="titulo">PÊNDULO SIMPLES</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitulo">simulação interativa &nbsp;·&nbsp; runge-kutta 4ª ordem</p>',
    unsafe_allow_html=True
)
st.markdown('<hr>', unsafe_allow_html=True)

# ── Sidebar — parâmetros ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<span style="color:{AMBER};font-weight:bold;font-family:monospace;font-size:0.85rem">PARÂMETROS</span>', unsafe_allow_html=True)
    st.markdown('---')

    L  = st.slider('Comprimento  L  (m)',      0.3,  3.0,  1.6,  0.05)
    m  = st.slider('Massa  m  (kg)',            0.1,  5.0,  1.0,  0.1)
    th = st.slider('Ângulo inicial  θ₀  (°)',   5.0, 85.0, 55.0,  1.0)
    g  = st.slider('Gravidade  g  (m/s²)',      1.0, 25.0,  9.81, 0.01)

    st.markdown('---')
    st.markdown(f'<span style="color:{MUTED};font-size:0.75rem;font-family:monospace">Arraste os sliders para atualizar a simulação em tempo real.</span>', unsafe_allow_html=True)

    st.markdown('---')
    # Planetas preset
    st.markdown(f'<span style="color:{AMBER};font-weight:bold;font-family:monospace;font-size:0.85rem">PRESETS DE GRAVIDADE</span>', unsafe_allow_html=True)
    st.markdown(f'<span style="color:{MUTED};font-size:0.75rem;font-family:monospace">Lua: 1.62 · Marte: 3.72 · Terra: 9.81 · Júpiter: 24.79</span>', unsafe_allow_html=True)

# ── RK4 ───────────────────────────────────────────────────────────────────────
DT    = 0.016
T_SIM = 14.0

@st.cache_data
def compute(g, L, m, theta0_deg):
    theta0 = np.radians(theta0_deg)
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
    vp = L*np.abs(om)
    Ek = 0.5*m*vp**2
    Ep = m*g*L*(1-np.cos(th))
    Et = Ek + Ep
    Tp = 2*np.pi*np.sqrt(L/g)
    return t, th, om, xb, yb, Ek, Ep, Et, Tp

t_arr, theta, omega, xb, yb, Ek, Ep, Et, Tp = compute(g, L, m, th)
Et0  = Et[0]
vmax = L * np.abs(omega).max()

# ── Métricas ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric('Período  T', f'{Tp:.3f} s')
c2.metric('Energia total  E₀', f'{Et0:.3f} J')
c3.metric('Vel. máxima  v', f'{vmax:.2f} m/s')
c4.metric('ω máximo', f'{np.abs(omega).max():.2f} rad/s')
c5.metric('h máxima', f'{L*(1-np.cos(np.radians(th))):.3f} m')

st.markdown('<hr>', unsafe_allow_html=True)

# ── Gráficos ──────────────────────────────────────────────────────────────────
col_sim, col_graficos = st.columns([1, 1.3])

plot_cfg = dict(
    paper_bgcolor=BG,
    plot_bgcolor=PANEL,
    font=dict(family='monospace', color=WHITE, size=11),
    margin=dict(l=50, r=20, t=40, b=40),
    xaxis=dict(gridcolor='#0E1020', zerolinecolor='#252B45', color=MUTED),
    yaxis=dict(gridcolor='#0E1020', zerolinecolor='#252B45', color=MUTED),
)

# ── Simulação (trajetória do pêndulo) ─────────────────────────────────────────
with col_sim:
    st.markdown(f'<span style="color:{MUTED};font-family:monospace;font-size:0.8rem;font-weight:bold">TRAJETÓRIA</span>', unsafe_allow_html=True)

    # Arco percorrido
    arc_a = np.linspace(-np.radians(th), np.radians(th), 300)

    fig_sim = go.Figure()

    # Arco fantasma
    fig_sim.add_trace(go.Scatter(
        x=L*np.sin(arc_a), y=-L*np.cos(arc_a),
        mode='lines',
        line=dict(color='#252B45', width=1, dash='dash'),
        showlegend=False, hoverinfo='skip'
    ))

    # Rastro da trajetória
    fig_sim.add_trace(go.Scatter(
        x=xb, y=yb,
        mode='lines',
        line=dict(color=AMBER, width=1.5),
        opacity=0.25,
        showlegend=False, hoverinfo='skip'
    ))

    # Fio (posição inicial)
    fig_sim.add_trace(go.Scatter(
        x=[0, xb[0]], y=[0, yb[0]],
        mode='lines',
        line=dict(color='#6878A0', width=2.5),
        showlegend=False, hoverinfo='skip'
    ))

    # Pivô
    fig_sim.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(color='#5A6080', size=12, symbol='circle'),
        showlegend=False, hoverinfo='skip'
    ))

    # Bob
    fig_sim.add_trace(go.Scatter(
        x=[xb[0]], y=[yb[0]],
        mode='markers',
        marker=dict(color=AMBER, size=22, symbol='circle',
                    line=dict(color='#FFD166', width=1.5)),
        showlegend=False,
        hovertemplate='x = %{x:.3f} m<br>y = %{y:.3f} m<extra></extra>'
    ))

    # h=0
    xlim = L * 1.35
    fig_sim.add_shape(type='line', x0=-xlim, x1=xlim, y0=-L, y1=-L,
                      line=dict(color=TEAL, width=0.8, dash='dot'))
    fig_sim.add_annotation(x=xlim*0.9, y=-L, text='h = 0',
                            showarrow=False, font=dict(color=TEAL, size=10))

    fig_sim.update_layout(
        **plot_cfg,
        xaxis=dict(**plot_cfg['xaxis'], range=[-xlim, xlim], scaleanchor='y'),
        yaxis=dict(**plot_cfg['yaxis'], range=[-L*1.35, L*0.55]),
        height=420,
        title=dict(text='', x=0.5),
    )
    st.plotly_chart(fig_sim, use_container_width=True)

# ── Energia × Tempo + Espaço de Fases ─────────────────────────────────────────
with col_graficos:
    fig2 = make_subplots(
        rows=2, cols=1,
        subplot_titles=['ENERGIA × TEMPO', 'ESPAÇO DE FASES'],
        vertical_spacing=0.14,
    )

    # Energia × Tempo
    fig2.add_trace(go.Scatter(x=t_arr, y=Ek, name='Ek  cinética',
        line=dict(color=AMBER, width=2), fill='tozeroy',
        fillcolor='rgba(245,166,35,0.06)'), row=1, col=1)
    fig2.add_trace(go.Scatter(x=t_arr, y=Ep, name='Ep  potencial',
        line=dict(color=TEAL, width=2), fill='tozeroy',
        fillcolor='rgba(0,201,167,0.06)'), row=1, col=1)
    fig2.add_trace(go.Scatter(x=t_arr, y=Et, name='E  total',
        line=dict(color=GOLD, width=1.2, dash='dash')), row=1, col=1)

    # Espaço de Fases
    # Colorir por velocidade angular
    fig2.add_trace(go.Scatter(
        x=np.degrees(theta), y=omega,
        mode='lines',
        line=dict(color=AMBER, width=1.8),
        name='fase',
        showlegend=False,
    ), row=2, col=1)

    # Ponto inicial
    fig2.add_trace(go.Scatter(
        x=[np.degrees(theta[0])], y=[omega[0]],
        mode='markers',
        marker=dict(color=TEAL, size=10, symbol='circle'),
        name='início', showlegend=False,
    ), row=2, col=1)

    fig2.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        font=dict(family='monospace', color=WHITE, size=10),
        margin=dict(l=50, r=20, t=50, b=40),
        height=500,
        legend=dict(
            bgcolor='#06080F', bordercolor='#252B45',
            font=dict(color=WHITE, size=10), x=0.75, y=0.98
        ),
    )
    for i in [1, 2]:
        fig2.update_xaxes(gridcolor='#0E1020', zerolinecolor='#252B45',
                          color=MUTED, row=i, col=1)
        fig2.update_yaxes(gridcolor='#0E1020', zerolinecolor='#252B45',
                          color=MUTED, row=i, col=1)

    fig2.update_xaxes(title_text='t (s)', row=1, col=1)
    fig2.update_yaxes(title_text='E (J)', row=1, col=1)
    fig2.update_xaxes(title_text='ângulo (°)', row=2, col=1)
    fig2.update_yaxes(title_text='vel. angular (rad/s)', row=2, col=1)

    # Títulos dos subplots na cor certa
    for ann in fig2.layout.annotations:
        ann.font.color = MUTED
        ann.font.size  = 11

    st.plotly_chart(fig2, use_container_width=True)

# ── Rodapé ────────────────────────────────────────────────────────────────────
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown(
    f'<p style="text-align:center;font-family:monospace;font-size:0.75rem;color:{MUTED}">'
    f'mechanics-code &nbsp;·&nbsp; flaviocoding22 &nbsp;·&nbsp; '
    f'L = {L} m &nbsp;·&nbsp; m = {m} kg &nbsp;·&nbsp; '
    f'θ₀ = {th:.0f}° &nbsp;·&nbsp; g = {g} m/s²</p>',
    unsafe_allow_html=True
)