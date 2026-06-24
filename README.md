# 🔭 mechanics-code

Simulações de mecânica clássica desenvolvidas em Python com `numpy` e `matplotlib`.

---

## 📁 Estrutura

```
mechanics-code/
├── pendulo-simples/
│   ├── pendulo_v1.py … pendulo_v4.py   # versões iniciais (estática → interativa)
│   ├── pendulo_v5.py                   # RK4 + espaço de fases, layout redesenhado
│   ├── pendulo_v6.py                   # diagramas minimalistas, painéis minimizáveis, presets de g
│   ├── pendulo_v7.py                   # atrito viscoso, período exato, atalhos, tema neon
│   ├── pendulo_v8.py   ★ atual         # + comparação linear, período medido, velocidade e salvar PNG
│   └── pendulo_streamlit.py            # versão web (Streamlit)
└── README.md
```

---

## 🌀 Pêndulo Simples

Simulação animada de um pêndulo com visualização da troca de energia mecânica,
integração numérica por **Runge-Kutta de 4ª ordem (RK4)** e espaço de fases (θ × ω).

### Versões

| Arquivo | Descrição |
|---|---|
| `pendulo_v1.py` | Estática — animação + gráficos de Eₖ, Eₚ e E total |
| `pendulo_v2.py` | Interativa — sliders (m, L, θ₀, g) + pausar/continuar |
| `pendulo_v3.py` – `pendulo_v4.py` | Refinamentos da interface interativa |
| `pendulo_v5.py` | RK4 + espaço de fases; layout em duas colunas redesenhado |
| `pendulo_v6.py` | Diagramas minimalistas; painéis de gráfico minimizáveis; presets de gravidade |
| `pendulo_v7.py` | Atrito viscoso, período exato (grandes amplitudes), energia dissipada, atalhos e tema neon |
| **`pendulo_v8.py`** ★ | + comparação linear × não-linear, período medido, **força externa / ressonância**, **análise de caos** (Poincaré + bifurcação), arco de ângulo, velocidade, exportar PNG/CSV e animação por blitting (≈15× mais FPS) |
| `pendulo_streamlit.py` | Versão web (Streamlit) |

> **Recomendado:** `pendulo_v8.py` — é a versão mais completa.

### Como executar

**1. Instale as dependências:**
```bash
pip install numpy matplotlib PyQt5
```

**2. Execute a versão atual:**
```bash
python pendulo-simples/pendulo_v8.py
```

### ✨ Recursos da v8

- **Comparação linear × não-linear:** sobreposição "fantasma" do pêndulo de
  pequenas oscilações na simulação e no espaço de fases (tecla `c`) — torna
  visível por que a aproximação de pequeno ângulo erra em grandes amplitudes.
- **Três períodos lado a lado:** `T₀` (linear), `T` (amplitude exata via AGM) e
  `T` medido direto da trajetória integrada — validação do integrador.
- **Força externa / ressonância:** torque forçante `A·cos(ω_d·t)` opcional. Com
  a frequência forçante `Ω` próxima de `ω₀ = √(g/L)`, a amplitude e a energia
  crescem (ressonância) — o cabeçalho mostra `ω₀` e a razão `Ω/ω₀`, acende um
  aviso **RESSONÂNCIA**, e a tecla `b` (ou o botão) abre a **curva de
  ressonância** (amplitude × Ω), comparando a previsão **linear** com a
  resposta **não-linear real** — que mostra o pico "encurvado" e o salto
  (histerese) característicos do pêndulo de verdade.
- **Análise de caos** (pêndulo forçado): **seção de Poincaré** (`k`) revela o
  atrator estranho, e **diagrama de bifurcação** (`j`) mostra a cascata de
  duplicação de período rumo ao caos (varrendo a força A).
- **Arco de ângulo ao vivo** na simulação e **controle de velocidade** (0,5× / 1× / 2×).
- **Exportar** a janela em **PNG** (`s`) e a trajetória em **CSV** (`d`).
- **Animação otimizada por blitting:** só os artistas que se movem são
  redesenhados a cada quadro (o fundo estático fica cacheado) — cerca de **15×
  mais FPS** que redesenhar a figura inteira, mantendo a fluidez mesmo cheia.
- **Atrito viscoso (γ)**, **energia dissipada (ΔE)**, **painéis minimizáveis** e
  **presets de gravidade** com realce do ativo (herdados do v7).

### ⌨️ Atalhos de teclado

| Tecla | Ação |
|---|---|
| `espaço` | pausar / continuar |
| `r` | reset |
| `e` / `f` | minimizar painel de energia / espaço de fases |
| `c` | comparar com a aproximação linear |
| `v` | velocidade da animação (0,5× / 1× / 2×) |
| `b` | abrir a curva de ressonância |
| `k` / `j` | seção de Poincaré / diagrama de bifurcação |
| `s` / `d` | salvar PNG / exportar CSV |
| `←` / `→` | avançar / voltar quadro a quadro (pausado) |

### Parâmetros configuráveis

| Parâmetro | Intervalo | Padrão |
|---|---|---|
| Comprimento L | 0.3 – 3.0 m | 1.6 m |
| Massa m | 0.1 – 5.0 kg | 1.0 kg |
| Ângulo inicial θ₀ | 5° – 85° | 55° |
| Gravidade g | 1.0 – 25.0 m/s² | 9.81 m/s² |
| Atrito γ | 0.0 – 2.0 s⁻¹ | 0.0 (sem atrito) |
| Força externa A | 0.0 – 6.0 rad/s² | 0.0 (sem força) |
| Frequência forçante Ω | 0.1 – 6.0 rad/s | 2.5 rad/s |

Presets de gravidade: **Lua** 1.62 · **Marte** 3.71 · **Terra** 9.81 · **Júpiter** 24.79 m/s².

### 🔬 Física

A equação de movimento (com atrito viscoso e força externa opcionais) é
integrada por **RK4**:

$$\ddot{\theta} = -\frac{g}{L}\,\sin(\theta) \; - \; \gamma\,\dot{\theta} \; + \; A\cos(\omega_d\,t)$$

Com `γ = 0` e `A = 0` recai no pêndulo ideal e a energia mecânica total é conservada:

$$E = \tfrac{1}{2}mv^2 + mgh = \text{constante}$$

Com `γ > 0` há dissipação, e a energia perdida é `ΔE = E_0 - E(t)`.

O **período de pequenas oscilações** (aproximação linear) é:

$$T_0 = 2\pi\sqrt{\frac{L}{g}}$$

mas essa fórmula erra cada vez mais conforme a amplitude cresce. O **período exato**
usa a integral elíptica completa `K`, calculada pela média aritmético-geométrica (AGM):

$$T = 4\sqrt{\frac{L}{g}}\;K\!\left(\sin\tfrac{\theta_0}{2}\right) = \frac{2\pi\sqrt{L/g}}{\operatorname{AGM}\!\left(1,\,\cos\tfrac{\theta_0}{2}\right)}$$

Por exemplo, a 55° o período real é ~6% maior que `T₀`; a 85°, ~16% maior.

Para deixar isso tangível, a v8 também integra o **pêndulo linearizado**
(`θ'' = -(g/L)θ - γθ'`, válido só para θ pequeno) e o desenha como um
"fantasma" sobre a simulação e o espaço de fases — a defasagem entre as duas
soluções cresce visivelmente com a amplitude. E o **período medido** direto da
trajetória (tempo até o primeiro máximo de θ) confirma o valor exato `T` na
prática.

Com **força externa** (`A > 0`), o termo `A·cos(ω_d·t)` injeta energia. Quando a
frequência forçante `Ω` se aproxima da **frequência natural** `ω₀ = √(g/L)`, a
amplitude e a energia crescem fortemente — **ressonância** (visível como uma
espiral que cresce no espaço de fases e como `E(t)` subindo no gráfico de
energia). Como o pêndulo é não-linear, o pico de ressonância **se encurva**
para frequências menores (o período cresce com a amplitude) e aparece um
**salto/histerese** — a curva de ressonância (tecla `b`) mostra isso comparando
a previsão linear com a resposta real. Para forçamento forte em grande
amplitude, o movimento pode ficar **caótico**.

---

## ⚙️ Requisitos

- Python 3.10+
- numpy
- matplotlib
- PyQt5

---

## 👤 Autor

**flaviocoding22** — trabalho do Leopoldo