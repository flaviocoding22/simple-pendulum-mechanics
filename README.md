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
│   ├── pendulo_v7.py   ★ atual         # + atrito viscoso, período exato e atalhos de teclado
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
| **`pendulo_v7.py`** ★ | + atrito viscoso opcional, período exato (grandes amplitudes), energia dissipada, atalhos de teclado e realce do preset ativo |
| `pendulo_streamlit.py` | Versão web (Streamlit) |

> **Recomendado:** `pendulo_v7.py` — é a versão mais completa.

### Como executar

**1. Instale as dependências:**
```bash
pip install numpy matplotlib PyQt5
```

**2. Execute a versão atual:**
```bash
python pendulo-simples/pendulo_v7.py
```

### ✨ Recursos da v7

- **Atrito viscoso opcional (γ):** com `γ = 0` é o pêndulo ideal; com `γ > 0` a
  energia decai e o espaço de fases vira uma espiral que converge ao repouso.
- **Período exato para grandes amplitudes:** mostra o período linear `T₀` e o
  período real `T` (corrigido pela amplitude) lado a lado, com o erro percentual.
- **Energia dissipada (ΔE)** em tempo real no painel da simulação.
- **Painéis de gráfico minimizáveis:** clique na barra de título (ou use `e`/`f`)
  para esconder/mostrar cada gráfico — o layout se reorganiza sem deixar vão vazio.
- **Presets de gravidade:** Lua, Marte, Terra e Júpiter (o ativo fica realçado).

### ⌨️ Atalhos de teclado

| Tecla | Ação |
|---|---|
| `espaço` | pausar / continuar |
| `r` | reset |
| `e` / `f` | minimizar painel de energia / espaço de fases |
| `←` / `→` | avançar / voltar quadro a quadro (pausado) |

### Parâmetros configuráveis

| Parâmetro | Intervalo | Padrão |
|---|---|---|
| Comprimento L | 0.3 – 3.0 m | 1.6 m |
| Massa m | 0.1 – 5.0 kg | 1.0 kg |
| Ângulo inicial θ₀ | 5° – 85° | 55° |
| Gravidade g | 1.0 – 25.0 m/s² | 9.81 m/s² |
| Atrito γ | 0.0 – 2.0 s⁻¹ | 0.0 (sem atrito) |

Presets de gravidade: **Lua** 1.62 · **Marte** 3.71 · **Terra** 9.81 · **Júpiter** 24.79 m/s².

### 🔬 Física

A equação de movimento (com atrito viscoso opcional) é integrada por **RK4**:

$$\ddot{\theta} = -\frac{g}{L}\,\sin(\theta) \; - \; \gamma\,\dot{\theta}$$

Com `γ = 0` recai no pêndulo ideal e a energia mecânica total é conservada:

$$E = \tfrac{1}{2}mv^2 + mgh = \text{constante}$$

Com `γ > 0` há dissipação, e a energia perdida é `ΔE = E_0 - E(t)`.

O **período de pequenas oscilações** (aproximação linear) é:

$$T_0 = 2\pi\sqrt{\frac{L}{g}}$$

mas essa fórmula erra cada vez mais conforme a amplitude cresce. O **período exato**
usa a integral elíptica completa `K`, calculada pela média aritmético-geométrica (AGM):

$$T = 4\sqrt{\frac{L}{g}}\;K\!\left(\sin\tfrac{\theta_0}{2}\right) = \frac{2\pi\sqrt{L/g}}{\operatorname{AGM}\!\left(1,\,\cos\tfrac{\theta_0}{2}\right)}$$

Por exemplo, a 55° o período real é ~6% maior que `T₀`; a 85°, ~16% maior.

---

## ⚙️ Requisitos

- Python 3.10+
- numpy
- matplotlib
- PyQt5

---

## 👤 Autor

**flaviocoding22** — trabalho do Leopoldo