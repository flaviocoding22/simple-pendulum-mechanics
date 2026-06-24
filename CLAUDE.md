# CLAUDE.md

Orientações para o Claude Code trabalhar neste repositório.

## Visão geral

`mechanics-code` — simulações de mecânica clássica em Python (`numpy` + `matplotlib`).
O artefato principal é o **pêndulo simples interativo**, que evoluiu de `v1` a `v8`.

- **Arquivo atual / recomendado:** `pendulo-simples/pendulo_v8.py`
- Versões anteriores (`v1`…`v7`) são mantidas como histórico; não precisam ser editadas.
- `pendulo-simples/pendulo_streamlit.py` é uma variante web (Streamlit), independente.

Cada `pendulo_vN.py` é um **script único, autocontido** (sem pacote/módulos próprios).

## Rodar

```bash
pip install numpy matplotlib PyQt5
python pendulo-simples/pendulo_v8.py
```

Abre uma janela interativa (backend Qt). É uma animação em tempo real com sliders,
botões e atalhos de teclado.

## Testar/verificar sem GUI (importante)

Não há suíte de testes. Para validar mudanças sem abrir janela, rode o script
**offscreen** com o backend `Agg`, importando via `importlib` e exercitando as
funções. Padrão usado neste projeto:

```python
import matplotlib; matplotlib.use('Agg')
import importlib.util, warnings, os
warnings.filterwarnings('error', message='.*missing from font.*')  # glyph faltante = erro
spec = importlib.util.spec_from_file_location('p8', os.path.abspath('pendulo_v8.py'))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
for _ in range(60): mod.update(0)          # avança quadros
mod.fig.canvas.draw()                       # força render (pega glyph faltante)
```

Notas ao testar:
- **Sempre** trate "missing from font" como erro (veja as fontes abaixo).
- Para **renderizar um PNG** da janela principal, primeiro desligue o flag de
  animação dos artistas (senão o `blit` os exclui do `savefig`):
  `for a in mod.ANIM: a.set_animated(False)` → `fig.savefig(...)` → religue.
- Os atalhos podem ser disparados com `mod.on_key(types.SimpleNamespace(key='c'))`.
- Limpe quaisquer `pendulo_v8_*.png/.csv` gerados pelos testes (export salva na
  pasta do script).
- O console do Windows é cp1252: **não** imprima caracteres unicode (●, θ…) no
  `print` dos testes — imprima flags/booleanos.

## Arquitetura do v8 (mapa mental)

- `compute(g, L, m, θ0, γ, A, ω_d)` — integra por **RK4** o pêndulo não-linear
  (`θ'' = -(g/L)sinθ - γθ' + A·cos(ω_d·t)`) **e** o linearizado (fantasma de
  comparação). Devolve trajetórias, energias e os períodos.
- `state` (dict) — todos os parâmetros e flags de UI.
- `rebuild()` — recomputa e (re)cria todos os artistas dentro de `aobj` (dict) e
  monta `ANIM` (lista dos artistas redesenhados por quadro).
- `_draw_frame(i)` — atualiza os artistas para o quadro `i`.
- `update()` / `FuncAnimation` — laço de animação (`blit=True`).
- `relayout()` + `reflow_controls()` — posicionam eixos/controles (coords em
  fração da figura) e tratam os **painéis minimizáveis**.
- Janelas de análise (popups separados): `show_resonance` (curva linear ×
  não-linear), `show_poincare` (seção de Poincaré), `show_bifurcation` (diagrama
  de bifurcação, integrador **vetorizado** `bifurcation_data`).

## Convenções e armadilhas (ler antes de editar)

### Fontes
- Corpo em **Segoe UI** (sans), números em **Consolas** (mono) — `rcParams`.
- ⚠️ **Segoe UI não tem** vários glyphs de UI: `▾ ▸ ▶ ✓ ↺ ◈`. Use os que ele tem:
  `▼ ► ‖ ▪ ●`. Botões "confirmar/refresh" ficam **sem ícone**.
- DejaVu Sans é o fallback, mas o fallback **emite warning** — por isso o teste
  com warnings-as-error. Qualquer glyph novo precisa existir no Segoe UI.

### Animação por blitting (`blit=True`)
- Só os artistas em `ANIM` (com `animated=True`) são redesenhados por quadro; o
  resto é fundo cacheado. Ganho de ~15× de FPS vs. redesenhar a figura inteira.
- `update()` retorna **apenas artistas em eixos visíveis** — ao minimizar um
  painel, seus artistas são pulados (senão dá *glitch* ao blitar eixo colapsado).
- `toggle_panel()` faz `fig.canvas.draw()` **síncrono** (não `draw_idle`) após o
  relayout, para refazer o fundo cacheado de uma vez.
- `save_png()` desliga `animated` antes do `savefig` e religa depois.

### Performance (lições aprendidas)
- **Nunca** use `imshow`/gradiente redesenhado por quadro: com `blit=False` (ou
  dentro de eixo animado) isso **dobrou** o tempo por quadro e matou a fluidez.
  Mantenha primitivas vetoriais leves no laço.
- Cálculos pesados (varredura de bifurcação) rodam **em popup, sob demanda**, e
  são **vetorizados com numpy** (integrar todos os A de uma vez) — não no laço de
  animação.

### Estilo visual
- Tema neon sobre quase-preto (`BG = '#06070C'`). As constantes `AMBER/TEAL/
  GOLD/BLUE/VIOLET/RED/MINT` são os acentos por papel; mudar a cor de um papel é
  só mudar a constante.
- Layout é todo manual (`fig.add_axes([x,y,w,h])` em fração da figura). Não há
  `tight_layout`/gridspec — posições são calculadas à mão e reflowadas.

## Física implementada

- RK4 (livre, amortecido e forçado) + versão linearizada para comparação.
- Período **exato** para grande amplitude via integral elíptica (AGM); período
  **medido** da trajetória; período linear `T₀ = 2π√(L/g)`.
- Pêndulo **forçado/ressonância**: curva de ressonância (forma fechada linear +
  varredura não-linear real, com encurvamento e histerese).
- **Caos**: seção de Poincaré (atrator estranho) e diagrama de bifurcação
  (cascata de duplicação de período).

## Convenções gerais do repositório

- Comentários e textos de UI em **português**.
- Sem dependências além de `numpy`, `matplotlib`, `PyQt5` (e `streamlit` só na
  variante web). Não introduza novas dependências sem necessidade.
- O `README.md` documenta as versões e recursos para o usuário final — mantenha-o
  em dia ao adicionar recursos ao `v8`.
