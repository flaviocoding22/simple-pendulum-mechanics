[README.md](https://github.com/user-attachments/files/28123046/README.md)
# 🔬 Pêndulo Simples — Conservação de Energia Mecânica

Simulação animada de um pêndulo simples com visualização em tempo real da conservação de energia mecânica, desenvolvida em Python com `matplotlib` e `numpy`.

---

## 📽️ O que a simulação mostra

| Painel | Descrição |
|--------|-----------|
| **Diagrama do Pêndulo** | Animação do bob com rastro gradiente, seta de velocidade e barra de altura |
| **Energia × Tempo** | Curvas de Eₖ, Eₚ e E_total desenhadas progressivamente |
| **Espaço de Fases** | Órbita elíptica (θ × ω) com ponto animado |
| **Ângulo × Tempo** | Oscilação senoidal do ângulo θ |
| **Proporção de Energia** | Barra dinâmica mostrando % de Eₖ e Eₚ em tempo real |

---

## ⚙️ Parâmetros da simulação

```
Comprimento do fio  →  L  = 1.6 m
Massa do bob        →  m  = 1.0 kg
Ângulo inicial      →  θ₀ = 55°
Gravidade           →  g  = 9.81 m/s²
Período analítico   →  T  ≈ 2π√(L/g)
```

---

## 🧮 Física envolvida

A energia mecânica total se conserva na ausência de atrito:

$$E_{mec} = E_k + E_p = \frac{1}{2}mv^2 + mgh = \text{constante}$$

A equação do movimento é integrada numericamente pelo método de **Runge-Kutta de 4ª ordem (RK4)**:

$$\ddot{\theta} = -\frac{g}{L}\sin(\theta)$$

Sem a aproximação de pequenos ângulos — válido para qualquer amplitude.

---

## 🚀 Como executar

### 1. Instale as dependências

```bash
pip install numpy matplotlib
```

> No Spyder: cole `!pip install numpy matplotlib` no console IPython e pressione Enter.

### 2. Execute o script

```bash
python pendulo.py
```

> No Spyder: abra o arquivo e pressione **F5**.

---

## 📁 Estrutura do repositório

```
pendulo-energia/
│
├── pendulo.py      # Simulação principal
└── README.md       # Este arquivo
```

---

## 🛠️ Tecnologias

- **Python 3.x**
- **NumPy** — cálculo numérico e integração RK4
- **Matplotlib** — visualização e animação

---

## 📄 Licença

Este projeto é de uso livre para fins educacionais.
