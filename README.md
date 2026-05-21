# 🔭 mechanics-code

Simulações de mecânica clássica desenvolvidas em Python com `numpy` e `matplotlib`.

---

## 📁 Estrutura

```
mechanics-code/
├── pendulo-simples/
│   ├── pendulo_v1.py   # Simulação estática com gráficos de energia
│   └── pendulo_v2.py   # Simulação interativa com controles em tempo real
└── README.md
```

---

## 🌀 Pêndulo Simples

Simulação animada de um pêndulo simples com visualização da conservação de energia mecânica, integração numérica pelo método RK4 e espaço de fases.

### Versões

| Arquivo | Descrição |
|---|---|
| `pendulo_v1.py` | Versão inicial — animação + gráficos de Eₖ, Eₚ e E total |
| `pendulo_v2.py` | Versão interativa — sliders para massa, comprimento, ângulo e gravidade; botão pausar/continuar |

### Como executar

**1. Instale as dependências:**
```bash
pip install numpy matplotlib PyQt5
```

**2. Execute o script desejado:**
```bash
python pendulo-simples/pendulo_v2.py
```

### Parâmetros configuráveis (v2)

| Parâmetro | Intervalo | Padrão |
|---|---|---|
| Comprimento L | 0.3 – 3.0 m | 1.6 m |
| Massa m | 0.1 – 5.0 kg | 1.0 kg |
| Ângulo inicial θ₀ | 5° – 85° | 55° |
| Gravidade g | 1.0 – 25.0 m/s² | 9.81 m/s² |

### Física

A equação de movimento é integrada numericamente pelo método **Runge-Kutta de 4ª ordem (RK4)**:

$$\ddot{\theta} = -\frac{g}{L} \sin(\theta)$$

A energia mecânica total é conservada:

$$E_{mec} = \frac{1}{2}mv^2 + mgh = \text{constante}$$

---

## ⚙️ Requisitos

- Python 3.10+
- numpy
- matplotlib
- PyQt5

---

## 👤 Autor

**flaviocoding22** — trabalho do Leopoldo
