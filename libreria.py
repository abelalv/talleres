import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, FloatSlider

# Definimos la sucesión
def cardiac_signal(n, S0, alpha):
    return S0 * np.sin(alpha / n)

# Función para graficar
def plot_sequence(S0=1.0, alpha=1.0, N=100):
    ns = np.arange(1, N + 1)
    signals = cardiac_signal(ns, S0, alpha)

    plt.figure(figsize=(7, 4))
    plt.plot(ns, signals, 'o-', label=r"$S_n = S_0 \sin(\alpha/n)$")
    plt.xlabel("n (número de medición)")
    plt.ylabel("Señal medida (mV)")
    plt.title(rf"Sucesión cardíaca: $S_0={S0}$, $\alpha={alpha}$, $N={N}$")
    plt.axhline(0, linestyle='--')
    plt.grid(True)
    plt.legend()
    plt.show()

# Widget interactivo con 3 sliders: S0, alpha y N
interact(
    plot_sequence,
    S0=FloatSlider(value=1.0, min=0.1, max=3.0, step=0.1, description="S₀"),
    alpha=FloatSlider(value=1.0, min=0.1, max=5.0, step=0.1, description="α"),
    N=IntSlider(value=100, min=10, max=300, step=10, description="N")
)
