import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# a) Ley Débil de los Grandes Números
n_values = 1000000
uniform_samples = np.random.rand(n_values)
partial_means = np.cumsum(uniform_samples) / np.arange(1, n_values + 1)

plt.figure(figsize=(10, 6))
plt.plot(partial_means, label='Medias Parciales')
plt.axhline(y=0.5, color='r', linestyle='--', label='Media de la distribución')
plt.xlabel('n')
plt.ylabel('Media')
plt.legend()
plt.title('Ley Débil de los Grandes Números')
plt.show()

# b) Teorema del Límite Central
n_values = [20, 40, 60, 80, 100]
N_values = [50, 100, 1000, 10000]

plt.figure(figsize=(15, 12))

for n in n_values:
    centered_means = []
    for _ in range(max(N_values)):
        sample = np.random.rand(n)
        mean = np.mean(sample)
        centered_means.append(mean - 0.5)
    plt.subplot(2, 2, n_values.index(n) + 1)
    plt.hist(centered_means, bins=30, density=True, label=f'n = {n} Promedios')
    x = np.linspace(-0.5, 0.5, 100)
    plt.plot(x, norm.pdf(x, loc=0, scale=np.sqrt(1/(12*n))), label='Distribución Normal')
    plt.xlabel('Valor')
    plt.ylabel('Densidad de Probabilidad')
    plt.legend()
    plt.title(f'Teorema del Límite Central - n = {n}')

plt.tight_layout()
plt.show()
