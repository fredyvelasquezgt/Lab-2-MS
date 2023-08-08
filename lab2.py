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

