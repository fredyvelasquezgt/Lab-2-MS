import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Función para generar n muestras aleatorias uniformemente distribuidas en el intervalo (0, 1)
def generar_muestras_uniformes(n):
    return np.random.uniform(0, 1, n)

# Función para calcular la media de un conjunto de muestras
def calcular_media(muestras):
    return np.mean(muestras)

# Función para calcular la varianza de un conjunto de muestras
def calcular_varianza(muestras):
    return np.var(muestras)

# Función para calcular la media normalizada de una muestra
def calcular_media_normalizada(media_muestra, mu, var, n):
    return (media_muestra - mu) / (np.sqrt(var / n))

# Función para graficar el histograma y la densidad normal
def graficar_histograma_y_densidad_normal(datos, titulo, bins=30):
    datos = np.array(datos)  # Convertir datos a un array numpy
    plt.figure(figsize=(8, 6))
    plt.hist(datos, bins=bins, density=True, alpha=0.6, label='Histograma')
    x = np.linspace(datos.min(), datos.max(), 100)
    plt.plot(x, norm.pdf(x), 'r', label='Densidad normal N(0,1)')
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia relativa')
    plt.title(titulo)
    plt.legend()
    plt.show()


# Función para graficar la distribución acumulada empírica y la distribución normal
def graficar_distribucion_acumulada_empirica(datos):
    plt.figure(figsize=(8, 6))
    datos_ordenados = np.sort(datos)
    y = np.arange(1, len(datos) + 1) / len(datos)
    plt.plot(datos_ordenados, y, marker='.', linestyle='none', label='Frecuencia relativa acumulada')
    x = np.linspace(-4, 4, 100)
    plt.plot(x, norm.cdf(x), 'r', label='Distribución acumulada N(0,1)')
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia relativa acumulada')
    plt.title('Función de distribución acumulada empírica y N(0,1)')
    plt.legend()
    plt.show()

# Tamaños de muestra y cantidad de experimentos para las pruebas
ns = [20, 40, 60, 80, 100]
Ns = [50, 100, 1000, 10000]

mu = 0.5  # Media de la distribución uniforme en (0, 1)
variance = 0.05  # Nueva varianza de la distribución uniforme en (0, 1)

# Generar gráficas para cada valor de n
for n in ns:
    # Realizar N experimentos y calcular las medias muestrales
    sample_means = [calcular_media(generar_muestras_uniformes(n)) for _ in range(max(Ns))]
    # Calcular las medias normalizadas para la distribución normal N(0,1)
    normalized_means = [calcular_media_normalizada(mean, mu, variance, n) for mean in sample_means]
    # Graficar el histograma y la densidad normal
    graficar_histograma_y_densidad_normal(normalized_means, f'Histograma y densidad normal para n={n}')

# Generar gráficas de distribución acumulada empírica para cada valor de N
for N in Ns:
    # Realizar N experimentos y calcular las medias muestrales para cada n
    sample_means = [calcular_media(generar_muestras_uniformes(n)) for _ in range(N)]
    # Calcular las medias normalizadas para la distribución normal N(0,1) para cada n
    normalized_means = [calcular_media_normalizada(mean, mu, variance, n) for n, mean in zip(ns, sample_means)]
    # Graficar la distribución acumulada empírica y la distribución normal
    graficar_distribucion_acumulada_empirica(normalized_means)
