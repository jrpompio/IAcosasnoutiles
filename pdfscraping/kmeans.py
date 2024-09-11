import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Ruta del archivo CSV
file_path = 'incisoA.csv'

# Leer el archivo CSV
df = pd.read_csv(file_path)

# Reemplazar comas por puntos en las columnas DPIR y FPI
df['DPIR'] = df['DPIR'].str.replace(',', '.').astype(float)
df['FPI'] = df['FPI'].str.replace(',', '.').astype(float)

# Seleccionar las columnas numéricas para el clustering
X = df[['Abonados', 'DPIR', 'FPI']]

# Algoritmo del codo
inertia = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    inertia.append(kmeans.inertia_)

# Calcular la pendiente de las diferencias de inercia
dif_inertia = np.diff(inertia)

# Comparar las pendientes entre los puntos
ratios = []
for i in range(1, len(dif_inertia)):
    ratio = abs(dif_inertia[i-1] / dif_inertia[i])
    ratios.append(ratio)

# Encontrar el valor de k donde la relación es más cercana a 1
k_optimo = np.argmin(np.abs(np.array(ratios) - 1)) + 2  # +2 para ajustar a k real

# Graficar el método del codo
plt.figure(figsize=(10, 5))
plt.plot(K, inertia, 'bx-')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Inercia')
plt.title('Método del Codo para Selección de k')
plt.axvline(x=k_optimo, color='r', linestyle='--', label=f'k óptimo={k_optimo}')
plt.legend()
plt.grid(True)
plt.show()

# Aplicar K-means con el valor óptimo de k
kmeans = KMeans(n_clusters=k_optimo, random_state=0).fit(X)

# Añadir los resultados del agrupamiento al DataFrame
df['Cluster'] = kmeans.labels_

# Graficar los clusters en función de DPIR y FPI
plt.figure(figsize=(10, 5))
plt.scatter(df['DPIR'], df['FPI'], c=df['Cluster'], cmap='viridis', s=100, alpha=0.7)
plt.xlabel('DPIR')
plt.ylabel('FPI')
plt.title(f'Agrupamiento con K-means (k={k_optimo})')
plt.grid(True)
plt.show()
