import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# Cargar el archivo CSV proporcionado por el usuario
data = pd.read_csv('incisoA.csv')

# Convertir las columnas DPIR y FPI a valores numéricos
data['DPIR'] = data['DPIR'].str.replace(',', '.').astype(float)
data['FPI'] = data['FPI'].str.replace(',', '.').astype(float)

# Crear un gráfico de dispersión para observar los puntos iniciales
plt.figure(figsize=(8, 6))
plt.scatter(data['FPI'], data['DPIR'], alpha=0.7)
plt.title('Relación entre FPI y DPIR (Datos Iniciales)')
plt.xlabel('Frecuencia promedio de interrupciones (FPI)')
plt.ylabel('Duración promedio de interrupciones reportadas (DPIR)')
plt.grid(True)
plt.show()

# Preparar los datos para el algoritmo K-means
X = data[['FPI', 'DPIR']].values

# Aplicar el método del codo
sse = []
k_range = range(1, 11)  # Probar para valores de K de 1 a 10
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

# Graficar el método del codo
plt.figure(figsize=(8, 6))
plt.plot(k_range, sse, marker='o')
plt.title('Método del Codo para Determinar el Número Óptimo de Clusters')
plt.xlabel('Número de clusters (K)')
plt.ylabel('Suma de los errores cuadráticos (SSE)')
plt.grid(True)
plt.show()

# Método del codo: cálculo de la distancia perpendicular
x1, y1 = 1, sse[0]  # Punto inicial (K=1, SSE para K=1)
x2, y2 = len(sse), sse[-1]  # Punto final (K máximo, SSE para K máximo)

# Distancias de cada punto a la línea (K=1 a K=10)
distancias = []
for i in range(len(sse)):
    x0 = i + 1
    y0 = sse[i]
    numerador = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominador = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    distancia = numerador / denominador
    distancias.append(distancia)

# Seleccionar el K con la máxima distancia
k_selected = distancias.index(max(distancias)) + 1
print(f'El número óptimo de clusters seleccionado según el método del codo es: K = {k_selected}')

# Crear el modelo KMeans con el valor óptimo de K
kmeans = KMeans(n_clusters=k_selected, random_state=0)

# Ajustar el modelo a los datos
kmeans.fit(X)

# Obtener los clusters asignados a cada punto
labels = kmeans.labels_

# Obtener los centroides de cada cluster
centroids = kmeans.cluster_centers_

# Crear un gráfico de dispersión para visualizar los clusters después de aplicar K-means
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X')  # Centroides en rojo
plt.title(f'Agrupamiento K-means de Circuitos Eléctricos (K={k_selected})')
plt.xlabel('Frecuencia promedio de interrupciones (FPI)')
plt.ylabel('Duración promedio de interrupciones reportadas (DPIR)')
plt.grid(True)
plt.show()
