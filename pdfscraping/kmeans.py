import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

class AutoKMeans:
    def __init__(self, file_path):
        """
        Inicializa el objeto con el archivo CSV de datos.
        """
        self.file_path = file_path
        self.data = None
        self.X = None
        self.sse = []
        self.k_selected = None

    def load_data(self):
        """
        Carga y preprocesa los datos del archivo CSV.
        """
        self.data = pd.read_csv(self.file_path)
        # Convertir las columnas DPIR y FPI a valores numéricos
        self.data['DPIR'] = self.data['DPIR'].str.replace(',', '.').astype(float)
        self.data['FPI'] = self.data['FPI'].str.replace(',', '.').astype(float)
        # Preparar los datos para el algoritmo K-means
        self.X = self.data[['FPI', 'DPIR']].values

    def visualize_data(self):
        """
        Visualiza los datos iniciales en un gráfico de dispersión.
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(self.data['FPI'], self.data['DPIR'], alpha=0.7)
        plt.title('Relación entre FPI y DPIR (Datos Iniciales)')
        plt.xlabel('Frecuencia promedio de interrupciones (FPI)')
        plt.ylabel('Duración promedio de interrupciones reportadas (DPIR)')
        plt.grid(True)
        plt.show()

    def compute_elbow_method(self, k_range=range(1, 11)):
        """
        Calcula el valor óptimo de K utilizando el método del codo.
        """
        # Calcular SSE para diferentes valores de K
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(self.X)
            self.sse.append(kmeans.inertia_)

        # Cálculo de la distancia perpendicular para seleccionar el K óptimo
        x1, y1 = 1, self.sse[0]  # Punto inicial (K=1, SSE para K=1)
        x2, y2 = len(self.sse), self.sse[-1]  # Punto final (K máximo, SSE para K máximo)

        # Distancias de cada punto a la línea (K=1 a K=10)
        distancias = []
        for i in range(len(self.sse)):
            x0 = i + 1
            y0 = self.sse[i]
            numerador = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
            denominador = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            distancia = numerador / denominador
            distancias.append(distancia)

        # Seleccionar el K con la máxima distancia
        self.k_selected = distancias.index(max(distancias)) + 1

        # Graficar el método del codo con la línea vertical en K seleccionado
        plt.figure(figsize=(8, 6))
        plt.plot(k_range, self.sse, marker='o')
        plt.axvline(x=self.k_selected, color='r', linestyle='--', label=f'K óptimo = {self.k_selected}')
        plt.title('Método del Codo para Determinar el Número Óptimo de Clusters')
        plt.xlabel('Número de clusters (K)')
        plt.ylabel('Suma de los errores cuadráticos (SSE)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def apply_kmeans(self):
        """
        Aplica el algoritmo K-means utilizando el K óptimo seleccionado.
        """
        kmeans = KMeans(n_clusters=self.k_selected, random_state=0)
        kmeans.fit(self.X)

        # Obtener los clusters asignados a cada punto
        labels = kmeans.labels_
        # Obtener los centroides de cada cluster
        centroids = kmeans.cluster_centers_

        # Crear un gráfico de dispersión para visualizar los clusters después de aplicar K-means
        plt.figure(figsize=(8, 6))
        plt.scatter(self.X[:, 0], self.X[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X')  # Centroides en rojo
        plt.title(f'Agrupamiento K-means de Circuitos Eléctricos (K={self.k_selected})')
        plt.xlabel('Frecuencia promedio de interrupciones (FPI)')
        plt.ylabel('Duración promedio de interrupciones reportadas (DPIR)')
        plt.grid(True)
        plt.show()

# Ejemplo de uso
if __name__ == "__main__":
    auto_kmeans = AutoKMeans('incisoA.csv')
    auto_kmeans.load_data()
    auto_kmeans.visualize_data()
    auto_kmeans.compute_elbow_method()
    auto_kmeans.apply_kmeans()

    auto_kmeans = AutoKMeans('incisoB.csv')
    auto_kmeans.load_data()
    auto_kmeans.visualize_data()
    auto_kmeans.compute_elbow_method()
    auto_kmeans.apply_kmeans()
    

