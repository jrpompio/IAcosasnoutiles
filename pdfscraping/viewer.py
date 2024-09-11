import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo CSV proporcionado por el usuario
data = pd.read_csv('incisoA.csv')

# Convertir las columnas DPIR y FPI a valores numéricos
data['DPIR'] = data['DPIR'].str.replace(',', '.').astype(float)
data['FPI'] = data['FPI'].str.replace(',', '.').astype(float)

# Crear un gráfico de dispersión para observar los puntos
plt.figure(figsize=(8, 6))
plt.scatter(data['FPI'], data['DPIR'], alpha=0.7)
plt.title('Relación entre FPI y DPIR')
plt.xlabel('Frecuencia promedio de interrupciones (FPI)')
plt.ylabel('Duración promedio de interrupciones reportadas (DPIR)')
plt.grid(True)
plt.show()
