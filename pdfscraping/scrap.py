import tabula
import pandas as pd

pdf_path = "ic2023.pdf"
fileName = "incisoB"

dfs = tabula.read_pdf(pdf_path, pages='57-66', multiple_tables=True)

# Lista para guardar los nombres de los archivos CSV generados
csv_files = []

# Guarda cada DataFrame como un archivo CSV
for i, df in enumerate(dfs):
    csv_path = f"{fileName}_{i + 1}.csv"
    df.to_csv(csv_path, index=False)
    csv_files.append(csv_path)  # Agrega el nombre del archivo a la lista
    print(f"Tabla {i + 1} guardada como {csv_path}")

# Función para corregir secuencias de comas en el archivo CSV
def fix_commas_in_csv(file_path):
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        content = file.read()
    
    # Reemplaza las secuencias de dos comas seguidas por una sola coma
    content = content.replace(',,', ',')
    
    # Sobrescribe el archivo con el contenido corregido
    with open(file_path, 'w', newline='', encoding='utf-8') as file:
        file.write(content)

# Corrige las secuencias de comas en cada archivo CSV
for file in csv_files:
    fix_commas_in_csv(file)

# Imprime la lista de nombres de archivos CSV generados
print("\nArchivos CSV generados y corregidos:")
for file in csv_files:
    print(file)
