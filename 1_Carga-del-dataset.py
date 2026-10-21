# Importar librerías
import pandas as pd

# Cargar el dataset
df = pd.read_csv("slump_test.csv")

# Visualizar la información
print(df.head())      # Muestra las primeras 5 filas
df.info()             # Muestra columnas, tipos de datos y valores no nulos
print(df.shape)       # Muestra (cantidad de filas, cantidad de columnas)
print(df.isnull().sum())  # Cantidad de valores nulos por columna