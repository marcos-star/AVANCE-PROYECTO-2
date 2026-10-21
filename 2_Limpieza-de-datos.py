
import pandas as pd
df = pd.read_csv("slump_test.csv")

# Eliminar la columna 'No' (solo es un índice, y evita que filas idénticas pasen como distintas)
df = df.drop(columns="No")

# Valores nulos por columna
print("Valores nulos por columna:")
print(df.isnull().sum())

# Filas duplicadas
duplicados = df.duplicated().sum()
print(f"Filas duplicadas: {duplicados}")

# Eliminar duplicados
df = df.drop_duplicates()
print(f"Filas duplicadas eliminadas: {duplicados}")
print(f"Dimensiones después de eliminar duplicados: {df.shape}")