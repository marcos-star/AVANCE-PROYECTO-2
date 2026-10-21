import pandas as pd
from sklearn.linear_model import LinearRegression

# Cargar el dataset, normalizar encabezados y eliminar filas duplicadas
df = pd.read_csv("slump_test.csv")
df.columns = df.columns.str.strip()
df = df.drop(columns="No")
df = df.drop_duplicates()

# Definir la variable objetivo y las variables de entrada
fc = "Compressive Strength (28-day)(Mpa)"
X = df[["Cement", "Slag", "Fly ash", "Water", "SP", "Coarse Aggr.", "Fine Aggr."]]
y = df[fc]

# Entrenar el modelo de regresión lineal múltiple
modelo = LinearRegression()
modelo.fit(X, y)

# Predecir la resistencia para una mezcla nueva
mezcla_nueva = pd.DataFrame({
    "Cement": [250], "Slag": [100], "Fly ash": [100], "Water": [190],
    "SP": [8], "Coarse Aggr.": [900], "Fine Aggr.": [700],
})
resistencia = modelo.predict(mezcla_nueva)[0]
print(f"Resistencia predicha: {resistencia:.2f} MPa")

# Calcular R² sobre los mismos datos usados para entrenar
r2 = modelo.score(X, y)
print(f"R² en los datos de entrenamiento: {r2:.4f}")