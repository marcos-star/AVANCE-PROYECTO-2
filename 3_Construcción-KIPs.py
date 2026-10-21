import pandas as pd

# Cargar el dataset y eliminar filas duplicadas
df = pd.read_csv("slump_test.csv")
df.columns = df.columns.str.strip()
df = df.drop(columns="No")
df = df.drop_duplicates()

# Definir la variable objetivo y las otras variables de salida
fc = "Compressive Strength (28-day)(Mpa)"
slump = "SLUMP(cm)"
flow = "FLOW(cm)"

# Calcular KPIs
resistencia_promedio = df[fc].mean()
resistencia_maxima = df[fc].max()
resistencia_minima = df[fc].min()
cantidad_registros = df.shape[0]

slump_promedio = df[slump].mean()
slump_maximo = df[slump].max()
slump_minimo = df[slump].min()

flow_promedio = df[flow].mean()
flow_maximo = df[flow].max()
flow_minimo = df[flow].min()

cemento_promedio = df["Cement"].mean()
agua_promedio = df["Water"].mean()
relacion_agua_cemento = (df["Water"] / df["Cement"]).mean()

# Mostrar resultados
print(f"Resistencia promedio: {resistencia_promedio:.2f} MPa")
print(f"Resistencia máxima: {resistencia_maxima:.2f} MPa")
print(f"Resistencia mínima: {resistencia_minima:.2f} MPa")
print(f"Número de registros: {cantidad_registros}")
print(f"Slump promedio: {slump_promedio:.2f} cm")
print(f"Slump máximo: {slump_maximo:.2f} cm")
print(f"Slump mínimo: {slump_minimo:.2f} cm")
print(f"Flow promedio: {flow_promedio:.2f} cm")
print(f"Flow máximo: {flow_maximo:.2f} cm")
print(f"Flow mínimo: {flow_minimo:.2f} cm")
print(f"Cemento promedio: {cemento_promedio:.2f} kg/m3")
print(f"Agua promedio: {agua_promedio:.2f} kg/m3")
print(f"Relación agua/cemento promedio: {relacion_agua_cemento:.2f}")