import pandas as pd
import numpy as np
import plotly.express as px

# Cargar el dataset y preparar la variable objetivo
df = pd.read_csv("slump_test.csv")
df.columns = df.columns.str.strip()
df = df.drop(columns="No")
df = df.drop_duplicates()
fc = "Compressive Strength (28-day)(Mpa)"

# Calcular estadísticos
media = df[fc].mean()
mediana = df[fc].median()
desviacion = df[fc].std()

# Moda: centro del intervalo (bin) con más mezclas
nbins = 12
conteos, bordes = np.histogram(df[fc], bins=nbins)
i = conteos.argmax()
moda = (bordes[i] + bordes[i + 1]) / 2

# Crear el histograma
fig = px.histogram(
    df,
    x=fc,
    nbins=nbins,
    title="Distribución de la resistencia del concreto a 28 días",
)

# Líneas de media, mediana y moda
fig.add_vline(x=media, line_dash="dash", line_color="red",
              annotation_text=f"Media: {media:.2f}", annotation_position="top left")
fig.add_vline(x=mediana, line_dash="dashdot", line_color="blue",
              annotation_text=f"Mediana: {mediana:.2f}", annotation_position="top right")
fig.add_vline(x=moda, line_dash="dot", line_color="green",
              annotation_text=f"Moda: {moda:.2f}", annotation_position="bottom right")

# Desviación estándar en una esquina
fig.add_annotation(
    text=f"Desviación estándar: {desviacion:.2f} MPa",
    xref="paper", yref="paper", x=0.98, y=0.98,
    showarrow=False, bgcolor="white",
)

fig.update_layout(
    xaxis_title="Resistencia a 28 días (MPa)",
    yaxis_title="Cantidad de mezclas",
)

fig.show()