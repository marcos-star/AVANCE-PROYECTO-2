import pandas as pd
import plotly.express as px

# Cargar el dataset y preparar la variable objetivo
df = pd.read_csv("slump_test.csv")
df.columns = df.columns.str.strip()
df = df.drop(columns="No")
df = df.drop_duplicates()
fc = "Compressive Strength (28-day)(Mpa)"

# Crear la dispersión de cemento vs. resistencia
fig = px.scatter(
    df,
    x="Cement",
    y=fc,
    trendline="ols",
    title="Cemento vs. resistencia a 28 días",
)
fig.update_layout(
    xaxis_title="Cemento (kg/m3)",
    yaxis_title="Resistencia a 28 días (MPa)",
)

fig.show()