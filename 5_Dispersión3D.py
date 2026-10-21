import pandas as pd
import plotly.express as px

# Carga del archivo
df = pd.read_csv("slump_test.csv")

# Limpieza de datos
df.columns = df.columns.str.strip()
df = df.drop(columns="No")
df = df.drop_duplicates()
fc = "Compressive Strength (28-day)(Mpa)"

# Dispersión 3D: cemento, agua y resistencia
fig = px.scatter_3d(
    df,
    x="Cement",
    y="Water",
    z=fc,
    color=fc,
    color_continuous_scale="Viridis",
    title="Cemento, agua y resistencia a 28 días",
)
fig.update_layout(
    scene=dict(
        xaxis_title="Cemento (kg/m3)",
        yaxis_title="Agua (kg/m3)",
        zaxis_title="Resistencia (MPa)",
    ),
    coloraxis_colorbar_title="MPa",
    width=900,
    height=700,
)
fig.update_traces(marker=dict(size=5))

fig.show()