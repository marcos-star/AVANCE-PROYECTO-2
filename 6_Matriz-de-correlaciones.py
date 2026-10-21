import pandas as pd
import plotly.express as px

# Carga del archivo
df = pd.read_csv("slump_test.csv")

# Limpieza de datos
df.columns = df.columns.str.strip()
df = df.drop(columns="No")
df = df.drop_duplicates()

# Matriz de correlación completa
corr = df.corr()

fig = px.imshow(
    corr,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="Matriz de correlación de las variables del concreto",
)
fig.update_layout(width=900, height=800)
fig.update_xaxes(tickangle=-45)

fig.show()