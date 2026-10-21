import pandas as pd
import plotly.express as px

# Carga del archivo
df = pd.read_csv("slump_test.csv")

# Limpieza de datos
df.columns = df.columns.str.strip()
df = df.drop(columns="No")
df = df.drop_duplicates()
fc = "Compressive Strength (28-day)(Mpa)"

# Gráfico Slump vs. Flow
fig = px.scatter(
    df,
    x="SLUMP(cm)",
    y="FLOW(cm)",
    color=fc,
    color_continuous_scale="Viridis",
    trendline="ols",
    trendline_scope="overall",
    trendline_color_override="red",
    title="Slump vs. flow",
)

fig.update_traces(showlegend=False)

fig.update_layout(
    xaxis_title="Slump (cm)",
    yaxis_title="Flow (cm)",
    coloraxis_colorbar_title="Resistencia (MPa)",
    width=900,
    height=650,
    margin=dict(b=120),
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            x=0.0,
            y=-0.18,
            xanchor="left",
            yanchor="top",
            showactive=True,
            buttons=[
                dict(label="Mostrar recta", method="restyle", args=[{"visible": [True, True]}]),
                dict(label="Ocultar recta", method="restyle", args=[{"visible": [True, False]}]),
            ],
        )
    ],
)

fig.show()