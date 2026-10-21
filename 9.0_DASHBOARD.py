import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, html, dcc, dash_table, Input, Output
from dash.exceptions import PreventUpdate
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------- Datos ----------
df = pd.read_csv("slump_test.csv")
df.columns = df.columns.str.strip()
df = df.drop(columns="No").drop_duplicates().reset_index(drop=True)

fc = "Compressive Strength (28-day)(Mpa)"
slump, flow = "SLUMP(cm)", "FLOW(cm)"
FEATS = ["Cement", "Slag", "Fly ash", "Water", "SP", "Coarse Aggr.", "Fine Aggr."]
X, y = df[FEATS], df[fc]

# ---------- Modelos ----------
modelos = {
    "Regresión lineal": make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

filas = []
for nombre, m in modelos.items():
    cv = cross_validate(m, X, y, cv=kf,
                        scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"])
    filas.append({
        "Modelo": nombre,
        "R2": cv["test_r2"].mean(),
        "MAE (MPa)": -cv["test_neg_mean_absolute_error"].mean(),
        "RMSE (MPa)": -cv["test_neg_root_mean_squared_error"].mean(),
    })
tabla = pd.DataFrame(filas).round(3)
mejor = tabla.loc[tabla["R2"].idxmax(), "Modelo"]

pred = pd.Series(cross_val_predict(modelos[mejor], X, y, cv=kf), index=df.index)
modelo_final = modelos[mejor].fit(X, y)

coef = pd.DataFrame({
    "Componente": FEATS,
    "Coeficiente": modelos["Regresión lineal"].fit(X, y)[-1].coef_,
}).sort_values("Coeficiente")
fig_coef = px.bar(coef, x="Coeficiente", y="Componente", orientation="h",
                  color="Coeficiente", color_continuous_scale="RdBu",
                  title="Influencia de cada componente (regresión lineal estandarizada)")
fig_coef.update_layout(coloraxis_showscale=False)

# ---------- Utilidades ----------
def kpi(titulo, valor):
    return html.Div([html.Span(titulo), html.H3(valor)], className="kpi")


def fig_hist(d):
    media, mediana, desv = d[fc].mean(), d[fc].median(), d[fc].std()
    cont, bordes = np.histogram(d[fc], bins=12)
    moda = (bordes[cont.argmax()] + bordes[cont.argmax() + 1]) / 2
    f = px.histogram(d, x=fc, nbins=12, title="Distribución de la resistencia a 28 días")
    f.add_vline(x=media, line_dash="dash", line_color="red", annotation_text=f"Media: {media:.2f}")
    f.add_vline(x=mediana, line_dash="dashdot", line_color="blue",
                annotation_text=f"Mediana: {mediana:.2f}", annotation_position="bottom right")
    f.add_vline(x=moda, line_dash="dot", line_color="green",
                annotation_text=f"Moda: {moda:.2f}", annotation_position="top left")
    f.add_annotation(text=f"Desv. estándar: {desv:.2f} MPa", xref="paper", yref="paper",
                     x=0.98, y=0.98, showarrow=False, bgcolor="white")
    f.update_layout(xaxis_title="Resistencia (MPa)", yaxis_title="Cantidad de mezclas")
    return f


def fig_cemento(d):
    f = px.scatter(d, x="Cement", y=fc, trendline="ols", title="Cemento vs. resistencia")
    f.update_layout(xaxis_title="Cemento (kg/m3)", yaxis_title="Resistencia (MPa)")
    return f


def fig_slump_flow(d):
    f = px.scatter(d, x=slump, y=flow, color=fc, color_continuous_scale="Viridis",
                   trendline="ols", trendline_scope="overall",
                   trendline_color_override="red", title="Slump vs. flow")
    f.update_traces(showlegend=False)
    f.update_layout(xaxis_title="Slump (cm)", yaxis_title="Flow (cm)",
                    coloraxis_colorbar_title="MPa")
    return f


def fig_corr(d):
    f = px.imshow(d.corr(), text_auto=".2f", color_continuous_scale="RdBu_r",
                  zmin=-1, zmax=1, title="Matriz de correlación")
    f.update_xaxes(tickangle=-45)
    return f


def fig_3d(d):
    f = px.scatter_3d(d, x="Cement", y="Water", z=fc, color=fc,
                      color_continuous_scale="Viridis",
                      title="Cemento, agua y resistencia")
    f.update_traces(marker=dict(size=4))
    f.update_layout(coloraxis_colorbar_title="MPa")
    return f


def fig_real_pred(d):
    f = px.scatter(x=d[fc], y=pred.loc[d.index],
                   title=f"Resistencia real vs. predicha ({mejor}, validación cruzada)",
                   labels={"x": "Real (MPa)", "y": "Predicha (MPa)"})
    f.add_shape(type="line", x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(),
                line=dict(color="red", dash="dash"))
    return f


# ---------- Layout ----------
DEF = dict(zip(FEATS, [250, 100, 100, 190, 8, 900, 700]))
lo, hi = float(np.floor(y.min())), float(np.ceil(y.max()))

app = Dash(__name__)
app.title = "Dashboard de concreto"
app.index_string = """<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
body{font-family:system-ui,sans-serif;background:#f4f6f8;color:#1f2933;margin:0;padding:24px}
h1{margin:0 0 16px}.card{background:#fff;border:1px solid #e3e8ee;border-radius:10px;padding:14px;margin-bottom:16px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:16px}
.kpi{background:#fff;border:1px solid #e3e8ee;border-radius:10px;padding:12px}
.kpi span{font-size:12px;color:#6b7785}.kpi h3{margin:4px 0 0;font-size:22px;color:#2f6fed}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:16px}
.form{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.form label{display:flex;flex-direction:column;font-size:13px;gap:4px}
#pred{font-size:28px;font-weight:700;color:#2f6fed;margin-top:12px}
</style></head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>"""

app.layout = html.Div([
    html.H1("Dashboard de resistencia del concreto"),
    html.Div([
        html.Label("Filtrar por resistencia a 28 días (MPa)"),
        dcc.RangeSlider(id="rango", min=lo, max=hi, step=0.5, value=[lo, hi],
                        tooltip={"placement": "bottom"}),
    ], className="card"),
    html.Div(id="kpis", className="kpis"),
    html.Div([dcc.Graph(id=i) for i in
              ["g-hist", "g-cem", "g-sf", "g-corr", "g-3d", "g-real"]] +
             [dcc.Graph(figure=fig_coef)], className="grid"),
    html.Div([
        html.H3("Comparación de modelos (validación cruzada, 5 folds)"),
        dash_table.DataTable(tabla.to_dict("records"),
                             [{"name": c, "id": c} for c in tabla.columns],
                             style_cell={"textAlign": "left", "padding": "6px"}),
    ], className="card", style={"marginTop": "16px"}),
    html.Div([
        html.H3(f"Predicción de resistencia ({mejor})"),
        html.Div([html.Label([f, dcc.Input(id=f"in-{i}", type="number", value=DEF[f], debounce=True)])
                  for i, f in enumerate(FEATS)], className="form"),
        html.Div(id="pred"),
    ], className="card"),
])


# ---------- Callbacks ----------
@app.callback(
    Output("kpis", "children"), Output("g-hist", "figure"), Output("g-cem", "figure"),
    Output("g-sf", "figure"), Output("g-corr", "figure"), Output("g-3d", "figure"),
    Output("g-real", "figure"),
    Input("rango", "value"),
)
def actualizar(rango):
    d = df[df[fc].between(rango[0], rango[1])]
    if len(d) < 3:
        raise PreventUpdate
    kpis = [
        kpi("Registros", f"{len(d)}"),
        kpi("Resistencia promedio", f"{d[fc].mean():.2f} MPa"),
        kpi("Resistencia máxima", f"{d[fc].max():.2f} MPa"),
        kpi("Resistencia mínima", f"{d[fc].min():.2f} MPa"),
        kpi("Slump promedio", f"{d[slump].mean():.2f} cm"),
        kpi("Flow promedio", f"{d[flow].mean():.2f} cm"),
        kpi("Cemento promedio", f"{d['Cement'].mean():.1f} kg/m3"),
        kpi("Agua promedio", f"{d['Water'].mean():.1f} kg/m3"),
        kpi("Relación agua/cemento", f"{(d['Water'] / d['Cement']).mean():.2f}"),
    ]
    return (kpis, fig_hist(d), fig_cemento(d), fig_slump_flow(d),
            fig_corr(d), fig_3d(d), fig_real_pred(d))


@app.callback(Output("pred", "children"), [Input(f"in-{i}", "value") for i in range(len(FEATS))])
def predecir(*valores):
    if None in valores:
        return "Completa todos los campos"
    p = modelo_final.predict(pd.DataFrame([valores], columns=FEATS))[0]
    return f"{p:.2f} MPa"


if __name__ == "__main__":
    app.run(debug=True)