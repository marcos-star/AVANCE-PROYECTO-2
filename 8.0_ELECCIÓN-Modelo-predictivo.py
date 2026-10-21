import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_validate, cross_val_predict

# Carga del archivo
df = pd.read_csv("slump_test.csv")

# Limpieza de datos
df.columns = df.columns.str.strip()
df = df.drop(columns="No")
df = df.drop_duplicates()

# Variables predictoras y objetivo
fc = "Compressive Strength (28-day)(Mpa)"
X = df[["Cement", "Slag", "Fly ash", "Water", "SP", "Coarse Aggr.", "Fine Aggr."]]
y = df[fc]

# Modelos a comparar
modelos = {
    "Regresión lineal": make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
}

# Validación cruzada
kf = KFold(n_splits=5, shuffle=True, random_state=42)
resultados = []
for nombre, modelo in modelos.items():
    cv = cross_validate(
        modelo, X, y, cv=kf,
        scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"],
    )
    resultados.append({
        "Modelo": nombre,
        "R2": cv["test_r2"].mean(),
        "MAE (MPa)": -cv["test_neg_mean_absolute_error"].mean(),
        "RMSE (MPa)": -cv["test_neg_root_mean_squared_error"].mean(),
    })
resultados = pd.DataFrame(resultados).round(3)
print(resultados.to_string(index=False))

# Predicciones del mejor modelo con validación cruzada
mejor = resultados.loc[resultados["R2"].idxmax(), "Modelo"]
pred = cross_val_predict(modelos[mejor], X, y, cv=kf)

# Gráfico: real vs. predicho
fig1 = px.scatter(
    x=y,
    y=pred,
    title=f"Resistencia real vs. predicha ({mejor})",
    labels={"x": "Resistencia real (MPa)", "y": "Resistencia predicha (MPa)"},
)
fig1.add_shape(
    type="line",
    x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(),
    line=dict(color="red", dash="dash"),
)
fig1.update_layout(width=800, height=600)
fig1.show()

# Gráfico: coeficientes del modelo lineal estandarizado
lineal = modelos["Regresión lineal"].fit(X, y)
coef = pd.DataFrame({
    "Componente": X.columns,
    "Coeficiente": lineal[-1].coef_,
}).sort_values("Coeficiente")

fig2 = px.bar(
    coef,
    x="Coeficiente",
    y="Componente",
    orientation="h",
    color="Coeficiente",
    color_continuous_scale="RdBu",
    title="Influencia de cada componente en la resistencia",
)
fig2.update_layout(width=800, height=500, coloraxis_showscale=False)
fig2.show()

# Predicción para una mezcla nueva
mezcla = pd.DataFrame({
    "Cement": [250], "Slag": [100], "Fly ash": [100], "Water": [190],
    "SP": [8], "Coarse Aggr.": [900], "Fine Aggr.": [700],
})
print(f"\nResistencia predicha: {lineal.predict(mezcla)[0]:.2f} MPa")