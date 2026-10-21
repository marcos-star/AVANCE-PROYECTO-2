import json
import webbrowser
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_validate, cross_val_predict

PLANTILLA = r"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dashboard de concreto</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body{font-family:system-ui,sans-serif;background:#f4f6f8;color:#1f2933;margin:0;padding:24px}
h1{margin:0 0 16px}
.card{background:#fff;border:1px solid #e3e8ee;border-radius:10px;padding:14px;margin-bottom:16px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:16px}
.kpi{background:#fff;border:1px solid #e3e8ee;border-radius:10px;padding:12px}
.kpi span{font-size:12px;color:#6b7785}.kpi h3{margin:4px 0 0;font-size:22px;color:#2f6fed}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:16px}
.grid .card{margin:0;padding:6px;height:460px;box-sizing:border-box;overflow:hidden;min-width:0}
.grid .card>div{width:100%;height:100%}
.form{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.form label{display:flex;flex-direction:column;font-size:13px;gap:4px}
.form input{padding:6px;border:1px solid #cbd2d9;border-radius:6px}
#pred{font-size:28px;font-weight:700;color:#2f6fed;margin-top:12px}
table{border-collapse:collapse}th,td{text-align:left;padding:6px 14px;border-bottom:1px solid #e3e8ee}
.rango{position:relative;height:28px;margin:8px 8px 0}
.rango .pista{position:absolute;top:12px;left:0;right:0;height:4px;background:#d5dbe3;border-radius:2px}
.rango .activa{position:absolute;top:12px;height:4px;background:#2f6fed;border-radius:2px}
.rango input{position:absolute;left:0;width:100%;top:4px;margin:0;background:none;pointer-events:none;-webkit-appearance:none;appearance:none}
.rango input::-webkit-slider-thumb{pointer-events:auto;-webkit-appearance:none;width:18px;height:18px;border-radius:50%;background:#2f6fed;border:2px solid #fff;box-shadow:0 0 0 1px #2f6fed;cursor:pointer}
.rango input::-moz-range-thumb{pointer-events:auto;width:14px;height:14px;border-radius:50%;background:#2f6fed;border:2px solid #fff;cursor:pointer}
.rango input::-webkit-slider-runnable-track{background:none}
.rango input::-moz-range-track{background:none}
#valores{font-size:14px;margin-top:6px}
#aviso{color:#b42318;display:none;margin-top:6px}
</style></head><body>
<h1>Dashboard de resistencia del concreto</h1>

<div class="card">
  <label>Filtrar por resistencia a 28 días (MPa)</label>
  <div class="rango">
    <div class="pista"></div><div class="activa" id="activa"></div>
    <input type="range" id="lo"><input type="range" id="hi">
  </div>
  <div id="valores"></div>
  <div id="aviso">Hay menos de 3 mezclas en este rango. Amplía el filtro.</div>
</div>

<div class="kpis" id="kpis"></div>
<div class="grid">
  <div class="card"><div id="g-hist"></div></div>
  <div class="card"><div id="g-cem"></div></div>
  <div class="card"><div id="g-sf"></div></div>
  <div class="card"><div id="g-corr"></div></div>
  <div class="card"><div id="g-3d"></div></div>
  <div class="card"><div id="g-real"></div></div>
</div>

<div class="card" style="margin-top:16px">
  <h3>Comparación de modelos (validación cruzada, 5 folds)</h3>
  <table id="tabla"></table>
</div>

<div class="card">
  <h3>Predicción de resistencia (regresión lineal)</h3>
  <div class="form" id="form"></div>
  <div id="pred"></div>
</div>

<script>
const D = __DATA__;
const FC = D.fc, F = D.filas;
const FEATS = Object.keys(D.lin.coef);
const COLS = [...FEATS, "SLUMP(cm)", "FLOW(cm)", FC];
const NOMBRES = {[FC]: "Resistencia 28d", "SLUMP(cm)": "Slump", "FLOW(cm)": "Flow"};
const cfg = {responsive: true, displaylogo: false};
function fallo(id, e) {
  console.error(id, e);
  document.getElementById(id).innerHTML = `<p style="padding:16px;color:#b42318">No se pudo dibujar este gráfico: ${e && e.message ? e.message : e}</p>`;
}
function react(id, ...args) {
  try { const p = Plotly.react(id, ...args); if (p && p.catch) p.catch(e => fallo(id, e)); }
  catch (e) { fallo(id, e); }
}
const col = (d, k) => d.map(r => r[k]);
const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
const std = a => { const m = mean(a); return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - 1)); };
const median = a => { const s = [...a].sort((x, y) => x - y), n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };
const ols = (x, y) => { const mx = mean(x), my = mean(y); let n = 0, d = 0;
  x.forEach((v, i) => { n += (v - mx) * (y[i] - my); d += (v - mx) ** 2; }); const m = n / d; return [m, my - m * mx]; };
const pearson = (x, y) => { const mx = mean(x), my = mean(y); let n = 0, a = 0, b = 0;
  x.forEach((v, i) => { n += (v - mx) * (y[i] - my); a += (v - mx) ** 2; b += (y[i] - my) ** 2; }); const den = Math.sqrt(a * b); return den > 0 ? n / den : null; };
function moda(a, bins = 12) {
  const mn = Math.min(...a), mx = Math.max(...a), w = (mx - mn) / bins || 1, c = Array(bins).fill(0);
  a.forEach(v => c[Math.min(bins - 1, Math.floor((v - mn) / w))]++);
  return mn + (c.indexOf(Math.max(...c)) + 0.5) * w;
}

const ys = col(F, FC);
const lo = Math.floor(Math.min(...ys)), hi = Math.ceil(Math.max(...ys));
const $lo = document.getElementById("lo"), $hi = document.getElementById("hi");
[$lo, $hi].forEach(e => { e.min = lo; e.max = hi; e.step = 0.5; });
$lo.value = lo; $hi.value = hi;

function actualizar() {
  let a = +$lo.value, b = +$hi.value;
  if (a > b) { [a, b] = [b, a]; }
  const pct = v => (v - lo) / (hi - lo) * 100;
  const act = document.getElementById("activa");
  act.style.left = pct(a) + "%"; act.style.width = (pct(b) - pct(a)) + "%";
  document.getElementById("valores").textContent = `Rango: ${a.toFixed(1)} a ${b.toFixed(1)} MPa`;
  const d = F.filter(r => r[FC] >= a && r[FC] <= b);
  const aviso = document.getElementById("aviso");
  if (d.length < 3) { aviso.style.display = "block"; return; }
  aviso.style.display = "none";

  const r = col(d, FC), cem = col(d, "Cement"), agua = col(d, "Water");
  const kpi = (t, v) => `<div class="kpi"><span>${t}</span><h3>${v}</h3></div>`;
  document.getElementById("kpis").innerHTML = [
    kpi("Registros", d.length),
    kpi("Resistencia promedio", mean(r).toFixed(2) + " MPa"),
    kpi("Resistencia máxima", Math.max(...r).toFixed(2) + " MPa"),
    kpi("Resistencia mínima", Math.min(...r).toFixed(2) + " MPa"),
    kpi("Slump promedio", mean(col(d, "SLUMP(cm)")).toFixed(2) + " cm"),
    kpi("Flow promedio", mean(col(d, "FLOW(cm)")).toFixed(2) + " cm"),
    kpi("Cemento promedio", mean(cem).toFixed(1) + " kg/m3"),
    kpi("Agua promedio", mean(agua).toFixed(1) + " kg/m3"),
    kpi("Relación agua/cemento", mean(d.map(x => x.Water / x.Cement)).toFixed(2)),
  ].join("");

  // Histograma
  const me = mean(r), md = median(r), mo = moda(r), sd = std(r);
  const linea = (x, color, dash) => ({type: "line", xref: "x", yref: "paper", x0: x, x1: x, y0: 0, y1: 1, line: {color, dash}});
  const ann = (x, t, color, y) => ({x, y, xref: "x", yref: "paper", text: t, showarrow: false, font: {color}, xanchor: "left"});
  react("g-hist", [{x: r, type: "histogram", nbinsx: 12, marker: {color: "#4c78a8"}}], {
    title: "Distribución de la resistencia a 28 días",
    xaxis: {title: "Resistencia (MPa)"}, yaxis: {title: "Cantidad de mezclas"},
    shapes: [linea(me, "red", "dash"), linea(md, "blue", "dashdot"), linea(mo, "green", "dot")],
    annotations: [ann(me, `Media: ${me.toFixed(2)}`, "red", 1), ann(md, `Mediana: ${md.toFixed(2)}`, "blue", 0.92),
      ann(mo, `Moda: ${mo.toFixed(2)}`, "green", 0.84),
      {xref: "paper", yref: "paper", x: 0.98, y: 0.7, xanchor: "right", showarrow: false, bgcolor: "white",
       text: `Desv. estándar: ${sd.toFixed(2)} MPa`}],
    margin: {t: 50}
  }, cfg);

  // Cemento vs resistencia
  const [m, c0] = ols(cem, r), x0 = Math.min(...cem), x1 = Math.max(...cem);
  react("g-cem", [
    {x: cem, y: r, mode: "markers", type: "scatter", name: "Mezclas", marker: {color: "#4c78a8"}},
    {x: [x0, x1], y: [m * x0 + c0, m * x1 + c0], mode: "lines", name: "Tendencia (OLS)", line: {color: "red"}}],
    {title: "Cemento vs. resistencia", xaxis: {title: "Cemento (kg/m3)"}, yaxis: {title: "Resistencia (MPa)"},
     showlegend: false, margin: {t: 50}}, cfg);

  // Slump vs flow
  const sl = col(d, "SLUMP(cm)"), fl = col(d, "FLOW(cm)"), [m2, c2] = ols(sl, fl);
  const s0 = Math.min(...sl), s1 = Math.max(...sl);
  react("g-sf", [
    {x: sl, y: fl, mode: "markers", type: "scatter",
     marker: {color: r, colorscale: "Viridis", showscale: true, colorbar: {title: "MPa"}}},
    {x: [s0, s1], y: [m2 * s0 + c2, m2 * s1 + c2], mode: "lines", line: {color: "red"}}],
    {title: "Slump vs. flow", xaxis: {title: "Slump (cm)"}, yaxis: {title: "Flow (cm)"}, showlegend: false, margin: {t: 50}}, cfg);

  // Correlación
  const z = COLS.map(a => COLS.map(b => (v => v === null ? null : +v.toFixed(2))(pearson(col(d, a), col(d, b)))));
  const lab = COLS.map(c => NOMBRES[c] || c);
  react("g-corr", [{z, x: lab, y: lab, type: "heatmap", colorscale: "RdBu", reversescale: true, zmin: -1, zmax: 1,
    texttemplate: "%{z:.2f}"}],
    {title: "Matriz de correlación", xaxis: {tickangle: -45}, yaxis: {autorange: "reversed"}, margin: {t: 50, b: 110, l: 110}}, cfg);

  // 3D
  react("g-3d", [{x: cem, y: agua, z: r, mode: "markers", type: "scatter3d",
    marker: {size: 4, color: r, colorscale: "Viridis", colorbar: {title: "MPa"}}}],
    {title: "Cemento, agua y resistencia",
     scene: {xaxis: {title: "Cement"}, yaxis: {title: "Water"}, zaxis: {title: "Resistencia"}}, margin: {t: 50}}, cfg);

  // Real vs predicha (diagonal con el rango completo de los datos)
  const gmin = Math.min(...ys), gmax = Math.max(...ys);
  react("g-real", [
    {x: r, y: col(d, "pred"), mode: "markers", type: "scatter", marker: {color: "#4c78a8"}},
    {x: [gmin, gmax], y: [gmin, gmax], mode: "lines", line: {color: "red", dash: "dash"}}],
    {title: `Resistencia real vs. predicha (${D.mejor}, validación cruzada)`,
     xaxis: {title: "Real (MPa)"}, yaxis: {title: "Predicha (MPa)"}, showlegend: false, margin: {t: 50}}, cfg);
}
let pendiente = false;
function programar() {
  if (pendiente) return;
  pendiente = true;
  requestAnimationFrame(() => { pendiente = false; try { actualizar(); } catch (e) { console.error(e); } });
}
$lo.addEventListener("input", programar);
$hi.addEventListener("input", programar);

// Tabla de modelos
document.getElementById("tabla").innerHTML =
  "<tr><th>Modelo</th><th>R2</th><th>MAE (MPa)</th><th>RMSE (MPa)</th></tr>" +
  D.tabla.map(t => `<tr><td>${t.modelo}</td><td>${t.r2.toFixed(3)}</td><td>${t.mae.toFixed(3)}</td><td>${t.rmse.toFixed(3)}</td></tr>`).join("");

// Simulador
const form = document.getElementById("form");
FEATS.forEach((f, i) => {
  const [mn, mx, mu] = D.lin.rango[f];
  form.insertAdjacentHTML("beforeend",
    `<label>${f}<input type="number" id="in-${i}" value="${mu.toFixed(1)}" min="${mn}" max="${mx}" step="any"></label>`);
});
function predecir() {
  let p = D.lin.intercepto;
  for (let i = 0; i < FEATS.length; i++) {
    const v = parseFloat(document.getElementById("in-" + i).value);
    if (isNaN(v)) { document.getElementById("pred").textContent = "Completa todos los campos"; return; }
    p += D.lin.coef[FEATS[i]] * v;
  }
  document.getElementById("pred").textContent = p.toFixed(2) + " MPa";
}
form.addEventListener("input", predecir);

actualizar(); predecir();
</script></body></html>
"""

# Datos
carpeta = Path(__file__).parent
df = pd.read_csv(carpeta / "slump_test.csv")
df.columns = df.columns.str.strip()
df = df.drop(columns="No").drop_duplicates().reset_index(drop=True)
fc = "Compressive Strength (28-day)(Mpa)"
feats = ["Cement", "Slag", "Fly ash", "Water", "SP", "Coarse Aggr.", "Fine Aggr."]
X, y = df[feats], df[fc]

# Modelos (validación cruzada)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
modelos = {
    "Regresión lineal": make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
}
tabla = []
for n, mod in modelos.items():
    cv = cross_validate(mod, X, y, cv=kf,
                        scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"])
    tabla.append({"modelo": n, "r2": cv["test_r2"].mean(),
                  "mae": -cv["test_neg_mean_absolute_error"].mean(),
                  "rmse": -cv["test_neg_root_mean_squared_error"].mean()})
mejor = max(tabla, key=lambda t: t["r2"])["modelo"]
pred = cross_val_predict(modelos[mejor], X, y, cv=kf)

# Modelo lineal para el simulador
lin = LinearRegression().fit(X, y)

# Una fila por mezcla (con su predicción): el filtro del slider se aplica en el navegador
filas = df.copy()
filas["pred"] = np.round(pred, 2)

datos = {
    "fc": fc,
    "filas": filas.to_dict("records"),
    "tabla": tabla,
    "mejor": mejor,
    "lin": {"intercepto": float(lin.intercept_),
            "coef": dict(zip(feats, map(float, lin.coef_))),
            "rango": {f: [float(X[f].min()), float(X[f].max()), float(X[f].mean())] for f in feats}},
}

html = PLANTILLA.replace("__DATA__", json.dumps(datos, ensure_ascii=False))
salida = carpeta / "web.html"
salida.write_text(html, encoding="utf-8")
print(f"web.html generado en: {salida}")
webbrowser.open(salida.as_uri())