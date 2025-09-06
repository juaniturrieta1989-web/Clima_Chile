import streamlit as st
import pandas as pd
import numpy as np
import requests
import altair as alt
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime

# -----------------------------
# Configuración general de la app
# -----------------------------
st.set_page_config(page_title="Clima_Chile", page_icon="⛅", layout="wide")

# Encabezado y descripción general (texto para usuarios no técnicos)
st.title("⛅ Clima en Chile – Últimos 12 meses")
st.caption(
    "Aplicación educativa para explorar datos climáticos históricos diarios en ciudades de Chile. "
    "Selecciona una ciudad y un rango de fechas, revisa indicadores clave, visualiza tendencias y descarga los datos."
)

st.markdown(
    """
    Esta aplicación consulta la **Open-Meteo Archive API** para obtener variables diarias: 
    **temperatura máxima y mínima, precipitación acumulada** y **viento máximo a 10 m**. 
    La idea es ofrecer una vista clara de la **evolución temporal** y un **resumen mensual**, además de permitir la **descarga en CSV**.
    
    **¿Cómo usarla?**
    1) En la barra lateral, elige **Ciudad**, **Rango de fechas** y **Variables** a mostrar.
    2) Revisa los **indicadores principales** (panel de KPIs).
    3) Explora la **tabla**, los **gráficos** y el **resumen mensual**.
    4) Si quieres trabajar con los datos por tu cuenta, usa el botón **Descargar CSV**.
    """
)

# -----------------------------
# Utilidades (constantes y diccionarios)
# -----------------------------
CITIES = {
    "Santiago": (-33.45, -70.66),
    "Valparaíso": (-33.05, -71.62),
    "Concepción": (-36.83, -73.05),
    "Antofagasta": (-23.65, -70.40),
    "Coyhaique": (-45.58, -72.07),
}

DEFAULT_END = date.today() - timedelta(days=1)
DEFAULT_START = DEFAULT_END - timedelta(days=365)

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "windspeed_10m_max",
]

VAR_LABELS = {
    "temperature_2m_max": "T° Máx (°C)",
    "temperature_2m_min": "T° Mín (°C)",
    "precipitation_sum": "Precipitación (mm)",
    "windspeed_10m_max": "Viento máx (km/h)",
}

# Paleta de colores para las variables
PALETTE = {
    "temperature_2m_max": "#e45756",
    "temperature_2m_min": "#4ca2ff",
    "precipitation_sum": "#6cc24a",
    "windspeed_10m_max": "#FFA500",
}

# -----------------------------
# Funciones de datos
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_openmeteo_daily(lat, lon, start, end):
    """
    Descarga datos diarios de Open-Meteo (API de archivo / histórico).
    Retorna DataFrame con columnas: date, temperature_2m_max, temperature_2m_min,
    precipitation_sum, windspeed_10m_max.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "daily": ",".join(DAILY_VARS),
        "timezone": "auto",
        "windspeed_unit": "kmh",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    if "daily" not in data:
        return pd.DataFrame()
    daily = data["daily"]

    df = pd.DataFrame({"date": pd.to_datetime(daily["time"])})
    for k in DAILY_VARS:
        df[k] = daily.get(k, [np.nan] * len(df))
    return df.sort_values("date").reset_index(drop=True)


def compute_kpis(df):
    """Calcula y devuelve un diccionario con KPIs del periodo seleccionado."""
    if df.empty:
        return {
            "días": 0,
            "T° media": np.nan,
            "Total precip": np.nan,
            "Día más caluroso": None,
        }

    temp_mean = (df["temperature_2m_max"] + df["temperature_2m_min"]) / 2
    total_days = len(df)
    tmedia = round(float(temp_mean.mean()), 1) if not temp_mean.empty else np.nan
    total_pp = round(float(df["precipitation_sum"].sum()), 1)

    # Día más caluroso (maneja vacíos con seguridad)
    if df["temperature_2m_max"].notna().any():
        hottest_idx = df["temperature_2m_max"].idxmax()
        hottest_day = df.loc[hottest_idx, "date"].date()
    else:
        hottest_day = None

    return {
        "días": total_days,
        "T° media": tmedia,
        "Total precip": total_pp,
        "Día más caluroso": hottest_day,
    }


def monthly_summary(df):
    """Agrega por mes: promedios de Tmax/Tmin, viento y precipitación total."""
    if df.empty:
        return df
    m = df.copy()
    m["month"] = m["date"].dt.to_period("M").dt.to_timestamp()
    agg = (
        m.groupby("month")
        .agg(
            Tmax=("temperature_2m_max", "mean"),
            Tmin=("temperature_2m_min", "mean"),
            PrecipTotal=("precipitation_sum", "sum"),
            VientoMax=("windspeed_10m_max", "mean"),
        )
        .reset_index()
    )
    return agg


def download_button_csv(df, filename="clima_chile.csv"):
    """Crea un botón de descarga de CSV en Streamlit."""
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar CSV", data=csv, file_name=filename, mime="text/csv"
    )

# -----------------------------
# UI – Sidebar (filtros)
# -----------------------------
with st.sidebar:
    st.header("Parámetros")
    st.markdown(
        """
        **Personaliza la consulta:**
        - Elige la **ciudad** de interés.
        - Define el **rango de fechas** (recomendado ≤ 12 meses para rapidez).
        - Selecciona las **variables** a visualizar.
        
        _Mientras más amplio el rango, mayor será el volumen de datos descargados._
        """
    )

    city = st.selectbox("Ciudad", list(CITIES.keys()), index=0)
    dr = st.date_input(
        "Rango de fechas (máx. ~1 año)",
        (DEFAULT_START, DEFAULT_END),
        min_value=date(1979, 1, 1),
        max_value=DEFAULT_END,
    )
    if isinstance(dr, tuple):
        start_date, end_date = dr
    else:
        start_date, end_date = DEFAULT_START, DEFAULT_END

    if start_date > end_date:
        st.error("El inicio debe ser anterior al fin.")

    selected_vars = st.multiselect(
        "Variables a visualizar",
        options=[VAR_LABELS[v] for v in DAILY_VARS],
        default=[
            VAR_LABELS["temperature_2m_max"],
            VAR_LABELS["temperature_2m_min"],
            VAR_LABELS["precipitation_sum"],
        ],
        help="Elige qué series quieres ver en los gráficos de líneas.",
    )

    # Mapeo etiqueta -> clave de variable
    inv_labels = {v: k for k, v in VAR_LABELS.items()}
    vars_keys = [inv_labels[v] for v in selected_vars] if selected_vars else DAILY_VARS

# -----------------------------
# Descarga de datos
# -----------------------------
lat, lon = CITIES[city]
with st.spinner(f"Descargando datos para {city} ({start_date} → {end_date})…"):
    df = fetch_openmeteo_daily(lat, lon, start_date, end_date)

if df.empty:
    st.warning("No se recibieron datos para el rango seleccionado.")
    st.stop()

# -----------------------------
# Panel de KPIs (indicadores principales)
# -----------------------------
st.markdown(
    """
    ### 📊 Indicadores principales
    Estos valores resumen el comportamiento del clima en el periodo seleccionado:
    - **Días analizados**: cantidad total de registros en el rango.
    - **Temperatura media**: promedio entre las T° mínimas y máximas diarias.
    - **Precipitación total**: suma de las lluvias registradas.
    - **Día más caluroso**: fecha con la mayor temperatura máxima.
    """
)

k = compute_kpis(df)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Días", k["días"])  # cantidad de días
c2.metric("T° media (°C)", "-" if np.isnan(k["T° media"]) else k["T° media"])  
c3.metric("Precipitación total (mm)", "-" if np.isnan(k["Total precip"]) else k["Total precip"])  
caloroso_str = k["Día más caluroso"].strftime("%Y-%m-%d") if k["Día más caluroso"] else "-"
c4.metric("Día más caluroso", caloroso_str)

st.divider()

# -----------------------------
# Tabla de datos + descarga
# -----------------------------
st.subheader(f"📑 Datos diarios – {city}")
st.markdown(
    """
    Explora los **registros diarios** descargados desde la API. Puedes ordenar por columna y usar el buscador integrado.
    Si necesitas trabajar con los datos por fuera de esta app, usa el botón **Descargar CSV**.
    """
)

st.dataframe(df, use_container_width=True, height=320)
download_button_csv(
    df, f"clima_{city.lower().replace(' ', '')}_{start_date}_{end_date}.csv"
)

st.divider()

# -----------------------------
# Visualizaciones (Altair / Matplotlib)
# -----------------------------
left, right = st.columns([2, 1])

with left:
    st.subheader("📈 Evolución temporal")
    st.markdown(
        """
        Cada línea representa una variable climática. **Pasa el cursor** para ver los valores exactos.
        Esto ayuda a identificar **patrones estacionales** y **eventos extremos** (olas de calor, lluvias intensas, etc.).
        """
    )

    base = alt.Chart(df).encode(x=alt.X("date:T", title="Fecha"))
    layers = []
    for v in vars_keys:
        layers.append(
            base.mark_line(point=False, color=PALETTE.get(v, "#999")).encode(
                y=alt.Y(f"{v}:Q", title=VAR_LABELS[v]),
                tooltip=[
                    alt.Tooltip("date:T", title="Fecha"),
                    alt.Tooltip(f"{v}:Q", title=VAR_LABELS[v], format=".2f"),
                ],
            ).interactive()
        )
    chart = alt.layer(*layers).resolve_scale(y="independent")
    st.altair_chart(chart, use_container_width=True)

with right:
    st.subheader("📊 Distribución de valores")
    st.markdown(
        """
        El histograma muestra **con qué frecuencia** se observan los valores de una variable. 
        Sirve para detectar rangos típicos y outliers.
        """
    )

    num_col = st.selectbox(
        "Variable",
        df.select_dtypes(include="number").columns.map(lambda c: VAR_LABELS.get(c, c)),
    )
    # Revertir etiqueta a clave de columna
    label_to_key = {v: k for k, v in VAR_LABELS.items()}
    num_key = label_to_key.get(num_col, num_col)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(df[num_key].dropna(), bins=20, edgecolor="white")
    ax.set_title(f"Histograma: {VAR_LABELS.get(num_key, num_key)}")
    ax.grid(alpha=0.2)
    st.pyplot(fig, use_container_width=True)

st.divider()

# -----------------------------
# Resumen mensual
# -----------------------------
st.subheader("📅 Resumen mensual")
st.markdown(
    """
    - **Líneas**: promedios de **T° máxima** (roja) y **T° mínima** (azul).
    - **Barras**: **precipitación total mensual** (verde).
    
    Esta vista agrega los datos por mes para facilitar comparaciones y ver tendencias generales.
    """
)

m = monthly_summary(df)
st.dataframe(m, use_container_width=True)

chart_m = alt.Chart(m).encode(x=alt.X("month:T", title="Mes"))
l1 = chart_m.mark_line(color=PALETTE["temperature_2m_max"]).encode(y=alt.Y("Tmax:Q", title="Tmax/Tmin"))
l2 = chart_m.mark_line(color=PALETTE["temperature_2m_min"]).encode(y="Tmin:Q")
b1 = chart_m.mark_bar(color=PALETTE["precipitation_sum"]).encode(y=alt.Y("PrecipTotal:Q", title="Precipitación (mm)"))

st.altair_chart(alt.layer(b1, l1, l2).resolve_scale(y="independent"), use_container_width=True)

# Opción de descarga del resumen mensual
with st.expander("💾 Descargar resumen mensual"):
    st.markdown("Descarga el resumen mensual en CSV para usar en informes o clases.")
    download_button_csv(m, f"clima_resumen_mensual_{city.lower().replace(' ', '')}_{start_date}_{end_date}.csv")

st.divider()

# -----------------------------
# Metodología y fuente de datos (ayuda)
# -----------------------------
with st.expander("ℹ️ Metodología y fuente de datos"):
    st.markdown(
        """
        **Fuente:** [Open-Meteo Archive API](https://archive-api.open-meteo.com/v1/archive) (sin autenticación).
        
        **Variables:**
        - `temperature_2m_max` / `temperature_2m_min` (°C)
        - `precipitation_sum` (mm)
        - `windspeed_10m_max` (km/h)
        
        **Zona horaria:** la API ajusta automáticamente según la ubicación.
        
        **Tratamiento de datos:**
        - Se construye un DataFrame con índice temporal.
        - Resumen mensual: promedio de temperaturas, suma de precipitación y promedio de viento máximo.
        - Los valores faltantes se conservan como `NaN` para no distorsionar promedios.
        
        **Notas:**
        - Rango recomendado: ≤ 12 meses para tiempos de respuesta ágiles.
        - Si el rango es muy grande o remoto, podrían no existir datos para algunos días.
        """
    )

st.caption("Fuente de datos: Open-Meteo Archive API • App educativa desarrollada en Python (VS Code) con Streamlit, Pandas, Altair y Matplotlib.")
