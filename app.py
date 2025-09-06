# ============================================
#  CLIMA_CHILE — APP STREAMLIT (con comentarios)
#  Objetivo: consultar, procesar y visualizar
#  datos climáticos diarios desde Open-Meteo.
# ============================================

# ---- Importaciones principales ----
# streamlit: framework para la app web
# pandas/numpy: manipulación numérica y tabular
# requests: consumo de la API HTTP
# altair/matplotlib: gráficos interactivos/estáticos
# datetime: manejo de fechas
import streamlit as st
import pandas as pd
import numpy as np
import requests
import altair as alt
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime

# ---- Configuración base de la app ----
# title/icon/layout: parámetros visuales generales
st.set_page_config(page_title="Clima_Chile", page_icon="⛅", layout="wide")


# =====================================================
# 1) UTILIDADES / CONSTANTES
#    - CITIES: coordenadas por ciudad (lat, lon)
#    - DEFAULT_START/END: rango por defecto (últimos ~12 meses)
#    - DAILY_VARS: variables a solicitar a la API
#    - VAR_LABELS: etiquetas legibles para la UI
# =====================================================
CITIES = {
    "Santiago": (-33.45, -70.66),
    "Valparaíso": (-33.05, -71.62),
    "Concepción": (-36.83, -73.05),
    "Antofagasta": (-23.65, -70.40),
    "Coyhaique": (-45.58, -72.07),
}

# Tomamos “ayer” como fin por defecto y restamos ~1 año
DEFAULT_END = date.today() - timedelta(days=1)
DEFAULT_START = DEFAULT_END - timedelta(days=365)

# Variables diarias que expone Open-Meteo (archivo histórico)
DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "windspeed_10m_max",
]

# Etiquetas amigables para mostrar en la interfaz
VAR_LABELS = {
    "temperature_2m_max": "T° Máx (°C)",
    "temperature_2m_min": "T° Mín (°C)",
    "precipitation_sum": "Precipitación (mm)",
    "windspeed_10m_max": "Viento máx (km/h)",
}


# =====================================================
# 2) FUNCIÓN: fetch_openmeteo_daily
#    - Llama al endpoint de archivo histórico
#    - Recibe lat/lon y fechas (date/datetime)
#    - Devuelve un DataFrame ordenado por fecha
#    - Se cachea para evitar llamadas repetidas iguales
# =====================================================
@st.cache_data(show_spinner=False)
def fetch_openmeteo_daily(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    """
    Descarga datos diarios de Open-Meteo (API de archivo / histórico).
    Retorna DataFrame con columnas:
    - date, temperature_2m_max, temperature_2m_min, precipitation_sum, windspeed_10m_max
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "daily": ",".join(DAILY_VARS),
        "timezone": "auto",        # deja que la API ajuste la zona horaria
        "windspeed_unit": "kmh",   # unifica la unidad del viento
    }

    # Petición HTTP con timeout; raise_for_status lanza excepción en códigos 4xx/5xx
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    # Si no vuelve el bloque "daily", devolvemos vacío
    if "daily" not in data:
        return pd.DataFrame()

    daily = data["daily"]

    # Construimos el DataFrame base con la columna de fechas
    df = pd.DataFrame({"date": pd.to_datetime(daily["time"])})

    # Para cada variable pedida, intentamos mapear su lista. Si no existe, rellenamos con NaN
    for k in DAILY_VARS:
        df[k] = daily.get(k, [np.nan] * len(df))

    # Ordenamos por fecha (por prolijidad) y reiniciamos índice
    return df.sort_values("date").reset_index(drop=True)


# =====================================================
# 3) FUNCIÓN: compute_kpis
#    - Calcula indicadores generales del rango consultado
#    - Maneja DataFrame vacío y valores faltantes
# =====================================================
def compute_kpis(df: pd.DataFrame) -> dict:
    # Si no hay datos, devolvemos un set de KPIs “vacío” coherente
    if df.empty:
        return {"días": 0, "T° media": np.nan, "Total precip": np.nan, "Día más caluroso": None}

    # Temperatura media diaria = (Tmax + Tmin) / 2; luego promedio global del periodo
    temp_mean = (df["temperature_2m_max"] + df["temperature_2m_min"]) / 2

    kpis = {
        "días": len(df),
        "T° media": round(temp_mean.mean(), 1),
        "Total precip": round(df["precipitation_sum"].sum(), 1),
    }

    # Fecha con la Tmax más alta; si no hubiera datos, devolvemos None
    if df["temperature_2m_max"].notna().any():
        hottest_idx = df["temperature_2m_max"].idxmax()
        hottest_day = df.loc[hottest_idx, "date"].date()
    else:
        hottest_day = None

    kpis["Día más caluroso"] = hottest_day
    return kpis


# =====================================================
# 4) FUNCIÓN: monthly_summary
#    - Agrega por mes: medias de Tmax/Tmin/Viento y suma de precipitación
#    - Devuelve un DataFrame por mes (columna 'month')
# =====================================================
def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    m = df.copy()
    # Convertimos la fecha a periodo mensual y lo llevamos a timestamp (inicio de mes)
    m["month"] = m["date"].dt.to_period("M").dt.to_timestamp()

    # Agregados por mes (nombres de columnas de salida a la izquierda)
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


# =====================================================
# 5) FUNCIÓN: download_button_csv
#    - Genera un botón de Streamlit para descargar un CSV
#    - No altera el DataFrame; solo serializa y expone
# =====================================================
def download_button_csv(df: pd.DataFrame, filename: str = "clima_chile.csv") -> None:
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar CSV", data=csv, file_name=filename, mime="text/csv")


# =====================================================
# 6) INTERFAZ (UI) — SIDEBAR Y PARÁMETROS
#    - Selección de ciudad, rango de fechas y variables a graficar
#    - Validaciones mínimas de rango
# =====================================================
st.title("⛅ Clima en Chile – Últimos 12 meses")
st.caption("Datos diarios desde Open-Meteo (histórico). Exporta, filtra y visualiza.")

with st.sidebar:
    st.header("Parámetros")

    # Ciudad a consultar -> usamos el diccionario CITIES para obtener lat/lon
    city = st.selectbox("Ciudad", list(CITIES.keys()), index=0)

    # Selector de rango de fechas (Streamlit retorna una tupla (start, end))
    dr = st.date_input(
        "Rango de fechas (máx. 1 año aprox.)",
        (DEFAULT_START, DEFAULT_END),
        min_value=date(1979, 1, 1),
        max_value=DEFAULT_END,
    )

    # Normalizamos el resultado del date_input
    if isinstance(dr, tuple):
        start_date, end_date = dr
    else:
        start_date, end_date = DEFAULT_START, DEFAULT_END

    # Regla básica: inicio no puede ser posterior al fin
    if start_date > end_date:
        st.error("El inicio debe ser anterior al fin.")

    # Permite elegir qué series aparecerán en el gráfico de líneas
    selected_vars = st.multiselect(
        "Variables a visualizar",
        options=[VAR_LABELS[v] for v in DAILY_VARS],
        default=[
            VAR_LABELS["temperature_2m_max"],
            VAR_LABELS["temperature_2m_min"],
            VAR_LABELS["precipitation_sum"],
        ],
    )

    # Invertimos el dict de etiquetas para traducir “etiqueta visible” -> “nombre de columna”
    inv_labels = {v: k for k, v in VAR_LABELS.items()}
    vars_keys = [inv_labels[v] for v in selected_vars] if selected_vars else DAILY_VARS


# =====================================================
# 7) DESCARGA DE DATOS
#    - Obtenemos lat/lon desde la ciudad elegida
#    - Llamamos a la función que consume la API
#    - Cortamos la ejecución si no hay datos
# =====================================================
lat, lon = CITIES[city]
with st.spinner(f"Descargando datos para {city} ({start_date} → {end_date})..."):
    df = fetch_openmeteo_daily(lat, lon, start_date, end_date)

if df.empty:
    st.warning("No se recibieron datos para el rango seleccionado.")
    st.stop()


# =====================================================
# 8) KPIs / MÉTRICAS
#    - Mostramos 4 indicadores principales del periodo
#    - Formateamos la fecha del día más caluroso con fallback
# =====================================================
k = compute_kpis(df)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Días", k["días"])
c2.metric("T° media (°C)", k["T° media"])
c3.metric("Precipitación total (mm)", k["Total precip"])

# Si no hay fecha (None), mostramos “-” para evitar errores de strftime
caloroso_date = k["Día más caluroso"].strftime("%Y-%m-%d") if k["Día más caluroso"] else "-"
c4.metric("Día más caluroso", caloroso_date)

st.divider()


# =====================================================
# 9) TABLA + DESCARGA
#    - Vista tabular de los registros diarios
#    - Botón para descargar el CSV con nombre dinámico
# =====================================================
st.subheader(f"Datos diarios – {city}")
st.dataframe(df, use_container_width=True, height=300)

# Construimos un nombre de archivo simple: ciudad_sin_espacios_inicio_fin.csv
csv_name = f"clima_{city.lower().replace(' ', '')}_{start_date}_{end_date}.csv"
download_button_csv(df, csv_name)

st.divider()


# =====================================================
# 10) VISUALIZACIONES
#     A) Gráfico de líneas (Altair) con escala Y independiente por serie
#     B) Histograma (Matplotlib) para explorar la distribución
# =====================================================
left, right = st.columns([2, 1])

with left:
    st.subheader("Evolución temporal")

    # Gráfico base: eje X temporal (fecha); luego apilamos capas por variable
    base = alt.Chart(df).encode(x=alt.X("date:T", title="Fecha"))
    layers = []

    # Paleta simple para cada variable (fallback '#999' si faltara la clave)
    palette = {
        "temperature_2m_max": "#e45756",
        "temperature_2m_min": "#4ca2ff",
        "precipitation_sum": "#6cc24a",
        "windspeed_10m_max": "#FFA500",
    }

    # Creamos una capa por variable seleccionada
    for v in vars_keys:
        layers.append(
            base.mark_line(point=False, color=palette.get(v, "#999")).encode(
                y=alt.Y(f"{v}:Q", title=VAR_LABELS[v]),
                tooltip=[
                    alt.Tooltip("date:T", title="Fecha"),
                    alt.Tooltip(f"{v}:Q", title=VAR_LABELS[v], format=".2f"),
                ],
            ).interactive()
        )

    # Unimos todas las capas y permitimos escalas Y independientes
    chart = alt.layer(*layers).resolve_scale(y="independent")
    st.altair_chart(chart, use_container_width=True)

with right:
    st.subheader("Histogramas")

    # Selector de columna numérica pero mostrando la etiqueta amigable
    num_col = st.selectbox(
        "Variable",
        df.select_dtypes(include="number").columns.map(lambda c: VAR_LABELS.get(c, c)),
    )

    # Mapeamos la etiqueta visible de vuelta al nombre real de columna
    label_to_key = {v: k for k, v in VAR_LABELS.items()}
    num_key = label_to_key.get(num_col, num_col)

    # Histograma con 20 bins; bordes blancos para claridad visual
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(df[num_key].dropna(), bins=20, edgecolor="white")
    ax.set_title(f"Histograma: {VAR_LABELS.get(num_key, num_key)}")
    ax.grid(alpha=0.2)
    st.pyplot(fig, use_container_width=True)

st.divider()


# =====================================================
# 11) RESUMEN MENSUAL
#     - Tabla agregada y gráfico combinado:
#       * Líneas: Tmax / Tmin
#       * Barras: Precipitación mensual
# =====================================================
st.subheader("Resumen mensual")
m = monthly_summary(df)
st.dataframe(m, use_container_width=True)

chart_m = alt.Chart(m).encode(x=alt.X("month:T", title="Mes"))
l1 = chart_m.mark_line(color="#e45756").encode(y=alt.Y("Tmax:Q", title="Tmax/Tmin"))
l2 = chart_m.mark_line(color="#4ca2ff").encode(y="Tmin:Q")
b1 = chart_m.mark_bar(color="#6cc24a").encode(y=alt.Y("PrecipTotal:Q", title="Precipitación (mm)"))

st.altair_chart(alt.layer(b1, l1, l2).resolve_scale(y="independent"), use_container_width=True)

# Pie de página con la fuente utilizada
st.caption("Fuente de datos: Open-Meteo Archive API")
