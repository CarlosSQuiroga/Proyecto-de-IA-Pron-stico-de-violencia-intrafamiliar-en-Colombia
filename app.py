"""App de Streamlit: pronóstico mensual de casos denunciados de violencia
intrafamiliar por departamento en Colombia.

Estructura:
- prediccion.py  -> carga el modelo y calcula el pronóstico (misma lógica del notebook)
- app.py         -> interfaz: entradas en la barra lateral y tres pestañas
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import prediccion as pr

st.set_page_config(
    page_title="Pronóstico de violencia intrafamiliar",
    page_icon="📈",
    layout="wide",
)

AZUL = "#5B8DEF"      
AMBAR = "#F2B544"     
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
         "agosto", "septiembre", "octubre", "noviembre", "diciembre"]


def nombre_mes(fecha):
    return f"{MESES[fecha.month - 1]} {fecha.year}"


def formato(n):
    return f"{int(round(n)):,}".replace(",", ".")


def estilo(fig, alto=380):
    fig.update_layout(
        template="plotly_dark", height=alto,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.12, x=0),
    )
    return fig


# Carga (en caché para que la app responda rápido)
@st.cache_resource
def modelo():
    return pr.cargar_artefactos()


@st.cache_data
def historial():
    return pr.cargar_historial()


try:
    art = modelo()
    hist = historial()
except FileNotFoundError as e:
    st.error(f"No se encontró el archivo {e.filename}. "
             "Copia modelo_violencia.pkl e historial_mensual.csv en la misma carpeta que app.py.")
    st.stop()

ultimo_mes = hist["FECHA"].max()
departamentos = sorted(hist["DEPARTAMENTO"].unique())

# Barra lateral: entradas del usuario
with st.sidebar:
    st.header("Consulta")
    depto = st.selectbox(
        "Departamento", departamentos,
        index=departamentos.index("SANTANDER") if "SANTANDER" in departamentos else 0,
    )
    n_meses = st.slider("Meses a pronosticar", 1, 12, 6)

    serie = pr.serie_departamento(hist, depto)

    st.divider()
    ajustar = st.toggle("Simular otros casos recientes", value=False,
                        help="Cambia los casos de los últimos meses para ver cómo responde el pronóstico.")
    if ajustar:
        st.caption("Casos registrados en cada mes. Edítalos y el pronóstico se recalcula.")
        serie = serie.copy()
        meses_editables = [ultimo_mes - pd.DateOffset(months=k) for k in (2, 1, 0)]
        for fecha in meses_editables:
            serie.loc[fecha] = st.number_input(
                nombre_mes(fecha).capitalize(), min_value=0, step=1,
                value=int(serie.get(fecha, 0)), key=f"{depto}-{fecha:%Y%m}",
            )

    st.divider()
    st.caption(f"Datos de la Policía Nacional hasta {nombre_mes(ultimo_mes)}.")

# Ejecución del modelo
pron = pr.pronosticar(depto, serie, art, n_meses)
siguiente = pron.iloc[0]
ultimo_real = serie.loc[ultimo_mes]
cambio = (siguiente["PRONOSTICO"] - ultimo_real) / ultimo_real * 100 if ultimo_real else 0

st.title("¿Cuántos casos de violencia intrafamiliar podemos anticipar?")
st.write("Pronóstico mensual de casos denunciados por departamento, "
         "a partir del historial, el mes del año y la tendencia.")

tab_pron, tab_expl, tab_modelo = st.tabs(["Pronóstico", "Exploración", "Sobre el modelo"])

# Pestaña 1: Pronóstico
with tab_pron:
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Casos esperados en {nombre_mes(siguiente['FECHA'])}",
              formato(siguiente["PRONOSTICO"]), f"{cambio:+.1f}% frente al último mes")
    c2.metric(f"Total esperado en {n_meses} {'mes' if n_meses == 1 else 'meses'}",
              formato(pron["PRONOSTICO"].sum()))
    c3.metric("Error promedio del modelo", "± 45 casos",
              help="MAE en el conjunto de prueba (mayo 2019 – mayo 2021).")

    recientes = serie[serie.index > ultimo_mes - pd.DateOffset(months=36)]
    fig = go.Figure()
    fig.add_scatter(x=recientes.index, y=recientes.values, name="Casos reales",
                    mode="lines", line=dict(color=AZUL, width=2.5))
    # El pronóstico arranca desde el último dato real para que la línea sea continua
    fig.add_scatter(x=[ultimo_mes, *pron["FECHA"]], y=[ultimo_real, *pron["PRONOSTICO"]],
                    name="Pronóstico", mode="lines+markers",
                    line=dict(color=AMBAR, width=2.5, dash="dash"))
    fig.update_layout(title=f"{depto.title()}: últimos 36 meses y pronóstico",
                      yaxis_title="Casos por mes")
    st.plotly_chart(estilo(fig, 420), use_container_width=True)

    tabla = pron.assign(Mes=pron["FECHA"].map(nombre_mes),
                        **{"Casos esperados": pron["PRONOSTICO"].map(formato)})
    st.dataframe(tabla[["Mes", "Casos esperados"]], hide_index=True, use_container_width=True)
    if n_meses > 3:
        st.caption("Cada mes pronosticado se usa para calcular el siguiente, "
                   "así que la incertidumbre crece con el horizonte.")

# Pestaña 2: Exploración
with tab_expl:
    nacional = hist.groupby("FECHA", as_index=False)["CANTIDAD"].sum()
    fig = px.line(nacional, x="FECHA", y="CANTIDAD",
                  title="Casos denunciados en Colombia por mes",
                  labels={"FECHA": "", "CANTIDAD": "Casos"})
    fig.update_traces(line_color=AZUL)
    fig.add_vrect(x0="2020-03-25", x1="2020-08-31", fillcolor="gray", opacity=0.2,
                  line_width=0, annotation_text="Cuarentena", annotation_position="top left")
    st.plotly_chart(estilo(fig), use_container_width=True)

    col_a, col_b = st.columns(2)
    anios = sorted(hist["FECHA"].dt.year.unique())
    anio = col_a.selectbox("Año", anios, index=len(anios) - 2)
    ranking = (hist[hist["FECHA"].dt.year == anio]
               .groupby("DEPARTAMENTO", as_index=False)["CANTIDAD"].sum()
               .nlargest(10, "CANTIDAD").sort_values("CANTIDAD"))
    fig = px.bar(ranking, x="CANTIDAD", y="DEPARTAMENTO", orientation="h",
                 title=f"Departamentos con más casos en {anio}",
                 labels={"CANTIDAD": "Casos", "DEPARTAMENTO": ""})
    fig.update_traces(marker_color=AZUL)
    col_a.plotly_chart(estilo(fig, 420), use_container_width=True)

    por_mes = (serie.groupby(serie.index.month).mean()
               .rename(index=lambda m: MESES[m - 1][:3]))
    fig = px.bar(x=por_mes.index, y=por_mes.values,
                 title=f"{depto.title()}: promedio de casos según el mes del año",
                 labels={"x": "", "y": "Casos promedio"})
    fig.update_traces(marker_color=AMBAR)
    col_b.plotly_chart(estilo(fig, 420), use_container_width=True)

# Pestaña 3: Sobre el modelo
with tab_modelo:
    st.subheader("Modelo usado: regresión lineal")
    st.write(
        "Se compararon tres modelos de regresión con y sin PCA. La regresión lineal "
        "tuvo el menor error y fue la única que superó a la línea base de repetir el mes anterior."
    )
    # Resultados del notebook en el conjunto de prueba (mayo 2019 – mayo 2021)
    resultados = pd.DataFrame({
        "Modelo": ["Línea base (mes anterior)", "Linear Regression", "KNN (k=3)", "Random Forest",
                   "Linear Regression + PCA", "KNN + PCA", "Random Forest + PCA"],
        "MAE": [47.815, 44.559, 62.803, 54.299, 51.226, 64.820, 67.118],
        "RMSE": [131.298, 115.277, 147.211, 147.778, 127.312, 149.138, 189.463],
        "R²": [0.963, 0.971, 0.953, 0.953, 0.965, 0.952, 0.922],
    })
    st.dataframe(resultados, hide_index=True, use_container_width=True)

    st.subheader("Variables que usa")
    st.write(
        "Casos de los 3 meses anteriores, casos del mismo mes del año anterior, "
        "promedio de los últimos 3 meses, año, mes y departamento."
    )
    st.subheader("Limitaciones")
    st.write(
        "El modelo pronostica casos **denunciados**, no la violencia real: el subregistro es alto "
        "y en la cuarentena de 2020 las denuncias cayeron mientras las llamadas de auxilio aumentaron. "
        "Trabaja con totales por departamento y mes; no predice casos individuales ni identifica personas."
    )
