"""Lógica del modelo: carga de archivos y pronóstico.

Está separada de la interfaz (app.py) para que el cálculo de las
variables sea idéntico al del notebook y se pueda probar por aparte.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
RUTA_MODELO = BASE / "modelo_violencia.pkl"
RUTA_HISTORIAL = BASE / "historial_mensual.csv"


def cargar_artefactos(ruta=RUTA_MODELO):
    """Modelo, scaler, columnas y variables numéricas guardados en el notebook."""
    return joblib.load(ruta)


def cargar_historial(ruta=RUTA_HISTORIAL):
    """Casos reales por departamento y mes."""
    h = pd.read_csv(ruta, parse_dates=["FECHA"])
    return h[["DEPARTAMENTO", "FECHA", "CANTIDAD"]].sort_values(["DEPARTAMENTO", "FECHA"])


def serie_departamento(historial, depto):
    """Serie mensual de un departamento, indexada por fecha."""
    d = historial[historial["DEPARTAMENTO"] == depto]
    return d.set_index("FECHA")["CANTIDAD"].astype(float)


def construir_fila(depto, fecha, serie, art):
    """Arma las variables de un mes exactamente como en el notebook:
    AÑO, rezagos 1-3 y 12, media móvil de 3 meses y one-hot de departamento y mes."""
    def rezago(k):
        return float(serie.get(fecha - pd.DateOffset(months=k), 0.0))  # mes sin registro = 0 casos

    fila = {
        "AÑO": fecha.year,
        "LAG_1": rezago(1),
        "LAG_2": rezago(2),
        "LAG_3": rezago(3),
        "LAG_12": rezago(12),
        "MEDIA_3M": np.mean([rezago(1), rezago(2), rezago(3)]),
        "DEPARTAMENTO": depto,
        "MES": fecha.month,
    }
    X = pd.get_dummies(pd.DataFrame([fila]), columns=["DEPARTAMENTO", "MES"], dtype=int)
    # Mismas columnas y mismo orden que en el entrenamiento; las categorías de
    # referencia (eliminadas con drop_first) quedan representadas por ceros.
    X = X.reindex(columns=art["columnas"], fill_value=0)
    X[art["features_num"]] = art["scaler"].transform(X[art["features_num"]])
    return X


def pronosticar(depto, serie, art, n_meses=1):
    """Pronóstico recursivo: cada mes pronosticado se usa como rezago del siguiente."""
    serie = serie.copy()
    ultimo = serie.index.max()
    filas = []
    for k in range(1, n_meses + 1):
        fecha = ultimo + pd.DateOffset(months=k)
        pred = float(art["modelo"].predict(construir_fila(depto, fecha, serie, art))[0])
        pred = max(pred, 0.0)  # no existen casos negativos
        serie.loc[fecha] = pred
        filas.append({"FECHA": fecha, "PRONOSTICO": round(pred)})
    return pd.DataFrame(filas)
