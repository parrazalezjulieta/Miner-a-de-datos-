from math import sqrt
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parent
ARCHIVO_ENTRADA = BASE_DIR / "datos" / "Recipe_Reviews_Clean.csv"
CARPETA_RESULTADOS = BASE_DIR / "datos" / "resultados_practica_8"
FORMATO_FECHA = "%d/%m/%Y %H:%M"
DURACION_PRONOSTICO = 30


def cargar_promedios_diarios() -> tuple[pd.DataFrame, int, int, int]:
    """Carga reseñas válidas y calcula promedios solo para días observados."""
    if not ARCHIVO_ENTRADA.is_file():
        raise FileNotFoundError(f"No se encontro el archivo de entrada: {ARCHIVO_ENTRADA}")

    datos = pd.read_csv(ARCHIVO_ENTRADA, usecols=["created_at", "stars"])
    fechas = pd.to_datetime(datos["created_at"], format=FORMATO_FECHA, errors="coerce")
    estrellas = pd.to_numeric(datos["stars"], errors="coerce")
    fechas_invalidas = int(fechas.isna().sum())
    calificaciones_invalidas = int((~estrellas.between(1, 5)).sum())

    reseñas_validas = fechas.notna() & estrellas.between(1, 5)
    datos_validos = pd.DataFrame(
        {
            "fecha": fechas[reseñas_validas].dt.normalize(),
            "estrellas": estrellas[reseñas_validas],
        }
    )
    diarios = (
        datos_validos.groupby("fecha")["estrellas"]
        .agg(promedio_estrellas="mean", cantidad_resenas_validas="size")
        .sort_index()
        .reset_index()
    )
    if len(diarios) < 2:
        raise ValueError("Se necesitan al menos dos dias observados para evaluar el modelo.")

    return diarios, int(reseñas_validas.sum()), fechas_invalidas, calificaciones_invalidas


def guardar_grafica(
    diarios: pd.DataFrame,
    fechas_pronostico: pd.DatetimeIndex,
    predicciones_historicas: np.ndarray,
    predicciones_futuras: np.ndarray,
) -> None:
    figura, eje = plt.subplots(figsize=(11, 5.5))
    eje.scatter(
        diarios["fecha"],
        diarios["promedio_estrellas"],
        s=20,
        alpha=0.75,
        color="#286b8e",
        label="Promedio diario observado",
        zorder=3,
    )
    eje.plot(
        diarios["fecha"],
        predicciones_historicas,
        color="#334d3d",
        linewidth=2,
        label="Regresion lineal historica",
    )
    eje.plot(
        fechas_pronostico,
        predicciones_futuras,
        color="#d56b3f",
        linewidth=2.2,
        label="Pronostico de 30 dias",
    )
    limite_periodos = diarios["fecha"].iloc[-1] + pd.Timedelta(hours=12)
    eje.axvline(limite_periodos, color="#777777", linestyle="--", linewidth=1)
    eje.set_title("Promedio diario de estrellas y pronóstico de los siguientes 30 días")
    eje.set_xlabel("Fecha")
    eje.set_ylabel("Promedio de estrellas")
    eje.set_ylim(1, 5)
    eje.xaxis.set_major_locator(mdates.AutoDateLocator())
    eje.xaxis.set_major_formatter(mdates.ConciseDateFormatter(eje.xaxis.get_major_locator()))
    eje.grid(axis="y", alpha=0.25)
    eje.legend(loc="best")
    figura.tight_layout()
    figura.savefig(CARPETA_RESULTADOS / "pronostico_estrellas.png", dpi=160)
    plt.close(figura)


def main() -> None:
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    diarios, total_validas, fechas_invalidas, calificaciones_invalidas = (
        cargar_promedios_diarios()
    )
    diarios.to_csv(
        CARPETA_RESULTADOS / "promedio_diario_estrellas.csv",
        index=False,
        date_format="%Y-%m-%d",
    )

    fechas = pd.to_datetime(diarios["fecha"])
    origen = fechas.iloc[0]
    dias_transcurridos = (fechas - origen).dt.days.to_numpy(dtype=float).reshape(-1, 1)
    valores = diarios["promedio_estrellas"].to_numpy(dtype=float)
    indice_corte = int(len(diarios) * 0.8)
    if indice_corte < 1 or indice_corte >= len(diarios):
        raise ValueError("La division cronologica 80/20 no deja datos suficientes.")

    modelo_evaluacion = LinearRegression()
    modelo_evaluacion.fit(dias_transcurridos[:indice_corte], valores[:indice_corte])
    predicciones_prueba_sin_limitar = modelo_evaluacion.predict(
        dias_transcurridos[indice_corte:]
    )
    predicciones_prueba = np.clip(predicciones_prueba_sin_limitar, 1, 5)
    prueba_recortadas = int(
        np.count_nonzero(predicciones_prueba != predicciones_prueba_sin_limitar)
    )
    metricas = {
        "MAE": mean_absolute_error(valores[indice_corte:], predicciones_prueba),
        "RMSE": sqrt(mean_squared_error(valores[indice_corte:], predicciones_prueba)),
        "R2": r2_score(valores[indice_corte:], predicciones_prueba),
    }
    pd.DataFrame(
        [{"metrica": nombre, "valor": valor} for nombre, valor in metricas.items()]
    ).to_csv(CARPETA_RESULTADOS / "metricas_regresion_lineal.csv", index=False)

    modelo_completo = LinearRegression()
    modelo_completo.fit(dias_transcurridos, valores)
    predicciones_historicas = modelo_completo.predict(dias_transcurridos)
    ultima_fecha = fechas.iloc[-1]
    fechas_pronostico = pd.date_range(
        start=ultima_fecha + pd.Timedelta(days=1),
        periods=DURACION_PRONOSTICO,
        freq="D",
    )
    dias_pronostico = (
        (fechas_pronostico - origen).days.to_numpy(dtype=float).reshape(-1, 1)
    )
    predicciones_futuras_sin_limitar = modelo_completo.predict(dias_pronostico)
    predicciones_futuras = np.clip(predicciones_futuras_sin_limitar, 1, 5)
    cantidad_recortadas = int(
        np.count_nonzero(predicciones_futuras != predicciones_futuras_sin_limitar)
    )
    pd.DataFrame(
        {
            "fecha": fechas_pronostico,
            "promedio_estrellas_pronosticado": predicciones_futuras,
        }
    ).to_csv(
        CARPETA_RESULTADOS / "pronostico_30_dias.csv",
        index=False,
        date_format="%Y-%m-%d",
    )

    guardar_grafica(diarios, fechas_pronostico, predicciones_historicas, predicciones_futuras)

    fecha_prueba_inicio = fechas.iloc[indice_corte]
    fecha_prueba_fin = fechas.iloc[-1]
    texto = [
        "Procedimiento: se leyo Recipe_Reviews_Clean.csv sin modificarlo. created_at se",
        f"interpreto con el formato {FORMATO_FECHA}; se conservaron estrellas entre 1 y 5.",
        "Las reseñas se agruparon por dia observado; los dias sin reseñas se excluyeron.",
        "Se entreno una regresion lineal con dias transcurridos y promedio diario de estrellas.",
        f"Reseñas validas: {total_validas}; dias observados: {len(diarios)}.",
        f"Fechas analizadas: {fechas.iloc[0]:%Y-%m-%d} a {ultima_fecha:%Y-%m-%d}.",
        f"Fechas con formato invalido descartadas: {fechas_invalidas}; calificaciones invalidas: {calificaciones_invalidas}.",
        f"Evaluacion cronologica: {indice_corte} dias de entrenamiento y {len(diarios) - indice_corte} de prueba.",
        f"Periodo de prueba: {fecha_prueba_inicio:%Y-%m-%d} a {fecha_prueba_fin:%Y-%m-%d}.",
        f"MAE: {metricas['MAE']:.4f}; RMSE: {metricas['RMSE']:.4f}; R2: {metricas['R2']:.4f}.",
        f"Predicciones de prueba limitadas al rango 1-5: {prueba_recortadas} de {len(predicciones_prueba)}.",
        f"Pronostico: {len(fechas_pronostico)} dias, del {fechas_pronostico[0]:%Y-%m-%d} al {fechas_pronostico[-1]:%Y-%m-%d}.",
        f"Predicciones futuras limitadas al rango 1-5: {cantidad_recortadas} de {DURACION_PRONOSTICO}.",
        "Conclusion: el R2 negativo indica capacidad predictiva limitada en el periodo de prueba.",
        "El pronostico extiende la tendencia lineal observada y es una estimacion agregada,",
        "no una garantia del promedio real de cada dia.",
    ]
    (CARPETA_RESULTADOS / "resultados_modelo.txt").write_text(
        "\n".join(texto) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()