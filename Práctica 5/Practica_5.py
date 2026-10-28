from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression


BASE_DIR = Path(__file__).resolve().parent
ARCHIVO_ENTRADA = BASE_DIR / "datos" / "resumen_usuarios.csv"
CARPETA_RESULTADOS = BASE_DIR / "datos" / "resultados_practica_5"
COLUMNAS_NECESARIAS = ["num_reviews", "avg_interaccion_total"]


def cargar_datos() -> tuple[pd.DataFrame, int]:
    """Carga el resumen y conserva filas válidas para las dos variables del modelo."""
    if not ARCHIVO_ENTRADA.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {ARCHIVO_ENTRADA}")

    datos = pd.read_csv(ARCHIVO_ENTRADA)
    filas_cargadas = len(datos)
    faltantes = [columna for columna in COLUMNAS_NECESARIAS if columna not in datos.columns]
    if faltantes:
        raise ValueError(
            "Faltan columnas necesarias en resumen_usuarios.csv: " + ", ".join(faltantes)
        )

    datos = datos[COLUMNAS_NECESARIAS].copy()
    for columna in COLUMNAS_NECESARIAS:
        datos[columna] = pd.to_numeric(datos[columna], errors="coerce")
        datos[columna] = datos[columna].replace([float("inf"), float("-inf")], pd.NA)

    datos = datos.dropna(subset=COLUMNAS_NECESARIAS).copy()
    print(f"Filas cargadas: {filas_cargadas}")
    print(f"Filas utilizadas después de limpiar: {len(datos)}")

    if len(datos) < 2:
        raise ValueError(
            "Datos insuficientes: se necesitan al menos dos filas válidas para ajustar el modelo."
        )

    return datos, filas_cargadas


def ajustar_modelo(datos: pd.DataFrame) -> tuple[LinearRegression, pd.Series, float]:
    """Ajusta la regresión y calcula predicciones y R² sobre los mismos datos."""
    X = datos[["num_reviews"]]
    y = datos["avg_interaccion_total"]

    modelo = LinearRegression()
    modelo.fit(X, y)
    predicciones = modelo.predict(X)
    # R² mide la proporción de variabilidad de y explicada por el modelo en estos datos.
    r_cuadrado = modelo.score(X, y)
    return modelo, pd.Series(predicciones, index=datos.index), r_cuadrado


def guardar_graficas(
    datos: pd.DataFrame,
    modelo: LinearRegression,
    predicciones: pd.Series,
) -> None:
    """Crea y guarda la recta ajustada y la comparación real frente a predicho."""
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    extremos_x = pd.DataFrame(
        {"num_reviews": [datos["num_reviews"].min(), datos["num_reviews"].max()]}
    )
    valores_recta = modelo.predict(extremos_x)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        datos["num_reviews"],
        datos["avg_interaccion_total"],
        alpha=0.55,
        label="Usuarios",
    )
    ax.plot(extremos_x["num_reviews"], valores_recta, color="crimson", label="Regresión lineal")
    ax.set_title("Número de reseñas e interacción promedio")
    ax.set_xlabel("Número de reseñas por usuario")
    ax.set_ylabel("Interacción total promedio")
    pendiente = float(modelo.coef_[0])
    intercepto = float(modelo.intercept_)
    r_cuadrado = modelo.score(datos[["num_reviews"]], datos["avg_interaccion_total"])
    ax.legend(loc="upper right")
    ax.text(
        0.98,
        0.86,
        f"Ecuación: y = {pendiente:.6f}x {intercepto:+.6f}\nR² = {r_cuadrado:.6f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "gray", "alpha": 0.85},
    )
    fig.tight_layout()
    fig.savefig(CARPETA_RESULTADOS / "modelo_lineal.png", dpi=150)
    plt.close(fig)

    valores_reales = datos["avg_interaccion_total"]
    minimo = min(valores_reales.min(), predicciones.min())
    maximo = max(valores_reales.max(), predicciones.max())
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(valores_reales, predicciones, alpha=0.55, label="Observaciones")
    ax.plot([minimo, maximo], [minimo, maximo], "--", color="crimson", label="Referencia (y = x)")
    ax.set_title("Valores reales frente a valores predichos")
    ax.set_xlabel("Interacción total promedio real")
    ax.set_ylabel("Interacción total promedio predicha")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CARPETA_RESULTADOS / "reales_vs_predichos.png", dpi=150)
    plt.close(fig)


def guardar_resultados(
    datos: pd.DataFrame,
    modelo: LinearRegression,
    predicciones: pd.Series,
    r_cuadrado: float,
) -> None:
    """Guarda las métricas y los valores usados para revisar las predicciones."""
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    # La pendiente indica el cambio estimado de interacción por cada reseña adicional.
    pendiente = float(modelo.coef_[0])
    # El intercepto es el valor estimado de interacción cuando num_reviews vale cero.
    intercepto = float(modelo.intercept_)
    metricas = pd.DataFrame(
        [
            {"metrica": "R^2", "valor": r_cuadrado},
            {"metrica": "pendiente", "valor": pendiente},
            {"metrica": "intercepto", "valor": intercepto},
            {"metrica": "observaciones_utilizadas", "valor": len(datos)},
        ]
    )
    metricas.to_csv(CARPETA_RESULTADOS / "resultados_practica_5.csv", index=False)

    comparacion = pd.DataFrame(
        {
            "num_reviews": datos["num_reviews"],
            "valor_real": datos["avg_interaccion_total"],
            "valor_predicho": predicciones,
        }
    )
    comparacion.to_csv(
        CARPETA_RESULTADOS / "valores_reales_y_predichos.csv", index=False
    )


def main() -> None:
    datos, _ = cargar_datos()
    modelo, predicciones, r_cuadrado = ajustar_modelo(datos)
    pendiente = float(modelo.coef_[0])
    intercepto = float(modelo.intercept_)

    print("\n=== MODELO LINEAL ===")
    print(f"Ecuación: avg_interaccion_total = {pendiente:.6f} * num_reviews + {intercepto:.6f}")
    print(f"Pendiente: {pendiente:.6f}")
    print(f"Intercepto: {intercepto:.6f}")
    print(f"R^2: {r_cuadrado:.6f}")

    guardar_graficas(datos, modelo, predicciones)
    guardar_resultados(datos, modelo, predicciones, r_cuadrado)
    print(f"\nResultados guardados en: {CARPETA_RESULTADOS}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as error:
        raise SystemExit(f"Error al ejecutar la Práctica 5: {error}") from error
