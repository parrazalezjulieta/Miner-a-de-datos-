from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CARPETA_GRAFICAS = BASE_DIR / "datos" / "graficas_practica_3"
ARCHIVO_DATOS = BASE_DIR / "datos" / "Recipe_Reviews_Clean.csv"
ARCHIVO_USUARIOS = BASE_DIR / "datos" / "resumen_usuarios.csv"
ARCHIVO_RECETAS = BASE_DIR / "datos" / "resumen_recetas.csv"

COLUMNAS_INTERACCION = ["thumbs_up", "thumbs_down", "reply_count"]
GRUPOS_REPUTACION = [
    ("Grupo 1: reputación 0", lambda serie: serie == 0),
    ("Grupo 2: reputación 1", lambda serie: serie == 1),
    ("Grupo 3: reputación 10–20", lambda serie: serie.between(10, 20)),
    ("Grupo 4: reputación 30–510", lambda serie: serie.between(30, 510)),
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cargar los datos de la Práctica 1 y los resúmenes de la Práctica 2."""
    datos = pd.read_csv(ARCHIVO_DATOS)
    usuarios = pd.read_csv(ARCHIVO_USUARIOS)
    recetas = pd.read_csv(ARCHIVO_RECETAS)
    return datos, usuarios, recetas


def crear_carpeta_graficas() -> None:
    """Crear la carpeta exclusiva para las gráficas de la Práctica 3."""
    CARPETA_GRAFICAS.mkdir(parents=True, exist_ok=True)


def preparar_interaccion_total(datos: pd.DataFrame) -> pd.DataFrame:
    """Calcular interaccion_total si aún no existe en el dataset."""
    resultado = datos.copy()
    if "interaccion_total" not in resultado.columns:
        columnas = [pd.to_numeric(resultado[columna], errors="coerce").fillna(0) for columna in COLUMNAS_INTERACCION]
        resultado["interaccion_total"] = sum(columnas)
    else:
        resultado["interaccion_total"] = pd.to_numeric(
            resultado["interaccion_total"], errors="coerce"
        )
    return resultado


def guardar_grafica(figura: plt.Figure, nombre_archivo: str) -> None:
    """Guardar una figura dentro de la carpeta de la Práctica 3."""
    figura.tight_layout()
    figura.savefig(CARPETA_GRAFICAS / nombre_archivo, dpi=120, bbox_inches="tight")
    plt.close(figura)


def crear_grafica_pastel(datos: pd.DataFrame) -> None:
    """Crear la distribución de reseñas por cantidad de estrellas."""
    distribucion = (
        pd.to_numeric(datos["stars"], errors="coerce")
        .value_counts()
        .reindex(range(6), fill_value=0)
    )
    figura, ax = plt.subplots(figsize=(8, 6))
    ax.pie(
        distribucion.values,
        labels=[f"{estrella} estrellas" for estrella in distribucion.index],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title("Distribución de calificaciones por estrellas")
    guardar_grafica(figura, "pastel_distribucion_estrellas.png")


def crear_histograma(datos: pd.DataFrame) -> None:
    """Crear el histograma de la interacción total."""
    figura, ax = plt.subplots(figsize=(9, 6))
    ax.hist(datos["interaccion_total"].dropna(), bins=30, color="steelblue", edgecolor="black")
    ax.set_xlabel("Interacción total")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de la interacción total")
    ax.grid(axis="y", alpha=0.3)
    guardar_grafica(figura, "histograma_interaccion_total.png")


def clasificar_reputacion(datos: pd.DataFrame) -> pd.DataFrame:
    """Aplicar los cuatro grupos de reputación usados en la Práctica 2."""
    resultado = datos.copy()
    reputacion = pd.to_numeric(resultado["user_reputation"], errors="coerce")
    resultado["grupo_reputacion"] = pd.NA

    for nombre_grupo, condicion in GRUPOS_REPUTACION:
        resultado.loc[condicion(reputacion), "grupo_reputacion"] = nombre_grupo

    return resultado.dropna(subset=["grupo_reputacion"])


def crear_boxplot(datos: pd.DataFrame) -> None:
    """Comparar la interacción total entre los grupos de reputación."""
    datos_clasificados = clasificar_reputacion(datos)
    etiquetas = [nombre for nombre, _ in GRUPOS_REPUTACION]
    valores = [
        datos_clasificados.loc[
            datos_clasificados["grupo_reputacion"] == etiqueta, "interaccion_total"
        ].dropna()
        for etiqueta in etiquetas
    ]

    figura, ax = plt.subplots(figsize=(11, 6))
    ax.boxplot(valores, labels=etiquetas, patch_artist=True)
    ax.set_xlabel("Grupo de reputación")
    ax.set_ylabel("Interacción total")
    ax.set_title("Interacción total por grupo de reputación")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)
    guardar_grafica(figura, "boxplot_interaccion_por_reputacion.png")


def crear_dispersion(usuarios: pd.DataFrame) -> None:
    """Crear la relación entre reseñas y promedio de interacción por usuario."""
    figura, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        usuarios["num_reviews"],
        usuarios["avg_interaccion_total"],
        alpha=0.65,
        color="darkorange",
        edgecolors="black",
        linewidths=0.3,
    )
    ax.set_xlabel("num_reviews")
    ax.set_ylabel("avg_interaccion_total")
    ax.set_title("Cantidad de reseñas vs promedio de interacción por usuario")
    ax.grid(alpha=0.3)
    guardar_grafica(figura, "dispersion_resenas_interaccion_usuario.png")


def crear_grafica_barras(recetas: pd.DataFrame) -> None:
    """Crear barras para las diez recetas con más reseñas."""
    top_recetas = recetas.nlargest(10, "num_reviews").reset_index(drop=True)
    etiquetas = top_recetas["recipe_code"].astype(str).tolist()

    figura, ax = plt.subplots(figsize=(10, 6))
    ax.bar(etiquetas, top_recetas["num_reviews"], color="seagreen", edgecolor="black")
    ax.set_xlabel("Código de receta")
    ax.set_ylabel("Cantidad de reseñas")
    ax.set_title("Top 10 recetas por cantidad de reseñas")
    ax.grid(axis="y", alpha=0.3)
    guardar_grafica(figura, "05_barras_recetas.png")


def generar_graficas(
    datos: pd.DataFrame, usuarios: pd.DataFrame, recetas: pd.DataFrame
) -> None:
    """Ejecutar las cinco gráficas mediante una configuración automatizada."""
    graficas = [
        (crear_grafica_pastel, datos),
        (crear_histograma, datos),
        (crear_boxplot, datos),
        (crear_dispersion, usuarios),
        (crear_grafica_barras, recetas),
    ]
    for crear_grafica, fuente in graficas:
        crear_grafica(fuente)


def main() -> None:
    crear_carpeta_graficas()
    datos, usuarios, recetas = load_data()
    datos = preparar_interaccion_total(datos)
    generar_graficas(datos, usuarios, recetas)
    print(f"Cinco gráficas generadas en: {CARPETA_GRAFICAS}")


if __name__ == "__main__":
    main()