from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
ARCHIVO_ENTRADA = BASE_DIR / "datos" / "resumen_usuarios.csv"
CARPETA_RESULTADOS = BASE_DIR / "datos" / "resultados_practica_6"
COLUMNA_OBJETIVO = "avg_interaccion_total"
COLUMNAS_PREDICTORAS = [
    "user_reputation",
    "num_reviews",
    "avg_stars",
    "avg_best_score",
]
CLASES = ["Participación baja", "Participación media", "Participación alta"]
RANDOM_STATE = 42
NUM_VECINOS = 5


def cargar_datos() -> tuple[pd.DataFrame, int, int]:
    """Carga el resumen, convierte variables y descarta filas incompletas o inválidas."""
    if not ARCHIVO_ENTRADA.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {ARCHIVO_ENTRADA}")

    datos = pd.read_csv(ARCHIVO_ENTRADA)
    filas_cargadas = len(datos)
    columnas_necesarias = COLUMNAS_PREDICTORAS + [COLUMNA_OBJETIVO]
    faltantes = [columna for columna in columnas_necesarias if columna not in datos.columns]
    if faltantes:
        raise ValueError(
            "Faltan columnas necesarias en resumen_usuarios.csv: " + ", ".join(faltantes)
        )

    datos = datos[columnas_necesarias].copy()
    for columna in columnas_necesarias:
        datos[columna] = pd.to_numeric(datos[columna], errors="coerce")
        datos[columna] = datos[columna].replace([float("inf"), float("-inf")], pd.NA)

    datos = datos.dropna(subset=columnas_necesarias).copy()
    datos = datos[
        (datos[COLUMNA_OBJETIVO] >= 0) & (datos["num_reviews"] > 0)
    ].copy()
    filas_descartadas = filas_cargadas - len(datos)

    if len(datos) < 2:
        raise ValueError("Datos insuficientes: se necesitan al menos dos filas válidas.")

    return datos, filas_cargadas, filas_descartadas


def crear_categorias(datos: pd.DataFrame) -> tuple[pd.Series, float]:
    """Asigna clases: cero es baja; positivos se dividen por su mediana."""
    interaccion = datos[COLUMNA_OBJETIVO]
    positivos = interaccion[interaccion > 0]
    if positivos.empty:
        raise ValueError("No hay valores positivos para definir participación media y alta.")

    # La actividad nula define participación baja; la mediana positiva separa media y alta.
    umbral = float(positivos.median())
    categorias = pd.Series(index=datos.index, dtype="string")
    categorias.loc[interaccion == 0] = CLASES[0]
    categorias.loc[(interaccion > 0) & (interaccion <= umbral)] = CLASES[1]
    categorias.loc[interaccion > umbral] = CLASES[2]

    if categorias.isna().any() or categorias.nunique() != len(CLASES):
        raise ValueError("No fue posible asignar correctamente las tres categorías.")

    return categorias, umbral


def dividir_y_escalar(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """Divide los datos 80/20 y ajusta el escalador solo con entrenamiento."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    if len(X_train) < NUM_VECINOS:
        raise ValueError("No hay suficientes registros de entrenamiento para K=5.")

    escalador = StandardScaler()
    X_train_escalado = escalador.fit_transform(X_train)
    X_test_escalado = escalador.transform(X_test)
    return X_train_escalado, X_test_escalado, y_train, y_test, escalador


def entrenar_modelo(X_train: pd.DataFrame, y_train: pd.Series) -> KNeighborsClassifier:
    """Entrena KNN con cinco vecinos, un valor moderado que reduce sensibilidad al ruido."""
    modelo = KNeighborsClassifier(n_neighbors=NUM_VECINOS)
    modelo.fit(X_train, y_train)
    return modelo


def evaluar_modelo(
    modelo: KNeighborsClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> tuple[dict[str, float], dict[str, dict[str, float]], pd.DataFrame, pd.Series]:
    """Calcula todas las métricas exclusivamente sobre el conjunto de prueba."""
    predicciones = modelo.predict(X_test)
    metricas = {
        "accuracy": float(accuracy_score(y_test, predicciones)),
        "precision_macro": float(
            precision_score(y_test, predicciones, labels=CLASES, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_test, predicciones, labels=CLASES, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_test, predicciones, labels=CLASES, average="macro", zero_division=0)
        ),
    }
    reporte = classification_report(
        y_test,
        predicciones,
        labels=CLASES,
        target_names=CLASES,
        output_dict=True,
        zero_division=0,
    )
    matriz = confusion_matrix(y_test, predicciones, labels=CLASES)
    matriz_df = pd.DataFrame(matriz, index=CLASES, columns=CLASES)
    predicciones_series = pd.Series(predicciones, index=y_test.index, name="prediccion")
    return metricas, reporte, matriz_df, predicciones_series


def guardar_matriz_confusion(matriz: pd.DataFrame) -> None:
    """Genera una matriz de confusión legible con ejes y categorías."""
    figura, eje = plt.subplots(figsize=(8, 6))
    imagen = eje.imshow(matriz.values, cmap="Blues")
    figura.colorbar(imagen, ax=eje, label="Número de usuarios")
    eje.set_title("Matriz de confusión del modelo KNN")
    eje.set_xlabel("Categoría predicha")
    eje.set_ylabel("Categoría real")
    eje.set_xticks(range(len(CLASSES := matriz.columns)), labels=CLASSES, rotation=15, ha="right")
    eje.set_yticks(range(len(matriz.index)), labels=matriz.index)

    for fila in range(len(matriz.index)):
        for columna in range(len(matriz.columns)):
            eje.text(columna, fila, str(matriz.iat[fila, columna]), ha="center", va="center")

    figura.tight_layout()
    figura.savefig(CARPETA_RESULTADOS / "matriz_confusion.png", dpi=150, bbox_inches="tight")
    plt.close(figura)


def guardar_resultados(
    metricas: dict[str, float],
    reporte: dict[str, dict[str, float]],
    matriz: pd.DataFrame,
    datos: pd.DataFrame,
    filas_cargadas: int,
    filas_descartadas: int,
    umbral: float,
    total_entrenamiento: int,
    total_prueba: int,
) -> None:
    """Guarda métricas, reporte por clase, gráfica y configuración del modelo."""
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"metrica": nombre, "valor": valor} for nombre, valor in metricas.items()]
    ).to_csv(CARPETA_RESULTADOS / "metricas_clasificacion.csv", index=False)

    reporte_df = pd.DataFrame(
        [
            {
                "categoria": categoria,
                "precision": reporte[categoria]["precision"],
                "recall": reporte[categoria]["recall"],
                "f1-score": reporte[categoria]["f1-score"],
                "support": reporte[categoria]["support"],
            }
            for categoria in CLASES
        ]
    )
    reporte_df.to_csv(CARPETA_RESULTADOS / "reporte_clasificacion.csv", index=False)
    guardar_matriz_confusion(matriz)

    resumen = [
        "=== Clasificación KNN: Práctica 6 ===",
        f"Archivo de entrada: {ARCHIVO_ENTRADA.relative_to(BASE_DIR)}",
        f"Registros cargados: {filas_cargadas}",
        f"Registros válidos utilizados: {len(datos)}",
        f"Registros descartados: {filas_descartadas}",
        f"Variables predictoras: {', '.join(COLUMNAS_PREDICTORAS)}",
        f"Variable para definir categorías (no usada como predictor): {COLUMNA_OBJETIVO}",
        "Regla de categorías: baja = 0; media = (0, mediana positiva]; alta = mayor que la mediana positiva.",
        f"Mediana de avg_interaccion_total entre valores positivos: {umbral:.6f}",
        f"División entrenamiento/prueba: {total_entrenamiento}/{total_prueba} (80 %/20 %)",
        f"K: {NUM_VECINOS} (valor moderado para equilibrar ruido y suavizado)",
        f"random_state: {RANDOM_STATE}",
        "Estandarización: StandardScaler ajustado solo con entrenamiento.",
        "Métricas globales: precision, recall y F1 son promedios macro.",
        "",
        "Métricas del conjunto de prueba:",
    ]
    resumen.extend(f"{nombre}: {valor:.6f}" for nombre, valor in metricas.items())
    resumen.extend(
        [
            "",
            "Matriz de confusión (filas = real, columnas = predicha):",
            matriz.to_string(),
            "",
            "Reporte por categoría:",
            reporte_df.to_string(),
        ]
    )
    (CARPETA_RESULTADOS / "resultados_modelo.txt").write_text(
        "\n".join(resumen) + "\n", encoding="utf-8"
    )


def main() -> None:
    datos, filas_cargadas, filas_descartadas = cargar_datos()
    y, umbral = crear_categorias(datos)
    X = datos[COLUMNAS_PREDICTORAS]
    X_train, X_test, y_train, y_test, _ = dividir_y_escalar(X, y)
    modelo = entrenar_modelo(X_train, y_train)
    metricas, reporte, matriz, _ = evaluar_modelo(modelo, X_test, y_test)

    guardar_resultados(
        metricas,
        reporte,
        matriz,
        datos,
        filas_cargadas,
        filas_descartadas,
        umbral,
        len(X_train),
        len(X_test),
    )

    print(f"Filas cargadas: {filas_cargadas}")
    print(f"Filas utilizadas: {len(datos)}; filas descartadas: {filas_descartadas}")
    print(f"Umbral entre participación media y alta: {umbral:.6f}")
    print("\nMétricas sobre el conjunto de prueba:")
    for nombre, valor in metricas.items():
        print(f"{nombre}: {valor:.6f}")
    print("\nMatriz de confusión (filas = real, columnas = predicha):")
    print(matriz)
    print(f"\nResultados guardados en: {CARPETA_RESULTADOS}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, OSError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as error:
        raise SystemExit(f"Error al ejecutar la Práctica 6: {error}") from error