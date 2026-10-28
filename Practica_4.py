from pathlib import Path

import pandas as pd
from scipy import stats


BASE_DIR = Path(__file__).resolve().parent
ARCHIVO_USUARIOS = BASE_DIR / "datos" / "resumen_usuarios.csv"
ARCHIVO_RESULTADOS = BASE_DIR / "datos" / "resultados_practica_4.xlsx"

ALPHA = 0.05
COLUMNAS_NECESARIAS = [
    "user_id",
    "user_reputation",
    "num_reviews",
    "avg_interaccion_total",
]
GRUPOS_REPUTACION = [
    ("Grupo 1: reputacion 0", lambda serie: serie == 0),
    ("Grupo 2: reputacion 1", lambda serie: serie == 1),
    ("Grupo 3: reputacion 10-20", lambda serie: serie.between(10, 20)),
    ("Grupo 4: reputacion 30-510", lambda serie: serie.between(30, 510)),
]


def cargar_y_preparar_datos() -> pd.DataFrame:
    """Cargar y clasificar el resumen con una fila por usuario."""
    datos = pd.read_csv(ARCHIVO_USUARIOS)
    faltantes = [columna for columna in COLUMNAS_NECESARIAS if columna not in datos.columns]
    if faltantes:
        raise ValueError(
            "Faltan columnas necesarias en resumen_usuarios.csv: " + ", ".join(faltantes)
        )

    datos = datos[COLUMNAS_NECESARIAS].copy()
    for columna in ["user_reputation", "num_reviews", "avg_interaccion_total"]:
        datos[columna] = pd.to_numeric(datos[columna], errors="coerce")

    if datos["user_id"].duplicated().any():
        raise ValueError("resumen_usuarios.csv debe contener una sola fila por user_id.")

    datos = datos.dropna(subset=COLUMNAS_NECESARIAS).copy()
    datos["grupo_reputacion"] = pd.NA
    for nombre_grupo, condicion in GRUPOS_REPUTACION:
        datos.loc[condicion(datos["user_reputation"]), "grupo_reputacion"] = nombre_grupo

    return datos.dropna(subset=["grupo_reputacion"]).copy()


def crear_descriptivos(datos: pd.DataFrame) -> pd.DataFrame:
    """Calcular estadisticos del promedio de interaccion por usuario."""
    filas = []
    for nombre_grupo, _ in GRUPOS_REPUTACION:
        valores = datos.loc[
            datos["grupo_reputacion"] == nombre_grupo, "avg_interaccion_total"
        ]
        filas.append(
            {
                "grupo_reputacion": nombre_grupo,
                "cantidad_usuarios": valores.count(),
                "media": valores.mean(),
                "mediana": valores.median(),
                "desviacion_estandar": valores.std(),
                "minimo": valores.min(),
                "Q1": valores.quantile(0.25),
                "Q3": valores.quantile(0.75),
                "maximo": valores.max(),
            }
        )
    return pd.DataFrame(filas)


def obtener_grupos(datos: pd.DataFrame) -> dict[str, pd.Series]:
    """Separar los promedios individuales de cada usuario por grupo."""
    return {
        nombre_grupo: datos.loc[
            datos["grupo_reputacion"] == nombre_grupo, "avg_interaccion_total"
        ].dropna()
        for nombre_grupo, _ in GRUPOS_REPUTACION
    }


def revisar_supuestos(
    datos: pd.DataFrame, grupos: dict[str, pd.Series]
) -> tuple[pd.DataFrame, float, float]:
    """Revisar normalidad descriptiva, asimetria y homogeneidad de varianzas."""
    filas = []
    for nombre_grupo, valores in grupos.items():
        # Shapiro admite como maximo 5000 observaciones; se usa una muestra reproducible.
        muestra = valores.sample(n=min(len(valores), 5000), random_state=42)
        if len(muestra) >= 3:
            estadistico, p_value = stats.shapiro(muestra)
        else:
            estadistico, p_value = float("nan"), float("nan")
        filas.append(
            {
                "grupo_reputacion": nombre_grupo,
                "n": len(valores),
                "shapiro_estadistico": estadistico,
                "shapiro_p_value": p_value,
                "asimetria": valores.skew(),
            }
        )

    estadistico_levene, p_value_levene = stats.levene(*grupos.values(), center="median")
    supuestos = pd.DataFrame(filas)
    return supuestos, estadistico_levene, p_value_levene


def decidir_prueba(
    supuestos: pd.DataFrame,
    p_value_levene: float,
) -> tuple[str, str]:
    """Elegir prueba considerando objetivo, varianzas y desequilibrio muestral."""
    tamanos = supuestos["n"]
    razon_tamanos = tamanos.max() / tamanos.min()
    varianzas_diferentes = p_value_levene <= ALPHA

    if varianzas_diferentes and razon_tamanos >= 1.5 and tamanos.min() >= 30:
        return (
            "welch",
            "Se compara la media del promedio de interaccion por usuario. Levene indica "
            "varianzas distintas y los tamanos son muy desiguales; por ello se usa Welch, "
            "que no combina las varianzas. El grupo menor tiene un tamano moderado. Shapiro "
            "y la asimetria se reportan como diagnosticos, no como veto automatico; la "
            "asimetria marcada aconseja cautela al interpretar la aproximacion de medias.",
        )

    normalidad_no_rechazada = (supuestos["shapiro_p_value"] > ALPHA).all()
    varianzas_homogeneas = p_value_levene > ALPHA
    if normalidad_no_rechazada and varianzas_homogeneas:
        return (
            "anova",
            "Se aplica ANOVA de una via: la normalidad no se rechaza en los grupos y "
            "Levene no detecta varianzas diferentes.",
        )

    return (
        "kruskal",
        "Los diagnosticos no respaldan ANOVA clasico ni cumplen el criterio documentado "
        "para Welch. Se usa Kruskal-Wallis como prueba de rangos entre distribuciones; "
        "no es una prueba directa de igualdad de medias.",
    )


def resultado_no_aplicado(prueba: str, razon: str) -> pd.DataFrame:
    return pd.DataFrame(
        [{"prueba": prueba, "ejecutada": False, "razon": razon}]
    )


def ejecutar_games_howell(grupos: dict[str, pd.Series]) -> pd.DataFrame:
    """Comparar pares tras Welch con el ajuste Games-Howell."""
    resultado = stats.tukey_hsd(*grupos.values(), equal_var=False)
    intervalo = resultado.confidence_interval(confidence_level=1 - ALPHA)
    nombres = list(grupos)
    filas = []
    for indice_a in range(len(nombres)):
        for indice_b in range(indice_a + 1, len(nombres)):
            filas.append(
                {
                    "grupo_a": nombres[indice_a],
                    "grupo_b": nombres[indice_b],
                    "diferencia_de_medias": resultado.statistic[indice_a, indice_b],
                    "p_value_ajustado": resultado.pvalue[indice_a, indice_b],
                    "IC_inferior": intervalo.low[indice_a, indice_b],
                    "IC_superior": intervalo.high[indice_a, indice_b],
                    "alpha": ALPHA,
                }
            )
    return pd.DataFrame(filas)


def imprimir_resumen(
    descriptivos: pd.DataFrame,
    supuestos: pd.DataFrame,
    estadistico_levene: float,
    p_value_levene: float,
    decision: str,
    razon_decision: str,
    anova: pd.DataFrame,
    kruskal: pd.DataFrame,
    pruebas_t: pd.DataFrame,
) -> None:
    print("=== PRACTICA 4: INTERACCION TOTAL POR REPUTACION ===")
    print("\n=== DESCRIPTIVOS ===")
    print(descriptivos.to_string(index=False))
    print("\n=== SUPUESTOS ===")
    print(supuestos.to_string(index=False))
    print(f"\nLevene (estadistico): {estadistico_levene}")
    print(f"Levene (p-value): {p_value_levene}")
    print(f"\nPrueba seleccionada: {decision}")
    print(f"Criterio: {razon_decision}")
    print("\n=== ANOVA ===")
    print(anova.to_string(index=False))
    print("\n=== KRUSKAL-WALLIS ===")
    print(kruskal.to_string(index=False))
    print("\n=== COMPARACIONES POSTERIORES ===")
    print(pruebas_t.to_string(index=False))


def guardar_resultados(
    datos: pd.DataFrame,
    descriptivos: pd.DataFrame,
    supuestos: pd.DataFrame,
    decision: str,
    razon_decision: str,
    estadistico_levene: float,
    p_value_levene: float,
    anova: pd.DataFrame,
    kruskal: pd.DataFrame,
    comparaciones: pd.DataFrame,
) -> None:
    """Guardar datos y resultados en hojas separadas."""
    with pd.ExcelWriter(ARCHIVO_RESULTADOS, engine="openpyxl") as escritor:
        datos.to_excel(escritor, sheet_name="Datos_Usuario", index=False)
        descriptivos.to_excel(escritor, sheet_name="Descriptivos", index=False)
        supuestos.to_excel(escritor, sheet_name="Supuestos", index=False, startrow=0)
        pd.DataFrame(
            [{
                "prueba_seleccionada": decision,
                "criterio": razon_decision,
                "levene_estadistico": estadistico_levene,
                "levene_p_value": p_value_levene,
                "alpha": ALPHA,
            }]
        ).to_excel(
            escritor,
            sheet_name="Supuestos",
            index=False,
            startrow=len(supuestos) + 3,
        )
        pruebas = pd.concat([anova, kruskal, comparaciones], ignore_index=True, sort=False)
        pruebas.to_excel(escritor, sheet_name="Pruebas", index=False)


def main() -> None:
    datos = cargar_y_preparar_datos()
    grupos = obtener_grupos(datos)
    descriptivos = crear_descriptivos(datos)
    supuestos, estadistico_levene, p_value_levene = revisar_supuestos(datos, grupos)
    decision, razon_decision = decidir_prueba(supuestos, p_value_levene)

    anova = resultado_no_aplicado("ANOVA", "No fue la prueba seleccionada.")
    kruskal = resultado_no_aplicado("Kruskal-Wallis", "No fue la prueba seleccionada.")
    comparaciones = resultado_no_aplicado(
        "Comparaciones posteriores", "No se requieren para la prueba seleccionada."
    )

    if decision == "welch":
        resultado_welch = stats.f_oneway(*grupos.values(), equal_var=False)
        anova = pd.DataFrame(
            [{
                "prueba": "ANOVA de Welch",
                "estadistico_F": resultado_welch.statistic,
                "p_value": resultado_welch.pvalue,
                "alpha": ALPHA,
                "ejecutada": True,
                "razon": razon_decision,
            }]
        )
        if resultado_welch.pvalue < ALPHA:
            comparaciones = ejecutar_games_howell(grupos)
            comparaciones["prueba"] = "Games-Howell"
            comparaciones["ejecutada"] = True
        else:
            comparaciones = resultado_no_aplicado(
                "Games-Howell",
                "No se aplica porque la comparacion global de Welch no fue significativa.",
            )
    elif decision == "anova":
        resultado_anova = stats.f_oneway(*grupos.values())
        anova = pd.DataFrame(
            [{
                "prueba": "ANOVA de una via",
                "estadistico_F": resultado_anova.statistic,
                "p_value": resultado_anova.pvalue,
                "alpha": ALPHA,
                "ejecutada": True,
                "razon": razon_decision,
            }]
        )
        if resultado_anova.pvalue < ALPHA:
            comparaciones = ejecutar_games_howell(grupos)
            comparaciones["prueba"] = "Games-Howell"
            comparaciones["ejecutada"] = True
        else:
            comparaciones = resultado_no_aplicado(
                "Comparaciones posteriores",
                "No se aplica porque ANOVA no fue significativo.",
            )
    else:
        resultado_kruskal = stats.kruskal(*grupos.values())
        kruskal = pd.DataFrame(
            [{
                "prueba": "Kruskal-Wallis",
                "estadistico_H": resultado_kruskal.statistic,
                "p_value": resultado_kruskal.pvalue,
                "alpha": ALPHA,
                "ejecutada": True,
                "razon": razon_decision,
            }]
        )
        comparaciones = resultado_no_aplicado(
            "Comparaciones posteriores",
            "No se aplican pruebas t: Kruskal-Wallis fue seleccionado y no se definio una prueba post hoc para rangos.",
        )

    guardar_resultados(
        datos,
        descriptivos,
        supuestos,
        decision,
        razon_decision,
        estadistico_levene,
        p_value_levene,
        anova,
        kruskal,
        comparaciones,
    )
    imprimir_resumen(
        descriptivos,
        supuestos,
        estadistico_levene,
        p_value_levene,
        decision,
        razon_decision,
        anova,
        kruskal,
        comparaciones,
    )
    print(f"\nResultados guardados en: {ARCHIVO_RESULTADOS}")


if __name__ == "__main__":
    main()
