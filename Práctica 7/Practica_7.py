from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
ARCHIVO_ENTRADA = BASE_DIR / "datos" / "resumen_usuarios.csv"
CARPETA_RESULTADOS = BASE_DIR / "datos" / "resultados_practica_7"
COLUMNAS_VARIABLES = [
    "user_reputation",
    "num_reviews",
    "avg_stars",
    "avg_thumbs_up",
    "avg_thumbs_down",
    "avg_reply_count",
    "avg_best_score",
]
COLUMNAS_LOG1P = [columna for columna in COLUMNAS_VARIABLES if columna != "avg_stars"]
RANDOM_STATE = 42
VALORES_K = range(2, 11)
MAX_MUESTRA_SILUETA = 3000


def cargar_datos() -> tuple[pd.DataFrame, int, int]:
    """Carga usuarios y elimina identificadores"""
    if not ARCHIVO_ENTRADA.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {ARCHIVO_ENTRADA}")

    datos = pd.read_csv(ARCHIVO_ENTRADA)
    filas_cargadas = len(datos)
    columnas_faltantes = [
        columna
        for columna in ["user_id", *COLUMNAS_VARIABLES]
        if columna not in datos.columns
    ]
    if columnas_faltantes:
        raise ValueError(
            "Faltan columnas necesarias en resumen_usuarios.csv: "
            + ", ".join(columnas_faltantes)
        )

    columnas_salida = ["user_id"]
    if "user_name" in datos.columns:
        columnas_salida.append("user_name")
    datos = datos[columnas_salida + COLUMNAS_VARIABLES].copy()
    datos["user_id"] = datos["user_id"].astype("string").str.strip()
    datos["user_id"] = datos["user_id"].replace("", pd.NA)
    for columna in COLUMNAS_VARIABLES:
        datos[columna] = pd.to_numeric(datos[columna], errors="coerce")
    datos[COLUMNAS_VARIABLES] = datos[COLUMNAS_VARIABLES].replace(
        [np.inf, -np.inf], np.nan
    )

    datos = datos.dropna(subset=["user_id", *COLUMNAS_VARIABLES]).copy()
    datos = datos[
        (datos["num_reviews"] > 0)
        & (datos[COLUMNAS_VARIABLES] >= 0).all(axis=1)
        & datos["avg_stars"].between(0, 5)
    ].copy()
    datos = datos.drop_duplicates(subset="user_id", keep="first")
    filas_descartadas = filas_cargadas - len(datos)

    if len(datos) < 11:
        raise ValueError("Se necesitan al menos 11 usuarios válidos para evaluar K=2..10.")

    return datos.reset_index(drop=True), filas_cargadas, filas_descartadas


def ajustar_escalador(
    datos: pd.DataFrame,
) -> tuple[np.ndarray, StandardScaler]:
    """Comprime colas largas con log1p y estandariza las variables resultantes."""
    valores = datos[COLUMNAS_VARIABLES].copy()
    valores[COLUMNAS_LOG1P] = np.log1p(valores[COLUMNAS_LOG1P])
    escalador = StandardScaler()
    valores_escalados = escalador.fit_transform(valores)
    return valores_escalados, escalador


def evaluar_cantidades_grupos(
    valores_escalados: np.ndarray,
) -> tuple[pd.DataFrame, dict[int, KMeans]]:
    """Entrena K=2..10 y calcula inercia y silueta en una muestra reproducible."""
    tamanio_muestra = min(MAX_MUESTRA_SILUETA, len(valores_escalados))
    resultados = []
    modelos = {}

    for k in VALORES_K:
        modelo = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        etiquetas = modelo.fit_predict(valores_escalados)
        if len(np.unique(etiquetas)) < 2:
            raise ValueError(f"K={k} produjo menos de dos grupos distintos.")
        silueta = silhouette_score(
            valores_escalados,
            etiquetas,
            sample_size=tamanio_muestra,
            random_state=RANDOM_STATE,
        )
        resultados.append(
            {"k": k, "inercia": float(modelo.inertia_), "coeficiente_silueta": float(silueta)}
        )
        modelos[k] = modelo

    return pd.DataFrame(resultados), modelos


def seleccionar_k(metricas: pd.DataFrame) -> int:
    """Estima el codo por distancia máxima a la cuerda de la curva de inercia."""
    x = metricas["k"].to_numpy(dtype=float)
    y = metricas["inercia"].to_numpy(dtype=float)
    x_normalizado = (x - x.min()) / (x.max() - x.min())
    rango_y = y.max() - y.min()
    y_normalizado = (y - y.min()) / rango_y if rango_y else np.zeros_like(y)

    inicio = np.array([x_normalizado[0], y_normalizado[0]])
    fin = np.array([x_normalizado[-1], y_normalizado[-1]])
    cuerda = fin - inicio
    puntos = np.column_stack((x_normalizado, y_normalizado))
    distancias = np.abs(
        cuerda[0] * (inicio[1] - puntos[:, 1])
        - (inicio[0] - puntos[:, 0]) * cuerda[1]
    ) / np.linalg.norm(cuerda)
    distancias[[0, -1]] = -1
    return int(metricas.iloc[int(np.argmax(distancias))]["k"])


def diagnosticar_extremos(
    datos: pd.DataFrame, etiquetas: np.ndarray
) -> tuple[pd.DataFrame, list[str]]:
    """Resume valores atípicos IQR y muestra en qué grupo cae cada máximo."""
    salida = []
    mascaras_atipicos = pd.DataFrame(index=datos.index)
    for columna in COLUMNAS_VARIABLES:
        q1 = float(datos[columna].quantile(0.25))
        q3 = float(datos[columna].quantile(0.75))
        rango_iqr = q3 - q1
        limite_inferior = q1 - 1.5 * rango_iqr
        limite_superior = q3 + 1.5 * rango_iqr
        mascara = (datos[columna] < limite_inferior) | (datos[columna] > limite_superior)
        mascaras_atipicos[columna] = mascara
        indice_maximo = datos[columna].idxmax()
        grupo_maximo = int(etiquetas[datos.index.get_loc(indice_maximo)])
        usuario = str(datos.loc[indice_maximo, "user_id"])
        nombre = str(datos.loc[indice_maximo, "user_name"]) if "user_name" in datos else ""
        salida.append(
            {
                "variable": columna,
                "q1": q1,
                "q3": q3,
                "limite_inferior_iqr": limite_inferior,
                "limite_superior_iqr": limite_superior,
                "cantidad_atipicos_iqr": int(mascara.sum()),
                "maximo": float(datos.loc[indice_maximo, columna]),
                "user_id_maximo": usuario,
                "user_name_maximo": nombre,
                "cluster_del_maximo": grupo_maximo,
            }
        )

    cantidad_atipicos_por_usuario = mascaras_atipicos.any(axis=1).to_numpy()
    lineas_grupos = []
    for grupo in sorted(np.unique(etiquetas)):
        seleccion = etiquetas == grupo
        total_grupo = int(seleccion.sum())
        atipicos_grupo = int(cantidad_atipicos_por_usuario[seleccion].sum())
        lineas_grupos.append(
            f"Grupo {grupo}: {atipicos_grupo}/{total_grupo} usuarios con al menos un valor fuera de los límites IQR."
        )
    return pd.DataFrame(salida), lineas_grupos


def guardar_graficas(
    metricas: pd.DataFrame,
    valores_escalados: np.ndarray,
    etiquetas: np.ndarray,
    k_seleccionado: int,
) -> None:
    """Guarda las curvas de selección y una proyección PCA de los grupos."""
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    figura, eje = plt.subplots(figsize=(8, 5))
    eje.plot(metricas["k"], metricas["inercia"], marker="o", color="#176b87")
    eje.set_title("Método del codo para K-Means")
    eje.set_xlabel("Número de grupos (K)")
    eje.set_ylabel("Inercia")
    eje.set_xticks(metricas["k"])
    eje.axvline(k_seleccionado, color="#555555", linestyle="--", label=f"Codo estimado = {k_seleccionado}")
    eje.grid(alpha=0.25)
    eje.legend()
    figura.tight_layout()
    figura.savefig(CARPETA_RESULTADOS / "metodo_del_codo.png", dpi=160, bbox_inches="tight")
    plt.close(figura)

    figura, eje = plt.subplots(figsize=(8, 5))
    eje.plot(
        metricas["k"], metricas["coeficiente_silueta"], marker="o", color="#d2693c"
    )
    eje.axvline(k_seleccionado, color="#555555", linestyle="--", label=f"K elegido = {k_seleccionado}")
    eje.set_title("Coeficiente de silueta por cantidad de grupos")
    eje.set_xlabel("Número de grupos (K)")
    eje.set_ylabel("Coeficiente de silueta")
    eje.set_xticks(metricas["k"])
    eje.grid(alpha=0.25)
    eje.legend()
    figura.tight_layout()
    figura.savefig(
        CARPETA_RESULTADOS / "coeficiente_silueta.png", dpi=160, bbox_inches="tight"
    )
    plt.close(figura)

    proyeccion = PCA(n_components=2).fit_transform(valores_escalados)
    figura, eje = plt.subplots(figsize=(9, 6))
    for grupo in sorted(np.unique(etiquetas)):
        seleccion = etiquetas == grupo
        eje.scatter(
            proyeccion[seleccion, 0],
            proyeccion[seleccion, 1],
            s=18,
            alpha=0.65,
            label=f"Grupo {grupo}",
        )
    eje.set_title(f"Grupos de usuarios con K-Means (K={k_seleccionado})")
    eje.set_xlabel("Componente principal 1")
    eje.set_ylabel("Componente principal 2")
    eje.legend(title="Grupo", loc="best")
    figura.text(
        0.5,
        0.015,
        "Proyección PCA en 2D; los datos originales tienen 7 dimensiones.",
        ha="center",
        fontsize=9,
    )
    figura.tight_layout(rect=(0, 0.04, 1, 1))
    figura.savefig(
        CARPETA_RESULTADOS / "clusters_usuarios.png", dpi=160, bbox_inches="tight"
    )
    plt.close(figura)


def guardar_resultados(
    datos: pd.DataFrame,
    filas_cargadas: int,
    filas_descartadas: int,
    metricas: pd.DataFrame,
    modelo: KMeans,
    escalador: StandardScaler,
    k_seleccionado: int,
    tamanio_muestra_silueta: int,
) -> pd.DataFrame:
    """Escribe métricas, usuarios con grupo, centroides y resumen de ejecución."""
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    etiquetas = modelo.labels_
    usuarios = datos[[columna for columna in ("user_id", "user_name") if columna in datos]].copy()
    usuarios["cluster"] = etiquetas
    usuarios.to_csv(CARPETA_RESULTADOS / "usuarios_agrupados.csv", index=False)

    conteos = pd.Series(etiquetas).value_counts().sort_index()
    centroides_transformados = pd.DataFrame(
        escalador.inverse_transform(modelo.cluster_centers_), columns=COLUMNAS_VARIABLES
    )
    centroides = centroides_transformados.copy()
    centroides[COLUMNAS_LOG1P] = np.expm1(centroides[COLUMNAS_LOG1P]).clip(lower=0)
    centroides.insert(0, "usuarios", [int(conteos.loc[i]) for i in range(k_seleccionado)])
    centroides.insert(0, "cluster", range(k_seleccionado))
    centroides.to_csv(CARPETA_RESULTADOS / "centroides_clusters.csv", index=False)

    filas_metricas = []
    for _, fila in metricas.iterrows():
        filas_metricas.extend(
            [
                {"metrica": "inercia", "k": int(fila["k"]), "grupo": "", "valor": fila["inercia"]},
                {
                    "metrica": "coeficiente_silueta",
                    "k": int(fila["k"]),
                    "grupo": "",
                    "valor": fila["coeficiente_silueta"],
                },
            ]
        )
    filas_metricas.extend(
        {
            "metrica": "usuarios_por_grupo",
            "k": k_seleccionado,
            "grupo": int(grupo),
            "valor": int(cantidad),
        }
        for grupo, cantidad in conteos.items()
    )
    pd.DataFrame(filas_metricas).to_csv(
        CARPETA_RESULTADOS / "metricas_clustering.csv", index=False
    )

    diagnostico, lineas_grupos_extremos = diagnosticar_extremos(datos, etiquetas)
    diagnostico.to_csv(
        CARPETA_RESULTADOS / "diagnostico_valores_extremos.csv", index=False
    )

    silueta_final = float(
        metricas.loc[metricas["k"] == k_seleccionado, "coeficiente_silueta"].iloc[0]
    )
    inercia_final = float(modelo.inertia_)
    resumen = [
        "=== Agrupamiento K-Means: Práctica 7 ===",
        f"Archivo de entrada: {ARCHIVO_ENTRADA.relative_to(BASE_DIR)}",
        f"Registros cargados: {filas_cargadas}",
        f"Usuarios válidos utilizados: {len(datos)}",
        f"Registros descartados o duplicados: {filas_descartadas}",
        f"Variables: {', '.join(COLUMNAS_VARIABLES)}",
        "Variable redundante excluida: avg_interaccion_total, suma de thumbs_up, thumbs_down y reply_count.",
        "Transformación: log1p aplicada a las variables no acotadas; avg_stars queda sin transformar.",
        "Estandarización: StandardScaler ajustado con todos los usuarios válidos tras la transformación.",
        f"K evaluados: {', '.join(str(k) for k in metricas['k'])}",
        f"Muestra usada para silueta: {tamanio_muestra_silueta} usuarios, random_state={RANDOM_STATE}.",
        f"K seleccionado: {k_seleccionado}, estimación geométrica del codo en la curva de inercia.",
        "La silueta se informa como evaluación de separación/cohesión; no se maximiza para escoger K.",
        f"Inercia del modelo final: {inercia_final:.6f}",
        f"Coeficiente de silueta del modelo final: {silueta_final:.6f}",
        "Los números de grupo son etiquetas sin interpretación semántica asignada.",
        "Los centroides se guardan en la escala original de las variables.",
        "La gráfica PCA muestra en 2D datos que originalmente tienen 7 dimensiones.",
        "Diagnóstico de extremos: regla descriptiva de 1.5*IQR sobre variables originales; ningún usuario se elimina por ser atípico.",
        "",
        "Valores atípicos por variable (IQR), máximo y grupo de ese máximo:",
    ]
    resumen.extend(
        f"{fila.variable}: {fila.cantidad_atipicos_iqr} atípicos; máximo={fila.maximo:.6g}; "
        f"usuario={fila.user_id_maximo} ({fila.user_name_maximo}); grupo={fila.cluster_del_maximo}."
        for fila in diagnostico.itertuples(index=False)
    )
    resumen.extend(["", "Usuarios con al menos un valor atípico por grupo:"])
    resumen.extend(lineas_grupos_extremos)
    resumen.extend(
        [
            "",
        "Usuarios por grupo:",
        ]
    )
    resumen.extend(f"Grupo {grupo}: {cantidad}" for grupo, cantidad in conteos.items())
    resumen.extend(["", "Evaluación por K:", metricas.to_string(index=False)])
    (CARPETA_RESULTADOS / "resultados_modelo.txt").write_text(
        "\n".join(resumen) + "\n", encoding="utf-8"
    )
    return centroides


def main() -> None:
    """Ejecuta la preparación, selección, entrenamiento y exportación completa."""
    datos, filas_cargadas, filas_descartadas = cargar_datos()
    valores_escalados, escalador = ajustar_escalador(datos)
    metricas, _ = evaluar_cantidades_grupos(valores_escalados)
    k_seleccionado = seleccionar_k(metricas)
    modelo = KMeans(
        n_clusters=k_seleccionado, random_state=RANDOM_STATE, n_init=10
    ).fit(valores_escalados)
    tamanio_muestra_silueta = min(MAX_MUESTRA_SILUETA, len(datos))

    guardar_graficas(metricas, valores_escalados, modelo.labels_, k_seleccionado)
    centroides = guardar_resultados(
        datos,
        filas_cargadas,
        filas_descartadas,
        metricas,
        modelo,
        escalador,
        k_seleccionado,
        tamanio_muestra_silueta,
    )

    silueta_final = metricas.loc[
        metricas["k"] == k_seleccionado, "coeficiente_silueta"
    ].iloc[0]
    print(f"Usuarios válidos: {len(datos)}; descartados: {filas_descartadas}")
    print(f"K seleccionado por el codo: {k_seleccionado}")
    print(f"Inercia final: {modelo.inertia_:.6f}")
    print(f"Coeficiente de silueta final: {silueta_final:.6f}")
    print("Usuarios por grupo:")
    print(centroides[["cluster", "usuarios"]].to_string(index=False))
    print(f"Resultados guardados en: {CARPETA_RESULTADOS}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, OSError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as error:
        raise SystemExit(f"Error al ejecutar la Práctica 7: {error}") from error