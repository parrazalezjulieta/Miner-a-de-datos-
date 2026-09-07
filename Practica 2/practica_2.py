from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt




BASE_DIR = Path(__file__).resolve().parent
CARPETA_GRAFICAS = BASE_DIR / "datos" / "graficas"
ARCHIVO_DATOS = BASE_DIR / "datos" / "Recipe_Reviews_Clean.csv"
ARCHIVO_USUARIOS = BASE_DIR / "datos" / "resumen_usuarios.csv"
ARCHIVO_RECETAS = BASE_DIR / "datos" / "resumen_recetas.csv"
ARCHIVO_EXCEL = BASE_DIR / "datos" / "resultados_practica_2.xlsx"


def load_data(file_name: str) -> pd.DataFrame:
    """Carga el CSV y devuelve un DataFrame nuevo en memoria."""
    return pd.read_csv(file_name)


def show_initial_information(df: pd.DataFrame) -> None:
    """Mostrar la estructura básica y los valores faltantes del dataset."""
    print("=== INFORMACIÓN GENERAL DEL DATASET ===")
    print(f"Filas: {df.shape[0]}")
    print(f"Columnas: {df.shape[1]}")
    print("\nNombres de las columnas:")
    print(df.columns.tolist())
    print("\nTipos de datos:")
    print(df.dtypes)
    print("\nValores faltantes por columna:")
    print(df.isna().sum())
    print("\nPrimeras filas:")
    print(df.head().to_string(index=False))


def identify_entities(df: pd.DataFrame) -> None:
    """Mostrar entidades conceptuales y sus columnas."""
    entidades = {
        "USUARIO": ["user_id", "user_name", "user_reputation"],
        "RESEÑA": [
            "comment_id",
            "created_at",
            "stars",
            "reply_count",
            "thumbs_up",
            "thumbs_down",
            "best_score",
            "text",
        ],
        "RECETA": ["recipe_code", "recipe_number", "recipe_name"],
    }

    print("\n=== ENTIDADES ===")
    for entidad, columnas in entidades.items():
        columnas_presentes = [columna for columna in columnas if columna in df.columns]
        print(f"{entidad}: {columnas_presentes}")


def identify_relationships(df: pd.DataFrame) -> None:
    """Mostrar identificadores y conteos que representan las relaciones."""
    print("\n=== RELACIONES ===")
    print("USUARIO (user_id) -> RESEÑA (comment_id) -> RECETA (recipe_code)")
    print(f"Usuarios únicos: {df['user_id'].nunique()}")
    print(f"Reseñas únicas: {df['comment_id'].nunique()}")
    print(f"Recetas únicas: {df['recipe_code'].nunique()}")
    print("\nCantidad de reseñas por usuario:")
    print(df.groupby("user_id").size().rename("num_reviews").to_string())
    print("\nCantidad de reseñas por receta:")
    print(df.groupby("recipe_code").size().rename("num_reviews").to_string())


def create_derived_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Crear interaccion_total en una copia."""
    resultado = df.copy()
    resultado["interaccion_total"] = (
        resultado["thumbs_up"] + resultado["thumbs_down"] + resultado["reply_count"]
    )
    return resultado


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Calcular medidas descriptivas para las variables."""
    variables = [
        "user_reputation",
        "stars",
        "thumbs_up",
        "thumbs_down",
        "reply_count",
        "best_score",
        "interaccion_total",
    ]
    resultados = []

    for variable in variables:
        serie = df[variable].dropna()
        resultados.append(
            {
                "variable": variable,
                "count": serie.count(),
                "sum": serie.sum(),
                "mean": serie.mean(),
                "median": serie.median(),
                "mode": serie.mode().iloc[0] if not serie.mode().empty else pd.NA,
                "minimum": serie.min(),
                "maximum": serie.max(),
                "range": serie.max() - serie.min(),
                "variance": serie.var(),
                "standard_deviation": serie.std(),
                "Q1": serie.quantile(0.25),
                "Q3": serie.quantile(0.75),
                "skewness": serie.skew(),
                "kurtosis": serie.kurtosis(),
            }
        )

    tabla = pd.DataFrame(resultados).set_index("variable")
    print("\n=== ESTADÍSTICA DESCRIPTIVA ===")
    print(tabla.to_string())
    print("\nReferencia: df.describe()")
    print(df[variables].describe().to_string())
    return tabla


def analyze_stars(df: pd.DataFrame) -> None:
    """Mostrar la distribución de stars, y conservar los registros con cero."""
    total = len(df)
    cantidad_ceros = (df["stars"] == 0).sum()
    porcentaje_ceros = cantidad_ceros / total * 100 if total else 0

    print("\n=== ANÁLISIS DE STARS ===")
    print(f"Registros con stars = 0: {cantidad_ceros}")
    print(f"Porcentaje con stars = 0: {porcentaje_ceros:.2f}%")

    distribucion = df["stars"].value_counts(dropna=False).sort_index().rename("cantidad").to_frame()
    distribucion["porcentaje"] = distribucion["cantidad"] / total * 100 if total else 0
    print("\nCantidad y porcentaje por valor de stars:")
    print(distribucion.to_string())
    moda = df["stars"].mode().iloc[0]
    print(f"\nModa de stars: {moda}")
    resumen = pd.DataFrame(
        {
            "metrica": [
                "cantidad_stars_0",
                "porcentaje_stars_0",
                "moda_stars",
            ],
            "valor": [cantidad_ceros, porcentaje_ceros, moda],
        }
    )
    return resumen, distribucion.reset_index()


def analysis_by_user(df: pd.DataFrame) -> pd.DataFrame:
    """Calcular métricas por usuario y guarda el resumen completo."""
    resumen = (
        df.groupby("user_id", as_index=False)
        .agg(
            user_name=("user_name", "first"),
            user_reputation=("user_reputation", "first"),
            num_reviews=("comment_id", "count"),
            avg_stars=("stars", "mean"),
            avg_thumbs_up=("thumbs_up", "mean"),
            avg_thumbs_down=("thumbs_down", "mean"),
            avg_reply_count=("reply_count", "mean"),
            avg_best_score=("best_score", "mean"),
            avg_interaccion_total=("interaccion_total", "mean"),
        )
        .sort_values("num_reviews", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    guardar_sin_sobrescribir(resumen, ARCHIVO_USUARIOS)
    print("\n=== MÉTRICAS AGRUPADAS POR USUARIO ===")
    print(f"Usuarios únicos: {len(resumen)}")
    print("Primeros 10 usuarios:")
    print(resumen.head(10).to_string(index=False))
    print("Usuario(s) con mayor cantidad de reseñas:")
    print(resumen[resumen["num_reviews"] == resumen["num_reviews"].max()].to_string(index=False))
    print(f"Tabla completa guardada en: {ARCHIVO_USUARIOS}")
    return resumen


def analysis_by_reputation(df: pd.DataFrame) -> pd.DataFrame:
    """Crear grupos de reputación."""
    resumen_usuarios = pd.read_csv(ARCHIVO_USUARIOS)

    print("\n=== DISTRIBUCIÓN DE USER_REPUTATION ===")
    print(resumen_usuarios["user_reputation"].describe().to_string())
    print("\nValores de user_reputation:")
    print(resumen_usuarios["user_reputation"].value_counts().sort_index().to_string())

    condiciones = [
        resumen_usuarios["user_reputation"] == 0,
        resumen_usuarios["user_reputation"] == 1,
        resumen_usuarios["user_reputation"].between(10, 20),
        resumen_usuarios["user_reputation"].between(30, 510),
    ]
    grupos = ["Grupo 1", "Grupo 2", "Grupo 3", "Grupo 4"]
    rangos = ["0", "1", "10-20", "30-510"]
    resumen_usuarios["grupo_reputacion"] = pd.Series(
        pd.NA, index=resumen_usuarios.index, dtype="string"
    )
    resumen_usuarios["rango_reputacion"] = pd.Series(
        pd.NA, index=resumen_usuarios.index, dtype="string"
    )
    for condicion, grupo, rango in zip(condiciones, grupos, rangos):
        resumen_usuarios.loc[condicion, "grupo_reputacion"] = grupo
        resumen_usuarios.loc[condicion, "rango_reputacion"] = rango

    if resumen_usuarios["grupo_reputacion"].isna().any():
        raise ValueError("Hay usuarios sin grupo de reputación asignado.")

    print("\nGrupos de user_reputation:")
    print(pd.DataFrame({"grupo_reputacion": grupos, "rango_reputacion": rangos}).to_string(index=False))

    resumen = (
        resumen_usuarios.groupby("grupo_reputacion", observed=False)
        .agg(
            cantidad_usuarios=("user_id", "count"),
            cantidad_resenas=("num_reviews", "sum"),
            promedio_stars=("avg_stars", "mean"),
            promedio_thumbs_up=("avg_thumbs_up", "mean"),
            promedio_thumbs_down=("avg_thumbs_down", "mean"),
            promedio_reply_count=("avg_reply_count", "mean"),
            promedio_best_score=("avg_best_score", "mean"),
            promedio_interaccion_total=("avg_interaccion_total", "mean"),
        )
        .reset_index()
    )
    resumen["rango_reputacion"] = resumen["grupo_reputacion"].map(
        dict(zip(grupos, rangos))
    )
    resumen = resumen[
        [
            "grupo_reputacion",
            "rango_reputacion",
            "cantidad_usuarios",
            "cantidad_resenas",
            "promedio_stars",
            "promedio_thumbs_up",
            "promedio_thumbs_down",
            "promedio_reply_count",
            "promedio_best_score",
            "promedio_interaccion_total",
        ]
    ]
    print("\n=== MÉTRICAS AGRUPADAS POR REPUTACIÓN ===")
    print(resumen.to_string(index=False))
    return resumen


def analysis_by_recipe(df: pd.DataFrame) -> pd.DataFrame:
    """Calcular métricas por receta y guardar el resumen completo."""
    resumen = (
        df.groupby(["recipe_code", "recipe_name"], as_index=False)
        .agg(
            num_reviews=("comment_id", "count"),
            avg_stars=("stars", "mean"),
            avg_thumbs_up=("thumbs_up", "mean"),
            avg_thumbs_down=("thumbs_down", "mean"),
            avg_reply_count=("reply_count", "mean"),
            avg_best_score=("best_score", "mean"),
            avg_interaccion_total=("interaccion_total", "mean"),
        )
        .sort_values("num_reviews", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    guardar_sin_sobrescribir(resumen, ARCHIVO_RECETAS)
    print("\n=== MÉTRICAS AGRUPADAS POR RECETA ===")
    print("Primeras 10 recetas:")
    print(resumen.head(10).to_string(index=False))
    print(f"Tabla completa guardada en: {ARCHIVO_RECETAS}")
    return resumen


def analyze_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Muestra métricas temporales básicas sin hacer análisis de series de tiempo."""
    fechas = pd.to_datetime(df["created_at"], format="%d/%m/%Y %H:%M", errors="raise")
    print("\n=== MÉTRICAS TEMPORALES BÁSICAS ===")
    print(f"Fecha mínima: {fechas.min()}")
    print(f"Fecha máxima: {fechas.max()}")
    print(f"Cantidad de fechas diferentes: {fechas.nunique()}")
    return pd.DataFrame(
        {
            "metrica": ["fecha_minima", "fecha_maxima", "cantidad_fechas_diferentes"],
            "valor": [fechas.min(), fechas.max(), fechas.nunique()],
        }
    )


def guardar_resultados_excel(
    df_inicial: pd.DataFrame,
    df: pd.DataFrame,
    estadisticas: pd.DataFrame,
    resumen_stars: pd.DataFrame,
    distribucion_stars: pd.DataFrame,
    resumen_usuarios: pd.DataFrame,
    resumen_reputacion: pd.DataFrame,
    resumen_recetas: pd.DataFrame,
    resumen_fechas: pd.DataFrame,
) -> None:
    """Guardar los resultados actuales."""
    info_general = pd.DataFrame(
        {
            "elemento": ["cantidad_filas", "cantidad_columnas", "nombres_columnas"],
            "valor": [
                df_inicial.shape[0],
                df_inicial.shape[1],
                ", ".join(df_inicial.columns),
            ],
        }
    )
    info_columnas = pd.DataFrame(
        {
            "columna": df_inicial.columns,
            "tipo_dato": df_inicial.dtypes.astype(str).values,
            "valores_faltantes": df_inicial.isna().sum().values,
        }
    )
    variables = estadisticas.index.tolist()
    estadisticas_excel = estadisticas.reset_index().rename(
        columns={
            "variable": "variable",
            "minimum": "min",
            "maximum": "max",
            "standard_deviation": "std",
        }
    )
    describe_excel = df[variables].describe().reset_index()

    with pd.ExcelWriter(ARCHIVO_EXCEL, engine="openpyxl", mode="w") as escritor:
        info_general.to_excel(escritor, sheet_name="Info_Inicial", index=False)
        fila_columnas = len(info_general) + 3
        info_columnas.to_excel(
            escritor, sheet_name="Info_Inicial", index=False, startrow=fila_columnas
        )
        fila_primeras = fila_columnas + len(info_columnas) + 3
        df_inicial.head().to_excel(
            escritor, sheet_name="Info_Inicial", index=False, startrow=fila_primeras
        )

        estadisticas_excel.to_excel(
            escritor, sheet_name="Estadisticas_Descriptivas", index=False
        )
        describe_excel.to_excel(
            escritor,
            sheet_name="Estadisticas_Descriptivas",
            index=False,
            startrow=len(estadisticas_excel) + 3,
        )

        resumen_stars.to_excel(escritor, sheet_name="Analisis_Stars", index=False)
        distribucion_stars.to_excel(
            escritor,
            sheet_name="Analisis_Stars",
            index=False,
            startrow=len(resumen_stars) + 3,
        )
        resumen_usuarios.to_excel(escritor, sheet_name="Por_Usuario", index=False)
        resumen_reputacion.to_excel(escritor, sheet_name="Por_Reputacion", index=False)
        resumen_recetas.to_excel(escritor, sheet_name="Por_Receta", index=False)
        resumen_fechas.to_excel(escritor, sheet_name="Fechas", index=False)

    print(f"\nArchivo Excel creado correctamente: {ARCHIVO_EXCEL}")


def guardar_sin_sobrescribir(df: pd.DataFrame, ruta: Path) -> None:
    
    if ruta.exists():
        print(f"El archivo ya existe y no será sobrescrito: {ruta}")
        return
    df.to_csv(ruta, index=False, encoding="utf-8-sig")


def crear_carpeta_graficas() -> None:
    """Crea la carpeta de gráficas si no existe."""
    CARPETA_GRAFICAS.mkdir(parents=True, exist_ok=True)


def grafica_linea_estadisticas(estadisticas: pd.DataFrame) -> None:
    """Generar gráfica lineal para variables de estadísticas descriptivas."""
    variables = estadisticas.index.tolist()
    metricas = ["mean", "median"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, metrica in enumerate(metricas):
        valores = estadisticas[metrica].values
        x_pos = range(len(variables))
        
        axes[idx].plot(x_pos, valores, marker="o", linewidth=2, markersize=6)
        axes[idx].set_xticks(x_pos)
        axes[idx].set_xticklabels(variables, rotation=45, ha="right")
        axes[idx].set_xlabel("Variable")
        axes[idx].set_ylabel(metrica.capitalize())
        axes[idx].set_title(f"Estadísticas Descriptivas - {metrica.capitalize()}")
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    archivo = CARPETA_GRAFICAS / "linea_estadisticas_descriptivas.png"
    plt.savefig(archivo, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Gráfica guardada: {archivo}")


def configurar_etiquetas_intervalos(ax, cantidad: int, nombre: str, max_intervalos: int = 5) -> None:

    paso = max(1, (cantidad + max_intervalos - 1) // max_intervalos)
    posiciones = list(range(0, cantidad, paso))
    etiquetas = []
    for inicio in posiciones:
        fin = min(inicio + paso, cantidad)
        etiquetas.append(f"{nombre} {inicio + 1}-{fin}")
    ax.set_xticks(posiciones)
    ax.set_xticklabels(etiquetas, rotation=45, ha="right")


def grafica_linea_por_usuario(resumen_usuarios: pd.DataFrame) -> None:
    
    metricas = [
        ("num_reviews", "Cantidad de reseñas por usuario"),
        ("avg_stars", "Promedio de estrellas por usuario"),
        ("avg_interaccion_total", "Promedio de interacción total por usuario"),
    ]
    
    # Generar gráficas individuales
    nombres_archivos = [
        "linea_resenas_por_usuario.png",
        "linea_estrellas_por_usuario.png",
        "linea_interaccion_por_usuario.png",
    ]
    
    for idx, (metrica, titulo) in enumerate(metricas):
        fig, ax = plt.subplots(figsize=(12, 5))
        valores = resumen_usuarios[metrica].values
        x_pos = range(len(valores))
        
        ax.plot(x_pos, valores, linewidth=1.5, markersize=3, marker=".")
        configurar_etiquetas_intervalos(ax, len(valores), "Usuarios")
        ax.set_xlabel("Usuarios")
        ax.set_ylabel(metrica.replace("avg_", "").replace("num_", "").replace("_", " ").capitalize())
        ax.set_title(titulo)
        ax.grid(True, alpha=0.3)
        
        archivo = CARPETA_GRAFICAS / nombres_archivos[idx]
        plt.savefig(archivo, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"Gráfica guardada: {archivo}")


def grafica_linea_por_receta(resumen_recetas: pd.DataFrame) -> None:
    """Generar gráficas lineales por receta con intervalos en el eje X."""
    metricas = [
        ("num_reviews", "Cantidad de reseñas por receta"),
        ("avg_stars", "Promedio de estrellas por receta"),
        ("avg_interaccion_total", "Promedio de interacción total por receta"),
    ]
    
    nombres_archivos = [
        "linea_resenas_por_receta.png",
        "linea_estrellas_por_receta.png",
        "linea_interaccion_por_receta.png",
    ]
    
    for idx, (metrica, titulo) in enumerate(metricas):
        fig, ax = plt.subplots(figsize=(12, 5))
        valores = resumen_recetas[metrica].values
        x_pos = range(len(valores))
        
        ax.plot(x_pos, valores, linewidth=1.5, markersize=4, marker=".")
        configurar_etiquetas_intervalos(ax, len(valores), "Recetas")
        ax.set_xlabel("Recetas")
        ax.set_ylabel(metrica.replace("avg_", "").replace("num_", "").replace("_", " ").capitalize())
        ax.set_title(titulo)
        ax.grid(True, alpha=0.3)

        archivo = CARPETA_GRAFICAS / nombres_archivos[idx]
        plt.savefig(archivo, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"Gráfica guardada: {archivo}")


def grafica_barras_reputacion(resumen_reputacion: pd.DataFrame) -> None:
    """Genera gráficas de barras para los intervalos de reputación."""
    rangos = resumen_reputacion["rango_reputacion"].tolist()
    etiquetas_x = [f"Reputación {r}" for r in rangos]
    
    # Primera gráfica: cantidad de usuarios
    fig, ax = plt.subplots(figsize=(10, 5))
    valores = resumen_reputacion["cantidad_usuarios"].values
    ax.bar(etiquetas_x, valores, color="steelblue", edgecolor="black")
    ax.set_xlabel("Intervalo de Reputación")
    ax.set_ylabel("Cantidad de usuarios")
    ax.set_title("Usuarios por intervalo de reputación")
    ax.grid(True, alpha=0.3, axis="y")
    
    # Agregar valores en las barras
    for i, v in enumerate(valores):
        ax.text(i, v, str(int(v)), ha="center", va="bottom")
    
    plt.tight_layout()
    archivo = CARPETA_GRAFICAS / "barras_reputacion_usuarios.png"
    plt.savefig(archivo, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Gráfica guardada: {archivo}")
    
    # Segunda gráfica promedio de interacción total
    fig, ax = plt.subplots(figsize=(10, 5))
    valores = resumen_reputacion["promedio_interaccion_total"].values
    ax.bar(etiquetas_x, valores, color="coral", edgecolor="black")
    ax.set_xlabel("Intervalo de Reputación")
    ax.set_ylabel("Promedio de interacción total")
    ax.set_title("Promedio de interacción total por intervalo de reputación")
    ax.grid(True, alpha=0.3, axis="y")
    
    # Agregar valores en las barras
    for i, v in enumerate(valores):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    
    plt.tight_layout()
    archivo = CARPETA_GRAFICAS / "barras_reputacion_interaccion.png"
    plt.savefig(archivo, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Gráfica guardada: {archivo}")


def generar_graficas(
    estadisticas: pd.DataFrame,
    resumen_usuarios: pd.DataFrame,
    resumen_reputacion: pd.DataFrame,
    resumen_recetas: pd.DataFrame,
) -> None:
    """Genera todas las gráficas y las guarda en la carpeta."""
    crear_carpeta_graficas()
    print("\n=== GENERANDO GRÁFICAS ===")
    grafica_linea_estadisticas(estadisticas)
    grafica_linea_por_usuario(resumen_usuarios)
    grafica_linea_por_receta(resumen_recetas)
    grafica_barras_reputacion(resumen_reputacion)
    print("Todas las gráficas han sido generadas correctamente.\n")



def main() -> None:
    """Ejecuta la Práctica 2 en un orden lógico."""
    df_inicial = load_data(str(ARCHIVO_DATOS))
    show_initial_information(df_inicial)
    identify_entities(df_inicial)
    identify_relationships(df_inicial)
    df = create_derived_variables(df_inicial)
    estadisticas = descriptive_statistics(df)
    resumen_stars, distribucion_stars = analyze_stars(df)
    resumen_usuarios = analysis_by_user(df)
    resumen_reputacion = analysis_by_reputation(df)
    resumen_recetas = analysis_by_recipe(df)
    resumen_fechas = analyze_dates(df)
    guardar_resultados_excel(
        df_inicial,
        df,
        estadisticas,
        resumen_stars,
        distribucion_stars,
        resumen_usuarios,
        resumen_reputacion,
        resumen_recetas,
        resumen_fechas,
    )
    generar_graficas(estadisticas, resumen_usuarios, resumen_reputacion, resumen_recetas)


if __name__ == "__main__":
    main()
