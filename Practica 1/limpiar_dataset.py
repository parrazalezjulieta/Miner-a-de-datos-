from html import unescape
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
ARCHIVO_ORIGINAL = BASE_DIR / "datos" / "Recipe Reviews and User Feedback Dataset.csv"
ARCHIVO_LIMPIO = BASE_DIR / "datos" / "Recipe_Reviews_Clean.csv"


def reparar_mojibake(valor):
    """Decodifica entidades HTML y corrige mojibake solo si parece evidente."""
    if not isinstance(valor, str):
        return valor

    reparado = unescape(valor)
    marcadores = ("Ã", "Â", "â", "ð", "�")
    if not any(marcador in reparado for marcador in marcadores):
        return reparado

    try:
        candidato = reparado.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return reparado

    if candidato.count("�") <= reparado.count("�") and candidato != reparado:
        return candidato
    return reparado


def encontrar_columna_indice(dataframe):
    """Devuelve la columna que solo enumera las filas desde cero."""
    for columna in dataframe.columns:
        valores = pd.to_numeric(dataframe[columna], errors="coerce")
        if columna.lower().startswith("unnamed:") and valores.notna().all():
            return columna

    for columna in dataframe.columns:
        valores = pd.to_numeric(dataframe[columna], errors="coerce")
        consecutivo = range(len(dataframe))
        if valores.notna().all() and valores.astype("int64").tolist() == list(consecutivo):
            return columna
    raise ValueError("No se encontró una columna índice consecutiva desde cero.")


def mostrar_faltantes(dataframe, titulo):
    faltantes = dataframe.isna().sum()
    print(f"\n{titulo}")
    print(faltantes[faltantes > 0].to_string() if (faltantes > 0).any() else "No hay valores faltantes.")


def main():
    if not ARCHIVO_ORIGINAL.exists():
        raise FileNotFoundError(f"No existe el archivo original: {ARCHIVO_ORIGINAL}")

    # Lectura inicial: el archivo original nunca se modifica.
    datos = pd.read_csv(ARCHIVO_ORIGINAL)
    print("=== ESTADO INICIAL ===")
    print(f"Filas y columnas: {datos.shape}")
    print("Columnas:")
    print(datos.columns.tolist())
    print("Tipos de datos:")
    print(datos.dtypes)
    mostrar_faltantes(datos, "Valores faltantes antes de eliminar datos")

    columna_indice = encontrar_columna_indice(datos)
    datos = datos.drop(columns=columna_indice)
    print(f"\nColumna de índice eliminada: {columna_indice}")

    if "created_at" not in datos.columns:
        raise KeyError("No existe la columna 'created_at'.")

    timestamps = pd.to_numeric(datos["created_at"], errors="coerce")
    fechas = pd.to_datetime(timestamps, unit="s", errors="coerce")
    fechas_invalidas = fechas.isna().sum()
    if fechas_invalidas:
        raise ValueError(f"created_at contiene {fechas_invalidas} valores no válidos.")
    datos["created_at"] = fechas
    print(f"created_at convertido correctamente: {datos['created_at'].notna().sum()} fechas válidas.")

    columnas_texto = datos.select_dtypes(include=["object", "string"]).columns
    for columna in columnas_texto:
        datos[columna] = datos[columna].map(reparar_mojibake)
    print(f"Columnas de texto limpiadas: {list(columnas_texto)}")

    datos = datos.sort_values("created_at", ascending=True, kind="stable").reset_index(drop=True)
    orden_correcto = datos["created_at"].is_monotonic_increasing
    if not orden_correcto:
        raise ValueError("No se pudo ordenar created_at correctamente.")
    print(f"Orden cronológico ascendente verificado: {orden_correcto}")

    # El ordenamiento usa fechas datetime; este formato es solo la representación del CSV.
    datos["created_at"] = datos["created_at"].dt.strftime("%d/%m/%Y %H:%M")

    print("\n=== ESTADO FINAL ===")
    print(f"Filas y columnas: {datos.shape}")
    print("Columnas:")
    print(datos.columns.tolist())
    print("Tipos de datos:")
    print(datos.dtypes)
    print("\nPrimeros 5 registros:")
    print(datos.head(5).to_string(index=False))
    print("\nÚltimos 5 registros:")
    print(datos.tail(5).to_string(index=False))
    print("\nAlgunos valores de created_at:")
    print(datos["created_at"].head().to_string(index=False))
    print(f"Fechas en orden ascendente: {orden_correcto}")
    mostrar_faltantes(datos, "Valores faltantes después de la limpieza")

    datos.to_csv(ARCHIVO_LIMPIO, index=False, encoding="utf-8-sig")
    print(f"\nArchivo limpio guardado en: {ARCHIVO_LIMPIO}")


if __name__ == "__main__":
    main()