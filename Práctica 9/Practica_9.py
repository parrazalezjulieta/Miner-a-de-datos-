from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from wordcloud import STOPWORDS, WordCloud


BASE_DIR = Path(__file__).resolve().parent
ARCHIVO_ENTRADA = BASE_DIR / "datos" / "Recipe_Reviews_Clean.csv"
CARPETA_RESULTADOS = BASE_DIR / "datos" / "resultados_practica_9"
ARCHIVOS_PALABRAS = re.compile(r"[a-záéíóúüñ]+(?:['’][a-záéíóúüñ]+)?")
STOPWORDS_ESPANOL = set(
    "a al algo algunas algunos ante antes como con contra cual cuando de del desde "
    " donde dos el ella ellas ellos en entre era eres es esa esas ese eso esos esta "
    "estaba estaban estado estamos estan estar estas este esto estos estoy fue fueron "
    "ha habia han hasta hay la las le les lo los mas me mi mientras mismo mucha muchas "
    "mucho muchos muy nada ni no nos nuestra nuestras nuestro nuestros o otra otras "
    "otro otros para pero poco por porque que quien quienes se sea ser si sido sin sobre "
    "son su sus te tiene tienen todo todos tu tus un una unas uno unos y ya yo"
    .split()
)
STOPWORDS_TEXTO = STOPWORDS | ENGLISH_STOP_WORDS | STOPWORDS_ESPANOL


class ExtractorTexto(HTMLParser):
    """Extrae el texto visible de comentarios que contienen etiquetas HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.fragmentos: list[str] = []

    def handle_data(self, data: str) -> None:
        self.fragmentos.append(data)


def extraer_texto_visible(comentario: str) -> str:
    extractor = ExtractorTexto()
    extractor.feed(comentario)
    return " ".join(extractor.fragmentos)


def limpiar_comentario(comentario: str) -> list[str]:
    palabras = ARCHIVOS_PALABRAS.findall(comentario.lower())
    return [palabra for palabra in palabras if palabra not in STOPWORDS_TEXTO]


def main() -> None:
    if not ARCHIVO_ENTRADA.is_file():
        raise FileNotFoundError(f"No se encontro el archivo de entrada: {ARCHIVO_ENTRADA}")

    datos = pd.read_csv(ARCHIVO_ENTRADA, usecols=["text"], dtype={"text": "string"})
    conteo_palabras: Counter[str] = Counter()
    comentarios_analizados = 0

    for comentario in datos["text"].dropna():
        texto_visible = extraer_texto_visible(str(comentario)).strip()
        if not texto_visible:
            continue

        comentarios_analizados += 1
        conteo_palabras.update(limpiar_comentario(texto_visible))

    if not conteo_palabras:
        raise ValueError("No quedaron palabras para analizar despues de limpiar los comentarios.")

    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    total_palabras = sum(conteo_palabras.values())
    palabras_diferentes = len(conteo_palabras)
    promedio_palabras = total_palabras / comentarios_analizados
    palabras_frecuentes = conteo_palabras.most_common(15)

    pd.DataFrame(palabras_frecuentes, columns=["palabra", "frecuencia"]).to_csv(
        CARPETA_RESULTADOS / "frecuencia_palabras.csv", index=False, encoding="utf-8"
    )

    nube = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        max_words=150,
        collocations=False,
        random_state=42,
        colormap="viridis",
    ).generate_from_frequencies(conteo_palabras)
    figura, eje = plt.subplots(figsize=(12, 7))
    eje.imshow(nube, interpolation="bilinear")
    eje.axis("off")
    figura.tight_layout(pad=0)
    figura.savefig(CARPETA_RESULTADOS / "nube_palabras.png", dpi=160, bbox_inches="tight")
    plt.close(figura)

    principales = ", ".join(
        f"{palabra} ({frecuencia})" for palabra, frecuencia in palabras_frecuentes[:5]
    )
    texto = [
        "PRACTICA 9: ANALISIS DE TEXTO",
        "Fuente: Recipe_Reviews_Clean.csv, columna text. El archivo fuente no se modifico.",
        "Se extrajo texto visible, se convirtio a minusculas y se excluyeron stopwords en espanol e ingles.",
        "Los comentarios vacios se excluyeron; las metricas de palabras cuentan tokens tras quitar stopwords.",
        "",
        f"Comentarios analizados: {comentarios_analizados}",
        f"Cantidad total de palabras: {total_palabras}",
        f"Numero de palabras diferentes: {palabras_diferentes}",
        f"Promedio de palabras por comentario: {promedio_palabras:.2f}",
        "",
        "Las 15 palabras mas frecuentes:",
        *[f"{indice}. {palabra}: {frecuencia}" for indice, (palabra, frecuencia) in enumerate(palabras_frecuentes, start=1)],
        "",
        "Interpretacion:",
        f"Destacan {principales}. Su frecuencia sugiere que estos temas o descripciones aparecen repetidamente en las opiniones sobre las recetas.",
        "Los terminos favorables pueden reflejar satisfaccion con el sabor o la preparacion; menciones a ingredientes, tiempo o cambios pueden apuntar a preferencias y sugerencias.",
        "La frecuencia no determina por si sola si una opinion es positiva o negativa: el contexto de cada comentario es necesario para interpretar el sentimiento.",
    ]
    (CARPETA_RESULTADOS / "analisis_texto.txt").write_text(
        "\n".join(texto) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()