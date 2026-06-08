from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize
import string
import spacy

nlp = spacy.load("es_core_news_sm")

# FUNCIONES

def quitarStopwords_spa(texto):
    espanol = stopwords.words("spanish")
    texto_limpio = [w.lower() for w in texto if w.lower() not in espanol
                    and w not in string.punctuation
                    and w not in ["'s", '|', '--', "''", "``", "-", ".-", "."]]
    return " ".join(texto_limpio)

def lematizar(texto):
    txt = nlp(texto)
    texto_limpio = [token.lemma_ for token in txt]
    return " ".join(texto_limpio)

# FUNCION PIPELINE (Tokenizado -> Stopwords -> Lematizacion)
def procesar(texto):
   tokens = word_tokenize(texto, language='spanish')
   sin_stopwords = quitarStopwords_spa(tokens)
   lematizado = lematizar(sin_stopwords)
   return lematizado

# Funcion que Vectoriza el corpus y devuelve un DataFrame con frecuencias
def obtener_frecuencias(corpus_procesado, ngram_range):
    vectorizer = CountVectorizer(ngram_range=ngram_range, min_df=2)
    X = vectorizer.fit_transform(corpus_procesado)
    df = pd.DataFrame(
        X.sum(axis=0).T,
        index=vectorizer.get_feature_names_out(),
        columns=['freq']
    ).sort_values(by='freq', ascending=True)
    return df

def graficar_ngrams(df_bigrams, df_trigrams):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize= (20, 8))

    df_bigrams.tail(15).plot(kind='barh', title='2-Grams', ax=ax1)
    df_trigrams.tail(15).plot(kind='barh', title='3-Grams', ax=ax2)
    
    plt.tight_layout(w_pad=5)
    plt.show()

# PROGRAMA PRINCIPAL

# Cargar corpus y dividir en párrafos (cada párrafo = un documento)
corpus_raw = PlaintextCorpusReader(".", "CorpusEducacion.txt")
parrafos = [p.strip() for p in corpus_raw.raw().split("\n") if p.strip()]

# Procesar cada parrafo
corpus_procesado = [procesar(p) for p in parrafos]

# Obtener frecuencias para 2-grams y 3-grams
df_bigrams  = obtener_frecuencias(corpus_procesado, ngram_range=(2, 2))
df_trigrams = obtener_frecuencias(corpus_procesado, ngram_range=(3, 3))


# RESULTADOS EN CONSOLA

print("=" * 50)
print("BIGRAMS")
print("=" * 50)
print(df_bigrams.sort_values(by='freq', ascending=False).head(15))

print()

print("=" * 50)
print("TRIGRAMS")
print("=" * 50)
print(df_trigrams.sort_values(by='freq', ascending=False).head(15))

# Grafico

graficar_ngrams(df_bigrams, df_trigrams)