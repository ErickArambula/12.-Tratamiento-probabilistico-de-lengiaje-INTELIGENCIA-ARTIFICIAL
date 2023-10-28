import nltk
from nltk import bigrams, FreqDist
from nltk.corpus import reuters

# Descargar el corpus de Reuters (si aún no lo tienes)
nltk.download('reuters')

# Obtener un subconjunto de documentos de Reuters
documents = reuters.fileids()[:100]

# Construir un corpus de palabras
words = reuters.words(documents)

# Crear bigramas (pares de palabras consecutivas)
bi_grams = list(bigrams(words))

# Calcular la frecuencia de los bigramas
bi_gram_freq = FreqDist(bi_grams)

# Calcular la probabilidad condicionada de un bigrama dado
word1 = "oil"
word2 = "prices"
probability = bi_gram_freq[(word1, word2)] / bi_gram_freq[word1]

print(f"La probabilidad de '{word2}' dado '{word1}' es: {probability:.4f}")
