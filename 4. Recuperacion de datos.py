from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Corpus de documentos
corpus = [
    "Este es un documento de ejemplo.",
    "Este documento es diferente.",
    "Otro documento en el corpus.",
    "Un documento más para probar.",
]

# Consulta del usuario
query = "documento de ejemplo"

# Crear un vectorizador TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# Vectorizar la consulta del usuario
query_vector = vectorizer.transform([query])

# Calcular similitud coseno entre la consulta y los documentos
cosine_scores = cosine_similarity(query_vector, X)

# Obtener los documentos más relevantes
most_similar_doc = cosine_scores.argsort()[0][::-1]

print("Documentos más relevantes para la consulta:")
for i, doc_idx in enumerate(most_similar_doc):
    print(f"{i + 1}. {corpus[doc_idx]}")
