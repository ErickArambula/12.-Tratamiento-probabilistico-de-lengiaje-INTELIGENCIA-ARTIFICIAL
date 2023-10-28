import nltk
from nltk.translate import AlignmentModel, IBMModel1
from nltk.corpus import comtrans

# Descargar el corpus paralelo para inglés y español (si aún no lo tienes)
nltk.download('comtrans')

# Obtener oraciones alineadas en inglés y español
aligned_sentences = comtrans.aligned_sents()

# Dividir las oraciones en inglés y español
eng_sents = [" ".join(a.words) for a in aligned_sentences]
esp_sents = [" ".join(a.mots) for a in aligned_sentences]

# Crear un modelo de alineación
alignment_model = AlignmentModel(IBMModel1, eng_sents, esp_sents)

# Traducir una oración del inglés al español
english_sentence = "This is a simple example."
translation = alignment_model.translate(english_sentence.split())
print("Oración en inglés:", english_sentence)
print("Traducción al español:", " ".join(translation))
