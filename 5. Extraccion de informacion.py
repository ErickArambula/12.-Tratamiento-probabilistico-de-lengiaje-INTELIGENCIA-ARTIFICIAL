import spacy

# Cargar el modelo en inglés
nlp = spacy.load("en_core_web_sm")

# Texto de ejemplo
text = "El 5 de noviembre de 2020, OpenAI anunció el lanzamiento de GPT-3, un modelo de lenguaje avanzado."

# Procesar el texto con spaCy
doc = nlp(text)

# Extraer entidades con etiquetas
entities = [(ent.text, ent.label_) for ent in doc.ents]

# Imprimir las entidades encontradas
for entity, label in entities:
    print(f"Entidad: {entity}, Etiqueta: {label}")
