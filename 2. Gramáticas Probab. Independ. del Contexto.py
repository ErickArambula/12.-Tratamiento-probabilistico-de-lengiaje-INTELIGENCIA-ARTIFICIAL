import nltk
from nltk import CFG, PCFG, ViterbiParser

# Definir una gramática probabilística
grammar = PCFG.fromstring("""
    S -> NP VP [1.0]
    NP -> Det N [0.7] | Det N PP [0.3]
    VP -> V [0.4] | V NP [0.4] | V NP PP [0.2]
    PP -> P NP [1.0]
    Det -> 'the' [1.0]
    N -> 'cat' [0.4] | 'dog' [0.6]
    V -> 'chased' [0.7] | 'ate' [0.3]
    P -> 'with' [1.0]
    """)

# Crear un analizador Viterbi para la gramática
parser = ViterbiParser(grammar)

# Analizar una oración
sentence = "the cat chased the dog with the bone".split()
for tree in parser.parse(sentence):
    tree.pretty_print()
