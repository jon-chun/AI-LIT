from pysbd.utils import PySBDFactory

nlp = spacy.blank('en')
# Caution: works with spaCy<=2.x.x
nlp.add_pipe(PySBDFactory(nlp))

doc = nlp('My name is Jonas E. Smith. Please turn to p. 55.')
print(list(doc.sents))
# [My name is Jonas E. Smith., Please turn to p. 55.]