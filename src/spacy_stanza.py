# https://spacy.io/universe/project/spacy-stanza

import stanza
import spacy_stanza

stanza.download("en")
nlp = spacy_stanza.load_pipeline("en")

doc = nlp("Barack Obama was born in Hawaii. He was elected president in 2008.")
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.dep_, token.ent_type_)
print(doc.ents)