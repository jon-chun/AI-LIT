# https://spacy.io/universe/project/spacy-cleaner

import spacy
import spacy_cleaner
from spacy_cleaner.processing import removers, replacers, mutators

model = spacy.load("en_core_web_sm")
pipeline = spacy_cleaner.Pipeline(
    model,
    removers.remove_stopword_token,
    replacers.replace_punctuation_token,
    mutators.mutate_lemma_token,
)

texts = ["Hello, my name is Cellan! I love to swim!"]

pipeline.clean(texts)
# ['hello _IS_PUNCT_ Cellan _IS_PUNCT_ love swim _IS_PUNCT_']