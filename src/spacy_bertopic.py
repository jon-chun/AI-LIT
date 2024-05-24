import spacy
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
nlp = spacy.load('en_core_web_md', exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])

topic_model = BERTopic(embedding_model=nlp)
topics, probs = topic_model.fit_transform(docs)

print(f"TOPICS:\n{topics}")

print(f"PROBS:\n{probs}")

print("\n\n")

fig = topic_model.visualize_topics()
fig.show()