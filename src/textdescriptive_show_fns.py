import time
import json
import pandas as pd

import spacy

import textdescriptives as td

SPACY_FLAG = True

# Instantiate a DataFrame
df = pd.DataFrame()


if SPACY_FLAG is False:

    td.get_valid_metrics()
    # {'quality', 'readability', 'all', 'descriptive_stats', 'dependency_distance', 'pos_proportions', 'information_theory', 'coherence'}

    text = "The world is changed. I feel it in the water. I feel it in the earth. I smell it in the air. Much that once was is lost, for none now live who remember it."
    # will automatically download the relevant model (´en_core_web_lg´) and extract all metrics


    df = td.extract_metrics(text=text, lang="en", metrics=None)

    # specify spaCy model and which metrics to extract
    df = td.extract_metrics(text=text, spacy_model="en_core_web_lg", metrics=["readability", "coherence"])

    print(df)

else:

    # load your favourite spacy model (remember to install it first using e.g. `python -m spacy download en_core_web_sm`)
    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("textdescriptives/all") 
    doc = nlp("The world is changed. I feel it in the water. I feel it in the earth. I smell it in the air. Much that once was is lost, for none now live who remember it.")

    # access some of the values
    readability_dt = doc._.readability  
    token_len_dt = doc._.token_length

    print("\n\ndoc._.readability =====")
    print(json.dumps(doc._.readability, indent=4))
    print("\n\ndoc._.token_length =====")
    print(json.dumps(doc._.token_length, indent=4))


df = td.extract_df(doc, metrics = ["descriptive_stats", "readability", "dependency_distance", "pos_proportions", "coherence", "quality", "information_theory"])
# print(df)
# Print DataFrame in vertical style with fixed-width formatting
max_col_width = max(len(col) for col in df.columns)  # Find the maximum column name length

for col in df.columns:
    print(f"{col:<{max_col_width}} : {df.at[0, col]}")