from gensim.test.utils import datapath
from gensim import utils
import pandas as pd
class MyCorpus:
    """An iterator that yields sentences (lists of str)."""

    def __iter__(self):
        df = pd.read_csv('/data/Dynamically Generated Hate Dataset v0.2.3.csv')
        for index, row in df.iterrows():
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(row["text"])

import gensim.models

sentences = MyCorpus()
model = gensim.models.Word2Vec(sentences=sentences)
vec_king = model.wv['king']
for index, word in enumerate(model.wv.index_to_key):
    if index == 10:
        break
    print(f"word #{index}/{len(model.wv.index_to_key)} is {word}")

import tempfile

with tempfile.NamedTemporaryFile(prefix='gensim-model-', delete=False) as tmp:
    temporary_filepath = tmp.name
    model.save(temporary_filepath)

    # The model is now safely stored in the filepath.
    # You can copy it to other machines, share it with others, etc.
    #
    # To load a saved model:

    new_model = gensim.models.Word2Vec.load(temporary_filepath)
    df = pd.read_csv('/data/Dynamically Generated Hate Dataset v0.2.3.csv')
    text = df.iloc[0]["text"]
    print(new_model.wv.index_to_key[text[0]])


