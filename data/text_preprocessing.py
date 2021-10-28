"""This file uses various libraries to clean the text."""

from bs4 import BeautifulSoup
import spacy
import unidecode
from word2number import w2n
import contractions

nlp = spacy.load('en_core_web_md')

# words to exclude from the stop word list.
# Words such as "no" and "not" can change drastically the meaning of a sentence.
deselect_stop_words = ['no', 'not']
for w in deselect_stop_words:
    nlp.vocab[w].is_stop = False


def remove_whitespace(text):
    text = text.strip()
    return " ".join(text.split())


def remove_accented_chars(text):
    text = unidecode.unidecode(text)
    return text


def expand_contractions(text):
    text = contractions.fix(text)
    return text


# The method for text preprocessing. Flags allow to turn chosen features on and off.
# By default, all actions are applied.
def text_preprocessing(text, accented_chars=True, contractions=True,
                       convert_num=True, extra_whitespace=True,
                       lemmatization=True, lowercase=True, punctuations=True,
                       remove_num=True, special_chars=True,
                       stop_words=True):
    if extra_whitespace:  # remove extra whitespaces
        text = remove_whitespace(text)
    if accented_chars:  # remove accented characters
        text = remove_accented_chars(text)
    if contractions:  # expand contractions
        text = expand_contractions(text)
    if lowercase:  # convert all characters to lowercase
        text = text.lower()

    doc = nlp(text)  # tokenise text

    clean_text = []

    for token in doc:  # for each token apply the required changes
        flag = True
        edit = token.text
        # remove stop words
        if stop_words and token.is_stop and token.pos_ != 'NUM':
            flag = False
        # remove punctuations
        if punctuations and token.pos_ == 'PUNCT' and flag:
            flag = False
        # remove special characters
        if special_chars and token.pos_ == 'SYM' and flag:
            flag = False
        # remove numbers
        if remove_num and (token.pos_ == 'NUM' or token.text.isnumeric()) and flag:
            flag = False
        # convert number words to numeric numbers
        if convert_num and token.pos_ == 'NUM' and flag:
            edit = w2n.word_to_num(token.text)
        # convert tokens to base form
        elif lemmatization and token.lemma_ != "-PRON-" and flag:
            edit = token.lemma_
        # append tokens edited and not removed to list
        if edit != "" and flag:
            clean_text.append(edit)
    return clean_text
