from collections import Counter

import numpy as np
import pandas as pd


__all__ = ['get_word_vector']


def __extract_terms(series:pd.Series) -> list:
    terms = list()
    for row in series.values:
        row = row.split(',')
        terms.append([t.strip() for t in row])
    return terms


def __list_flatten(x):
    flatten = list()
    for i in x:
        if isinstance(i, list):
            flatten += __list_flatten(i)
        else:
            flatten.append(i)
    return flatten


def get_word_vector(series:pd.Series, length:int,
                    pad_term:str) -> np.array:
    lines = __extract_terms(series)
    count = Counter(__list_flatten(lines))
    unique_terms = [t[0] for t in count.most_common()]
    # build vocaburary dictionary
    vocab = dict()
    for i, v in enumerate(unique_terms):
        if v != pad_term:
            vocab[v] = i + 1
    vocab[pad_term] = 0 # pad term is 0 in dict.
    # build word vector
    word_vectors = list()
    for line in lines:
        # truncation
        line = line[:length]
        # padding word
        while len(line) < length:
            line.append(pad_term)
        # get word vector
        vec = [vocab[word] for word in line]
        word_vectors.append(vec)
    return word_vectors