from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.matcher import Matcher
import pandas as pd
import requests
import urllib
import bs4 as bs
import urllib.request
import re
import numpy as np
import spacy
from collections import Counter
import pandas_datareader as web
import datetime
from pandas.tseries.offsets import BDay
import mlfinlab

corpus = pd.read_csv('../data/corpus/minutes.csv', index_col=0)['corpus'][0]

nlp = spacy.load('en_core_web_sm')
doc1 = nlp(corpus)

# ! KEEP THIS BLOCK FOR COMPELTE PREPROCESSING
# The first function below eliminates stop words and punctuation
# The second function lemmatizes remaining words

def is_token_allowed(token):
    '''
        Only allow valid tokens which are not stop words
        and punctuation symbols.
    '''
    if (not token or not token.string.strip() or
            token.is_stop or token.is_punct):
        return False
    return True


def preprocess_token(token):
    # Reduce token to its lowercase lemma form
    return token.lemma_.strip().lower()


complete_filtered_tokens = [preprocess_token(token)
                            for token in doc1 if is_token_allowed(token)]

# List of filtered words; have not run word frequency/importance at this point
complete_filtered_tokens