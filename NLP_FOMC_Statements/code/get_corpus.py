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

domain = 'https://www.federalreserve.gov/'

def statements_corpus():

    statements = pd.read_csv('../data/urls/statements.csv', index_col=0)

    for x in statements.index:

        extension = statements.loc[x, 'statement_urls']
        url = domain + extension

        source = requests.get(url).text
        soup = bs.BeautifulSoup(source, 'lxml')
        corpus = []

        for t in soup.find('div', id='article').findAll('p'):

            corpus = str(corpus) + ' ' + t.text

        statements.at[x, 'corpus'] = corpus

    statements.to_csv('../data/corpus/statements.csv')


def minutes_corpus():

    minutes = pd.read_csv('../data/urls/minutes.csv', index_col=0)

    for x in minutes.index:

        extension = minutes.loc[x, 'minutes_urls']
        url = domain + extension

        source = requests.get(url).text
        soup = bs.BeautifulSoup(source, 'lxml')
        corpus = []

        for t in soup.find('div', id='article').findAll('p'):

            corpus = str(corpus) + ' ' + t.text

        minutes.at[x, 'corpus'] = corpus

    minutes.to_csv('../data/corpus/minutes.csv')

    print(minutes)

# TODO: Finish Implementation Notes scrapings

# def implem_notes_corpus():

#     notes = pd.read_csv('../data/urls/implementation_notes.csv', index_col=0)

#     for x in notes.index[:1]:

#         extension = notes.loc[x, 'implem notes urls']
#         url = domain + extension

#         source = requests.get(url).text
#         soup = bs.BeautifulSoup(source, 'lxml')
#         corpus = []

#         for t in soup.find('div', class_='col-xs-12 col-sm-8').findAll('p'):

#             corpus = str(corpus) + ' ' + t.text

#         notes.at[x, 'corpus'] = corpus

#     notes.to_csv('../data/corpus/implementation_notes.csv')

#     print(notes)