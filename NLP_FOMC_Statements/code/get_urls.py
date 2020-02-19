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

url = 'https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm'

source = requests.get(url).text
soup = bs.BeautifulSoup(source, 'lxml')

# Get all links from fomc calendar page
links = []
for link in soup.findAll('a', href=True):
    links.append(link.get('href'))


# Get all links for meeting minutes
minutes = []
for x in links:
    if 'monetarypolicy/fomcminutes' in x:
        minutes.append(x)

minutes = pd.DataFrame(minutes, columns=['minutes_urls'])

for x in minutes.index:
    temp = minutes.loc[x, 'minutes_urls'][:-4]
    date = temp[-8:]
    date = datetime.datetime.strptime(date, '%Y%m%d').date()
    minutes.at[x, 'date'] = date

minutes.set_index('date', inplace=True)
minutes.to_csv('../data/urls/minutes.csv')

# Get all links for statements
statements = []
for x in links:
    if '/pressreleases/monetary' in x:
        if 'a.htm' in x:
            statements.append(x)

statements = pd.DataFrame(statements, columns=['statement_urls'])

for x in statements.index:
    
    temp = statements.loc[x, 'statement_urls'][34:]
    date = temp[:8]
    date = datetime.datetime.strptime(date, '%Y%m%d').date()
    statements.at[x, 'date'] = date

statements.set_index('date', inplace=True)
statements.to_csv('../data/urls/statements.csv')



# Get all links for statements
implementation_note = []

for x in links:
    if '/pressreleases/monetary' in x:
        if 'a1.htm' in x:
            implementation_note.append(x)

implementation_note = pd.DataFrame(implementation_note, columns=['implem notes urls'])

for x in implementation_note.index:
    
    temp = implementation_note.loc[x, 'implem notes urls'][34:]
    date = temp[:8]
    date = datetime.datetime.strptime(date, '%Y%m%d').date()
    implementation_note.at[x, 'date'] = date

implementation_note.set_index('date', inplace=True)
implementation_note.to_csv('../data/urls/implementation_notes.csv')