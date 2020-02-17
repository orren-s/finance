from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.matcher import Matcher
import pandas as pd
import requests
import pandas as pd
import bs4 as bs
import urllib.request
import re
import numpy as np
import spacy
from collections import Counter
import pandas_datareader as web
import datetime
from pandas.tseries.offsets import BDay

start = '2018-01-01'
today = datetime.datetime.today()

# Scrape Motley Fool for Text data

# url = 'https://www.fool.com/earnings/call-transcripts/2020/01/28/apple-inc-aapl-q1-2020-earnings-call-transcript.aspx'
url = 'https://www.fool.com/earnings/call-transcripts/2019/04/30/apple-inc-aapl-q2-2019-earnings-call-transcript.aspx'
url = 'https://www.fool.com/earnings/call-transcripts/2019/07/30/apple-inc-aapl-q3-2019-earnings-call-transcript.aspx'

source = requests.get(url).text
soup = bs.BeautifulSoup(source, 'lxml')

# Locate Article Content
soup = soup.find('span', class_='article-content')

try:
    date_of_call = datetime.datetime.strptime(
        soup.find('span', id='date').text, '%B %d, %Y')
except ValueError:
    date_of_call = datetime.datetime.strptime(
        soup.find('span', id='date').text, '%b %d, %Y')


ticker = soup.find('span', class_='ticker').text.split(':')[1].replace(')', '')

Bdays_after_call_20 = date_of_call + BDay(20)
Bdays_after_call_40 = date_of_call + BDay(40)
Bdays_after_call_60 = date_of_call + BDay(60)

# Get historical price data from Datareader
price_df = web.DataReader(ticker, 'yahoo', start, today)

# Scrape all <p> and combine them into one string
earnings_call = []

for t in soup.findAll('p'):

    earnings_call = str(earnings_call) + ' ' + t.text

print(earnings_call)


# Initialize Spacy on document
nlp = spacy.load('en_core_web_sm')
doc1 = nlp(earnings_call)

# Remove stop words and punctuation symbols
words = [token.text for token in doc1 if not token.is_stop and not token.is_punct]
word_freq = Counter(words)

# 5 commonly occurring words with their frequencies
common_words = word_freq.most_common(5)
print(common_words)

# Unique words
unique_words = [word for (word, freq) in word_freq.items() if freq == 1]
print(unique_words)


for token in doc1:
    print(token, token.tag_, token.pos_, spacy.explain(token.tag_))


nouns = []
adjectives = []
for token in doc1:
    if token.pos_ == 'NOUN':
        nouns.append(token)
    if token.pos_ == 'ADJ':
        adjectives.append(token)

nouns

adjectives


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


# Run TF-IDF on Lemmatized Bag of Words

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(complete_filtered_tokens)
print(vectorizer.get_feature_names())

tfidf_bow = vectorizer.get_feature_names()
tfidf_bow

# Run a quick sentiment analyzer for fun
sid = SentimentIntensityAnalyzer()
sid.polarity_scores(earnings_call)


# Generate DataFrame of Features with indiv. Earnings Calls as observations in each row

df = pd.DataFrame(index=[date_of_call])
df['corpus'] = earnings_call
df['tf_idf_words'] = [tfidf_bow]

# Note: Days are business days
df['% Return 20-Days After Call'] = price_df['Close'].loc[:
                                                          Bdays_after_call_20].pct_change(20)[-1]
df['20th Business Day After'] = price_df['Close'].loc[:Bdays_after_call_20].index[-1].date()

df['% Return 40-Days After Call'] = price_df['Close'].loc[:
                                                          Bdays_after_call_40].pct_change(40)[-1]
df['40th Business Day After Call'] = price_df['Close'].loc[:Bdays_after_call_40].index[-1].date()

df['% Return 60-Days After Call'] = price_df['Close'].loc[:
                                                          Bdays_after_call_60].pct_change(60)[-1]
df['60th Business Day After Call'] = price_df['Close'].loc[:Bdays_after_call_60].index[-1].date()
