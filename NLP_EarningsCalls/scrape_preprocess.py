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

start = '2018-01-01'
today = datetime.datetime.today()

# Scrape Motley Fool for Text data

def earnings_meta_data(url):
        
    source = requests.get(url).text
    soup = bs.BeautifulSoup(source, 'lxml')

    # Locate Article Content
    soup = soup.find('span', class_='article-content')

    try:
        date_of_call = datetime.datetime.strptime(
            soup.find('span', id='date').text, '%B %d, %Y')
    except ValueError:
        try:
            date_of_call = datetime.datetime.strptime(
                soup.find('span', id='date').text, '%b %d, %Y')
        except ValueError:
            date_of_call = datetime.datetime.strptime(
                soup.find('span', id='date').text, '%b. %d, %Y')


    ticker = soup.find('span', class_='ticker').text.split(':')[1].replace(')', '')

    Bdays_after_call_1 = date_of_call + BDay(1)
    Bdays_after_call_2 = date_of_call + BDay(2)
    Bdays_after_call_3 = date_of_call + BDay(3)
    Bdays_after_call_4 = date_of_call + BDay(4)
    Bdays_after_call_5 = date_of_call + BDay(5)
    Bdays_after_call_6 = date_of_call + BDay(6)
    Bdays_after_call_7 = date_of_call + BDay(7)
    Bdays_after_call_8 = date_of_call + BDay(8)
    Bdays_after_call_9 = date_of_call + BDay(9)
    Bdays_after_call_10 = date_of_call + BDay(10)
    Bdays_after_call_15 = date_of_call + BDay(15)
    Bdays_after_call_20 = date_of_call + BDay(20)
    Bdays_after_call_25 = date_of_call + BDay(25)
    Bdays_after_call_30 = date_of_call + BDay(30)
    Bdays_after_call_35 = date_of_call + BDay(35)
    Bdays_after_call_40 = date_of_call + BDay(40)
    Bdays_after_call_45 = date_of_call + BDay(45)
    Bdays_after_call_50 = date_of_call + BDay(50)
    Bdays_after_call_55 = date_of_call + BDay(55)
    Bdays_after_call_60 = date_of_call + BDay(60)

    # Get historical price data from Datareader
    try:
        price_df = web.DataReader(ticker, 'yahoo', start, today)
    except KeyError:
        print('Historical Data Unavailable for: ', ticker)

    # Scrape all <p> and combine them into one string
    earnings_call = []

    for t in soup.findAll('p'):

        earnings_call = str(earnings_call) + ' ' + t.text

    # print(earnings_call)


    # Initialize Spacy on document
    nlp = spacy.load('en_core_web_sm')
    doc1 = nlp(earnings_call)

    # Remove stop words and punctuation symbols
    words = [token.text for token in doc1 if not token.is_stop and not token.is_punct]
    word_freq = Counter(words)

    # 5 commonly occurring words with their frequencies
    common_words = word_freq.most_common(5)
    # print(common_words)

    # Unique words
    unique_words = [word for (word, freq) in word_freq.items() if freq == 1]
    # print(unique_words)


    # for token in doc1:
    #     print(token, token.tag_, token.pos_, spacy.explain(token.tag_))


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
    # print(vectorizer.get_feature_names())

    tfidf_bow = vectorizer.get_feature_names()
    tfidf_bow

    # Run a quick sentiment analyzer for fun
    sid = SentimentIntensityAnalyzer()
    sid.polarity_scores(earnings_call)


    # Generate DataFrame of Features with indiv. Earnings Calls as observations in each row

    df = pd.DataFrame(index=[date_of_call])
    df['ticker'] = ticker
    df['corpus'] = earnings_call
    df['tf_idf_words'] = [tfidf_bow]

    # Note: Days are business days
    df['% Return 1-Day After Call'] = price_df['Close'].loc[:Bdays_after_call_1].pct_change(1)[-1]
    df['1st Business Day After'] = price_df['Close'].loc[:Bdays_after_call_1].index[-1].date()
    df['% Return 2-Days After Call'] = price_df['Close'].loc[:Bdays_after_call_2].pct_change(1)[-1]
    df['2nd Business Day After'] = price_df['Close'].loc[:Bdays_after_call_2].index[-1].date()
    df['% Return 3-Days After Call'] = price_df['Close'].loc[:Bdays_after_call_3].pct_change(1)[-1]
    df['3rd Business Day After'] = price_df['Close'].loc[:Bdays_after_call_3].index[-1].date()
    df['% Return 4-Days After Call'] = price_df['Close'].loc[:Bdays_after_call_4].pct_change(1)[-1]
    df['4th Business Day After'] = price_df['Close'].loc[:Bdays_after_call_4].index[-1].date()
    df['% Return 5-Days After Call'] = price_df['Close'].loc[:Bdays_after_call_5].pct_change(1)[-1]
    df['5th Business Day After'] = price_df['Close'].loc[:Bdays_after_call_5].index[-1].date()
    df['% Return 6-Days After Call'] = price_df['Close'].loc[:Bdays_after_call_6].pct_change(1)[-1]
    df['6th Business Day After'] = price_df['Close'].loc[:Bdays_after_call_6].index[-1].date()
    df['% Return 7-Days After Call'] = price_df['Close'].loc[:Bdays_after_call_7].pct_change(1)[-1]
    df['7th Business Day After'] = price_df['Close'].loc[:Bdays_after_call_7].index[-1].date()
    df['% Return 8-Days After Call'] = price_df['Close'].loc[:Bdays_after_call_8].pct_change(1)[-1]
    df['8th Business Day After'] = price_df['Close'].loc[:Bdays_after_call_8].index[-1].date()
    df['% Return 9-Days After Call'] = price_df['Close'].loc[:Bdays_after_call_9].pct_change(1)[-1]
    df['9th Business Day After'] = price_df['Close'].loc[:Bdays_after_call_9].index[-1].date()
    df['% Return 10-Days After Call'] = price_df['Close'].loc[:Bdays_after_call_10].pct_change(1)[-1]
    df['10th Business Day After'] = price_df['Close'].loc[:Bdays_after_call_10].index[-1].date()
    df['% Return 15-Days After Call'] = price_df['Close'].loc[:Bdays_after_call_15].pct_change(1)[-1]
    df['15th Business Day After'] = price_df['Close'].loc[:Bdays_after_call_15].index[-1].date()
    df['% Return 20-Days After Call'] = price_df['Close'].loc[:Bdays_after_call_20].pct_change(1)[-1]
    df['20th Business Day After'] = price_df['Close'].loc[:Bdays_after_call_20].index[-1].date()
    df['% Return 25-Days After Call'] = price_df['Close'].loc[:Bdays_after_call_25].pct_change(1)[-1]
    df['25th Business Day After'] = price_df['Close'].loc[:Bdays_after_call_25].index[-1].date()
    df['% Return 30-Days After Call'] = price_df['Close'].loc[:Bdays_after_call_30].pct_change(1)[-1]
    df['30th Business Day After'] = price_df['Close'].loc[:Bdays_after_call_30].index[-1].date()
    df['% Return 35-Days After Call'] = price_df['Close'].loc[:Bdays_after_call_35].pct_change(1)[-1]
    df['35th Business Day After'] = price_df['Close'].loc[:Bdays_after_call_35].index[-1].date()
    df['% Return 40-Days After Call'] = price_df['Close'].loc[:Bdays_after_call_40].pct_change(1)[-1]
    df['40th Business Day After'] = price_df['Close'].loc[:Bdays_after_call_40].index[-1].date()
    df['% Return 45-Days After Call'] = price_df['Close'].loc[:Bdays_after_call_45].pct_change(1)[-1]
    df['45th Business Day After'] = price_df['Close'].loc[:Bdays_after_call_45].index[-1].date()
    df['% Return 50-Days After Call'] = price_df['Close'].loc[:Bdays_after_call_50].pct_change(1)[-1]
    df['50th Business Day After'] = price_df['Close'].loc[:Bdays_after_call_50].index[-1].date()
    df['% Return 55-Days After Call'] = price_df['Close'].loc[:Bdays_after_call_55].pct_change(1)[-1]
    df['55th Business Day After'] = price_df['Close'].loc[:Bdays_after_call_55].index[-1].date()
    df['% Return 60-Days After Call'] = price_df['Close'].loc[:Bdays_after_call_60].pct_change(1)[-1]
    df['60th Business Day After'] = price_df['Close'].loc[:Bdays_after_call_60].index[-1].date()


    call_polarity_score = sid.polarity_scores(df['corpus'][0])
    df['neg_score_corpus'] = call_polarity_score['neg']
    df['neutral_score_corpus'] = call_polarity_score['neu']
    df['pos_score_corpus'] = call_polarity_score['pos']


    return df


# Read in csv that contains earnings call urls
links = pd.read_csv('./data/earnings_links.csv', index_col=0)

# Create empty dataframe to house dataframes generated in earnings_meta_data loop
data = pd.DataFrame()

# Loop over earnings_meta_data with the earnings transcript urls
for u in links['earnings_links']:

    if 'RCL' in u:
        pass
    else:
        temp = earnings_meta_data(url=u)

        data = data.append(temp)
        print(data)

print('All done.')