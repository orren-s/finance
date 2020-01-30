import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model as skl
import sklearn.metrics as skm
import sklearn as sk
import sklearn.decomposition as skd
import statsmodels.api as sm
import datetime
import sklearn.ensemble.forest as ens
import warnings
warnings.filterwarnings('ignore')
import pandas_datareader as web
from datetime import timedelta


# Load VIX historical data via pandas_datareader

start = pd.to_datetime('2002-01-01')
end = pd.datetime.today()

vix = web.DataReader('^VIX','yahoo',start, end)['Adj Close']
vix = vix.loc[:'2019-12-20']


# Create a feature set of 20 columns, with each column being n-days back from most recent value

d = datetime.timedelta(days=1)

vix_lags = pd.Series()

for x in range(1,21):

    temp = vix.shift(x)
    
    vix_lags = pd.concat([vix_lags, temp], axis=1).rename(columns={'Adj Close':f'Lag {x}'})

del vix_lags[0]
# print(vix_lags)

from sklearn.model_selection import train_test_split

# Use Random Foreset Regression on the lagged features to predict the next days value

rfr = sk.ensemble.RandomForestRegressor(n_estimators=100, max_depth=50, random_state=0)

X = vix_lags.dropna()
y = vix.loc['2002-01-31':].values.reshape(-1,1)

X_train = X[:2500]
X_test = X[2500:]
y_train = y[:2500]
y_test = y[2500:]

model_fit = rfr.fit(X_train, y_train)
score = model_fit.score(X_test, y_test)

print('R2 RFR Train-Test: ', score)




# Create empty dataframes to store data generated in for loop below
results = pd.DataFrame()
mean = pd.DataFrame()

vixLagsDropna = vix_lags.dropna()

# Do a walk-forward test to simulate what the accuracy would be if you were to re-run the algorithm in real time each day going back to January 31, 2003

for t in vix.loc['2003-01-31':].index:


    rfr = sk.ensemble.RandomForestRegressor(n_estimators=10, max_depth=5, random_state=0)

    X = vixLagsDropna.loc[:t-d]
    y = vix[20:].loc[:t-d].values.reshape(-1,1)

    model = rfr.fit(X, y)
    pred = vixLagsDropna.loc[t].values.reshape(1,-1) 
    predict = model.predict(pred)

    mean.at[t, 'Mean'] = vix[20:].loc[:t].mean()
    print('mean ',mean[-1:])
    results.at[t, 'Predicted Vix'] = predict
    print('prediction ', results[-1:])


# Save historical mean and results to csvs

mean.to_csv('analysis/data/findings/vix_WF_mean.csv')
results.to_csv('analysis/data/findings/vix_WF_preds.csv')

results = pd.read_csv('analysis/data/findings/vix_WF_preds.csv', index_col=0, parse_dates=True)
mean = pd.read_csv('analysis/data/findings/vix_WF_mean.csv', index_col=0, parse_dates=True)


# Using the mean and actual results, manually calculate the R2

yk = results['Predicted Vix'].values
yfk = vix.loc['2003-01-31':'2019-12-20'].values
ymk = mean['Mean'].values

walk_for_r2 = 1 - np.sum( (yk-yfk)**2 )/np.sum( (yk-mean['Mean'])**2 )

print('R2 RFR Walk-Forward: %.04f' % walk_for_r2)

# Result: 0.9559




'''
Below is an attempt to do a Markov Autoregression on the predict VIX values above.
The reason is that I was curious to see how the states would change if I were
to simulate this as if it were happening in real time. However, my computer could not
handle the computational power required by Statsmodels sm.tsa.MarkovAutoregression()
function. I left this code in, however, for future reference for myself.

Beneath this chunk, I settled on skipping the walk-forward of predicted values,
and running a MarkovAR on the historical VIX values to see how the SP500 returns might
differ given the volatility state.
'''


# Use forecasted vix in Markov AR

results.index = results.index.to_period('D')
vix.index = vix.index.to_period('D')

state0_OOS = pd.DataFrame()

for x in results.index:

    fcast = results['Predicted Vix'].loc[x]
    hist = vix.loc[:x-1]
    fcast = pd.Series(data=fcast, index=[x])
    
    df = hist.append(fcast)
    df = df.diff().dropna()
    # df = df.loc[x-1000:x]
    # print(df)
    vix_regimes = sm.tsa.MarkovAutoregression(df, k_regimes=2, order=5, switching_variance=False).fit()

    # print(vix_regimes.smoothed_marginal_probabilities[0][-1:].values)

    state0_OOS.at[x,'State0_OOS'] = vix_regimes.smoothed_marginal_probabilities[0][-1:].values

    print(state0_OOS)


state0_OOS.to_csv('state0.csv')



'''
Below, I use the historical values of the VIX with a MarkovAR to see if there
are any patterns or anomalies in the SP500 returns given the state of Volatility(i.e.
high or low vol)

Note: you must run the above code at least for a few dozen iterations to generate
the state0_OOS dataframe
'''


spy = web.DataReader('SPY','yahoo',start, end)['Adj Close']
spy_ret = spy.pct_change().dropna()

spy = web.DataReader('SPY','yahoo',start, end)
spy_OC_range = spy['Close'] - spy['Open']
spy_HL_range = spy['High'] - spy['Low']

spy_HL_range = pd.DataFrame(spy_HL_range, columns=['High-Low'])
spy_OC_range = pd.DataFrame(spy_OC_range, columns=['Close-Open'])

vix_regimes = sm.tsa.MarkovAutoregression(vix.diff().dropna(), k_regimes=2, order=5, switching_variance=False).fit()

state0 = vix_regimes.smoothed_marginal_probabilities[0]

state0_OOS = pd.DataFrame(state0_OOS)
spy_ret = pd.DataFrame(data=spy_ret.values, index=spy_ret.index)
spy_ret.rename(columns={0:'Spy'}, inplace=True)
spy_ret = spy_ret.to_period(freq='D')




# vol_spy = pd.concat([spy_ret, state0_OOS], axis=1).dropna(axis=1)
vol_spy = state0_OOS.merge(spy_ret, how='inner',left_index=True, right_index=True)
vol_spy.rename(columns={'State0_OOS':'state0'}, inplace=True)


# vol_spy = vol_spy.loc['2006-12-31':'2009-01-01']

ret_abv_mean = vol_spy[vol_spy['state0'] > vol_spy['state0'].mean() + vol_spy['state0'].std()]
ret_blw_mean = vol_spy[vol_spy['state0'] < vol_spy['state0'].mean() - vol_spy['state0'].std()]

ret_abv_75th = vol_spy[vol_spy['state0'] > vol_spy['state0'].quantile(q=.75)]
ret_blw_25th = vol_spy[vol_spy['state0'] < vol_spy['state0'].quantile(q=.25)]


print('State0 is the high volatility regime')
print('')
print('Avg Spy Return Above State0 Mean: %.04f' % ret_abv_mean['Spy'].mean())
print('Max Spy Return Above State0 Mean: %.04f' % ret_abv_mean['Spy'].max())
print('Min Spy Return Above State0 Mean: %.04f' % ret_abv_mean['Spy'].min())
print('')
print('Avg Spy Return Below State0 Mean: %.04f' % ret_blw_mean['Spy'].mean())
print('Max Spy Return Below State0 Mean: %.04f' % ret_blw_mean['Spy'].max())
print('Min Spy Return Below State0 Mean: %.04f' % ret_blw_mean['Spy'].min())


ret_abv_mean['Spy'].hist(bins=ret_abv_mean['Spy'].count())
ret_blw_mean['Spy'].hist(bins=ret_blw_mean['Spy'].count())