import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as sct
import sklearn.linear_model as skl
import sklearn.metrics as skm
import sklearn.decomposition as skd
import statsmodels.api as sm
import pandas_datareader as web

'''
Estimateafive-variableVARusingq3.pkl:’DPCERA3M086SBEA’,’UNRATE’,’CPIAUCSL’, ’S&P 500’, ’GS10’. Choose the lag order by using the AIC, with a maximum lag of 12
'''

# ! Question 1
print('')
print('**** Question 1 ***')
print('')
macro = pd.read_pickle("/Users/OrrenSingh/repositories/public_github/master_program_sample/homework/hw_10/q3.pkl")

estimate_VAR = sm.tsa.VAR(endog=macro[['DPCERA3M086SBEA','UNRATE','CPIAUCSL','S&P 500','GS10']].dropna()).fit(maxlags=12,ic='aic')

# print(estimate_VAR.summary())
print(macro[['DPCERA3M086SBEA','UNRATE','CPIAUCSL','S&P 500','GS10']].dropna())

fitted = estimate_VAR.fittedvalues
print(fitted)
print('')
print('The AIC limited the VAR to 9 lags')
print('The fitted values are what the Vector Autoregression forecasts from the endogenous variables in our dataset.\n'
        'The Vector Autoregression calculates phi (which is the variables coefficient for a given lag), \n'
        'then adds them up to give us our forecasted value.\n')
print("In other words, from the class notes, 'VAR is defining a system of equations'.")
print("Again, from the notes, 'the VAR is saying that each Ynt is linearly related to a constant and N*p regressors: the p lags of that N-vector y.'")
print("")
print("In my own words, this means that via the autocorrelation that exists, the regressed lags allow us to predict the next variable.")
fitted.plot()


#######################################################

'''
Estimate a five-variable VAR using the Fama French five-factor model. Choose the lag order by using the AIC, with a maximum lag of 12.
'''

# ! Question 2

print('')
print('**** Question 2 ***')
print('')

ff_25 = web.DataReader('F-F_Research_data_5_Factors_2x3','famafrench',start='1963-07-01')

ff_estimate_VAR = sm.tsa.VAR(endog=ff_25[0].dropna()).fit(maxlags=12,ic='aic')
# print(ff_estimate_VAR.summary())

ff_fitted = ff_estimate_VAR.fittedvalues
ff_fitted.plot()
print('')
print(ff_25[0])
print(ff_fitted)
ff_estimate_VAR.resid.plot()
print('')
print('The AIC limited the VAR to 1 lags')
print('The fitted values are what the Vector Autoregression forecasts from the endogenous variables in our dataset.\n'
        'The Vector Autoregression calculates phi (which is the variables coefficient for a given lag), \n'
        'then adds them up to give us our forecasted value.\n')
print("In other words, from the class notes, 'VAR is defining a system of equations'.")
print("Again, from the notes, 'the VAR is saying that each Ynt is linearly related to a constant and N*p regressors: the p lags of that N-vector y.'")
print("")
print("In my own words, this means that via the autocorrelation that exists, the regressed lags allow us to predict the next variable.")
plt.show()


