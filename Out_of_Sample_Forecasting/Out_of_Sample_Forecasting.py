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


macro = pd.read_pickle("/Users/OrrenSingh/repositories/public_github/master_program_sample/homework/hw_11/q3.pkl")[['DPCERA3M086SBEA','UNRATE','CPIAUCSL','S&P 500','GS10']].dropna()

estimate_VAR = sm.tsa.VAR(endog=macro).fit(maxlags=12,ic='aic')



# ! Question 1

'''
Estimate a five-variable VAR using q3.pkl: ’DPCERA3M086SBEA’, ’UNRATE’, ’CPIAUCSL’, ’S&P 500’, ’GS10’. Choose the lag order by using the AIC, with a maximum lag of 12.
'''

lags = estimate_VAR.k_ar
ISR2 = pd.DataFrame(index=[0],columns=macro.columns,data=np.nan)

for x in macro.columns:
    y = macro[x].values[lags:]
    yf = estimate_VAR.fittedvalues[x].values[:]
    ISR2[x] = skm.r2_score(y,yf)

print('')
print('*** Question 1 ***')
print('')
print('R2 Values: ')
print(ISR2)
print('')
print(f'Chosen lag length: {estimate_VAR.k_ar}')
print('')


# ! Question 2

'''
Estimate the five-variable VAR on the training sample ending in 1999-12. Use these estimates to construct out-of-sample forecasts for 2000-01 onwards
'''

train_is = sm.tsa.VAR(endog=macro.loc[:'1999-12']).fit(maxlags=12, ic='aic')
train_lags = train_is.k_ar
train_ISR2 = pd.DataFrame(index=[0],columns=macro.columns,data=np.nan)

# In-Sample Benchmark R2

for x in macro.columns:
    y = macro[x].loc[:'1999-12'].values[train_is.k_ar:]
    yf = train_is.fittedvalues[x].values[:]
    train_ISR2[x] = skm.r2_score(y,yf)

# OOS Forecast

OOSfore= pd.DataFrame(index=macro.loc['2000-01':].index, columns=macro.columns,data=np.nan)
OOSmean = pd.DataFrame(index=macro.loc['2000-01':].index, columns=macro.columns,data=np.nan)

for x in macro.loc['2000-01':].index:
    f = train_is.forecast(y=macro.loc[x-train_is.k_ar:x-1].values,steps=1)
    OOSfore.loc[x] = f
    OOSmean.loc[x] = macro.loc[:'1999-12'].mean()


# OOS R2
OOSR2 = pd.DataFrame(index=[0],columns=macro.columns,data=np.nan)

for x in macro.columns:
    y = macro[x].loc['2000-01':].values
    yf = OOSfore[x].values
    ym = OOSmean[x].values
    OOSR2[x] = 1 - np.sum( (y-yf)**2 )/np.sum( (y-ym)**2 )

    # OOSR2[x] = skm.r2_score(y,yf)


print('*** Question 2 ***')
print('')
print('Benchmark Training-Sample R2:')
print(train_ISR2)
print('')
print('OOS R2')
print(OOSR2)
print('')
print(f'Chosen lag length: {train_is.k_ar}')
print('')

# ! Question 3

'''
Estimate the five-variable VAR on an expanding window basis to construct out-of-sample
forecasts for 2000-01 onwards
'''

# do all the OOS forecasts using an expanding window
kar = pd.DataFrame(index=macro.loc['2000-01':].index, columns=['k_ar'], data=np.nan)
OOSfore2_exp = pd.DataFrame(index=macro.loc['2000-01':].index, columns=macro.columns,data=np.nan)
OOSR2_exp = pd.DataFrame(index=[0],columns=macro.columns,data=np.nan)
OOS2_mean = pd.DataFrame(index=macro.loc['2000-01':].index, columns=macro.columns,data=np.nan)

for x in macro.loc['2000-01':].index:
    estVAR = sm.tsa.VAR(macro.loc[:x-1]).fit(maxlags=12,ic='aic')
    kar.loc[x] = estVAR.k_ar
    OOSfore2_exp.loc[x] = estVAR.forecast(y=macro.loc[x-estVAR.k_ar-1:x-1].values,steps=1)
    OOS2_mean.loc[x] = macro.loc[:].mean()


for x in macro.columns:
    y = macro[x].loc['2000-01':].values
    yf = OOSfore2_exp[x].values
    ym = OOS2_mean[x].values
    OOSR2_exp[x] = 1 - np.sum( (y-yf)**2 )/np.sum( (y-ym)**2 )

print('*** Question 3 ***')
    
print('Benchmark Training-Sample R2:')
print(train_ISR2)
print('')
print('OOS Expanding R2')
print(OOSR2_exp)
print('')
kar.plot()
plt.show()
