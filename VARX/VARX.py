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

macro = pd.read_pickle("/Users/OrrenSingh/repositories/public_github/master_program_sample/homework/hw_12/q3.pkl").drop(['ACOGNO', 'S&P 500', 'ANDENOx', 'TWEXMMTH', 'UMCSENTx', 'VXOCLSx'],axis=1)

factors = web.DataReader('F-F_Research_Data_5_Factors_2x3','famafrench',start='1926-01-01')[0].replace(to_replace=[-99.99,-999],value=np.nan).drop(['RF'],axis=1)

names_macro = macro.columns
names_factors = factors.columns

alldata = macro.shift(1).join(factors).dropna()

# ! Question 1

'''
EstimateaVARX(1)forthefivefactors,usingasexogenousvariablesallthemacroseries. Remember to lag the macro series.
'''

print('*** Question 1 ***')
estVAR = sm.tsa.VAR(endog=alldata[names_factors],
                    exog=alldata[names_macro]).fit(1)

print("The in-sample R2 for each factor is:")
for name in names_factors:
    y = estVAR.fittedvalues[name] + estVAR.resid[name]
    yh = estVAR.fittedvalues[name]
    print(" for %s is %0.2f%%" % ( name, 100*skm.r2_score(y,yh)) )
print('')
print('Sample period used to construct R2')
print('The sample range is',estVAR.fittedvalues.index[0], 'through',estVAR.fittedvalues.index[-1])
print('')



# ! Question 2

'''
EstimateaVARX(1)asinthepreviousquestion,butinsteadconstructexpanding-window
out-of-sample forecasts from 1980-01 onwards
'''

print('*** Question 2 ***')

OOSfore= pd.DataFrame(index=alldata[names_factors].loc['1980-01':].index, columns=alldata[names_factors].columns,data=np.nan)

OOSmean = pd.DataFrame(index=alldata[names_factors].loc['1980-01':].index, columns=alldata[names_factors].columns,data=np.nan)

for x in alldata.loc['1980-01':].index:

    estVAR = sm.tsa.VAR(endog=alldata[names_factors].loc[:x-1], exog=alldata[names_macro].loc[:x-1]).fit(1)

    OOSfore.loc[x] = estVAR.forecast(y=alldata[names_factors].loc[x-estVAR.k_ar-1:x-1].values, exog_future=alldata[names_macro].loc[x-1:x-1].values, steps=1)

    OOSmean.loc[x] = alldata[names_factors].mean()


# OOS R2
OOSR2 = pd.DataFrame(index=[0],columns=alldata[names_factors].columns,data=np.nan)

for x in alldata[names_factors].columns:
    y = alldata[x].loc['1980-01':].values
    yf = OOSfore[x].values
    ym = OOSmean[x].values
    OOSR2[x] = 1 - np.sum( (y-yf)**2 )/np.sum( (y-ym)**2 )

    print(" for %s is %0.2f%%" % ( x, 100* OOSR2[x]) )

print('')



# ! Question 3

'''
EstimateaVAR(1)forthefivefactors,onthesamesampleasforquestion#1
'''

print('*** Question 3 ***')

OOSfore3= pd.DataFrame(index=alldata[names_factors].index, columns=alldata[names_factors].columns,data=np.nan)

OOSmean3 = pd.DataFrame(index=alldata[names_factors].index, columns=alldata[names_factors].columns,data=np.nan)

# for x in alldata[names_factors].index:

estVAR3 = sm.tsa.VAR(endog=alldata[names_factors]).fit(1)

for name in names_factors:
    y = estVAR3.fittedvalues[name] + estVAR3.resid[name]
    yh = estVAR3.fittedvalues[name]
    print(" for %s is %0.2f%%" % ( name, 100*skm.r2_score(y,yh)) )

print('')



# ! Question 4

'''
Estimate a VAR(1) for the five factors as in the previous question, but instead construct expanding-window out-of-sample forecasts from 1980-01 onwards
'''

print('*** Question 4 ***')

OOSfore4= pd.DataFrame(index=alldata[names_factors].loc['1980-01':].index, columns=alldata[names_factors].columns,data=np.nan)

OOSmean4 = pd.DataFrame(index=alldata[names_factors].loc['1980-01':].index, columns=alldata[names_factors].columns,data=np.nan)

for x in alldata.loc['1980-01':].index:

    estVAR4 = sm.tsa.VAR(alldata[names_factors].loc[:x-1]).fit(1)

    OOSfore4.loc[x] = estVAR4.forecast(y=alldata[names_factors].loc[x-estVAR.k_ar-1:x-1].values, steps=1)

    OOSmean4.loc[x] = alldata[names_factors].mean()

# OOS R2
OOSR2_4 = pd.DataFrame(index=[0],columns=alldata[names_factors].columns,data=np.nan)

for x in alldata[names_factors].columns:
    y = alldata[x].loc['1980-01':].values
    yf = OOSfore4[x].values
    ym = OOSmean4[x].values
    OOSR2_4[x] = 1 - np.sum( (y-yf)**2 )/np.sum( (y-ym)**2 )

    print(" for %s is %0.2f%%" % ( x, 100* OOSR2_4[x]) )

print('')

# ! Question 5

'''
Interpretthecomparisonbetweenin-sampleandout-of-sampleR2s,fortheVARXof#1-2 and the VAR of #3-4. Why might the VARX R2s be so different while the VAR R2s were not?
'''

print('*** Question 5 ***')

print('The reason for the huge difference in R2s for #1 and #2 is that the added exog variables\n'
'were likely overfitting the data. In fact, not "likely"; they were indeed causing the predictions\n'
'to be overfitted, and thus drive up the in-sample R2 and deliver horrific out of sample R2. \n The added exog variables turned out to just be noise in this case.')
print('')
print('Comparing the R2 of #3 and #4, the in-sample estimate delivers promising results \n'
'Even though the values are low, they are still positive; some profitable trading strategies can be conjured from positive even as low as these. \nHowever, the out of sample R2 are a swift reality check, \nquite opposite to the check you thought you would be cashing after seeing #3s R2, \n as the out of sample R2s show that there is no real profitable relationship amongst these variables')