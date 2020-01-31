import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as sct
import sklearn.linear_model as skl
import sklearn.metrics as skm
import sklearn.decomposition as skd
import sklearn.cross_decomposition as skcd
import statsmodels.api as sm
import datetime
import pandas_datareader.data as web
import warnings
warnings.filterwarnings("ignore") 

p = print

macro = pd.read_pickle("/Users/OrrenSingh/repositories/public_github/master_program_sample/homework/hw_14/q3.pkl").drop(['ACOGNO', 'S&P 500', 
                      'ANDENOx', 'TWEXMMTH', 'UMCSENTx', 'VXOCLSx'],axis=1)
factors = web.DataReader('F-F_Research_Data_5_Factors_2x3','famafrench',
                         start='1926-01-01'
                    )[0].replace(to_replace=[-99.99,-999],
                    value=np.nan).drop(['RF'],axis=1)

names_macro = macro.columns
names_factors = factors.columns

alldata = macro.shift(1).join(factors).dropna()

macro_zscore = sct.zscore(alldata[names_macro].values,axis=0)

# ! Question 1

'''
Estimateaone-factorthree-passregressionforecastsfromthemacrodataforeachreturn factor, out-of-sample for 1980-01 onwards. Report the OOS R2 versus the historical mean benchmark
'''

p('*** Question 1 ***')
print('')
for names in names_factors:

    # three-pass reg filter for market excess return
    y = alldata[names]
    X = macro_zscore
    # first-pass
    Phitilde = la.lstsq(a=sm.add_constant(y),
                        b=X,
                        rcond=None)[0]
    Phi = Phitilde[1,:]
    # second-pass
    factor = la.lstsq(a=sm.add_constant(Phi),
                    b=X.T,
                    rcond=None)[0][1,:]
    # third-pass
    reg = sm.OLS(endog=y,
                exog=sm.add_constant(factor)).fit()


    OOSfore = pd.DataFrame(index=alldata.loc['1980-01':].index,
                        columns=[0],data=np.nan)
    OOSmean = OOSfore.copy()
    for t in alldata.loc['1980-01':].index:
        y = alldata[names].loc[:t-1]
        X = sct.zscore(alldata[names_macro].loc[:t,:].values, axis=0)
        X1 = X[:-1,:]
        X2 = X[-1,:]
        # first-pass
        Phi = la.lstsq(a=sm.add_constant(y),
                    b=X1,
                    rcond=None)[0][1,:]
        # second-pass
        factor = la.lstsq(a=sm.add_constant(Phi),
                        b=X1.T,
                        rcond=None)[0][1,:]
        # third-pass
        reg = sm.OLS(endog=y,
                exog=sm.add_constant(factor)).fit()
        
        # OOS forecast
        OOSfore.loc[t] = reg.params[0] + reg.params[1]*(
                sm.OLS(endog=X2, exog=sm.add_constant(Phi)).fit().params[1]
                )
        OOSmean.loc[t] = y.mean()

    oosr2 = (1 - np.sum( (alldata[names].loc['1980-01':].values-OOSfore.values)**2 )/
            np.sum( (alldata[names].loc['1980-01':].values-OOSmean.values)**2 )
            )
    print("R2 for %s is %0.4f%%" % ( names, 100*oosr2))

print('')

# ! Question 2

'''
Estimateout-of-sampleforecastsfromthemacrodataforeachreturnfactor,for1980-01 onwards, using ridge regression with a single alpha for every time period. State what is the alpha. Report the OOS R2 versus the historical mean benchmark.
'''

print('*** Question 2 ***')
print('')

OOSfore = pd.DataFrame(index=alldata.loc['1980-01':].index,
                    data=np.nan,columns=factors.columns)

OOSmean = OOSfore.copy()

for names in names_factors:
    for t in alldata.loc['1980-01':].index:

        z = sct.zscore(alldata[names_macro].loc[:t].values,axis=0)

        est_ridge = skl.Ridge(alpha=10).fit(X=z[:-1, :], y=alldata[names].loc[:t-1])
    
        OOSfore.at[t, names] = z[-1,:].T @ est_ridge.coef_

        OOSmean.at[t,names] = alldata[names].loc[:t-1].mean()
        
    yk = alldata[names].loc['1980-01':].values
    yfk = OOSfore[names].values
    ymk = OOSmean[names].values
    R2 = 1 - np.sum( (yk-yfk)**2 )/np.sum( (yk-ymk)**2 )
    print("R2 for %s is %0.4f%%" % ( names, 100*R2))
print('The alpha that I used for the Ridge Regression is 10')
print('')

# ! Question 3

'''
Estimateout-of-sampleforecastsfromthemacrodataforeachreturnfactor,for1980-01 onwards, using LASSO with a single alpha for every time period. State what is the alpha. Report the OOS R2 versus the historical mean benchmark
'''

print('*** Question 3 ***')
print('')

OOSfore = pd.DataFrame(index=alldata.loc['1980-01':].index,
                    data=np.nan,columns=factors.columns)

OOSmean = OOSfore.copy()

for names in names_factors:
    for t in alldata.loc['1980-01':].index:

        z = sct.zscore(alldata[names_macro].loc[:t].values,axis=0)

        est_ridge = skl.Lasso(alpha=.25).fit(X=z[:-1, :], y=alldata[names].loc[:t-1])
    
        OOSfore.at[t, names] = z[-1,:].T @ est_ridge.coef_

        OOSmean.at[t,names] = alldata[names].loc[:t-1].mean()
        
    yk = alldata[names].loc['1980-01':].values
    yfk = OOSfore[names].values
    ymk = OOSmean[names].values
    R2 = 1 - np.sum( (yk-yfk)**2 )/np.sum( (yk-ymk)**2 )
    print("R2 for %s is %0.4f%%" % ( names, 100*R2))
print('The gamma that I used for the Lasso Regression is .25')
print('')
