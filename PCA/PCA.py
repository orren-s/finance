import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as sct
import scipy
import sklearn.linear_model as skl
import sklearn.metrics as skm
import sklearn.decomposition as skd
import statsmodels.api as sm
import datetime
import pandas_datareader.data as web

macro = pd.read_pickle("/Users/OrrenSingh/repositories/public_github/master_program_sample/homework/hw_13/q3.pkl").drop(['ACOGNO', 'S&P 500', 
                      'ANDENOx', 'TWEXMMTH', 'UMCSENTx', 'VXOCLSx'],axis=1)

factors = web.DataReader('F-F_Research_Data_5_Factors_2x3','famafrench',
                         start='1926-01-01')[0].replace(to_replace=[-99.99,-999], value=np.nan).drop(['RF'],axis=1)

names_macro = macro.columns
names_factors = factors.columns

alldata = macro.shift(1).join(factors).dropna()

macrodata = alldata[names_macro].values

macro_zscore = sct.zscore(macrodata,axis=0)

pca = skd.PCA(n_components=10,svd_solver='full')
X = pca.fit_transform(macro_zscore)
pcafit = pca.fit(macro_zscore)

#! Question 1

'''
Estimate the first 10 principal components (PCs) of the full sample of (z-scored) macro
data.
'''

print('### Question 1 ###')
print('')
pca_sum = 0
count = 0
for x in pcafit.explained_variance_ratio_:
    pca_sum += x
    count += 1
    print("CumSum of Explained Var. for PCA 1-%s: %0.4f" % (count, pca_sum))
print('')

tmp = np.corrcoef(X,rowvar=False)

print("Maximum Magnitude Corr. for PC is ", np.max(np.abs(np.triu(tmp, k=1))))

print('')

# ! Question 2

'''
Estimate an in-sample regression of each factor return on the first 5 macro PCs and report
the R2 for each factor forecast.
'''

print('### Question 2 ###')
print('')

pca_5 = skd.PCA(n_components=5,svd_solver='full')
X5 = pca_5.fit_transform(macro_zscore)

for x in alldata[names_factors].columns:
    reg = sm.OLS(endog=alldata[x],
             exog=sm.add_constant(X5)).fit()

    print("R2 for %s is %0.4f%%" % ( x, 100*reg.rsquared))

print('')

print('It tells us that the PCs are orthogonal to each other\n'
'and are thus uncorrelated')

# ! Question 3

'''
Estimate an expanding window out-of-sample forecast for each factor return on the first
5 macro PCs for 1980-01 onwards. Be sure to z-score and fit the PCs on training sample
only. Report the out-of-sample R2 for each factor, versus the historical mean benchmark
'''

print('### Question 3 ###')
print('')


OOSfore = pd.DataFrame(index=alldata.loc['1980-01':].index,
                    data=np.nan,columns=factors.columns)

OOSmean = OOSfore.copy()




for names in names_factors:
    for t in alldata.loc['1980-01':].index:
        y = alldata[names].loc[:t-1]
        x = sct.zscore(alldata[names_macro].loc[:t].values,axis=0)
        z = pca_5.fit_transform(x)
        reg = sm.OLS(endog=y,exog=sm.add_constant(z[:-1,:])).fit()
        OOSfore.at[t, names] = sm.add_constant(z)[-1,:]@reg.params
        OOSmean.at[t,names] = alldata[names].loc[:t-1].mean()
        
    yk = alldata[names].loc['1980-01':].values
    yfk = OOSfore[names].values
    ymk = OOSmean[names].values
    R2 = 1 - np.sum( (yk-yfk)**2 )/np.sum( (yk-ymk)**2 )
    print("R2 for %s is %0.4f%%" % ( names, 100*R2))