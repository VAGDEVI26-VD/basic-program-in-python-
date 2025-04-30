import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


inv=pd.read_csv(r"C:\Users\Lenovo\Documents\sypder\Investment.csv")
inv.describe()
inv.isnull().sum()
x=inv.iloc[:,:-1]
y=inv.iloc[:,4]

x=pd.get_dummies(x,dtype=int)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

#model build
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train, y_train)

y_pred=regressor.predict(x_test)


# feature elimination--->> to get the right attribute to invest
m_slope=regressor.coef_
print(m_slope)

c_inter=regressor.intercept_
print(c_inter)

x=np.append(arr=np.full(((50,1)),42467).astype(int),values=x,axis=1)

import statsmodels.api as sm
x_opt=x[:,[0,1,2,3,4,5]]
#ordinaryleastSquare
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()


import statsmodels.api as sm
x_opt=x[:,[0,1,2,3,5]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
x_opt=x[:,[0,1,2,3]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
x_opt=x[:,[0,1,3]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
x_opt=x[:,[0,1]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()


bias=regressor.score(x_train,y_train)
print(bias)

variance=regressor.score(x_test,y_test)
print(variance)



