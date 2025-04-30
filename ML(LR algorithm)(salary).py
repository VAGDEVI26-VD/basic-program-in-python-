import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

salary=pd.read_csv(r"C:\Users\Lenovo\Documents\Notes\MARCH\19th work\Salary_Data.csv")
salary.describe()
salary.isnull().sum()
x=salary.iloc[:,:-1]
y=salary.iloc[:,-1]

# .iloc[] is an indexer used for integer-location-based indexing of data in a DataFrame
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
x_train=x_train.values.reshape(-1,1)
x_test=x_test.values.reshape(-1,1)


#inearRegression fits a linear model with coefficients w = (w1, ..., wp)
# to minimize the residual sum of squares between the observed 
#targets in the dataset, and the targets predicted by the 
#linear approximation.====>>residual=error we can reduce by loss function or 
#   ---->> reduce by the distance of actual and predict

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

#comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
#print(comparison)

#visualization:train
plt.scatter(x_train, y_train, color = 'red')  # Real salary data (training)
plt.plot(x_train, regressor.predict(x_train), color = 'blue')  # Predicted regression line
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#test
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience(Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()



coef=print(f"Coefficient:{regressor.coef_}")
intercept=print(f"Intercept:{regressor.intercept_}")

comparison=pd.DataFrame({'Actual':y_test,'predicted':y_pred})
print(comparison)

# future pred code:whose expr 12 0r 20 year future :m=coef,c=intercept,x=12

exp_12_future_pred= 9312*12+26780
exp_12_future_pred

bias=regressor.score(x_train,y_train)
print(bias)

variance=regressor.score(x_test,y_test)
print(variance)

# we implement statistic to this dataset :

salary.mean()
salary["Salary"].mean()
salary.median()
salary["Salary"].median()
salary.mode()
salary["Salary"].mode()
salary.var()
salary["Salary"].var()
salary.std()
salary["Salary"].std()

from scipy.stats import variation
variation(salary.values) # this will give cv of entire dataframe 
variation(salary["Salary"])

salary.corr()
salary["Salary"].corr(salary["YearsExperience"])
salary.skew()
salary["Salary"].skew()
salary.sem() # this will give standard error of entire dataframe   
salary["Salary"].sem()  # this will give standard error particular column

#calculate z-score

import scipy.stats as stats
salary.apply(stats.zscore)
   
stats.zscore(salary["Salary"])

# degree of freedom

a=salary.shape[0]
b=salary.shape[1]
degree_of_freedom=a-b
print(degree_of_freedom)

# SSR(Sum of Squares Regression)
y_mean=np.mean(y)
SSR=np.sum((y_pred-y_mean)**2)
print(SSR)

#SSE(SUM OF SQUARES OF ERROR)
y=y[0:6]
SSE=np.sum((y-y_pred)**2)
print(SSE)

#SST(SUM OF SQUARES TOTAL)
mean_total=np.mean(salary.values)
SST=np.mean((salary.values-mean_total)**2)
print(SST)

# R-SQUARE:
r_square=SSR/SST
r_square
 