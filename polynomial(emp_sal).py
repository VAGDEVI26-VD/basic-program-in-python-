import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv(r"C:\Users\Lenovo\Documents\Notes\APRIL\emp_sal.csv")
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)

plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title('linear regression model(linear regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

m=lin_reg.coef_
print(m)

c=lin_reg.intercept_
print(c)

lin_reg_pred = lin_reg.predict([[6.5]])
lin_reg_pred


#polynomial regression (NON linear model)====> change degree to 1 to 2

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures()  # degree
x_poly=poly_reg.fit_transform(x)

poly_reg.fit(x_poly,y)

lin_reg_2=LinearRegression()
lin_reg_2.fit(x_poly,y)

plt.scatter(x,y,color='red')
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)),color='blue')
plt.title('polymodel (polynomial regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#prediction
poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(poly_model_pred)

#support vector regression model  (SVR)

from sklearn.svm import SVR
svr_reg=SVR() #hyperparameter tuning
svr_reg.fit(x,y)

#predict
svr_model_pred=svr_reg.predict([[6.5]])
svr_model_pred

# K-NEARST NEIGHBOUR(KNN)

from sklearn.neighbors import KNeighborsRegressor
knn_reg=KNeighborsRegressor(n_neighbors=4,weights='uniform')
knn_reg.fit(x,y)

knn_model_reg=knn_reg.predict([[6.5]])
knn_model_reg


# DECISION TREE

from sklearn.tree import DecisionTreeRegressor
dec_reg=DecisionTreeRegressor(criterion="absolute_error",splitter="random")
dec_reg.fit(x,y)

dec_model_pred=dec_reg.predict([[6.5]])
dec_model_pred


# RANDOM FOREST

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(random_state=0)
rf_reg.fit(x,y)

rf_model_pred=rf_reg.predict([[6.5]])
rf_model_pred










