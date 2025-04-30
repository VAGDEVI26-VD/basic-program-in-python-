import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv(r"C:\Users\Lenovo\Documents\Notes\APRIL\22nd- Cross validation\Wholesale customers data.csv")
data.head()

data.shape
data.info()
data.descrbe()
data.isnull().sum()

x=data.drop(['Channel'],axis=1)
y=data['Channel']

x.head()
y.head()

y[y==2]=0
y[y==1]=1

y.head()


import xgboost as xgb
data_dmatrix=xgb.DMatrix(data=x,label=y)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)


from xgboost import XGBClassifier
params={'objective': "binary:logistic","max_depth":4,'alpha':10,'learning_rate':1.0,'n_estimators':100}

xgb_clf=XGBClassifier(**params)
xgb_clf.fit(x_train,y_train)

y_pred=xgb_clf.predict(x_test)


from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test, y_pred)
ac


# K-FOLD:
    
from xgboost import cv
params = {"objective":"binary:logistic",'colsample_bytree': 0.3,'learning_rate': 0.1,'max_depth': 5, 'alpha': 10}
xgb_cv = cv(dtrain=data_dmatrix, params=params, nfold=3,num_boost_round=50, early_stopping_rounds=10, metrics="auc", as_pandas=True, seed=123)

xgb_cv.head()


#feature importance of xgboost:
    
xgb.plot_importance(xgb_clf)
plt.figure(figsize=(16,12))
plt.show()
