import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv(r"C:\Users\Lenovo\Documents\sypder\Data.csv")

# independent variable
x=df.iloc[:,:-1].values
# dependent variable
y=df.iloc[:,3].values

from sklearn.impute import SimpleImputer

imputer=SimpleImputer()

#imputer=SimpleImputer(strategy="mean")
#imputer=SimpleImputer(strategy="median")
#imputer=SimpleImputer(strategy="most_frequent")

# fit and transform used to fit the data and transform 
#         ---->>  the data for missing values

imputer=imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])

from sklearn.preprocessing import LabelEncoder
Labelencoder_x=LabelEncoder()
Labelencoder_x.fit_transform(x[:,0])
x[:,0]=Labelencoder_x.fit_transform(x[:,0])

#
Labelencoder_y=LabelEncoder()
y=Labelencoder_y.fit_transform(y)

# split data

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,train_size=0.8,random_state=0)
#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)








