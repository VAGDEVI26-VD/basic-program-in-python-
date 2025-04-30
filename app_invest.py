import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


st.title("Analysis on Investment")
st.write("This application analyzes the all the investment data from all the differnt categories")

st.title("📂 Upload Your File")
upload_your_file=st.file_uploader("Upload your file,type=None")
inv=None
if upload_your_file is None:
    inv=pd.read_csv(upload_your_file)
    st.write("File uploaded")
    st.subheader("File symbol DATASET")
    st.write(inv.head())
else:
    st.warning("Please upload a file.")

#inv.describe()
#inv.isnull().sum()
x=inv.iloc[:,:-1]
y=inv.iloc[:,4]

x=pd.get_dummies(x,dtype=int)

#from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

#model build
#from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train, y_train)

y_pred=regressor.predict(x_test)


# feature elimination--->> to get the right attribute to invest
m_slope=regressor.coef_
print(m_slope)

c_inter=regressor.intercept_
print(c_inter)

x=np.append(arr=np.full(((50,1)),42467).astype(int),values=x,axis=1)

#import statsmodels.api as sm
x_opt=x[:,list(range(x.shape[1]))]
#ordinaryleastSquare
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
summary_text=regressor_OLS.summary()

st.subheader("Result")
st.text(summary_text)


bias=regressor.score(x_train,y_train)
print(bias)

variance=regressor.score(x_test,y_test)
print(variance)

st.subheader("🎯 Model Evaluation")
st.write(f"**Training Accuracy (Bias):** {bias:.2f}")
st.write(f"**Testing Accuracy (Variance):** {variance:.2f}")