import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import r2_score

import streamlit as st

st.title("CSV File user:")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type= ["CSV"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    st.write("Preview of the dataset:")
    st.dataframe(data.head())
    st.write(f"Number of Rows: {data.shape[0]}")
    st.write(f"Number of Columns: {data.shape[1]}")
    st.write(f"Number of Rows: {data.shape[0]}")
    st.write(f"Number of Columns: {data.shape[1]}")
    st.dataframe(data.head())
    
    #data.replace('?', np.nan, inplace=True)
    

    drop_cols = st.multiselect("Select columns to drop", data.columns)
    
    # Drop selected columns and display the updated DataFrame
    if drop_cols:
        data = data.drop(columns=drop_cols)
        st.write(f"Dataset after dropping columns: {"drop_cols"}")
        st.dataframe(data.head())

    st.dataframe(data.drop([]))

    selected_column = st.selectbox("Select a column to create dummy variables:", data.columns)
    if st.button("Create Dummy Variables"):
        data_encoded = pd.get_dummies(data, columns=[selected_column])
        st.write(f"DataFrame with Dummy Variables for '{selected_column}':")
        st.dataframe(data_encoded)
    
    dependent_variable=st.selectbox("Select a column to create a dependent variable(y):",data.columns)

    if st.button("separate variables"):


        x=data.drop(columns=dependent_variable)
        y=data.drop(columns=x)
        st.dataframe(x)
        st.dataframe(y)
    

    data = data.apply(pd.to_numeric, errors='coerce')
    data.fillna(data.median(numeric_only=True), inplace=True)
    fill_method = st.selectbox("Select a method to fill missing values:", ["Median", "Mean", "Mode"])
    if st.button("Fill Missing Values"):
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        if fill_method == "Median":
            data.fillna(data.median(), inplace=True)
        elif fill_method == "Mean":
            data.fillna(data.mean(), inplace=True)
        elif fill_method == "Mode":
            data.fillna(data.mode().iloc[0], inplace=True)

        st.write("Missing Values are Filled:")
        st.dataframe(data)



    
else:
    st.write("Please upload a CSV file to display its contents.")




# streamlit run "C:\Users\Lenovo\Documents\VS CODE\Linear_regression\lasso_app.py" 