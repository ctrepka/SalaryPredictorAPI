# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

def create_pkl():
    # Importing the dataset
    dataset = pd.read_csv('Salary_Data.csv')

    #get all columns in all rows, except for last column of each row (row to be predicted)
    X = dataset.iloc[:, :-1].values
    #set y equal to last column of all rows
    y = dataset.iloc[:, -1].values


    # Training the Multiple Linear Regression model on the Training set
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    #use OneHotEncoder to get unique values from column at index 0, which are categorical (teacher or coder)
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
    #transform array to append HotCoded categorical values to each row
    X = np.array(ct.fit_transform(X))

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state=0)

    #import LinearRegression
    from sklearn.linear_model import LinearRegression

    #initialize LinearRegression object model
    regressor = LinearRegression()
    #training the model from the x_train and y_train models
    regressor.fit(X_train, y_train)

    #creating dump / binary stream out of trained ML Dataset
    pickle.dump(regressor, open('model.pkl', 'wb'))






