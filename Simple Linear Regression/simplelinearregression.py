# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 21:31:21 2020

@author: Saurav Chaudhary
"""

#Simple Linear Regression
#Data Preprocessing

#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the DataSet
dataset = pd.read_csv('Salary_Data.csv')
#Years of Experience
X = dataset.iloc[:,:-1].values
#Salary
Y = dataset.iloc[:,1].values

#Spliting the dataset into Training Set and Test Set
from sklearn.model_selection import train_test_split 
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)

#Fitting Simple Linear Regression to The Training Set
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#Predicting the Test Set Results
#A vector of prediction of dependent variables
Y_Pred = regressor.predict(X_test)

#Visualising the Training set results
plt.scatter(X_train,Y_train,color = 'red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Salary Vs Experience(Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

#Visualising the Test set results
plt.scatter(X_test,Y_test,color = 'red')
#We dont need to change this because our regressor is already trained on training set
#So doing this we will get only some new points
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Salary Vs Experience(Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()