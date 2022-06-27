# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 17:58:12 2022

@author: ruben
"""

import matplotlib.pyplot as plt
import pandas as pd

#exporting and understanding our data
data = pd.read_csv('SeoulBikeData.csv', encoding= 'unicode_escape')
print('Our data labeled in a table format:')
print(data.head())

print('Relevant statistical quantities')
print(data.describe)

print('Relevant computation quantities')
print (data.dtypes)
print (data.info())


#labeling our data
#X stands for data/ y stands for target 
X = data[['Hour',
          'Temperature(°C)',
          'Rainfall(mm)'
]].values

y = data['Rented Bike Count']
print(X,y)

print("X_shape: ", X.shape)
print("y_shape: ", y.shape)

#visualizing the data 
plt.scatter(X.T[0], y, s = 4, label='hours')
plt.scatter(X.T[1], y, s = 4,label='temperature')
plt.scatter(X.T[2], y, s = 4,label='rainfall')
plt.legend()
plt.show()

#separating in three plots for better understanding the units
fig, ax = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(17, 6)) 

ax[0].scatter(X.T[0], y, color='r', s = 4)
ax[0].set_title('Renting by hours of the date')
ax[0].set_ylabel('y label features')
ax[0].set_xlabel('X label features')

ax[1].scatter(X.T[1], y, color='b', s = 4)
ax[1].set_title('Renting by Temperature(°C)')
ax[1].set_ylabel('y label features')
ax[1].set_xlabel('X label features')

ax[2].scatter(X.T[2], y, color='g', s = 4)
ax[2].set_title('Renting by Rainfall(mm)')
ax[2].set_ylabel('y label features')
ax[2].set_xlabel('X label features')

plt.show()

#Let's begin the machine learning exploration!
from sklearn import linear_model
from sklearn.model_selection import train_test_split 

#splitting the data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

plt.scatter(X_test.T[1], y_test, s = 4, label = 'data points')
plt.title('Target points for model testing')
plt.show()

#model / algorithm and training - LINEAR REGRESSION
l_reg = linear_model.LinearRegression()
model  = l_reg.fit(X_train, y_train)
predictions = model.predict(X_test)
print('Predictions: ', predictions)
print('R^2 value: ', l_reg.score(X,y))
print('coefficient factor: ', l_reg.coef_)
print('intercept: ', l_reg.intercept_)

#identificamos overfitting mesmo realizando o split de teste.
#R^2 aprox accuracy na regressão linear, muito falha no teste de cross-validation
plt.scatter(X_test.T[1], y_test, s = 4, label = 'data points')
plt.scatter(X_test.T[1], predictions, color="black", s = 4, label = 'model predictions')
plt.title('Model predictions under linear regression regime')
plt.ylabel('y label features')
plt.xlabel('X label features')
plt.ylim([0, 3000])
plt.legend()
plt.show()

#model / algorithm and training - KNN REGRESSION
from sklearn import neighbors

knn = neighbors.KNeighborsRegressor(n_neighbors= 5, weights= 'uniform')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

knn.fit(X_train, y_train)

predictions = knn.predict(X_test)

print("predictions: ", predictions)
print("R^2 value: ", knn.score(X,y))


plt.scatter(X_test.T[1], y_test, s = 4, label = 'data points')
plt.scatter(X_test.T[1], predictions, color="black", s = 4, label = 'model predictions')
plt.title('Model predictions under KNN regression regime')
plt.ylabel('y label features')
plt.xlabel('X label features')
plt.ylim([0, 3000])
plt.legend()
plt.show()

#model / algorithm and training - SVM REGRESSION
from sklearn import svm

model =  svm.SVR(kernel='poly')
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("predictions: ", predictions)
print('R^2 value: ', model.score(X,y))

plt.scatter(X_test.T[1], y_test, s = 4, label = 'data points')
plt.scatter(X_test.T[1], predictions, color="black", s = 4, label = 'model predictions')
plt.title('Model predictions under SVM regression regime')
plt.ylabel('y label features')
plt.xlabel('X label features')
plt.ylim([0, 3000])
plt.legend()
plt.show()

