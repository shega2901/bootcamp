# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 18:14:13 2021

@author: Oleksandr Sheyngart


Logistic regression model on IRIS
"""


from sklearn.datasets import load_iris
iris_dataset = load_iris()


for x in range(len(iris_dataset['target_names'])):
    print("Flower: {}, value in target's array: {}".format(iris_dataset['target_names'][x],x))
    
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=42)
'''
print('Shape of X_train: {}'.format(X_train.shape))
print('Shape of y_train: {}'.format(y_train.shape))
print('Shape of X_test: {}'.format(X_test.shape))
print('Shape of y_test: {}'.format(y_test.shape))
'''

#Building a model using method Logistic regression 
#Classifier implementing the Logistic regression: LogisticRegression

'''
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
print(X_train)
X_test=sc.transform(X_test)
print(X_test)
'''
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression(C=1)
logreg.fit(X_train, y_train)

#getting a forecast
#new flower: sepal's length = 5cm, sepal's width = 2.9 cm, petal's length = 1 cm, petal's width = 0.2 cm
import numpy as np
X_new=np.array([[5,2.9,1,0.2]])
print("\nShape of array X_new: {}".format(X_new.shape))

#Determine the name of the flower
prediction = logreg.predict(X_new)
for i in range(len(prediction)):
     print("Sizes if flower: {}\nName of the flower: {}".format(X_new[i],iris_dataset['target_names'][prediction[i]]))

#Correctness of the model
#Using the method 'score' of object 'logreg'. Using X_test,y_test
y_pred=logreg.predict(X_test)
print("Test: {}\nPredict: {}".format(y_test,y_pred))
accuracy = logreg.score(X_test,y_test)
print("Accuracy of the model: {:.4f}" .format(accuracy))
