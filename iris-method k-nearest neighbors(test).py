# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 18:14:13 2021

@author: Oleksandr Sheyngart

k-nearest neighbors model on IRIS

"""

from sklearn.datasets import load_iris
iris_dataset = load_iris()


#Dataset IRIS
print('Keys of iris_dataset: \n{}'.format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:1]+"\n...")
print('Target names: {}'.format(iris_dataset['target_names']))
print('feature names: {}'.format(iris_dataset['feature_names']))

#data's array
print("Type of data's array: {}".format(type(iris_dataset['data'])))
print("Shape data's array: {}".format(iris_dataset['data'].shape))
print('Data:\n{}'.format(iris_dataset['data'][:5]))



#target's array
print("Type of target's array: {}".format(type(iris_dataset['target'])))
print("Shape target's array: {}".format(iris_dataset['target'].shape))
print('Target: \n{}'.format(iris_dataset['target']))
for x in range(len(iris_dataset['target_names'])):
    print("Flower: {}, value in target's array: {}".format(iris_dataset['target_names'][x],x))

    
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)
print('Shape of X_train: {}'.format(X_train.shape))
print('Shape of y_train: {}'.format(y_train.shape))
print('Shape of X_test: {}'.format(X_test.shape))
print('Shape of y_test: {}'.format(y_test.shape))




#Building a model using method k-nearest neighbors 
#(Classifier implementing the k-nearest neighbors vote: KNeighborsClassifier)
from sklearn.neighbors import KNeighborsClassifier
#number neibhbors = 1
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)



#getting a forecast
#new flower: sepal's length = 5cm, sepal's width = 2.9 cm, petal's length = 1 cm, petal's width = 0.2 cm
import numpy as np
X_new=np.array([[5,2.9,1,0.2]])
print("\nShape of array X_new: {}".format(X_new.shape))
#Determine the name of the flower
prediction = knn.predict(X_new)
for i in range(len(prediction)):
     print("Sizes if flower: {}\nName of the flower: {}".format(X_new[i],iris_dataset['target_names'][prediction[i]]))




#Correctness of the model
#Using the method 'score' of object 'knn'. Using X_test,y_test
y_pred=knn.predict(X_test)
print("Test: {}\nPredict: {}".format(y_test,y_pred))
accuracy = knn.score(X_test,y_test)
print("Accuracy of the model: {:.2f}" .format(accuracy))
