# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 13:10:38 2020

@author: Trill
"""

import os
import numpy as np
import pandas as pd
import string


def get_mnist_dataset():
    test_data_path = os.path.join("./datasets",
        "sign-language-mnist", 
        "sign_mnist_test","")
    train_data_path = os.path.join("./datasets", 
        "sign-language-mnist", 
        "sign_mnist_train","")

    test_pd = pd.read_csv(test_data_path + "sign_mnist_test.csv", 
        skiprows=1)
    train_pd = pd.read_csv(train_data_path + "sign_mnist_train.csv", 
        skiprows=1)
    
    return train_pd, test_pd

train_pd, test_pd = get_mnist_dataset()
train_images, test_images = train_pd.values[:,1:], test_pd.values[:,1:]
train_labels, test_labels = train_pd.values[:,0], test_pd.values[:,0]

class_names = list(string.ascii_lowercase)

#%% 
# Normalize data
x_train = train_images / 255
x_test = test_images / 255

# Principal Component Analysis (PCA) 
# Reduce dimensions to contain 95% of variance and discard the rest 

from sklearn.decomposition import PCA 

pca = PCA(n_components=0.95) 
train_reduced = pca.fit_transform(x_train)
test_reduced = pca.fit_transform(x_test) 

print(train_reduced.shape)
print(test_reduced.shape)
# Remember to decompress when showing the picture. 
# Some data will be lost
# train_recovered = pca.inverse_transform(train_reduced)
# test_recovered = pca.inverse_transform(test_reduced)


#%% 

# Linear Regression Model using Grid Search 
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

model = svm.SVC(gamma=0.001)

tuning_parameters = {
    'kernel':('linear', 'rbf'), 
    'C':[1, 10]
}

CV=5
VERBOSE=0 

# Grid search for model 
grid_tuned = GridSearchCV(model, tuning_parameters, cv=CV, scoring='f1_micro', verbose=VERBOSE, n_jobs=None, iid=True)
grid_tuned.fit(train_reduced, train_labels)



