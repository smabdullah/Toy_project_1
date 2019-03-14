# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 11:24:00 2019

@author: SM
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 10:00:00 2019

@author: SM Abdullah
"""
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing the dataset
dataset = pd.read_csv('airfoil_self_noise.dat', sep = '\t', header = None, names = [
        'Frequency', 'Angle of attack', 'Chord length', 'Free-stream velocity', 'suction thickness', 'sound pressure level'])
# feature matrix
X = dataset.iloc[:,:-1].values
# target vector
y = dataset.iloc[:,5].values

# Spliting the dataset into trining_set and test_set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Linear model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

# Predict y
y_pred = regressor.predict(X_test)

# Confusion matrix
from sklearn.metrics import r2_score, mean_squared_error
R2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)


