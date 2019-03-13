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

