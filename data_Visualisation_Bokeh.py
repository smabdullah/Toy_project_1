# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 13:21:10 2019

@author: SM
"""
# Importing library
import pandas as pd
import numpy as np
from bokeh.plotting import figure, output_file, show

# Read data
dataset = pd.read_csv('airfoil_self_noise.dat', sep = '\t', header = None, names = [
        'Frequency', 'Angle of attack', 'Chord length', 'Free-stream velocity', 'suction thickness', 'sound pressure level'])
# feature matrix
X = dataset.iloc[:,:-1].values
# target vector
y = dataset.iloc[:,5].values