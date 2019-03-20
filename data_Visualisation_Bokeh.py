# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 13:21:10 2019

@author: SM
"""
# Importing library
import pandas as pd
import numpy as np
from bokeh.plotting import figure, output_file, save

# Read data
dataset = pd.read_csv('airfoil_self_noise.dat', sep = '\t', header = None, names = [
        'Frequency', 'Angle of attack', 'Chord length', 'Free-stream velocity', 'suction thickness', 'sound pressure level'])
# feature matrix
X = dataset.iloc[:,:-1].values
# target vector
y = dataset.iloc[:,5].values

# Scale the feature matrix and the dependant variable
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X = sc_x.fit_transform(X)

# Visualise some data
p = figure(plot_width = 800, x_axis_label = 'Features',
           y_axis_label = 'sound pressure level')
p.circle(X[:,0], y, size = 10, fill_color = 'red', fill_alpha = 0.5)
p.circle(X[:,1], y, size = 15, fill_color = 'blue', fill_alpha = 0.3)
p.circle_cross(X[:,2], y, size = 20, fill_color = 'lightblue', fill_alpha = 0.3)
p.circle_x(X[:,3], y, size = 20, fill_color = 'maroon', fill_alpha = 0.8)
p.diamond(X[:,4], y, size = 20, fill_color = 'coral', fill_alpha = 0.1)
output_file('out_1.html', title = 'Feature vs sound pressure', mode = 'inline')
save(p)
