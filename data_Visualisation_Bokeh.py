# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 13:21:10 2019

@author: SM
"""
# Importing library
import pandas as pd
import numpy as np
from bokeh.layouts import gridplot # for plotting subfigures
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

# Visualise data using multiple 1-D plot
output_file('Data_visualisation.html')

p1 = figure(width=250, height=250, title = 'Frequency vs sound pressure')
p1.circle(X[:,0], y, size = 10, color = 'indigo', alpha = 0.5)

p2 = figure(width=250, height=250, title = 'Angle of attack vs sound pressure')
p2.diamond(X[:,1], y, size = 10, color = 'navy', alpha = 0.5)

p3 = figure(width=250, height=250, title = 'chord length vs sound pressure')
p3.square(X[:,2], y, size = 10, color = 'firebrick', alpha = 0.5)

p4 = figure(width=250, height=250, title = 'Free-stream velocity vs sound pressure')
p4.triangle(X[:,3], y, size = 10, color = 'olive', alpha = 0.5)

p5 = figure(width=250, height=250, title = 'suction thickness vs sound pressure')
p5.circle_x(X[:,4], y, size = 10, color = 'sandybrown', alpha = 0.5)

# put all the sub-plots in a gridplot
p = gridplot([[p1, p2, p3, p4, p5]])

save(p)