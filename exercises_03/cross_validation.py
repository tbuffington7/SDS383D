import numpy as np
import matplotlib.pyplot as plt
import pdb
from methods import *


#Create a vector of x values. This is the training set.
x = np.linspace(0,10,100)
#Use sin(x) as the function and add Gaussian noise
y = [np.sin(i) + np.random.normal(0,0.1) for i in x]

#Define x_star: the x locations we're interested in predicting y
x_star_vec = np.linspace(0,10,100)

#Initializing array of h values for comparison
h_vec = [0.05, 0.1, 1.0]

#initialize y_star: the corresponding predictions for each h
y_star_mat = np.zeros((len(x_star_vec), len(h_vec)))

#Fitting the predictions
for i,h_val in enumerate(h_vec):
    for j, x_star in enumerate(x_star_vec):
        weights = weight_calc(x, x_star,h_val, gaussian_kernel)
        y_star_mat[j,i] = weights@y #matrix multiplciation

#Generating new data
y_new = [np.sin(i) + np.random.normal(0,0.1) for i in x_star_vec]

error = np.zeros(len(h_vec))
for i in range(len(h_vec)):
    error[i] = error_calc(y_new, y_star_mat[:,i])










