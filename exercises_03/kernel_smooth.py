import numpy as np
import matplotlib.pyplot as plt
import pdb


#Create a vector of x values
x = np.linspace(0,10,100)
#Use sin(x) as the function and add Gaussian noise
y = np.sin(x) + np.random.normal(0,0.05)

#Define x_star: the x locations we're interested in predicting y
x_star = np.linspace(0,10,10)

#initialize y_star: the corresponding predictions
y_star = np.zeros(len(x_star))

def gaussian_kernel(x):
    #Takes an array x and returns array of Gaussian kernels of same length
    return '[
