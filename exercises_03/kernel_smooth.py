import numpy as np
import matplotlib.pyplot as plt
import pdb
from methods import *


#Create a vector of x values
x = np.linspace(0,10,100)
#Use sin(x) as the function and add Gaussian noise
y = [np.sin(i) + np.random.normal(0,0.1) for i in x]

#Define x_star: the x locations we're interested in predicting y
x_star_vec = np.linspace(0,10,100)

#initialize y_star: the corresponding predictions
y_star_vec = np.zeros(len(x_star_vec))

#Initializing array of h values for comparison
h_vec = [0.05, 0.1, 1.0]

#Making the plot of the noisy data outside of for loops
plt.scatter(x,y, marker = 'x')
for h in h_vec:
    y_star_vec = pred(x,y,x_star_vec,h)
    plt.plot(x_star_vec, y_star_vec)

plt.legend(['h=0.05', 'h=0.1', 'h=1.0', 'raw_data'])
plt.savefig('gaussian_kernel')
plt.close()


