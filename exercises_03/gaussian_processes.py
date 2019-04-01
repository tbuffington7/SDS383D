import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb




def matern(x, b, tau_sq1, tau_sq2, ):
    #Determines the covariance function for a set of points and returns the value of random process
    #initializing the matrix
    cov_mat = np.zeros([len(x), len(x)])
    for i,x1 in enumerate(x):
        for j,x2 in enumerate(x):
            cov_mat[i,j] = tau_sq1*np.exp(-.5*(np.linalg.norm([x1,x2])/b)**2)+tau_sq2*float(x1==x2)

    return cov_mat
#Defining the set of points over a unit interval
x = np.linspace(0,1,100)
mean = np.zeros(len(x))
C = matern(x,1,10**-0, 0.0)

plt.plot(x,np.random.multivariate_normal(mean,C))
plt.show()

