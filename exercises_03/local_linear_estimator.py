from __future__ import division
import numpy as np
import pdb
import pandas as pd
import matplotlib.pyplot as plt
import pdb
from numpy.linalg import inv
from kernel_smooth import gaussian_kernel
from cross_validation import error_calc
from scipy.optimize import minimize

def local_poly_estimate(x_data, y_data,x_star,h, kernel = gaussian_kernel, d=2):
    """

    Description

    Parameters
    ----------
    x_data: array_like
        The x values from the observed data
    y_data: array_like
        The y values from the observed data
    x_star: array_like
        The x locations to make predictions for
    h: Float
        The bandwidth
    kernel: func
        The kernel function to use
    d: int
        The order of the polynomial

    Returns
    -------
    y_star: array_like
        The predicted y_values corresponding to x_star
    H: array_like
        The projection matrix 
    """
    #Creating numpy array versions of the x and y data so everything plays nicely
    x = np.array(x_data)
    y = np.array(y_data)

    y_star = np.zeros(len(x_star))
    #initialize H, the projection matrix
    H = np.zeros([len(x_star), len(x_data)])
    #Creating the R matrix
    d = 2
    for i,x_val in enumerate(x_star):
        #Making the general R matrix
        R = np.zeros([len(x),d])
        for j in range(d):
            R[:,j] = (x - x_val)**j
        #Making the W matrix
        W = np.diag(1/h * kernel((x-x_val)/h))

        a_hat = inv(np.transpose(R)@W@R)@np.transpose(R)@W@y
        H[i,:] = (inv(np.transpose(R)@W@R)@np.transpose(R)@W)[0,:]
        y_star[i] = a_hat[0]


    return y_star, H

def leave_one_out(x_data, y_data, estimator,h, kernel=gaussian_kernel):
    #Creating numpy array versions of the x and y data so everything plays nicely
    x = np.array(x_data)
    y = np.array(y_data)
    error = np.zeros(len(x_data))
    for i in range(len(x_data)):
        y_star = estimator(np.delete(x,i), np.delete(y,i), [x[i]], h, kernel)[0]
        error[i] = error_calc(y_star, [y_data[i]])
    return np.mean(error)





#First load the data
df = pd.read_csv('utilities.csv')

df = df.sort_values('temp').reset_index()
#Getting desired x and y from the dataframe
x_data = df['temp']
y_data = df['gasbill']/df['billingdays']

#Getting the optimal h
h = minimize(lambda x: leave_one_out(x_data, y_data, local_poly_estimate, x), 1.0)['x']
print(h)
#Generating array of x values we want to predict
#x_star = np.linspace(min(x_data),max(x_data),50)
y_star, H = local_poly_estimate(x_data, y_data, x_data, h)

#Getting the residuals
residuals = y_data - y_star
plt.scatter(x_data, residuals)
plt.show()
plt.close()

#Determining the lower and upper bound to plot the fit
RSS = (np.sum(residuals**2))**.5
sigma_2 = RSS/(len(x_data) + 2*np.matrix.trace(H) + np.matrix.trace(np.transpose(H)@H))
lower = y_star - 1.96*sigma_2**.5
upper = y_star + 1.96*sigma_2**.5
pdb.set_trace()


#Making the plot of the estimate
plt.scatter(x_data,y_data,marker='x')
plt.plot(x_data,y_star,color='red')
plt.plot(x_data,lower, color='red',linestyle='--')
plt.plot(x_data,upper, color='red', linestyle='--')
plt.show()
plt.savefig('local_estimate')


