from __future__ import division
import numpy as np
import pdb
import pandas as pd
import matplotlib.pyplot as plt
import pdb
from methods import *
from numpy.linalg import inv
from scipy.optimize import minimize

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
plt.savefig('local_estimate')


