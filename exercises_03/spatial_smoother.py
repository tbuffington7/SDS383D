from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
from methods import *
import seaborn as sns
from tqdm import tqdm

df = pd.read_csv('utilities.csv')
x = np.array(df['temp'].copy()).astype(float)
y = np.array(df['gasbill']/df['billingdays'])
sigma_2 = 1
y_pred,cov = gp_predict(x,y,x,sigma_2, 30,100,10**-6, cov_fun=matern52)
#make estimate of sigma_2 after fitting function
sigma_2 = np.sum((y_pred-y)**2)/(float(len(x))-1.0)
#then run it again with the more informed guess for sigma_2
y_pred,cov = gp_predict(x,y,x,sigma_2, 30,100,10**-6,cov_fun=matern52)
se = np.sqrt(np.diag(cov))
#generating the confidence bands
lower = y_pred - 1.96*se
upper = y_pred + 1.96*se



#then make the plot
sns.set()
plt.plot(x, y_pred, linestyle='--', color='k')
plt.fill_between(x,lower, upper)
plt.scatter(x,y,marker='x', color='mediumvioletred')
plt.xlabel('Average temperature ($^o$ F)')
plt.ylabel('Average daily cost (USD)')
plt.savefig('confidence_band')
plt.close()

n = 10
b_vec = np.linspace(40,80,n)
tau_vec = np.linspace(20,80,n)
p = np.zeros((n,n))
for i,b in tqdm(enumerate(b_vec)):
    for j,tau in enumerate(tau_vec):
        p[i,j] = log_likelihood(x, y, sigma_2, b, tau)

pdb.set_trace()

