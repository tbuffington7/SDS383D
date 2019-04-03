from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
from methods import *
import seaborn as sns
from tqdm import tqdm
import matplotlib

df = pd.read_csv('utilities.csv')
df = df.sort_values('temp').reset_index()
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
plt.show()
plt.close()

n =20 
b_vec = np.linspace(40,80,n)
tau_vec = np.linspace(20,80,n)
p = np.zeros((n,n))
for i,b in tqdm(enumerate(b_vec)):
    for j,tau in enumerate(tau_vec):
        p[i,j] = log_likelihood(x, y, sigma_2, b, tau)

#Then making the plot
fig,ax = plt.subplots()
cs = plt.contour(tau_vec, b_vec, p)
norm= matplotlib.colors.Normalize(vmin=cs.cvalues.min(), vmax=cs.cvalues.max())
sm = plt.cm.ScalarMappable(norm=norm, cmap = cs.cmap)
sm.set_array([])
cb = fig.colorbar(sm, ticks=cs.levels)
cb.set_label('log likelihood')
ax.set_xlabel(r'$\tau_1^2$')
ax.set_ylabel('b')
plt.savefig('contour')

#Printing the optimal values
print('optimal tau_1^2 is: '+ str(tau_vec[np.argwhere(p==np.max(p))[0][1]]))
print('optimal b is: '+ str(b_vec[np.argwhere(p==np.max(p))[0][0]]))




