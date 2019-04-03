import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
from methods import *

#Defining the set of points over a unit interval
x = np.linspace(0,1,100)
mean = np.zeros(len(x))

#b_vec = [10**-6, 1.0]
#tau1_sq = [10**-6, 1.0]
#tau2_sq = [10**-6, 1.0]


C = matern(x,.05,.01, 10**-6)
plt.plot(x,np.random.multivariate_normal(mean,C))
C = matern52(x,.05,.01, 10**-6)
plt.plot(x,np.random.multivariate_normal(mean,C))
plt.legend(['squared exponential', 'matern52'])
plt.savefig('base_case')
plt.close()

C = matern(x,.2,.01, 10**-6)
plt.plot(x,np.random.multivariate_normal(mean,C))
C = matern52(x,.2,.01, 10**-6)
plt.plot(x,np.random.multivariate_normal(mean,C))
plt.legend(['squared exponential', 'matern52'])
plt.savefig('increased_b')
plt.close()


C = matern(x,.05,1.0, 10**-6)
plt.plot(x,np.random.multivariate_normal(mean,C))
C = matern52(x,.05,1.0, 10**-6)
plt.plot(x,np.random.multivariate_normal(mean,C))
plt.legend(['squared exponential', 'matern52'])
plt.savefig('increased_tau1')
plt.close()


C = matern(x,.05,.01, 1.0)
plt.plot(x,np.random.multivariate_normal(mean,C))
C = matern52(x,.05,.01, 1.0)
plt.plot(x,np.random.multivariate_normal(mean,C))
plt.legend(['squared exponential', 'matern52'])
plt.savefig('increased_tau2')
plt.close()


