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


C = matern(x,10**-6,10**-6, 10**-6)
plt.plot(x,np.random.multivariate_normal(mean,C))
plt.savefig('base_case')
plt.close()

C = matern(x,10**-3,10**-6, 10**-6)
plt.plot(x,np.random.multivariate_normal(mean,C))
plt.savefig('increased_b')
plt.close()


C = matern(x,10**-6,10**-3, 10**-6)
plt.plot(x,np.random.multivariate_normal(mean,C))
plt.savefig('increased_tau1')
plt.close()


C = matern(x,10**-6,10**-6, 10**-3)
plt.plot(x,np.random.multivariate_normal(mean,C))
plt.savefig('increased_tau2')
plt.close()


C = matern52(x,10**-6,10**-6, 10**-6)
plt.plot(x,np.random.multivariate_normal(mean,C))
plt.savefig('matern52_case')
plt.close()
