import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from numpy.linalg import inv
import pdb


'''
Recap of definitions derived in written notes:
m_star = inv(np.transpose(x)@Lambda@x + K) @ (np.transpose(x) @ Lambda @ y + K @ m)
K_star = np.transpose(x)@Lambda@x + K
d_star = d + n
eta_star = eta + np.transpose(y) @ Lambda @ y + np.transpose(m) @ K @ m - m_star.transpose() @ K_star @ m 



Conditional distributions:
p(beta|y,w, Lambda) ~ MVN(m_star, (w*K_star)^-1)
p(w|y,Lambda) ~ Gamma(d_star/2, eta_star/2)
p(l[i], y, beta, w ) ~ Gamma((h+1)/2, w/2.0*(y - x@beta)**2 + h/2.0)

'''


def gibbs_sampler(x,y,iterations=1000,burn=100, h=1.0, Kvals = 0.01, d = 1.0, eta=1.0, w=1.0):

    n = len(y) 
    p = len(x.columns)

    #initialize parameters
    K = np.diag(np.ones(p))*Kvals #A vague prior for the K precision
    m = np.zeros(p)
    Lambda = np.diag(np.ones(n))
    beta = np.zeros(p)

    #Storing the results in a dictionary
    results = {

            'beta_trace': np.zeros([iterations, p]),
            'Lambda_trace': np.zeros([iterations, n]),
            'w_trace': np.zeros(iterations),

    }

    for i in range(iterations):

        #Update beta
        m_star = inv(np.transpose(x) @ Lambda @ x + K) @ (np.transpose(x) @ Lambda @ y + K @ m)
        pdb.set_trace()
        K_star = np.transpose(x) @ Lambda @ x + K
        beta = stats.multivariate_normal.rvs(mean = m_star, cov=inv(w*K_star))

        #Update w
        d_star = d + n
        eta_star = eta + np.transpose(y) @ Lambda @ y + np.transpose(m) @ K @ m - np.transpose(m_star) @ inv(K_star) @ m_star
        w = stats.gamma.rvs(d_star/2.0, (2.0/eta_star))

        #Update lambda
        # Lambda = np.diag([stats.gamma.rvs((h+1.0)/2.0, (w/2.0*(y[j] - np.transpose(x.iloc[j])@beta)**2 + h/2.0)**-1.0)  for j in range(n)])

        lambda_diag = np.zeros(n)
        for j in range(n):
            lambda_diag[j] = stats.gamma.rvs((h+1)/2, (2/(h+w*(y[j] - np.transpose(x.iloc[j])@beta)**2)))
        Lambda = np.diag(lambda_diag)


        results['beta_trace'][i,:] = beta
        results['Lambda_trace'][i,:] = np.diagonal(Lambda)
        results['w_trace'][i] = w

        print('{:d} iterations out of {:d} complete.'.format(i,iterations))

    #Postprocessing traces
    for key in results.keys():
        #Applying burn in
        results[key] = results[key][burn:]


    #Making trace plots and getting mean predictors
    results['beta_mean'] =  np.zeros(p)
    for j in range(p):
        plt.plot(range(len(results['beta_trace'][:,j])), results['beta_trace'][:,j])
        plt.xlabel('Iteration')
        plt.ylabel('beta')
        plt.savefig('trace'+str(j))
        plt.close()
        results['beta_mean'][j] = np.mean(results['beta_trace'][:,j])


    return results




#Loading the data into a pandas dataframe
df = pd.read_csv('gdpgrowth.csv')
#append an intercept column of ones
df['intercept'] = np.ones(len(df))



#Pulling x and y from the dataset
x = df[ [ 'intercept', 'DEF60']].copy()
y = df['GR6096']


results = gibbs_sampler(x,y)


plt.scatter(x['DEF60'],y)
plt.plot(x['DEF60'], x@results['beta_mean'], linestyle='--', color='red')
plt.savefig('heavy_tail_fit')



