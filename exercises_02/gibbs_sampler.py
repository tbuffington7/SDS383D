import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from numpy.linalg import inv
import pdb

class heavy_tailed:
    """
    Notes
    -----

    This is a heavy tailed Bayesian class

    Recap of definitions derived in written notes:
    m_star = inv(np.transpose(x)@Lambda@x + K) @ (np.transpose(x) @ Lambda @ y + K @ m)
    K_star = np.transpose(x)@Lambda@x + K
    d_star = d + n
    eta_star = eta + np.transpose(y) @ Lambda @ y + np.transpose(m) @ K @ m - m_star.transpose() @ K_star @ m 



    Conditional distributions:
    p(beta|y,w, Lambda) ~ MVN(m_star, (w*K_star)^-1)
    p(w|y,Lambda) ~ Gamma(d_star/2, eta_star/2)
    p(l[i], y, beta, w ) ~ Gamma((h+1)/2, w/2.0*(y - x@beta)**2 + h/2.0)

    Attributes
    ----------
    x: array_like
        The dataset after an intercept is fitted to it
    y: array_like
        The y data
    traces: dict
        A dictionary of the traces for each parameter
`   beta_mean: array_like
        The mean of the posterior distributions for each predictor

    """
    def __init__(self,x_data,y_data,iterations=1000, h=1.0, Kvals = 0.01, d = 1.0, 
                   eta=1.0, w=1.0, add_intercept=True):
        """

        A heavy tailed linear error model. Involves a Gibbs sampler.

        Parameters
        ----------
        x: array_like
            The x values from the input data
        y: array_like
            The y values from the input data
        iterations: int
            The number of iterations for the Gibbs sampler to complete
        h: float
            Gamma distribution hyperparamter
        Kvals: float
            The values to populate the diagonal of the K matrix
        d: float
            Gamma prior hyper parameter
        eta: float
            Gamma distribution hyper parameter
        w: float
            omega prior as described in written notes
        add_intercept: bool
            Option to add column of ones to x_data

        Returns
        -------
        A heavy_tailed object with attributes described in the class docstring

        """

        #This is flipped, but it's a quirk of numpy
        n = np.shape([x_data])[1]
        p = np.shape([x_data])[0]

        #If add_intercept is True, add a column of ones to the x matrix
        if add_intercept == True:
            p += 1
            self.x = np.ones([n,p])
            self.x[:,1:] = np.transpose([x_data])
        else:
            self.x = np.array(x_data)

        self.y = y_data

        #Storing the traces in a dictionary
        self.traces = {

                'beta_trace': np.zeros([iterations, p]),
                'Lambda_trace': np.zeros([iterations, n]),
                'w_trace': np.zeros(iterations),

        }

        self.gibbs_sampler(iterations, n, p, h, Kvals, d, eta, w)

    def gibbs_sampler(self,iterations, n, p, h, Kvals, d, eta, w):

        """

        Description

        Parameters
        ----------
        iterations: int
            The number of iterations to complete
        n: int
            The number of observations
        p: int
            The number of predictors
        Kvals: float
            The value to fill the diagonal terms of the K matrix
        d: float
            parameter for the gamma prior
        eta: float
            parameter for the gamma prior
        w: float
            The prior for omega

        Populates self.trace and self.beta_mean

        """
        #initialize parameters
        K = np.diag(np.ones(p))*Kvals
        m = np.zeros(p)
        Lambda = np.diag(np.ones(n))
        beta = np.zeros(p)

        for i in range(iterations):

            #Update beta
            m_star = inv(np.transpose(self.x) @ Lambda @ self.x + K) \
            @ (np.transpose(self.x) @ Lambda @ self.y + K @ m)

            K_star = np.transpose(self.x) @ Lambda @ self.x + K
            beta = stats.multivariate_normal.rvs(mean = m_star, cov=inv(w*K_star))

            #Update w
            d_star = d + n

            eta_star = eta + np.transpose(self.y) @ Lambda @ self.y + np.transpose(m) @ K @ m - \
            np.transpose(m_star) @ inv(K_star) @ m_star

            w = stats.gamma.rvs(d_star/2.0, (2.0/eta_star))

            lambda_diag = np.zeros(n)
            for j in range(n):
                lambda_diag[j] = stats.gamma.rvs((h+1)/2, (2/(h+w*(self.y[j] - \
                np.transpose(self.x[j,:])@beta)**2)))

            Lambda = np.diag(lambda_diag)


            self.traces['beta_trace'][i,:] = beta
            self.traces['Lambda_trace'][i,:] = np.diagonal(Lambda)
            self.traces['w_trace'][i] = w

            print('{:d} iterations out of {:d} complete.'.format(i,iterations))
        self.update_mean()

    def update_mean(self):
        #Updates the means
        self.beta_mean =  np.zeros(np.shape(self.x)[1])
        for j in range(np.shape(self.x)[1]):
            self.beta_mean[j] = np.mean(self.traces['beta_trace'][:,j])

    def burn(self, burn):
        #Removes first burn entries in the traces, then updates means
        for key in self.traces.keys():
            #Applying burn in
            self.traces[key] = self.traces[key][burn:]
        self.update_mean()

    def plot_traces(self):
        #Making trace plots and getting mean predictors
        for j in range(np.shape(self.x)[1]):
            plt.plot(range(len(self.traces['beta_trace'][:,j])), self.traces['beta_trace'][:,j])
            plt.xlabel('Iteration')
            plt.ylabel('beta')
            plt.savefig('trace'+str(j))
            plt.close()


#Loading the data into a pandas dataframe
df = pd.read_csv('gdpgrowth.csv')
#append an intercept column of ones



#Pulling x and y from the dataset
x_data = df['DEF60']
y_data = df['GR6096']

results = heavy_tailed(x_data,y_data)
results.plot_traces()

plt.scatter(x_data,y_data)
plt.plot(x_data, results.x@results.beta_mean, linestyle='--', color='red')
plt.savefig('heavy_tail_fit')



