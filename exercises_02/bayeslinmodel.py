import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pdb


def bayes_linear_model(x_data,y_data,K=None, Lambda=None, m_prior=None, add_intercept = True):
    """

    Returns fit parameters for a Bayseian linear model

    Parameters
    ----------
    x: array_like with shape (nxp)
        The x data
    y: array_like with shape(nx1)
        The y data
    K: array_like with shape(pxp)
        Prior precison matrix
    Lambda: array_like with shape (nxn)

    Returns
    -------
    beta_hat: array_like with shape (nx1)
        An array of length two whose first entry is the fitted intercept and whose
        second entry is the fitted slope from a Bayesian linear model
    x: array_like with shape (nxp)
        The X matrix that is multiplied with beta_hat to give predicted y values
        Can be different than input argument x_data when it has a column of ones appended to it
    """

    #This is flipped, but it's a quirk of numpy
    n = np.shape([x_data])[1]
    p = np.shape([x_data])[0]

    #If add_intercept is True, add a column of ones to the x matrix
    if add_intercept == True:
        p += 1
        x = np.ones([n,p])
        x[:,1:] = np.transpose([x_data])
    else:
        x = np.array(x_data)

    #If K is not defined, give it very small values (close to OLS estimate)
    if K is None:
        K = np.diag(np.ones(p)*10**-12)

    #If Lambda is not defined, just make it identity matrix
    if Lambda is None:
        Lambda = np.identity(n)
    #if prior is not specified, make it zero for all parameters
    if m_prior is None:
        m_prior = np.zeros(p)

    #Evaluting posterior
    #First evaluate posterior precision matrix (defined as capital omega in the problem)
    post_prec_mat = np.linalg.inv(K+x.transpose()@Lambda@x)

    #Then evalute beta according to the derived equation
    beta_hat = post_prec_mat@(x.transpose()@Lambda@y + K@m_prior)

    return beta_hat, x

#Loading the data into a pandas dataframe
df = pd.read_csv('gdpgrowth.csv')

#Pulling x and y from the dataset
x_data = df['DEF60']
y = df['GR6096']


K = np.diag([.1,.1]) #A vague prior for the K precision

#Note the captial X has a column of ones for the intercept
beta_hat, X = bayes_linear_model(x_data,y)

#Make scatterplot of raw data
plt.scatter(x_data,y)
#Let's define X as the vector that contains both the intercept and the predictor
plt.plot(x_data, X@beta_hat, linestyle='--', color='red')


#Also plot least squares fit
plt.plot(x_data, np.poly1d(np.polyfit(x_data, y, 1))(x_data),linestyle='--', color='green' )
plt.xlabel('Defense spending (fraction of GDP)')
plt.ylabel('GDP growth rate')
plt.legend(['Bayesian linear model', 'least squares fit'])


plt.savefig('linear_bayes')

