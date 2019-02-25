import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pdb

#Loading the data into a pandas dataframe
df = pd.read_csv('gdpgrowth.csv')
#append an intercept column of ones
df['intercept'] = np.ones(len(df))



#Pulling x and y from the dataset
x = df[ [ 'intercept', 'DEF60']].copy()
y = df['GR6096']




#defining the priors
n = len(y) #The number of observations
p = len(x.columns)
Lambda = np.identity(n) #The known matrix defined in the problem

K = np.diag([.1,.1]) #A vague prior for the K precision
m_prior = np.zeros(p)

#Evaluting posterior
#First evaluate precision matrix (defined as capital omega in the problem)
prec_mat = np.linalg.inv(K+x.transpose()@Lambda@x)

#Then evalute beta according to the derived equation
beta_hat = prec_mat@(x.transpose()@Lambda@y + K@m_prior)

plt.scatter(x['DEF60'],y)
plt.plot(x['DEF60'], x@beta_hat, linestyle='--', color='red')


#Also plot least squares fit
plt.plot(x['DEF60'], np.poly1d(np.polyfit(x['DEF60'], y, 1))(x['DEF60']),linestyle='--', color='green' )



plt.savefig('linear_bayes')





if __name__ == "__main__":
    import doctest
    doctest.testmod()

