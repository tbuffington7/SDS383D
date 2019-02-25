import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pdb



'''
Recap of definitions derived in written notes:
m_star = np.linalg.inv(x.transpose()@Lambda@x + K) @ (x.transpose() @ Lambda @ y + K @ m)
K_star = np.linalg.inv(x.transpose()@Lambda@x + K)
d_star = d + n
eta_star = eta + y.transpose() @ Lambda @ y + m.transpose() @ K @ m - m_star.transpose() @ K_star @ m 



Conditional distributions:
p(beta|y,w, Lambda) ~ MVN(m_star, (w*K_star)^-1)
p(w|y,Lambda) ~ Gamma(d_star/2, eta_star/2)
p(l[i], y, beta, w ) ~ Gamma((h+1)/2, w/2.0*(y - x@beta)**2 + h/2.0)

'''


def gibbs_sampler(x,y,iterations=1000,burn=100,thin=2, h=0.01, Kvals = 0.01, d = 0.01, eta=0.01, w=1.0):

	n = len(y) 
	p = len(x.columns)

	#initialize parameters
	K = np.diag(np.ones(p))*Kvals #A vague prior for the K precision
	m = np.zeros(p)
	Lambda = np.diag(np.ones(n)) * 0.1
	beta = np.zeros(p)

	#Storing the results in a dictionary
	results = {

			'beta_trace': np.zeros([iterations, p]),
			'Lambda_trace': np.zeros([iterations, n]),
			'w_trace': np.zeros(iterations),

	}



	for i in range(iterations):

		#Update beta
		m_star = np.linalg.inv(x.transpose()@Lambda@x + K) @ (x.transpose() @ Lambda @ y + K @ m)
		K_star = np.linalg.inv(x.transpose()@Lambda@x + K)
		beta = np.random.multivariate_normal(m_star, np.linalg.inv(w*K_star))

		#Update lambda
		Lambda = np.diag(np.random.gamma((h+1.0)/2.0, (w/2.0*(y - x@beta)**2 + h/2.0)**-1 ))

		#Update w
		d_star = d + n
		eta_star = eta + y.transpose() @ Lambda @ y + m.transpose() @ K @ m - m_star.transpose() @ K_star @ m 
		w = np.random.gamma(d_star/2.0, 2.0/eta_star)

		results['beta_trace'][i,:] = beta
		results['Lambda_trace'][i,:] = np.diagonal(Lambda)
		results['w_trace'][i] = w


		print('{:d} iterations out of {:d} complete.'.format(i,iterations))

	#Postprocessing traces
	for key in results.keys():
		#Applying burn in
		results[key] = results[key][burn:]
		#Applying thinning
		results[key] = results[key][::thin]

	#Making trace plots and getting mean predictors
	results['beta_mean'] =  np.zeros(p)
	for j in range(p):
		plt.plot(range(len(results['beta_trace'][:,j])), results['beta_trace'][:,j])
		plt.xlabel('Iteration')
		plt.ylabel('beta')
		plt.savefig('trace'+str(j))
		plt.close()
		results['beta_mean'][j] = np.mean(results['beta_trace'][j])


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



