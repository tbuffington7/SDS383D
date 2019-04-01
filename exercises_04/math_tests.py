import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb
import pymc3 as pm

#loading the data
data = pd.read_csv('mathtest.csv')

#Our best guess for the priors is the pooled statistics
p_mean = np.mean(data['mathscore'])
p_sigma = np.std(data['mathscore'])
p_tau = 5

data = data[data['school'] < 5]

def school_sampler(x):
    with pm.Model() as model:
        sigma = pm.Uniform('sigma',p_sigma-5, p_sigma+5)
        tau = pm.Uniform('tau',0,10)
        theta= pm.Normal('theta', mu=p_mean, sd= tau*sigma)
        y = pm.Normal('y', mu=theta, sd=sigma, observed=x['mathscore'])
        trace = pm.sample(2000, tune = 1000, cores=4)

    d = {
            'sample_size':len(x),
            'theta_mean':pm.summary(trace)['mean']['theta'],
            'sigma_mean':pm.summary(trace)['mean']['sigma'],
            'tau_mean':pm.summary(trace)['mean']['tau'],
            'shrinkage_coeff': (np.mean(x['mathscore'])-pm.summary(trace)['mean']['theta'])/np.mean(x['mathscore'])
    }
    return pd.Series(d, index=d.keys())





summary = data.groupby('school').apply(school_sampler).reset_index(drop=True)


pdb.set_trace()


