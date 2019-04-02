import numpy as np
import matplotlib.pyplot as plt
import pdb
from methods import *

#initializing an x array of a unit interval 
x_vec = np.sort(np.random.uniform(size=500))

#We'll let our smooth function be x^3 and the wiggly function be sin(0.05x)
#store all y data in a dictionary
y = {}
y['smooth_low_noise'] = [i**3 + np.random.normal(scale=0.05) for i in x_vec]
y['wiggly_low_noise'] = [np.sin(50*i) + np.random.normal(scale=0.05) for i in x_vec]
y['smooth_high_noise'] = [i**3 +  np.random.normal(scale=0.2) for i in x_vec]
y['wiggly_high_noise'] = [np.sin(50*i) + np.random.normal(scale=0.2) for i in x_vec]


names = ['smooth_low_noise','smooth_high_noise', 'wiggly_low_noise','wiggly_high_noise']
colors = ['black','black','red','red']
linetypes = ['--','-','--','-']

#Making plots to determine best h
h_vec = np.logspace(-10,-1,100) #Use log spacing at recommendation of James
#Make dictionary of best h_values for later plotting
best_h = {}
for i,name in enumerate(names):
    train_x, train_y, test_x, test_y = train_test_split(x_vec, y[name])
    error = np.zeros(len(h_vec))
    for j,h in enumerate(h_vec):
        y_pred_test = pred(train_x,train_y,test_x,h)
        error[j] = error_calc(y_pred_test,test_y)
    #Saving the h value that generates smallest error
    best_h[name] = h_vec[np.nanargmin(error)]
    plt.plot(h_vec,error, color=colors[i], linestyle=linetypes[i])


plt.xlabel('h')
plt.ylabel('error')
plt.ylim([0.0,0.1])
plt.legend(names)
plt.savefig('h_select')
plt.close()

#Then making plots of best fits
for name in names:
    plt.scatter(x_vec,y[name], marker='x')
    plt.plot(x_vec, pred(x_vec, y[name], x_vec, best_h[name]),color='red',linewidth=2)
    plt.savefig(name)
    plt.close()





