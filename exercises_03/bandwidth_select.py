import numpy as np
import matplotlib.pyplot as plt
import pdb
from kernel_smooth import weight_calc
from kernel_smooth import gaussian_kernel
from kernel_smooth import pred
from cross_validation import error_calc

#initializing an x array of a unit interval 
x_vec = np.sort(np.random.uniform(size=500))

#We'll let our smooth function be x^3 and the wiggly function be sin(0.05x)
#store all y data in a dictionary
y = {}
y['smooth_low_noise'] = [i**3 + np.random.normal(scale=0.05) for i in x_vec]
y['wiggly_low_noise'] = [np.sin(50*i) + np.random.normal(scale=0.05) for i in x_vec]
y['smooth_high_noise'] = [i**3 +  np.random.normal(scale=0.2) for i in x_vec]
y['wiggly_high_noise'] = [np.sin(50*i) + np.random.normal(scale=0.2) for i in x_vec]

def train_test_split(x,y, test_size=0.25):

    """

    Splits a dataset into a training and test set

    Parameters
    ----------
    x: array_like
        The x values for a dataset
    y: array_like
        The corresponding y values from the dataset
    test_size: float
        The fraction of the dataset to allocate to the test set

    Returns
    -------
    train_x: array_like
        The x values from the training set
    train_y: array_like
        The corresponding y values from the training set
    test_x: array_like
        The x values from the test set
    test_y: array_like
        The corresponding y values from the test set

    Raises
    ------
    ValueError
        when x and y are not the same length

    """
    if len(x) != len(y):
        raise ValueError('x and y should be the same length')

    #First determine the absolute number of points for the test set
    test_size = int(round(test_size*len(x)))

    #Randomly select indices for the test set
    test_indices = np.random.choice(np.array(range(len(x))), test_size, replace=False)

    #Then make the test set
    test_x = np.array(x)[test_indices]
    test_y = np.array(y)[test_indices]

    #Then make the training set 
    train_x = np.delete(np.array(x),test_indices)
    train_y = np.delete(np.array(y),test_indices)

    return train_x, train_y, test_x, test_y

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





