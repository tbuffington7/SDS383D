import numpy as np
import matplotlib.pyplot as plt
import pdb


def gaussian_kernel(x):
    """

    Takes in an array of x values and returns corresponding Gaussian kernel values

    Parameters
    ----------
     x: array_like
        An array of x values

    Returns
    -------
    gaussian_kernel(x): array_like
        An array of same length as x with corresponding kernel values

    """
    return np.array([1.0/(2.0*np.pi)**.5 * np.exp(-i**2/2) for i in x])

def weight_calc(x, x_star, h, kernel_func=gaussian_kernel):
    """

    Evaluates the normalized weights for based on a kernel function

    Parameters
    ----------
    x: array_like
        the array of x values from the data
    x_star: float
        The x value corresponding to the desired prediction location
    h: float
        The bandwidth
    kernel_func: function
        The desired kernel function. Default is gaussian


    Returns
    -------
    weight_calc: array_like
        The weight of each x data point for the desired prediction location
    """
    raw_weights = 1.0/h*kernel_func((x-x_star)/h)
    norm_weights =  np.array(raw_weights/np.sum(raw_weights))
    return norm_weights

def pred(x,y, x_star, h, kernel_func=gaussian_kernel):

    """

    Makes predictions for an array x_star for a given dataset of x and y

    Parameters
    ----------
    x: array_like
        the array of x values from the data
    y: array_like
        the array of corresponding y values
    x_star: float
        The x value corresponding to the desired prediction location
    h: float
        The bandwidth
    kernel_func: function
        The desired kernel function. Default is gaussian

    Returns
    -------
    y_star: array_like
        The predicted values corresponding to x_star
    """

    y_star = np.zeros(len(x_star))
    for i, x_val in enumerate(x_star):
        weights = weight_calc(x, x_val,h, kernel_func)
        y_star[i] = weights@y #matrix multiplciation

    return y_star

#Create a vector of x values
x = np.linspace(0,10,100)
#Use sin(x) as the function and add Gaussian noise
y = [np.sin(i) + np.random.normal(0,0.1) for i in x]

#Define x_star: the x locations we're interested in predicting y
x_star_vec = np.linspace(0,10,100)

#initialize y_star: the corresponding predictions
y_star_vec = np.zeros(len(x_star_vec))

#Initializing array of h values for comparison
h_vec = [0.05, 0.1, 1.0]

#Making the plot of the noisy data outside of for loops
plt.scatter(x,y, marker = 'x')
for h in h_vec:
    y_star_vec = pred(x,y,x_star_vec,h)
    plt.plot(x_star_vec, y_star_vec)

plt.legend(['h=0.05', 'h=0.1', 'h=1.0', 'raw_data'])
plt.savefig('gaussian_kernel')
plt.close()


