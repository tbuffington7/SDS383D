import numpy as np
import pandas as pd

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

def error_calc(x1, x2):
    """

    Determines the mean squared error between two vectors.
    x1 and x2 must be the same length

    Parameters
    ----------
    x1 : array_like
        The first array
    x2: array_like
        The second array

    Returns
    -------
    mean_error: float
        The average squared error between x1 and x2

    Raises:
    ------
    ValueError
        if the two arrays are not the same length
    """
    if len(x1) != len(x2):
        raise ValueError('arrays must be the same length')

    total_error = 0.0
    for i in range(len(x1)):
        total_error = total_error + (x1[i] - x2[i])**2

    mean_error = total_error / float(len(x1))
    return mean_error
