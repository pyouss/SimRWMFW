#!/usr/bin/python3
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
import numpy as np


def compute_gradient(x, x_data, y_data):
    """
    Compute the gradient of the loss function for logistic regression.
    
    Args:
        x: The current model parameters.
        x_data: The data used to compute the gradient.
        y_data: The labels corresponding to the data.
        
    Returns:
        The gradient of the loss function.
    """
    # Get the number of data points
    data_size = x_data.shape[1]
    # Compute the linear combination of the data and model parameters
    z = x.T @ x_data
    # Compute the softmax function
    tmp_exp = np.exp(z)
    tmp_denominator = np.sum(tmp_exp, axis=0)
    tmp_exp = tmp_exp / tmp_denominator
    # Compute the gradient
    tmp_exp[y_data, range(data_size)] = tmp_exp[y_data, range(data_size)] - 1
    return (x_data / data_size) @ tmp_exp.T

def lmo(o, r):
    """
    Compute the result of the linear minimization oracle.
    
    Args:
        o: The objective function.
        r: The radius of the LMO.
        
    Returns:
        The result of the LMO.
    """
    # Initialize the result
    res = np.zeros(o.shape)
    # Get the indices of the maximum values in each column of the objective function
    max_rows = np.argmax(np.abs(o), axis=0)
    # Compute the values of the result using the maximum values and the radius
    values = -r * np.sign(o[max_rows, range(o.shape[1])])
    # Set the values in the result
    res[max_rows, range(o.shape[1])] = values
    return res



def loss(x, x_data, y_data):
    """
    Compute the loss for the given data and weights.
    
    Args:
        x: The weights.
        x_data: The data.
        y_data: The labels.
        
    Returns:
        The loss.
    """
    # Get the number of data points
    data_size = x_data.shape[1]
    # Compute the logits
    z = x.T @ x_data
    # Compute the exponentiated logits
    tmp_exp = np.exp(z)
    # Compute the numerator of the cross-entropy loss
    tmp_numerator = tmp_exp[y_data, range(data_size)]
    # Return the mean of the cross-entropy loss
    return - np.mean(np.log(tmp_numerator / np.sum(tmp_exp, axis=0)))

