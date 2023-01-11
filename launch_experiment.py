#!/usr/bin/python3
# Import necessary libraries
import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import figure, axes, title
from unittest import result
from configparser import ConfigParser
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
import mat73
import utils.logistic_regression as log_r
from utils.create_graph import transition_matrix, graph_name
import utils.create_graph as cg
import sys


# Start timer
start = time.time()

# Set options for Pandas DataFrame display
pd.set_option('display.max_rows', 100000)
pd.set_option('display.max_columns', 100000)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Read configuration files
param_config = ConfigParser()
param_config.read("config/param.conf")

# Get data and algorithm configurations
datainfo = param_config["DATAINFO"]
algoinfo = param_config["ALGOCONFIG"]
fwinfo = param_config["FWCONFIG"]
expinfo = param_config["EXPERIMENTCONFIG"]

# Get number of features and classes
f = int(datainfo["f"])
dataset = datainfo["dataset"]
c = int(datainfo["c"])

# Get algorithm parameters
batch_size = int(algoinfo["batch_size"])
L = int(algoinfo["l"])
T = int(algoinfo["t"])
r = float(algoinfo["r"])
num_nodes = int(algoinfo["num_nodes"])
dataname = dataset.split(".", 1)[0]
dim = c * f  # dimension of the output and messages
num_trials = int(expinfo["num_trials"])

# Get learning rates
eta = float(algoinfo["eta"])
eta_exp = float(algoinfo["eta_exp"])
rho = float(algoinfo["rho"])
rho_exp = float(algoinfo["rho_exp"])

# Regularization parameter
reg = float(algoinfo["reg"]) / np.sqrt(100)

# Get Frank-Wolfe learning rates
eta_fw = float(fwinfo["eta"])
eta_exp_fw = float(fwinfo["eta_exp"])
L_fw = int(fwinfo["l"])

# Get chosen algorithm
algo = algoinfo["algo"]

# Get shape of data
shape = (f, c)

# Read data from file
path = f"dataset/{dataset}"
data = pd.read_csv(path, header=None).to_numpy()
y_data = np.zeros(data.shape[0], dtype=int)
x_data = data[:, 1:].T
y_data = data[:, 0].astype('int64')


def loss_offline(x, t):
    """Compute the loss function on all data up to time t.

    Args:
        x: The current point in the optimization.
        t: The time up to which to compute the loss function.

    Returns:
        The loss function value at x.
    """
    return log_r.loss(x, x_data[:, :(t+1) * batch_size], y_data[:(t+1) * batch_size])

def loss_online(x, t):
    """Compute the loss function on a batch of data at time t.

    Args:
        x: The current point in the optimization.
        t: The time at which to compute the loss function.

    Returns:
        The loss function value at x.
    """
    k = t * batch_size
    return log_r.loss(x, x_data[:, k:k + batch_size], y_data[k:k + batch_size])

def compute_gradient_offline(x, t):
    """Compute the gradient of the loss function on all data up to time t.

    Args:
        x: The current point in the optimization.
        t: The time up to which to compute the gradient.

    Returns:
        The gradient of the loss function at x.
    """
    return log_r.compute_gradient(x, x_data[:, :(t+1)*batch_size], y_data[:(t+1)*batch_size])

def compute_gradient_online(x, t):
    """Compute the gradient of the loss function on a batch of data at time t.

    Args:
        x: The current point in the optimization.
        t: The time at which to compute the gradient.

    Returns:
        The gradient of the loss function at x.
    """
    k = t * batch_size
    j = k + batch_size
    return log_r.compute_gradient(x, x_data[:, k:j], y_data[k:j])

def compute_stoch_gradient_online(x, t, s):
    """Compute a stochastic gradient of the loss function on a batch of data at time t.

    Args:
        x: The current point in the optimization.
        t: The time at which to compute the gradient.
        s: The size of the stochastic gradient.

    Returns:
        The stochastic gradient of the loss function at x.
    """
    k = t * batch_size
    j = k + batch_size
    sub_batch = np.floor(np.random.rand(s)*batch_size).astype(int) + k
    return log_r.compute_gradient(x, x_data[:, sub_batch], y_data[sub_batch])


def compute_gradient_dist_online(x, t, n, i):
    """
    Compute the gradient of the loss function on a partition of a batch of data at time t.
    
    Args:
        x: The current point in the optimization.
        t: The time at which to compute the gradient.
        n: The number of partitions to divide the batch of data at time t into.
        i: The index of the partition to compute the gradient for, where i is an integer from 0 to n-1.
        
    Returns:
        The gradient of the loss function at x.
    """
    # Compute the start and end indices of the partition
    k = t* batch_size + i* int(batch_size/n)
    j = k +  int(batch_size/n) 
    
    # Return the gradient of the loss function on the partition of the data
    return log_r.compute_gradient(x, x_data[:, k:j], y_data[k:j])



def next_agent(P, i):
    """
    Select the next agent in a process, given a transition probability matrix P and the current agent i.
    
    Args:
        P: A matrix encoding the probabilities of transitioning from one agent to another.
        i: The index of the current agent.
        
    Returns:
        The index of the next agent.
    """
    # Extract the indices of the non-zero elements of the i-th row of P
    nz_P = P[i,:].nonzero()[0]
    
    # Generate a random number uniformly from the interval [0,1)
    r = np.random.rand()
    
    # Set the cumulative probability to the first non-zero element of the i-th row of P
    p = P[i, nz_P][0]
    
    # Iterate over the non-zero elements of the i-th row of P
    for k in range(len(nz_P)):
        # If the random number is less than the cumulative probability, return the index of the corresponding agent
        if r < p:
            return nz_P[k]
        # Otherwise, add the current probability to the cumulative probability and continue to the next element
        p += P[i, nz_P][k]
    
    # If the loop completes, return the index of the last agent in the list of non-zero elements
    return nz_P[k]


def update_x(x, v, eta_coef, eta_exp, t):
    """
    Update the value of x using an exponential decay learning rate.
    
    Args:
        x: The current value of x.
        v: The value to update x towards.
        eta_coef: A coefficient that determines the initial learning rate.
        eta_exp: An exponent that determines the decay rate of the learning rate.
        t: The current time step.
        
    Returns:
        The updated value of x.
    """
    # Compute the learning rate using an exponential decay function
    eta = min(pow(eta_coef / (t + 1), eta_exp), 1.0)
    
    # Update x using the learning rate
    return x + eta * (v - x)

def FW(t):
    """
    Perform the Frank-Wolfe algorithm to optimize the loss function on all data up to time t.
    
    Args:
        t: The time up to which to optimize the loss function.
        
    Returns:
        The optimal value of x.
    """
    # Initialize x to a vector of zeros
    x = np.zeros(shape)
    
    # Perform iterations of the Frank-Wolfe algorithm
    for l in range(L_fw):  # L_fw is not defined in this version of the function
        # Compute the gradient of the loss function at x
        gradient = compute_gradient_offline(x, t)
        
        # Compute the linear minimization oracle at the gradient
        v = log_r.lmo(gradient, r)
        
        # Update x using the learning rate
        x = update_x(x, v, eta_fw, eta_exp_fw, l)
    
    # Return the optimal value of x
    return x


def MFW():
    """
    Perform the Meta Frank-Wolfe algorithm in an online setting.
    
    Returns:
        A list of the optimized values of x at each time step.
    """
    # Initialize x, v, and o to sparse matrices of zeros with the specified shape
    x = lil_matrix(shape)
    v = lil_matrix(shape)
    o = [lil_matrix(shape) for _ in range(L+1)]
    
    # Initialize a and g to sparse matrices of zeros with the specified shape
    a = lil_matrix(shape)
    g = lil_matrix(shape)
    
    # Initialize a list to store the optimized values of x at each time step
    res = [lil_matrix(shape) for _ in range(T)]
    
    # Iterate over time steps
    for t in range(T):
        # Initialize x to a vector of zeros
        x = np.zeros(shape)
        
        # Iterate over the levels of the Meta Frank-Wolfe algorithm
        for l in range(0,L+1):
            # Compute the learning rate using an exponential decay function
            eta_l = min(eta / pow((l + 1),eta_exp), 1.0)
            
            # Compute the variance reduction coefficient using an exponential decay function
            rho_l = min(rho / pow((l + 1),rho_exp), 1.0)
            
            # Generate noise uniformly from the interval [-0.5, 0.5)
            noise = -0.5 + np.random.rand(shape[0],shape[1])
            
            # Compute the linear minimization oracle at o[l]*reg + noise
            v = log_r.lmo(o[l]*reg+noise,r)
            
            # Perform variance reduction
            g = (1-rho_l) * g + rho_l * compute_stoch_gradient_online(x,t,4)
            o[l] = o[l] + g
            
            # Update x using the learning rate
            x = x + eta_l * (v - x)
        
        # Append the optimized value of x at this time step to the list of results
        res[t] = x
    
    # Return the list of optimized values of x
    return res


def MFW_dense():
    """
    Perform the Meta Frank-Wolfe algorithm in an online setting using dense arrays.
    
    Returns:
        A list of the optimized values of x at each time step.
    """
    # Initialize x, v, and o to arrays of zeros with the specified shape
    x = np.zeros(shape)
    v = np.zeros(shape)
    o = [np.zeros(shape) for _ in range(L+1)]
    
    # Initialize a and g to arrays of zeros with the specified shape
    a = np.zeros(shape)
    g = np.zeros(shape)
    
    # Initialize a list to store the optimized values of x at each time step
    res = [np.zeros(shape) for _ in range(T)]
    
    # Iterate over time steps
    for t in range(T):
        # Initialize x to a vector of zeros
        x = np.zeros(shape)
        
        # Iterate over the levels of the Meta Frank-Wolfe algorithm
        for l in range(0,L+1):
            # Compute the learning rate using an exponential decay function
            eta_l = min(eta / pow((l + 1),eta_exp), 1.0)
            
            # Compute the variance reduction coefficient using an exponential decay function
            rho_l = min(rho / pow((l + 1),rho_exp), 1.0)
            
            # Generate noise uniformly from the interval [-0.5, 0.5)
            noise = -0.5 + np.random.rand(shape[0],shape[1])
            
            # Compute the linear minimization oracle at o[l]*reg + noise
            v = log_r.lmo(o[l]*reg+noise,r)
            
            # Perform variance reduction
            g = (1-rho_l) * g + rho_l * compute_stoch_gradient_online(x,t,4)
            o[l] = o[l] + g
            
            # Update x using the learning rate
            x = x + eta_l * (v - x)
        
        # Append the optimized value of x at this time step to the list of results
        res[t] = x
    
    # Return the list of optimized values of x
    return res


def RWMFW(P):
    """
    Perform the Random Walk Meta Frank-Wolfe algorithm in an online setting.
    
    Args:
        P: A matrix of transition probabilities between nodes in a graph.
    
    Returns:
        A list of the optimized values of x at each time step.
    """
    # Initialize x, v, and o to sparse matrices with the specified shape
    x = lil_matrix(shape)
    v = lil_matrix(shape)
    o = [lil_matrix(shape) for _ in range(L+1)]
    
    # Initialize g to a sparse matrix with the specified shape
    g = lil_matrix(shape)
    
    # Initialize a list to store the optimized values of x at each time step
    res = [lil_matrix(shape) for _ in range(T)]
    
    # Get the number of nodes in the graph
    n = P.shape[0]
    
    # Iterate over time steps
    for t in range(T):
        # Select the starting node for the random walk using the next_agent function
        i = next_agent((1/n)*np.ones((1,n)),0)
        
        # Initialize x to a vector of zeros
        x = np.zeros(shape)
        
        # Iterate over the levels of the Random Walk Meta Frank-Wolfe algorithm
        for l in range(0,L+1):
            # Compute the learning rate using an exponential decay function
            eta_l = min(eta / pow((l + 1),eta_exp), 1.0)
            
            # Compute the variance reduction coefficient using an exponential decay function
            rho_l = min(rho / pow((l + 1),rho_exp), 1.0)

             # Generate noise uniformly from the interval [-0.5, 0.5)
            noise = -0.5 + np.random.rand(shape[0],shape[1])
            
            # Compute the linear minimization oracle at o[l]*reg + noise
            v = log_r.lmo(o[l]*reg+noise,r)
            
            # Use the subgradient corresponding to the current node in the random walk to update o[l]
            g = compute_gradient_dist_online(x,t,n,i)
            
            # Uncomment the following line to perform variance reduction
            #g = (1-rho_l) * g + rho_l * compute_gradient_dist_online(x,t,n,i)
            
            o[l] = o[l] + g
            
            # Update x using the learning rate
            x = x + eta_l * (v - x)
            
            # Update the current node in the random walk using the next_agent function
            i = next_agent(P,i) 
        
        # Append the optimized value of x at this time step to the list of results
        res[t] = x
    
    # Return the list of optimized values of x
    return res


def result_path():
    """
    Generate a pathname for the results of the optimization algorithm.
    
    Returns:
        A string representing the pathname for the results.
    """
    # Construct the pathname using the algorithm name, graph name, data name, batch size, number of nodes, number of time steps, and number of levels
    path = f"{algo.lower()}-{graph_name}-{dataname}-batch_size{str(batch_size)}-N{str(num_nodes)}-T{str(T)}-L{str(L)}"
    
    # Return the pathname
    return path


def draw_regret(regret, name):
    """
    Plot the regret curve for the given regret values and save it to a file with the given name.
    
    Args:
        regret: A list of regret values at each time step.
        name: A string representing the name of the file to save the plot to.
    """
    # Set the size of the figure and create a new figure
    figure(1, figsize=(10, 6))
    
    # Generate the list of x-axis values
    x_axis = [i for i in range(1, T+1)]
    
    # Plot the regret curve
    plt.plot(x_axis, regret)
    
    # Set the title of the plot
    title = name
    plt.title(title)
    
    # Label the x-axis
    plt.xlabel("Number of Rounds T")
    
    # Label the y-axis
    plt.ylabel("Regret")
    
    # Save the plot to a file with the given name
    plt.savefig(name)
    
    # Clear the plot
    plt.clf()

def save_multiple_regrets(regrets, regret_file_name):
    """
    Save the given list of regrets to a CSV file with the given name.
    
    Args:
        regrets: A list of lists, where each inner list represents the regret at each time step for a single run of the algorithm.
        regret_file_name: A string representing the name of the file to save the regrets to.
    """
    # Create a Pandas DataFrame from the list of regrets
    df_regrets = pd.DataFrame(regrets)
    
    # Save the DataFrame to a CSV file with the given name
    df_regrets.to_csv(regret_file_name+".csv", index = False, header = False)
    
    # Print a message indicating where the regrets are saved
    print(f'Regrets are in the file : {regret_file_name}.csv')


def save_regret_file(regret, regret_file_name):
    """
    Save the given regret curve to a CSV file with the given name.
    
    Args:
        regret: A list representing the regret at each time step.
        regret_file_name: A string representing the name of the file to save the regrets to.
    """
    # Create a Pandas DataFrame from the list of regrets
    df_regret = pd.DataFrame(regret).T
    
    # Save the DataFrame to a CSV file with the given name
    df_regret.to_csv(regret_file_name+".csv", index = False, header = False)
    
    # Print a message indicating where the regret curve is saved
    print(f"The regret is in the file: {regret_file_name}.csv")


def compute_offline_optimal(t=T,available=False, optimal_file = None):
    """
    Compute the offline optimal solution at time t. If the `available` flag is set to True and an `optimal_file` is provided, 
    the offline optimal solution will be read from the file. Otherwise, the offline optimal solution will be computed using the 
    FW algorithm.
    
    Args:
        t: The time at which to compute the offline optimal solution.
        available: A boolean flag indicating whether or not the offline optimal solution is available in a file.
        optimal_file: The name of the file containing the offline optimal solution, if available.
    
    Returns:
        The offline optimal solution at time t.
    """
    if available:
        # Read the offline optimal solution from the given file
        offline_optimal = pd.read_csv(optimal_file, header=None).to_numpy()
    else:
        # Compute the offline optimal solution using the FW algorithm
        offline_optimal = FW(t)
    return offline_optimal

def regret(online_output, offline_optimal=None, fixed=False):
    """
    Compute the regret of the given online output. If the `fixed` flag is set to True and an `offline_optimal` solution is provided, 
    the regret will be computed using this offline solution. Otherwise, the offline optimal solution will be computed at each round 
    using the `compute_offline_optimal` function.
    
    Args:
        online_output: The online output for which to compute the regret.
        offline_optimal: The offline optimal solution, if provided.
        fixed: A boolean flag indicating whether or not to use the provided offline optimal solution.
    
    Returns:
        The regret of the given online output.
    """
    if fixed:
        # Use the provided offline optimal solution to compute the regret
        node_loss = [(loss_online(online_output[t], t) - loss_online(offline_optimal, t)) for t in range(T)]
    else:
        # Compute the offline optimal solution at each round and use it to compute the regret
        node_loss = [loss_online(online_output[t], t) - loss_online(compute_offline_optimal(t), t) for t in range(T)]
    # Compute the cumulative sum of the node losses to obtain the regret
    regrets_res = np.cumsum(node_loss)
    return regrets_res


def exit_error(msg):
    """
    Print an error message and exit the program.
    
    Args:
        msg: The error message to print.
    """
    # Print the error message
    print("Error: " + msg)
    # Exit the program
    exit()




# This function runs the specified algorithm for a given number of trials and computes the regret of the online output.
# It also saves plots of the regret for each trial and the average regret across all trials.
def main():
    start = time.time()
    print(f"Experiment Parameters:")
    print(f"\tNumber of trials: {num_trials}")
    print(f"\tDataset: {dataset}")
    print(f"\tNumber of features: {f}")
    print(f"\tNumber of classes: {c}")
    print(f"\tBatch size: {batch_size}")
    print(f"\tNumber of rounds: {T}")
    print(f"\tLearning rate: {eta}")
    print(f"\tLearning rate decay exponent: {eta_exp}")
    print(f"\tRegularization parameter: {reg}")
    print(f"\tChosen algorithm: {algo.upper()}")
    if (algo.upper() == 'MFW'):
    	print(f"\tCentralized settings.")
    else :
    	print(f"\tDecentralized settings: ")
    	print(f"\tGraph : {cg.graph_name}")
    print("")
    print("")
    print(f"Computing offline optimal solution...")

    # Compute the offline optimal solution
    offline_optimal = compute_offline_optimal()

    # Initialize an empty list to store the regrets for each trial
    regrets = []
    # Run the specified number of trials
    for trial in range(num_trials):
        print(f"Running trial {trial+1}/{num_trials} with {algo.upper()} algorithm...")
        # Initialize the online output for this trial
        online_output = [np.zeros(shape) for _ in range(T)]
        # Run the specified algorithm
        if algo.upper() == "RWMFW":
            online_output = RWMFW(transition_matrix)
        elif algo.upper() == "MFW":
            graph_name = "centralized"
            num_does = 1
            online_output = MFW()
        else:
            algorithms = {"mfw", "rwmfw"}
            exit_error(f"Algorithm in parameter is wrong. \n 'algo' not in {algorithms} ")
        # Compute the regret for this trial and append it to the list of regrets
        regrets.append(regret(online_output, offline_optimal, fixed=True))
        # Save a plot of the regret for this trial
        draw_regret(regrets[trial], f"regrets/{result_path()}_{trial}.png")

    print("")
    print("")
    # Save plots of the regrets for all trials
    save_multiple_regrets(regrets, f"regrets/{result_path()}")
    # Compute the average regret across all trials
    regrets = np.sum(np.array(regrets), axis=0) / num_trials
    # Save a plot of the average regret
    draw_regret(regrets, f"regrets/average_{result_path()}_trials{num_trials}.png")
    # Save the average regret to a file
    save_regret_file(regrets, f"regrets/average_{result_path()}_trials{num_trials}")

    print(f"Regret plots saved to regrets/{result_path()}*.png")
    print(f"Average regret saved to regrets/average_{result_path()}_trials{num_trials}")

    end = time.time()
    print(f"Total time taken: {end - start}s")





if __name__ == "__main__":
	main()
	



