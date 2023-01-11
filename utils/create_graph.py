#!/bin/usr/python3

# Import necessary libraries and modules
import os
import sys
import shutil
import numpy as np
sys.path.append('..')
from configparser import ConfigParser

# Read the configuration files for the graph and the algorithm
graph_config = ConfigParser()
graph_config.read("config/graph.conf")
param_config = ConfigParser()
param_config.read("config/param.conf")

# Get the type and parameters of the graph
graph_type = graph_config["GRAPHTYPE"]
graph_param = graph_config[graph_type["type"] + "PARAM"]

# Get the algorithm configuration
algoconfig = param_config["ALGOCONFIG"]

# Initialize the number of nodes to zero
nb_nodes = 0


def complete_graph():
    """
    Create a complete graph with n0 nodes.
    
    Returns:
        A complete graph represented as an adjacency matrix.
    """
    global nb_nodes  # number of nodes in the graph
    # Get the number of nodes from the graph parameters
    n0 = int(graph_param["n0"])
    # Set the global variable for the number of nodes
    nb_nodes = n0
    # Create an adjacency matrix for the complete graph with n0 nodes
    return np.ones((n0, n0))

def grid_graph():
    """
    Create a grid graph with n0 rows and n1 columns.
    
    Returns:
        A grid graph represented as an adjacency matrix.
    """
    global nb_nodes  # number of nodes in the graph
    # Get the number of rows and columns from the graph parameters
    n0 = int(graph_param["n0"])
    n1 = int(graph_param["n1"])
    # Calculate the total number of nodes in the graph
    nb_nodes = n0 * n1
    # Initialize an adjacency matrix for the graph
    res = np.zeros((nb_nodes, nb_nodes))
    # Iterate over the nodes and set the edges for the grid graph
    for i in range(nb_nodes-1):
        if i % n1 == n1-1:  # last column
            res[i, i+n1] = 1  # edge to the node below
        elif i >= (n0-1)*n1:  # last row
            res[i, i+1] = 1  # edge to the node on the right
        else:  # other nodes
            res[i, i+1] = 1  # edge to the node on the right
            res[i, i+n1] = 1  # edge to the node below
    # Set the edges between nodes symmetrically
    for i in range(nb_nodes):
        for j in range(i):
            res[i, j] = res[j, i]
    # Add the identity matrix to the adjacency matrix to include self-loops
    return res + np.diag(np.ones(nb_nodes))

def line_graph():
    """
    Create a line graph with n0 nodes.
    
    Returns:
        A line graph represented as an adjacency matrix.
    """
    global nb_nodes  # number of nodes in the graph
    # Get the number of nodes from the graph parameters
    n0 = int(graph_param["n0"])
    # Set the global variable for the number of nodes
    nb_nodes = n0
    # Initialize the adjacency matrix for the line graph
    res = np.zeros((n0, n0))
    # Set the edges between nodes
    i = list(range(1, n0)) + list(range(n0-1))
    j = list(range(n0-1)) + list(range(1, n0))
    res[i, j] = 1
    # Add the identity matrix to the adjacency matrix to include self-loops
    return res + np.diag(np.ones(n0))

def cycle_graph():
    """
    Create a cycle graph with n0 nodes.
    
    Returns:
        A cycle graph represented as an adjacency matrix.
    """
    global nb_nodes  # number of nodes in the graph
    # Get the number of nodes from the graph parameters
    n0 = int(graph_param["n0"])
    # Set the global variable for the number of nodes
    nb_nodes = n0
    # Initialize the adjacency matrix for the cycle graph
    res = np.zeros((nb_nodes, nb_nodes))
    # Set the edges between nodes for the cycle graph
    res[0, 1] = 1
    res[0, n0-1] = 1
    res[n0-1, n0-2] = 1
    res[n0-1, 0] = 1
    for i in range(1, n0-1):
        res[i, i-1] = 1
        res[i, i+1] = 1
    # Add the identity matrix to the adjacency matrix to include self-loops
    return res + np.diag(np.ones(n0))


def compute_metropolis_transition_matrix(G):
    """
    Compute the transition matrix for the Metropolis-Hastings algorithm using the given graph.
    
    Args:
        G: The graph represented as an adjacency matrix.
    
    Returns:
        The transition matrix for the Metropolis-Hastings algorithm.
    """
    # Initialize the transition matrix
    P = np.zeros(G.shape)
    # Set the transition probabilities between nodes
    for i in range(G.shape[0]):
        for j in range(G.shape[0]):
            if i != j and G[i, j] == 1:  # i and j are neighbors
                P[i, j] = np.min([1 / np.sum(G[i,:]), 1 / np.sum(G[j,:])])
    # Set the self-loop probabilities
    for i in range(G.shape[0]):
        P[i, i] = 1 - np.sum(P[i,:])
    return P


# Create a dictionary of functions for creating different types of graphs
create_graph_functions = {"complete": complete_graph, "grid": grid_graph, "line": line_graph, "cycle": cycle_graph}

# Get the name of the desired graph from the graph parameters
graph_name = graph_param["name"]

# Use the appropriate function to create the graph
G = create_graph_functions[graph_name]()

# Compute the transition matrix for the Metropolis-Hastings algorithm using the graph
transition_matrix = compute_metropolis_transition_matrix(G)



def one_param_graph():
    """
    Return the name of a graph with one parameter.
    
    Returns:
        A tuple containing a boolean indicating success and the name of the graph.
    """
    # Get the number of nodes from the graph parameters
    n0 = graph_param["n0"]
    # Return the name of the graph
    return True, graph_param["name"] + str(n0)

def two_param_graph():
    """
    Return the name of a graph with two parameters.
    
    Returns:
        A tuple containing a boolean indicating success and the name of the graph.
    """
    # Get the number of nodes from the graph parameters
    n0 = graph_param["n0"]
    n1 = graph_param["n1"]
    # Ensure that n0 <= n1
    if n0 > n1:
        temp = n0
        n0 = n1
        n1 = temp
    # Return the name of the graph
    return True, graph_param["name"] + str(n0) + "_" + str(n1)

def error_graph():
    """
    Return an error message for an unsupported graph type.
    
    Returns:
        A tuple containing a boolean indicating failure and an error message.
    """
    return False, "Error: Unsupported graph type"

# Create a dictionary of functions for getting the names of different types of graphs
graph_name_functions = {"complete": one_param_graph, "grid": two_param_graph, "line": one_param_graph, "cycle": one_param_graph}

# Get the name of the graph using the appropriate function
graph_name_result = graph_name_functions[graph_param["name"]]()

# Check if the function call was successful
if graph_name_result[0]:
    # Set the name of the graph
    graph_name = graph_name_result[1]
else:
    # Set an error message
    graph_name = "Graph name error."


if __name__ == "__main__":
	print(graph_name)
	print(G)
	print(P)