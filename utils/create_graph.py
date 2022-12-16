import os
import sys
import shutil
import numpy as np
sys.path.append('..')
from configparser import ConfigParser

#Read config files
graph_config = ConfigParser()
graph_config.read("config/graph.conf")
param_config = ConfigParser()
param_config.read("config/param.conf")

graph_type = graph_config["GRAPHTYPE"]

graph_param = graph_config[graph_type["type"]+"PARAM"]

algoconfig = param_config["ALGOCONFIG"]

nb_nodes = 0


def complete_graph():
	global nb_nodes
	n0 = int(graph_param["n0"])
	nb_nodes = n0
	return np.ones((n0,n0))  

def grid_graph():
	global nb_nodes
	n0 = int(graph_param["n0"])
	n1 = int(graph_param["n1"])
	nb_nodes = n0*n1
	res = np.zeros((nb_nodes, nb_nodes))
	for i in range(nb_nodes-1):
		if i % n1 == n1-1:
		    res[i,i+n1] = 1
		elif i >= (n0-1)*n1:
		    res[i,i+1] = 1
		else:
		    res[i,i+1] = 1
		    res[i,i+n1] = 1
	for i in range(nb_nodes):
		for j in range(i):
		    res[i,j] = res[j,i]
	return res + np.diag(np.ones(nb_nodes))

def line_graph():
	global nb_nodes
	n0 = int(graph_param["n0"])
	nb_nodes = n0
	i = list(range(1,n0))+list(range(n0-1)) 
	j = list(range(n0-1))+list(range(1,n0))
	res = np.zeros((n0,n0))
	res[i,j] = 1
	return res + np.diag(np.ones(n0))

def cycle_graph():
	global nb_nodes
	n0 = int(graph_param["n0"])
	nb_nodes = n0
	res = np.zeros((nb_nodes, nb_nodes))
	res[0, 1] = 1
	res[0, n0-1] = 1
	res[n0-1, n0-2] = 1
	res[n0-1, 0] = 1
	for i in range(1, n0-1):
		res[i, i-1] = 1
		res[i, i+1] = 1
	return res + np.diag(np.ones(n0))


def compute_metropolis_transition_matrix(G):
	P = np.zeros(G.shape)
	for i in range(G.shape[0]):
		for j in range(G.shape[0]):
			if i != j and G[i,j] == 1 :
				P[i, j] = np.min([ 1/np.sum(G[i,:]), 1/np.sum(G[j,:]) ])
	for i in range(G.shape[0]):
		P[i, i] = 1 - np.sum(P[i,:])
	return P


create_graph_functions = {"complete" : complete_graph, "grid": grid_graph, "line": line_graph, "cycle": cycle_graph}


G = create_graph_functions[graph_param["name"]]()

transition_matrix = compute_metropolis_transition_matrix(G)



def one_param_graph() :
		n0 = graph_param["n0"]
		return True,graph_param["name"]+str(n0)
	
def two_param_graph():
	n0 = graph_param["n0"]
	n1 = graph_param["n1"]
	if n0 > n1:
		temp = n0
		n0 = n1
		n1 = temp
	return True,graph_param["name"]+str(n0)+"_"+str(n1)
	
def error_graph():
	return False,"Error graph type"

graph_name_functions = {"complete" : one_param_graph, "grid": two_param_graph, "line": one_param_graph, "cycle": one_param_graph}
graph_name = graph_name_functions[graph_param["name"]]()
if graph_name[0]:
	graph_name = graph_name[1]
else :
	graph_name = "Graph name error."

if __name__ == "__main__":
	print(graph_name)
	print(G)
	print(P)