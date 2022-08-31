import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import figure, axes, title
from unittest import result
from configparser import ConfigParser
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
import mat73
import logistic_regression as log_r
from create_graph import transition_matrix,graph_name
import sys
sys.path.append('..')


start = time.time()

pd.set_option('display.max_rows', 100000)
pd.set_option('display.max_columns', 100000)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


param_config = ConfigParser()
param_config.read("config/param.conf")

datainfo = param_config["DATAINFO"]
algoinfo = param_config["ALGOCONFIG"]
fwinfo = param_config["FWCONFIG"]

f = int(datainfo["f"])   # number of features
dataset = datainfo["dataset"]
c = int(datainfo["c"])   # number of classes
batch_size = int(algoinfo["batch_size"])
L = int(algoinfo["l"])
T = int(algoinfo["t"])
r = float(algoinfo["r"])
num_nodes = int(algoinfo["num_nodes"])
dataname = dataset.split(".", 1)[0]
dim = c * f # dimension of the output and messages



eta = float(algoinfo["eta"])
eta_exp = float(algoinfo["eta_exp"])
rho = float(algoinfo["rho"])
rho_exp = float(algoinfo["rho_exp"])
reg = float(algoinfo["reg"])

eta_fw = float(fwinfo["eta"])
eta_exp_fw = float(fwinfo["eta_exp"])
L_fw = int(fwinfo["l"])

algo = algoinfo["algo"]


shape = (f,c)

path = "dataset/"+dataset
data = pd.read_csv(path, header = None).to_numpy()
y_data = np.zeros(data.shape[0],dtype=int)
x_data = data[:,1:].T
y_data = data[:,0].astype('int64')


def loss_offline(x,t):
	return log_r.loss(x,x_data[:,:(t+1) * batch_size],y_data[:(t+1) * batch_size])

def loss_online(x,t):
	k = t * batch_size
	return log_r.loss(x,x_data[:,k:k + batch_size],y_data[k:k + batch_size])



def compute_gradient_offline(x,t):
	return log_r.compute_gradient(x,x_data[:,:(t+1)*batch_size],y_data[:(t+1)*batch_size])

def compute_gradient_online(x,t):
	k = t* batch_size
	j = k + batch_size
	return log_r.compute_gradient(x,x_data[:,k:j], y_data[k:j])

def compute_gradient_dist_online(x,t,n,i):
	k = t* batch_size + i* int(batch_size/n)
	j = k +  int(batch_size/n) 
	return log_r.compute_gradient(x,x_data[:,k:j], y_data[k:j])


def next_agent(P,i):
	nz_P = P[i,:].nonzero()[0]
	r = np.random.rand()
	p = P[i,nz_P][0]
	for k in range(len(nz_P)):
		if r < p :
			break	
		p += P[i,nz_P][k]
	return nz_P[k]
	

def update_x(x, v, eta_coef, eta_exp, t):
    eta = min(pow(eta_coef / (t + 1),eta_exp), 1.0)
    return x + eta*(v - x)

def FW(t):
	x = np.zeros(shape)
	for l in range(L_fw):
		gradient = compute_gradient_offline(x,t)
		v = log_r.lmo(gradient,r)		                
		x = update_x(x, v, eta_fw, eta_exp_fw, l)
	return x

def MFW():
	x = lil_matrix(shape)
	v = lil_matrix(shape)
	o = [lil_matrix(shape) for _ in range(L+1)]
	a = lil_matrix(shape)
	g = lil_matrix(shape)
	res = [lil_matrix(shape) for _ in range(T)]
	global reg
	reg = reg / np.sqrt(100)
	for t in range(T):
		x = np.zeros(shape)
		g = compute_gradient_online(x,t)
		o[0] = o[0] + g
		
		for l in range(1,L+1):
			eta_l = min(eta / pow((l + 1),eta_exp), 1.0)
			rho_l = min(rho / pow((l + 1),rho_exp), 1.0)

			noise = -0.5 + np.random.rand(shape[0],shape[1])
			
			v = log_r.lmo(o[l]*reg+noise,r)
			
			x = x + eta_l * (v - x)
			g = (1-rho_l) * g + rho_l * compute_gradient_online(x,t)
			g = compute_gradient_online(x,t)
			o[l] = o[l] + g
		res[t] = x
	return res

def RWMFW(P):
	x = lil_matrix(shape)
	v = lil_matrix(shape)
	o = [lil_matrix(shape) for _ in range(L+1)]
	a = lil_matrix(shape)
	g = lil_matrix(shape)
	res = [lil_matrix(shape) for _ in range(T)]
	global reg
	reg = reg / np.sqrt(100)
	n = P.shape[0]
	i = next_agent((1/n)*np.ones((1,n)),0)
	for t in range(T):
		x = np.zeros(shape)
		node_x[i] = x
		g = compute_gradient_online(x,t)
		o[0] = o[0] + g
		
		for l in range(1,L+1):
			eta_l = min(eta / pow((l + 1),eta_exp), 1.0)
			rho_l = min(rho / pow((l + 1),rho_exp), 1.0)

			noise = -0.5 + np.random.rand(shape[0],shape[1])
			
			v = log_r.lmo(o[l]*reg+noise,r)
			
			x = x + eta_l * (v - x)
			i = next_agent(P,i)

			g = compute_gradient_dist_online(x,t,n,i)
			o[l] = o[l] + g
		res[t] = x
	return res

def result_path():
	path = algo.lower()+"-"+graph_name+"-batch_size"+str(batch_size)+"-N"+str(num_nodes)+"-T"+str(T)+"-L"+str(L)
	return path


def draw_regret(regret, name):
    figure(1, figsize=(10, 6))
    x_axis = [i for i in range(1, T+1)]
    plt.plot(x_axis, regret)
    title = result_path()
    plt.title(title)
    plt.xlabel("Number of Rounds T")
    plt.ylabel("Regret")
    plt.savefig(name)


def compute_offline_optimal(t):
	offline_optimal = FW(t)
	return offline_optimal

def regret(online_output,offline_optimal=None,fixed=False):
	if fixed :
		node_loss = [loss_online(online_output[t],t) -  loss_online(offline_optimal,t)  for t in range(T)]	
	else : 
		node_loss = [loss_online(online_output[t],t) - loss_online(compute_offline_optimal(t),t) for t in range(T)]
	regrets = np.cumsum(node_loss)
	return regrets


def exit_error(msg):
	print("Error : " + msg)
	exit()
if __name__ == "__main__":
	start = time.time()	

	

	offline_optimal = compute_offline_optimal(T)
	if algo.upper() == "RWOFW":
		online_output = RWMFW(transition_matrix)
	elif algo.upper() == "MFW":
		online_output = MFW()
	else :
		algorithms = {"mfw","rwofw"}
		exit_error(f"Algorithm in parameter is wrong. \n 'algo' not in {algorithms} ")	
	regrets = regret(online_output,offline_optimal,fixed=True)
	
	draw_regret(regrets,"regrets/"+result_path()+".png")
	print("The regret is in the file " + "regrets/"+result_path()+".png")
	
	end = time.time()
	print("Time taken : " + str(end - start)+ "s")
	



