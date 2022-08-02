import os
import json
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import figure, axes, title
from unittest import result
from scipy.special import softmax
from configparser import ConfigParser
from scipy.sparse import csr_matrix
import mat73

start = time.time()

pd.set_option('display.max_rows', 100000)
pd.set_option('display.max_columns', 100000)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)



#Read config.ini file
param_config = ConfigParser()
param_config.read("config/param.conf")

datainfo = param_config["DATAINFO"]
algoinfo = param_config["ALGOCONFIG"]
fwinfo = param_config["FWCONFIG"]

f = int(datainfo["f"])   # number of features
dataset = datainfo["dataset"]
c = int(datainfo["c"])   # number of classes
decentralized_batch_size = int(algoinfo["batch_size"])
L = int(algoinfo["l"])
T = int(algoinfo["t"])
r = float(algoinfo["r"])
num_nodes = int(algoinfo["num_nodes"])
dataname = dataset.split(".", 1)[0]
dim = c * f # dimension of the output and messages
batch_size = num_nodes * decentralized_batch_size



eta = float(algoinfo["eta"])
eta_exp = float(algoinfo["eta_exp"])
rho = float(algoinfo["rho"])
rho_exp = float(algoinfo["rho_exp"])
reg = float(algoinfo["reg"])

eta_fw = float(fwinfo["eta"])
eta_exp_fw = float(fwinfo["eta_exp"])
L_fw = int(fwinfo["l"])


shape = (f,c)
x = np.ones(shape)
y = np.zeros(shape)
d = np.zeros(shape)
h = np.zeros(shape)
v = np.zeros(shape)



path = "dataset/"+dataset
data = pd.read_csv(path, header = None)
data_tmp = data.to_numpy()
data = data_tmp[:,1:]
data[:,0] = data[:,0] -1 
print(data[:,0])

x_data = np.zeros([f,data.shape[0]],dtype="float64")
y_data = np.zeros([data.shape[0], c],dtype=int)
x_data = data[:,1:].T
x_data = x_data
y_data[range(data.shape[0]),data[:,0].astype('int64')] = 1




def loss_not_used(x,x_data,y_data):
	z =  x_data @ x
	n = x_data.shape[0]
	z =  x_data @ x
	exp_z = np.exp(-z)
	return 1/n *(np.trace(z @ y_data.T) + np.sum(np.log(np.sum(np.exp(-z),axis=1)))) 


def loss(x,x_data,y_data):
	data_size = x_data.shape[1]
	z = x.T @ x_data
	#print(x.T.shape)
	#print(x_data.shape)
	tmp_exp = np.exp(z)
	#print(tmp_exp.shape)
	tmp_numerator = np.zeros((1,data_size))
	#print(tmp_numerator.shape)
	for i in range(data_size):
		j = y_data[i].nonzero()
		tmp_numerator[0,i] = tmp_exp[j,i]
	res =  np.log(tmp_numerator / np.sum(tmp_exp,axis=0))
	return - np.mean(res)

def loss_offline(x,t):
	return loss(x,x_data[:,:(t+1) * batch_size],y_data[:(t+1) * batch_size])

def loss_online(x,t):
	k = t * batch_size
	return loss(x,x_data[:,k:k + batch_size],y_data[k:k + batch_size])

def compute_gradient(x,x_data,y_data):
	data_size = x_data.shape[1]
	z = x.T @ x_data
	tmp_exp = np.exp(z)
	tmp_denominator= np.sum(tmp_exp,axis=0)
	tmp_exp = np.divide(tmp_exp,tmp_denominator)
	for i in range(data_size):
		j = y_data[i].nonzero()
		tmp_exp[j,i] = tmp_exp[j,i] - 1
	return (x_data / data_size) @ tmp_exp.T


def compute_gradient_offline(x,t):
	return compute_gradient(x,x_data[:,:(t+1)*batch_size],y_data[:(t+1)*batch_size])

def compute_gradient_online(x,t):
	k = t* batch_size
	j = k + batch_size
	return compute_gradient(x,x_data[:,k:j], y_data[k:j])

def lmo(z):
	res = csr_matrix(np.zeros(shape))
	max_rows = np.argmax(np.abs(z), axis=0)
	values = -r * np.sign(z[max_rows,range(c)])
	res[max_rows, range(c)] = values
	return res

# def lmo(V):
# 	radius = r
# 	num_rows, num_cols = V.shape
# 	rows = np.argmax(np.abs(V),0)
# 	cols = np.arange(num_cols)
# 	flatten_V = V.T.ravel()[rows + cols * num_rows]
# 	values = -radius * np.sign(flatten_V)
# 	res = csr_matrix((values,(rows, cols)), shape=V.shape)
# 	return res




def update_x(x, v, eta_coef, eta_exp, t):
    eta = min(pow(eta_coef / (t + 1),eta_exp), 1.0)
    return x + eta*(v - x)

def FW(t):
	x = np.zeros(shape)
	for l in range(L_fw):
		gradient = compute_gradient_offline(x,t)
		v = lmo(gradient)		                
		x = update_x(x, v, eta_fw, eta_exp_fw, t)
	return x

def MFW():
	x = csr_matrix(np.zeros(shape))
	v = csr_matrix(np.zeros(shape))
	o = [csr_matrix(np.zeros(shape)) for _ in range(L+1)]
	a = csr_matrix(np.zeros(shape))
	g = csr_matrix(np.zeros(shape))
	res = [csr_matrix(np.zeros(shape)) for _ in range(T)]
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
			
			v = lmo(o[l]*reg+noise)
			
			x = x + eta_l * (v - x)
			g = (1-rho_l) * g + rho_l * compute_gradient_online(x,t)
			g = compute_gradient_online(x,t)
			o[l] = o[l] + g
			

		res[t] = x
	return res

def result_path():
	path = "batch_size"+str(batch_size)+"-N"+str(num_nodes)+"-T"+str(T)+"-L"+str(L)+"-2"
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


def compute_offline_optimal():
	offline_optimal = pd.read_csv("dataset/optimal_lr.csv", header=None).to_numpy()
	#print(shape,offline_optimal.shape)
	#print(offline_optimal[:,1:].shape)
	#offline_optimal = FW(T)
	return offline_optimal[:,1:]

def regret(online_output,offline_optimal):
	#node_loss = [loss_online(online_output[t],t) - loss_online(offline_optimal,t) for t in range(T)]
	node_loss = [loss_online(online_output[t],t) -  loss_online(offline_optimal,t)  for t in range(T)]	
	regrets = np.cumsum(node_loss)
	#regrets = [ cum_online_loss[t]  for t in range(T)] 
	#regrets = [ cummulitative_node_loss[t] - loss_offline(offline_optimal,t) for t in range(T)]
	return regrets



if __name__ == "__main__":
	start = time.time()	

	

	offline_optimal = compute_offline_optimal()
	opt_loss = [loss_online(offline_optimal,t) for t in range(T)]
	#print(loss_offline(offline_optimal,T))
	#print(sum(opt_loss)/T)
	#print(loss_offline(np.ones(shape),T))
	#print(np.max(compute_gradient_offline(offline_optimal,T),axis=0))
	print("---------------------------------------------")
	A = lmo(compute_gradient_offline(offline_optimal,T))
	#print(max(A[A.nonzero()]))
	online_output = MFW()
	
	
	regrets = regret(online_output,offline_optimal)
	
	draw_regret(regrets,"regrets/"+result_path()+".png")
	print("The regret is in the file " + "regrets/"+result_path()+".png")
	

	end = time.time()
	print("Time taken : " + str(end - start)+ "s")



