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
dim = f * c # dimension of the output and messages
batch_size = num_nodes * decentralized_batch_size



eta = float(algoinfo["eta"])
eta_exp = float(algoinfo["eta_exp"])
rho = float(algoinfo["rho"])
rho_exp = float(algoinfo["rho_exp"])


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
data = data.to_numpy()
print(data.shape[0])

x_data = np.zeros([data.shape[0], f])
y_data = np.zeros([data.shape[0], c])
x_data = data[:,1:]
print(data[50666:50669,0])
y_data [range(data.shape[0]),data[:,0].astype('int64')] = 1
print("ok")


def loss(x,x_data,y_data):
	z =  x_data @ x
	n = x_data.shape[0]
	return 1/n *(np.trace(z @ y_data.T) + np.sum(np.log(np.sum(np.exp(-z),axis=1)+1e-8))) 


def loss_offline(x,t):
	return loss(x,x_data[:(t+1) * batch_size],y_data[:(t+1) * batch_size])

def loss_online(x,t):
	k = t * batch_size
	return loss(x,x_data[k:k + batch_size],y_data[k:k + batch_size])

def compute_gradient(x,x_data,y_data):
	z = x_data @ x
	p = softmax(-z, axis=1)
	n = x_data.shape[0] 
	mu = 0.01
	gradient = 1/n * (x_data.T @ (y_data - p)) + 2 * mu * x
	return gradient

def compute_gradient_offline(x,t):
	return compute_gradient(x,x_data[:(t+1)*batch_size],y_data[:(t+1)*batch_size])

def compute_gradient_online(x,t):
	k = t* batch_size
	j = k + batch_size
	return compute_gradient(x,x_data[k:j], y_data[k:j])

def lmo(o):
	res = np.zeros(shape)
	res[np.argmax(abs(o), axis=0), range(c)] = -r * np.sign(np.max(o,axis=0))
	return res

def update_x(x, v, eta_coef, eta_exp, t):
    eta = min(pow(eta_coef / (t + 1),eta_exp), 1.0)
    return x + eta*(x - v)

def FW(t):
	x = np.zeros(shape)
	for l in range(L_fw):
		gradient = compute_gradient_offline(x,t)
		v = lmo(gradient)		                
		x = update_x(x, v, eta_fw, eta_exp_fw, t)
	return x

def MFW():
	xs = [np.zeros(shape) for _ in range(L+1)]
	v = np.zeros(shape)
	o = [np.zeros(shape) for _ in range(L+1)]
	a = np.zeros(shape)
	g = np.zeros(shape)
	res = [np.zeros(shape) for _ in range(T)]
	for t in range(T):
		xs[0] = np.zeros(shape)
		for l in range(1,L+1):
			eta_l = min(eta / pow((l + 1),eta_exp), 1.0)
			v = lmo(o[l])
			xs[l] = update_x(xs[l-1], v, eta_l, 1, t)
		res[t] = xs[L]
		g = compute_gradient_online(xs[0],t)
		o[0] = g		
		for l in range(1,L+1):
			rho_l = min(rho / pow((l + 1),rho_exp), 1.0)
			g = (1-rho_l)* g + rho_l * compute_gradient_online(xs[l],t)
			o[l] = o[l-1] + g
	return res

def result_path():
	path = "batch_size"+str(batch_size)+"-N"+str(num_nodes)+"-T"+str(T)+"-L"+str(L)
	return path


def draw_regret(regret, name):
    figure(1, figsize=(10, 6))
    x_axis = [i for i in range(1, T+1)]
    plt.scatter(x_axis, regret)
    title = result_path()
    plt.title(title)
    plt.xlabel("Number of Rounds T")
    plt.ylabel("Regret")
    plt.savefig(name)




def regret():
	offline_optimal = FW(T-1)
	node_loss = [loss_online(results[t],t) for t in range(T)]
	cummulitative_node_loss = [sum(node_loss[:t]) for t in range(T)] 
	regrets = [ cummulitative_node_loss[t] - loss_offline(offline_optimal,t) for t in range(T)]

	draw_regret(regrets,"regrets/"+result_path()+".png")
	print("The regret is in the file " + "regrets/"+result_path()+".png")


if __name__ == "__main__":
	start = time.time()	
	results = MFW()
	regret()
	end = time.time()
	print("Time taken : " + str(end - start)+ "s")