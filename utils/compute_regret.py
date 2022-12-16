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
expinfo = param_config["EXPERIMENTCONFIG"]

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
num_trials = int(expinfo["num_trials"])


eta = float(algoinfo["eta"])
eta_exp = float(algoinfo["eta_exp"])
rho = float(algoinfo["rho"])
rho_exp = float(algoinfo["rho_exp"])
reg = float(algoinfo["reg"])/np.sqrt(100)

eta_fw = float(fwinfo["eta"])
eta_exp_fw = float(fwinfo["eta_exp"])
L_fw = int(fwinfo["l"])

algo = algoinfo["algo"]


shape = (f,c)

path = f"dataset/{dataset}"
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


def compute_stoch_gradient_online(x,t,s):
	k = t * batch_size
	j = k + batch_size
	#sub_batch = random.sample(range(k,j),s)
	sub_batch = np.floor(np.random.rand(s)*batch_size).astype(int) + k
	return log_r.compute_gradient(x,x_data[:,sub_batch],y_data[sub_batch])


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
	L_fw = 100
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
	for t in range(T):
		x = np.zeros(shape)
		
		for l in range(0,L+1):
			eta_l = min(eta / pow((l + 1),eta_exp), 1.0)
			rho_l = min(rho / pow((l + 1),rho_exp), 1.0)

			noise = -0.5 + np.random.rand(shape[0],shape[1])
			v = log_r.lmo(o[l]*reg+noise,r)
			
			
			g = (1-rho_l) * g + rho_l * compute_stoch_gradient_online(x,t,4)
			o[l] = o[l] + g
			x = x + eta_l * (v - x) 
		res[t] = x
	return res


def MFW_dense():
	x = np.zeros(shape)
	v = np.zeros(shape)
	o = [np.zeros(shape) for _ in range(L+1)]
	a = np.zeros(shape)
	g = np.zeros(shape)
	res = [np.zeros(shape) for _ in range(T)]
	for t in range(T):
		x = np.zeros(shape)
		
		for l in range(0,L+1):
			eta_l = min(eta / pow((l + 1),eta_exp), 1.0)
			rho_l = min(rho / pow((l + 1),rho_exp), 1.0)
			print(f'{eta_l=}')
			print(f'{rho_l=}')
			print(f'{reg=}')

			noise = -0.5 + np.random.rand(shape[0],shape[1])
			v = log_r.lmo(o[l]*reg+noise,r)
			
			
			g = (1-rho_l) * g + rho_l * compute_stoch_gradient_online(x,t,4)
			o[l] = o[l] + g
			x = x + eta_l * (v - x) 
		res[t] = x
	return res


def RWMFW(P):
	x = lil_matrix(shape)
	v = lil_matrix(shape)
	o = [lil_matrix(shape) for _ in range(L+1)]
	g = lil_matrix(shape)
	res = [lil_matrix(shape) for _ in range(T)]
	n = P.shape[0]
	
	for t in range(T):
		i = next_agent((1/n)*np.ones((1,n)),0)
		x = np.zeros(shape)
		
		for l in range(0,L+1):
			eta_l = min(eta / pow((l + 1),eta_exp), 1.0)
			rho_l = min(rho / pow((l + 1),rho_exp), 1.0)

			noise = -0.5 + np.random.rand(shape[0],shape[1])
			
			v = log_r.lmo(o[l]*reg+noise,r)
			
			g = compute_gradient_dist_online(x,t,n,i)
			#g = (1-rho_l) * g + rho_l * compute_gradient_dist_online(x,t,n,i)
			o[l] = o[l] + g
			x = x + eta_l * (v - x)
			i = next_agent(P,i) 
		res[t] = x
	return res


def DMFW(P):
	print(f'{reg=}')
	n = P.shape[0]
	x = [lil_matrix(shape) for _ in range(n)]
	y = [lil_matrix(shape) for _ in range(n)]
	v = [lil_matrix(shape) for _ in range(n)] 
	o = [[lil_matrix(shape) for _ in range(L+1)] for _ in range(n)]
	g = [lil_matrix(shape) for _ in range(n)]
	d = [lil_matrix(shape) for _ in range(n)]
	res = [lil_matrix(shape) for _ in range(T)]
	
	for t in range(T):
		for i  in range(n):
			x[i] = np.zeros(shape)
			g[i] = compute_gradient_online(x[i],t)
			o[i][0] = o[i][0] + g[i]
		
		for l in range(1,L+1):
			eta_l = min(eta / pow((l + 1),eta_exp), 1.0)
			rho_l = min(rho / pow((l + 1),rho_exp), 1.0)

			noise = -0.5 + np.random.rand(shape[0],shape[1])
			
			v[i] = log_r.lmo(o[i][l]*reg+noise,r)
			for i in range(n):
				y[i] = np.zeros(shape)
				for j in P[i,:].nonzero()[0]:
					y[i] += x[j] * P[i,j]

			x[i] = y[i] + eta_l * (v[i] - x[i])
			for i in range(n):
				g[i] = (1-rho_l) * d[i] + rho_l * compute_gradient_dist_online(x[i],t,n,i)
			
			for i in range(n):
				d[i] = np.zeros(shape)
				for j in P[i,:].nonzero()[0]:
					d[i] += g[j] * P[i,j]
			

			for i in range(n):
				g[i] = compute_gradient_dist_online(x[i],t,n,i)
			o[i][l] = o[i][l] + d[i]

		res[t] = x[0]
	return res



def result_path():
	path = f"{algo.lower()}-{graph_name}-{dataname}-batch_size{str(batch_size)}-N{str(num_nodes)}-T{str(T)}-L{str(L)}"
	return path


def draw_regret(regret, name):
    figure(1, figsize=(10, 6))
    x_axis = [i for i in range(1, T+1)]
    plt.plot(x_axis, regret)
    title = name
    plt.title(title)
    plt.xlabel("Number of Rounds T")
    plt.ylabel("Regret")
    plt.savefig(name)
    plt.clf()

def save_multiple_regrets(regrets, regret_file_name):
	df_regrets = pd.DataFrame(regrets)
	df_regrets.to_csv(regret_file_name+".csv", index = False, header = False)
	print(f'Regrets are in the file : {regret_file_name}.csv')


def save_regret_file(regret, regret_file_name):
	df_regret = pd.DataFrame(regret).T
	df_regret.to_csv(regret_file_name+".csv", index=False, header = False)    
	print(f"The regret is in the file: {regret_file_name}.csv")


def compute_offline_optimal(t=T,available=False, optimal_file = None):
	if available :
		offline_optimal = pd.read_csv(optimal_file, header=None).to_numpy()
		return offline_optimal
	offline_optimal = FW(t)
	return offline_optimal

def regret(online_output,offline_optimal=None,fixed=False):
	if fixed :
		node_loss = [(loss_online(online_output[t],t) -  loss_online(offline_optimal,t)) for t in range(T)]	
	else : 
		node_loss = [loss_online(online_output[t],t) - loss_online(compute_offline_optimal(t),t) for t in range(T)]
	regrets_res = np.cumsum(node_loss)
	return regrets_res


def exit_error(msg):
	print("Error : " + msg)
	exit()

if __name__ == "__main__":
	start = time.time()	
	print(f"{dataset[7:-4]=}")
	if dataset[7:-4].upper() == "MNIST":
		offline_optimal = compute_offline_optimal()
	else:
		offline_optimal = compute_offline_optimal(t=None,available=True, optimal_file="dataset/cifar10_optimal.csv")
	
	
	regrets = []
	for trial in range(num_trials):
		online_output = [ np.zeros(shape) for _ in range(T)]
		if algo.upper() == "RWMFW":
			online_output = RWMFW(transition_matrix)
		elif algo.upper() == "MFW":
			graph_name = "centralized"
			num_does = 1
			online_output = MFW()
		elif algo.upper() == "DMFW":
			online_output = DMFW(transition_matrix)
		else :
			algorithms = {"mfw","rwmfw","dmfw"}
			exit_error(f"Algorithm in parameter is wrong. \n 'algo' not in {algorithms} ")	
		regrets.append(regret(online_output,offline_optimal,fixed=True))

		draw_regret(regrets[trial],f"regrets/{result_path()}_{trial}.png")

	save_multiple_regrets(regrets,f"regrets/{result_path()}")
	regrets = np.sum(np.array(regrets), axis=0)/num_trials
	draw_regret(regrets,f"regrets/average_{result_path()}_trials{num_trials}.png")
	save_regret_file(regrets,f"regrets/average_{result_path()}_trials{num_trials}")
	
	print(f"The regret is in the file regrets/{result_path()}.png")

	end = time.time()
	print("Time taken : " + str(end - start)+ "s")
	



