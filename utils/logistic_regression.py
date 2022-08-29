from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
import numpy as np

def compute_gradient(x,x_data,y_data):
	data_size = x_data.shape[1]
	z = x.T @ x_data
	tmp_exp = np.exp(z)
	tmp_denominator= np.sum(tmp_exp,axis=0)
	tmp_exp = tmp_exp / tmp_denominator
	tmp_exp[y_data,range(data_size)] = tmp_exp[y_data,range(data_size)] - 1
	return (x_data / data_size) @ tmp_exp.T

def lmo(o,r):
	res = np.zeros(o.shape)
	max_rows = np.argmax(np.abs(o), axis=0)
	values = -r * np.sign(o[max_rows,range(o.shape[1])])
	res[max_rows, range(o.shape[1])] = values
	return res


def loss(x,x_data,y_data):
	data_size = x_data.shape[1]
	z = x.T @ x_data
	tmp_exp = np.exp(z)
	tmp_numerator = tmp_exp[y_data,range(data_size)]
	return - np.mean(np.log(tmp_numerator / np.sum(tmp_exp,axis=0)) )
