#!/usr/bin/python3

import subprocess
import os
import sys

def runcmd(cmd, verbose = False, *args, **kwargs):
	process = subprocess.Popen(
		cmd,
		stdout = subprocess.PIPE,
		stderr = subprocess.PIPE,
		text = True,
		shell = True
	)
	std_out, std_err = process.communicate()

	if verbose:
		print(std_out.strip(), std_err)
	pass

if len(sys.argv) == 2 :
	if sys.argv[1].upper() == "MNIST":
		if not os.path.exists("./dataset"):
			os.makedirs("dataset")
		runcmd("wget -O sorted_mnist.csv https://edge-intelligence.imag.fr/preprocessed_mnist_dataset/sorted_mnist.csv", verbose = False)
		runcmd("mv sorted_mnist.csv dataset/")
		print("You can find the dataset in : `dataset/sorted_mnist.csv`")

	if sys.argv[1].upper() == "CIFAR10":
		if not os.path.exists("./dataset"):
			os.makedirs("dataset")
		runcmd("wget -O sorted_cifar10.csv https://edge-intelligence.imag.fr/preprocessed_cifar10_dataset/sorted_cifar10.csv", verbose = False)
		runcmd("mv sorted_cifar10.csv dataset/")
		print("You can find the dataset in : `dataset/sorted_cifar10.csv")