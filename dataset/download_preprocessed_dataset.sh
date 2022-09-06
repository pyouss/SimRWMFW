#!/bin/bash

if [ $# -eq 0 ];then 
	echo Error : Argument missing.
	echo Arguments options = \[ mnist, cifar10 \]
	echo Example to download the preprocessed mnist run the following command : 
	echo "	./download_preprocess_dataset.sh mnist"
	echo To download the preprocessed mnist run the following command : 
	echo "	./download_preprocess_dataset.sh cifar10"
fi

if [[ $1 == "mnist" ]];then
	wget -O sorted_mnist.csv https://edge-intelligence.imag.fr/preprocessed_mnist_dataset/sorted_mnist.csv
fi

if [[ $1 == "cifar10" ]];then
	wget -O sorted_cifar10.csv https://edge-intelligence.imag.fr/preprocessed_cifar10_dataset/sorted_cifar10.csv
fi