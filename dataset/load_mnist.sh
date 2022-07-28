#!/bin/bash


if [[ $1 == "sort" ]];then
	python3 sorted_mnist.py 
elif [[ $1 == "shuffle" ]];then 
	python3 shuffled_mnist.py
elif [[ $1 == "init" ]];then 
	wget -c https://pjreddie.com/media/files/mnist_train.csv -O mnist0.csv 
fi 