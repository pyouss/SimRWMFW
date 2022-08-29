#!/bin/bash

res=$(python3 p_config.py $@)

if [[ $res == "True" ]];then
	less config/graph.conf
	less config/param.conf
else
	echo $res
fi