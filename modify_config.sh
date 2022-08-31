#!/bin/bash

res=$(python3 utils/p_config.py $@)

if [[ $res == "1" ]];then
	less config/graph.conf
	less config/param.conf
else
	echo $res
fi