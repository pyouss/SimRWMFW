#!/bin/bash

local_config_envi="../config/envi.conf"
node_list_path="../config/node.conf"
local_config_param="../config/param.conf"

function get_config(){
    sed -nr "/^\[$1\]/ { :l /^$2[ ]*=/ { s/[^=]*=[ ]*//; p; q;}; n; b l;}" $3
}

# Read config data from config.ini
user=$(get_config ENVCONFIG user $local_config_envi)
server=$(get_config ENVCONFIG server $local_config_envi)
num_nodes=$(get_config ALGOCONFIG num_nodes $local_config_param)

scp $1 $user@$server:dataset/$1

for((i=0;i<$num_nodes;i++));  
do
    name="node_"${i}   
    node=$(get_config NODEINFO $name $node_list_path)
    ssh $user@$server walt node cp dataset/$1 ${node}:/persist/$1
done 
