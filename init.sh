#!/bin/bash

function create_dir(){
    dirname=$1
    if [ ! -d $dirname  ];then
    mkdir $dirname
    fi
}

create_dir "regrets"


echo "Set up is ready to use."