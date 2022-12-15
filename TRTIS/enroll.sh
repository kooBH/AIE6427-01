#!/bin/bash

PATH+=:/usr/bin

for ID in ../data/in/*;
do 
    echo $ID;
    IFS='/' read -ra ADDR <<< $ID
    for PATH in  ../data/in/${ADDR[3]}/*.wav;
        do
        echo $PATH
        /usr/bin/python infer.py --enroll ${ADDR[3]} --path $PATH
        done

done