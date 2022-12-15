#!/bin/bash

PATH+=:/usr/bin

for PATH in ../data/sample/*/*;
do 
    #echo $PATH
    #/usr/bin/python infer.py --path $PATH --version 1
    /usr/bin/python infer.py --path $PATH --version 2

done