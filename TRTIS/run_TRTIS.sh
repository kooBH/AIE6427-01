#!/bin/bash

## cuda 11.8, driver 520
VERSION=22.10

## cuda 11.7, driver 515
#VERSION=22.05

# docker run --gpus 1 -it --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v /home/kbh/work/AIE6427-01/TRTIS/repository:/models nvcr.io/nvidia/tritonserver:22.10-py3  /bin/bash 
docker run --gpus 1 --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v /home/kbh/work/AIE6427-01/TRTIS/repository:/models nvcr.io/nvidia/tritonserver:${VERSION}-py3   tritonserver --model-repository=/models 