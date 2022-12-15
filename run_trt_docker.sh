#!/bin/bash

docker run --gpus all -it --rm -v /home/kbh/work/AIE6427-01-CS:/host_ws nvcr.io/nvidia/tensorrt:22.10-py3

#docker run --gpus all -it --rm -v /home/kbh/work/AIE6427-01:/host_ws nvcr.io/nvidia/tensorrt:22.05-py3
