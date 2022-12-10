#!/bin/bash

VERSION=v0
DEVICE=cuda:1

# 2022-12-10, margin 0.15
VERSION=v1
python train.py --config config/${VERSION}.yaml --default config/default.yaml --version ${VERSION} --device ${DEVICE}