#!/bin/bash


#/workspace/tensorrt/bin/trtexec --onnx=/host_ws/rawnet3/chkpt/rawnet3.onnx --saveEngine=/host_ws/rawnet3/chkpt/rawnet3.engine --verbose --dumpLayerInfo --dumpProfile

##22.10
VERSION=v0
PATH_OUT=/host_ws/TRTIS/repository/rawnet3/${VERSION}/model.plan
/workspace/tensorrt/bin/trtexec --onnx=/host_ws/rawnet3/chkpt/rawnet3_${VERSION}.onnx --saveEngine=${PATH_OUT} --minShapes=input:1x64000   --buildOnly


VERSION=v1
PATH_OUT=/host_ws/TRTIS/repository/rawnet3/${VERSION}/model.plan
/workspace/tensorrt/bin/trtexec --onnx=/host_ws/rawnet3/chkpt/rawnet3_${VERSION}.onnx --saveEngine=${PATH_OUT} --minShapes=input:1x64000   --buildOnly

##220.05
#/workspace/tensorrt/bin/trtexec --onnx=/host_ws/rawnet3/chkpt/rawnet3.onnx --saveEngine=${PATH_OUT} --minShapes=input:1x64000  --optShapes --maxShapes  --buildOnly
