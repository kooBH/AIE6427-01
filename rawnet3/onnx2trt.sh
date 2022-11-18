#!/bin/bash

PATH_OUT=/host_ws/rawnet3/chkpt/rawnet3.engine
PATH_OUT=/host_ws/TRTIS/repository/rawnet3/1/model.plan

#/workspace/tensorrt/bin/trtexec --onnx=/host_ws/rawnet3/chkpt/rawnet3.onnx --saveEngine=/host_ws/rawnet3/chkpt/rawnet3.engine --verbose --dumpLayerInfo --dumpProfile

##22.10
/workspace/tensorrt/bin/trtexec --onnx=/host_ws/rawnet3/chkpt/rawnet3.onnx --saveEngine=${PATH_OUT} --minShapes=input:1x64000   --buildOnly

##220.05
#/workspace/tensorrt/bin/trtexec --onnx=/host_ws/rawnet3/chkpt/rawnet3.onnx --saveEngine=${PATH_OUT} --minShapes=input:1x64000  --optShapes --maxShapes  --buildOnly
