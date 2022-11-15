#!/bin/bash

#/workspace/tensorrt/bin/trtexec --onnx=/host_ws/rawnet3/chkpt/rawnet3.onnx --saveEngine=/host_ws/rawnet3/chkpt/rawnet3.engine --verbose --dumpLayerInfo --dumpProfile
/workspace/tensorrt/bin/trtexec --onnx=/host_ws/rawnet3/chkpt/rawnet3.onnx --saveEngine=/host_ws/rawnet3/chkpt/rawnet3.engine --minShapes=input:1x64000   --buildOnly

