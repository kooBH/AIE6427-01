# AIE6427-01  

## What this do  

Speaker Recognition Using Rawnet3 on Trition Inference Server.  

---

## NOTE

* for all ```.sh``` scripts use may need to change hard-coded paths  
+ All versions of environments should be matched. to match pytorch version I used pytorch docker.
    + nvcr.io/nvidia/pytorch:22.10-py3
    + nvcr.io/nvidia/tensorrt:22.10-py3
    + nvcr.io/nvidia/tritonserver:22.10-py3
    + Driver Version : 520.56.06
    + CUDA Version 11.8

    If any of these dosen't match, there may be error.  

## How to Train  
Run ```rawnet3/train.sh```. see ```rawnet3/train_d1.sh``` for detail usage.  
Use Voxceleb2 as train Voxceleb1 as test,  need to specify dataset path in ```rawnet3/config/default.yaml``` ```.data.vox1``` ,```.data2.vox2```  
and also specify ```log.root```.

## How to Convert Model  

Coversion procedure is ```pytorch```->```ONNX```->```TensorRT```  

Run ```rawnet3/torch2onnx.py``` with  ```rawnet3/torch2onnx.sh```.  
And to use ```trtexec```, run ```run_trt_docker.sh```.  
In the docker, run ```rawnet3/onxx2trt.sh```

## Triton Inferense Server  

[model.plan of v2](https://drive.google.com/file/d/1xk_3on9PGGPc2BOyWZmDtpOrLpY4zPLD/view?usp=sharing)  
Place it into
```
TRTIS/repository/rawnet3/2/model.plan
```

## How to Use Triton Inferense Server  
Run ```TRTIS/run_TRTIS.sh``` for inferense Server.  
Run ```TRTIS/run_client.sh``` for client Server.  
In the client server, first run ```TRTIS/install_client_dependency.sh``` for dependency and  use ```/host_ws/TRTIS/infer.py``` for client-side infernece.  

see ```TRTIS/enroll.sh``` and ```TRTIS/eval.sh``` for detail usage.  



# Reference

[Triton inference server - server](https://github.com/triton-inference-server/server)  
  
[Triton inference server - client](https://github.com/triton-inference-server/client)  

[rawnet3](https://github.com/clovaai/voxceleb_trainer)



[Triton Inference Server에서 TensorRT Engine Inference](https://velog.io/@pjs102793/Triton-Inference-Server%EC%97%90%EC%84%9C-TensorRT-Engine-Inference)
