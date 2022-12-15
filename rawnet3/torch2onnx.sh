#/bin/bash

VERSION=v0
python torch2onnx.py  --config config/${VERSION}.yaml --default config/default.yaml --chkpt /home/nas/user/kbh/chkpt/${VERSION}/epoch_9_loss_5.163061041251538.pt -v ${VERSION}

VERSION=v1
python torch2onnx.py  --config config/${VERSION}.yaml --default config/default.yaml --chkpt /home/nas/user/kbh/chkpt/${VERSION}/epoch_19_loss_2.242471431697972.pt -v ${VERSION}


#VERSION=v3
#python torch2onnx.py  --config config/${VERSION}.yaml --default config/default.yaml --chkpt /home/nas/user/kbh/chkpt/${VERSION}/epoch_19_loss_2.242471431697972.pt


