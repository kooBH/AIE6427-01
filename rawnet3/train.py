import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

import argparse
import os
import numpy as np

from tensorboardX import SummaryWriter

from model.RawNet3 import MainModel
from classifier.aamsoftmax import aamsoftmax
from module.dataset import VoxDataset

from ptUtils.hparams import HParam
from ptUtils.writer import MyWriter

from module.common import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help="yaml for configuration")
    parser.add_argument('--default', type=str, required=True,
                    help="default configuration")

    parser.add_argument('--version_name', '-v', type=str, required=True,
                        help="version of current training")
    parser.add_argument('--chkpt',type=str,required=False,default=None)
    parser.add_argument('--step','-s',type=int,required=False,default=0)
    parser.add_argument('--device','-d',type=str,required=False,default="cuda:0")
    args = parser.parse_args()

    hp = HParam(args.config,args.default)
    print("NOTE::Loading configuration : "+args.config)

    device = args.device
    version = args.version_name
    torch.cuda.set_device(device)

    batch_size = hp.train.batch_size
    num_epochs = hp.train.epoch
    num_workers = hp.train.num_workers

    best_loss = 1e7

    ## load

    modelsave_path = hp.log.root +'/'+'chkpt' + '/' + version
    log_dir = hp.log.root+'/'+'log'+'/'+version

    os.makedirs(modelsave_path,exist_ok=True)
    os.makedirs(log_dir,exist_ok=True)

    writer = MyWriter(hp, log_dir)

    ## target
    train_dataset = VoxDataset(hp, is_vox1=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)

    model = MainModel().to(device)
    classifier = aamsoftmax(hp).to(device)

    if not args.chkpt == None : 
        print('NOTE::Loading pre-trained model : '+ args.chkpt)
        model.load_state_dict(torch.load(args.chkpt, map_location=device))

    criterion = None
    if hp.loss.type == "CrossEntropyLoss" :
        criterion = nn.CrossEntropyLoss()
    else : 
        raise Exception("ERROR::Unknown loss : {}".format(hp.loss.type))

    optimizer = torch.optim.Adam(model.parameters(), lr=hp.train.adam)
    scaler = torch.cuda.amp.GradScaler()

    if hp.scheduler.type == "CosineAnnealingLR" : 
       scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp.scheduler.CosineAnnealingLR.T_max, eta_min=hp.scheduler.CosineAnnealingLR.eta_min) 
    else :
        raise Exception("Unsupported sceduler type")

    step = args.step
    """
    cuda.amp
    [Pytorch] apex / amp 모델 학습 빠르게 시키는 법.
    https://dbwp031.tistory.com/33
    """

    for epoch in range(num_epochs):
        ### TRAIN ####
        model.train()
        train_loss=0
        for i, (data,label) in enumerate(train_loader):
            step +=1
            data = data.to(device)
            label = label.to(device)
            
            loss = run(data,label,model,classifier,criterion)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
           
            print('TRAIN::{} : Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(version,epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

            if step %  hp.train.summary_interval == 0:
                writer.log_value(loss,step,'train loss : '+hp.loss.type)

        train_loss = train_loss/len(train_loader)
        torch.save(model.state_dict(), str(modelsave_path)+'/epoch_{}_loss_{}.pt'.format(epoch,train_loss))
        scheduler.step()
            