import torch
import sys, time, os, argparse
sys.path.append("model")
import RawNet3 as RawNet3
import numpy as np


if __name__ == "__main__" : 
    n = 1000
    # Load model
    model  = (RawNet3.MainModel(
        #nOut=512,
        nOut=256,
        encoder_type="ECA",
        sinc_stride=10,
        max_frame = 200,
        sr=16000
        ))
    #model.load_state_dict(torch.load("chkpt/model.pt"))
    #model.load_state_dict(torch.load("chkpt/rawnet3.pt"))
    model.load_state_dict(torch.load("chkpt/model.pt")["model"])
    model.eval()
    model = model.to("cuda:1")
    print("pytorch model loaded")

    print("warming up")
    for i in range(int(n/10)) : 
        x = torch.rand(1,64000).to("cuda:1")
        feat = model(x)


    print("Run")
    x = torch.rand(1,64000).to("cuda:1")
    tic = time.time()
    for i in range(n) : 
        feat = model(x)
    toc = time.time()
    print('Elapsed time for %d inferences for 4 sec data: %s' % (n,toc - tic))

    x = np.load("input.npy")
   # x = np.ones(x.shape)
    feat = model(torch.from_numpy(x).to("cuda:1").float())
    print(feat.shape)
    feat = feat.detach().cpu().numpy()
    print(np.mean(np.abs(feat)))
    print(np.sum(np.abs(feat)))
    print(np.max(np.abs(feat)))
    print(np.min(np.abs(feat)))
    np.save("pytorch.npy",feat)
