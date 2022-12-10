import torch
import torch.nn

def run(waveform,label,embed,classify,criterion,ret_output=False): 
    waveform = waveform

    feature = embed(waveform)
    output = classify(feature,label)

    loss = criterion(output,label)

    if ret_output :
        return output, loss
    else : 
        return loss
