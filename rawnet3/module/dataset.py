from os.path import join
from glob import glob
import librosa as rs
import torch
import numpy as np

class VoxDataset(torch.utils.data.Dataset):
    def __init__(self,hp, is_vox1=False):
        self.hp = hp
        self.is_vox1 = is_vox1

        self.len_data = int(hp.data.len_sec*hp.data.sr)

        if is_vox1 :
            root = hp.data.vox1
            # root/id_spk/id_ytb/id_f.wav
            self.list_data = glob(join(root,"**","*.wav"),recursive=True)
            list_id = glob(join(root,"id*"))
        else  :
            root = hp.data.vox2
            # root/id_spk/id_f.wav
            self.list_data = glob(join(root,"**","*.wav"))
            list_id = glob(join(root,"id*"))

        list_id = [x.split('/')[-1] for x in list_id ]

        self.dict_id = {x : i for i,x in enumerate(list_id)}

    def __getitem__(self, idx):
        path_data = self.list_data[idx]

        data = rs.load(path_data,sr=self.hp.data.sr,mono=True)[0]

        ## sampling
        # cut
        if len(data) > self.len_data :
            idx_start = np.random.randint(len(data)-self.len_data)
            data = data[idx_start:idx_start + self.len_data]
        # pad
        else : 
            shortage = self.len_data - len(data)
            data = np.pad(data,(0,shortage))

        ## Augmentation

        ## Label  
        if self.is_vox1 : 
            id_data = path_data.split("/")[-3]
        # root/id_spk(idxxxxx)/id_f.wav
        else : 
            id_data = path_data.split("/")[-2]
        
        label = self.dict_id[id_data]

        return data,label

    def __len__(self):
        return len(self.list_data)