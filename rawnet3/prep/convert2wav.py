import os,glob
import numpy as np
import random
import librosa as rs
import soundfile as sf

from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count

# Due to 'PySoundFile failed. Trying audioread instead' 
import warnings
warnings.filterwarnings('ignore')

dir_voxceleb2 = "/home/data2/kbh/voxceleb2"
dir_out = "/home/data2/kbh/voxceleb2_wav"

list_id = glob.glob(os.path.join(dir_voxceleb2,"*"))

def process(idx):
    list_audio = glob.glob(os.path.join(list_id[idx],"**","*.m4a"))

    id_audio = list_audio[0].split("/")[-3]

    os.makedirs(os.path.join(dir_out,id_audio),exist_ok=True)

    for path_audio in list_audio :
        name_audio = path_audio.split("/")[-1]
        name_audio2 = name_audio.split(".")[0] + ".wav"

        x = rs.load(path_audio,sr=16000)[0]
        sf.write(os.path.join(dir_out,id_audio,name_audio2),x,16000)
   
if __name__=='__main__': 
    cpu_num = cpu_count()

    arr = list(range(len(list_id)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(process, arr), total=len(arr),ascii=True,desc='processing'))


