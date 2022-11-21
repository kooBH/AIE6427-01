import os,glob
import numpy as np
import random

from tqdm.auto import tqdm


dir_voxceleb2 = "/home/data2/kbh/voxceleb2"
ws = os.path.dirname(os.path.abspath(__file__))
ratio_train  = 0.9

list_id = glob.glob(os.path.join(dir_voxceleb2,"*"))
list_num = []

f_train = open(os.path.join(ws,"..","data","train.txt"),"w")
f_test = open(os.path.join(ws,"..","data","test.txt"),"w")

for idx in tqdm(range(len(list_id))) : 
    list_audio = glob.glob(os.path.join(list_id[idx],"**","*.m4a"))
    list_num.append(len(list_audio))

    random.shuffle(list_audio)
    idx_train = int(ratio_train * len(list_audio))

    list_train = list_audio[:idx_train]
    list_test = list_audio[idx_train:]

    for path_train in list_train :
        f_train.write("{}\n".format(path_train))

    for path_test in list_test :
        f_test.write("{}\n".format(path_test))

f_train.close()
f_test.close()

print(np.min(list_num))
print(np.max(list_num))
print(np.mean(list_num))


