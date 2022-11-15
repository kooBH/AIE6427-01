import numpy as np

#x = np.random.rand(1,64000)
#np.save("input.npy",x)

a = np.load("TRT.npy")
b = np.load("pytorch.npy")

print("For Identical input ")
print("TRT output mean : {}".format(np.mean(a)))
print("pytorch output mean : {}".format(np.mean(b)))

print(np.sum(np.abs(a-b)))