import os
import numpy as np 
import matplotlib.pyplot as plt



bg_path = "./background/one_sky_backgrouod.npy"


bg = np.load(bg_path)

zeros = np.zeros((1,6,384,384))

zeros[:,:,128:256,128:256] = bg[:,:,256:384, 256:384]
zeros /= np.max(zeros)

np.save("./background/center_128.npy",zeros)
print(zeros.shape)
plt.imshow(zeros[0,0,:,:])
plt.show()

print(np.max(zeros))

