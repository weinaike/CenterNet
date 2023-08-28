import scipy.io as sciio
import numpy as np
import glob
import os 

path_list = ['230822_1532', '230818_1001', '230821_0915', '230817_1647', '230822_1831', '230822_1149', 
             '230821_1615', '230822_0935', '230818_1503', '230821_1408', '230816_1422', '230817_1418', 
             '230821_1432', '230817_1201', '230822_1039', '230822_1403', '230822_1714', '230816_1614', 
             '230817_1020', '230821_1136', '230821_1025', '230818_1356', '230818_1112']
path_list.sort()

print(path_list)


coff = np.array([0.3227,0.3219,0.3208,0.3205,0.4532,0.4438])

for path in path_list:
    mats = glob.glob(path+"/*.mat")
    mats.sort()
    batch = list()
    for mat in mats:
        print(mat)
        try:
            sky = sciio.loadmat(mat)["SkyBK_MS"].transpose(2,0,1)      
        except:
            continue
        print(sky.shape) 
        sky = np.array(sky, dtype=float)
        sky = sky / coff.reshape(len(coff), 1, 1)
        sky /= np.max(sky)
        sky = sky[:,::2, ::2]
        batch.append(sky)
    batch = np.array(batch, dtype=float)
    np.save(os.path.join("background/{}.npy".format(path)), batch)

