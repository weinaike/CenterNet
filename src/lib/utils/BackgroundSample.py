#coding=utf-8

import os
import numpy as np
import json
import scipy.io as sciio
# bg_dict = { -2: "../../../data/background/sunny_sky_backgrouod.npy",  
#             -3: "../../../data/background/cloudy_sky_backgrouod.npy",
#             -4: '../../../data/background/230822_1532.npy', 
#             -5: '../../../data/background/230818_1001.npy', 
#             -6: '../../../data/background/230821_0915.npy', 
#             -7: '../../../data/background/230817_1647.npy', 
#             -8: '../../../data/background/230822_1831.npy', 
#             -9: '../../../data/background/230822_1149.npy', 
#             -10: '../../../data/background/230821_1615.npy', 
#             -11: '../../../data/background/230822_0935.npy', 
#             -12: '../../../data/background/230818_1503.npy', 
#             -13: '../../../data/background/230821_1408.npy', 
#             -14: '../../../data/background/230816_1422.npy', 
#             -15: '../../../data/background/230817_1418.npy', 
#             -16: '../../../data/background/230821_1432.npy', 
#             -17: '../../../data/background/230817_1201.npy', 
#             -18: '../../../data/background/230822_1039.npy', 
#             -19: '../../../data/background/230822_1403.npy', 
#             -20: '../../../data/background/230822_1714.npy', 
#             -21: '../../../data/background/230816_1614.npy', 
#             -22: '../../../data/background/230817_1020.npy', 
#             -23: '../../../data/background/230821_1136.npy', 
#             -24: '../../../data/background/230821_1025.npy', 
#             -25: '../../../data/background/230818_1356.npy',
#             -26: '../../../data/background/230818_1112.npy'
#         }
def calc_prmse(rect_patchs):
    [c, h, w] = rect_patchs.shape
    tmp = 0
    for i in range(c):
        tmp += np.mean(np.power(rect_patchs[i,:,:],2))
    return np.sqrt(tmp/c)


def gen_compressive_backgournd(otf_fft, sky_bg):

    [wave_count,h,w ] = sky_bg.shape
    sky_bg /= np.max(sky_bg)
    prmse = calc_prmse(sky_bg)

    # expand_h = 1024
    # expand_w = 1024       
    num , expand_h, expand_w = otf_fft.shape
    centery = expand_h//2
    centerx = expand_w//2
    pad_h_1 = (expand_h - h)//2
    pad_h_2 = expand_h - h - pad_h_1
    pad_w_1 = (expand_w - w)//2
    pad_w_2 = expand_w - w - pad_w_1
    sky_bg_3d = np.pad(sky_bg, ((0,0),(pad_h_1, pad_h_2),(pad_w_1,pad_w_2)),mode='constant', constant_values=0)
    sky_bg_fft = np.fft.fft2(np.fft.ifftshift(sky_bg_3d,(1,2)))

    fft_dot = np.multiply(sky_bg_fft, otf_fft)   
    image = np.fft.ifft2(fft_dot)
    image = np.fft.fftshift(image,(1,2))
    image_amp = np.abs(image)

    Intensity = np.sum(image_amp,0)

    # crop
    crop_Intensity = Intensity[(centery-h//2):(centery+h//2), (centerx-w//2):(centerx+w//2)]

    return crop_Intensity, prmse

def save_background(id, key, otf_fft, path, sky_bg):

    background, prmse = gen_compressive_backgournd(otf_fft, sky_bg)
    # print(path, target)
    np.save(os.path.join(path,"background_{}_{:05d}.npy".format(key,id)),background)
    with open(os.path.join(path,"background_{}_{:05d}.json".format(key,id)), "w") as fp:
        json.dump(prmse, fp)
    # mat = dict()
    # h,w = background.shape
    # mat["compress_bg"] = background[h//2-192:h//2+192, w//2-192:w//2+192]
    # mat["prmse"] = prmse
    # sciio.savemat(os.path.join(path,"background_{}_{:05d}.mat".format(key,id)), mat)

if __name__ == "__main__":
    import multiprocessing as mp
    import argparse 
    parser = argparse.ArgumentParser()
    # basic experiment setting
    parser.add_argument('--otf_file', default="../../../data/PSF0620_04_4_40.npy", type=str, help='')
    parser.add_argument('--save_path', default=None, type=str, help='')
    parser.add_argument('--jobs', default=20, type=int,help='')
    args = parser.parse_args()

    # pool = mp.Pool(processes=args.jobs)
    otf_file = args.otf_file
    save_path = args.save_path

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    otf_list = np.load(otf_file)  
    [c,h,w] = otf_list.shape
    
    expand_h = 1024
    expand_w = 1024       
    pad_h_1 = (expand_h - h)//2
    pad_h_2 = expand_h - h - pad_h_1
    pad_w_1 = (expand_w - w)//2
    pad_w_2 = expand_w - w - pad_w_1
    otf_3d = np.pad(otf_list, ((0,0),(pad_h_1, pad_h_2),(pad_w_1,pad_w_2)),mode='constant', constant_values=0)
    otf_fft = np.fft.fft2(np.fft.ifftshift(otf_3d,(1,2)))

    bg_dict = { -4: '../../../data/background/230822_1532.npy', 
                -5: '../../../data/background/230818_1001.npy', 
                -6: '../../../data/background/230821_0915.npy', 
                -7: '../../../data/background/230817_1647.npy', 
                -8: '../../../data/background/230822_1831.npy', 
                -9: '../../../data/background/230822_1149.npy', 
                -10: '../../../data/background/230821_1615.npy', 
                -11: '../../../data/background/230822_0935.npy', 
                -12: '../../../data/background/230818_1503.npy', 
                -13: '../../../data/background/230821_1408.npy', 
                -14: '../../../data/background/230816_1422.npy', 
                -15: '../../../data/background/230817_1418.npy', 
                -16: '../../../data/background/230821_1432.npy', 
                -17: '../../../data/background/230817_1201.npy', 
                -18: '../../../data/background/230822_1039.npy', 
                -19: '../../../data/background/230822_1403.npy', 
                -20: '../../../data/background/230822_1714.npy', 
                -21: '../../../data/background/230816_1614.npy', 
                -22: '../../../data/background/230817_1020.npy', 
                -23: '../../../data/background/230821_1136.npy', 
                -24: '../../../data/background/230821_1025.npy', 
                -25: '../../../data/background/230818_1356.npy',
                -26: '../../../data/background/230818_1112.npy'
            }
    bg_dict_val = { -2: "../../../data/background/sunny_sky_backgrouod.npy",  
                -3: "../../../data/background/cloudy_sky_backgrouod.npy",
                #-2: "../../../data/background/one_sky_backgrouod.npy"
            }
    
    temp = dict()
    if "val" in save_path:
        temp = bg_dict_val
    else:
        temp = bg_dict

    for key, val in temp.items():
        sky_bgs = np.load(val)
        num, c, h, w = sky_bgs.shape
        
        pool = mp.Pool(processes=args.jobs)
        for i in range(num):
            #save_background(i,key,otf_fft,save_path, sky_bgs[i,:,:,:])
            pool.apply_async(save_background, (i,key,otf_fft,save_path, sky_bgs[i,:,:,:]))
        print("---{}----".format(key))
        pool.close()
        pool.join()
        print("----over----")
