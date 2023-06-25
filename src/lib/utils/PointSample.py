#coding=utf-8

import os
import torch
import torch.utils.data
import random
import numpy as np
import json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def gen_weight_and_label(wave_count, labels, weight_mode="onehot", have_noise = True):
    # labels ：取值（0-9） 点的标签类别，len(labels): 二分类，五分类，十分类
    # wave_count: 点光源包含的波长数量， 和PSF0406_02_256by256.npy文件中的波长对应。
    # weight_mode : 点的叠加分布，onehot型(onehot)，twohot型(twohot)，高斯型(gauss)，指数型(exp)
    label = random.choice(labels)
    weight =  np.zeros(wave_count)

    if weight_mode == "onehot":
        weight[label] = 1.0
    elif weight_mode == "twohot":
        if label >= wave_count - 1:
            assert(0)
        weight[label] = 0.5  # 权重比例可调
        weight[label+1] = 0.5

    elif weight_mode == "twohot_far":
        if label >= wave_count - 5:
            assert(0)
        weight[label] = 0.5
        weight[label+5] = 0.5

    elif weight_mode == "gauss":
        x = np.linspace(0, wave_count - 1, wave_count)
        weight = np.exp(-1 * (x - label)**2 / ((wave_count/2)**2 ))
        # logging.debug("label:{}, weight:{}\n".format(label, weight))
    elif weight_mode == "exp":
        assert(0)
    else:
        assert(0)

    if have_noise:
        noise = np.random.uniform(low = 0., high = 0.3 , size = (wave_count) )  
        weight = weight + noise
        # print(weight)
    return weight, label

def gen_point_patchs(wave_count, labels, point_type="ones", weight_mode="onehot", have_noise = True, point_len = 3):
    # labels ：取值（0-9） 点的标签类别，len(labels): 二分类，五分类，十分类
    # wave_count: 点光源包含的波长数量， 和PSF0406_02_256by256.npy文件中的波长对应。
    # point_type: 点的形状分布，全1型(ones)，高斯型(gauss)，随机型(rand)
    # weight_mode : 点的叠加分布，onehot型(onehot)，twohot型(twohot)，高斯型(gauss)，指数型(exp)

    weight, label = gen_weight_and_label(wave_count, labels, weight_mode, have_noise)
    [c] = weight.shape
    point_patchs = np.zeros((c,point_len,point_len))
    if point_type == "ones":
        point_patchs = np.ones((c,point_len,point_len))
    elif point_type == "rand":
        for i in range(c):
            point_patch = np.random.rand(point_len,point_len)
            point_patch[1,1] = 1
            point_patchs[i,:,:] = point_patch
    elif point_type == "ones_rand":
        point_patchs = np.random.rand(c,point_len,point_len) * 0.2 + 0.8
    else:
        assert(0)
    point_patchs = point_patchs* weight.reshape(c,1,1)
    # print(point_patchs)
    return point_patchs, label


# 点光源输入，与波长叠加，输出像强度图样
# point_patch:[10,3,3],不同波长的点光源
# otf_list：[10,256,256],不同波长的光学传递函数
# 输出
# smaple:输出样本，[h,w]:[384,384]
# [centery,centerx]: 点目标位置[h,w]
def gen_point_psf(point_patch, otf_list): 
    assert(otf_list.shape[0] == point_patch.shape[0])
    [c,h,w] = otf_list.shape
    [pc,ph,pw] = point_patch.shape
    
    # centerx = w//4 + w//2#int(w*random.uniform(0,1))
    # centery = h//4 + h//2#int(h*random.uniform(0,1))
    centerx = w//2 + int(w*random.uniform(0,0.5))
    centery = h//2 + int(h*random.uniform(0,0.5))

    point_source_plan = np.zeros((pc,h+h//2, w+w//2))
    point_source_plan[:,(centery-ph//2-1):(centery+ph//2), (centerx-pw//2-1):(centerx+pw//2)] = point_patch

    otf_3d = np.pad(otf_list, ((0,0),(h//4,h//4),(w//4,w//4)),mode='constant', constant_values=0)
    point_image_amp = np.zeros(otf_3d.shape)
    if False:
        with torch.no_grad:
            point_source_plan = torch.tensor(point_source_plan)
            otf_3d = torch.tensor(otf_3d)
            torch.fft.fft2(point_source_plan[0,:,:])
            for i in list(range(c)):
                point_source_fft = torch.fft.fft2(point_source_plan[i,:,:])
                # print(point_source_plan[0,:,:])
                # plt.imshow(torch.abs(point_source_fft))
                # plt.show()
                otf_fft = torch.fft.fft2(otf_3d[i,:,:])
                
                fft_dot = torch.mul(point_source_fft, otf_fft)
                
                point_image = torch.fft.ifft2(fft_dot)
                point_image = torch.fft.fftshift(point_image)
                
                point_image_amp[i,:,:] = torch.abs(point_image).numpy()
    else:
        point_source_fft = np.fft.fft2(point_source_plan)     # fft2 ,对于 cpu 数据，ubuntu平台计算结果有问题，cuda计算正常（但是数据搬移耗时长，意义不大），
        otf_fft = np.fft.fft2(otf_3d)

        fft_dot = np.multiply(point_source_fft, otf_fft)
        
        point_image = np.fft.ifft2(fft_dot)
        point_image = np.fft.fftshift(point_image,(1,2))
        point_image_amp = np.abs(point_image)

    Intensity = np.sum(point_image_amp,0)

    # print(Intensity)
    return Intensity, [centery,centerx]

def gen_sample(otf_list, labels, point_type="ones", weight_mode = "gauss", have_noise = True ):
    point_len = 3
    [wave_count,h,w ] = otf_list.shape
    point_patchs, label = gen_point_patchs(wave_count, labels, 
                                           point_type=point_type, weight_mode=weight_mode, 
                                           have_noise=have_noise, point_len=point_len) 
    sample, [ch,cw] = gen_point_psf(point_patchs, otf_list) 
      
    sample = (sample-np.min(sample))/(np.max(sample)-np.min(sample)) 
    [h,w] = sample.shape
    if have_noise:
        sigm = 0.4
        sample = sample + sigm * np.random.rand(h,w)
        sample = (sample-np.min(sample))/(np.max(sample)-np.min(sample)) 

    sample = sample.reshape(1,h,w) 
    # print(sample)
    return sample, [label, cw, ch, point_len, point_len]


def gen_multi_point_sample(otf_list, labels, obj_width, point_type="ones", weight_mode = "gauss", have_noise = True ):

    point_len = 3
    [wave_count,h,w ] = otf_list.shape

    obj_num = random.randint(1,3)
    target = list()
    sample = None
    for i in range(obj_num):
        point_patchs, label = gen_point_patchs(wave_count, labels, 
                                            point_type=point_type, weight_mode=weight_mode, 
                                            have_noise=have_noise, point_len=point_len) 
        temp, [ch,cw] = gen_point_psf(point_patchs, otf_list) 
        target.append([label, cw, ch, obj_width, obj_width])
        if sample is None:
            sample = temp
        else:
            sample += temp
      
    sample = (sample-np.min(sample))/(np.max(sample)-np.min(sample)) 
    [h,w] = sample.shape
    if have_noise:
        sigm = 0.4
        sample = sample + sigm * np.random.rand(h,w)
        sample = (sample-np.min(sample))/(np.max(sample)-np.min(sample)) 

    sample = sample.reshape(1,h,w) 
    # print(sample)
    return sample, target



def gen_merge_sample(otf_list, labels, obj_width, point_type="ones", weight_mode = "gauss", have_noise = True , noise_sig = 0.1):
    [wave_count,h,w ] = otf_list.shape
    #点目标， 其宽度obj_len, 要求奇数
    obj_num = random.randint(1,3)
    #方块目标， 其宽度rect_len, 要求奇数
    rect_len = 99
    rect_patchs, _ = gen_point_patchs(wave_count,[0],point_type=point_type, weight_mode="gauss", have_noise=have_noise, point_len=rect_len)
    target = list()
    for i in range(obj_num):
        obj_len = 5
        point_patchs, label = gen_point_patchs(wave_count, labels, point_type, weight_mode, have_noise, point_len=obj_len) 
        
        # 点目标和方块目标融合
        startx = random.randint(obj_len, rect_len - obj_len * 2) 
        starty = random.randint(obj_len, rect_len - obj_len * 2) 
        endx = startx + obj_len
        endy = starty + obj_len
        rect_patchs[:,starty:endy, startx:endx] = point_patchs
        
        point_cx = (startx + endx) // 2
        point_cy = (starty + endy) // 2

        last_h = point_cy - rect_len//2
        last_w = point_cx - rect_len//2
        # last_h = rect_center[0] + point_cy - rect_len//2
        # last_w = rect_center[1] + point_cx - rect_len//2

        target.append([label, last_w, last_h, obj_width, obj_width ])
    # np.save("rect_patchs.npy",rect_patchs)
    #获取样本sample，及加载在其上的块目标的中心位置，rect_center[h,w]
    sample, rect_center = gen_point_psf(rect_patchs, otf_list)
    for i in range(len(target)):
        target[i][1] += rect_center[1] 
        target[i][2] += rect_center[0] 

    sample = (sample-np.min(sample))/(np.max(sample)-np.min(sample)) 
    [h,w] = sample.shape
    if False:#have_noise:
        sigm = noise_sig * random.uniform(0,1)
        # sample = np.multiply(sample, 1 + noise_sig * (np.random.rand(h,w) - 0.5)) 
        sample = sample + sigm * np.random.rand(h,w)
        # sample = (sample-np.min(sample))/(np.max(sample)-np.min(sample)) 

    sample = sample.reshape(1,h,w) 

    # print("test")
    return sample, target

def save_merge_point(id, otf_file):
    
    path = otf_file[:-4]
    labels = [1,3,5]

    otf_list = np.load(otf_file)
    
    point_len = 111 
    point_type = "ones"
    noise_sigma = 0.01

    sample, target = gen_merge_sample(otf_list,labels,point_len,point_type,"gauss",False,noise_sigma)
    # sample, target = gen_multi_point_sample(otf_list, labels, point_len, point_type, "gauss", False)
    if not os.path.exists(path):
        os.mkdir(path)
    # print(path)
    np.save(os.path.join(path,"sample_{:05d}.npy".format(id)),sample)
    with open(os.path.join(path,"sample_{:05d}.json".format(id)), "w") as fp:
        json.dump(target, fp)


if __name__ == "__main__":
    import multiprocessing as mp


    pool = mp.Pool(processes=20)
    otf_file = "../../../data/PSF0620_04_4_40.npy"
    num  = 10000
    for i in range(num):
        pool.apply_async(save_merge_point, (i,otf_file,))
        # gen_merge_sample(otf_list, labels, point_len, point_type, weight_mode = "gauss", have_noise = True , noise_sig = noise_sigma)
    print("----start----")
    pool.close()
    pool.join()
    print("----over----")
    # path = otf_file[:-4]
    # samples = list()
    # targets = list()    
    # for i in range(int(num * 0.9)):
    #     sample = np.load(os.path.join(path,"sample_{:05d}.npy".format(i)))
    #     with open(os.path.join(path,"sample_{:05d}.json".format(i)), "r") as fp:
    #         target = json.load(fp)
    #     samples.append(sample)
    #     targets.append(target)

    # np.save("{}_{}_train.npy".format(otf_file,"sample"),np.array(samples))
    # with open("{}_{}_train.json".format(otf_file,"sample"), "w") as fp:
    #     json.dump(targets, fp)

    # samples = list()
    # targets = list()    
    # for i in range(int(num * 0.9)+1,num):
    #     sample = np.load(os.path.join(path,"sample_{:05d}.npy".format(i)))
    #     with open(os.path.join(path,"sample_{:05d}.json".format(i)), "r") as fp:
    #         target = json.load(fp)
    #     samples.append(sample)
    #     targets.append(target)

    # np.save("{}_{}_val.npy".format(otf_file,"sample"),np.array(samples))
    # with open("{}_{}_val.json".format(otf_file,"sample"), "w") as fp:
    #     json.dump(targets, fp)