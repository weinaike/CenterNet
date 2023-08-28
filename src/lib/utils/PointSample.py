#coding=utf-8

import os
import random
import numpy as np
import json
import math
import time
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

debug = False
def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def closest_power_of_two(n):
    power = math.floor(math.log2(n)) + 1
    result = 2 ** power
    return result


def get_true_background(rect_len: int):
    
    sky_bg = None
    bg_label = None

    bg_dict = { -2: "../../../data/background/sunny_sky_backgrouod.npy",  
                -3: "../../../data/background/cloudy_sky_backgrouod.npy",
                -4: '../../../data/background/230822_1532.npy', 
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
    if debug:
        bg_dict = {-2: "../../../data/background/one_sky_backgrouod.npy"}
        # bg_dict = {-3: "../../../data/background/cloudy_sky_backgrouod.npy"}

    bg_label = random.choice(list(bg_dict.keys()))
    try:
        sky_bg = np.load(bg_dict[bg_label])
    except:
        items = bg_dict[bg_label].split("/")
        sky_bg = np.load(os.path.join(".","../",items[-3], items[-2],items[-1]))

    # n:样本数， c:光谱通道数， h: 高， w:宽
    n, c, h, w = sky_bg.shape
    n_idx = random.choice(range(n))
    step = 1
    rect_len = rect_len * step
    h_s = random.choice(range(h - rect_len))
    h_e = h_s + rect_len
    w_s = random.choice(range(w - rect_len))
    w_e = w_s + rect_len
    
    rect_patchs = sky_bg[n_idx, :, h_s:h_e:step, w_s:w_e:step]

    if debug:
        np.save("debug/background.npy",rect_patchs)

    return rect_patchs, bg_label



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
        weight[label] = 1.0  # 权重比例可调
        weight[label+1] = 1.0

    elif weight_mode == "twohot_far":
        if label >= wave_count - 5:
            assert(0)
        weight[label] = 0.5
        weight[label+5] = 0.5
    elif weight_mode == "all_one":
        weight =  np.random.uniform(low = 0., high = 1.0 , size = (wave_count) ) * 0.1 + 0.9
        weight = weight/np.max(weight)
        label = -1
        return weight, label
    elif weight_mode == "gauss":
        x = np.linspace(0, wave_count - 1, wave_count)
        weight = np.exp(-1 * (x - label)**2 / ((wave_count/2)**2 ))
        # logging.debug("label:{}, weight:{}\n".format(label, weight))
    elif weight_mode == "rand":
        while True:
            weight =  np.random.uniform(low = 0., high = 1.0 , size = (wave_count) )  
            label = np.argmax(weight)
            if label in labels:
                break   
        weight = weight/np.max(weight)
        label = int(np.argmax(weight))
        return weight, label
    else:
        assert(0)

    if have_noise :
        noise = np.random.uniform(low = 0., high = 0.3 , size = (wave_count) )  
        weight = weight + noise
        weight = weight/np.max(weight)
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
            point_patch[point_len//2,point_len//2] = 1
            point_patchs[i,:,:] = point_patch
        point_patchs = point_patchs * random.uniform(0.8, 1.0)
    elif point_type == "ones_rand":
        point_patchs = np.random.rand(c,point_len,point_len) * 0.1 + 0.9
        point_patchs[:,point_len//2,point_len//2] = np.ones([c])
    elif point_type == "gauss":
        point_patchs = np.repeat(gaussian2D((point_len,point_len),1).reshape(1,point_len,point_len),c,axis=0)
    else:
        assert(0)
    point_patchs = point_patchs* weight.reshape(c,1,1)
    # print(point_patchs)
    return point_patchs, label, list(weight)


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
    assert(pw < (w//2))
    assert(ph < (h//2))
    
    centerx = w//4 + w//2#int(w*random.uniform(0,1))
    centery = h//4 + h//2#int(h*random.uniform(0,1))
    # centerx = w//2 + int(w*random.uniform(0,0.5))
    # centery = h//2 + int(h*random.uniform(0,0.5))

    point_source_plan = np.zeros((pc,h+h//2, w+w//2))
    point_source_plan[:,(centery-ph//2):(centery+ph//2+1), (centerx-pw//2):(centerx+pw//2+1)] = point_patch
    # print("point_source_plan",point_source_plan.shape)
    # print((centery-ph//2),(centery+ph//2+1), (centerx-pw//2),(centerx+pw//2+1))
    # print(centery, centerx)

    otf_3d = np.pad(otf_list, ((0,0),(h//4,h//4),(w//4,w//4)),mode='constant', constant_values=0)
    # print("otf_3d",otf_3d.shape)
    point_image_amp = np.zeros(otf_3d.shape)

    point_source_fft = np.fft.fft2(np.fft.ifftshift(point_source_plan,(1,2)))     # fft2 ,对于 cpu 数据，ubuntu平台计算结果有问题，cuda计算正常（但是数据搬移耗时长，意义不大），
    otf_fft = np.fft.fft2(np.fft.ifftshift(otf_3d,(1,2)))

    fft_dot = np.multiply(point_source_fft, otf_fft)
    
    point_image = np.fft.ifft2(fft_dot)
    point_image = np.fft.fftshift(point_image,(1,2))
    point_image_amp = np.abs(point_image)

    Intensity = np.sum(point_image_amp,0)

    # print(Intensity)
    return Intensity, [centery,centerx]

def gen_point_psf_new(point_patch, otf_list): 
    assert(otf_list.shape[0] == point_patch.shape[0])
    [c,h,w] = otf_list.shape
    [pc,ph,pw] = point_patch.shape
    
    expand_h = closest_power_of_two(h + ph)
    expand_w = closest_power_of_two(w + pw)
    
    
    centerx = expand_w // 2
    centery = expand_h // 2


    # padding 
    pad_ph_1 = (expand_h - ph)//2
    pad_ph_2 = expand_h - ph - pad_ph_1

    pad_pw_1 = (expand_w - pw)//2
    pad_pw_2 = expand_w - pw - pad_pw_1
    point_source_plan = np.pad(point_patch, ((0,0),(pad_ph_1,pad_ph_2),(pad_pw_1,pad_pw_2)),mode='constant', constant_values=0)


    pad_h_1 = (expand_h - h)//2
    pad_h_2 = expand_h - h - pad_h_1

    pad_w_1 = (expand_w - w)//2
    pad_w_2 = expand_w - w - pad_w_1

    otf_3d = np.pad(otf_list, ((0,0),(pad_h_1, pad_h_2),(pad_w_1,pad_w_2)),mode='constant', constant_values=0)

    point_image_amp = np.zeros(otf_3d.shape)

    point_source_fft = np.fft.fft2(np.fft.ifftshift(point_source_plan,(1,2)))     # fft2 ,对于 cpu 数据，ubuntu平台计算结果有问题，cuda计算正常（但是数据搬移耗时长，意义不大），
    otf_fft = np.fft.fft2(np.fft.ifftshift(otf_3d,(1,2)))

    fft_dot = np.multiply(point_source_fft, otf_fft)
    
    point_image = np.fft.ifft2(fft_dot)
    point_image = np.fft.fftshift(point_image,(1,2))

    # crop
    crop_point_image = point_image[:, (centery-ph//2):(centery+ph//2), (centerx-pw//2):(centerx+pw//2)]


    point_image_amp = np.abs(crop_point_image)

    Intensity = np.sum(point_image_amp,0)
    [Ic,Ih,Iw] = point_patch.shape

    return Intensity, [Ih//2,Iw//2]


def gen_centers(rect_len:int, edge:int, obj_num:int):
    pts = list()
    if obj_num == 2:
        radius = random.randint(0, rect_len - edge*2) // 2
        theta = random.uniform(0, 2*3.14)

        x_len = int(math.cos(theta) * radius)
        y_len = int(math.sin(theta) * radius)

        resdiualx = (rect_len - abs(x_len) * 2 - edge * 2) // 2
        resdiualy = (rect_len - abs(y_len) * 2 - edge * 2 ) // 2

        cx = random.randint(rect_len//2 - resdiualx, rect_len//2 + resdiualx)
        cy = random.randint(rect_len//2 - resdiualy, rect_len//2 + resdiualy)

        p1x = cx + x_len
        p1y = cy + y_len
        p2x = cx - x_len
        p2y = cy - y_len

        pts.append([p1x,p1y])
        pts.append([p2x,p2y])
    else:
        for i in range(obj_num):
            startx = random.randint(edge, rect_len - edge) 
            starty = random.randint(edge, rect_len - edge) 
            pts.append([startx, starty])
    return pts

def calc_prmse(rect_patchs):
    [c, h, w] = rect_patchs.shape
    tmp = 0
    for i in range(c):
        tmp += np.mean(np.power(rect_patchs[i,:,:],2))
    return np.sqrt(tmp/c)



def gen_merge_sample(otf_list, labels, obj_len, point_type="ones", weight_mode = "gauss", have_noise = True , psnr = 20, true_bg = False):
    [wave_count,h,w ] = otf_list.shape
    #点目标， 其宽度obj_len, 要求奇数
    obj_num = 1
    if random.random() > 0.2:
        obj_num = 2
    
    #方块目标， 其宽度rect_len, 要求奇数
    rect_len = 384
    if true_bg:
        rect_patchs, bg_label = get_true_background(rect_len)        
    else:
        rect_patchs, bg_label, _ = gen_point_patchs(wave_count,[0],"ones_rand", weight_mode="all_one", have_noise=have_noise, point_len=rect_len)
    

    if bg_label < 0:
        prmse = calc_prmse(rect_patchs)
        # print("prmse:",prmse)
        # print("max:",np.max(rect_patchs))
        # print("norm:",np.sqrt(np.linalg.norm(rect_patchs.ravel())))
        rect_patchs = rect_patchs / prmse * 10**(-psnr/20)
    target = list()
   

    objs = list()
    for i in range(obj_num):
        objs.append(gen_point_patchs(wave_count, labels, point_type, weight_mode, have_noise, point_len=obj_len) )
    if debug:
        for i in range(len(objs)):
            np.save("debug/point_{}.npy".format(i), objs[i][0])

    pts = list()

    if rect_len > h:
        edge = h // 2
    else:
        edge = obj_len
    pts = gen_centers(rect_len, edge, obj_num)

    for i in range(obj_num):
        point_patchs = objs[i][0]
        label = objs[i][1]
        weight = objs[i][2]
        # 点目标和方块目标融合
        startx = pts[i][0]
        starty = pts[i][1]
        endx = startx + obj_len
        endy = starty + obj_len
        rect_patchs[:,starty:endy, startx:endx] = point_patchs
        
        point_cx = (startx + endx) // 2
        point_cy = (starty + endy) // 2

        # 标签, 中心坐标x, 中心坐标y, 宽， 高， 标签权重， 背景标签， 信噪比， 
        target.append([label, point_cx, point_cy, obj_len, obj_len] + weight + [bg_label] + [psnr])
    
    if debug:
        np.save("debug/rect_patchs.npy",rect_patchs)
    #获取样本sample，及加载在其上的块目标的中心位置，rect_center[h,w]
    sample, rect_center = gen_point_psf_new(rect_patchs, otf_list)


    for i in range(len(target)):
        target[i][1] += rect_center[1] - rect_len//2
        target[i][2] += rect_center[0] - rect_len//2
        target[i].append(rect_center[1])
        target[i].append(rect_center[0])

    sample = (sample-np.min(sample))/(np.max(sample)-np.min(sample)) 
    [h,w] = sample.shape

    sample = sample.reshape(1,h,w) 

    # print(target)
    return sample, target

def save_merge_point(id, otf_file, labels = [1,3,5], point_len = 5 , point_type = "ones", weight_mode = "gauss", psnr = 20, true_bg = False, save_path = None):
    
    path = otf_file[:-4]
    if save_path is not None :
        path = save_path

    otf_list = np.load(otf_file)  
    sample, target = gen_merge_sample(otf_list,labels,point_len,point_type, weight_mode,True, psnr, true_bg)

    if not os.path.exists(path):
        os.mkdir(path)
    # print(path, target)
    np.save(os.path.join(path,"sample_{:05d}.npy".format(id)),sample)
    with open(os.path.join(path,"sample_{:05d}.json".format(id)), "w") as fp:
        json.dump(target, fp)


if __name__ == "__main__":
    import multiprocessing as mp
    import argparse 
    parser = argparse.ArgumentParser()
    # basic experiment setting
    parser.add_argument('--otf_file', default="../../../data/PSF0620_04_4_40.npy", type=str, help='')
    parser.add_argument('--save_path', default=None, type=str, help='')
    parser.add_argument('--point_len', default=5, type=int,help='')
    parser.add_argument('--point_type', default="ones", type=str, help='')
    parser.add_argument('--weight_mode', default="gauss", type=str, help='')
    parser.add_argument('--noise_sigma', default=0.01, type=float,help='')
    parser.add_argument('--num', default=10000, type=int,help='')
    parser.add_argument('--labels', nargs='+',default=[1,3,5], type=int, help='a list of integers')
    parser.add_argument('--merge_bg', action='store_true', help='point object merge with backgroud')
    parser.add_argument('--true_bg', action='store_true', help='point object merge true backgroud')
    parser.add_argument('--psnr', default=20, type=float,help='')
    parser.add_argument('--jobs', default=20, type=int,help='')
    args = parser.parse_args()

    pool = mp.Pool(processes=args.jobs)
    otf_file = args.otf_file
    labels = args.labels
    point_len = args.point_len
    point_type = args.point_type
    noise_sigma = args.noise_sigma
    merge_bg = args.merge_bg
    true_bg = args.true_bg
    weight_mode = args.weight_mode
    psnr = args.psnr
    save_path = args.save_path

    num  = args.num

    for i in range(num):
        # save_merge_point(i,otf_file,labels, point_len, point_type, weight_mode,  psnr, true_bg, save_path)
        pool.apply_async(save_merge_point, (i,otf_file,labels, point_len, point_type, weight_mode, psnr, true_bg, save_path))
    print("----start----")
    pool.close()
    pool.join()
    print("----over----")
    if False:
        path = otf_file[:-4]
        samples = list()
        targets = list()    
        for i in range(int(num * 0.9)):
            sample = np.load(os.path.join(path,"sample_{:05d}.npy".format(i)))
            with open(os.path.join(path,"sample_{:05d}.json".format(i)), "r") as fp:
                target = json.load(fp)
            samples.append(sample)
            targets.append(target)

        np.save("{}_{}_train.npy".format(otf_file,"sample"),np.array(samples))
        with open("{}_{}_train.json".format(otf_file,"sample"), "w") as fp:
            json.dump(targets, fp)

        samples = list()
        targets = list()    
        for i in range(int(num * 0.9)+1,num):
            sample = np.load(os.path.join(path,"sample_{:05d}.npy".format(i)))
            with open(os.path.join(path,"sample_{:05d}.json".format(i)), "r") as fp:
                target = json.load(fp)
            samples.append(sample)
            targets.append(target)

        np.save("{}_{}_val.npy".format(otf_file,"sample"),np.array(samples))
        with open("{}_{}_val.json".format(otf_file,"sample"), "w") as fp:
            json.dump(targets, fp)
