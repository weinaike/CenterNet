from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt


from opts import opts
from lib.utils.image import draw_umich_gaussian, draw_msra_gaussian
from utils.utils import AverageMeter
from datasets.point import PointOTF
from detectors.ctdet import CtdetDetector

def gen_color(cls):
    cmap = {0:(139, 0 , 255),
            1:(0, 0, 255), 
            2:(0, 127, 255), 
            3:(0, 255, 0), 
            4:(255, 255, 0), 
            5:(255, 0, 0)}
    return cmap[cls]

def add_circle(img, anns, radius = 3): 
    mask = img.copy()
    for ann in anns:
        px = int(ann[1])
        py = int(ann[2])
        cls = int(ann[0])
        conf = min(float(ann[5]),1.0)
        c = gen_color(cls)
        name = "C"
        # if cls == 5:
        #     name = "C3:{:.2f}".format(conf)
        # if cls == 3:
        #     name = "C2:{:.2f}".format(conf)
        # if cls == 1:
        #     name = "C1:{:.2f}".format(conf)
        cv2.circle(mask,(px,py), radius, c, -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(mask, name+str(cls), (px + 5 , py - 5), font, 0.5 , c,  thickness=1, lineType=cv2.LINE_AA)
    return mask

def add_rect(img, anns, gt = True):
    mask = img.copy()
    for ann in anns:
        px = int(ann[1])
        py = int(ann[2])
        w = int(ann[3])
        h = int(ann[4])
        cls = int(ann[0])
        conf = min(float(ann[5]),1.0)
        c = gen_color(cls)
        name = "C"
        wl = w//2
        wr = w - wl
        hl = h//2
        hr = h - hl
        

        pt1 = (px - wl , py - hl)
        pt2 = (px + wr, py + hr)
        if gt:
            cv2.rectangle(mask, pt1, pt2, c, 1 )
        else:
            cv2.rectangle(mask, pt1, pt2, (0,0,0), 1)
            cv2.imwrite("C_{}.bmp".format(cls+1), cv2.cvtColor(mask[py - 10: py + 10, px - 10 : px + 10,:], cv2.COLOR_BGR2RGB) )
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(mask, name+str(cls+1) + ":{:.2f}".format(conf), (px + 5 , py - 5), font, 1 , (0,0,0),  thickness=2, lineType=cv2.LINE_AA)
    return mask


def gen_background(heatmap, anns, width = 111):
    h,w=heatmap.shape
    mask = np.zeros((h,w,3))
    rectx = anns[0][-2]
    recty = anns[0][-1]

    x1 = rectx - width//2
    x2 = rectx + width//2
    y1 = recty - width//2
    y2 = recty + width//2
    mask[y1:y2, x1:x2, :] = 1
    return np.ascontiguousarray(mask * 255 , dtype=np.uint8)


def add_gt_mask(heatmap, anns, radius=3, factor= 1.0): 
    h,w=heatmap.shape
    mask = np.zeros((h,w,3))
    mask = np.ascontiguousarray(mask, dtype=np.uint8)
    for ann in anns:
        px = int(ann[1])
        py = int(ann[2])
        cls = int(ann[0])
        color = gen_color(cls)
        cv2.circle(mask, (px,py), radius, color,-1)
        # if cls == 5:
        #     mask[:,:,0] = draw_msra_gaussian(mask[:,:,0],(px,py), radius)
        # if cls == 3:
        #     mask[:,:,1] = draw_msra_gaussian(mask[:,:,1],(px,py), radius)
        # if cls == 1:
        #     mask[:,:,2] = draw_msra_gaussian(mask[:,:,2],(px,py), radius)
        # name = ""
        # for it in ann[5:11]:
        #     name += "_{:.1f}".format(it) 
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(mask, name, (px + 5 , py - 5), font, 0.5 , (255,255,255),  thickness=1, lineType=cv2.LINE_AA)
    return mask


def show(opt):
    print("----------------test-------------")
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    # Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, PointOTF)
    print(opt)
    # Logger(opt)
    # Detector = detector_factory[opt.task]  
    detector = CtdetDetector(opt)    
    
    idx = 1
    for idx in range(idx,3):
        print("\n-------------{}-------------".format(idx))
        img = np.load("sample_{:05d}.npy".format(idx))
        with open("sample_{:05d}.json".format(idx), "r") as fp:
            gts = json.load(fp)

        ret = detector.run(img)
        print("------------gt-----------")
        for gt in gts :
            print(gt)
        print("---------predict---------")
        predicts = list()
        for cls_id, dets in ret['results'].items():
            for det in dets:
                if det[4] > opt.vis_thresh:
                    label = opt.labels[cls_id-1]
                    xc = int((det[0]+det[2])/2)
                    yc = int((det[1]+det[3])/2)
                    width = int(det[2] - det[0])
                    height = int(det[3] - det[1])
                    conf = det[4]
                    
                    print(label, xc, yc, width, height, conf)
                    predicts.append([label, xc, yc, width, height, conf])
        
        c,h,w=img.shape        
        # Heatmap
        heatmap = np.zeros((h,w))
        backgroud = gen_background(heatmap, gts,384)

        points_mask = add_gt_mask(heatmap, gts, radius=3)

        # image
        img = np.concatenate((img,img,img), axis=0)
        img = img.transpose(1,2,0)*255
        img = np.ascontiguousarray(img, dtype=np.uint8)

        # object 
        obj = cv2.addWeighted(backgroud, 0.1, points_mask,0.9, 0)

        # image + heatmap
        # 
        result = add_rect(img, gts, True)
        result = add_rect(result, predicts, False)
        # result = add_circle(result, predicts)

        # plot
        plt.rcParams['figure.figsize'] = (12.0, 5) 
        fig, ax = plt.subplots(1,3)    
        ax[0].imshow(obj)
        ax[0].axis("off")
        ax[0].set_title("Point Object With Background")
        ax[1].imshow(img)
        ax[1].axis("off")
        ax[1].set_title("Compressive Image")
        ax[2].imshow(result)
        ax[2].axis("off")
        ax[2].set_title("Predict Result")        
        plt.savefig("../exp/ctdet/infrared_point_res34_384/debug/predict_{}.png".format(idx))
        # plt.show()
        plt.close()
        cv2.imwrite("debug/result_{}.bmp".format(idx), cv2.cvtColor(result, cv2.COLOR_BGR2RGB) )

if __name__ == '__main__':
    opt = opts().parse()
    show(opt)
