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

def add_circle(img, anns, radius = 3): 
    mask = img.copy()
    for ann in anns:
        px = int(ann[1])
        py = int(ann[2])
        cls = int(ann[0])
        c = (255,0,0)
        if cls == 5:
            c = (255,0,0)
        if cls == 3:
            c = (0,255,0)
        if cls == 1:
            c = (0,0,255)
        cv2.circle(mask,(px,py), radius, c, -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(mask, str(cls), (px + 5 , py - 5), font, 0.5 , c,  thickness=1, lineType=cv2.LINE_AA)
    return mask

def gen_background(heatmap, anns, width = 111):
    h,w=heatmap.shape
    mask = np.zeros((h,w,3))
    rectx = anns[0][5]
    recty = anns[0][6]

    x1 = rectx - width//2
    x2 = rectx + width//2
    y1 = recty - width//2
    y2 = recty + width//2
    mask[y1:y2, x1:x2, :] = 1
    return np.ascontiguousarray(mask * 255 , dtype=np.uint8)


def add_gt_mask(heatmap, anns, radius=3, factor= 1.0): 
    h,w=heatmap.shape
    mask = np.zeros((h,w,3))
    for ann in anns:
        px = int(ann[1])
        py = int(ann[2])
        cls = int(ann[0])

        if cls == 5:
            mask[:,:,0] = draw_msra_gaussian(mask[:,:,0],(px,py), radius)
        if cls == 3:
            mask[:,:,1] = draw_msra_gaussian(mask[:,:,1],(px,py), radius)
        if cls == 1:
            mask[:,:,2] = draw_msra_gaussian(mask[:,:,2],(px,py), radius)
    return np.ascontiguousarray(mask * 255 * factor, dtype=np.uint8)


def show(opt):
    print("----------------test-------------")
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    # Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, PointOTF)
    print(opt)
    # Logger(opt)
    # Detector = detector_factory[opt.task]  
    detector = CtdetDetector(opt)    
    
    idx = 9996
    for idx in range(9995, 10000):
        print("\n-------------{}-------------".format(idx))
        img = np.load("../data/PSF0620_04_4_40/sample_{:05d}.npy".format(idx))
        with open("../data/PSF0620_04_4_40/sample_{:05d}.json".format(idx), "r") as fp:
            gts = json.load(fp)

        ret = detector.run(img)
        print("------------gt-----------")
        for gt in gts :
            print(gt)
        print("---------predict---------")
        predicts = list()
        for cls_id, dets in ret['results'].items():
            for det in dets:
                if det[4] > 0.5:
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
        backgroud = gen_background(heatmap, gts)

        points_mask = add_gt_mask(heatmap, gts, radius=3)

        # image
        img = np.concatenate((img,img,img), axis=0)
        img = img.transpose(1,2,0)*255
        img = np.ascontiguousarray(img, dtype=np.uint8)

        # object 
        obj = cv2.addWeighted(backgroud, 0.5, points_mask, 0.5, 0)

        # image + heatmap
        result = add_circle(img, predicts)

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

if __name__ == '__main__':
    opt = opts().parse()
    show(opt)
