from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import json
import numpy as np

import matplotlib.pyplot as plt


from opts import opts

from utils.utils import AverageMeter
from datasets.dataset_factory import dataset_factory

from detectors.detector_factory import detector_factory


def show(opt):
    print("----------------test-------------")
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)
    # Logger(opt)
    Detector = detector_factory[opt.task]  
    detector = Detector(opt)    
    
    idx = 9999
    img = np.load("../data/PSF0620_04_4_40/sample_{:05d}.npy".format(idx))
    with open("../data/PSF0620_04_4_40/sample_{:05d}.json".format(idx), "r") as fp:
        gts = json.load(fp)
    print("image size : ",img.shape)
    # print(img_info)
    ret = detector.run(img)
    print("------------gt-----------")
    for gt in gts :
        print(gt)
    print("---------predict---------")
    for cls_id, dets in ret['results'].items():
       for det in dets:
           if det[4] > 0.2:
               label = opt.labels[cls_id-1]
               xc = int((det[0]+det[2])/2)
               yc = int((det[1]+det[3])/2)
               width = int(det[2] - det[0])
               height = int(det[3] - det[1])
               conf = det[4]
               
               print(label, xc, yc, width, height, conf)


if __name__ == '__main__':
    opt = opts().parse()
    show(opt)
