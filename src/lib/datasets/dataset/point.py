from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import pycocotools.coco as coco
import numpy as np
import torch
import json
import os

import torch.utils.data as data

class PointOTF(data.Dataset):
  mean = np.array([0.485, 0.456, 0.406],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.229, 0.224, 0.225],
                   dtype=np.float32).reshape(1, 1, 3)
  default_resolution = [384, 384]
  num_classes = 4
  split = "train"
  def __init__(self, opt, split):
    super(PointOTF, self).__init__()
    self.split = split
    self.opt = opt
  def __len__(self):
    return 128
   