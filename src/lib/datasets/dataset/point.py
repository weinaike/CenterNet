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
  mean = np.array([0., 0., 0.],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([1.0, 1.0, 1.0],
                   dtype=np.float32).reshape(1, 1, 3)
  default_resolution = [384, 384]
  num_classes = 4
  def __init__(self, opt, split):
    super(PointOTF, self).__init__()
    self.split = split
    self.opt = opt
    if self.split == "train":
      self.num_samples = 2048
    else:
      self.num_samples = 256
  def __len__(self):   
    return self.num_samples