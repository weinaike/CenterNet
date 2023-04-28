from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import pycocotools.coco as coco
import numpy as np
import torch
import json
import os

import torch.utils.data as data

from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
from utils.PointSample import gen_multi_point_sample, gen_merge_sample
import math
class PointOTF(data.Dataset):
  mean = np.array([0., 0., 0.],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([1.0, 1.0, 1.0],
                   dtype=np.float32).reshape(1, 1, 3)
  default_resolution = [384, 384]
  




  def __init__(self, opt, split):
    super(PointOTF, self).__init__()
    self.split = split
    self.opt = opt
    if self.split == "train":
      self.num_samples = self.opt.sample_num
    else:
      self.num_samples = 256
    self.num_classes = len(self.opt.labels)


    self.otf_file = self.opt.otf_file
    self.point_type = self.opt.point_type
    self.weight_mode = self.opt.weight_mode  
    self.labels = self.opt.labels
    self.have_noise = self.opt.have_noise

    self.max_objs = self.opt.K
    self.class_name = ['__background__'].extend(self.labels)
    self._valid_ids = np.arange(1, len(self.labels) + 1, dtype=np.int32)
    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}

    self.otf_list = np.load(self.otf_file)
    
    [c,h,w] = self.otf_list.shape
    self.h = h
    self.w = w
    self.wave_count = c

    self.point_len = self.opt.point_len

  def __len__(self):   
    return self.num_samples

  def __getitem__(self, index):
    otf_noise = np.random.rand(self.wave_count, self.h, self.w) * 0.1 + 0.9
    otf_list = np.multiply(self.otf_list, otf_noise)
    img = None
    target = None
    if self.opt.merge_bg:
      img, target = gen_merge_sample(otf_list, self.labels, self.point_len, self.point_type, self.weight_mode, self.have_noise, self.opt.sigma)
    else:
      img, target = gen_multi_point_sample(otf_list, self.labels, self.point_len, self.point_type, self.weight_mode, self.have_noise)
    
    obj_num = len(target)
    
    anns = target
    for i in range(obj_num):        
      anns[i][0] = self.labels.index(target[i][0]) 

    num_objs = min(len(anns), self.max_objs)

    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    if self.opt.keep_res:
      input_h = (height | self.opt.pad) + 1
      input_w = (width | self.opt.pad) + 1
      s = np.array([input_w, input_h], dtype=np.float32)
    else:
      s = max(img.shape[0], img.shape[1]) * 1.0
      input_h, input_w = self.opt.input_h, self.opt.input_w
    
    output_h = input_h // self.opt.down_ratio
    output_w = input_w // self.opt.down_ratio
    num_classes = self.num_classes
    trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)
    dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
    cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)
    
    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian

    gt_det = []
    for k in range(num_objs):
      ann = anns[k]
      bbox = np.array([ann[1] - ann[3]//2, ann[2] - ann[4]/2, ann[1] + ann[3]/2,ann[2] + ann[4]/2], dtype=np.float32) / self.opt.down_ratio
      # print("bbox",bbox)
      cls_id = ann[0]

      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      if h > 0 and w > 0:
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        radius = self.opt.hm_gauss if self.opt.mse_loss else radius
        ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        draw_gaussian(hm[cls_id], ct_int, radius)
        wh[k] = 1. * w, 1. * h
        ind[k] = ct_int[1] * output_w + ct_int[0]
        # print(ct_int[1], output_w , ct_int[0], ind[k], k)
        reg[k] = ct - ct_int
        reg_mask[k] = 1
        cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
        cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
        if self.opt.dense_wh:
          draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
        gt_det.append([ct[0] - w / 2, ct[1] - h / 2, 
                       ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
    
    ret = {'input': img, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}
    if self.opt.dense_wh:
      hm_a = hm.max(axis=0, keepdims=True)
      dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
      ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
      del ret['wh']
    elif self.opt.cat_spec_wh:
      ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
      del ret['wh']
    if self.opt.reg_offset:
      ret.update({'reg': reg})
    if self.opt.debug > 0 or not self.split == 'train':
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
               np.zeros((1, 6), dtype=np.float32)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': 0, 'ann':anns}
      ret['meta'] = meta
    return ret