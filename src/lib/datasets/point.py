from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import pycocotools.coco as coco
import numpy as np
import torch
import json
import os
import glob
import random

import torch.utils.data as data

from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
from utils.PointSample import gen_merge_sample
import math

def get_file_list(file, mode):
  imgs = list()
  targets = list()
  with open(file) as f:
    data = json.load(f)
  
    for key, vals in data.items():
      items = key.split("_")
      use = False
      if mode == "all":
        use = True

      elif "single" == mode:
        if "1" == items[0]:
          use = True
      elif "double" == mode:
        if "2" == items[0]:
          use = True
      elif "single_5x" == mode:
        if "1" == items[0] and "5x" == items[2]:
          use = True
      elif "single_10x" == mode:
        if "1" == items[0] and "10x" == items[2]:
          use = True
      elif "single_50x" == mode:
        if "1" == items[0] and "50x" == items[2]:
          use = True
      elif "double_5x" == mode:
        if "2" == items[0] and "5x" == items[2]:
          use = True
      elif "double_10x" == mode:
        if "2" == items[0] and "10x" == items[2]:
          use = True
      elif "double_50x" == mode:
        if "2" == items[0] and "50x" == items[2]:
          use = True

      elif "double_50x_" in mode:
        dist = mode.split("_")[2]
        if "2" == items[0] and "50x" == items[2] and (dist == items[3] or dist == items[1]):
          use = True

      elif "double_10x_" in mode:
        dist = mode.split("_")[2]
        if "2" == items[0] and "10x" == items[2] and (dist == items[3] or dist == items[1]):
          use = True

      elif "double_5x_" in mode:
        dist = mode.split("_")[2]
        if "2" == items[0] and "5x" == items[2] and (dist == items[3] or dist == items[1]):
          use = True


      elif "single_50x_" in mode:
        dist = mode.split("_")[2]
        if "1" == items[0] and "50x" == items[2] and dist == items[1]:
          use = True

      elif "single_10x_" in mode:
        dist = mode.split("_")[2]
        if "1" == items[0] and "10x" == items[2] and dist == items[1]:
          use = True

      elif "single_5x_" in mode:
        dist = mode.split("_")[2]
        if "1" == items[0] and "5x" == items[2] and dist == items[1]:
          use = True

      if use:
        for val in vals:
          imgs.append(val[:-4] + "npy")
          targets.append(val)


  return imgs, targets

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
    self.force_merge_labels = opt.force_merge_labels

    self.val_num = 1000
    if self.split == "train":
      self.num_samples = self.opt.sample_num
    else:
      self.num_samples = self.val_num
    self.num_classes = len(self.opt.labels)

    self.otf_file = self.opt.otf_file
    self.point_type = self.opt.point_type
    self.weight_mode = self.opt.weight_mode  
    self.labels = self.opt.labels
    self.have_noise = self.opt.have_noise

    ##---------------------强制调整标签----------------
    if self.force_merge_labels:
      self.num_classes = 1
      self.labels = [0]

    self.max_objs = self.opt.K
    self.class_name = ['__background__'].extend(self.labels)
    self._valid_ids = np.arange(1, len(self.labels) + 1, dtype=np.int32)
    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}

    self.point_len = self.opt.point_len
    self.dataset_path = self.opt.dataset_path
    self.data_mode = self.opt.data_mode

    self.imgs = list()
    self.targets = list()



    if False:
      self.otf_list = np.load(self.otf_file)
      [c,h,w] = self.otf_list.shape
      self.h = h
      self.w = w
      self.wave_count = c

      for i in range(self.num_samples):
        otf_noise = np.random.rand(self.wave_count, self.h, self.w) * 0.1 + 0.9
        otf_list = np.multiply(self.otf_list, otf_noise)
        if self.opt.merge_bg:
          img, target = gen_merge_sample(otf_list, self.labels, self.point_len, self.point_type, self.weight_mode, self.have_noise, self.opt.noise_sigma)
        self.imgs.append(img)
        self.targets.append(target)
    elif self.dataset_path is not None:
      if "json" in self.dataset_path:
        self.bgs = list()
        self.bgs_prmse = list()
        self.psnrs = self.opt.psnr

        with open(self.dataset_path) as f:
          path_json = json.load(f)
        if self.split == "train":
          path = path_json["train"]
          bg_path = path_json["background"]
        else:
          path = path_json["val"]       
          bg_path = path_json["background_val"]
        files = glob.glob(path + "/*.npy")
        for file in files:
          json_file = os.path.join(path,file[:-4]+".json")
          with open(json_file) as f:
            targets = json.load(f)
            objs_num = len(targets)
          use = False
          if self.data_mode == "single" and objs_num == 1:
            use = True
          if self.data_mode == "all":
            use = True
          if self.data_mode == "double" and objs_num ==2:
            use = True
          if use:
            self.imgs.append(os.path.join(path,file))
            self.targets.append(json_file)
        self.num_samples = len(self.imgs)
        
        
        bg_files = glob.glob(bg_path + "/*.npy")
        for file in bg_files:
          self.bgs.append(os.path.join(bg_path,file))
          self.bgs_prmse.append(os.path.join(path,file[:-4]+".json"))
        



      else:
        if self.split == "train":
          file = os.path.join(self.dataset_path, "train.json")
          self.imgs, self.targets = get_file_list(file, self.data_mode)
        else:
          file = os.path.join(self.dataset_path, "val.json")
          self.imgs, self.targets = get_file_list(file, self.data_mode)
        self.num_samples = len(self.imgs)
        if self.opt.sample_num <  self.num_samples:
          self.num_samples = self.opt.sample_num
    else:
      path = self.otf_file[:-4]
      all_sample = len(os.listdir(path)) // 2

      start = 0
      end = all_sample
      if self.split == "train":
        if (all_sample - self.val_num) < self.num_samples:
          print("num of sample is not enough")
          assert(0)
        end = self.num_samples
      else:
        start = all_sample - self.val_num
      print("file index of sample ", start, end)
      for i in range(start, end):  
        sample = np.load(os.path.join(path,"sample_{:05d}.npy".format(i)))
        with open(os.path.join(path,"sample_{:05d}.json".format(i)), "r") as fp:
            target = json.load(fp)
        self.imgs.append(sample)
        self.targets.append(target)

  def __len__(self):   
    return self.num_samples

  def __getitem__(self, index):
    img = None
    target = None
    if False:
      otf_noise = np.random.rand(self.wave_count, self.h, self.w) * 0.1 + 0.9
      otf_list = np.multiply(self.otf_list, otf_noise)

      if self.opt.merge_bg:
        img, target = gen_merge_sample(otf_list, self.labels, self.point_len, self.point_type, self.weight_mode, self.have_noise, self.opt.noise_sigma)
    elif self.dataset_path is not None:

      if "json" in self.dataset_path:
        img_file = self.imgs[index]
        img = np.load(img_file)
        [imgc,imgh,imgw] = img.shape 
        target_file = self.targets[index]
        with open(target_file) as fp:
          target = json.load(fp)

        idx = random.randint(0,len(self.bgs)-1)
        bg_file = self.bgs[idx]
        bg = np.load(bg_file)
        with open(self.bgs_prmse[idx]) as f:
          prmse = json.load(f)
        [bgh,bgw] = bg.shape


        h_s = random.choice(range(bgh - imgh))
        h_e = h_s + imgh
        w_s = random.choice(range(bgw - imgw))
        w_e = w_s + imgw
        
        crop_bg = bg[h_s:h_e, w_s:w_e]

        crop_bg = crop_bg.reshape(1,imgh,imgw)
        psnr = random.choice(self.psnrs)
        if self.split == "train":
          psnr = random.randint(self.psnrs[0],self.psnrs[-1])
        img += crop_bg/prmse * 10**(-psnr/20) 
        img /= np.max(img)

      else:
        img_file = self.imgs[index]
        img = np.load(img_file)
        target_file = self.targets[index]
        with open(target_file) as fp:
          target = json.load(fp)
        if self.have_noise and np.random.uniform(0,1) > 0.3 and self.split == "train":
          [c, h,w] = img.shape
          sigm = self.opt.noise_sigma * np.random.uniform(0,1)
          img = img + sigm * np.random.rand(c, h, w)
    else:
      img = self.imgs[index]
      target = self.targets[index]
      if self.have_noise:
          [c, h,w] = img.shape
          sigm = self.opt.noise_sigma * np.random.uniform(0,1)
          # sample = np.multiply(sample, 1 + noise_sig * (np.random.rand(h,w) - 0.5)) 
          img = img + sigm * np.random.rand(c, h, w)
          # sample = (sample-np.min(sample))/(np.max(sample)-np.min(sample)) 
         
    obj_num = len(target)
    
    anns = target
    for i in range(obj_num):
      if self.force_merge_labels:
        anns[i][0] = self.labels.index(0) 
      else:
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
      bbox = np.array([ann[1] - ann[3]/2, ann[2] - ann[4]/2, ann[1] + ann[3]/2,ann[2] + ann[4]/2], dtype=np.float32) / self.opt.down_ratio
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
  


  
