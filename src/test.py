from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import json
import cv2
import numpy as np
import random
import time
from progress.bar import Bar
import torch

import pickle as cPickle

import matplotlib.pyplot as plt

from external.nms import soft_nms
from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from datasets.dataset_factory import  dataset_factory
from detectors.detector_factory import detector_factory


def voc_ap(rec, prec, use_07_metric=False):
  """ ap = voc_ap(rec, prec, [use_07_metric])
  Compute VOC AP given precision and recall.
  If use_07_metric is true, uses the
  VOC 07 11 point method (default:False).
  """
  if use_07_metric:
    # 11 point metric
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
      if np.sum(rec >= t) == 0:
        p = 0
      else:
        p = np.max(prec[rec >= t])
        # print(t, p)
      ap = ap + p / 11.
  else:
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap

def voc_eval_new(dets, annos, images, classname, ovthresh=0.5, use_07_metric=False):

  class_recs = {}
  npos = 0
  for image in images:
    R = [obj for obj in annos[image] if obj['name'] == classname]
    bbox = np.array([x['bbox'] for x in R])
    det = [False] * len(R)
    npos = npos + len(R)
    class_recs[image] = {'bbox': bbox, 'det': det}
  # 获取这个类别，在这张图片中的所有目标，按图谱名称索引

  # 该类目标的所有检测结果，包括image_ids，confidence， BB
  # dets 按类别分
  image_ids = [x[5] for x in dets]
  confidence = np.array([float(x[4]) for x in dets])
  BB = np.array([[float(z) for z in x[:4]] for x in dets])

  nd = len(image_ids) #图片数量
  tp = np.zeros(nd) #每个图片的结果
  fp = np.zeros(nd)

  if BB.shape[0] > 0:
    # sort by confidence
    sorted_ind = np.argsort(-confidence) #排序
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind] #排序， 

    # go down dets and mark TPs and FPs
    for d in range(nd):
      R = class_recs[image_ids[d]]
      bb = BB[d, :].astype(float)
      ovmax = -np.inf
      BBGT = R['bbox'].astype(float)

      if BBGT.size > 0:
        # compute overlaps
        # intersection
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

      if ovmax > ovthresh:
        if not R['det'][jmax]:
          tp[d] = 1.
          R['det'][jmax] = 1
        else:
          fp[d] = 1.
      else:
        fp[d] = 1.

  # compute precision recall
  fp = np.cumsum(fp)
  tp = np.cumsum(tp)
  rec = tp / float(npos)
  
  # avoid divide by zero in case the first detection matches a difficult
  # ground truth
  prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
  ap = voc_ap(rec, prec, use_07_metric)

  return rec, prec, ap

def test(opt):
  print("----------------test-------------")
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  np.random.seed(opt.seed)
  random.seed(opt.seed)
  torch.manual_seed(opt.seed)
  torch.cuda.manual_seed(opt.seed)
  torch.cuda.manual_seed_all(opt.seed)  # if you are using multi-GPU.


  # Logger(opt)
  Detector = detector_factory[opt.task]
  
  split = 'val' if not opt.trainval else 'test'
  dataset = Dataset(opt, split)
  detector = Detector(opt)
  data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

  # torch.backends.cudnn.benchmark = False
  # torch.backends.cudnn.deterministic = True
  num_iters = len(dataset)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}

  ind = 0
  cls_names = dataset.labels

  annos_cls = {}
  results = {}
  for cls in cls_names:
    results.update({cls:list()})

  image_ids = list()
  cost = 0

  for sample in data_loader:
    ind = ind + 1
    # img_id = dataset.images[ind]
    # img_info = dataset.coco.loadImgs(ids=[img_id])[0]
    # img_path = os.path.join(dataset.img_dir, img_info['file_name'])
    # print(sample)
    img_path = sample["input"]
    img_info = sample["meta"]
    # print(img_info)
    start = time.time()
    ret = detector.run(sample, img_info)
    end = time.time()
    cost += end-start
    
    if opt.debug > 0 and ind > 20:
      print("break for index is more than 20")
      break


    anns_gt = img_info["gt_det"]
    bs, obj_num, obj_ann = anns_gt.size()

    if bs != 1:
      print("not support batch_size more than 1")
      assert(0)
    # load anno
    gt_objs =  list()
    
    expand = 0
    for j in range(obj_num):
      bbox = anns_gt[0][j][0:4].numpy()*opt.down_ratio
      bbox[0]-=expand
      bbox[1]-=expand
      bbox[2]+=expand
      bbox[3]+=expand
      cls_id =  int(anns_gt[0][j][5])
      name = cls_names[cls_id]
      gt_obj = {"bbox": bbox, "name":name}
      gt_objs.append(gt_obj)
    if len(gt_objs) > 0:
      annos_cls.update({ind:gt_objs})
    image_ids.append(ind)

    # load detection

    for cls_id, dets in ret['results'].items():
      name = cls_names[cls_id - 1]
      
      for det in dets:
        det_new = list(det)
        det_new.append(ind)
        results[name].append(det_new)

    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
    bar.next()

  bar.finish()
  # dataset.run_eval(results, opt.save_dir)
  if not os.path.exists(opt.save_dir):
    os.mkdir(opt.save_dir)

  if ind == 0:
    return
  file = open(os.path.join(opt.save_dir, 'map.txt'), 'w')
  file.writelines("commit:{}, num of data is {}, cost time of once inference {}\n".format(opt.commit, ind, cost/ind))
  # <4px, <4px, 3<px, <2px,  <1px , <0.5px 
  threshs = [0.01, 0.02, 0.087, 0.22, 0.47, 0.67]
  for th in threshs:
    aps = []
    print("thresh: {:.2f}".format(th))
    file.writelines("thresh: {:.2f}\n".format(th))
    for cls in cls_names:
      rec, prec, ap = voc_eval_new(results[cls], annos_cls, image_ids, cls, ovthresh=th)
      aps += [ap]
      print(('AP for {} = {:.4f} '.format(cls, ap,)))
      file.writelines('AP for {} = {:.4f} \n'.format(cls, ap))
      with open(os.path.join(opt.save_dir, '{}_pr.pkl'.format(cls)), 'wb') as f:
        cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
      if opt.debug > 0:
        plt.figure()
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.title('PR cruve')
        plt.plot(rec, prec, '-r')
        plt.savefig(os.path.join(opt.save_dir, 'thresh_{}_label_{}_PR.jpg'.format(th,cls)))
        plt.close()
      
    print(('Mean AP = {:.4f}\n'.format(np.mean(aps))))
    file.writelines(('Mean AP = {:.4f}\n'.format(np.mean(aps))))
  file.close()
  print('~~~~~~~~')


if __name__ == '__main__':
  opt = opts().parse()
  test(opt)
