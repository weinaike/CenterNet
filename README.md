# Direct object detection with snapshot multispectral compressed imaging in a short-wave infrared band

论文地址：https://doi.org/10.1364/OL.517284


## 算法仓库结构

```
├── data （存放数据）
├── exp  （训练与测试输出目录）
├── experiments （训练与测试脚本文件）
├── images	（测试样本与示例）
├── models	（模型存放地址）
└── src		（源码库）
    └── lib
        ├── datasets  （数据集定义）
        ├── detectors （目标检测前向推理）
        ├── external （nms库）
        ├── models （模型定义）
        │   └── networks
        ├── trains （训练代码）
        └── utils  （数据生成与调试工具）

```


## 算法仓库如何使用

执行以下命令，可开启训练任务

```python

cd src
# train & test
python main.py ctdet --exp_id infrared_point_res50_384 --arch res_50 --dataset point --mse_loss \
                     --batch_size 64 --num_epochs 50 --lr_step 40 --gpus 0 --debug 0 \
                     --hm_weight 1e5 --labels 0 1 2 3 4 5  --have_noise True --noise_sigma 0.02\
                     --sample_num 60000  --hm_gauss 3 --val_intervals 1 \
                     --dataset_path ../data/PSF0815_6_IR_30_384_PSNR_init_bg/  --data_mode all \
                     --commit "train all data with resnet50 [dataset is PSF0815_6_IR_30_384_PSNR_init_bg], input size is 384, with 70% add 0.02 noise for train , val without noise"
                     #--force_merge_labels --commit "force merge label for all data, just eval local precise"
                     #--load_model ../exp/ctdet/infrared_point_res34_384/logs_2023-08-02-10-01/model_best.pth

cd ..
```

训练脚本在experiments下，如 ctdet_infrared_point_res34_true_bg.sh



## 如何获取数据
