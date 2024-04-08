# Direct object detection with snapshot multispectral compressed imaging in a short-wave infrared band

论文地址：[https://doi.org/10.1364/OL.517284](https://doi.org/10.1364/OL.517284)


## 算法仓库结构


```
├── data （存放数据）
├── exp  （训练与测试输出目录）
├── experiments （训练与测试脚本文件，调试中的）
├── images	（测试样本与示例）
├── models	（模型存放地址）
|—— script       (训练与测试脚本， 论文应用)
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

论文中

执行以下命令，可开启训练任务

```python

cd src
# train & test
python main.py ctdet --exp_id seed_res50_all_1000 --arch res_50 --dataset point --mse_loss \
                     --batch_size 64 --num_epochs 40 --lr_step '30' --gpus 1 --debug 0 \
                     --hm_weight 1e5 --labels 0 1 2 3 4 5  --have_noise False --noise_sigma 0.02\
                     --sample_num 60000  --hm_gauss 3 --val_intervals 2 \
                     --dataset_path ../data/train_val_30_seed.json  --data_mode all --psnr 1000 1001 \
                     --commit "train all data [train_val_30_seed] with resnet50 psnr=1000, without noise" \
                     #  --load_model ../models/model_sup30_psnr1000_50.pth
                     #--force_merge_labels --commit "force merge label for all data, just eval local precise"

cd ..

```

训练脚本在script下，如 ctdet_infrared_res50_seed_psnr.sh

训练与测试使用的数据 

```
train_val_30_seed.json   #表示基于30超像素调制psf生成
train_val_60_seed.json   #表示基于60超像素调制psf生成
train_val_90_seed.json   #表示基于90超像素调制psf生成
train_val_120_seed.json  #表示基于120超像素调制psf生成

```

它们包含四个部分

```json
{"train":"data/psf_30_230729_234218_init_nobg_seed",
"val":"data/psf_30_230729_234218_init_nobg_seed_val",
"background":"data/psf_30_230729_234218_init_background",
"background_val":"data/psf_30_230729_234218_init_background_val"
}
```

train：表示无背景的点目标调制图样训练集， val：表示无背景的点目标调制图样的测试集

background：表示由采集的纯天空背景的调制图样训练集， background_val：表示由采集的纯天空背景的调制图样测试集

实际训练与测试过程中， 

train 与 background， 按照PNSR信噪比要求，随机组合生成训练样本

val 与 background_val， 按照PNSR信噪比要求， 随机组合生成测试/验证样本



## 如何生成数据

仿真训练数据的生成方法：

1. train/val无背景的点目标调制样本生成方法

```python
###########生成训练数据############
cd src/lib/utils/
# python PointSample_NoBG.py --otf_file ../../../data/psf_30_230729_234218_init.npy --point_type gauss --weight_mode onehot \
#                        --point_len 5  --labels 0  --save_path ../../../data/one_background --num 10 --jobs 16

python PointSample_NoBG.py --otf_file ../../../data/psf_60_230729_130106_init.npy --point_type gauss --weight_mode rand \
                        --point_len 5  --labels 0 1 2 3 4 5  --save_path ../../../data/psf_60_230729_130106_init_nobg_seed --num 20000 --jobs 16

# python PointSample_NoBG.py --otf_file ../../../data/psf_90_230730_080929_init.npy --point_type gauss --weight_mode rand \
#                         --point_len 5  --labels 0 1 2 3 4 5  --save_path ../../../data/psf_90_230730_080929_init_nobg_seed --num 20000 --jobs 16

# python PointSample_NoBG.py --otf_file ../../../data/psf_120_230801_180614_init.npy --point_type gauss --weight_mode rand \
#                         --point_len 5  --labels 0 1 2 3 4 5  --save_path ../../../data/psf_120_230801_180614_init_nobg_seed --num 20000 --jobs 16

```

脚本见：experiments/gen_data_noBG.sh

2. background_val/background 天空背景的调制样本仿真方法

```python
#experiments/gen_backgournd.sh
###########生成训练数据############
cd src/lib/utils/

python BackgroundSample.py --otf_file ../../../data/PSF1110_6_IR_120_init.npy \
                          --save_path ../../../data/PSF1110_6_IR_120_init_background_val --jobs 10



python BackgroundSample.py --otf_file ../../../data/PSF1110_6_IR_120_init.npy \
                          --save_path ../../../data/PSF1110_6_IR_120_init_background --jobs 10


python BackgroundSample.py --otf_file ../../../data/PSF1110_6_IR_120_final.npy \
                          --save_path ../../../data/PSF1110_6_IR_120_final_background_val --jobs 10


python BackgroundSample.py --otf_file ../../../data/PSF1110_6_IR_120_final.npy \
                          --save_path ../../../data/PSF1110_6_IR_120_final_background --jobs 10



cd ../../../


```

## 背景调制图样

天空背景调制图样，由实验设备采集获取

经过简单的归一化处理得到，PSF1110_6_IR_30_init.npy， 表示11月10日，采集的红外6组滤光片的PSF调制数据。

而天空背景的的数据较大， 上传[百度网盘](https://pan.baidu.com/s/12vaHgK3WWUAllUIOE09p8w?pwd=xmh8)，提取码: xmh8
