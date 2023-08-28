import json
import os 
import glob
import math
import argparse 


def parse_dict(psnr_dict:dict):
    target_dict = dict()
    for key in psnr_dict.keys():
        path = psnr_dict[key]
        files = os.listdir(path)

        for file in files:
            if "json" in file:
                file = os.path.join(path, file)
            else:
                continue
            with open(file) as f:
                targets = json.load(f)
            
            point_num = len(targets)
            bg_label = targets[0][-4]
            # psnr= targets[0][-3]
            psnr = key
            distance = 0           
            if point_num == 2:
                x1 = targets[0][1]
                y1 = targets[0][2]
                x2 = targets[1][1]
                y2 = targets[1][2]
                distance = int(math.sqrt((x1-x2)**2+(y1-y2)**2))

            distance_label = distance // 10 * 10
            key_name = "{}_{}_{}_{}".format(point_num, bg_label, psnr, distance_label)
            file_train_path = "../data/"+file
            if key_name in target_dict.keys():
                target_dict[key_name].append(file_train_path)
            else:
                target_dict[key_name] = [file_train_path]
    return target_dict


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # basic experiment setting
    parser.add_argument('--mode', default="train", type=str, help='')
    parser.add_argument('--path', default="PSF0815_6_IR_30_384", type=str, help='')
    args = parser.parse_args()

    path = args.path
    mode = args.mode
    #### 5x : 10dB   10x:  20dB    50x: 40dB
    psnr_dict = {"5x": "PSF0815_6_IR_30_psnr_10_384_init_bg",
                "10x": "PSF0815_6_IR_30_psnr_20_384_init_bg",
                "50x": "PSF0815_6_IR_30_psnr_40_384_init_bg"}
    
    val_psnr_dict = {"5x": "PSF0815_6_IR_30_psnr_10_384_init_bg_val",
            "10x": "PSF0815_6_IR_30_psnr_20_384_init_bg_val",
            "50x": "PSF0815_6_IR_30_psnr_40_384_init_bg_val"}

    if not os.path.exists(path):
        os.mkdir(path)
    if mode == "train":
        target_dict = parse_dict(psnr_dict)
        print("----------train------------")
        print(target_dict.keys())
        #dict_keys(['2_-1_10x_60', '2_-1_10x_40', '2_-1_10x_30', '1_-1_10x_0', '2_-1_10x_70', '2_-1_10x_50', 
        # '2_-1_10x_20', '2_-1_10x_80', '2_-1_10x_90', '2_-1_10x_0', '2_-1_10x_10', '2_-1_5x_60', '1_-1_5x_0', 
        # '2_-1_5x_20', '2_-1_5x_0', '2_-1_5x_50', '2_-1_5x_10', '2_-1_5x_90', '2_-1_5x_70', '2_-1_5x_80', '2_-1_5x_40', 
        # '2_-1_5x_30', '2_-1_50x_10', '1_-1_50x_0', '2_-1_50x_20', '2_-1_50x_0', '2_-1_50x_40', '2_-1_50x_70', 
        # '2_-1_50x_60', '2_-1_50x_90', '2_-1_50x_50', '2_-1_50x_80', '2_-1_50x_30'])
        for key , val in target_dict.items():
            print(key, len(val))
        with open(os.path.join(path, "train.json"),'w') as f:
            json.dump(target_dict,f)
    
    if mode == "val":
        target_dict = parse_dict(val_psnr_dict)
        print("----------val------------")
        print(target_dict.keys())
        for key , val in target_dict.items():
            print(key, len(val))
        with open(os.path.join(path, "val.json"),'w') as f:
            json.dump(target_dict,f)

