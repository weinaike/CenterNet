import test

import os
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # basic experiment setting
    parser.add_argument('--model_path', default="../exp/ctdet/infrared_point_res34_384/logs_2023-08-18-11-22-53/model_best.pth", type=str, help='')
    parser.add_argument('--dataset_path', default="../data/PSF0815_6_IR_30_384/", type=str, help='')
    parser.add_argument('--gpu', default=1, type=int, help='')
    args = parser.parse_args()

    model_path = args.model_path
    dataset_path = args.dataset_path
    gpu = args.gpu

    data_commit_dict = {
        # "all":"train with all data, test with all data",
        # "single":"train with all data, test with all single data",        
        # "single_5x":"train with all data, test with single and snr[5x] data",
        # "single_10x":"train with all data, test with single and snr[10x] data",
        # "single_50x":"train with all data, test with single and snr[50x] data",
        # "double":"train with all data, test with all double data",
        # "double_5x":"train with all data, test with double and snr[5x] data",
        # "double_10x":"train with all data, test with double and snr[10x] data",
        # "double_50x":"train with all data, test with double and snr[50x] data",
        # "double_10x_10":"train with all data, test with double and snr[10x] and dist[10] data",
        # "double_10x_20":"train with all data, test with double and snr[10x] and dist[20] data",
        # "double_10x_30":"train with all data, test with double and snr[10x] and dist[30] data",
        # "double_10x_40":"train with all data, test with double and snr[10x] and dist[40] data",
        "double_10x_50":"train with all data, test with double and snr[10x] and dist[50] data",
        # "double_10x_60":"train with all data, test with double and snr[10x] and dist[60] data",
        # "double_10x_70":"train with all data, test with double and snr[10x] and dist[70] data",
        # "double_10x_80":"train with all data, test with double and snr[10x] and dist[80] data",
        # "double_10x_90":"train with all data, test with double and snr[10x] and dist[90] data",
        # "double_10x_100":"train with all data, test with double and snr[10x] and dist[100] data",
        # "double_10x_110":"train with all data, test with double and snr[10x] and dist[110] data",
        # "double_10x_120":"train with all data, test with double and snr[10x] and dist[120] data",
        # "double_10x_130":"train with all data, test with double and snr[10x] and dist[130] data",

        # "single_10x_-2":"train with all data, test with single and snr[10x] data of sunny background",
        # "single_10x_-3":"train with all data, test with single and snr[10x] data of cloudy background",
        # "double_10x_-2":"train with all data, test with double and snr[10x] data of sunny background",
        # "double_10x_-3":"train with all data, test with double and snr[10x] data of cloudy background",
        
    }


    for mode, commit in data_commit_dict.items():
        cmd = "python test.py ctdet --exp_id infrared_point_res34_384 --arch res_34 --dataset point --mse_loss --not_prefetch_test \
                    --debug 0  --vis_thresh 0.5  --labels 0 1 2 3 4 5  --have_noise False --point_len 5 --hm_gauss 3  \
                    --gpus {} --dataset_path '{}' --data_mode {} --commit '{}' --load_model {}".format(gpu, dataset_path, mode, commit, model_path)

        os.system(cmd)