cd src
# train
# python main.py ctdet --exp_id point_res34_384_mesloss_hm_merge_5 --arch res_34 --dataset point --mse_loss \
#                      --batch_size 64 --num_epochs 70 --lr_step 45,60 --gpus 0  --debug 0 \
#                      --hm_weight 1e5 --labels 0 2 4 6 8 --have_noise True --noise_sigma 0.1 \
#                      --sample_num 8192 --point_type ones_rand --point_len 111 --merge_bg --hm_gauss 3 
# test
python test.py ctdet --exp_id point_res34_384_mesloss_hm_merge_5 --arch res_34 --dataset point --mse_loss --resume --not_prefetch_test \
                     --debug 0  --vis_thresh 0.5  --labels 0 2 4 6 8 --have_noise True  --point_type ones_rand --point_len 111 --merge_bg --hm_gauss 3

cd ..













# 以下配置， 能够实现mAP ： 1， 
# 可以增加目标点的噪声， 但是其他噪声需要抑制，包括背景随机噪声，与背景光谱随机权重噪声。
# python main.py ctdet --exp_id point_res34_384_mesloss_hm_merge --arch res_34 --dataset point --mse_loss \
#                      --batch_size 64 --num_epochs 70 --lr_step 45,60 --gpus 0  --debug 0 \
#                      --hm_weight 1e5 --labels 0 --have_noise True --sample_num 2048 --point_type ones_rand --point_len 111 --merge_bg --hm_gauss 3 \
#                      --load_model ../exp/ctdet/point_res34_384_mesloss_hm_merge/model_last.pth


# 以下配置，可实现检测mAP ：1
# 即无噪声条件, 纯亮斑
# python main.py ctdet --exp_id point_res34_384_mesloss_hm_merge --arch res_34 --dataset point --mse_loss \
#                      --batch_size 64 --num_epochs 70 --lr_step 45,60 --gpus 0  --debug 0 \
#                      --hm_weight 1e5 --labels 0 --have_noise False --sample_num 2048 --point_type ones --point_len 111 --merge_bg --hm_gauss 3