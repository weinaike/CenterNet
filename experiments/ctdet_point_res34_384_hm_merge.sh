cd src
# train
python main.py ctdet --exp_id point_res34_384_mesloss_hm_merge --arch res_34 --dataset point --mse_loss \
                     --batch_size 64 --num_epochs 70 --lr_step 45,60 --gpus 0  --debug 0 \
                     --hm_weight 1e5 --wh_weight 1 --labels 0 2 4 6 8 --sample_num 8192 --point_len 5 --merge_bg --hm_gauss 1
# test
# python test.py ctdet --exp_id point_res34_384_mesloss_hm_merge --arch res_34 --dataset point --mse_loss --resume --not_prefetch_test \
#                      --debug 0  --vis_thresh 0.5  --labels 0 2 4 6 8 

cd ..
