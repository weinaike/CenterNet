cd src
# train
# python main.py ctdet --exp_id point_res34_384_mesloss_hm_5 --arch res_34 --dataset point --mse_loss \
#                      --batch_size 64 --num_epochs 70 --lr_step 45,60 --gpus 1  --debug 0 \
#                      --hm_weight 10000 --labels 0 2 4 6 8 --sample_num 8192

# test
python test.py ctdet --exp_id point_res34_384_mesloss_hm_5 --arch res_34 --dataset point --mse_loss --resume --not_prefetch_test \
                     --debug 0  --vis_thresh 0.5  --labels 0 2 4 6 8 

cd ..
