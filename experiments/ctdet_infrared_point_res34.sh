cd src
# train
python main.py ctdet --exp_id infrared_point_res34_384 --arch res_34 --dataset point --mse_loss \
                     --batch_size 64 --num_epochs 100 --lr_step 80 --gpus 0  --debug 0 \
                     --hm_weight 1e5 --labels 1 3 5 --have_noise True --noise_sigma 0.1 \
                     --sample_num 8192 --point_type ones_rand --point_len 111 --hm_gauss 3
# test
python test.py ctdet --exp_id infrared_point_res34_384 --arch res_34 --dataset point --mse_loss --not_prefetch_test \
                    --debug 0  --vis_thresh 0.5  --labels 1 3 5  --have_noise True --noise_sigma 0.1 \
                    --point_type ones_rand --point_len 111 --hm_gauss 3 --resume


cd ..


#8192: