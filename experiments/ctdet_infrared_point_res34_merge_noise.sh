cd src
# train
python main.py ctdet --exp_id infrared_point_res34_384 --arch res_34 --dataset point --mse_loss \
                     --batch_size 64 --num_epochs 100 --lr_step 80 --gpus 0  --debug 0 \
                     --hm_weight 1e5 --labels 1 3 5 --have_noise True --noise_sigma 0.01 \
                     --sample_num 8192 --point_type ones_rand --point_len 111 --merge_bg --hm_gauss 3 \
                    --otf_file ../data/PSF0620_04_4_40.npy
# test
python test.py ctdet --exp_id infrared_point_res34_384 --arch res_34 --dataset point --mse_loss --not_prefetch_test \
                    --debug 0  --vis_thresh 0.5  --labels 1 3 5  --have_noise True --noise_sigma 0.01 \
                    --point_type ones_rand --point_len 111 --merge_bg --hm_gauss 3 --resume --otf_file ../data/PSF0620_04_4_40.npy


cd ..


#8192: