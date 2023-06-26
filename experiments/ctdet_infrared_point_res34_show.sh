cd src
# test
python show.py ctdet --exp_id infrared_point_res34_384 --arch res_34 --dataset point --mse_loss --not_prefetch_test \
                    --debug 0  --vis_thresh 0.5  --labels 1 3 5  --have_noise True --noise_sigma 0.01 \
                    --point_type ones_rand --point_len 111 --merge_bg --hm_gauss 3 --resume --otf_file ../data/PSF0620_04_4_40.npy


cd ..


#8192: