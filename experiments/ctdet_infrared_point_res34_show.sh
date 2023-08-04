cd src
# show
python show.py ctdet --exp_id infrared_point_res34_384 --arch res_34  \
                    --debug 0  --vis_thresh 0.5  --labels 0 1 2 3 4 5 \
                    --load_model ../exp/ctdet/infrared_point_res34_384/logs_2023-08-02-13-59/model_best.pth



# test only
# python test.py ctdet --exp_id infrared_point_res34_384 --arch res_34 --dataset point --mse_loss --not_prefetch_test \
#                     --debug 0  --vis_thresh 0.5  --labels 0 1 2 3 4 5  --have_noise True --noise_sigma 0.01 --gpus 1 \
#                     --point_type ones_rand --point_len 5 --hm_gauss 3  --otf_file ../data/PSF0620_04_4_40.npy \
#                     --load_model ../exp/ctdet/infrared_point_res34_384/logs_2023-08-02-13-59/model_last.pth

cd ..


#8192: