cd src/lib/utils/
python PointSample.py --otf_file ../../../data/PSF0620_04_4_40.npy --point_type ones_rand --weight_mode rand \
                        --point_len 5 --noise_sigma 0.01 --labels 0 1 2 3 4 5 --merge_bg --nsr 0.02
cd ../../../



cd src
# train & test
python main.py ctdet --exp_id infrared_point_res34_384 --arch res_34 --dataset point --mse_loss \
                     --batch_size 64 --num_epochs 50 --lr_step 40 --gpus 0  --debug 0 \
                     --hm_weight 1e5 --labels 0 1 2 3 4 5  --have_noise True --noise_sigma 0.01 \
                     --sample_num 9000  --point_type ones_rand --weight_mode rand --point_len 5 --merge_bg --hm_gauss 3 \
                     --otf_file ../data/PSF0620_04_4_40.npy --val_intervals 1 \
                    #  --load_model ../exp/ctdet/infrared_point_res34_384/logs_2023-08-02-10-01/model_best.pth


# test only
# python test.py ctdet --exp_id infrared_point_res34_384 --arch res_34 --dataset point --mse_loss --not_prefetch_test \
#                     --debug 0  --vis_thresh 0.5  --labels 1 3 5  --have_noise True --noise_sigma 0.01 --gpus 1 \
#                     --point_type ones_rand --point_len 111 --hm_gauss 3  --otf_file ../data/PSF0620_04_4_40_2.npy \
#                     --load_model ../exp/ctdet/infrared_point_res34_384/logs_2023-06-27-17-31/model_last.pth



cd ..
