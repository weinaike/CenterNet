cd src
# show
python show.py ctdet --exp_id seed_res50_all_scratch_new --arch res_50  --gpus 1 --dataset_path "../data/train_val_30_seed.json" --data_mode single \
                    --debug 0  --vis_thresh 0.5  --labels 0 1 2 3 4 5 --otf_file "../data/psf_30_230729_234218_init.npy" --psnr 20 \
                    --load_model ../exp/ctdet/seed_res50_all_scratch_new/logs_2023-10-19-11-46-23/model_best.pth



# test only
# python test.py ctdet --exp_id infrared_point_res50_384 --arch res_50 --dataset point --mse_loss --not_prefetch_test \
#                     --debug 0  --vis_thresh 0.5  --labels 0 1 2 3 4 5  --have_noise False --point_len 5 --hm_gauss 3  \
#                     --gpus 1 --dataset_path ../data/PSF0815_6_IR_30/  --data_mode single_10x \
#                     --commit "train with all data, test with single point data [psnr 10db]" \
#                     --load_model ../exp/ctdet/infrared_point_res34_384/logs_2023-08-17-22-32/model_best.pth
cd ..


#8192:
