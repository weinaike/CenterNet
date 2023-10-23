
cd src

# test only

python test.py ctdet --exp_id seed_res50_all_scratch_new --arch res_50 --dataset point --mse_loss --not_prefetch_test \
                    --debug 0  --vis_thresh 0.5  --labels 0 1 2 3 4 5  --have_noise False --point_len 5 --hm_gauss 3  \
                    --gpus 1 --dataset_path ../data/train_val_30_seed.json --data_mode single --psnr 20 \
                    --commit "train with all data, test with single point data [psnr 20db]" \
                    --load_model ../exp/ctdet/seed_res50_all_scratch_new/logs_2023-10-19-11-46-23/model_best.pth

# python test.py ctdet --exp_id seed_res50_all_scratch --arch res_50 --dataset point --mse_loss --not_prefetch_test \
#                     --debug 0  --vis_thresh 0.5  --labels 0 1 2 3 4 5  --have_noise False --point_len 20 --hm_gauss 3  \
#                     --gpus 1 --dataset_path ../data/train_val_30.json --data_mode single --psnr 20 \
#                     --commit "train with all data, test with single point data [psnr 20db]" \
#                     --load_model ../exp/ctdet/seed_res50_all_scratch/logs_2023-10-14-21-27-50/model_best.pth




cd ..
