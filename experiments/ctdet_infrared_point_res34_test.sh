cd src
# test only
python test.py ctdet --exp_id infrared_point_res50_384 --arch res_50 --dataset point --mse_loss --not_prefetch_test \
                    --debug 0  --vis_thresh 0.5  --labels 0 1 2 3 4 5  --have_noise False --point_len 5 --hm_gauss 3  \
                    --gpus 1 --dataset_path ../data/train_val.json --data_mode single --psnr 20 \
                    --commit "train with all data, test with single point data [psnr 20db]" \
                    --load_model ../exp/ctdet/infrared_point_res50_384/logs_2023-09-05-09-24-43/model_best.pth
cd ..
