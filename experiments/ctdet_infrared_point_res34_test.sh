
cd src

# test only

python test.py ctdet --exp_id infrared_point_res50_384_single --arch res_50 --dataset point --mse_loss --not_prefetch_test \
                    --debug 0  --vis_thresh 0.5  --labels 0 1 2 3 4 5  --have_noise False --point_len 5 --hm_gauss 3  \
                    --gpus 1 --dataset_path ../data/train_val_30_seed.json --data_mode single --psnr 20 \
                    --commit "train with all data, test with single point data [psnr 20db]" \
                    --load_model ../exp/ctdet/infrared_point_res50_384_single_seed/logs_2023-10-10-17-17-14/model_best.pth

# python test.py ctdet --exp_id infrared_point_res50_384_single --arch res_50 --dataset point --mse_loss --not_prefetch_test \
#                     --debug 0  --vis_thresh 0.5  --labels 0 1 2 3 4 5  --have_noise False --point_len 5 --hm_gauss 3  \
#                     --gpus 1 --dataset_path ../data/train_val_30.json --data_mode single --psnr 20 \
#                     --commit "train with all data, test with single point data [psnr 20db]" \
#                     --load_model ../exp/ctdet/infrared_point_res50_384_single/logs_2023-10-08-15-55-35/model_best.pth

# python test.py ctdet --exp_id infrared_point_res50_384_single --arch res_50 --dataset point --mse_loss --not_prefetch_test \
#                     --debug 0  --vis_thresh 0.5  --labels 0 1 2 3 4 5  --have_noise False --point_len 5 --hm_gauss 3  \
#                     --gpus 1 --dataset_path ../data/train_val_30.json --data_mode single --psnr 20 \
#                     --commit "train with all data, test with single point data [psnr 20db]" \
#                     --load_model ../exp/ctdet/infrared_point_res50_384_single/logs_2023-10-08-15-55-35/model_best.pth
# python test.py ctdet --exp_id infrared_point_res50_384_single --arch res_50 --dataset point --mse_loss --not_prefetch_test \
#                     --debug 0  --vis_thresh 0.5  --labels 0 1 2 3 4 5  --have_noise False --point_len 5 --hm_gauss 3  \
#                     --gpus 1 --dataset_path ../data/train_val_30.json --data_mode single --psnr 20 \
#                     --commit "train with all data, test with single point data [psnr 20db]" \
#                     --load_model ../exp/ctdet/infrared_point_res50_384_single/logs_2023-10-08-15-55-35/model_best.pth
# python test.py ctdet --exp_id infrared_point_res50_384_single --arch res_50 --dataset point --mse_loss --not_prefetch_test \
#                     --debug 0  --vis_thresh 0.5  --labels 0 1 2 3 4 5  --have_noise False --point_len 5 --hm_gauss 3  \
#                     --gpus 1 --dataset_path ../data/train_val_30.json --data_mode single --psnr 20 \
#                     --commit "train with all data, test with single point data [psnr 20db]" \
#                     --load_model ../exp/ctdet/infrared_point_res50_384_single/logs_2023-10-08-15-55-35/model_best.pth




# python test.py ctdet --exp_id infrared_point_res50_384_single --arch res_50 --dataset point --mse_loss --not_prefetch_test \
#                     --debug 0  --vis_thresh 0.5  --labels 0 1 2 3 4 5  --have_noise False --point_len 5 --hm_gauss 3  \
#                     --gpus 1 --dataset_path ../data/train_val_60.json --data_mode single --psnr 20 \
#                     --commit "train with all data, test with single point data [psnr 20db]" \
#                     --load_model ../exp/ctdet/infrared_point_res50_384_single/logs_2023-10-08-15-56-24/model_best.pth

# python test.py ctdet --exp_id infrared_point_res50_384_single --arch res_50 --dataset point --mse_loss --not_prefetch_test \
#                     --debug 0  --vis_thresh 0.5  --labels 0 1 2 3 4 5  --have_noise False --point_len 5 --hm_gauss 3  \
#                     --gpus 1 --dataset_path ../data/train_val_60.json --data_mode single --psnr 20 \
#                     --commit "train with all data, test with single point data [psnr 20db]" \
#                     --load_model ../exp/ctdet/infrared_point_res50_384_single/logs_2023-10-08-15-56-24/model_best.pth
                    
# python test.py ctdet --exp_id infrared_point_res50_384_single --arch res_50 --dataset point --mse_loss --not_prefetch_test \
#                     --debug 0  --vis_thresh 0.5  --labels 0 1 2 3 4 5  --have_noise False --point_len 5 --hm_gauss 3  \
#                     --gpus 1 --dataset_path ../data/train_val_60.json --data_mode single --psnr 20 \
#                     --commit "train with all data, test with single point data [psnr 20db]" \
#                     --load_model ../exp/ctdet/infrared_point_res50_384_single/logs_2023-10-08-15-56-24/model_best.pth

# python test.py ctdet --exp_id infrared_point_res50_384_single --arch res_50 --dataset point --mse_loss --not_prefetch_test \
#                     --debug 0  --vis_thresh 0.5  --labels 0 1 2 3 4 5  --have_noise False --point_len 5 --hm_gauss 3  \
#                     --gpus 1 --dataset_path ../data/train_val_60.json --data_mode single --psnr 20 \
#                     --commit "train with all data, test with single point data [psnr 20db]" \
#                     --load_model ../exp/ctdet/infrared_point_res50_384_single/logs_2023-10-08-15-56-24/model_best.pth

# python test.py ctdet --exp_id infrared_point_res50_384_single --arch res_50 --dataset point --mse_loss --not_prefetch_test \
#                     --debug 0  --vis_thresh 0.5  --labels 0 1 2 3 4 5  --have_noise False --point_len 5 --hm_gauss 3  \
#                     --gpus 1 --dataset_path ../data/train_val_60.json --data_mode single --psnr 20 \
#                     --commit "train with all data, test with single point data [psnr 20db]" \
#                     --load_model ../exp/ctdet/infrared_point_res50_384_single/logs_2023-10-08-15-56-24/model_best.pth




# python test.py ctdet --exp_id infrared_point_res50_384_single_single --arch res_50 --dataset point --mse_loss --not_prefetch_test \
#                     --debug 0  --vis_thresh 0.5  --labels 0 1 2 3 4 5  --have_noise False --point_len 5 --hm_gauss 3  \
#                     --gpus 1 --dataset_path ../data/train_val_90.json --data_mode single --psnr 20 \
#                     --commit "train with all data, test with single point data [psnr 20db]" \
#                     --load_model ../exp/ctdet/infrared_point_res50_384_single/logs_2023-10-08-15-56-47/model_best.pth                                        

                                                                                
# python test.py ctdet --exp_id infrared_point_res50_384_single --arch res_50 --dataset point --mse_loss --not_prefetch_test \
#                     --debug 0  --vis_thresh 0.5  --labels 0 1 2 3 4 5  --have_noise False --point_len 5 --hm_gauss 3  \
#                     --gpus 1 --dataset_path ../data/train_val_90.json --data_mode single --psnr 20 \
#                     --commit "train with all data, test with single point data [psnr 20db]" \
#                     --load_model ../exp/ctdet/infrared_point_res50_384_single/logs_2023-10-08-15-56-47/model_best.pth     
                                                                                
# python test.py ctdet --exp_id infrared_point_res50_384_single --arch res_50 --dataset point --mse_loss --not_prefetch_test \
#                     --debug 0  --vis_thresh 0.5  --labels 0 1 2 3 4 5  --have_noise False --point_len 5 --hm_gauss 3  \
#                     --gpus 1 --dataset_path ../data/train_val_90.json --data_mode single --psnr 20 \
#                     --commit "train with all data, test with single point data [psnr 20db]" \
#                     --load_model ../exp/ctdet/infrared_point_res50_384_single/logs_2023-10-08-15-56-47/model_best.pth     

                                                                                
# python test.py ctdet --exp_id infrared_point_res50_384_single --arch res_50 --dataset point --mse_loss --not_prefetch_test \
#                     --debug 0  --vis_thresh 0.5  --labels 0 1 2 3 4 5  --have_noise False --point_len 5 --hm_gauss 3  \
#                     --gpus 1 --dataset_path ../data/train_val_90.json --data_mode single --psnr 20 \
#                     --commit "train with all data, test with single point data [psnr 20db]" \
#                     --load_model ../exp/ctdet/infrared_point_res50_384_single/logs_2023-10-08-15-56-47/model_best.pth     
                                                                                
# python test.py ctdet --exp_id infrared_point_res50_384_single --arch res_50 --dataset point --mse_loss --not_prefetch_test \
#                     --debug 0  --vis_thresh 0.5  --labels 0 1 2 3 4 5  --have_noise False --point_len 5 --hm_gauss 3  \
#                     --gpus 1 --dataset_path ../data/train_val_90.json --data_mode single --psnr 20 \
#                     --commit "train with all data, test with single point data [psnr 20db]" \
#                     --load_model ../exp/ctdet/infrared_point_res50_384_single/logs_2023-10-08-15-56-47/model_best.pth     




# python test.py ctdet --exp_id infrared_point_res50_384_single --arch res_50 --dataset point --mse_loss --not_prefetch_test \
#                     --debug 0  --vis_thresh 0.5  --labels 0 1 2 3 4 5  --have_noise False --point_len 5 --hm_gauss 3  \
#                     --gpus 1 --dataset_path ../data/train_val_120.json --data_mode single --psnr 20 \
#                     --commit "train with all data, test with single point data [psnr 20db]" \
#                     --load_model ../exp/ctdet/infrared_point_res50_384_single/logs_2023-10-08-15-57-09/model_best.pth


# python test.py ctdet --exp_id infrared_point_res50_384_single --arch res_50 --dataset point --mse_loss --not_prefetch_test \
#                     --debug 0  --vis_thresh 0.5  --labels 0 1 2 3 4 5  --have_noise False --point_len 5 --hm_gauss 3  \
#                     --gpus 1 --dataset_path ../data/train_val_120.json --data_mode single --psnr 20 \
#                     --commit "train with all data, test with single point data [psnr 20db]" \
#                     --load_model ../exp/ctdet/infrared_point_res50_384_single/logs_2023-10-08-15-57-09/model_best.pth

# python test.py ctdet --exp_id infrared_point_res50_384_single --arch res_50 --dataset point --mse_loss --not_prefetch_test \
#                     --debug 0  --vis_thresh 0.5  --labels 0 1 2 3 4 5  --have_noise False --point_len 5 --hm_gauss 3  \
#                     --gpus 1 --dataset_path ../data/train_val_120.json --data_mode single --psnr 20 \
#                     --commit "train with all data, test with single point data [psnr 20db]" \
#                     --load_model ../exp/ctdet/infrared_point_res50_384_single/logs_2023-10-08-15-57-09/model_best.pth
# python test.py ctdet --exp_id infrared_point_res50_384_single --arch res_50 --dataset point --mse_loss --not_prefetch_test \
#                     --debug 0  --vis_thresh 0.5  --labels 0 1 2 3 4 5  --have_noise False --point_len 5 --hm_gauss 3  \
#                     --gpus 1 --dataset_path ../data/train_val_120.json --data_mode single --psnr 20 \
#                     --commit "train with all data, test with single point data [psnr 20db]" \
#                     --load_model ../exp/ctdet/infrared_point_res50_384_single/logs_2023-10-08-15-57-09/model_best.pth

# python test.py ctdet --exp_id infrared_point_res50_384_single --arch res_50 --dataset point --mse_loss --not_prefetch_test \
#                     --debug 0  --vis_thresh 0.5  --labels 0 1 2 3 4 5  --have_noise False --point_len 5 --hm_gauss 3  \
#                     --gpus 1 --dataset_path ../data/train_val_120.json --data_mode single --psnr 20 \
#                     --commit "train with all data, test with single point data [psnr 20db]" \
#                     --load_model ../exp/ctdet/infrared_point_res50_384_single/logs_2023-10-08-15-57-09/model_best.pth

cd ..
