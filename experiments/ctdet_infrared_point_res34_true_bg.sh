


cd src
# train & test
python main.py ctdet --exp_id infrared_point_res50_384 --arch res_50 --dataset point --mse_loss \
                     --batch_size 64 --num_epochs 50 --lr_step 40 --gpus 0 --debug 0 \
                     --hm_weight 1e5 --labels 0 1 2 3 4 5  --have_noise True --noise_sigma 0.02\
                     --sample_num 60000  --hm_gauss 3 --val_intervals 1 \
                     --dataset_path ../data/PSF0815_6_IR_30_384_PSNR_init_bg/  --data_mode all \
                     --commit "train all data with resnet50 [dataset is PSF0815_6_IR_30_384_PSNR_init_bg], input size is 384, with 70% add 0.02 noise for train , val without noise"
                     #--force_merge_labels --commit "force merge label for all data, just eval local precise"
                     #--load_model ../exp/ctdet/infrared_point_res34_384/logs_2023-08-02-10-01/model_best.pth

cd ..
