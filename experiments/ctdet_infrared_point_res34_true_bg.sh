


cd src
# train & test
python main.py ctdet --exp_id infrared_point_res34_384 --arch res_34 --dataset point --mse_loss \
                     --batch_size 64 --num_epochs 50 --lr_step 40 --gpus 0  --debug 0 \
                     --hm_weight 1e5 --labels 0 1 2 3 4 5  --have_noise True --noise_sigma 0.01 \
                     --sample_num 60000  --hm_gauss 3 --val_intervals 1 \
                     --dataset_path ../data/PSF0815_6_IR_30/  --data_mode all --force_merge_labels \
                     --commit "force merge label for all data, just eval local precise"
                    #  --load_model ../exp/ctdet/infrared_point_res34_384/logs_2023-08-02-10-01/model_best.pth

cd ..
