


cd src
# train & test
python main.py ctdet --exp_id infrared_point_res101_384 --arch res_101 --dataset point --mse_loss \
                     --batch_size 64 --num_epochs 100 --lr_step 80 --gpus 0,1 --debug 0 \
                     --hm_weight 1e5 --labels 0 1 2 3 4 5  --have_noise False --noise_sigma 0.02\
                     --sample_num 60000  --hm_gauss 3 --val_intervals 5 \
                     --dataset_path ../data/train_val.json  --data_mode single \
                     --commit "train all data with resnet50 [dataset is PSF0815_6_IR_30_384_nobg merge background at training]"
                     #--force_merge_labels --commit "force merge label for all data, just eval local precise"
                     #--load_model ../exp/ctdet/infrared_point_res34_384/logs_2023-08-02-10-01/model_best.pth

cd ..
