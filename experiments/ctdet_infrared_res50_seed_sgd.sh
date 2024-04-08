


cd src
# train & test
python main.py ctdet --exp_id ifr_opti_res50_384_single_seed_sgd --arch res_50 --dataset point --mse_loss \
                     --batch_size 24 --num_epochs 60 --lr_step '50'  --gpus 0 --debug 0 \
                     --hm_weight 1e5 --labels 0 1 2 3 4 5  --have_noise False --noise_sigma 0.02\
                     --sample_num 60000  --hm_gauss 3 --val_intervals 2 \
                     --dataset_path ../data/train_val_1_init_seed.json  --data_mode single --psnr 1000 1001 --use_swats \
                     --commit "train single data of train_val_1_init_seed with resnet50, using random seed,  without noise very low psnr" \
                    #  --load_model ../models/model_init_1st_psnr1000_50.pth
                     #--force_merge_labels --commit "force merge label for all data, just eval local precise"

cd ..
