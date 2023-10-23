


cd src
# train & test
python main.py ctdet --exp_id seed_res50_all_scratch_new --arch res_50 --dataset point --mse_loss \
                    --batch_size 64 --num_epochs 50 --lr_step '40' --gpus 0 --debug 0 \
                    --hm_weight 1e5 --labels 0 1 2 3 4 5  --have_noise False --noise_sigma 0.02 \
                    --sample_num 60000  --hm_gauss 3 --val_intervals 2 \
                    --dataset_path ../data/train_val_30_seed.json  --data_mode all  --psnr -10 10 20 40 \
                    --commit "scratch from random, train all data [train_val_30_seed] with resnet50 psnr=-10 10 20 40, without noise"
                    #--load_model ../models/psnr1000/model_epoch_30_sup30.pth
                    #--force_merge_labels --commit "force merge label for all data, just eval local precise"

# python main.py ctdet --exp_id seed_res50_all_scratch_new --arch res_50 --dataset point --mse_loss \
#                     --batch_size 64 --num_epochs 50 --lr_step '40' --gpus 0 --debug 0 \
#                     --hm_weight 1e5 --labels 0 1 2 3 4 5  --have_noise False --noise_sigma 0.02 \
#                     --sample_num 60000  --hm_gauss 3 --val_intervals 2 \
#                     --dataset_path ../data/train_val_30_seed.json  --data_mode all  --psnr 1 5 10 20 40 \
#                     --commit "scratch imagenet, train all data [train_val_30_seed] with resnet50 psnr=1 10 20 40, without noise" \
#                     #--load_model ../models/psnr1000/model_epoch_30_sup30.pth
#                     #--force_merge_labels --commit "force merge label for all data, just eval local precise"


# python main.py ctdet --exp_id seed_res50_all_scratch_new --arch res_50 --dataset point --mse_loss \
#                     --batch_size 64 --num_epochs 50 --lr_step '40' --gpus 0 --debug 0 \
#                     --hm_weight 1e5 --labels 0 1 2 3 4 5  --have_noise False --noise_sigma 0.02 \
#                     --sample_num 60000  --hm_gauss 3 --val_intervals 2 \
#                     --dataset_path ../data/train_val_60_seed.json  --data_mode all  --psnr 1 5 10 20 40 \
#                     --commit "scratch imagenet, train all data [train_val_60_seed] with resnet50 psnr=1 10 20 40, without noise" \
#                     #--load_model ../models/psnr1000/model_epoch_30_sup30.pth
#                     #--force_merge_labels --commit "force merge label for all data, just eval local precise"

# python main.py ctdet --exp_id seed_res50_all_scratch_new --arch res_50 --dataset point --mse_loss \
#                     --batch_size 64 --num_epochs 50 --lr_step '40' --gpus 0 --debug 0 \
#                     --hm_weight 1e5 --labels 0 1 2 3 4 5  --have_noise False --noise_sigma 0.02 \
#                     --sample_num 60000  --hm_gauss 3 --val_intervals 2 \
#                     --dataset_path ../data/train_val_90_seed.json  --data_mode all  --psnr 1 5 10 20 40 \
#                     --commit "scratch imagenet, train all data [train_val_90_seed] with resnet50 psnr=1 10 20 40, without noise" \
#                     #--load_model ../models/psnr1000/model_epoch_30_sup30.pth
#                     #--force_merge_labels --commit "force merge label for all data, just eval local precise"


# python main.py ctdet --exp_id seed_res50_all_scratch_new --arch res_50 --dataset point --mse_loss \
#                     --batch_size 64 --num_epochs 50 --lr_step '40' --gpus 0 --debug 0 \
#                     --hm_weight 1e5 --labels 0 1 2 3 4 5  --have_noise False --noise_sigma 0.02 \
#                     --sample_num 60000  --hm_gauss 3 --val_intervals 2 \
#                     --dataset_path ../data/train_val_120_seed.json  --data_mode all  --psnr 1 5 10 20 40 \
#                     --commit "scratch imagenet, train all data [train_val_120_seed] with resnet50 psnr=1 10 20 40, without noise" \
#                     #--load_model ../models/psnr1000/model_epoch_30_sup30.pth
#                     #--force_merge_labels --commit "force merge label for all data, just eval local precise"


cd ..
