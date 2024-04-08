


cd src
# train & test
python main.py ctdet --exp_id seed_res50_all --arch res_50 --dataset point --mse_loss \
                     --batch_size 64 --num_epochs 50 --lr_step '40' --gpus 1 --debug 0 \
                     --hm_weight 1e5 --labels 0 1 2 3 4 5  --have_noise False --noise_sigma 0.02\
                     --sample_num 60000  --hm_gauss 3 --val_intervals 2 \
                     --dataset_path ../data/train_val_90_seed.json  --data_mode all  \
                     --commit "train all data [train_val_90_seed] with resnet50 psnr=10 20 40, without noise" \
                     --load_model ../models/psnr1000/model_epoch_30_sup90.pth
                     #--force_merge_labels --commit "force merge label for all data, just eval local precise"

python main.py ctdet --exp_id seed_res50_all --arch res_50 --dataset point --mse_loss \
                     --batch_size 64 --num_epochs 50 --lr_step '40' --gpus 1 --debug 0 \
                     --hm_weight 1e5 --labels 0 1 2 3 4 5  --have_noise False --noise_sigma 0.02\
                     --sample_num 60000  --hm_gauss 3 --val_intervals 2 \
                     --dataset_path ../data/train_val_120_seed.json  --data_mode all  \
                     --commit "train all data [train_val_120_seed] with resnet50 psnr=10 20 40, without noise" \
                     --load_model ../models/psnr1000/model_epoch_30_sup120.pth
                     #--force_merge_labels --commit "force merge label for all data, just eval local precise"

python main.py ctdet --exp_id seed_res50_all --arch res_50 --dataset point --mse_loss \
                     --batch_size 64 --num_epochs 50 --lr_step '40' --gpus 1 --debug 0 \
                     --hm_weight 1e5 --labels 0 1 2 3 4 5  --have_noise False --noise_sigma 0.02\
                     --sample_num 60000  --hm_gauss 3 --val_intervals 2 \
                     --dataset_path ../data/train_val_90_seed.json  --data_mode all  \
                     --commit "train all data [train_val_90_seed] with resnet50 psnr=10 20 40, without noise" \
                     --load_model ../models/psnr1000/model_epoch_30_sup90.pth
                     #--force_merge_labels --commit "force merge label for all data, just eval local precise"

python main.py ctdet --exp_id seed_res50_all --arch res_50 --dataset point --mse_loss \
                     --batch_size 64 --num_epochs 50 --lr_step '40' --gpus 1 --debug 0 \
                     --hm_weight 1e5 --labels 0 1 2 3 4 5  --have_noise False --noise_sigma 0.02\
                     --sample_num 60000  --hm_gauss 3 --val_intervals 2 \
                     --dataset_path ../data/train_val_120_seed.json  --data_mode all  \
                     --commit "train all data [train_val_120_seed] with resnet50 psnr=10 20 40, without noise" \
                     --load_model ../models/psnr1000/model_epoch_30_sup120.pth
                     #--force_merge_labels --commit "force merge label for all data, just eval local precise"
cd ..
