


cd src
# train & test
#python main.py ctdet --exp_id seed_res50_all_scratch --arch res_50 --dataset point --mse_loss \
#                     --batch_size 64 --num_epochs 50 --lr_step '40' --gpus 0 --debug 0 \
#                     --hm_weight 1e5 --labels 0 1 2 3 4 5  --have_noise False --noise_sigma 0.02\
#                     --sample_num 60000  --hm_gauss 3 --val_intervals 2 \
#                     --dataset_path ../data/train_val_30_seed.json  --data_mode all  \
#                     --commit "scratch , train all data [train_val_30_seed] with resnet50 psnr=10 20 40, without noise" \
#                     #--load_model ../models/psnr1000/model_epoch_30_sup30.pth
#                     #--force_merge_labels --commit "force merge label for all data, just eval local precise"
#
#python main.py ctdet --exp_id seed_res50_all_scratch --arch res_50 --dataset point --mse_loss \
#                     --batch_size 64 --num_epochs 50 --lr_step '40' --gpus 0 --debug 0 \
#                     --hm_weight 1e5 --labels 0 1 2 3 4 5  --have_noise False --noise_sigma 0.02\
#                     --sample_num 60000  --hm_gauss 3 --val_intervals 2 \
#                     --dataset_path ../data/train_val_60_seed.json  --data_mode all  \
#                     --commit "scratch, train all data [train_val_60_seed] with resnet50 psnr=10 20 40, without noise" \
#                     #--load_model ../models/psnr1000/model_epoch_30_sup60.pth
#                     #--force_merge_labels --commit "force merge label for all data, just eval local precise"
#
#python main.py ctdet --exp_id seed_res50_all_scratch --arch res_50 --dataset point --mse_loss \
#                     --batch_size 64 --num_epochs 50 --lr_step '40' --gpus 0 --debug 0 \
#                     --hm_weight 1e5 --labels 0 1 2 3 4 5  --have_noise False --noise_sigma 0.02\
#                     --sample_num 60000  --hm_gauss 3 --val_intervals 2 \
#                     --dataset_path ../data/train_val_30_seed.json  --data_mode all  \
#                     --commit "scratch, train all data [train_val_30_seed] with resnet50 psnr=10 20 40, without noise" \
#                     #--load_model ../models/psnr1000/model_epoch_30_sup30.pth
#                     #--force_merge_labels --commit "force merge label for all data, just eval local precise"

python main.py ctdet --exp_id res50_all_scratch --arch res_50 --dataset point --mse_loss \
                     --batch_size 24 --num_epochs 40 --lr_step '30' --gpus 0 --debug 0 \
                     --hm_weight 1e5 --labels 0 1 2 3 4 5  --have_noise False --noise_sigma 0.02\
                     --sample_num 60000  --hm_gauss 3 --val_intervals 2 \
                     --dataset_path ../data/train_val_120.json  --data_mode all  \
                     --commit "scratch, train all data [train_val_120] with resnet50 psnr=10 20 40, without noise, random init" \
                     #--load_model ../exp/ctdet/seed_res50_all_scratch/logs_2023-10-15-06-36-01/model_last.pth --resume
                     #--force_merge_labels --commit "force merge label for all data, just eval local precise"

cd ..
