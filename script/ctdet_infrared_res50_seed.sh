


cd src
# train & test
python main.py ctdet --exp_id seed_res50_all_1000 --arch res_50 --dataset point --mse_loss \
                     --batch_size 24 --num_epochs 120 --lr_step '90' --gpus 1 --debug 0 \
                     --hm_weight 1e5 --labels 0 1 2 3 4 5  --have_noise False --noise_sigma 0.02\
                     --sample_num 60000  --hm_gauss 3 --val_intervals 2 \
                     --dataset_path ../data/train_val_120_seed.json  --data_mode all --psnr 1000 1001 \
                     --commit "train all data [train_val_120_seed] with resnet50 psnr=1000, without noise" \
                     #  --load_model ../models/model_sup30_psnr1000_50.pth
                     #--force_merge_labels --commit "force merge label for all data, just eval local precise"

cd ..