


cd src
# train & test
python main.py ctdet --exp_id ifr_opti_res50_384_single_seed --arch res_50 --dataset point --mse_loss \
                     --batch_size 24 --num_epochs 70 --lr_step '50' --gpus 1 --debug 0 \
                     --hm_weight 1e5 --labels 0 1 2 3 4 5  --have_noise False --noise_sigma 0.02\
                     --sample_num 60000  --hm_gauss 3 --val_intervals 2 \
                     --dataset_path ../data/train_val_90_seed.json  --data_mode single \
                     --commit "train single data with resnet50 [dataset is train_val_90_seed merge background at training], without noise" \
                     --load_model ../models/model_sup30_psnr1000_50.pth
                     #--force_merge_labels --commit "force merge label for all data, just eval local precise"

cd ..