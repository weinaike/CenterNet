


cd src
# train & test
python main.py ctdet --exp_id ifr_opti_res50_384_single_seed --arch res_50 --dataset point --mse_loss \
                     --batch_size 24 --num_epochs 60 --lr_step '50' --gpus 0 --debug 0 \
                     --hm_weight 1e5 --labels 0 1 2 3 4 5  --have_noise False --noise_sigma 0.02\
                     --sample_num 60000  --hm_gauss 3 --val_intervals 2 \
                     --dataset_path ../data/train_val_1_init_seed.json  --data_mode single --psnr 1000 1001  \
                     --commit "train single data of train_val_1_init_seed with resnet50, using random seed,  without noise very low psnr" \
                     #--load_model ../exp/ctdet/infrared_point_res50_384_single/logs_2023-10-10-07-12-10/model_90.pth --resume
                     #--force_merge_labels --commit "force merge label for all data, just eval local precise"

cd ..


#['main.py', 'ctdet', '--exp_id', 'infrared_point_res50_384_single_seed',
# '--arch', 'res_50', '--dataset', 'point', '--mse_loss', '--batch_size',
# '24', '--num_epochs', '120', '--lr_step', '90', '--gpus', '0', '--debug', '0', 
#'--hm_weight', '1e5', '--labels', '0', '1', '2', '3', '4', '5', '--have_noise', 'False', 
#'--noise_sigma', '0.02', '--sample_num', '60000', '--hm_gauss', '3', '--val_intervals', '2', 
#'--dataset_path', '../data/train_val_90_seed.json', '--data_mode', 'single',
# '--commit', 'train single data with resnet50 [dataset is psf_90_xxxxxxxxxx_init merge background at training, with random seed], without noise']
