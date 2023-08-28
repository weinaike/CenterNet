###########生成训练数据############
cd src/lib/utils/
python PointSample.py --otf_file ../../../data/PSF0815_6_IR_30_init.npy --point_type gauss --weight_mode rand \
                        --point_len 5 --noise_sigma 0.01 --labels 0 1 2 3 4 5 --merge_bg --psnr 10 \
                        --save_path ../../../data/PSF0815_6_IR_30_psnr_10_384_init_bg --num 20000 --true_bg
cd ../../../


cd src/lib/utils/
python PointSample.py --otf_file ../../../data/PSF0815_6_IR_30_init.npy --point_type gauss --weight_mode rand \
                        --point_len 5 --noise_sigma 0.01 --labels 0 1 2 3 4 5 --merge_bg --psnr 20 \
                        --save_path ../../../data/PSF0815_6_IR_30_psnr_20_384_init_bg --num 20080 --true_bg
cd ../../../


cd src/lib/utils/
python PointSample.py --otf_file ../../../data/PSF0815_6_IR_30_init.npy --point_type gauss --weight_mode rand \
                        --point_len 5 --noise_sigma 0.01 --labels 0 1 2 3 4 5 --merge_bg --psnr 40 \
                        --save_path ../../../data/PSF0815_6_IR_30_psnr_40_384_init_bg --num 20000 --true_bg
cd ../../../
###########生成训练集json文件############
cd data
python parse_target_json.py --path PSF0815_6_IR_30_384_PSNR_init_bg --mode train
cd ../

###########生成验证集数据############
cd src/lib/utils/
python PointSample.py --otf_file ../../../data/PSF0815_6_IR_30_init.npy --point_type gauss --weight_mode rand \
                        --point_len 5 --noise_sigma 0.01 --labels 0 1 2 3 4 5 --merge_bg --psnr 10 \
                        --save_path ../../../data/PSF0815_6_IR_30_psnr_10_384_init_bg_val --num 2000 --true_bg
cd ../../../


cd src/lib/utils/
python PointSample.py --otf_file ../../../data/PSF0815_6_IR_30_init.npy --point_type gauss --weight_mode rand \
                        --point_len 5 --noise_sigma 0.01 --labels 0 1 2 3 4 5 --merge_bg --psnr 20 \
                        --save_path ../../../data/PSF0815_6_IR_30_psnr_20_384_init_bg_val --num 2000 --true_bg
cd ../../../


cd src/lib/utils/
python PointSample.py --otf_file ../../../data/PSF0815_6_IR_30_init.npy --point_type gauss --weight_mode rand \
                        --point_len 5 --noise_sigma 0.01 --labels 0 1 2 3 4 5 --merge_bg --psnr 40 \
                        --save_path ../../../data/PSF0815_6_IR_30_psnr_40_384_init_bg_val --num 2000 --true_bg
cd ../../../


###########生成验证集json文件############
cd data
python parse_target_json.py --path PSF0815_6_IR_30_384_PSNR_init_bg --mode val
cd ../
