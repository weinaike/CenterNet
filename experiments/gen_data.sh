###########生成训练数据############
cd src/lib/utils/
python PointSample.py --otf_file ../../../data/PSF0815_6_IR_30.npy --point_type ones_rand --weight_mode rand \
                        --point_len 5 --noise_sigma 0.01 --labels 0 1 2 3 4 5 --merge_bg --nsr 0.02 \
                        --save_path ../../../data/PSF0815_6_IR_30_nsr_0.02 --num 20000 --true_bg
cd ../../../


cd src/lib/utils/
python PointSample.py --otf_file ../../../data/PSF0815_6_IR_30.npy --point_type ones_rand --weight_mode rand \
                        --point_len 5 --noise_sigma 0.01 --labels 0 1 2 3 4 5 --merge_bg --nsr 0.1 \
                        --save_path ../../../data/PSF0815_6_IR_30_nsr_0.1 --num 20000 --true_bg
cd ../../../


cd src/lib/utils/
python PointSample.py --otf_file ../../../data/PSF0815_6_IR_30.npy --point_type ones_rand --weight_mode rand \
                        --point_len 5 --noise_sigma 0.01 --labels 0 1 2 3 4 5 --merge_bg --nsr 0.2 \
                        --save_path ../../../data/PSF0815_6_IR_30_nsr_0.2 --num 20000 --true_bg
cd ../../../
###########生成训练集json文件############
cd data
python parse_target_json.py --path PSF0815_6_IR_30 --mode train
cd ../

###########生成验证集数据############
cd src/lib/utils/
python PointSample.py --otf_file ../../../data/PSF0815_6_IR_30.npy --point_type ones_rand --weight_mode rand \
                        --point_len 5 --noise_sigma 0.01 --labels 0 1 2 3 4 5 --merge_bg --nsr 0.02 \
                        --save_path ../../../data/PSF0815_6_IR_30_nsr_0.02_val --num 2000 --true_bg
cd ../../../


cd src/lib/utils/
python PointSample.py --otf_file ../../../data/PSF0815_6_IR_30.npy --point_type ones_rand --weight_mode rand \
                        --point_len 5 --noise_sigma 0.01 --labels 0 1 2 3 4 5 --merge_bg --nsr 0.1 \
                        --save_path ../../../data/PSF0815_6_IR_30_nsr_0.1_val --num 2000 --true_bg
cd ../../../


cd src/lib/utils/
python PointSample.py --otf_file ../../../data/PSF0815_6_IR_30.npy --point_type ones_rand --weight_mode rand \
                        --point_len 5 --noise_sigma 0.01 --labels 0 1 2 3 4 5 --merge_bg --nsr 0.2 \
                        --save_path ../../../data/PSF0815_6_IR_30_nsr_0.2_val --num 2000 --true_bg
cd ../../../


###########生成验证集json文件############
cd data
python parse_target_json.py --path PSF0815_6_IR_30 --mode val
cd ../