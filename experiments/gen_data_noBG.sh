###########生成训练数据############
cd src/lib/utils/
python PointSample_NoBG.py --otf_file ../../../data/PSF0815_6_IR_30_init.npy --point_type gauss --weight_mode rand \
                        --point_len 5  --labels 0 1 2 3 4 5  --save_path ../../../data/PSF0815_6_IR_30_init_384_nobg --num 20000 --jobs 10
cd ../../../

