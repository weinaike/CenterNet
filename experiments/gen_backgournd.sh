###########生成训练数据############
cd src/lib/utils/
python BackgroundSample.py --otf_file ../../../data/PSF0815_6_IR_30_init.npy \
                         --save_path ../../../data/PSF0815_6_IR_30_init_background --jobs 10
cd ../../../

cd src/lib/utils/
python BackgroundSample.py --otf_file ../../../data/PSF0815_6_IR_30_init.npy \
                         --save_path ../../../data/PSF0815_6_IR_30_init_background_val --jobs 10
cd ../../../
