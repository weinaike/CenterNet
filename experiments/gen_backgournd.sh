###########生成训练数据############
cd src/lib/utils/

python BackgroundSample.py --otf_file ../../../data/PSF1110_6_IR_120_init.npy \
                          --save_path ../../../data/PSF1110_6_IR_120_init_background_val --jobs 10



python BackgroundSample.py --otf_file ../../../data/PSF1110_6_IR_120_init.npy \
                          --save_path ../../../data/PSF1110_6_IR_120_init_background --jobs 10


python BackgroundSample.py --otf_file ../../../data/PSF1110_6_IR_120_final.npy \
                          --save_path ../../../data/PSF1110_6_IR_120_final_background_val --jobs 10


python BackgroundSample.py --otf_file ../../../data/PSF1110_6_IR_120_final.npy \
                          --save_path ../../../data/PSF1110_6_IR_120_final_background --jobs 10



cd ../../../
