cd src
# train
python main.py ctdet --exp_id point_res18_384 --arch res_18 --dataset point --num_epochs 70 --lr_step 45,60 --gpus 0,1 --debug 0
cd ..
