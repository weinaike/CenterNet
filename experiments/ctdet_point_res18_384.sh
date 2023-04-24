cd src
# train
python main.py ctdet --exp_id point_res18_384 --arch res_18 --dataset point --batch_size 64 --num_epochs 70 --lr_step 45,60 --gpus  --debug 0 --resume 
# test
# python test.py ctdet --exp_id point_res18_384 --arch res_18 --dataset point --resume --not_prefetch_test --debug 4

cd ..
