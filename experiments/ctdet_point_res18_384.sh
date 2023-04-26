cd src
# train
# python main.py ctdet --exp_id point_res18_384 --arch res_34 --dataset point --mse_loss \
#                      --batch_size 64 --num_epochs 70 --lr_step 45,60 --gpus -1 --debug 5 \

# test
python test.py ctdet --exp_id point_res18_384 --arch res_18 --dataset point --load_model ../exp/ctdet/point_res18_384/model_best.pth  --not_prefetch_test --debug 0

cd ..
