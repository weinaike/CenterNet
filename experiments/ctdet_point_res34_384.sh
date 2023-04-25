cd src
# train
python main.py ctdet --exp_id point_res34_384_mesloss --arch res_34 --dataset point --mse_loss \
                     --batch_size 64 --num_epochs 70 --lr_step 45,60 --gpus 0  --debug 0 \
                     --hm_weight 1000

# test
# python test.py ctdet --exp_id point_res34_384 --arch res_34 --dataset point --resume --not_prefetch_test --debug 5

cd ..
