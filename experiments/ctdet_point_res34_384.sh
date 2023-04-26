cd src
# train
# python main.py ctdet --exp_id point_res34_384_mesloss --arch res_34 --dataset point --mse_loss \
#                      --batch_size 64 --num_epochs 70 --lr_step 45,60 --gpus 0  --debug 0 \
#                      --hm_weight 1000

# test
python test.py ctdet --exp_id point_res34_384_mesloss --arch res_34 --dataset point --mse_loss --resume --not_prefetch_test \
                    --debug 0 --vis_thresh 0.5 

cd ..
