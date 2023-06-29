cd src
# test
python show.py ctdet --exp_id infrared_point_res34_384 --arch res_34  \
                    --debug 0  --vis_thresh 0.5  --labels 1 3 5 \
                    --load_model ../exp/ctdet/infrared_point_res34_384/logs_2023-06-28-17-27/model_best.pth


cd ..


#8192: