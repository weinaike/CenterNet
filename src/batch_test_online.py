import test

import os
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # basic experiment setting
    parser.add_argument('-g','--gpu', default=0, type=int, help='')
    parser.add_argument('-c', '--complete',  action='store_true', help='force_merge_labels, label is 0, num is 1')
    args = parser.parse_args()

    gpu = args.gpu
    test_dir = 'seed_res50_all_scratch_new'

    train_dirs = {}
    train_dirs['train_val_30_seed.json'] = ["logs_2023-10-19-11-46-23"]
    # train_dirs['train_val_30_seed.json'] = ["logs_2023-10-14-21-27-50", "logs_2023-10-15-03-34-39"]
    # train_dirs['train_val_60_seed.json'] = ["logs_2023-10-15-00-31-30", "logs_2023-10-15-08-29-46"]
    # train_dirs['train_val_90_seed.json'] = ["logs_2023-10-14-21-28-30", "logs_2023-10-15-03-30-16"]
    # train_dirs['train_val_120_seed.json'] = ["logs_2023-10-15-00-28-05", "logs_2023-10-15-07-12-30"]
    datasets = list(train_dirs.keys())

    data_commit_dict =dict()
    #data_commit_dict['singe_-10db_all'] = ['single', -10, 0]
    #data_commit_dict['singe_00db_all'] = ['single', 0, 0]
    #data_commit_dict['singe_10db_all'] = ['single', 10, 0]
    # data_commit_dict['singe_20db_all'] = ['single', 20, 0]
    #data_commit_dict['singe_30db_all'] = ['single', 30, 0]
    #data_commit_dict['singe_40db_all'] = ['single', 40, 0]

    # data_commit_dict['singe_10db_sunny'] = ['single', 10, 0, 'sunny']
    # data_commit_dict['singe_10db_cloudy'] = ['single', 10, 0, 'cloudy']
    # data_commit_dict['singe_05db_sunny'] = ['single', 5, 0, 'sunny']
    # data_commit_dict['singe_05db_cloudy'] = ['single', 5, 0, 'cloudy']

    # data_commit_dict['singe_10db_all'] = ['single', 10, 0]
    # data_commit_dict['singe_20db_all'] = ['single', 20, 0]
    # data_commit_dict['singe_40db_all'] = ['single', 40, 0]
    # data_commit_dict['double_01db_all'] = ['double', 1, 0]
    # data_commit_dict['double_05db_all'] = ['double', 5, 0]
    # data_commit_dict['double_10db_all'] = ['double', 10, 0]
    # data_commit_dict['double_20db_all'] = ['double', 20, 0]
    # data_commit_dict['double_40db_all'] = ['double', 40, 0]
    #data_commit_dict['singe_-10db_sunny'] = ['single', -10, 0, 'sunny']
    #data_commit_dict['singe_-10db_cloudy'] = ['single', -10, 0, 'cloudy']
    # data_commit_dict['singe_0db_sunny'] = ['single', 0, 0, 'sunny']
    #data_commit_dict['singe_10db_sunny'] = ['single', 10, 0, 'sunny']
    # data_commit_dict['singe_0db_cloudy'] = ['single', 0, 0, 'cloudy']
    #data_commit_dict['singe_10db_cloudy'] = ['single', 10, 0, 'cloudy']
    #data_commit_dict['singe_20db_sunny'] = ['single', 20, 0, 'sunny']
    #data_commit_dict['singe_20db_cloudy'] = ['single', 20, 0, 'cloudy']
    #data_commit_dict['singe_30db_sunny'] = ['single', 30, 0, 'sunny']
    #data_commit_dict['singe_30db_cloudy'] = ['single', 30, 0, 'cloudy']
    for px in [10,20,30,40,50,60,70,80,90,100,110,120,130]:
        data_commit_dict['double_20db_{:03d}px'.format(px)] = ['double', 20, px]

    if args.complete:
        for i in range(len(datasets)):
            for j in range(1):
                for key,val in data_commit_dict.items():
                    commit = '{} for {}'.format(key, train_dirs[datasets[i]][j])
                    model_file = '../exp/ctdet/{}/{}/model_best.pth'.format(test_dir, train_dirs[datasets[i]][j])
                    exp_id = 'test_{}_{}'.format(test_dir, train_dirs[datasets[i]][j])
                    bg = 'all'
                    if len(val)>3:
                        bg = val[3]
                    cmd = "python test.py ctdet --exp_id {} --gpus {} --dataset_path '../data/{}' --data_mode {} --psnr {} --dist {} --bg {} --commit '{}' --load_model '{}'  \
                        --arch res_50 --dataset point --mse_loss --not_prefetch_test --debug 0  --vis_thresh 0.5  --labels 0 1 2 3 4 5  \
                        --have_noise False --point_len 5 --hm_gauss 3".format(exp_id, gpu, datasets[i], val[0], val[1], val[2], bg, commit, model_file)

                    os.system(cmd)



    keys = list(data_commit_dict.keys())
    all = dict()
    for key in keys:
        all[key] = list()

    for i in range(len(datasets)):
        for j in range(1):
            exp_dir = '../exp/ctdet/test_{}_{}'.format(test_dir, train_dirs[datasets[i]][j])
            sub_dirs = os.listdir(exp_dir)
            for sub_dir in sub_dirs:
                map_file = os.path.join(exp_dir,sub_dir,"map.txt")
                if os.path.exists(map_file):
                    with  open(map_file) as f: 
                        lines = f.readlines()
                    for key in keys:
                        cmap = list()
                        if key in lines[0]:                            
                            for m in range(2,9):
                                line = lines[m]
                                cmap.append(float(line.strip().split('=')[-1]))
                            cmap.append(train_dirs[datasets[i]][j])
                        if len(cmap)>0:
                            all[key].append(cmap)
    

    f = open('../exp/ctdet/result.txt','w')
    for key in keys:
        f.write(key+'\n')
        items = all[key]
        for item in items:
            out = ''
            for val in item:
                out += ' {}'.format(val)
            f.write(out+"\n")
        f.write('\n')



    dist_dict = dict()
    for i in range(len(datasets)):        
        for j in range(1):
            dist_dict[train_dirs[datasets[i]][j]] = list()    
    for key in keys:
        if 'px' in key:
            items = all[key]
            for item in items:            
                dist_dict[item[-1]].append(item[-2])

    for key, val in dist_dict.items():
        f.write(key+'\n')
        out = ''
        for v in val:
            out += ' {}'.format(v)
        f.write(out+"\n")


    for i in range(len(datasets)):
        for j in range(1):
            exp_dir = '../exp/ctdet/test_{}_{}'.format(test_dir, train_dirs[datasets[i]][j])
            sub_dirs = os.listdir(exp_dir)
            sub_dirs.sort()
            for sub_dir in sub_dirs:
                map_file = os.path.join(exp_dir,sub_dir,"map.txt")
                if not os.path.exists(map_file):
                    continue
                with open(map_file) as mf: 
                    lines = mf.readlines()
                f.write(lines[0])
                ths = list()
                aps = list()
                for line in lines:
                    if "thresh" in line:
                        ths.append(float(line.strip().split(':')[-1]))
                    if "Mean AP" in line:
                        aps.append(float(line.strip().split('=')[-1]))
                for th in ths:
                    f.write('{} '.format(th))
                f.write('\n')
                for ap in aps:
                    f.write('{} '.format(ap))
                f.write('\n')

    f.close()
