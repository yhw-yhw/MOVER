import pickle
import os
import numpy as np
from tqdm import tqdm
import sys

def split_all_plk(pkl_file):
    with open(pkl_file, 'rb') as fin:
        all_pkl = pickle.load(fin)

    save_dir = os.path.join(os.path.dirname(pkl_file), 'split')
    # import pdb;pdb.set_trace()
    os.makedirs(save_dir, exist_ok=True)
    length = all_pkl['global_orient'].shape[0]
    for i in tqdm(range(length)):
        tmp_file = os.path.join(save_dir, f'{i:06d}.pkl')
        tmp_dict = {}
        for key in all_pkl.keys():
            if key in ['betas', 'gender']:
                tmp_dict[key] = all_pkl[key]
            
            else:
                tmp_dict[key] = all_pkl[key][i:i+1]
        # print(f'save to {tmp_file}')
        with open(tmp_file, 'wb') as fout:
            pickle.dump(tmp_dict, fout)

if __name__ == '__main__':
    # pkl_file='/ps/scratch/hyi/HCI_dataset/20210209_experiments/PROX_sample/N3OpenArea_00157_02/0428_results/img_-1_1010_28.13.27.42.723611897_motion_smooth_new_camera/results/001_all.pkl'
    pkl_file = sys.argv[1]
    split_all_plk(pkl_file)
