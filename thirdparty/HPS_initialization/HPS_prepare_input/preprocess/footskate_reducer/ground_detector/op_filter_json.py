# Merge OpenPose json files into npy files
# Author: Yuliang Zou
# Date: 03/01/2019

import json
import numpy as np
import os
import pandas as pd

from glob import glob
import sys
from PIL import Image
from viz import vis_frame_fast
import cv2

def read_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    kps = []
    for people in data['people']:
        kp = np.array(people['pose_keypoints_2d']).reshape(-1, 3)
        kps.append(kp)
    # TODO: Support multi-person handling
    # Pick the most confident detection
    scores = [np.mean(kp[kp[:, 2] > -1, 2]) for kp in kps]
    try:
        kp = kps[np.argmax(scores)]
    except:
        kp = np.zeros((25, 3), dtype=np.float32)
    return kp

def save_json(ori_json_path, pose_kpts_2d, save_json_path):
    # import pdb;pdb.set_trace()
    with open(ori_json_path) as f:
        data = json.load(f)
    # print('diff: ')
    # print(np.array(data['people'][0]['pose_keypoints_2d']) - pose_kpts_2d.reshape(-1))
    data['people'][0]['pose_keypoints_2d'] = pose_kpts_2d.reshape(-1).tolist()

    with open(save_json_path, 'w') as fout:
        json.dump(data, fout)
    
    print(f'save to {save_json_path}')
    return data
def transfer_openpose_viz_format(result):
    pose_2d = np.array(result['people'][0]['pose_keypoints_2d']).reshape(-1, 3)
    
    save_result = {'result': [{
        'keypoints': pose_2d[:, :-1],
        'kp_score': pose_2d[:, -1],
    }]}
    return save_result


# NOTE: One Euro Filter
import math

class LowPassFilter(object):
    def __init__(self, alpha):
        self.__setAlpha(alpha)
        self.__y = self.__s = None

    def __setAlpha(self, alpha):
        alpha = float(alpha)
        if alpha<=0 or alpha>1.0:
            raise ValueError("alpha (%s) should be in (0.0, 1.0]"%alpha)
        self.__alpha = alpha

    def __call__(self, value, timestamp=None, alpha=None):        
        if alpha is not None:
            self.__setAlpha(alpha)
        if self.__y is None:
            s = value
        else:
            s = self.__alpha*value + (1.0-self.__alpha)*self.__s
        self.__y = value
        self.__s = s
        return s

    def lastValue(self):
        return self.__y


class OneEuroFilter(object):
    def __init__(self, freq, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        if freq<=0:
            raise ValueError("freq should be >0")
        if mincutoff<=0:
            raise ValueError("mincutoff should be >0")
        if dcutoff<=0:
            raise ValueError("dcutoff should be >0")
        self.__freq = float(freq)
        self.__mincutoff = float(mincutoff)
        self.__beta = float(beta)
        self.__dcutoff = float(dcutoff)
        self.__x = LowPassFilter(self.__alpha(self.__mincutoff))
        self.__dx = LowPassFilter(self.__alpha(self.__dcutoff))
        self.__lasttime = None
        
    def __alpha(self, cutoff):
        te    = 1.0 / self.__freq
        tau   = 1.0 / (2*np.pi*cutoff)
        return  1.0 / (1.0 + tau/te)

    def __call__(self, x, timestamp=None):
        # ---- update the sampling frequency based on timestamps
        if self.__lasttime and timestamp:
            self.__freq = 1.0 / (timestamp-self.__lasttime)
        self.__lasttime = timestamp
        # ---- estimate the current variation per second
        prev_x = self.__x.lastValue()
        dx = 0.0 if prev_x is None else (x-prev_x)*self.__freq # FIXME: 0.0 or value?
        edx = self.__dx(dx, timestamp, alpha=self.__alpha(self.__dcutoff))
        # ---- use it to update the cutoff frequency
        cutoff = self.__mincutoff + self.__beta*np.abs(edx)
        # ---- filter the given value
        return self.__x(x, timestamp, alpha=self.__alpha(cutoff))



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process openpose json through OneEuro Filter')
    parser.add_argument('--root', type=str, help='an integer for the accumulator')
    parser.add_argument('--dump', type=str, help='an integer for the accumulator')
    parser.add_argument('--img_dir', type=str, help='an integer for the accumulator')
    parser.add_argument('--viz',
                        type=lambda arg: arg.lower() == 'true',
                        default=False,
                        help='Print info messages during the process')


    args = parser.parse_args()

    # import pdb;pdb.set_trace()
    # root = sys.argv[1] #'/ps/scratch/hyi/HCI_dataset/20210204_capture/C0005/touch_rename_openpose'
    # dump = os.path.join(sys.argv[2], root.split('/')[-1])
    root = args.root
    dump = os.path.join(args.dump, root.split('/')[-1]) # ! warnings
    # dump = args.dump
    img_dir = args.img_dir
    viz = args.viz
    
    if not os.path.exists(dump):
        os.makedirs(dump)

    # freq and dcutoff are default values
    euro_config = {
        'freq': 120,
        'mincutoff': 1.0,    # 1.7
        'beta': 0.3,    # 0.3
        'dcutoff': 1.0
    }

    #folders = glob(os.path.join(root, '*'))
    folders = [root]
    for folder in folders:
        filenames = sorted(glob(os.path.join(folder, '*json')))
        num = len(filenames)
        temp = np.zeros((num, 25, 3), dtype=np.float32)
        for i, filename in enumerate(filenames):
            print(filename)
            tmp_json = read_json(filename)
            if tmp_json.shape[0] == 0:
                temp[i, :, :] = temp[i-1, :, :]
            else:
                temp[i, :, :] = tmp_json
            
        # Do linear interpolation
        for k in range(25):
            invalid = temp[:, k, 2]==0
            for i in range(2):
                curr = temp[:, k, i]
                curr[invalid] = np.nan
                # Forward interpolation
                ser = pd.Series(curr)
                ser = ser.interpolate()
                curr = ser.values
                # Backward interpolation
                curr = curr[::-1]
                ser = pd.Series(curr)
                ser = ser.interpolate()
                curr = ser.values
                curr = curr[::-1]
                # temp[:, k, i] = curr

                # One Euro Filter
                euro_f = OneEuroFilter(**euro_config)
                num_frames = curr.shape[0]
                for n in range(num_frames):
                    observed = curr[n]
                    filtered = euro_f(observed)
                    temp[n, k, i] = filtered

        if np.sum(np.isnan(temp)) > 0:
            print('error in nan.')
            import ipdb; ipdb.set_trace()

        save_name = os.path.join(dump, folder.split('/')[-1]+'.npy')
        print(f'save to {save_name}')
        np.save(save_name, temp)

        # save to new json
        print('filenames num: ', len(filenames))
        for i, filename in enumerate(filenames):
            basename = os.path.basename(filename)
            save_path = os.path.join(dump, basename)
            # import pdb;pdb.set_trace()
            new_result=save_json(filename, temp[i], save_path)
            
            if viz:
                img_path = os.path.join(img_dir, f'{i+1:06d}.jpg')
                save_img_path = os.path.join(dump, f'{i+1:06d}.jpg')
                img = cv2.imread(img_path)

                # import pdb;pdb.set_trace()
                viz_result = transfer_openpose_viz_format(new_result)
                save_img = vis_frame_fast(img, viz_result)
                cv2.imwrite(save_img_path, save_img)

