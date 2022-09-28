# Merge OpenPose json files into npy files
# Author: Yuliang Zou
# Date: 03/01/2019

import json
import numpy as np
import os
import pandas as pd

from glob import glob
import sys


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
    root = sys.argv[1] #'/ps/scratch/hyi/HCI_dataset/20210204_capture/C0005/touch_rename_openpose'
    #dump = os.path.join('/ps/scratch/hyi/HCI_dataset/20210204_capture/C0005/ground_detect', root.split('/')[-1])
    dump = os.path.join(sys.argv[2], root.split('/')[-1])
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
                #temp[i, :, :] = read_json(filename)
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
            import ipdb; ipdb.set_trace()

        save_name = os.path.join(dump, folder.split('/')[-1]+'.npy')
        print(f'save to {save_name}')
        np.save(save_name, temp)

