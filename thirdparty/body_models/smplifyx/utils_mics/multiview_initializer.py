import numpy as np
import copy
import os.path as osp
try:
    import cPickle as pickle
except ImportError:
    import pickle

def find_prev_result(result_path, curr, n):

    try:
        if int(curr) == 0:
            return None
    except ValueError:  # if result_folder is not ended with numbers
        return None

    prev = str(int(curr)-n).zfill(len(curr))
    prev_res = '{}'.format(result_path.replace(curr, prev))
    if osp.exists(prev_res):
        return prev_res
    else:
        return None

class MultiViewInitializer(object):
    def __init__(self, dataset_obj, result_path, time_str='', n=1, **kwargs):
        super(MultiViewInitializer, self).__init__()

        if result_path:
            curr = time_str.split('-')[-1].split('_')[-1]
            prev_res = find_prev_result(result_path, curr, n)

        self.init_dict = dict()
        # initialize with prev. frame result
        # if there's no prev result, initialize the translation from OpenPose
        if not result_path or not prev_res:
            for cam in range(len(dataset_obj)):
                pelvis = dataset_obj[cam].get('pelvis', None)
                if pelvis:
                    if pelvis[0]:
                        self.init_dict['transl'] = np.array(dataset_obj[cam]['pelvis'][0]).astype(np.float32)
                        self.init_dict['keypoints_3d'] = np.array(dataset_obj[cam]['keypoints_3d'][0]).astype(np.float32).reshape(-1,4)
                        break
            if 'transl' not in self.init_dict:
                self.init_dict['transl'] = np.array([0., 0., 0.]).astype(np.float32)
        else:
            with open(prev_res,'rb') as f:
                self.init_dict = pickle.load(f)

    def get_global_transl(self):
        return self.init_dict['transl']

    def get_init_params(self):
        return self.init_dict


# TODO: find out where to find it.
class VideoInitializer(object):
    def __init__(self, dataset_obj, result_path, time_str='', n=1, **kwargs):
        super(VideoInitializer, self).__init__()

        # import pdb;pdb.set_trace()
        if result_path:
            curr = time_str.split('-')[-1].split('_')[-1]
            prev_res = find_prev_result(result_path, curr, n)

        batch_size = len(dataset_obj['fn'])
        self.init_dict = dict()
        # initialize with prev. frame result
        # if there's no prev result, initialize the translation from OpenPose
        if not result_path or not prev_res:
            pelvis = dataset_obj.get('pelvis', None)
            if pelvis is not None:
                self.init_dict['transl'] = np.array(dataset_obj['pelvis']).astype(np.float32)
                
                # import pdb;pdb.set_trace()
                # if dataset_obj['keypoints_3d'].shape[1] > 25: # denotes add hands into 3d joints.
                #     # self.init_dict['keypoints_3d'] = np.array(dataset_obj['keypoints_3d'][:, :23]).astype(np.float32).reshape(batch_size, -1, 4)
                #     # self.init_dict['left_hand_keypoints_3d'] = np.array(dataset_obj['keypoints_3d'][:, 23:38]).astype(np.float32).reshape(batch_size, -1, 4)
                #     # self.init_dict['right_hand_keypoints_3d'] = np.array(dataset_obj['keypoints_3d'][:, 38:]).astype(np.float32).reshape(batch_size, -1, 4)
                # else:
                self.init_dict['keypoints_3d'] = np.array(dataset_obj['keypoints_3d']).astype(np.float32).reshape(batch_size, -1, 4)

            if 'transl' not in self.init_dict:
                self.init_dict['transl'] = np.zeros((batch_size, 3)).astype(np.float32)
        else:
            with open(prev_res,'rb') as f:
                self.init_dict = pickle.load(f)

    def get_global_transl(self):
        return self.init_dict['transl']

    def get_init_params(self):
        return self.init_dict