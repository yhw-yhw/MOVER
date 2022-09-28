import copy
import json
import numpy as np
import os
import sys

from glob import glob

import torch.utils.data as data

from utils import read_json

import ipdb
#np.seterr(all='raise')

## Human3.6M
#Pre-define data split
TRAIN_SUBJECTS = ['S1', 'S6', 'S7', 'S8']
VAL_SUBJECTS = ['S5']
TEST_SUBJECTS = ['S9', 'S11']
id2cam = {0: '54138969', 1: '55011271', 2: '58860488', 3: '60457274'}
mads_id2cam = {0: 'C0', 1: 'C1'}

## OpenPose Body25 index
HIP_IDX = 8
# Remaining keypoints
REST_INDS = list(range(9, 25))
# Foot keypoints: right, left (need to minus 1 because we've removed hip)
FOOT_INDS = [10, 23, 21, 22, 13, 20, 18, 19]

# NOTE: output_mode
# op: pose(2) + conf(1)
# kp: pose(2)

class DatasetHuman36m(data.Dataset):
    """Define dataset class for Human3.6M"""
    def __init__(self, root='../data/Human3.6M', mode='train', time_window=2, pose_norm=True, output_mode='kp'):
        """
        Args:
        mode - Different splits of Human3.6M
        time_window (int) - How many nearby (past or future) frames to see, length=2*time_window+1
        pose_norm - Normalize person height to 150 or not (pose)

        Current stat:
        Train
        Positive: [35368. 33936. 37512. 34080.]
        Negative: [22348. 25412. 22628. 25808.]
        Invalid: [45060. 43428. 42636. 42888.]

        Val
        Positive: [10036. 10040. 11216. 11240.]
        Negative: [5416. 5836. 4868. 5232.]
        Invalid: [15116. 14692. 14484. 14096.]
        """
        if mode == 'train':
            self.subjects = TRAIN_SUBJECTS
        elif mode == 'val':
            self.subjects = VAL_SUBJECTS
        else:
            raise NotImplementedError

        self.mode = mode
        self.time_window = time_window
        self.pose_norm = pose_norm
        self.output_mode = output_mode
        self.path_2d = os.path.join(root, 'openpose', 'openpose_flow')
        self.path_gt = os.path.join(root, 'noisy_ground_2', 'new_labels')
        self._make_index_human36m()


    def _make_index_human36m(self):
        """
        Create a dictionary for this usage:
        Given a key of format (subject, action), find the index range of it
        Left close, right open: [)
        """
        self.key_to_range = {}
        count = 0
        all_npy = glob(os.path.join(self.path_gt, '*npy'))
        for subject in self.subjects:
            for npy in all_npy:
                if subject != npy.split('/')[-1].split('_')[0]:
                    continue
                annot = np.load(npy)

                action = npy.split('_')[-1][:-4]
                # Sanity check: Ground truth 3D pose might have different number of frames with videos
                # And each camera might have different number of frames
                min_frame = annot.shape[0]
                for cam in id2cam:
                    npy_file = os.path.join(self.path_2d, subject+'_'+action+'_{}.npy'.format(id2cam[cam]))
                    if os.path.exists(npy_file):
                        temp = np.load(npy_file)
                        min_frame = min(temp.shape[0], min_frame)
                        flag = True
                    else:
                        flag = False
                        break

                if not flag:
                    continue

                key = (subject, action)
                num = 4*min_frame
                self.key_to_range[key] = (count, count+num)
                count += num
        self.length = count


    def __getitem__(self, index):
        """
        Args:
        index (int) - Index

        Returns:
        pose_2d - (Time x #Joint-1 x 3)
        foot_2d - (Optional) (Time x 8 x 3)
        label - (4, )
        """
        for key in self.key_to_range:
            start, end = self.key_to_range[key]
            if index >= start and index < end:
                num = end-start
                length = num//4
                cam = (index-start)//length
                sub_index = (index-start) % length
                break
        # Load openpose data
        npy_file = os.path.join(self.path_2d, '_'.join(key)+'_'+id2cam[cam]+'.npy')
        pose_2ds = np.load(npy_file)
        select = np.arange(sub_index-self.time_window, sub_index+self.time_window+1)
        # For boundaries, we do replication padding
        select = np.minimum(np.maximum(select, 0), length-1)
        pose_2d = copy.deepcopy(pose_2ds[select, :, :])
        # Hip centered, then remove it from the data
        pose_2d[:, :, :2] -= pose_2ds[sub_index, HIP_IDX, :2]
        pose_2d = pose_2d[:, REST_INDS, :]

        # Compute person height
        temp = pose_2d[self.time_window, :, :]
        select = temp[:, 2] > 0.2
        if np.sum(select) == 0:
            assert False
        temp = temp[select, :2]
        min_pt = np.min(temp, axis=0)
        max_pt = np.max(temp, axis=0)
        person_height = np.linalg.norm(max_pt-min_pt)
        scale = 150./person_height

        # Normalize pose
        if self.pose_norm:
            pose_2d[:, :, :2] *= scale

        # Load ground-contact label
        npy_file = os.path.join(self.path_gt, '{}_{}.npy'.format(key[0], key[1]))
        annot = np.load(npy_file)
        label = annot[sub_index, :].astype(np.float32)
        mask = (label!=-1).astype(np.float32)
        # Make -1 label to 0
        label = label*mask

        if self.output_mode == 'op':
            out = pose_2d[:, :, :3]
        elif self.output_mode == 'kp':
            out = pose_2d[:, :, :2]

        return out, label, mask


    def __len__(self):
        return self.length


    def __repr__(self):
        fmt_str = 'Dataset: Human3.6M\n'
        fmt_str += 'Split: '+self.mode+'\n'
        fmt_str += 'Number of datapoints: '+str(self.length)+'\n'
        return fmt_str


class DatasetMads(data.Dataset):
    """Define dataset class for MADS"""
    def __init__(self, root='data/MADS', mode='train', time_window=2, pose_norm=True, output_mode='kp'):
        """
        Args:
        mode - Different splits of MADS
        time_window (int) - How many nearby (past or future) frames to see, length=2*time_window+1
        pose_norm - Normalize person height to 150 or not (pose)
        out_foot - Output additional foot keypoints or not

        Current stat:
        Train
        Positive: [4256. 4064. 5002. 3628.]
        Negative: [2278. 2588. 2368. 3464.]
        Invalid: [13310. 13192. 12474. 12752.]
        """
        if mode == 'train':
            self.subjects = TRAIN_SUBJECTS
        else:
            raise NotImplementedError

        self.mode = mode
        self.time_window = time_window
        self.pose_norm = pose_norm
        self.output_mode = output_mode
        self.path_2d = os.path.join(root, 'openpose', 'openpose_flow')
        self.path_gt = os.path.join(root, 'noisy_ground_2_mads', 'new_labels')
        self.cams = ['C0', 'C1']
        self._make_index_mads()


    def _make_index_mads(self):
        """
        Create a dictionary for this usage:
        Given a key of format (subject, action), find the index range of it
        Left close, right open: [)
        """
        self.key_to_range = {}
        count = 0
        all_npy = glob(os.path.join(self.path_gt, '*npy'))
        for npy in all_npy:
            annot = np.load(npy)
            action = npy.split('/')[-1][:-4]
            min_frame = annot.shape[0]
            for cam in mads_id2cam:
                name = os.path.join(self.path_2d, '{}_{}.npy'.format(action, mads_id2cam[cam]))
                temp = np.load(name)
                min_frame = min(min_frame, temp.shape[0])

            key = action
            num = 2*min_frame
            self.key_to_range[key] = (count, count+num)
            count += num
        self.length = count


    def __getitem__(self, index):
        """
        Args:
        index (int) - Index

        Returns:
        pose_2d - (Time x #Joint-1 x 3)
        foot_2d - (Optional) (Time x 8 x 3)
        label - (4, )
        """
        for key in self.key_to_range:
            start, end = self.key_to_range[key]
            if index >= start and index < end:
                num = end-start
                length = num//2
                cam = (index-start)//length
                sub_index = (index-start) % length
                break
        # Load openpose data
        npy_file = os.path.join(self.path_2d, key+'_'+self.cams[cam]+'.npy')
        pose_2ds = np.load(npy_file)
        select = np.arange(sub_index-self.time_window, sub_index+self.time_window+1)
        # For boundaries, we do replication padding
        select = np.minimum(np.maximum(select, 0), length-1)
        pose_2d = copy.deepcopy(pose_2ds[select, :, :])
        # Hip centered, then remove it from the data
        pose_2d[:, :, :2] -= pose_2ds[sub_index, HIP_IDX, :2]
        pose_2d = pose_2d[:, REST_INDS, :]

        # Compute person height
        temp = pose_2d[self.time_window, :, :]
        select = temp[:, 2] > 0.2
        if np.sum(select) == 0:
            assert False
        temp = temp[select, :2]
        min_pt = np.min(temp, axis=0)
        max_pt = np.max(temp, axis=0)
        person_height = np.linalg.norm(max_pt-min_pt)
        scale = 150./person_height

        # Normalize pose
        if self.pose_norm:
            pose_2d[:, :, :2] *= scale

        # Load ground-contact label
        npy_file = os.path.join(self.path_gt, '{}.npy'.format(key))
        annot = np.load(npy_file)
        label = annot[sub_index, :].astype(np.float32)
        mask = (label!=-1).astype(np.float32)
        # Make -1 label to 0
        label = label*mask

        if self.output_mode == 'op':
            out = pose_2d[:, :, :3]
        elif self.output_mode == 'kp':
            out = pose_2d[:, :, :2]

        return out, label, mask


    def __len__(self):
        return self.length


    def __repr__(self):
        fmt_str = 'Dataset: MADS\n'
        fmt_str += 'Split: '+self.mode+'\n'
        fmt_str += 'Number of datapoints: '+str(self.length)+'\n'
        return fmt_str


# Concatenate different datasets during training
class DatasetConcat(data.Dataset):
    def __init__(self, dataset_list):
        self.dataset_list = dataset_list
        self.length = 0
        self.dataset_to_range = {}    # dataset_index: [start, end)
        start = -1
        end = 0
        for i, dataset in enumerate(dataset_list):
            start = end
            end = start+len(dataset)
            self.dataset_to_range[i] = (start, end)
            self.length += len(dataset)


    def __getitem__(self, index):
        for key in self.dataset_to_range:
            start, end = self.dataset_to_range[key]
            if index >= start and index < end:
                sub_index = index-start
                break
        return self.dataset_list[key].__getitem__(sub_index)


    def __len__(self):
        return self.length


    def __repr__(self):
        fmt_str = 'A concatenation of datasets, including:\n'
        for dataset in self.dataset_list:
            fmt_str += dataset.__repr__()
            fmt_str += '\n'
        return fmt_str


# For inference time only
# TODO: Support multi-person loading?
class DatasetInference(data.Dataset):
    """A general class for inference time data loading"""
    def __init__(self, root, time_window=2, pose_norm=True, output_mode='kp'):
        """
        Args:
        root - path to OpenPose output directory
        time_window (int) - How many nearby (past or future) frames to see, length=2*time_window+1
        pose_norm - Normalize person height to 150 or not (pose)
        """
        self.root = root
        self.time_window = time_window
        self.pose_norm = pose_norm
        self.output_mode = output_mode
        self._load_openpose_results()


    def _load_openpose_results(self):
        """
        Load all the json files
        """
        self.pose_2ds = np.load(self.root)
        self.length = self.pose_2ds.shape[0]


    def __getitem__(self, index):
        """
        Args:
        index (int) - Index

        Returns:
        pose_2d - (Time x #Joint-1 x 5)
        """
        select = np.arange(index-self.time_window, index+self.time_window+1)
        # For boundaries, we do replication padding
        select = np.minimum(np.maximum(select, 0), self.length-1)
        pose_2d = copy.deepcopy(self.pose_2ds[select, :, :])
        # Hip centered, then remove it from the data
        pose_2d[:, :, :2] -= self.pose_2ds[index, HIP_IDX, :2]
        pose_2d = pose_2d[:, REST_INDS, :]
        #print(pose_2d.shape)

        # Compute person height
        temp = pose_2d[self.time_window, :, :]
        select = temp[:, 2] > 0.2
        #print(temp.shape, select.shape)
        if np.sum(select) == 0:
            #assert False
            select = np.arange(select.shape[0])
        temp = temp[select, :2]
        min_pt = np.min(temp, axis=0)
        max_pt = np.max(temp, axis=0)
        person_height = np.linalg.norm(max_pt-min_pt)
        scale = 150./person_height

        # Normalize pose
        if self.pose_norm:
            pose_2d[:, :, :2] *= scale

        if self.output_mode == 'op':
            out = pose_2d[:, :, :3]
        elif self.output_mode == 'kp':
            out = pose_2d[:, :, :2]

        return out


    def __len__(self):
        return self.length


    def __repr__(self):
        fmt_str = 'Inference Time Dataset\n'
        fmt_str += 'Number of datapoints: '+str(self.length)+'\n'
        return fmt_str


class DatasetContact(data.Dataset):
    """Load the annotated ground-contact dataset (test time only)"""
    def __init__(self, root='data/contact_dataset', time_window=2, pose_norm=True, output_mode='kp'):
        self.vid_dir = os.path.join(root, 'videos')
        self.op_dir = os.path.join(root, 'openpose', 'openpose_flow')
        self.label_dir = os.path.join(root, 'labels')
        self.time_window = time_window
        self.pose_norm = pose_norm
        self.output_mode = output_mode

        self.vid_list = sorted(glob(os.path.join(self.vid_dir, '*mp4')))
        self.vid_idx = 0


    def _load_openpose_results(self, path):
        """
        Load all the json files
        """
        self.pose_2ds = np.load(path)
        length = self.pose_2ds.shape[0]
        return length


    def __getvid__(self):
        """Should call it first, before calling __getitem__"""
        vid = self.vid_list[self.vid_idx]
        op_path = os.path.join(self.op_dir, vid.split('/')[-1][:-4]+'.npy')
        label_path = os.path.join(self.label_dir, vid.split('/')[-1].replace('.mp4', '_ground.npy'))
        self.labels = np.load(label_path)

        num_frames = self._load_openpose_results(op_path)
        self.num_frames = num_frames

        self.vid_idx += 1
        if self.vid_idx >= len(self.vid_list):
            print('Finish loading all the video sequences!')
            self.vid_idx = 0
        return num_frames


    def __getitem__(self, index):
        """
        Args:
        index (int) - Index

        Returns:
        pose_2d - (Time x #Joint-1 x 3)
        foot_2d - (Optional) (Time x 8 x 3)
        """
        select = np.arange(index-self.time_window, index+self.time_window+1)
        # For boundaries, we do replication padding
        select = np.minimum(np.maximum(select, 0), self.num_frames-1)
        pose_2d = copy.deepcopy(self.pose_2ds[select, :, :])
        # Hip centered, then remove it from the data
        pose_2d[:, :, :2] -= self.pose_2ds[index, HIP_IDX, :2]
        pose_2d = pose_2d[:, REST_INDS, :]

        # Compute person height
        temp = pose_2d[self.time_window, :, :]
        select = temp[:, 2] > 0.2
        if np.sum(select) == 0:
            assert False
        temp = temp[select, :2]
        min_pt = np.min(temp, axis=0)
        max_pt = np.max(temp, axis=0)
        person_height = np.linalg.norm(max_pt-min_pt)
        scale = 150./person_height

        # Normalize pose
        if self.pose_norm:
            pose_2d[:, :, :2] *= scale

        label = self.labels[index, :]

        if self.output_mode == 'op':
            out = pose_2d[:, :, :3]
        elif self.output_mode == 'kp':
            out = pose_2d[:, :, :2]

        return out, label


    def __vidlen__(self):
        return len(self.vid_list)


    def __len__(self):
        return self.num_frames


    def __repr__(self):
        fmt_str = 'Ground Contact Dataset, including\n'
        for vid in self.vid_list:
            temp = vid.split('/')[-1]
            fmt_str += temp
        return fmt_str


if __name__ == '__main__':
    dataset1 = DatasetHuman36m(mode='train', output_mode='kp')
    dataset2 = DatasetMads(mode='train')
    dataset_list = [dataset1, dataset2]
    dataset = DatasetConcat(dataset_list)
