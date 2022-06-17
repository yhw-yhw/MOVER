# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp

import json

from collections import namedtuple

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
import h5py

from .utils import smpl_to_openpose

# VH joint definition.
from ...constants import IDX_MAPPING, JOINT_NAMES


Keypoints = namedtuple('Keypoints',
                       ['keypoints', 'gender_gt', 'gender_pd', 'pelvis', 'neck', 'keypoints_3d'])

Keypoints.__new__.__defaults__ = (None,) * len(Keypoints._fields)


def create_dataset(dataset='openpose', data_folder='data', **kwargs):
    if dataset.lower() == 'openpose':
        return OpenPose(data_folder, **kwargs)
    elif dataset.lower() == 'openpose_video':
        return OpenPose_Video(data_folder, **kwargs)
    elif dataset.lower() == 'pose2room':
        print('load Pose2Room dataset!!!')
        return OpenPose_Pose2Room(data_folder, **kwargs)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))


def read_keypoints(keypoint_fn, use_hands=True, use_face=True,
                   use_face_contour=False):
    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)
    # import pdb;pdb.set_trace()
    keypoints = []

    gender_pd = []
    gender_gt = []
    midhip = []
    neck = []
    pose_keypoints_3d = []
    for idx, person_data in enumerate(data['people']):
        body_keypoints = np.array(person_data['pose_keypoints_2d'],
                                  dtype=np.float32)
        body_keypoints = body_keypoints.reshape([-1, 3])
        if use_hands:
            left_hand_keyp = np.array(
                person_data['hand_left_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])
            right_hand_keyp = np.array(
                person_data['hand_right_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])

            body_keypoints = np.concatenate(
                [body_keypoints, left_hand_keyp, right_hand_keyp], axis=0)
        if use_face:
            # TODO: Make parameters, 17 is the offset for the eye brows,
            # etc. 51 is the total number of FLAME compatible landmarks
            face_keypoints = np.array(
                person_data['face_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])[17: 17 + 51, :]

            contour_keyps = np.array(
                [], dtype=body_keypoints.dtype).reshape(0, 3)
            if use_face_contour:
                contour_keyps = np.array(
                    person_data['face_keypoints_2d'],
                    dtype=np.float32).reshape([-1, 3])[:17, :]

            body_keypoints = np.concatenate(
                [body_keypoints, face_keypoints, contour_keyps], axis=0)

        if 'gender_pd' in person_data:
            gender_pd.append(person_data['gender_pd'])
        if 'gender_gt' in person_data:
            gender_gt.append(person_data['gender_gt'])

        keypoints.append(body_keypoints)

        midhip.append(person_data['pose_keypoints_3d'][4*8+0:4*8+3])
        neck.append(person_data['pose_keypoints_3d'][4*1+0:4*1+3])
        pose_keypoints_3d.append(person_data['pose_keypoints_3d'])
        # import pdb;pdb.set_trace()
    return Keypoints(keypoints=keypoints, gender_pd=gender_pd,
                     gender_gt=gender_gt, pelvis=midhip, neck=neck, 
                     keypoints_3d=pose_keypoints_3d)
    
def read_keypoints_VH(skeleton_joints, use_hands=True, use_face=True,
                   use_face_contour=False):
    
    smplx_joints_num = len(JOINT_NAMES)
    vh_2_smplx_joints = np.zeros((skeleton_joints.shape[0], \
                        smplx_joints_num, 4))
    # vh_2_smplx_joints[:, :, :-1] = skeleton_joints[:, IDX_MAPPING]
    # vh_2_smplx_joints[np.array(IDX_MAPPING)==-1] *= 0.0
    cnt = 0
    print('load kpts from VH !!!!!')
    for idx, map_i in enumerate(IDX_MAPPING):
        if map_i != -1:
            vh_2_smplx_joints[:, map_i, :-1] = skeleton_joints[:, idx]
            vh_2_smplx_joints[:, map_i, -1] += 1.0

        # if map_i != -1 and idx in valid_joint_ids:
        #     cnt += 1
        #     vh_2_smplx_joints[:, map_i, :-1] = skeleton_joints[:, cnt]
        #     vh_2_smplx_joints[:, map_i, -1] += 1.0

    # import pdb;pdb.set_trace()
    keypoints = []
    gender_pd = []
    gender_gt = []
    midhip = []
    neck = []
    pose_keypoints_3d = []
    # hand_left_keypoints_3d = []
    # right_left_keypoints_3d = []
    
    for idx, person_data in enumerate([vh_2_smplx_joints]):
        # assert use_hands == True and use_face == False
        body_keypoints = person_data[:, 23:53]
        keypoints.append(body_keypoints) # this is for hand.
        midhip.append(person_data[:, 4*0+0:4*0+3])
        neck.append(person_data[:, 4*12+0:4*12+3])
        pose_keypoints_3d.append(person_data) # all joints.
        
    return Keypoints(keypoints=keypoints, gender_pd=gender_pd,
                     gender_gt=gender_gt, pelvis=midhip, neck=neck, 
                     keypoints_3d=pose_keypoints_3d)


class OpenPose(Dataset):

    NUM_BODY_JOINTS = 25
    NUM_HAND_JOINTS = 20

    def __init__(self, data_folder, img_folder='images',
                 keyp_folder='keypoints',
                 use_hands=False,
                 use_face=False,
                 dtype=torch.float32,
                 model_type='smplx',
                 joints_to_ign=None,
                 use_face_contour=False,
                 openpose_format='coco25',
                 **kwargs):
        super(OpenPose, self).__init__()

        self.use_hands = use_hands
        self.use_face = use_face
        self.model_type = model_type
        self.dtype = dtype
        self.joints_to_ign = joints_to_ign
        self.use_face_contour = use_face_contour

        self.openpose_format = openpose_format

        self.num_joints = (self.NUM_BODY_JOINTS +
                           2 * self.NUM_HAND_JOINTS * use_hands)

        self.img_folder = osp.join(data_folder, img_folder)
        self.keyp_folder = osp.join(data_folder, keyp_folder)
        self.img_paths = [osp.join(self.img_folder, img_fn)
                          for img_fn in os.listdir(self.img_folder)
                          if img_fn.endswith('.png') or
                          img_fn.endswith('.jpg') or
                          img_fn.endswith('.bmp') and
                          not img_fn.startswith('.')]
        self.img_paths = sorted(self.img_paths)
        self.cnt = 0

    def get_model2data(self):
        return smpl_to_openpose(self.model_type, use_hands=self.use_hands,
                                use_face=self.use_face,
                                use_face_contour=self.use_face_contour,
                                openpose_format=self.openpose_format)

    def get_left_shoulder(self):
        return 2

    def get_right_shoulder(self):
        return 5

    def get_joint_weights(self):
        # The weights for the joint terms in the optimization
        optim_weights = np.ones(self.num_joints + 2 * self.use_hands +
                                self.use_face * 51 +
                                17 * self.use_face_contour,
                                dtype=np.float32)

        # Neck, Left and right hip
        # These joints are ignored because SMPL has no neck joint and the
        # annotation of the hips is ambiguous.
        if self.joints_to_ign is not None and -1 not in self.joints_to_ign:
            optim_weights[self.joints_to_ign] = 0.
        return torch.tensor(optim_weights, dtype=self.dtype)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        return self.read_item(img_path)

    def read_item(self, img_path):
        img = cv2.imread(img_path).astype(np.float32)[:, :, ::-1] / 255.0
        img_fn = osp.split(img_path)[1]
        img_fn, _ = osp.splitext(osp.split(img_path)[1])

        keypoint_fn = osp.join(self.keyp_folder,
                               img_fn + '_keypoints.json')
        keyp_tuple = read_keypoints(keypoint_fn, use_hands=self.use_hands,
                                    use_face=self.use_face,
                                    use_face_contour=self.use_face_contour)

        if len(keyp_tuple.keypoints) < 1:
            return {}
        keypoints = np.stack(keyp_tuple.keypoints)

        output_dict = {'fn': img_fn,
                       'img_path': img_path,
                       'keypoints': keypoints, 'img': img, 
                       'pelvis': keyp_tuple.pelvis, 'neck': keyp_tuple.neck, 
                       'keypoints_3d': keyp_tuple.keypoints_3d}
        if keyp_tuple.gender_gt is not None:
            if len(keyp_tuple.gender_gt) > 0:
                output_dict['gender_gt'] = keyp_tuple.gender_gt
        if keyp_tuple.gender_pd is not None:
            if len(keyp_tuple.gender_pd) > 0:
                output_dict['gender_pd'] = keyp_tuple.gender_pd
        return output_dict

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.cnt >= len(self.img_paths):
            raise StopIteration

        img_path = self.img_paths[self.cnt]
        self.cnt += 1

        return self.read_item(img_path)


class OpenPose_Pose2Room(Dataset):

    NUM_BODY_JOINTS = 25
    NUM_HAND_JOINTS = 20

    # import load dataset for optimization
    def __init__(self, data_folder, img_folder='images',
                 keyp_folder='keypoints',
                 use_hands=False,
                 use_face=False,
                 dtype=torch.float32,
                 model_type='smplx',
                 joints_to_ign=None,
                 use_face_contour=False,
                 openpose_format='coco25',
                 **kwargs):
        super(OpenPose_Pose2Room, self).__init__()
        self.use_hands = use_hands
        self.use_face = use_face
        self.model_type = model_type
        self.dtype = dtype
        self.joints_to_ign = joints_to_ign
        self.use_face_contour = use_face_contour

        self.openpose_format = openpose_format

        self.num_joints = (self.NUM_BODY_JOINTS +
                           2 * self.NUM_HAND_JOINTS * use_hands)

        self.img_folder = osp.join(data_folder, '.')
        
        # get all obj path.
        self.img_paths = [osp.join(self.img_folder, img_fn)
                          for img_fn in os.listdir(self.img_folder)
                          if img_fn.endswith('.hdf5') ]
        self.img_paths = sorted(self.img_paths)

        self.cnt = 0
        # import pdb;pdb.set_trace()

        self.single = kwargs.get('single')

    def get_model2data(self):
        # return smpl_to_openpose(self.model_type, use_hands=self.use_hands,
        #                         use_face=self.use_face,
        #                         use_face_contour=self.use_face_contour,
        #                         openpose_format=self.openpose_format)
        return None

    def get_left_shoulder(self):
        return 2

    def get_right_shoulder(self):
        return 5

    def get_joint_weights(self):
        # The weights for the joint terms in the optimization
        optim_weights = np.ones(self.num_joints + 2 * self.use_hands +
                                self.use_face * 51 +
                                17 * self.use_face_contour,
                                dtype=np.float32)

        # Neck, Left and right hip
        # These joints are ignored because SMPL has no neck joint and the
        # annotation of the hips is ambiguous.
        if self.joints_to_ign is not None and -1 not in self.joints_to_ign:
            optim_weights[self.joints_to_ign] = 0.
        return torch.tensor(optim_weights, dtype=self.dtype)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # one sequence is a video.
        assert len(idx) == 1
        img_path = self.img_paths[idx[0]]
        return self.read_item(img_path)

    def read_item(self, sample_file):
        
        sample_data = h5py.File(sample_file, "r")
        room_bbox = {}
        for key in sample_data['room_bbox'].keys():
            room_bbox[key] = sample_data['room_bbox'][key][:]
        skeleton_joints = sample_data['skeleton_joints'][:]
        
        object_nodes = []
        for idx in range(len(sample_data['object_nodes'])):
            object_node = {}
            node_data = sample_data['object_nodes'][str(idx)]
            for key in node_data.keys():
                if node_data[key].shape is None:
                    continue
                object_node[key] = node_data[key][:]
            object_nodes.append(object_node)
        
        
        # face and contour
        keyp_tuple = read_keypoints_VH(skeleton_joints, use_hands=self.use_hands,
                                    use_face=self.use_face,
                                    use_face_contour=self.use_face_contour)
        if self.single:
            batch_size = 1
        else:
            batch_size = keyp_tuple.keypoints[0].shape[0]
        
        
        sample_fn = [osp.split(sample_file)[1] for i in range(batch_size)]
        
        # change tmp path.
        tmp_img_fn = \
            '/ps/scratch/hyi/HCI_dataset/holistic_scene_human/smplifyx_test/00001/00/images/000001.jpg'
        img_fn = [tmp_img_fn for i in range(batch_size)]
        
        if len(keyp_tuple.keypoints) < 1:
            return {}
        # keypoints = np.stack(keyp_tuple.keypoints)
        assert len(keyp_tuple.keypoints) == 1
        keypoints = np.zeros((keyp_tuple.keypoints[0].shape[0], 68, 3))
        
        # import pdb;pdb.set_trace()
        # body & hand
        # keypoints_3d = np.concatenate([keyp_tuple.keypoints_3d[0], \
        #         keyp_tuple.keypoints[0]], 1)

        # All Skeletons.
        keypoints_3d = keyp_tuple.keypoints_3d[0]
        
        # import pdb;pdb.set_trace()
        if self.single:
            output_dict = {'fn': sample_fn,
                       'img_path': img_fn,
                       'keypoints': keypoints[:1], 
                       'pelvis': keyp_tuple.pelvis[:1], 
                       'neck': keyp_tuple.neck[:1], 
                       'keypoints_3d': keypoints_3d[:1]}
        else:
            output_dict = {'fn': sample_fn,
                       'img_path': img_fn,
                       'keypoints': keypoints, 
                       'pelvis': keyp_tuple.pelvis, 
                       'neck': keyp_tuple.neck, 
                       'keypoints_3d': keypoints_3d}
        if False:
            import pdb;pdb.set_trace()
            for tmp_i in range(len(img_fn)):
                save_name = os.path.basename(img_fn[tmp_i])
                tmp_save_name = os.path.join('/is/cluster/work/hyi/results/SceneGeneration/Pose2Room',\
                     save_name+f'{tmp_i}.ply')
                import trimesh
                output_ply = trimesh.Trimesh(keypoints_3d[tmp_i:tmp_i+1].reshape(-1, 4)[:, :-1], process=False)
                output_ply.export(tmp_save_name)
        print('kspt3d dim:', keypoints_3d.shape)
        return output_dict

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.cnt >= len(self.img_paths):
            raise StopIteration

        img_path = self.img_paths[self.cnt]
        self.cnt += 1

        return self.read_item(img_path)


class OpenPose_Video(OpenPose):

    NUM_BODY_JOINTS = 25
    NUM_HAND_JOINTS = 20

    def __init__(self, data_folder,
                 img_folder='images',
                 keyp_folder='keypoints',
                 use_hands=False,
                 use_face=False,
                 dtype=torch.float32,
                 model_type='smplx',
                 joints_to_ign=None,
                 use_face_contour=False,
                 openpose_format='coco25',
                 **kwargs):
        super(OpenPose, self).__init__()

        self.use_hands = use_hands
        self.use_face = use_face
        self.model_type = model_type
        self.dtype = dtype
        self.joints_to_ign = joints_to_ign
        self.use_face_contour = use_face_contour

        self.openpose_format = openpose_format

        self.num_joints = (self.NUM_BODY_JOINTS +
                           2 * self.NUM_HAND_JOINTS * use_hands)

        self.data_folder = data_folder
        self.img_folder = img_folder
        self.keyp_folder = keyp_folder
        # self.img_paths = [osp.join(self.img_folder, img_fn)
        #                   for img_fn in os.listdir(self.img_folder)
        #                   if img_fn.endswith('.png') or
        #                   img_fn.endswith('.jpg') or
        #                   img_fn.endswith('.bmp') and
        #                   not img_fn.startswith('.')]
        # self.img_paths = sorted(self.img_paths)

    def __getitem__(self, img_list):

        batch_size = len(img_list)
        all_input_list = []
        for idx, img_idx in enumerate(img_list):
            if type(img_idx) == int:
                img_idx = f'{img_idx:06d}'
                one_input = self.read_item(img_idx)
                all_input_list.append(one_input)

        # collate fn
        all_input_batch = self.collate_fn(all_input_list)
        return all_input_batch

    def collate_fn(self, input_list):
        # print(len(input_list))
        result = {}
        for one in input_list:
            # print(one.keys())
            for key, value in one.items():
                if key in result:
                    # print(key)
                    # # import pdb;pdb.set_trace()
                    # if 'keypoints' == key:
                    #     print(key, value.shape)
                    # elif 'keypoints_3d' == key:
                    #     print(key, len(value))

                    # TODO: check why exists 1x17x3, 1x0x3 kpts
                    if key == 'keypoints' and value.shape[1] != 118: 
                        result[key].append(np.zeros((1, 118, 3)))
                    else:
                        result[key].append(value)
                else:
                    print(f'key {key} not in one item')
                    if key == 'keypoints' and value.shape[1] != 118: 
                        result[key] = [np.zeros((1, 118, 3))]
                    else:
                        result[key] = [value]

        new_result = {}
        for key, value in result.items():
            # print(f'{key}')
            # import pdb;pdb.set_trace()
            if key not in ['fn', 'img_path', 'img']:
                # print(f'{key}: {value[0].shape}, {value[1].shape}')
                new_result[key] = np.concatenate(value, axis=0)
                
            else:
                new_result[key] = value
        return new_result

    def read_item(self, img_idx):
        img_path = os.path.join(self.data_folder, img_idx, '00', self.img_folder, img_idx+'.jpg')
        
        # img = cv2.imread(img_path).astype(np.float32)[:, :, ::-1] / 255.0
        
        img_fn = img_idx

        keypoint_fn = osp.join(self.data_folder, img_idx, '00', self.keyp_folder,
                               img_fn + '_keypoints.json')
        keyp_tuple = read_keypoints(keypoint_fn, use_hands=self.use_hands,
                                    use_face=self.use_face,
                                    use_face_contour=self.use_face_contour)

        if len(keyp_tuple.keypoints) < 1:
            return {}
        keypoints = np.stack(keyp_tuple.keypoints)

        #'img': img,
        output_dict = {'fn': img_fn,
                       'img_path': img_path,
                       'keypoints': keypoints,  
                       'pelvis': keyp_tuple.pelvis, 'neck': keyp_tuple.neck, 
                       'keypoints_3d': keyp_tuple.keypoints_3d}
        if keyp_tuple.gender_gt is not None:
            if len(keyp_tuple.gender_gt) > 0:
                output_dict['gender_gt'] = keyp_tuple.gender_gt
        if keyp_tuple.gender_pd is not None:
            if len(keyp_tuple.gender_pd) > 0:
                output_dict['gender_pd'] = keyp_tuple.gender_pd
        return output_dict



