import os
import cv2
import json
import torch
import pickle
import numpy as np
from tqdm import tqdm

from ...utils.kp_utils import convert_kps


def muco3dhp_extract(dataset_path, out_path, augmented=True):
    data = json.load(open(os.path.join(dataset_path, 'MuCo-3DHP.json')))
    # smpl_params = json.load(open(os.path.join(dataset_path, 'smpl_param.json')))

    image_data = data['images']
    annotations = data['annotations']

    imgnames_, scales_, centers_, parts_, poses_, shapes_, joints3d_, has_smpl_ = [], [], [], [], [], [], [], []

    for annot in tqdm(annotations):
        image_id = annot['image_id']
        img_filename = image_data[image_id]['file_name']

        img_prefix = 'augmented' if augmented else 'unaugmented'

        # filter data to select augmented/unaugmented samples
        if not img_filename.startswith(img_prefix):
            continue

        keypoints_2d = np.array(annot['keypoints_img'])
        keypoints_3d = np.array(annot['keypoints_cam']) / 1000
        keypoint_vis = np.array(annot['keypoints_vis'])[..., None]

        if keypoint_vis.sum() < 6:
            continue

        keypoints_2d = np.hstack([keypoints_2d, keypoint_vis])
        keypoints_2d = convert_kps(keypoints_2d[None, ...], src='muco3dhp', dst='spin')[0, 25:]

        keypoints_3d = np.hstack([keypoints_3d, keypoint_vis])
        keypoints_3d = convert_kps(keypoints_3d[None, ...], src='muco3dhp', dst='spin')[0, 25:]

        bbox = np.array(annot['bbox'])

        # pose = np.array(smpl_params[str(annot['id'])]['pose'])
        # shape = np.array(smpl_params[str(annot['id'])]['shape'])

        # has_smpl_.append(0)

        center = [bbox[0] + bbox[2] / 2., bbox[1] + bbox[3] / 2.]
        scale = max(bbox[2], bbox[3]) / 200

        imgnames_.append(os.path.join('images', img_filename))
        scales_.append(scale)
        centers_.append(center)
        parts_.append(keypoints_2d)
        joints3d_.append(keypoints_3d)
        # poses_.append(pose)
        # shapes_.append(shape)

    # store data
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    out_file = os.path.join(out_path, f'muco3dhp_train.npz')
    print(f'Saving {out_file}...')
    np.savez(
        out_file,
        imgname=imgnames_,
        center=centers_,
        scale=scales_,
        # pose=poses_,
        # shape=shapes_,
        part=parts_,
        S=joints3d_,
        # has_smpl=has_smpl_,
    )