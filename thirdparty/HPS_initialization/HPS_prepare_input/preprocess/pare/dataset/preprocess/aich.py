import os
import json
import numpy as np
from tqdm import tqdm
from os.path import join

from ...utils.kp_utils import convert_kps
from ...utils.geometry import rotation_matrix_to_angle_axis
from .run_eft import run_eft_step


def get_eft_fitter(split_idx=-1):
    from ...eft.eft_fitter import EFTFitter
    from ...eft.config import get_cfg_defaults
    # Prepare EFT config for running on in the wild images
    hparams = get_cfg_defaults()
    hparams.LOG = True
    hparams.LOG_DIR = 'logs/eft/aich'
    hparams.DATASET.RENDER_RES = hparams.DATASET.IMG_RES
    hparams.LOSS.OPENPOSE_TRAIN_WEIGHT = 0.0
    hparams.DATASET.VAL_DS = f'split_{split_idx}'
    hparams.MIN_EXEMPLAR_ITER = 20
    hparams.MAX_EXEMPLAR_ITER = 500
    hparams.MIN_LOSS = 0.01

    eft_fitter = EFTFitter(hparams=hparams)
    return eft_fitter


def aich_extract(dataset_path, out_path, split_idx=-1, num_splits=1):
    # convert joints to global order
    # joints_idx = [19, 20, 21, 22, 23, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0]

    # bbox expansion factor
    scaleFactor = 1.0

    eft_fitter = get_eft_fitter(split_idx=split_idx)

    # structs we need
    imgnames_, scales_, centers_, parts_, poses_, shapes_, joints3d_ = [], [], [], [], [], [], []

    # json annotation file
    json_path = os.path.join(dataset_path, 'annotations/keypoint_train_annotations.json')
    json_data = json.load(open(json_path, 'r'))

    if num_splits == 1:
        pass
    else:
        json_data = np.array_split(json_data, num_splits)[split_idx]

    counter = 0
    for img_data in tqdm(json_data):
        image_id = img_data['image_id']

        for k,v in img_data['keypoint_annotations'].items():
            keypoints = np.array(v).reshape((-1,3))

            try:
                keypoints = np.reshape(keypoints, (14, 3))
            except:
                print('There is no keypoint for this sample')
                continue

            keypoints[keypoints[:, 2] > 0, 2] = 1
            if sum(keypoints[5:, 2] > 0) < 6:
                continue

            img_name = f'images/train/{image_id}.jpg'

            bbox = np.array(img_data['human_annotations'][k])

            # keypoints
            part = convert_kps(keypoints[None, ...], src='aich', dst='spin')[0]

            center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            scale = scaleFactor * max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200

            eft_output = run_eft_step(
                eft_fitter, join(dataset_path, img_name), part.copy(), counter=counter,
                bbox=np.array([center[0], center[1], max(bbox[2], bbox[3]), max(bbox[2], bbox[3])]),
                debug=True,
            )
            counter += 1

            # import IPython; IPython.embed(); exit()

            pose = rotation_matrix_to_angle_axis(eft_output['pred_pose'][0]).reshape(-1).detach().cpu().numpy()
            betas = eft_output['pred_shape'].reshape(-1).detach().cpu().numpy()
            j3d = eft_output['smpl_joints3d'].detach().cpu().numpy()[0, 25:]

            S = np.hstack([j3d, np.ones([24, 1])])

            # store data
            imgnames_.append(img_name)
            centers_.append(center)
            scales_.append(scale)
            parts_.append(part[25:])

            poses_.append(pose)
            shapes_.append(betas)
            joints3d_.append(S)

    if num_splits == 1:
        # store the data struct
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        out_file = os.path.join(out_path, 'aich_train.npz')
    else:
        # store the data struct
        out_path = out_path + '/aich'
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        out_file = os.path.join(out_path, f'aich_train_{split_idx}-{num_splits}.npz')
    np.savez(
        out_file,
        imgname=imgnames_,
        center=centers_,
        scale=scales_,
        part=parts_,
        pose=poses_,
        shape=shapes_,
        S=joints3d_,
    )