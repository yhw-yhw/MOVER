import os
import json
import torch
import numpy as np
from tqdm import tqdm

from ...models import SMPL
from ...core import config
from ...utils.geometry import rotation_matrix_to_angle_axis

def eft_extract(dataset_path, out_path):

    json_files = {
        'coco_2014': 'COCO2014-All-ver01.json',
        'hr-lspet': 'LSPet_ver01.json',
        'mpii': 'MPII_ver01.json',
    } # COCO2014-Part-ver01.json

    img_dir = {
        'coco_2014': 'train2014/',
        'hr-lspet': '',
        'mpii': 'images/',
    }  # COCO2014-Part-ver01.json

    smpl = SMPL(
        config.SMPL_MODEL_DIR,
        batch_size=1,
        create_transl=False
    ).cuda()

    for dataset, json_f in json_files.items():

        # structs we use
        imgnames_, scales_, centers_, parts_, openposes_, poses_, shapes_, joints3d_ = [], [], [], [], [], [], [], []

        print(f'Processing {dataset} / {json_f}')

        annots = json.load(open(os.path.join(dataset_path, json_f)))['data']

        print('Loaded annotations...')

        for ann in tqdm(annots):
            betas = np.array(ann['parm_shape'])
            pose = rotation_matrix_to_angle_axis(
                torch.Tensor(ann['parm_pose'])
            ).reshape(-1).numpy()

            j3d = smpl(
                betas=torch.from_numpy(betas).unsqueeze(0).float().cuda(),
                body_pose=torch.from_numpy(pose[3:]).unsqueeze(0).float().cuda(),
                global_orient=torch.from_numpy(pose[:3]).unsqueeze(0).float().cuda()
            ).joints[0,25:].cpu().numpy()

            S = np.hstack([j3d, np.ones([24, 1])])

            imgnames_.append(img_dir[dataset]+ann['imageName'])
            scales_.append(ann['bbox_scale'])
            centers_.append(np.array(ann['bbox_center']))
            parts_.append(np.array(ann['gt_keypoint_2d'])[25:])
            openposes_.append(np.array(ann['gt_keypoint_2d'])[:25])
            poses_.append(pose)
            shapes_.append(betas)
            joints3d_.append(S)

        # store data
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        out_file = os.path.join(out_path, f'{dataset}_train_eft.npz')
        print(f'Saving {out_file}...')
        np.savez(
            out_file,
            imgname=imgnames_,
            center=centers_,
            scale=scales_,
            pose=poses_,
            shape=shapes_,
            part=parts_,
            openpose=openposes_,
            S=joints3d_,
        )
