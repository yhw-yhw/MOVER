import os
import json
import torch
import numpy as np
from tqdm import tqdm

from ...models import SMPL
from ...core import config
from ...utils.kp_utils import convert_kps
from ...utils.geometry import batch_rodrigues, batch_rot2aa

# Dictionary keys
# extri (4, 4)
# intri (3, 3)
# img_path ()
# bbox (2, 2)
# betas (1, 10)
# pose (1, 72)
# trans (1, 3)
# scale (1,)
# smpl_joints_2d (24, 2)
# smpl_joints_3d (24, 3)
# lsp_joints_2d (14, 2)
# lsp_joints_3d (14, 3)
# mask_path ()

scaleFactor = 1.2

def oh3d_train_extract(dataset_path, out_path):

    json_f = 'train/annots.json'

    smpl = SMPL(
        config.SMPL_MODEL_DIR,
        batch_size=1,
        create_transl=False
    ).cuda()

    # structs we use
    imgnames_, scales_, centers_, parts_, openposes_, poses_, shapes_, joints3d_ = [], [], [], [], [], [], [], []

    annots = json.load(open(os.path.join(dataset_path, json_f)))

    print('Loaded annotations...')

    for key,ann in tqdm(annots.items()):
        pose = np.array(ann['pose'])

        pose = torch.from_numpy(pose)
        camR = torch.from_numpy(np.array(ann['extri']))[:3,:3].unsqueeze(0)

        pose[:, :3] = rectify_pose(camR, pose[:, :3])

        betas = np.array(ann['betas'])

        j3d = smpl(
            betas=torch.from_numpy(betas).float().cuda(),
            body_pose=pose[:, 3:].float().cuda(),
            global_orient=pose[:, :3].float().cuda()
        ).joints[0, 25:].cpu().numpy()
        j3d -= np.array(j3d[2] + j3d[3]) / 2. # root center
        S = np.hstack([j3d, np.ones([24, 1])])

        # S = np.hstack([np.array(ann['smpl_joints_3d']), np.ones([24, 1])])

        bbox = np.array(ann['bbox']).reshape(-1)

        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        scale = scaleFactor * (bbox[3] - bbox[1]) / 200

        # import IPython; IPython.embed(); exit()
        imgnames_.append('train/' + ann['img_path'].replace("\\", '/'))

        j2d = np.expand_dims(np.concatenate([np.array(ann['lsp_joints_2d']), np.ones((14,1))], axis=-1), axis=0)
        j2d = convert_kps(j2d, src='common', dst='spin')[0]

        # import IPython; IPython.embed(); exit()

        scales_.append(scale)
        centers_.append(center)
        # parts_.append(np.concatenate([np.array(ann['smpl_joints_2d']), np.ones((24,1))], axis=-1))
        parts_.append(j2d[25:])
        # openposes_.append(np.zeros([25,3]))
        poses_.append(pose[0].numpy())
        shapes_.append(betas[0])
        joints3d_.append(S)

    # store data
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, f'3doh_train.npz')
    print(f'Saving {out_file}...')
    np.savez(
        out_file,
        imgname=imgnames_,
        center=centers_,
        scale=scales_,
        pose=poses_,
        shape=shapes_,
        part=parts_,
        # openpose=openposes_,
        S=joints3d_,
    )

def rectify_pose(camera_r, body_aa):
    body_r = batch_rodrigues(body_aa).reshape(-1,3,3)
    final_r = camera_r @ body_r
    body_aa = batch_rot2aa(final_r)
    return body_aa

