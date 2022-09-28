import os
import h5py
import numpy as np
import json
from os.path import join

def get_center_scale(joints, joints_vis):
    vis_joints =np.array([joints[i] for i, v in enumerate(joints_vis) if v > 0])

    ul = np.array([vis_joints[:, 0].min(), vis_joints[:, 1].min()])  # upper left
    lr = np.array([vis_joints[:, 0].max(), vis_joints[:, 1].max()])  # lower right

    w = lr[0] - ul[0]
    h = lr[1] - ul[1]
    c_x, c_y = ul[0] + w / 2, ul[1] + h / 2
    # to keep the aspect ratio
    w = h = np.where(w / h > 1, w, h)
    scale = h/200.

    return [c_x, c_y], scale

def mpii_extract(dataset_path, out_path):

    # structs we use
    imgnames_, scales_, centers_ = [], [], []

    scale_factor = 1.2
    # read annotation files
    annot_file = os.path.join(dataset_path, 'annotations', 'val.json')
    annot = json.load(open(annot_file))

    # go over all annotated examples
    for ann in annot:
        imgname = ann['image']
        # center = ann['center']
        # scale = ann['scale']

        joints = np.array(ann['joints'])
        joints_vis = np.array(ann['joints_vis'])
        center, scale = get_center_scale(joints, joints_vis)
        scale *= scale_factor

        # store data
        imgnames_.append(join('images', imgname))
        centers_.append(center)
        scales_.append(scale)

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    out_file = os.path.join(out_path, 'mpii_test.npz')
    np.savez(
        out_file,
        imgname=imgnames_,
        center=centers_,
        scale=scales_,
    )
