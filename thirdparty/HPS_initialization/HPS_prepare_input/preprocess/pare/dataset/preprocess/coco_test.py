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

def coco_extract(dataset_path, out_path):

    # structs we use
    imgnames_, scales_, centers_ = [], [], []

    scale_factor = 1.2

    # json annotation file
    json_path = os.path.join(dataset_path,
                             'annotations',
                             'person_keypoints_val2014.json')
    json_data = json.load(open(json_path, 'r'))

    print('loaded json annotations')
    imgs = {}
    for img in json_data['images']:
        imgs[img['id']] = img

    for annot in json_data['annotations']:
        # keypoints processing
        # keypoints = annot['keypoints']
        # keypoints = np.reshape(keypoints, (17, 3))
        # keypoints[keypoints[:, 2] > 0, 2] = 1
        # # check if all major body joints are annotated
        # if sum(keypoints[5:, 2] > 0) < 12:
        #     continue
        # image name
        image_id = annot['image_id']
        img_name = str(imgs[image_id]['file_name'])
        img_name_full = join('val2014', img_name)

        # scale and center
        bbox = annot['bbox']

        center = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]
        scale = scale_factor * max(bbox[2], bbox[3]) / 200

        if max(bbox[2], bbox[3]) < 100:
            continue

        # store data
        imgnames_.append(img_name_full)
        centers_.append(center)
        scales_.append(scale)

    print('done!')
        # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    out_file = os.path.join(out_path, 'coco_test.npz')
    np.savez(
        out_file,
        imgname=imgnames_,
        center=centers_,
        scale=scales_,
    )
