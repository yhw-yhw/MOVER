import os
os.environ["CDF_LIB"] = "/ps/scratch/mkocabas/data/cdf-dist-all/cdf37_1-dist/src/lib"

import sys
import cv2
import glob
import h5py
import torch
import argparse
import numpy as np
import pickle as pkl
from tqdm import tqdm
from spacepy import pycdf
# from .read_openpose import read_openpose
# from utils.geometry import batch_rodrigues, batch_rot2aa

joints_to_use = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 37])
joints_to_use = np.arange(0,156).reshape((-1,3))[joints_to_use].reshape(-1)

# Illustrative script for training data extraction
# No SMPL parameters will be included in the .npz file.
def h36m_train_extract(dataset_path, openpose_path, out_path, extract_img=False):

    # convert joints to global order
    h36m_idx = [11, 6, 7, 8, 1, 2, 3, 12, 24, 14, 15, 17, 18, 19, 25, 26, 27]
    global_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]

    # structs we use
    imgnames_, scales_, centers_, parts_, Ss_, openposes_  = [], [], [], [], [], []
    poses_, shapes_ = [], []
    # users in validation set
    user_list = [1, 5, 6, 7, 8]

    # go over each user
    for user_i in tqdm(user_list):
        user_name = 'S%d' % user_i
        print(user_name)
        # path with GT bounding boxes
        # path with GT bounding boxes
        bbox_path = os.path.join(dataset_path, 'annotations', user_name, 'MySegmentsMat', 'ground_truth_bb')
        # path with GT 3D pose
        pose_path = os.path.join(dataset_path, 'annotations', user_name, 'MyPoseFeatures', 'D3_Positions_mono')
        # path with GT 2D pose
        pose2d_path = os.path.join(dataset_path, 'annotations', user_name, 'MyPoseFeatures', 'D2_Positions')
        # path with videos
        # vid_path = os.path.join(dataset_path, user_name, 'Videos')

        # path with videos
        imgs_path = os.path.join(dataset_path, 'Images')
        # path with mosh data
        mosh_path = os.path.join(dataset_path, 'mosh', user_name)

        # go over all the sequences of each user
        seq_list = glob.glob(os.path.join(pose_path, '*.cdf'))
        seq_list.sort()
        for seq_i in tqdm(seq_list):
            # sequence info
            seq_name = seq_i.split('/')[-1]
            action, camera, _ = seq_name.split('.')
            action = action.replace(' ', '_')
            # irrelevant sequences
            if action == '_ALL':
                continue

            # 3D pose file
            poses_3d = pycdf.CDF(seq_i)['Pose'][0]

            # 2D pose file
            pose2d_file = os.path.join(pose2d_path, seq_name)
            poses_2d = pycdf.CDF(pose2d_file)['Pose'][0]

            # bbox file
            bbox_file = os.path.join(bbox_path, seq_name.replace('cdf', 'mat'))
            bbox_h5py = h5py.File(bbox_file)

            mosh_file = os.path.join(mosh_path, f'{action.replace("_", " ")}_poses.pkl')

            try:
                thetas, betas = read_mosh_data(mosh_file, cam=camera)
            except FileNotFoundError:
                print(f'{mosh_file} not found!!!')
                continue

            # video file
            # if extract_img:
            #     vid_file = os.path.join(vid_path, seq_name.replace('cdf', 'mp4'))
            #     imgs_path = os.path.join(dataset_path, 'images')
            #     vidcap = cv2.VideoCapture(vid_file)

            # go over each frame of the sequence
            for frame_i in tqdm(range(poses_3d.shape[0])):
                # read video frame
                # if extract_img:
                #     success, image = vidcap.read()
                #     if not success:
                #         break

                # check if you can keep this frame
                # if frame_i % 5 == 0 and (protocol == 1 or camera == '60457274'):
                # image name
                # imgname = '%s_%s.%s_%06d.jpg' % (user_name, action, camera, frame_i+1)
                imgname = os.path.join(dataset_path, 'Images', user_name, f'{action}.{camera}',
                                       f'{frame_i + 1:06d}.jpg')
                if user_name == 'S1' and 'TakingPhoto' in action:
                    temp_action = action.replace('TakingPhoto', 'Photo')
                    imgname = os.path.join(dataset_path, 'Images', user_name, f'{temp_action}.{camera}',
                                           f'{frame_i + 1:06d}.jpg')

                if user_name == 'S1' and 'WalkingDog' in action:
                    temp_action = action.replace('WalkingDog', 'WalkDog')
                    imgname = os.path.join(dataset_path, 'Images', user_name, f'{temp_action}.{camera}',
                                           f'{frame_i + 1:06d}.jpg')


                if not os.path.isfile(imgname):
                    print(imgname)
                    raise FileNotFoundError
                # save image
                if extract_img:
                    img_out = os.path.join(imgs_path, imgname)
                    cv2.imwrite(img_out, image)

                # read GT bounding box
                mask = bbox_h5py[bbox_h5py['Masks'][frame_i,0]].value.T
                ys, xs = np.where(mask==1)
                bbox = np.array([np.min(xs), np.min(ys), np.max(xs)+1, np.max(ys)+1])
                center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
                scale = 0.9*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200.

                # read GT 3D pose
                partall = np.reshape(poses_2d[frame_i,:], [-1,2])
                part17 = partall[h36m_idx]
                part = np.zeros([24,3])
                part[global_idx, :2] = part17
                part[global_idx, 2] = 1

                # read GT 3D pose
                Sall = np.reshape(poses_3d[frame_i,:], [-1,3])/1000.
                S17 = Sall[h36m_idx]
                S17 -= S17[0] # root-centered
                S24 = np.zeros([24,4])
                S24[global_idx, :3] = S17
                S24[global_idx, 3] = 1

                # TODO: run openpose for h36m
                # read openpose detections
                # json_file = os.path.join(openpose_path, 'coco',
                #     imgname.replace('.jpg', '_keypoints.json'))
                # openpose = read_openpose(json_file, part, 'h36m')
                openpose = np.zeros([25, 3])

                # read mosh data
                beta = betas[frame_i]
                theta = thetas[frame_i]

                # store data
                imgnames_.append(imgname)
                centers_.append(center)
                scales_.append(scale)
                parts_.append(part)
                Ss_.append(S24)
                openposes_.append(openpose)
                poses_.append(theta)
                shapes_.append(beta)

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'h36m_train.npz')
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       part=parts_,
                       S=Ss_,
                       pose=poses_,
                       shape=shapes_,
                       openpose=openposes_)

def load_camera_params( hf, path ):
    """Load h36m camera parameters
    Args
    hf: hdf5 open file with h36m cameras data
    path: path or key inside hf to the camera we are interested in
    Returns
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
    f: (scalar) Camera focal length
    c: 2x1 Camera center
    k: 3x1 Camera radial distortion coefficients
    p: 2x1 Camera tangential distortion coefficients
    name: String with camera id
    """

    R = hf[ path.format('R') ][:]
    R = R.T

    T = hf[ path.format('T') ][:]
    f = hf[ path.format('f') ][:]
    c = hf[ path.format('c') ][:]
    k = hf[ path.format('k') ][:]
    p = hf[ path.format('p') ][:]

    name = hf[ path.format('Name') ][:]
    name = "".join( [chr(item) for item in name] )

    return R, T, f, c, k, p, name


def load_cameras(bpath='/ps/scratch/mkocabas/data/h36m/cameras.h5', subjects=[1,5,6,7,8,9,11]):
    """Loads the cameras of h36m
    Args
    bpath: path to hdf5 file with h36m camera data
    subjects: List of ints representing the subject IDs for which cameras are requested
    Returns
    rcams: dictionary of 4 tuples per subject ID containing its camera parameters for the 4 h36m cams
    """
    rcams = {}

    with h5py.File(bpath,'r') as hf:
        for s in subjects:
            for c in range(4): # There are 4 cameras in human3.6m
                rcams[c+1] = load_camera_params(hf, 'subject%d/camera%d/{0}' % (s,c+1) )

    return rcams


def read_pkl(f):
    return pkl.load(open(f, 'rb'), encoding='latin1')


def read_mosh_data(f, frame_skip=4, cam=None):
    data = read_pkl(f)
    poses = data['pose_est_fullposes'][0::frame_skip,joints_to_use]
    shape = data['shape_est_betas'][:10]
    shape = np.tile(shape, [poses.shape[0], 1])

    camera_ids = {
        '54138969': 1,
        '55011271': 2,
        '58860488': 3,
        '60457274': 4,
    }

    rcams = load_cameras()
    camR = rcams[camera_ids[cam]][0]

    poses = torch.from_numpy(poses)
    camR = torch.from_numpy(camR).unsqueeze(0)

    poses[:,:3] = rectify_pose(camR, poses[:,:3])

    return poses.numpy(), shape


def rectify_pose(camera_r, body_aa):
    body_r = batch_rodrigues(body_aa).reshape(-1,3,3)
    final_r = camera_r @ body_r
    body_aa = batch_rot2aa(final_r)
    return body_aa