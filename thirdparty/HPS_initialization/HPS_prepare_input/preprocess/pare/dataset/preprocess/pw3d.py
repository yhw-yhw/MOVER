import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)


from ...utils.kp_utils import convert_kps
from ...core.config import DATASET_FILES, DATASET_FOLDERS, MMPOSE_CFG, MMPOSE_CKPT


def pw3d_extract(dataset_path, out_path, debug=False):
    # scale factor
    scaleFactor = 1.2

    pose_model = init_pose_model(MMPOSE_CFG, MMPOSE_CKPT, device='cuda')
    pose_dataset = pose_model.cfg.data['test']['type']

    # structs we use
    imgnames_, scales_, centers_, parts_ = [], [], [], []
    poses_, shapes_, genders_ = [], [], []

    mmpose_keypoints_, mmpose_keypoints_converted_ = [], []

    # get a list of .pkl files in the directory
    files = []
    for split in ['train', 'validation', 'test']:
        split_path = os.path.join(dataset_path, 'sequenceFiles', split)
        files += sorted([os.path.join(split_path, f) for f in os.listdir(split_path) if f.endswith('.pkl')])

    # go through all the .pkl files
    for filename in tqdm(files):
        print(filename)
        with open(filename, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            smpl_pose = data['poses']
            smpl_betas = data['betas']
            poses2d = data['poses2d']
            global_poses = data['cam_poses']
            genders = data['genders']
            valid = np.array(data['campose_valid']).astype(np.bool)
            num_people = len(smpl_pose)
            num_frames = len(smpl_pose[0])
            seq_name = str(data['sequence'])
            img_names = np.array(
                ['imageFiles/' + seq_name + '/image_%s.jpg' % str(i).zfill(5) for i in range(num_frames)])
            # get through all the people in the sequence
            for i in range(num_people):
                valid_pose = smpl_pose[i][valid[i]]
                valid_betas = np.tile(smpl_betas[i][:10].reshape(1, -1), (num_frames, 1))
                valid_betas = valid_betas[valid[i]]
                valid_keypoints_2d = poses2d[i][valid[i]]
                valid_img_names = img_names[valid[i]]
                valid_global_poses = global_poses[valid[i]]
                gender = genders[i]
                # consider only valid frames
                for valid_i in tqdm(range(valid_pose.shape[0])):
                    part = valid_keypoints_2d[valid_i, :, :].T
                    part = part[part[:, 2] > 0, :]
                    bbox = [min(part[:, 0]), min(part[:, 1]), max(part[:, 0]), max(part[:, 1])]
                    center = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2]
                    scale = scaleFactor * max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200

                    # transform global pose
                    pose = valid_pose[valid_i]
                    extrinsics = valid_global_poses[valid_i][:3, :3]
                    pose[:3] = cv2.Rodrigues(np.dot(extrinsics, cv2.Rodrigues(pose[:3])[0]))[0].T[0]

                    ######### MMPOSE #########
                    bbox.append(1.)
                    bbox = np.array(bbox)[None, ...]

                    pose_img_name = os.path.join(DATASET_FOLDERS['3dpw'], valid_img_names[valid_i])
                    pose_results = inference_top_down_pose_model(
                        pose_model,
                        pose_img_name,
                        bbox,
                        bbox_thr=0.3,
                        format='xyxy',
                        dataset=pose_dataset
                    )

                    if len(pose_results) > 1:
                        print('More than 1 pose result is detected')
                        breakpoint()

                    if debug and valid_i % 100 == 0:
                        # show the results
                        vis_pose_result(
                            pose_model,
                            pose_img_name,
                            pose_results,
                            dataset=pose_dataset,
                            kpt_score_thr=0.3,
                            show=True,
                            out_file=None,
                        )

                    # convert to SPIN joint format
                    joints2d = pose_results[0]['keypoints'][:23][None, ...]
                    joints2d = convert_kps(joints2d=joints2d, src='mmpose', dst='spin_op')[0]
                    mmpose_j2d = pose_results[0]['keypoints']

                    mmpose_keypoints_converted_.append(joints2d)
                    mmpose_keypoints_.append(mmpose_j2d)

                    ################################

                    imgnames_.append(valid_img_names[valid_i])
                    centers_.append(center)
                    scales_.append(scale)
                    poses_.append(pose)
                    shapes_.append(valid_betas[valid_i])
                    genders_.append(gender)

    # store data
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path,
                            '3dpw_all_test_with_mmpose.npz')
    np.savez(
        out_file,
        imgname=imgnames_,
        center=centers_,
        scale=scales_,
        pose=poses_,
        shape=shapes_,
        gender=genders_,
        mmpose_keypoints=mmpose_keypoints_,
        openpose=mmpose_keypoints_converted_,
    )
