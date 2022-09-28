import os
import cv2
import numpy as np
import pickle
import pandas as pd
import torch
from glob import glob
import ipdb
import json

# import sys
# sys.path.append('/home/stripathi/renderpeople_eval2/human_body_prior')
# from human_body_prior.human_body_prior.tools.model_loader import load_vposer

def focalLength_mm2px(focalLength,dslr_sens,  focalPoint):
    focal_pixel = (focalLength / dslr_sens) * focalPoint *2
    return focal_pixel

def get_cam_intrinsics(imgHeight, imgWidth):
    constants = {'focalLength': 50,
                 'dslr_sens_width': 36,
                 'dslr_sens_height': 20.25,
                 'camPosWorld': [0, 0, 170],
                 'camPitch': 0}
    cx, cy = imgWidth / 2, imgHeight / 2
    focalLength_x = focalLength_mm2px(constants['focalLength'], constants['dslr_sens_width'], cx)
    focalLength_y = focalLength_mm2px(constants['focalLength'], constants['dslr_sens_height'], cy)

    camMat = np.array([[focalLength_x, 0, cx],
                       [0, focalLength_y, cy],
                       [0, 0, 1]])

    return camMat

def openpose_json_to_numpy(keypoints, format='body_25'):
    '''
    Convert keypoint from openpose json to either body_25 or coco (17) format
    return: num_people x num_joints x 3(x,y,c)
    '''
    keypoints = keypoints['people']
    keypoints = [k['pose_keypoints_2d'] for k in keypoints]
    if format == 'body_25':
        keypoints = np.array(keypoints).reshape(-1, 25, 3)
    if format == 'coco':
        keypoints = np.array(keypoints).reshape(-1, 18, 3)
    return keypoints

def bbox_from_openpose(keypoints, rescale=1.2, detection_thresh=0.2):
    """Get center and scale for bounding box from openpose detections of a SINGLE person"""
    valid = keypoints[:, -1] > detection_thresh
    if sum(valid)>0:
        valid_keypoints = keypoints[valid][:, :-1]
    else:
        print('careful predicting with low confidence bbx')
        valid_keypoints = keypoints[:, :-1]
    center = valid_keypoints.mean(axis=0)
    center = center
    try:
        bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)).max()
    except:
        import ipdb; ipdb.set_trace()
    # adjust bounding box tightness
    scale = bbox_size / 200.0
    scale *= rescale
    return center, scale

def bbox_from_jointsGT(joints, rescale=1.2):
    joints = joints[:,:-1]
    # center = joints.mean(axis=0)
    bbox = [min(joints[:, 0]), min(joints[:, 1]), max(joints[:, 0]), max(joints[:, 1])]
    center = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2]
    scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200
    # center = (joints.max(axis=0) - joints.min(axis=0))/2
    # bbox_size = (joints.max(axis=0) - joints.min(axis=0)).max()
    # scale = bbox_size
    scale *= rescale
    return center, scale

def get_pose_and_beta_from_smpl_data(smpl_gt, vposer):
    # with torch.no_grad():
    #     pose_embedding = torch.tensor(smpl_gt['pose_embedding'], device='cuda')
    #     body_pose = vposer.decode(
    #         pose_embedding,
    #         output_type='aa').view(1, -1).cpu()
    body_pose = smpl_gt['body_pose']
    betas = smpl_gt['betas']
    return body_pose, betas

def agora_extract(df, out_path, vposer_ckpt, split_num):
    '''
    Load agora images, openpose 2d, GT 3D joints, and GT SMPL pose and beta params,
    and GT mesh vertices
    :param dataset_path: path to the SPIN dataframe (df_SPIN_matched.npy)
    :param out_path:
    :return:
    '''
    ######### TODO: Check if center and scale calculation make sense
    # scale factor
    scaleFactor = 1.2
    # structs we use
    imgnames_, imgpaths_, scales_, centers_ = [], [], [], []
    poses_, shapes_, genders_, verticesCam_ = [], [], [], []
    joints2dGT_, joints2dOpenpose_, joints3dCam_,  = [], [], []

    # load vposer
    # vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')
    # vposer = vposer.to(device='cuda')

    # load agora dataframe
    # df = pd.read_pickle(dataset_path)
    # go through all the images
    for idx in (range(10)): # len(df))):
        #image, keypoint
        # img_path = df.iloc[idx]['imgPath'].replace(david_path, my_path)
        img_path = df.iloc[idx]['imgPath']
        print(img_path)
        img_path = img_path.replace(
            '/ps/project/agora/render/unreal/images/20200930-flowers_5_15/trainset/ag_trainset_3dpeople_smpl_flowers_5_15_1280x720/',
            '/ps/scratch/ps_shared/4Muhammed/',
        )
        img_name = os.path.basename(img_path)
        # keypoint_path = img_path.replace('images', 'keypoints').replace('.png', '_keypoints.json')
        # with open(keypoint_path, 'rb') as fp:
        #     keypoints = json.load(fp)
        # keypoints = openpose_json_to_numpy(keypoints)
        # camera
        cam_poses = [df.iloc[idx]['camX'], df.iloc[idx]['camY'], df.iloc[idx]['camZ']]
        cam_yaw = df.iloc[idx]['camYaw']
        # body
        smpl_gt_data = df.iloc[idx, df.columns.get_loc('gt')]
        joints2dGT = df.iloc[idx, df.columns.get_loc('gt2d')] # 3d joints projected into camera from spin gt
        joints3dCam = df.iloc[idx, df.columns.get_loc('gt_joints3d_camCoords')]
        # verticesCam = df.iloc[idx, df.columns.get_loc('gt3dVerts')]
        verticesCam = df.iloc[idx, df.columns.get_loc('gt_vertex3d_camCoords')]
        genders = df.iloc[idx]['gender']
        body_names = df.iloc[idx]['Body']
        valid = df.iloc[idx]['isValid']
        is_kid = df.iloc[idx]['kid']
        valid_hand = df.iloc[idx]['isValidHand']
        # valid body
        body_names_valid = [b for (b, v, kid) in zip(body_names, valid, is_kid) if v and (not kid)]
        joints2dGT_valid = [p for (p, v, kid) in zip(joints2dGT, valid, is_kid) if v and (not kid)]
        smpl_gt_data_valid = [s for (s,v, kid) in zip(smpl_gt_data, valid, is_kid) if v and (not kid)]
        # keypoints_valid = [k for (k,v, kid) in zip(keypoints, valid, is_kid) if v and (not kid)]
        joints3dCam_valid = [j for (j,v, kid) in zip(joints3dCam, valid, is_kid) if v and (not kid)]
        verticesCam_valid = [w for (w,v, kid) in zip(verticesCam, valid, is_kid) if v and (not kid)]

        ### debug
        import cv2
        import matplotlib.pyplot as plt
        im = cv2.imread(img_path)

        # go through all persons in an image
        for jdx in range(len(body_names_valid)):
            smpl_gt_data_per_body = smpl_gt_data_valid[jdx]
            pose_per_body, betas_per_body = smpl_gt_data_per_body['body_pose'].squeeze().reshape(-1,), \
                                            smpl_gt_data_per_body['betas'][:,:10].squeeze()
            global_orient_per_body = smpl_gt_data_per_body['global_orient'].squeeze().reshape(-1,)
            pose_per_body = np.concatenate([global_orient_per_body, pose_per_body], axis=0)
            joints2dGT_per_body = joints2dGT_valid[jdx][:24]
            joints2dGT_per_body = np.hstack([joints2dGT_per_body, np.ones([24, 1])]) # make gt confindence 1
            joints3dCam_per_body = joints3dCam_valid[jdx][:24]
            joints3dCam_per_body = np.hstack([joints3dCam_per_body, np.ones([24, 1])])
            # joints2dOpenpose_per_body = joints2dGT_valid[jdx][:25] # Todo: Currenly both openpose and gt2d are the same
            # joints2dOpenpose_per_body = np.hstack([joints2dOpenpose_per_body, np.ones([25, 1])]) # make openpose confindence 1
            verticesCam_per_body = verticesCam_valid[jdx]
            center_per_body, scale_per_body = bbox_from_jointsGT(joints2dGT_per_body, rescale=scaleFactor)
            gender_per_body = genders[jdx]


            im = cv2.circle(im, (int(center_per_body[0]), int(center_per_body[1])), radius=10,
                            color=(0,0,255), thickness=5)
            pt1 = (int(center_per_body[0]), int(center_per_body[1] - scale_per_body * 100))
            pt2 = (int(center_per_body[0]), int(center_per_body[1] + scale_per_body * 100))
            im = cv2.line(im, pt1, pt2, color=(0,255,0), thickness=3)

            # import IPython; IPython.embed(); exit()

            imgnames_.append(img_name)
            imgpaths_.append(img_path)
            centers_.append(center_per_body)
            scales_.append(scale_per_body)
            poses_.append(pose_per_body)
            shapes_.append(betas_per_body)
            genders_.append(gender_per_body)
            verticesCam_.append(verticesCam_per_body)
            joints3dCam_.append(joints3dCam_per_body)
            joints2dGT_.append(joints2dGT_per_body)
            # joints2dOpenpose_.append(joints2dOpenpose_per_body)

        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        plt.show()
    # store data
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path,
        f'agora_train_{split_num}.npz')
    print(f'Split saved at:{out_file}')
    np.savez(out_file, imgpath=imgpaths_,
                       imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       pose=poses_,
                       shape=shapes_,
                       gender=genders_,
                       # openpose=joints2dOpenpose_,
                       S=joints3dCam_,
                       part=joints2dGT_
             )

def split_list(a, n):
    """
    Split list a in n parts. Returns a generator
    """
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def load_split_df_list(dir_path, split_count=4):
    """
    Returns a list of list by splitting all dfs into 4 parths
    """
    gt_files = [y for x in os.walk(dir_path) for y in glob(os.path.join(x[0], '*.npy'))]
    # gt_files_split = list(split_list(gt_files, split_count))
    gt_files_split = [gt_files[0:2]]
    # df_list = [pd.read_pickle(x) for x in next_split]
    # df_combined = pd.concat(df_list, ignore_index=True, sort=False)
    return gt_files_split



# dataset_path = '/mnt/efs-shared/shatripa_code/human/agora_data/50mm_zoom/df_groundTruth.npy'
# dataset_path = '/ps/project/common/renderpeople_initialfit/4Shashank/Experimental_training_dataframe/df_groundTruth.npy'
agora_training_path = '/ps/project/common/renderpeople_initialfit/4Shashank/training_dataframe'
out_path = 'data/dataset_extras'
vposer_ckpt = ''
gt_files_split = load_split_df_list(agora_training_path)

for i, gt_files in enumerate(gt_files_split):
    df_list = [pd.read_pickle(x) for x in gt_files]
    df_combined = pd.concat(df_list, ignore_index=True, sort=False)
    print(f'Number of bodies in this split #{i}: {len(df_combined)}')
    agora_extract(df_combined, out_path, vposer_ckpt, split_num=i)