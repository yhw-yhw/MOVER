import pickle
import joblib
import os
import sys
sys.path.append('/is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo')
# from phosa.pose_optimization_cooperative import get_cam_intrinsic
# import pyrender
from PIL import Image
import numpy as np
import torch
from easydict import EasyDict as edict

def get_cam_intrinsic(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    intrinsic = np.fromstring(' '.join(lines), dtype=np.float32, sep=' ').reshape((3, 3))
    return intrinsic

def estimate_translation_np(S, joints_2d, joints_conf, cam_inc):
    """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Input:
        S: (25, 3) 3D joint locations
        joints: (25, 3) 2D joint locations and confidence
    Returns:
        (3,) camera translation vector
    """

    num_joints = S.shape[0]
    # focal length
    # f = np.array([focal_length,focal_length])
    f = np.array([cam_inc[0][0], cam_inc[1][1]])
    # optical center
    # center = np.array([img_size/2., img_size/2.])
    center = np.array([cam_inc[0][2], cam_inc[1][2]])

    # transformations
    Z = np.reshape(np.tile(S[:,2],(2,1)).T,-1)
    XY = np.reshape(S[:,0:2],-1)
    O = np.tile(center,num_joints)
    F = np.tile(f,num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf),(2,1)).T,-1)

    # least squares
    Q = np.array([F*np.tile(np.array([1,0]),num_joints), F*np.tile(np.array([0,1]),num_joints), O-np.reshape(joints_2d,-1)]).T
    c = (np.reshape(joints_2d,-1)-O)*Z - F*XY

    # weighted least squares
    W = np.diagflat(weight2)
    Q = np.dot(W,Q)
    c = np.dot(W,c)

    # square matrix
    A = np.dot(Q.T,Q)
    b = np.dot(Q.T,c)

    # solution
    trans = np.linalg.solve(A, b)

    return trans


import sys
sys.path.append('/is/cluster/hyi/workspace/HCI/hdsr/projects/pare/pare')
# from utils.geometry import batch_rodrigues, batch_rot2aa
from geometry import batch_rodrigues, batch_rot2aa
def rectify_pose(camera_r, body_aa):
    # body_r = batch_rodrigues(body_aa).reshape(-1,3,3)
    body_r = body_aa.reshape(-1,3,3)
    final_r = camera_r @ body_r
    body_aa = batch_rot2aa(final_r)
    return body_aa


# import sys
# import os
# import argparse
# sys.path.append('/is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo')
# parser = argparse.ArgumentParser(description='')
# parser.add_argument('--img_path', type=str, default=None, help='')
# parser.add_argument('--img_folder', type=str, default=None, help='')
# parser.add_argument('--output_folder', type=str, default=None, help='')
# parser.add_argument('--idx', type=int, default=-1, help='')

# opt = parser.parse_args()

# img_path = opt.img_path
# img_folder = opt.img_folder
# output_folder = opt.output_folder
# idx = opt.idx

# import glob
# if img_path is None and img_folder is not None:
#     all_list = sorted(glob.glob(os.path.join(img_folder, '*.png')))
#     # import pdb; pdb.set_trace()
#     img_path = all_list[idx]

# save_path = os.path.join(output_folder, os.path.basename(img_path)+'_phosa.png')

# def reorganize_pare(result):
#     new_result = {}
#     for idx, value in result.items():
#         # import pdb;pdb.set_trace()
#         for key, part_value in value.items():
#             if part_value is None:
#                 continue
#             for idx in range(len(part_value)):
#                 if key not in new_result:
#                     new_result[key] = [part_value[idx]]
#                 else:
#                     new_result[key].append(part_value[idx])
#     return edict(new_result)

def reorganize_pare(vibe_results, nframes):
    frame_results = [{} for _ in range(nframes)]
    
    for person_id, person_data in vibe_results.items():
        for idx, frame_id in enumerate(person_data['frame_ids']):
            # import pdb;pdb.set_trace()
            # print(f'frame: {frame_id}')
            if person_data['joints3d'] is not None:
                frame_results[frame_id][person_id] = {
                    'pred_cam': person_data['pred_cam'][idx],
                    'orig_cam': person_data['orig_cam'][idx],
                    'verts': person_data['verts'][idx],
                    # 'joints2d': person_data['joints2d'][idx],
                    'betas': person_data['betas'][idx],
                    'pose': person_data['pose'][idx], 
                    'joints3d': person_data['joints3d'][idx],
                    'smpl_joints2d': person_data['smpl_joints2d'][idx],
                    'bboxes': person_data['bboxes'][idx]
                }
            else:
                print('false', person_data['orig_cam'][idx])
                frame_results[frame_id][person_id] = {
                    'pred_cam': np.zeros((3,)),
                    'orig_cam': np.zeros((3,)),
                    'verts': np.zeros((6890, 3)),
                    # 'joints2d': person_data['joints2d'][idx],
                    'betas': np.zeros((10,)),
                    'pose': np.zeros((24, 3, 3)), 
                    'joints3d': np.zeros((49, 3)),
                    'smpl_joints2d': np.zeros((49, 2)),
                    'bboxes': np.zeros((4,))
                }

    # naive depth ordering based on the scale of the weak perspective camera
    # find the largest one.
    # import pdb;pdb.set_trace()
    new_result = {}
    for frame_id, frame_data in enumerate(frame_results):
        # print(f'frame: {frame_id}')
        # sort based on y-scale of the cam in original image coords
        sort_idx = np.argsort([v['orig_cam'][1] for k,v in frame_data.items()])
        # print(sort_idx)
        # frame_results[frame_id] = OrderedDict(
        #     {list(frame_data.keys())[i]:frame_data[list(frame_data.keys())[i]] for i in sort_idx}
        # )
        # * missing frame
        if len(sort_idx) == 0:
            # import pdb;pdb.set_trace()
            for key in new_result.keys():
                new_result[key].append(new_result[key][-1])
            continue
        # * find the largest one.
        tmp_frame_data = frame_data[list(frame_data.keys())[sort_idx[-1]]]
        for key, part_value in tmp_frame_data.items():
            if key not in new_result:
                new_result[key] = [part_value]
            else:
                new_result[key].append(part_value)
    # import pdb;pdb.set_trace()
    return edict(new_result)

def get_idx_result(result, idx):
    single_frm_result = {}

    single_frm_result["pose_param"] = result.pose[idx]
    single_frm_result["betas_param"] = result.betas[idx]
    single_frm_result["verts"] = result.verts[idx]
    single_frm_result["orig_cam"] = result.orig_cam[idx]
    single_frm_result["pose"] = result.pose[idx]
    single_frm_result["joints3d"] = result.joints3d[idx]
    single_frm_result["smpl_joints2d"] = result.smpl_joints2d[idx]
    single_frm_result["bboxes"] = result.bboxes[idx]
    # single_frm_result["frame_ids"] = result.frame_ids[idx]
    single_frm_result["frame_ids"] = idx
    # assert result.frame_ids[idx] == idx

    return edict(single_frm_result)

def modify_json(json_file):
    with open(json_file, 'r') as f:
        results = json.load(f)
    # import pdb;pdb.set_trace()
    # results['people'][0]['pose_keypoints_3d'] = np.zeros((25, 4)).reshape(-1).tolist()
    results['people'][0]['pose_keypoints_3d'] = np.array([-0.22412225604057312, -0.6498373746871948, 2.9610018730163574, 
            0.0, 1.248719573020935, -1.0115410089492798, 3.6479299068450928, 0.0, -0.0635649710893631, -0.5221863389015198, 
            3.206291437149048, 0.0, -0.027896245941519737, -0.48878908157348633, 3.4668233394622803, 0.0, -0.0032307107467204332,
             -0.5475852489471436, 3.714127540588379, 0.0, -0.029576370492577553, -0.4711841642856598, 2.8522140979766846, 0.0, 0.051362618803977966, 
             -0.34874120354652405, 2.645429849624634, 0.0, 0.018674185499548912, -0.3764256238937378, 2.406602144241333, 0.0, 0.679316520690918, -1.2989616394042969, 
             4.081627368927002, 0.0, -0.06682337820529938, 0.043380845338106155, 3.1829376220703125, 0.0, -0.05268491804599762, 0.47060418128967285, 3.2917580604553223,
              0.0, -0.02126387506723404, 0.8698357343673706, 3.3812973499298096, 0.0, -0.08949077129364014, 0.07104416936635971, 2.985130548477173, 0.0,
               -0.047359202057123184, 0.5159651041030884, 3.022535800933838, 0.0, -0.003500135848298669, 0.9128473401069641, 3.057770252227783, 0.0,
                -0.21456217765808105, -0.6910736560821533, 2.9972853660583496, 0.0, -0.1919327974319458, -0.6841851472854614, 2.9299850463867188, 0.0, 
                -0.14674854278564453, -0.6980823278427124, 3.0769851207733154, 0.0, -0.13207750022411346, -0.6670486330986023, 2.879300594329834, 0.0, 
                -0.18096306920051575, 0.9664211273193359, 2.99873685836792, 0.0, -0.12371698021888733, 0.9613545536994934, 2.9702961444854736, 0.0, 
                0.027623290196061134, 0.9609692692756653, 3.082742691040039, 0.0, -0.2177031934261322, 0.9018409848213196, 3.4231624603271484, 0.0, 
                -0.1774619221687317, 0.8976943492889404, 3.456867218017578, 0.0, 0.021769778802990913, 0.925752580165863, 3.3631489276885986, 1.0]).tolist()
    return results

def preprocess_mv_input(img_dir, json_dir, output_dir):
    print(f'save to {output_dir}')
    for one in tqdm(sorted(os.listdir(img_dir))):
        # import pdb;pdb.set_trace()
        base, ext = os.path.splitext(one)
        if base == 'img': 
            continue
        os.makedirs(os.path.join(output_dir, base, '00', 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, base, '00', 'keypoints'), exist_ok=True)

        ori_img_p = os.path.join(img_dir, f'{one}')
        dst_path = os.path.join(output_dir, base, '00', 'images', f'{one}')
        if not os.path.exists(dst_path):
            os.system(f'ln -s {ori_img_p} {dst_path}')

        ori_json = os.path.join(json_dir, f'{base}_keypoints.json')
        new_result = modify_json(ori_json)
        dst_json = os.path.join(output_dir, base, '00', 'keypoints', f'{base}_keypoints.json')
        with open(dst_json, 'w') as f:
            json.dump(new_result, f)

## load model information
import time
import yaml

import smplx
from utils import JointMapper
from data_parser import create_dataset
from cmd_parser import parse_config
import os.path as osp
# import tqdm
from tqdm import tqdm
import trimesh

from constants import JOINT_NAMES, JOINT_MAP

SMPLX_OP_MAP = [JOINT_MAP[one] for one in JOINT_NAMES[:25]]
import json

def get_op_result(pare_kpts3d, op_conf=None):
    # import pdb; pdb.set_trace()
    # result = pare_kpts3d[SMPLX_OP_MAP]
    result = pare_kpts3d[:25, :]
    if op_conf is not None and op_conf.shape[0] != 0:
        # ! use OP confidence
        return np.concatenate((result, op_conf.reshape((result.shape[0], 1))), 1)
    else:
        return np.concatenate((result, np.ones((result.shape[0], 1))), 1)
    # import pdb;pdb.set_trace()
    # return np.stack((result[:, 0], -result[:, 1], -result[:, 2]), 1)

def load_camera_from_xml(xml_fn):
    import cv2
    fs = cv2.FileStorage(xml_fn, cv2.FILE_STORAGE_READ)
    extrinsics_mat = fs.getNode("CameraMatrix")
    extrinsics_mat = extrinsics_mat.mat()
    # import pdb;pdb.set_trace()
    intrinsics_mat = fs.getNode("Intrinsics")
    intrinsics_mat = intrinsics_mat.mat()

    return intrinsics_mat, extrinsics_mat[:, :-1]

def main2(**args):
    output_folder = args.pop('output_folder')
    output_folder = osp.expandvars(output_folder)
    if not osp.exists(output_folder):
        os.makedirs(output_folder)

    
    ## load camera rotation
    OBJ_DIR = args.pop('cam_dir')
    # GLOBAL_CAM_INC = get_cam_intrinsic(os.path.join(OBJ_DIR, 'cam_K.txt')) 
    # print('cam')
    # # print('ori: \n', GLOBAL_CAM_INC)
    # # GLOBAL_CAM_INC = scale_camera(GLOBAL_CAM_INC, 1)
    # print('init cam inc: \n')
    # print(GLOBAL_CAM_INC)
    # import scipy.io
    # GLOBAL_CAM_EXT = scipy.io.loadmat(os.path.join(OBJ_DIR, 'r_ex.mat'))['cam_R'] 
    # print('ori: \n', GLOBAL_CAM_EXT)
    # print('************debug**************')
    print(os.path.join(OBJ_DIR, '001.xml'))
    GLOBAL_CAM_INC, GLOBAL_CAM_EXT = load_camera_from_xml(os.path.join(OBJ_DIR, '001.xml'))
    print('load camera from ', os.path.join(OBJ_DIR, '001.xml'))


    
    # Store the arguments for the current experiment
    conf_fn = osp.join(output_folder, 'conf.yaml')
    with open(conf_fn, 'w') as conf_file:
        yaml.dump(args, conf_file)

    result_folder = args.pop('result_folder', 'results')
    result_folder = osp.join(output_folder, result_folder)
    if not osp.exists(result_folder):
        os.makedirs(result_folder)

    mesh_folder = args.pop('mesh_folder', 'meshes')
    mesh_folder = osp.join(output_folder, mesh_folder)
    if not osp.exists(mesh_folder):
        os.makedirs(mesh_folder)

    out_img_folder = osp.join(output_folder, 'images')
    if not osp.exists(out_img_folder):
        os.makedirs(out_img_folder)

    use_cuda = args.get('use_cuda', True)
    if use_cuda and not torch.cuda.is_available():
        print('CUDA is not available, exiting!')
        sys.exit(-1)

    visualize = args.get('visualize', True)

    check_inverse_feet = args.get('check_inverse_feet', True)
    # import pdb;pdb.set_trace()
    ## save to json file
    json_folder = args.pop('json_folder')
    save_new_json = args.pop('save_new_json')
    export_mesh = args.pop('export_mesh')
    print(f'export mesh {export_mesh}, save_new_json {save_new_json}, json folder {json_folder}')

    # data_folder = args.pop('data_folder', 'data')
    data_folder = args.pop('data_folder')
    img_folder = data_folder
    dataset_obj = create_dataset(data_folder=data_folder, **args)
    # dataset_obj = create_dataset(img_folder=img_folder, **args)
    
    # img_folder = '/ps/scratch/ps_shared/hyi/holistic_human_scene_capture/pare_results/pigraph_input_image_pare_hrnet/tmp_images/'
    ## load pare result
    # result_path = '/ps/scratch/hyi/HCI_dataset/holistic_scene_human/pigraph_input_image_high_resolution/pare_output.pkl'
    import glob
    nframes = len(glob.glob(os.path.join(img_folder, '*jpg'))) + \
            len(glob.glob(os.path.join(img_folder, '*png')))  
    result_path = args.pop('pare_result')
    pare_result = joblib.load(result_path)
    # print('nframes:',nframes)
    # print('img_folder:',img_folder)
    # print('nframes2:',len(glob.glob(os.path.join(img_folder, '*jpg'))))
    # test_path='/root/code/mover/preprocess/input_data/Color_flip/imgs'
    # print('nframes3:',len(glob.glob(os.path.join(test_path, '*jpg'))))
    pare_result_dict = reorganize_pare(pare_result, nframes)

    if not visualize:
        # TODO: generate mv input
        preprocess_mv_input(img_folder, json_folder, output_folder+'_ori_OP')
        # preprocess_mv_input(img_folder, json_folder, output_folder)
        preprocess_mv_input(img_folder, json_folder, output_folder+'_PARE3DJointOneConfidence_OP2DJoints')

    start = time.time()

    input_gender = args.pop('gender', 'neutral')
    gender_lbl_type = args.pop('gender_lbl_type', 'none')
    max_persons = args.pop('max_persons', -1)

    float_dtype = args.get('float_dtype', 'float32')
    if float_dtype == 'float64':
        dtype = torch.float64
    elif float_dtype == 'float32':
        dtype = torch.float32
    else:
        raise ValueError('Unknown float type {}, exiting!'.format(float_dtype))


    joint_mapper = JointMapper(dataset_obj.get_model2data())

    model_params = dict(model_path=args.get('model_folder'),
                            joint_mapper=joint_mapper,
                            create_global_orient=True,
                            create_body_pose=not args.get('use_vposer'),
                            create_betas=True,
                            create_left_hand_pose=True,
                            create_right_hand_pose=True,
                            create_expression=True,
                            create_jaw_pose=True,
                            create_leye_pose=True,
                            create_reye_pose=True,
                            create_transl=True,
                            dtype=dtype,
                            **args)
                            
    male_model = smplx.create(gender='male', **model_params)
    # SMPL-H has no gender-neutral model
    if args.get('model_type') != 'smplh':
        neutral_model = smplx.create(gender='neutral', **model_params)
    female_model = smplx.create(gender='female', **model_params)

    gender = input_gender
    if gender == 'neutral':
        body_model = neutral_model
    elif gender == 'female':
        body_model = female_model
    elif gender == 'male':
        body_model = male_model
    args['gender'] = gender


    for idx in tqdm(range(nframes)): # frame_ids
        one_result = get_idx_result(pare_result_dict, idx)
    
        # frame_id = one_result.frame_ids + 1 # Warnings
        frame_id = one_result.frame_ids  # Warnings
        img_path = os.path.join(img_folder, f'{frame_id:06d}.png')
        image = Image.open(img_path)
        w, h = image.size

        # version: 1
        # ! still bug in here.
        rot_joints3d = np.matmul(GLOBAL_CAM_EXT.T, one_result.joints3d.T).T
        rot_verts = np.matmul(GLOBAL_CAM_EXT.T, one_result.verts.T).T
    
        # TODO: estimated camera trans
        trans = estimate_translation_np(rot_joints3d, one_result.smpl_joints2d, 
                np.ones(one_result.smpl_joints2d.shape[0]), GLOBAL_CAM_INC)
        # save to mesh
        vertices = rot_verts + np.matmul(GLOBAL_CAM_EXT.T, trans).T 
        # ! 3D joints in world coordinates, which is different in multiview-smplifyx
        final_rot_joints3d = rot_joints3d + np.matmul(GLOBAL_CAM_EXT.T, trans).T 
        
        # this is used for smplify-x
        cam_final_rot_joints3d = np.matmul(GLOBAL_CAM_EXT, final_rot_joints3d.T).T
        cam_vertices = np.matmul(GLOBAL_CAM_EXT, vertices.T).T
        # version 2:
        # pelvis = one_result.joints3d[39]
        # joint3d_delta = one_result.joints3d - pelvis
        # verts_delta = one_result.verts - pelvis
        # rot_joints3d = np.matmul(GLOBAL_CAM_EXT.T, joint3d_delta.T).T + pelvis
        # rot_verts = np.matmul(GLOBAL_CAM_EXT.T, verts_delta.T).T + pelvis

        # trans = estimate_translation_np(rot_joints3d, one_result.smpl_joints2d, 
        #         np.ones(one_result.smpl_joints2d.shape[0]), GLOBAL_CAM_INC)
        # vertices = rot_verts + trans

        if visualize:
            import cv2
            cv2_img = cv2.imread(img_path)
            ori_proj = one_result.smpl_joints2d.T
            for j in range(ori_proj.shape[1]):
                print(ori_proj[0, j], ori_proj[1, j])
                cv2_img = cv2.circle(cv2_img, (int(ori_proj[0, j]), int(ori_proj[1, j])), \
                            radius=5, color=(255, 0, 0), thickness=4)
                            
            # visualize rot_joints3d
            # Original version: 0204 before
            tmp_rot = np.matmul(GLOBAL_CAM_EXT, final_rot_joints3d.T).T
            proj_3d = np.matmul(GLOBAL_CAM_INC, tmp_rot.T).T

            # tmp_rot = np.matmul(GLOBAL_CAM_EXT, final_rot_joints3d.T).T
            # proj_3d = np.matmul(GLOBAL_CAM_INC, tmp_rot.T).T

            # proj_3d = np.matmul(GLOBAL_CAM_INC, (rot_joints3d).T).T
            proj_x = proj_3d[:, 0] / proj_3d[:, 2]
            proj_y = proj_3d[:, 1] / proj_3d[:, 2]
            proj_xy = np.vstack((proj_x, proj_y))
            print(proj_xy)
            
            
            for j in range(proj_xy.shape[1]):
                print(proj_xy[0, j], proj_xy[1, j])
                cv2_img = cv2.circle(cv2_img, (int(proj_xy[0, j]), int(proj_xy[1, j])), \
                            radius=5, color=(0, 0, 255), thickness=4)
            cv2.imwrite("demo_bluePARE_redNew.png", cv2_img)
            cv2.waitKey(500)
            break
            import pdb; pdb.set_trace()


        if export_mesh:
            os.makedirs(os.path.join(output_folder+'_PARE3DJointOneConfidence_OP2DJoints', 'meshes'), exist_ok=True)
            mesh_fn = os.path.join(output_folder+'_PARE3DJointOneConfidence_OP2DJoints', 'meshes' , f'{frame_id:06d}.obj')
            # out_mesh = trimesh.Trimesh(vertices, body_model.faces, process=False) # before 06.27
            out_mesh = trimesh.Trimesh(cam_vertices, body_model.faces, process=False)
            out_mesh.export(mesh_fn)

            os.makedirs(os.path.join(output_folder+'_PARE3DJointOneConfidence_OP2DJoints', 'pare_results'), exist_ok=True)
            mesh_fn = os.path.join(output_folder+'_PARE3DJointOneConfidence_OP2DJoints', 'pare_results' , f'{frame_id:06d}.pkl')
            joblib.dump(one_result, mesh_fn)


        # output SMPL joints of openpose format
        if save_new_json:
            # json_fn = os.path.join(output_folder+'_ori_OP', f'{frame_id:06d}', '00/keypoints', f'{frame_id:06d}_keypoints.json')
            # with open(json_fn, 'r') as fin:
            #     result = json.load(fin)
            #     # import pdb;pdb.set_trace()
            #     op_conf = np.array(result['people'][0]['pose_keypoints_2d']).reshape(-1, 3)[:, -1]

            #     # kpts_3d = get_op_result(final_rot_joints3d) # before 06.27
            #     kpts_3d = get_op_result(cam_final_rot_joints3d)

            #     if idx % 50 == 0 and False:
            #         import matplotlib.pyplot as plt
            #         fig = plt.figure()
            #         ax = fig.add_subplot(111, projection='3d')
            #         m = '^'
            #         # for idx in range(kpts_3d.shape[0]):

            #         #     xs, ys, zs = kpts_3d[idx,0], kpts_3d[idx,1], kpts_3d[idx,2]
            #         #     ax.scatter(xs, ys, zs, marker=m)
            #         #     ax.text(xs, ys, zs, f'{idx}')
            #         for idx in range(final_rot_joints3d.shape[0]-24):
            #             xs, ys, zs = final_rot_joints3d[idx, 0], final_rot_joints3d[idx, 1], final_rot_joints3d[idx, 2]
            #             ax.scatter(xs, ys, zs, marker='o')
            #             ax.text(xs, ys, zs, f'{idx}')

            #         ax.set_xlabel('X Label')
            #         ax.set_ylabel('Y Label')
            #         ax.set_zlabel('Z Label')

            #         plt.show()

            #     # import pdb;pdb.set_trace()
            #     result['people'][0]['pose_keypoints_3d'] = kpts_3d.reshape(-1).tolist()
            #     # replace the whole body points
            #     # result['people'][0]['pose_keypoints_2d'] = np.concatenate((one_result.smpl_joints2d[:25,:],
            #     #                                                         np.ones((25, 1))), 1).reshape(-1).tolist()
            #     # replace the whole body without feet
            #     right_angle = result['people'][0]['pose_keypoints_2d'][3*11:3*12]
            #     left_angle = result['people'][0]['pose_keypoints_2d'][3*14:3*15]

            #     # TODO: check the shape
            #     if op_conf.shape[0] != 0:
            #         result['people'][0]['pose_keypoints_2d'][:3*19] = np.concatenate((one_result.smpl_joints2d[:19,:],
            #                                                             op_conf[:19].reshape(19, 1)), 1).reshape(-1).tolist()
            #     else:
            #         result['people'][0]['pose_keypoints_2d'][:3*19] = np.concatenate((one_result.smpl_joints2d[:19,:],
            #                                                             np.ones((19, 1))), 1).reshape(-1).tolist()
                
            #     # TODO: add right-left check process, use pare result as first prioritity
            #     pare_right_angle =  one_result.smpl_joints2d[11, :]
            #     pare_left_angle =  one_result.smpl_joints2d[14, :]
            #     # import pdb;pdb.set_trace()
            #     if check_inverse_feet and pare_right_angle.size > 0 and pare_left_angle.size > 0 and \
            #         len(right_angle) > 0 and len(left_angle) > 0 and \
            #         np.linalg.norm(pare_right_angle-right_angle[:-1]) > 100 and \
            #         np.linalg.norm(pare_right_angle-right_angle[:-1]) < 300 and \
            #         np.linalg.norm(pare_right_angle-right_angle[:-1]) - 4 * np.linalg.norm(pare_right_angle-left_angle[:-1]) > 0:
            #         print(f'inverse feet in {idx}, replace angle and feet in a reverse way')
            #         print(f'Ori: {np.linalg.norm(pare_right_angle-right_angle[:-1]) }, New: {np.linalg.norm(pare_right_angle-left_angle[:-1])}')
            #         result['people'][0]['pose_keypoints_2d'][3*11:3*12] = left_angle
            #         result['people'][0]['pose_keypoints_2d'][3*14:3*15] = right_angle
                    
            #         right_feet = result['people'][0]['pose_keypoints_2d'][3*22:3*25]
            #         left_feet = result['people'][0]['pose_keypoints_2d'][3*19:3*22]

            #         result['people'][0]['pose_keypoints_2d'][3*22:3*25] = left_feet
            #         result['people'][0]['pose_keypoints_2d'][3*19:3*22] = right_feet
            #     else:
            #         # Only replace the angles, if correct feet.
            #         result['people'][0]['pose_keypoints_2d'][3*11:3*12] = right_angle
            #         result['people'][0]['pose_keypoints_2d'][3*14:3*15] = left_angle
            # dst_json = os.path.join(output_folder, f'{frame_id:06d}', '00', 'keypoints', f'{frame_id:06d}_keypoints.json')
            # with open(dst_json, 'w') as f:
            #     json.dump(result, f)



            # ## with update pare3D joints and keep 2D joint in OpenPose
            # json_fn = os.path.join(output_folder+'_ori_OP', f'{frame_id:06d}', '00/keypoints', f'{frame_id:06d}_keypoints.json')
            # with open(json_fn, 'r') as fin:
            #     result = json.load(fin)
            #     # import pdb;pdb.set_trace()
            #     op_conf = np.array(result['people'][0]['pose_keypoints_2d']).reshape(-1, 3)[:, -1]

            #     kpts_3d = get_op_result(final_rot_joints3d, op_conf)

            #     # import pdb;pdb.set_trace()
            #     result['people'][0]['pose_keypoints_3d'] = kpts_3d.reshape(-1).tolist()
                
            #     # TODO: add right-left check process, use pare result as first prioritity
            #     pare_right_angle =  one_result.smpl_joints2d[11, :]
            #     pare_left_angle =  one_result.smpl_joints2d[14, :]
            #     # import pdb;pdb.set_trace()
            #     if pare_right_angle.size > 0 and pare_left_angle.size > 0 and \
            #         len(right_angle) > 0 and len(left_angle) > 0 and \
            #         np.linalg.norm(pare_right_angle-right_angle[:-1]) > 100 and \
            #         np.linalg.norm(pare_right_angle-right_angle[:-1]) < 300 and \
            #         np.linalg.norm(pare_right_angle-right_angle[:-1]) - 4 * np.linalg.norm(pare_right_angle-left_angle[:-1]) > 0:
            #         print(f'inverse feet in {idx}, replace angle and feet in a reverse way')
            #         print(f'Ori: {np.linalg.norm(pare_right_angle-right_angle[:-1]) }, New: {np.linalg.norm(pare_right_angle-left_angle[:-1])}')
            #         result['people'][0]['pose_keypoints_2d'][3*11:3*12] = left_angle
            #         result['people'][0]['pose_keypoints_2d'][3*14:3*15] = right_angle
                    
            #         right_feet = result['people'][0]['pose_keypoints_2d'][3*22:3*25]
            #         left_feet = result['people'][0]['pose_keypoints_2d'][3*19:3*22]

            #         result['people'][0]['pose_keypoints_2d'][3*22:3*25] = left_feet
            #         result['people'][0]['pose_keypoints_2d'][3*19:3*22] = right_feet
            #     else:
            #         # Only replace the angles, if correct feet.
            #         result['people'][0]['pose_keypoints_2d'][3*11:3*12] = right_angle
            #         result['people'][0]['pose_keypoints_2d'][3*14:3*15] = left_angle
                
            # dst_json = os.path.join(output_folder+'_PARE3DJoint_OP2DJoints', f'{frame_id:06d}', '00', 'keypoints', f'{frame_id:06d}_keypoints.json')
            # with open(dst_json, 'w') as f:
            #     json.dump(result, f)


            json_fn = os.path.join(output_folder+'_ori_OP', f'{frame_id:06d}', '00/keypoints', f'{frame_id:06d}_keypoints.json')
            with open(json_fn, 'r') as fin:
                result = json.load(fin)
                # import pdb;pdb.set_trace()
                op_conf = np.array(result['people'][0]['pose_keypoints_2d']).reshape(-1, 3)[:, -1]

                kpts_3d = get_op_result(final_rot_joints3d)

                # import pdb;pdb.set_trace()
                result['people'][0]['pose_keypoints_3d'] = kpts_3d.reshape(-1).tolist()
            
                right_angle = result['people'][0]['pose_keypoints_2d'][3*11:3*12]
                left_angle = result['people'][0]['pose_keypoints_2d'][3*14:3*15]
                
                # TODO: add right-left check process, use pare result as first prioritity
                pare_right_angle =  one_result.smpl_joints2d[11, :]
                pare_left_angle =  one_result.smpl_joints2d[14, :]
                # import pdb;pdb.set_trace()
                if check_inverse_feet and pare_right_angle.size > 0 and pare_left_angle.size > 0 and \
                    len(right_angle) > 0 and len(left_angle) > 0 and \
                    np.linalg.norm(pare_right_angle-right_angle[:-1]) > 100 and \
                    np.linalg.norm(pare_right_angle-right_angle[:-1]) < 300 and \
                    np.linalg.norm(pare_right_angle-right_angle[:-1]) - 4 * np.linalg.norm(pare_right_angle-left_angle[:-1]) > 0:
                    print(f'inverse feet in {idx}, replace angle and feet in a reverse way')
                    print(f'Ori: {np.linalg.norm(pare_right_angle-right_angle[:-1]) }, New: {np.linalg.norm(pare_right_angle-left_angle[:-1])}')
                    result['people'][0]['pose_keypoints_2d'][3*11:3*12] = left_angle
                    result['people'][0]['pose_keypoints_2d'][3*14:3*15] = right_angle
                    
                    right_feet = result['people'][0]['pose_keypoints_2d'][3*22:3*25]
                    left_feet = result['people'][0]['pose_keypoints_2d'][3*19:3*22]

                    result['people'][0]['pose_keypoints_2d'][3*22:3*25] = left_feet
                    result['people'][0]['pose_keypoints_2d'][3*19:3*22] = right_feet
                else:
                    # Only replace the angles, if correct feet.
                    result['people'][0]['pose_keypoints_2d'][3*11:3*12] = right_angle
                    result['people'][0]['pose_keypoints_2d'][3*14:3*15] = left_angle
                
            dst_json = os.path.join(output_folder+'_PARE3DJointOneConfidence_OP2DJoints', f'{frame_id:06d}', '00', 'keypoints', f'{frame_id:06d}_keypoints.json')
            with open(dst_json, 'w') as f:
                json.dump(result, f)

            # SMPL parameters 
            # pose = torch.from_numpy(one_result.pose).unsqueeze(0)
            # new_pose = pose.clone()
            # new_pose[:, :3] = rectify_pose(torch.from_numpy(GLOBAL_CAM_EXT.T).unsqueeze(0), pose[:, :3])
            # new_pose = new_pose.numpy()

        # generate new SMPL model
        
        # SMPL-SMPL-X
        # https://github.com/vchoutas/smplx/tree/master/transfer_model#smpl-to-smpl-x

        # load openpose joints

        # finish SMPL-X hand and face optimization


if __name__ == "__main__":
    args = parse_config()
    main1(**args)




