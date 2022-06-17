import trimesh
import numpy as np
import json
import scipy.io as sio
import os
from glob import glob
from easydict import EasyDict as edict
import joblib
from scipy.spatial.transform import Rotation as R
from PIL import Image
from tqdm import tqdm
import shutil
import copy
from loguru import logger
import pickle

import torch
import torch.nn.functional as F

from mover.utils.total3d_tool import load_meshes_from_obj
from mover.constants import NYU40CLASSES, IMAGE_SIZE
from mover.utils.total3d_tool import format_bbox, format_layout, format_mesh, format_obj_list
from mover.utils import get_instance_masks_within_global_cam, get_local_mask_with_roi_camera
from mover.utils.geometry import rotation_matrix_to_angle_axis

def get_idx_in_filter_img(idx, obj_list):
    if idx == -1:
        return -1
    else:
        basename = f'{idx+1:06d}.jpg'
        for i in range(len(obj_list)):
            if os.path.basename(obj_list[i]) == basename:
                return i        
        return -1

def load_scanned_mesh(basic_dir):
    scene_fn = os.path.join(basic_dir, 'scan_cam_cooridnates.ply')
    det_fn = os.path.join(basic_dir, 'bbox3d.json')
    if os.path.exists(det_fn):
        with open(det_fn, 'r') as fin:
            bbox3d_gt = json.load(fin)
    else:
        bbox3d_gt = None
    scene_mesh = trimesh.load(scene_fn, process=False)
    return scene_mesh, bbox3d_gt

def load_contact_label(ground_contact_path, img_list):
    try:
        ground_contact_array = np.load(ground_contact_path)
        idx_gc = [one-1 for one in img_list]
        ground_contact_value = ground_contact_array[idx_gc]
    except:
        ground_contact_value = np.zeros((len(img_list), 4))

    return ground_contact_value

def pre_match_3dbbox(bbox3d, bbox2d, bbox_cls):
    result = {}
    for key in sorted(bbox3d.keys()):
        for key, value in bbox3d[key].items():
            if key not in result:
                result[key] = [value]
            else:
                result[key].append(value)
    for key in result.keys():
        if key in ["centroid", "basis", "coeffs"]:
            result[key] = np.array(result[key])
    return result, np.ones(len(bbox3d.keys()))

def load_pre_cam(opt):
    if opt.load_scalenet_cam:
        if opt.cams_scalenet_fn is not None:
            cams_params = get_scalenet_parameters(opt.cams_scalenet_fn)
        else:
            cams_params = opt.cams_params
        print(f'load scalenet cam: {cams_params}')
    else:
        cams_params = None
    return cams_params

def load_scanned_data(opt, bdb2D_pos, size_cls):
    scanned_path = opt.scanned_path
    if scanned_path is not None and os.path.exists(scanned_path):
        scanned_scene, gt_3dbbox_results = load_scanned_mesh(scanned_path)
        if gt_3dbbox_results is not None:
            gt_3dbbox_results, error_valid_flag = pre_match_3dbbox(gt_3dbbox_results, bdb2D_pos, size_cls)
    else:
        scanned_scene = None
        gt_3dbbox_results = None
    return scanned_scene, gt_3dbbox_results

def load_init_scene(opt, filter=None):
    # optimize 3D scene initialization results; or load 3D understanding results
    # objects: mesh model init position;
    scene_result_dir = opt.scene_result_dir
    scene_result_path = opt.scene_result_path
    
    # load scalenet parameters
    cams_params = load_pre_cam(opt)

    if opt.USE_INIT_SCENE:
        if not 'cooperative' in scene_result_dir or not 'cooperative' in scene_result_path:
            if scene_result_path is not None:
                scene_result = load_total3d_result(scene_result_path, USE_CAD_SIZE=opt.USE_CAD_SIZE, filter=filter)
            else: # generate different scene initialization for each frame
                scene_result = load_total3d_result(os.path.join(scene_result_dir, f'frame{idx}'), USE_CAD_SIZE=opt.USE_CAD_SIZE, filter=filter)
        else:
            if scene_result_path is not None:
                scene_result = load_cooperative_result(scene_result_path)
            else:
                scene_result = load_cooperative_result(os.path.join(scene_result_dir, f'frame{idx}'))
        
        obj_size = scene_result['boxes_3d']['coeffs'] * 2
    else:
        # TODO: directly reconstruct scenes.
        pass

    return scene_result, obj_size 

def preprocess_input_mask(opt, mask_objs, mask_body, global_cam_inc, bdb2D_pos, IMAGE_SIZE, device):
    if opt.USE_MASK:
        # get mask input format
        masks_object, _, masks_person, target_masks =  get_instance_masks_within_global_cam(mask_objs, mask_body, image_size=IMAGE_SIZE)
        tmp1, cam_rois = get_local_mask_with_roi_camera(target_masks, global_cam_inc, bdb2D_pos)
        
        K_rois = torch.Tensor(cam_rois).to(device)
        tmp1=torch.from_numpy(tmp1).to(device)
        target_masks = tmp1

        masks_person=torch.from_numpy(masks_person).to(device)
        masks_object=torch.from_numpy(masks_object).to(device)

    else: 
        masks_object = None
        masks_person = None
        target_masks = None
        K_rois = None

    return masks_object, masks_person, target_masks, K_rois



def load_pare_smplifyx_result(pare_dir, img_path): 
    pare_fn = os.path.join(pare_dir, img_path[:-3]+'obj')
    pare_pkl_fn = os.path.join(pare_dir, img_path[:-3]+'pkl')    
    print(f'load pare smplx model: {pare_fn}')

    try: 
        mesh = trimesh.load(pare_fn, process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float32),
        faces = np.asarray(mesh.faces, dtype=np.int32),
        pare_result = joblib.load(pare_pkl_fn)
        pose_mat = pare_result['body_pose'].squeeze(0)
        body_pose = rotation_matrix_to_angle_axis(pose_mat)
        pare_result['body_pose'] = body_pose.reshape(-1, 63)
        pare_result['global_orient'] = rotation_matrix_to_angle_axis(pare_result['global_orient'].squeeze(0)).reshape(-1, 3)
        pare_result['flag'] = True
    
    except:
        print(f'error in path {img_path}')
        pare_fn = os.path.join(pare_dir, '000010.obj')
        pare_pkl_fn = os.path.join(pare_dir, '000010.pkl')
        
        mesh = trimesh.load(pare_fn, process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float32),
        faces = np.asarray(mesh.faces, dtype=np.int32),
        pare_result = joblib.load(pare_pkl_fn)
        pose_mat = pare_result['body_pose'].squeeze(0)
        body_pose = rotation_matrix_to_angle_axis(pose_mat)
        pare_result['body_pose'] = body_pose.reshape(-1, 63)
        pare_result['global_orient'] = rotation_matrix_to_angle_axis(pare_result['global_orient'].squeeze(0)).reshape(-1, 3)
        pare_result['flag'] = False

    return vertices[0], faces[0], pare_result


def load_smplifyx_result(pare_dir, img_path): 

    pare_fn = os.path.join(pare_dir, img_path[:-4], 'meshes/000.obj')
    pare_pkl_fn = os.path.join(pare_dir, img_path[:-4], 'results/000.pkl')
    print(f'load smpl model: {pare_fn}')

    try:
        mesh = trimesh.load(pare_fn, process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float32),
        faces = np.asarray(mesh.faces, dtype=np.int32),
        pare_result = joblib.load(pare_pkl_fn)
    except:
        # TODO: change to local path
        print(f'error in path {img_path}')
        pare_fn = '/ps/scratch/hyi/HCI_dataset/20210109_capture/C0034/hci_test_mv_smplify_result_pare3d_opfeetAnkles/000001/meshes/000.obj'
        pare_pkl_fn = '/ps/scratch/hyi/HCI_dataset/20210109_capture/C0034/hci_test_mv_smplify_result_pare3d_opfeetAnkles/000001/results/000.pkl'
        mesh = trimesh.load(pare_fn, process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float32),
        faces = np.asarray(mesh.faces, dtype=np.int32),
        pare_result = joblib.load(pare_pkl_fn)
        
    return vertices[0], faces[0], pare_result

def get_cam_intrinsic(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    intrinsic = np.fromstring(' '.join(lines), dtype=np.float32, sep=' ').reshape((3, 3))
    return intrinsic

def scale_camera(cam, scale=1):
    """ resize input in order to produce sampled depth map """
    new_cam = np.copy(cam)
    new_cam[0][0] = cam[0][0] * scale
    new_cam[1][1] = cam[1][1] * scale
    # principle point:
    new_cam[0][2] = cam[0][2] * scale
    new_cam[1][2] = cam[1][2] * scale
    return new_cam

def filter_by_2D_det_conf(bdb2D_pos, mask_objs, size_cls, thre=0.7): # thre=0.7
    conf = bdb2D_pos[:, -1]
    flag = conf >= thre

    f_bdb2D_pos = bdb2D_pos[flag]
    f_mask_objs = mask_objs[flag]
    f_size_cls = size_cls[flag]
    return f_bdb2D_pos, f_mask_objs, f_size_cls, flag

def load_detection_scene(det_fn, scale=1.0, mask=False, thre=0.7):
    with open(det_fn, 'r') as file:
        detections = json.load(file)
    bdb2D_pos, mask_objs, size_cls, bodybdb2D_pos, mask_body  = parse_detections(detections, mask=mask)

    if bdb2D_pos.shape[0] > 0:
        bdb2D_pos[:, :-1] = bdb2D_pos[:, :-1] * scale
        
    if mask and len(mask_objs) != 0:
        _, h, w = mask_objs.shape
    elif mask and len(mask_body) != 0:
        _, h, w = mask_body.shape

    if len(mask_objs) != 0 and mask_objs.shape[0] != 0:
        mask_objs = F.interpolate(torch.from_numpy(mask_objs).unsqueeze(0).float(), size=(int(h*scale), int(w*scale)), \
                        mode='nearest').squeeze(0).numpy().astype(bool)

    # not use body detection score
    bodybdb2D_pos = bodybdb2D_pos * scale
    if mask_body.shape[0] != 0:
        mask_body = F.interpolate(torch.from_numpy(mask_body).unsqueeze(0).float(), size=(int(h*scale), int(w*scale)), \
                        mode='nearest').squeeze(0).numpy().astype(bool) 
    
    if thre is not None:
        conf = bdb2D_pos[:, -1]
        flag = conf >= thre

        f_bdb2D_pos = bdb2D_pos[flag]
        f_mask_objs = mask_objs[flag]
        f_size_cls = size_cls[flag]
        return f_bdb2D_pos, f_mask_objs, f_size_cls, bodybdb2D_pos, mask_body, flag
    else:
        conf = bdb2D_pos[:, -1]
        flag = conf >= 0.0
        return bdb2D_pos, mask_objs, size_cls, bodybdb2D_pos, mask_body, flag


def load_detection(det_fn, scale=1.0, mask=False):
    with open(det_fn, 'r') as file:
        detections = json.load(file)
    bdb2D_pos, mask_objs, size_cls, bodybdb2D_pos, mask_body  = parse_detections(detections, mask=mask)

    if bdb2D_pos.shape[0] > 0:
        bdb2D_pos[:, :-1] = bdb2D_pos[:, :-1] * scale
        
    if mask and len(mask_objs) != 0:
        _, h, w = mask_objs.shape
    elif mask and len(mask_body) != 0:
        _, h, w = mask_body.shape

    if len(mask_objs) != 0 and mask_objs.shape[0] != 0:
        mask_objs = F.interpolate(torch.from_numpy(mask_objs).unsqueeze(0).float(), size=(int(h*scale), int(w*scale)), \
                        mode='nearest').squeeze(0).numpy().astype(bool) #Image.NEAREST)

    bodybdb2D_pos = bodybdb2D_pos * scale
    if mask_body.shape[0] != 0:
        mask_body = F.interpolate(torch.from_numpy(mask_body).unsqueeze(0).float(), size=(int(h*scale), int(w*scale)), \
                        mode='nearest').squeeze(0).numpy().astype(bool) 
    return bdb2D_pos, mask_objs, size_cls, bodybdb2D_pos, mask_body

def parse_detections(detections, mask=False):
    bdb2D_pos = []
    size_cls = []
    bodybdb2D_pos = []
    mask_body = []
    mask_objs = []
    for det in detections:
        if det['class'] == 'person': # ! fixed this bug on 02.12: add person detection results into objs
            bodybdb2D_pos.append(det['bbox'])
            if mask:
                mask_body.append(det['mask'])
        else:
            if mask:
                mask_objs.append(det['mask'])                
            bdb2D_pos.append(det['bbox'])
            size_cls.append(NYU40CLASSES.index(det['class']))
    return np.array(bdb2D_pos), np.array(mask_objs), np.array(size_cls), np.array(bodybdb2D_pos), np.array(mask_body)


def load_perframe_det_results(det_result_dir, img_list, width, filter_flag=None, device=None, use_make=True):
    perframe_det_bbox2D_list = []
    perframe_masks_list = []
    perframe_cam_rois_list = []
    
    for ord_idx, idx in enumerate(tqdm(img_list, desc="load perframe det")):
        if filter_flag is not None and filter_flag[ord_idx] == False:
            continue

        tmp_det_fn = os.path.join(det_result_dir, f'{idx:06d}/detections.json')
        tmp_bdb2D_pos, tmp_mask_objs, tmp_size_cls, tmp_bodybdb2D_pos, tmp_mask_body = load_detection(tmp_det_fn, \
                                scale=width/1920, mask=use_make)
        tmp_masks_person, tmp_masks_object, _, _ =  get_instance_masks_within_global_cam(tmp_mask_objs, \
                                tmp_mask_body, image_size=IMAGE_SIZE)

        if tmp_masks_object.shape[1] == 0: #! missing body
            tmp_masks_object = np.zeros((1, tmp_masks_person.shape[1], tmp_masks_person.shape[2]), dtype=np.uint8)

        if tmp_masks_person is None: #! missing detected object
            perframe_det_bbox2D = torch.cat([torch.from_numpy(tmp_bodybdb2D_pos).to(device)], 0)
            perframe_masks = np.concatenate([tmp_masks_object], 0)
        else:
            perframe_det_bbox2D = torch.cat([torch.from_numpy(tmp_bdb2D_pos).to(device), torch.from_numpy(tmp_bodybdb2D_pos).to(device)], 0) # warning: previous does not work well!
            perframe_masks = np.concatenate([tmp_masks_person, tmp_masks_object], 0)
        perframe_masks = torch.from_numpy(perframe_masks).to(device).type(torch.uint8)

        perframe_cam_rois = None
        perframe_det_bbox2D_list.append(perframe_det_bbox2D)
        perframe_masks_list.append(perframe_masks)
        perframe_cam_rois_list.append(perframe_cam_rois)
    
    return perframe_det_bbox2D_list, perframe_masks_list, perframe_cam_rois_list

def load_det_for_bodies(det_result_dir, img_list, tmp_save_dir, opt, filter_flag, device, preload=False):
    if det_result_dir is not None and det_result_dir != 'None': 
        perframe_det_path = os.path.join(tmp_save_dir, f'perframe_det_dict.pickle')
        if not preload or not os.path.exists(perframe_det_path): # ! will change it into False
            logger.info('load perframe det')
            perframe_det_bbox2D_list, perframe_masks_list, perframe_cam_rois_list = load_perframe_det_results(det_result_dir, img_list, opt.width, filter_flag, device, opt.USE_MASK)
            with open(os.path.join(tmp_save_dir, f'perframe_det_dict.pickle'), 'wb') as fout:
                pickle.dump( \
                    {'perframe_det_bbox2D_list': perframe_det_bbox2D_list, \
                    'perframe_masks_list': perframe_masks_list, \
                    'perframe_cam_rois_list': perframe_cam_rois_list}, \
                fout)
        else:
            logger.info('perload perframe det')
            with open(perframe_det_path, 'rb') as fin:
                perframe_det_dict = pickle.load(fin)
            perframe_det_bbox2D_list = perframe_det_dict['perframe_det_bbox2D_list']
            perframe_masks_list = perframe_det_dict['perframe_masks_list']
            perframe_cam_rois_list = perframe_det_dict['perframe_cam_rois_list']
        
    else:
        perframe_det_bbox2D_list, perframe_masks_list, perframe_cam_rois_list = None, None, None
    
    return perframe_det_bbox2D_list, perframe_masks_list, perframe_cam_rois_list

# def reform_bbox3d(obj_files, bboxes):
#     import copy
#     bboxes = copy.deepcopy(bboxes)
#     array_list = []
#     for obj_file in sorted(obj_files):
#         filename = '.'.join(os.path.basename(obj_file).split('.')[:-1])
#         obj_idx = int(filename.split('_')[0])
#         class_id = int(filename.split('_')[1].split(' ')[0])
#         assert bboxes['class_id'][obj_idx] == class_id
#         if class_id == 31:
#             array_list.append(False)
#             continue
#         else:
#             array_list.append(True)
#         points, faces = load_meshes_from_obj(obj_file)
#         mesh_center = (points.max(0) + points.min(0)) / 2.
#         points = points - mesh_center

#         mesh_coef = (points.max(0) - points.min(0)) / 2.
#         bboxes['coeffs'][obj_idx] = mesh_coef
#     return bboxes


# TODO: need to modified.
def load_total3d_result(result_dir, USE_CAD_SIZE=False, filter=None):
    pre_layout_data = sio.loadmat(os.path.join(result_dir, 'layout.mat'))['layout']
    pre_box_data = sio.loadmat(os.path.join(result_dir, 'bdb_3d.mat'))

    # * filter
    pre_boxes = format_bbox(pre_box_data, 'prediction', filter=filter)
    all_obj_list = format_obj_list(glob(os.path.join(result_dir, '*.obj')), filter=filter)

    pre_cam_R = sio.loadmat(os.path.join(result_dir, 'r_ex.mat'))['cam_R']
    
    if USE_CAD_SIZE:
        print(f'reform bbox3d based on objs')
        pre_boxes = reform_bbox3d(all_obj_list, pre_boxes)
    
    tmp_pre_boxes = copy.deepcopy(pre_boxes)
    tmp_pre_boxes['centroid'] *= 0.0
    tmp_pre_boxes['basis'] = np.identity(3, dtype=np.float32)[None, :,:].repeat(tmp_pre_boxes['basis'].shape[0], axis=0)
    
    objects_dict = format_mesh(all_obj_list, tmp_pre_boxes)

    trans_matrix = np.array([[[0, 0, 1], [0, -1, 0], [1, 0, 0]]])

    objects_dict["points"] = np.matmul(trans_matrix[0], objects_dict["points"].T).T
    pre_boxes['centroid'] = trans_matrix[0].dot(pre_boxes['centroid'].T).T
    pre_boxes['orient'] = R.from_matrix(pre_boxes['basis']).as_euler('zyx')[:, 1]

    z, _, x = R.from_matrix(pre_cam_R).as_euler('zyx',)
    pre_cam_R = R.from_euler('zyx', [-x, 0, -z]).as_matrix()

    return edict({
        'layout': pre_layout_data,
        'objs': objects_dict,
        'boxes_3d': pre_boxes,
        'pre_cam_R': pre_cam_R
    })


def process_input(scene_result, pare_result, bdb2D_pos, size_cls, obj_size,  device):
    new_result = {}
    for key, value in scene_result.items():
        if type(value) == edict:
            for t_k, t_v in value.items():
                if type(t_v) == np.ndarray:
                    new_result[key+'_'+t_k] = torch.Tensor(t_v).to(device)
                else:
                    new_result[key+'_'+t_k] = t_v
        elif type(value) == np.ndarray:
            new_result[key] = torch.Tensor(value).to(device)
        else:
            new_result[key] = value
    
    new_result['body'] = pare_result
    new_result['bdb2D_pos'] = torch.from_numpy(bdb2D_pos).to(device)
    new_result['size_cls'] = torch.from_numpy(size_cls).to(device)
    new_result['obj_size'] = torch.from_numpy(obj_size).to(device)
    return edict(new_result)

def get_obj_input(data_input, obj_idx=None):
    assert obj_idx is not None
    new_result = {}
    for key, value in data_input.items():
        if key in ['layout', 'pre_cam_R', 'body']:
            new_result[key] = value
        elif 'boxes' in key or key in ['bdb2D_pos', 'size_cls', 'obj_size', \
            'objs_fns', 'objs_idxs', 'objs_class_idxs']:
            new_result[key] = value[obj_idx:obj_idx+1]
        
    if obj_idx == 0:
        start_c_idx = start_f_idx = start_v_idx = 0
        end_v_idx = data_input.objs_points_idx_each_obj[obj_idx].long()
        end_f_idx = data_input.objs_faces_idx_each_obj[obj_idx].long()
        end_c_idx = data_input.objs_contact_cnt_each_obj[obj_idx].long()

    else:
        start_v_idx = data_input.objs_points_idx_each_obj[obj_idx-1].long()
        start_f_idx = data_input.objs_faces_idx_each_obj[obj_idx-1].long()
        start_c_idx = data_input.objs_contact_cnt_each_obj[obj_idx-1].long()
        end_v_idx = data_input.objs_points_idx_each_obj[obj_idx].long()
        end_f_idx = data_input.objs_faces_idx_each_obj[obj_idx].long()
        end_c_idx = data_input.objs_contact_cnt_each_obj[obj_idx].long()

    new_result['objs_points'] = data_input.objs_points[start_v_idx:end_v_idx]
    new_result['objs_faces'] = data_input.objs_faces[start_f_idx:end_f_idx] - start_v_idx
    new_result['objs_contact_idxs'] = data_input.objs_contact_idxs[start_c_idx:end_c_idx]
    new_result['objs_points_idx_each_obj'] = (end_v_idx - start_v_idx)[None]
    new_result['objs_faces_idx_each_obj'] = (end_f_idx - start_f_idx)[None]
    new_result['objs_contact_cnt_each_obj'] = (end_c_idx - start_c_idx)[None]

    assert len(new_result.keys()) == len(data_input.keys())
    return edict(new_result)

def get_posa_results(img_fn_list, posa_dir):
    if posa_dir is None:
        return None
    else:
        filter_contact_list = []
        filter_contact_torch_list = []
        for obj_fn in img_fn_list:
            basename = os.path.basename(obj_fn)
            idx = int(basename.split('.')[0])
            contact_file = os.path.join(posa_dir, f'{idx-1:06d}_sample_00.npy')
            filter_contact_list.append(contact_file)
            contact_labels = np.load(contact_file)
            contact_labels = torch.Tensor(contact_labels > 0.5).type(torch.uint8).cuda()
            filter_contact_torch_list.append(contact_labels)
        return torch.stack(filter_contact_torch_list)

def merge_scene_model(all_dirs, use_last=False):
    
    scene_model = None

    for sub_dir in sorted(all_dirs):
        if not use_last:
            tmp_name = 'model_st0_opt_orientation_translation.pth'
            print(f'load {tmp_name}')
        else:
            import glob
            tmp_name = glob.glob(os.path.join(sub_dir, f'model_obj*_opt_lridx1.pth'))[0]
            print(f'load {tmp_name}')
            tmp_name = tmp_name.split('/')[-1]

        model_path = os.path.join(sub_dir, tmp_name)
        if not os.path.exists(model_path):
            continue
    
        tmp_model = torch.load(model_path)
        if scene_model is None:
            scene_model = tmp_model
        else:
            merge_list = ['translations_object', 'rotations_object', 'int_scales_object']
            for key in merge_list:
                scene_model[key] = torch.cat([scene_model[key], tmp_model[key]])

    return scene_model
            