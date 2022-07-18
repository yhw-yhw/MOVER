# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

import os.path as osp

import time
import yaml
import torch

import smplx

# cur_path = os.getcwd()
# sys.path.append(cur_path + '/../')
# from ..smplifyx import *
from ..smplifyx.utils_mics.misc_utils import JointMapper
from ..smplifyx.utils_mics.data_parser import create_dataset
from .fit_video import fit_multi_view
from ..smplifyx.utils_mics.multiview_initializer import VideoInitializer
from ..smplifyx.utils_mics.camera import create_multicameras
from ..smplifyx.utils_mics.prior import create_prior

torch.backends.cudnn.enabled = False
import scipy.sparse as sparse
import json
import numpy as np
def faces_by_vertex_function(f, v, as_sparse_matrix=False):
        import scipy.sparse as sp
        if not as_sparse_matrix:
            faces_by_vertex = [[] for i in range(len(v))]
            for i, face in enumerate(f):
                faces_by_vertex[face[0]].append(i)
                faces_by_vertex[face[1]].append(i)
                faces_by_vertex[face[2]].append(i)
        else:
            row = f.flatten()
            col = np.array([range(f.shape[0])] * 3).T.flatten()
            data = np.ones(len(col))
            faces_by_vertex = sp.csr_matrix((data, (row, col)), shape=(v.shape[0], f.shape[0]))
        return faces_by_vertex


# add scene as constraint, then input multi-frame SMPLX models
def main_video(scene_prior, tb_debug, tb_logger, pre_smplx_model, not_running=False, posa_body_contact_labels=None, **args):
    # posa_body_contact_labels: for offline posa, input batch posa contact labels.

    
    ################################ Store the arguments for the current experiment
    output_folder = args.pop('output_folder')
    output_folder = osp.expandvars(output_folder)
    if not osp.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    conf_fn = osp.join(output_folder, 'conf.yaml')
    with open(conf_fn, 'w') as conf_file:
        yaml.dump(args, conf_file)

    result_folder = args.pop('result_folder', 'results')
    result_folder = osp.join(output_folder, result_folder)
    if not osp.exists(result_folder):
        os.makedirs(result_folder, exist_ok=True)

    mesh_folder = args.pop('mesh_folder', 'meshes')
    mesh_folder = osp.join(output_folder, mesh_folder)
    if not osp.exists(mesh_folder):
        os.makedirs(mesh_folder, exist_ok=True)

    out_img_folder = osp.join(output_folder, 'images')
    if not osp.exists(out_img_folder):
        os.makedirs(out_img_folder, exist_ok=True)

    use_cuda = args.get('use_cuda', True)
    if use_cuda and not torch.cuda.is_available():
        print('CUDA is not available, exiting!')
        sys.exit(-1)
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    float_dtype = args.get('float_dtype', 'float32')
    if float_dtype == 'float64':
        dtype = torch.float64
    elif float_dtype == 'float32':
        dtype = torch.float32
    else:
        raise ValueError('Unknown float type {}, exiting!'.format(float_dtype))


    ################################ creat dataloader
    # TODO: get input dataset
    batch_size = args.get('batch_size',1)
    img_list = args.get('img_list', [1])
    if -1 in img_list:
        img_list = [one for one in range(1, batch_size+1)]
    assert batch_size == len(img_list)

    # Warning: load ground contact value
    import numpy as np
    try:
        ground_contact_path = args.get('ground_contact_path', None)
        ground_contact_array = np.load(ground_contact_path)

        idx_gc = [one-1 for one in img_list]
        ground_contact_value = ground_contact_array[idx_gc]
    except:
        ground_contact_value = np.zeros((len(img_list), 4))
    scene_model = scene_prior['scene_model']
    ground_contact_value = torch.from_numpy(ground_contact_value).to(device=device,
                                                       dtype=dtype)


    ############################### load dataset.
    dataset_obj = create_dataset(**args)
    data_input = dataset_obj.__getitem__(img_list)
    # A weight for every joint of the model
    joint_weights = dataset_obj.get_joint_weights().to(device=device,
                                                       dtype=dtype)
    # Add a fake batch dimension for broadcasting
    joint_weights.unsqueeze_(dim=0)

    # initialization: batch_size x n_joints x 1/3
    # here it always uses the 3D skl from openpose
    start_opt_stage = args.pop('start_opt_stage', 0)
    # transl: bx3; kpt3d: bx25x4
    initialization = VideoInitializer(data_input, result_path=None, **args).get_init_params()
    
    ###################### run on Pose2Room;
    if args.get('dataset') == 'Pose2Room' and not args.get('single'):
        # update batch_size
        print('update batch size')
        new_batch_size = len(data_input['fn'])
        args.update({'batch_size': new_batch_size})

        # update file.
        img_list = [one for one in range(1, new_batch_size+1)]

        # import pdb;pdb.set_trace()
        # update gender 
        if 'Female' in data_input['fn'][0]:
            args.update({'gender': 'female'})
        else:
            args.update({'gender': 'male'})
        
        # update save dir
        output_folder = os.path.join(output_folder, \
                    data_input['fn'][0].split('.')[0])
        result_folder = args.pop('result_folder', 'results')
        result_folder = osp.join(output_folder, result_folder)
        if not osp.exists(result_folder):
            os.makedirs(result_folder, exist_ok=True)

        mesh_folder = args.pop('mesh_folder', 'meshes')
        mesh_folder = osp.join(output_folder, mesh_folder)
        if not osp.exists(mesh_folder):
            os.makedirs(mesh_folder, exist_ok=True)

        out_img_folder = osp.join(output_folder, 'images')
        if not osp.exists(out_img_folder):
            os.makedirs(out_img_folder, exist_ok=True)
            
        
    # Create the camera object
    # cam is a list
    xml_folder = args.get('calib_path', None)
    print('run: ', xml_folder)
    if xml_folder is not None:    
        if xml_folder != '':
            cameras = create_multicameras(xml_folder=xml_folder,
                            dtype=dtype,
                            **args)
        else:
            raise ValueError('Path must be specified!')
    
    # load original cam for Joint3d
    xml_folder = args.get('calib_path_oriJ3d', None)
    if xml_folder is not None and xml_folder != 'None':
        cameras_oriJ3d = create_multicameras(xml_folder=xml_folder,
                        dtype=dtype,
                        **args)
    else:
        cameras_oriJ3d = cameras

    # img and kpts are np.array
    # save img_fn list not load image array
    img = data_input['img_path'] 
    keypoints = torch.from_numpy(data_input['keypoints'])

    ################################ fitting settings
    start = time.time()

    input_gender = args.pop('gender', 'neutral')
    gender_lbl_type = args.pop('gender_lbl_type', 'none')
    max_persons = args.pop('max_persons', -1)
    
    joint_mapper = JointMapper(dataset_obj.get_model2data())

    if not_running:
        args.update({'batch_size': 1})
        # not influence data input
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
    
    print(args.get('model_type'))
    male_model = smplx.create(gender='male', **model_params)
    # SMPL-H has no gender-neutral model
    
    if args.get('model_type') != 'smplh':
        neutral_model = smplx.create(gender='neutral', **model_params)
    female_model = smplx.create(gender='female', **model_params)

    use_hands = args.get('use_hands', True)
    use_face = args.get('use_face', True)

    body_pose_prior = create_prior(
        prior_type=args.get('body_prior_type'),
        dtype=dtype,
        **args)

    jaw_prior, expr_prior = None, None
    if use_face:
        jaw_prior = create_prior(
            prior_type=args.get('jaw_prior_type'),
            dtype=dtype,
            **args)
        expr_prior = create_prior(
            prior_type=args.get('expr_prior_type', 'l2'),
            dtype=dtype, **args)

    left_hand_prior, right_hand_prior = None, None
    if use_hands:
        lhand_args = args.copy()
        lhand_args['num_gaussians'] = args.get('num_pca_comps')
        left_hand_prior = create_prior(
            prior_type=args.get('left_hand_prior_type'),
            dtype=dtype,
            use_left_hand=True,
            **lhand_args)

        rhand_args = args.copy()
        rhand_args['num_gaussians'] = args.get('num_pca_comps')
        right_hand_prior = create_prior(
            prior_type=args.get('right_hand_prior_type'),
            dtype=dtype,
            use_right_hand=True,
            **rhand_args)

    shape_prior = create_prior(
        prior_type=args.get('shape_prior_type', 'l2'),
        dtype=dtype, **args)

    angle_prior = create_prior(prior_type='angle', dtype=dtype)

    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        for c in cameras:
            c.to(device=device)
            if hasattr(c, 'rotation'):
                c.rotation.requires_grad = False
        for c in cameras_oriJ3d:
            c.to(device=device)
            if hasattr(c, 'rotation'):
                c.rotation.requires_grad = False
        

        # cameras = cameras.to(device=device)
        female_model = female_model.to(device=device)
        male_model = male_model.to(device=device)
        if args.get('model_type') != 'smplh':
            neutral_model = neutral_model.to(device=device)
        body_pose_prior = body_pose_prior.to(device=device)
        angle_prior = angle_prior.to(device=device)
        shape_prior = shape_prior.to(device=device)
        if use_face:
            expr_prior = expr_prior.to(device=device)
            jaw_prior = jaw_prior.to(device=device)
        if use_hands:
            left_hand_prior = left_hand_prior.to(device=device)
            right_hand_prior = right_hand_prior.to(device=device)
    else:
        device = torch.device('cpu')

    ################################ save result
    fn = "."
    curr_result_folder = osp.join(result_folder, fn)
    if not osp.exists(curr_result_folder):
        os.makedirs(curr_result_folder)
    curr_mesh_folder = osp.join(mesh_folder, fn)
    if not osp.exists(curr_mesh_folder):
        os.makedirs(curr_mesh_folder)

    curr_img_folder = osp.join(output_folder, 'images', fn)
    if not osp.exists(curr_img_folder):
        os.makedirs(curr_img_folder)


    ############################### gender
    if gender_lbl_type != 'none':
        if gender_lbl_type == 'pd' and 'gender_pd' in dataset_obj[0]:
            gender = dataset_obj[0]['gender_pd'][person_id]
        if gender_lbl_type == 'gt' and 'gender_gt' in dataset_obj[0]:
            gender = dataset_obj[0]['gender_gt'][person_id]
    else:
        gender = input_gender
    
    # if process prox dataset!
    if 'PROX_qualitative_all' in output_folder and not not_running:
        female_subjects_ids = [162, 3452, 159, 3403]
        recording_name = output_folder.split('/')[-2]
        subject_id = int(recording_name.split('_')[1])
        if subject_id in female_subjects_ids:
            gender = 'female'
        else:
            gender = 'male'

    if gender == 'neutral':
        body_model = neutral_model
    elif gender == 'female':
        body_model = female_model
    elif gender == 'male':
        body_model = male_model
    args['gender'] = gender
    
    
    curr_result_fn = []
    curr_mesh_fn = []
    out_img_fn = []
    for img_idx in img_list:
        curr_result_fn.append(osp.join(curr_result_folder,
                                    '{:03d}.pkl'.format(img_idx)))
        curr_mesh_fn.append(osp.join(curr_mesh_folder,
                                '{:03d}.obj'.format(img_idx)))

        out_img_fn.append(osp.join(curr_img_folder, '{:03d}.png'.format(img_idx)))
    os.makedirs(curr_result_folder, exist_ok=True)
    os.makedirs(curr_mesh_folder, exist_ok=True)
    os.makedirs(curr_img_folder, exist_ok=True)
    
    # ground_plane_value = scene_prior['ground_plane']
    if not_running:
        # modify for output
        fitting_body_use_dict = {}
        fitting_body_use_dict['keypoints'] = keypoints

        contact_vertices_ids = ftov = None
        contact = args['contact']
        body_segments_dir = args['body_segments_dir']
        contact_body_parts = args['contact_body_parts']
        if contact:
            contact_verts_ids = []
            for part in contact_body_parts:
                with open(os.path.join(body_segments_dir, part + '.json'), 'r') as f:
                    data = json.load(f)
                    contact_verts_ids.append(list(set(data["verts_ind"])))
            contact_verts_ids = np.concatenate(contact_verts_ids)
            
            # consumption
            vertices = body_model(return_verts=True, body_pose= torch.zeros((1, 63), dtype=dtype, device=device)).vertices
            # calculate normal map of the contact vertices;
            vertices_np = vertices[0].detach().cpu().numpy().squeeze()
            body_faces_np = body_model.faces_tensor.detach().cpu().numpy().reshape(-1, 3)
            
            # import pdb;pdb.set_trace()
            # from psbody.mesh import Mesh
            # m = Mesh(v=vertices_np, f=body_faces_np) # ! Warning: modified for cluster use.
            # ftov = m.faces_by_vertex(as_sparse_matrix=True)

            ftov = faces_by_vertex_function(body_faces_np, vertices_np, as_sparse_matrix=True)

            ftov = sparse.coo_matrix(ftov)
            indices = torch.LongTensor(np.vstack((ftov.row, ftov.col))).to(device)
            values = torch.FloatTensor(ftov.data).to(device)
            shape = ftov.shape
            ftov = torch.sparse.FloatTensor(indices, values, torch.Size(shape))

            fitting_body_use_dict['ftov'] = ftov
            fitting_body_use_dict['contact_verts_ids'] = contact_verts_ids
            # fitting_body_use_dict['rho_contact'] = args['rho_contact']
            fitting_body_use_dict['contact_angle'] = args['contact_angle']

            from ..smplifyx.utils_mics import misc_utils
            # fitting_body_use_dict['contact_robustifier'] = misc_utils.GMoF_unscaled(rho=args['rho_contact']) # in PROX
            fitting_body_use_dict['contact_robustifier'] = misc_utils.GMoF(rho=args['rho_contact']) # in PROX
        else:
            contact_verts_ids=None,
            ftov=None,
            
        return None, None, fitting_body_use_dict
    else:
        fitted_body_model, scene_model, fitting_body_use_dict =fit_multi_view(img, keypoints,
                        body_model=body_model,
                        cameras=cameras,
                        initialization = initialization,
                        joint_weights=joint_weights,
                        dtype=dtype,
                        output_folder=output_folder,
                        result_folder=curr_result_folder,
                        out_img_fn=out_img_fn,
                        result_fn=curr_result_fn,
                        mesh_fn=curr_mesh_fn,
                        shape_prior=shape_prior,
                        expr_prior=expr_prior,
                        body_pose_prior=body_pose_prior,
                        left_hand_prior=left_hand_prior,
                        right_hand_prior=right_hand_prior,
                        jaw_prior=jaw_prior,
                        angle_prior=angle_prior,
                        start_opt_stage=start_opt_stage,
                        # TODO: ground plane is not fixed should be adaptive from scene model
                        scene_model=scene_model,
                        ground_contact_value=ground_contact_value,
                        ## debug
                        tb_debug=tb_debug,
                        tb_logger=tb_logger,
                        pre_smplx_model=pre_smplx_model, # list of smplx model
                        ## depth
                        camera_3d=cameras_oriJ3d, # ! load original for joint 3d.
                        ## posa input
                        posa_body_contact_labels=posa_body_contact_labels,
                        **args)

        elapsed = time.time() - start
        time_msg = time.strftime('%H hours, %M minutes, %S seconds',
                                time.gmtime(elapsed))
        print('Processing the data took: {}'.format(time_msg))
        
        

        return fitted_body_model, scene_model, fitting_body_use_dict

from ..smplifyx.utils_mics.cmd_parser import parse_config

if __name__ == "__main__":
    args = parse_config()
    # 
    scene_prior = {}
    scene_prior['scene_model'] = None

    # tb debug
    TB_DEBUG = False
    if not TB_DEBUG:
        tb_logger = None
    else:
        from tensorboardX import SummaryWriter
        save_dir = args.get('save_dir')
        tb_logger = SummaryWriter(save_dir)

    main_video(scene_prior, tb_debug=TB_DEBUG, tb_logger=tb_logger, pre_smplx_model=[], \
                **args)