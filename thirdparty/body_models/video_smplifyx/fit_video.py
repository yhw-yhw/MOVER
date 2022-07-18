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


import time
try:
    import cPickle as pickle
except ImportError:
    import pickle

import sys
import os
import os.path as osp
import numpy as np
import torch

from tqdm import tqdm

from collections import defaultdict
import copy
import cv2
import PIL.Image as pil_img

from ..smplifyx. optimizers import optim_factory

from .fitting_video_monitor import *
from .fitting_video_loss import *
from .fitting_video_orientation_temporal_loss import *
from .create_loss import create_loss

import json
# from psbody.mesh import Mesh
import scipy.sparse as sparse
from loguru import logger
from ..smplifyx.utils_mics.utils import get_image
from ..constants import USE_PROX_VPOSER,\
    SKELETON_IDX, LEFT_HAND_IDX, RIGHT_HAND_IDX, USE_2022_VPOSER

if USE_2022_VPOSER: # TODO: need to check
    #Loading VPoser Body Pose Prior
    from human_body_prior.tools.model_loader import load_model
    from human_body_prior.models.vposer_model import VPoser
else:
    from human_body_prior.tools.model_loader import load_vposer
    from .vposer import VPoserDecoder

# TODO: body_model has its own attributes: betas, thetas; or could also get other betes as input 
# ! warning: 
# ! - update_scene
# ! - update_body
# ! - start_opt_stage
# ! - end_opt_stage

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

def fit_multi_view(img,
                keypoints,
                body_model,
                cameras,
                initialization,
                joint_weights,
                body_pose_prior,
                jaw_prior,
                left_hand_prior,
                right_hand_prior,
                shape_prior,
                expr_prior,
                angle_prior,
                result_fn='out.pkl',
                mesh_fn='out.obj',
                out_img_fn='overlay.png',
                loss_type='multiview_smplify',
                use_cuda=True,
                init_joints_idxs=(9, 12, 2, 5),
                use_face=True,
                use_hands=True,
                data_weights=None,
                body_pose_prior_weights=None,
                hand_pose_prior_weights=None,
                jaw_pose_prior_weights=None,
                shape_weights=None,
                expr_weights=None,
                hand_joints_weights=None,
                face_joints_weights=None,
                depth_loss_weight=1e2,
                interpenetration=True,
                coll_loss_weights=None,
                df_cone_height=0.5,
                penalize_outside=True,
                max_collisions=8,
                point2plane=False,
                part_segm_fn='',
                side_view_thsh=25.,
                rho=100,
                vposer_latent_dim=32,
                vposer_ckpt='',
                use_joints_conf=False,
                interactive=True,
                visualize=False,
                save_meshes=True,
                degrees=None,
                dtype=torch.float32,
                ign_part_pairs=None,
                left_shoulder_idx=2,
                right_shoulder_idx=5,
                start_opt_stage=0,
                end_opt_stage=7,
                batch_size=1,
                ####################
                ### PROX-POSA EXTENSION
                scene_model=None,
                update_scene=False,
                update_body=True,
                render_results=True,
                ## Camera
                camera_mode='moving',
                ## Groud Support Loss
                ground_plane_support=False,
                # ground_plane_value=None,
                gp_support_weights_init=0.0,
                gp_support_weights=None,
                ground_contact_support=False,
                ground_contact_value=None,
                gp_contact_weights=None,
                #penetration
                sdf_penetration=False,
                sdf_penetration_loss_weight=None,
                sdf_dir=None,
                cam2world_dir=None,
                #contact
                contact=False,
                rho_contact=1.0,
                contact_loss_weights=None,
                contact_angle=15,
                contact_body_parts=None,
                body_segments_dir=None,
                load_scene=False,
                scene_dir=None,
                scene=False,
                scene_loss_weight=None,
                
                ## Ordinal Depth
                ordinal_depth=False,
                ordinal_depth_loss_weight=None,
                perframe_mask=None,
                ##
                s2m=False,
                s2m_weights=None,
                m2s=False,
                m2s_weights=None,
                rho_s2m=1,
                rho_m2s=1,
                init_mode=None,
                trans_opt_stages=None,
                viz_mode='mv',
                ## debug
                tb_debug=True,
                tb_logger=None,
                ## pre-load model and Parameters
                pre_smplx_model=None,
                pre_load=False,
                pre_load_pare_pose=False,
                ## pare pose prior
                pare_pose_prior=False,
                pare_pose_weight=None,
                ## ! original camera for joint3d,
                camera_3d=None,
                ## offline posa
                posa_flag=False,
                online_pose=False,
                posa_body_contact_labels=None,
                **kwargs):
    
    ################################ Init
    # assert batch_size == 1, 'PyTorch L-BFGS only supports batch_size == 1'
    logger.info(f'run batch size: {batch_size}')
    assert 'transl' in initialization, 'We need to have global translation before fitting'

    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    if degrees is None:
        degrees = [0, 90, 180, 270]

    if data_weights is None:
        data_weights = [1, ] * 5

    if body_pose_prior_weights is None:
        body_pose_prior_weights = [4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78]

    msg = (
        'Number of Body pose prior weights {}'.format(
            len(body_pose_prior_weights)) +
        ' does not match the number of data term weights {}'.format(
            len(data_weights)))
    assert (len(data_weights) ==
            len(body_pose_prior_weights)), msg

    if use_hands:
        if hand_pose_prior_weights is None:
            hand_pose_prior_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of hand pose prior weights')
        assert (len(hand_pose_prior_weights) ==
                len(body_pose_prior_weights)), msg
        if hand_joints_weights is None:
            hand_joints_weights = [0.0, 0.0, 0.0, 1.0]
            msg = ('Number of Body pose prior weights does not match the' +
                   ' number of hand joint distance weights')
            assert (len(hand_joints_weights) ==
                    len(body_pose_prior_weights)), msg

    if shape_weights is None:
        shape_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
    msg = ('Number of Body pose prior weights = {} does not match the' +
           ' number of Shape prior weights = {}')
    assert (len(shape_weights) ==
            len(body_pose_prior_weights)), msg.format(
                len(shape_weights),
                len(body_pose_prior_weights))

    if use_face:
        if jaw_pose_prior_weights is None:
            jaw_pose_prior_weights = [[x] * 3 for x in shape_weights]
        else:
            jaw_pose_prior_weights = map(lambda x: map(float, x.split(',')),
                                         jaw_pose_prior_weights)
            jaw_pose_prior_weights = [list(w) for w in jaw_pose_prior_weights]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of jaw pose prior weights')
        assert (len(jaw_pose_prior_weights) ==
                len(body_pose_prior_weights)), msg

        if expr_weights is None:
            expr_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
        msg = ('Number of Body pose prior weights = {} does not match the' +
               ' number of Expression prior weights = {}')
        assert (len(expr_weights) ==
                len(body_pose_prior_weights)), msg.format(
                    len(body_pose_prior_weights),
                    len(expr_weights))

        if face_joints_weights is None:
            face_joints_weights = [0.0, 0.0, 0.0, 1.0]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of face joint distance weights')
        assert (len(face_joints_weights) ==
              len(body_pose_prior_weights)), msg

              
    if coll_loss_weights is None:
        coll_loss_weights = [0.0] * len(body_pose_prior_weights)
    msg = ('Number of Body pose prior weights does not match the' +
           ' number of collision loss weights')
    assert (len(coll_loss_weights) ==
            len(body_pose_prior_weights)), msg

    use_vposer = kwargs.get('use_vposer', True)
    vposer, pose_embedding = [None, ] * 2
    if use_vposer:
        pose_embedding = torch.zeros([batch_size, 32],
                                     dtype=dtype, device=device,
                                     requires_grad=True)
        if USE_PROX_VPOSER:
            vposer = VPoserDecoder(vposer_ckpt=vposer_ckpt, latent_dim=vposer_latent_dim,
                               dtype=dtype, **kwargs)
        elif USE_2022_VPOSER:
            #Loading VPoser Body Pose Prior
            from human_body_prior.tools.model_loader import load_model
            from human_body_prior.models.vposer_model import VPoser
            vposer, ps = load_model(expr_dir, model_code=VPoser,
                                        remove_words_in_model_weights='vp_model.',
                                        disable_grad=True)
        else:
            vposer_ckpt = osp.expandvars(vposer_ckpt)
            vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')

        vposer = vposer.to(device=device)
        vposer.eval()

    if use_vposer:
        # import pdb;pdb.set_trace()
        if USE_PROX_VPOSER:
            latent_mean = torch.zeros([batch_size, vposer_latent_dim],device=device,
            requires_grad=True, dtype=dtype)
            body_mean_pose = vposer(latent_mean).detach().cpu()
        else:
            body_mean_pose = torch.zeros([batch_size, vposer_latent_dim],
                                     dtype=dtype)
    else:
        body_mean_pose = body_pose_prior.get_mean().detach().cpu()

    
    # Create the search tree
    search_tree = None
    pen_distance = None
    filter_faces = None
    if interpenetration:
        from mesh_intersection.bvh_search_tree import BVH
        import mesh_intersection.loss as collisions_loss
        from mesh_intersection.filter_faces import FilterFaces

        assert use_cuda, 'Interpenetration term can only be used with CUDA'
        assert torch.cuda.is_available(), \
            'No CUDA Device! Interpenetration term can only be used' + \
            ' with CUDA'

        search_tree = BVH(max_collisions=max_collisions)

        pen_distance = \
            collisions_loss.DistanceFieldPenetrationLoss(
                sigma=df_cone_height, point2plane=point2plane,
                vectorized=True, penalize_outside=penalize_outside)

        if part_segm_fn:
            # Read the part segmentation
            part_segm_fn = os.path.expandvars(part_segm_fn)
            with open(part_segm_fn, 'rb') as faces_parents_file:
                face_segm_data = pickle.load(faces_parents_file,
                                             encoding='latin1')
            faces_segm = face_segm_data['segm']
            faces_parents = face_segm_data['parents']
            # Create the module used to filter invalid collision pairs
            filter_faces = FilterFaces(
                faces_segm=faces_segm, faces_parents=faces_parents,
                ign_part_pairs=ign_part_pairs).to(device=device)


    dataset_name = kwargs['dataset']
    ################################################################ end initialization

    ################################################################ start define optimize parameters

    # betas: Optional[Tensor] = None,
    # body_pose: Optional[Tensor] = None,
    # global_orient: Optional[Tensor] = None,
    # transl: Optional[Tensor] = None,
    betas = torch.zeros((1,  10), requires_grad=True, device=device) # use only one fixed betas

    # TODO: load pre-calculated SMPL-X model and parameters
    beta_precomputed = kwargs.get('beta_precomputed', False)

    # if beta_precomputed:
    #     beta_path = kwargs.get('beta_path',None)
    #     if beta_path:
    #         with open(beta_path,'rb') as pkl_f:
    #             # betas = pickle.load(pkl_f) # TODO: joblib and pickle is different
    #             import joblib
    #             pre_betas = joblib.load(pkl_f)['betas']
    #         betas_num = body_model.betas.shape[1]

    #         betas.data.copy_(torch.from_numpy(pre_betas[:betas_num]))
    #         betas.requires_grad = False  # betas provided externally, not optimized
    #     else:
    #         print('beta_precomputed == True but no beta files (.pkl) found.')
    #         exit()
            
    if start_opt_stage > 0: # the camera is modified from OpenGL CS to OpenCV CS.
        global_orient = torch.zeros((batch_size, 3), requires_grad=True, device=device)
    else:
        tmp_x_inverse = torch.zeros((batch_size, 3), device=device)
        tmp_x_inverse[:, 0] = -3.14
        global_orient = torch.clone(tmp_x_inverse)
        global_orient.requires_grad=True
    transl = torch.zeros((batch_size, 3), requires_grad=True, device=device)
    left_hand_pose = torch.zeros((batch_size, 12), requires_grad=True, device=device)
    right_hand_pose = torch.zeros((batch_size, 12), requires_grad=True, device=device)
    jaw_pose = torch.zeros((batch_size, 3), requires_grad=True, device=device)
    leye_pose = torch.zeros((batch_size, 3), requires_grad=True, device=device)
    reye_pose = torch.zeros((batch_size, 3), requires_grad=True, device=device)
    expression = torch.zeros((batch_size, 10), requires_grad=True, device=device)

    # import pdb;pdb.set_trace()
    # TODO: load pre-optimize stages' results.
    if pre_load:
        if type(pre_smplx_model) == list:
            for idx in range(len(pre_smplx_model)):
                # betas.data[idx:idx+1].copy_(torch.tensor(pre_smplx_model[idx]['betas']))
                betas.data.copy_(torch.tensor(pre_smplx_model[idx]['betas'])) # add on 06.18: it may leads to unmatching error from previous results.
                global_orient.data[idx:idx+1].copy_(torch.tensor(pre_smplx_model[idx]['global_orient']))
                transl.data[idx:idx+1].copy_(torch.tensor(pre_smplx_model[idx]['transl']))
                left_hand_pose.data[idx:idx+1].copy_(torch.tensor(pre_smplx_model[idx]['left_hand_pose']))
                right_hand_pose.data[idx:idx+1].copy_(torch.tensor(pre_smplx_model[idx]['right_hand_pose']))
                jaw_pose.data[idx:idx+1].copy_(torch.tensor(pre_smplx_model[idx]['jaw_pose']))
                leye_pose.data[idx:idx+1].copy_(torch.tensor(pre_smplx_model[idx]['leye_pose']))
                reye_pose.data[idx:idx+1].copy_(torch.tensor(pre_smplx_model[idx]['reye_pose']))
                expression.data[idx:idx+1].copy_(torch.tensor(pre_smplx_model[idx]['expression']))
                if use_vposer:
                    pose_embedding.data[idx:idx+1].copy_(torch.tensor(pre_smplx_model[idx]['pose_embedding']))
        elif type(pre_smplx_model) == dict:
            assert global_orient.shape[0] == pre_smplx_model['global_orient'].shape[0]
            betas.data.copy_(torch.tensor(pre_smplx_model['betas'])) # add on 06.18: it may leads to unmatching error from previous results.
            global_orient.data.copy_(torch.tensor(pre_smplx_model['global_orient']))
            transl.data.copy_(torch.tensor(pre_smplx_model['transl']))
            left_hand_pose.data.copy_(torch.tensor(pre_smplx_model['left_hand_pose']))
            right_hand_pose.data.copy_(torch.tensor(pre_smplx_model['right_hand_pose']))
            jaw_pose.data.copy_(torch.tensor(pre_smplx_model['jaw_pose']))
            leye_pose.data.copy_(torch.tensor(pre_smplx_model['leye_pose']))
            reye_pose.data.copy_(torch.tensor(pre_smplx_model['reye_pose']))
            expression.data.copy_(torch.tensor(pre_smplx_model['expression']))
            if use_vposer:
                pose_embedding.data.copy_(torch.tensor(pre_smplx_model['pose_embedding']))
        else:
            logger.info('Unknown pre load smplx model.')
            assert False
    
    # if pre_load_pare_pose:
    #     # TODO: load pare pose parameters.
    #     assert pre_load == False
    #     for idx in range(len(pre_smplx_model)):
    #         global_orient.data[idx:idx+1].copy_(pre_smplx_model[idx]['global_orient'].data)
    #         transl.data[idx:idx+1].copy_(pre_smplx_model[idx]['transl'].data)
    #         if use_vposer:
    #             # only have body pose
    #             pare_body_pose = pre_smplx_model[idx]['body_pose']
    #             pare_pose_output = vposer.encode(pare_body_pose.view(-1, 63))
    #             # import pdb;pdb.set_trace()
    #             # pare_pose_embedding = pare_pose_output.rsample()
    #             pare_pose_embedding = pare_pose_output.mean
    #             pose_embedding.data[idx:idx+1].copy_(pare_pose_embedding)
    
    # load pare pose as initialization.
    if pre_load_pare_pose:
        #     # TODO: load pare pose parameters.
        assert pre_load == False
        pare_pose_list = []
        pare_pose_flag = []
        for idx in range(len(pre_smplx_model)):
            pare_pose_list.append(pre_smplx_model[idx]['body_pose'].reshape(-1, 63))
            pare_pose_flag.append(pre_smplx_model[idx]['flag'])
        pare_pose = torch.cat(pare_pose_list)
        # import pdb;pdb.set_trace()
        pare_pose_flag = torch.Tensor(pare_pose_flag).to(pare_pose.device).reshape(-1, 1)

        if use_vposer:
            assert USE_PROX_VPOSER == False
            # only have body pose
            pare_body_pose = pre_smplx_model[idx]['body_pose']
            pare_pose_output = vposer.encode(pare_body_pose.view(-1, 63))
            # pare_pose_embedding = pare_pose_output.rsample()
            pare_pose_embedding = pare_pose_output.mean
            pose_embedding.data[idx:idx+1].copy_(pare_pose_embedding)
    else:
        pare_pose = None
        pare_pose_flag = None

    ################################################################ end

    ################################################################ define loss weight
    # Weights used for the pose prior and the shape prior
    opt_weights_dict = {'data_weight': data_weights,
                        'body_pose_weight': body_pose_prior_weights,
                        'shape_weight': shape_weights}
    if use_face:
        opt_weights_dict['face_weight'] = face_joints_weights
        opt_weights_dict['expr_prior_weight'] = expr_weights
        opt_weights_dict['jaw_prior_weight'] = jaw_pose_prior_weights

    if use_hands:
        opt_weights_dict['hand_weight'] = hand_joints_weights
        opt_weights_dict['hand_prior_weight'] = hand_pose_prior_weights
    if interpenetration:
        opt_weights_dict['coll_loss_weight'] = coll_loss_weights

    ## Ground Plane Support Loss
    if ground_plane_support:
        opt_weights_dict['gp_support_loss_weight'] = gp_support_weights
    if ground_contact_support:
        opt_weights_dict['gp_contact_loss_weight'] = gp_contact_weights
    if s2m:
        opt_weights_dict['s2m_weight'] = s2m_weights
    if m2s:
        opt_weights_dict['m2s_weight'] = m2s_weights
    if sdf_penetration: ## Human && Objs inter-penetration loss
        opt_weights_dict['sdf_penetration_loss_weight'] = sdf_penetration_loss_weight
    if contact:
        opt_weights_dict['contact_loss_weight'] = contact_loss_weights

    # TODO: addloss at five place: cfg, cmd_parser, input_def and in fit_mv, and in loss definition
    if scene:
        opt_weights_dict['scene_loss_weight'] = scene_loss_weight
    
    if pare_pose_prior:
        opt_weights_dict['pare_pose_weight'] = pare_pose_weight

    if ordinal_depth:
        opt_weights_dict['ordinal_depth_loss_weight'] = ordinal_depth_loss_weight

    # opt_weights_dict['smooth_2d_weight'] = smooth_2d_weight
    # opt_weights_dict['smooth_3d_weight'] = smooth_3d_weight

    ## Grond Contact Loss
    ground_contact_vertices_ids = None
    if ground_contact_support or True: 
        ground_contact_vertices_ids = []
        # for part in [ 'L_feet_front', 'L_feet_back', 'R_feet_front', 'R_feet_back']: # for self-defined feet contact
        for part in [ 'L_Leg', 'R_Leg']:
            with open(os.path.join(body_segments_dir, part + '.json'), 'r') as f:
                data = json.load(f)
                ground_contact_vertices_ids.append(list(set(data["verts_ind"])))
        ground_contact_vertices_ids = np.stack(ground_contact_vertices_ids)

    contact_vertices_ids = ftov = None
    if contact:
        contact_verts_ids = []
        for part in contact_body_parts:
            with open(os.path.join(body_segments_dir, part + '.json'), 'r') as f:
                data = json.load(f)
                contact_verts_ids.append(list(set(data["verts_ind"])))
        contact_verts_ids = np.concatenate(contact_verts_ids)
        
        vertices = body_model(return_verts=True, body_pose= torch.zeros((batch_size, 63), dtype=dtype, device=device)).vertices
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
    else:
        contact_verts_ids=None,
        ftov=None,
    
    keys = opt_weights_dict.keys()
    opt_weights = [dict(zip(keys, vals)) for vals in
                   zip(*(opt_weights_dict[k] for k in keys
                         if opt_weights_dict[k] is not None))]
    for weight_list in opt_weights:
        for key in weight_list:
            weight_list[key] = torch.tensor(weight_list[key],
                                            device=device,
                                            dtype=dtype)

    keypoint_data = keypoints.unsqueeze(1)
    gt_joints = keypoint_data[:, :, :, :2]
    if use_joints_conf:
        joints_conf = keypoint_data[:, :, :, 2].reshape(len(keypoints), -1)


    # Transfer the data to the correct device
    gt_joints = gt_joints.to(device=device, dtype=dtype)
    if use_joints_conf:
        joints_conf = joints_conf.to(device=device, dtype=dtype)

    # The indices of the joints used for the initialization of the camera
    init_joints_idxs = torch.tensor(init_joints_idxs, device=device)
    body_orientation_loss = create_loss('body_orient_multiview',
                                      init_joints_idxs=init_joints_idxs,
                                      depth_loss_weight=depth_loss_weight,
                                      dtype=dtype).to(device=device)

    # set scene parameters
    # if not update_scene:
    #     for param in scene_model.parameters():
    #         print(f'set params in scene from {param.requires_grad} to {False}')
    #         param.requires_grad = False
    # else:
    #     update_list= ['rotate_cam_roll', 'rotate_cam_pitch', 'rotations_object', 'translations_object']
    #     for key, param in scene_model.named_parameters():
    #         if key in update_list:
    #             print(f'set {key} grad: True')
    #             param.requires_grad = True
    
    ################################################################ define loss
    init_loss = create_loss("3D_joint_loss",
                               joint_weights=joint_weights,
                               rho=rho,
                               vposer=vposer,
                               pose_embedding=pose_embedding,
                               body_pose_prior=body_pose_prior,
                               shape_prior=shape_prior,
                               angle_prior=angle_prior,
                               interpenetration=interpenetration,
                               pen_distance=pen_distance,
                               search_tree=search_tree,
                               tri_filtering_module=filter_faces,
                               data_weight=1e1,
                               body_pose_weight=1.0,
                               shape_weight=1e2,
                               dtype=dtype,
                               ground_plane_support=ground_plane_support,
                               gp_support_loss_weight_init=gp_support_weights_init, #1e1
                               ).to(device=device)
    
    if dataset_name == 'Pose2Room':
        ################################
        ### with hand
        ################################
        # import pdb;pdb.set_trace()
        loss = create_loss("3D_joint_hands_loss",
                                joint_weights=joint_weights,
                                rho=rho,
                                use_joints_conf=use_joints_conf,
                                use_face=use_face, use_hands=use_hands,
                                vposer=vposer,
                                pose_embedding=pose_embedding,
                                body_pose_prior=body_pose_prior,
                                shape_prior=shape_prior,
                                angle_prior=angle_prior,
                                left_hand_prior=left_hand_prior,
                                right_hand_prior=right_hand_prior,
                                interpenetration=interpenetration,
                                pen_distance=pen_distance,
                                search_tree=search_tree,
                                tri_filtering_module=filter_faces,
                                data_weight=1e1,
                                body_pose_weight=1.0, #0.5,
                                shape_weight=5,
                                hand_joints_weights=4.0,
                                hand_prior_weight=4.78,
                                dtype=dtype,
                                #    ground_plane_support=ground_plane_support,
                                #    gp_support_loss_weight_init=gp_support_weights_init, #1e1
                                ).to(device=device)
        # use 3D joints for following fitting.
    else:
        # import pdb;pdb.set_trace()
        loss = create_loss("multiview_smplify",
                                joint_weights=joint_weights,
                                rho=rho,
                                use_joints_conf=use_joints_conf,
                                use_face=use_face, use_hands=use_hands,
                                vposer=vposer,
                                pose_embedding=pose_embedding,
                                body_pose_prior=body_pose_prior,
                                shape_prior=shape_prior,
                                angle_prior=angle_prior,
                                expr_prior=expr_prior,
                                left_hand_prior=left_hand_prior,
                                right_hand_prior=right_hand_prior,
                                jaw_prior=jaw_prior,
                                interpenetration=interpenetration,
                                pen_distance=pen_distance,
                                search_tree=search_tree,
                                tri_filtering_module=filter_faces,
                                dtype=dtype,
                                ##HDSR
                                ground_plane_support=ground_plane_support,
                                ground_contact_support=ground_contact_support,
                                ground_contact_vertices_ids=ground_contact_vertices_ids,
                                gp_support_loss_weight=gp_support_weights, # init sample, only one, will be replaced during optimization
                                gp_contact_loss_weight=gp_contact_weights,
                                sdf_penetration=sdf_penetration,
                                scene_model=scene_model,
                                sdf_penetration_loss_weight=sdf_penetration_loss_weight,
                                ## contact loss, TODO: change to POSA
                                contact=contact,
                                contact_verts_ids=contact_verts_ids,
                                rho_contact=rho_contact,
                                contact_angle=contact_angle,
                                ftov=ftov,
                                contact_loss_weight=contact_loss_weights,
                                ## scene loss:
                                scene=scene,
                                scene_loss_weight=scene_loss_weight,
                                ## depth_ordinal
                                ordinal_depth=ordinal_depth,
                                ordinal_depth_loss_weight=ordinal_depth_loss_weight,
                                ## pare pose prior
                                pare_pose_prior=pare_pose_prior,
                                pare_pose_weight=pare_pose_weight,
                                pare_body_pose=pare_pose,
                                pare_body_flag=pare_pose_flag,
                                #    ## video smooth
                                #    smooth_2d_weight=smooth_2d_weight,
                                #    smooth_3d_weight=smooth_3d_weight,
                                ## offline posa
                                posa_flag=posa_flag,
                                online_pose=online_pose,
                                posa_body_contact_labels=posa_body_contact_labels,
                                **kwargs)
    loss = loss.to(device=device)

    # not_running = kwargs.get('not_running', False)
    # if not_running: 
    #     logger.info("not running smplifyx !!!")
    #     return _, _, {'multiview_loss': loss}

    with FittingMonitor(
            batch_size=batch_size, visualize=visualize, **kwargs) as monitor:        

        # Warning: image needs to be original size
        H, _, _ = torch.tensor(get_image(img[0], width=1920, height=1080), dtype=dtype).shape

        data_weight = 1000 / H

         # Step 1: Initialization
         # Two options:
         # 1. Optimize the full pose using 3D skl provided externally (OpenPose or MvPose)
         # 2. Optimize the body orientation using the torso joints
        
        # ! load stage estimated body pose.
        if not pre_load: 
            # TODO: load prefixed results from SMPLify-X
            if start_opt_stage > 0:     # if the starting stage bigger than 0: fit to 3D skeletons
                if dataset_name == 'Pose2Room':
                    
                    # TODO: refactorize.
                    #! same as IDX_MAPPING from smpl_joints_map.
                    # if input 4 dimension, the last dim would be weight.
                    body_pose_joints = np.concatenate((initialization['keypoints_3d'][:, :, :-1], \
                                        initialization['keypoints_3d'][:, :, -1:] * 0.0,
                                        ), -1)
                    
                    body_pose_joints[:, SKELETON_IDX, -1] += 1.0
                    # body_pose_joints[:, LEFT_HAND_IDX, -1] += 1.0
                    # body_pose_joints[:, RIGHT_HAND_IDX, -1] += 1.0
                    gt_joints_3d = torch.tensor(body_pose_joints, device=device)
                    
                else:
                    # ! optimize in OpenGL camera CS, output into OpenCV camera CS
                    # Reset the parameters to mean pose
                    # TODO: kpts_3d is wrong, fixed on 17.01
                    # Add batch_size dimension
                    new_kpts_3d = initialization['keypoints_3d'][:, :15, :3]
                    
                    gt_joints_3d = torch.tensor(new_kpts_3d, device=device)

                if camera_3d is None:
                    camera_3d = cameras
                # ! only woking on single camera
                batch_rotation = camera_3d[0].rotation
                # ! modify gt_joints_3d from world CS into camera CS on 27.07.
                # gt_joints_3d = torch.transpose(torch.matmul(torch.transpose(batch_rotation, 2, 1), torch.transpose(gt_joints_3d, 2, 1)), 2, 1)

                # !!!  the initialization is wrong but the result is good.  Why use correct result, it is wierd.
                #gt_joints_3d[:, :, 1] = -1 * gt_joints_3d[:, :, 1] 

                # if beta_precomputed:
                #     body_model.reset_params(body_pose=body_mean_pose, betas=torch.tensor(betas[:betas_num], device=device))
                # else:
                #     body_model.reset_params(body_pose=body_mean_pose)

                init_params = [global_orient, transl]
                if use_vposer:
                    init_params.append(pose_embedding)
                
                init_optimizer, init_create_graph = optim_factory.create_optimizer(
                                                        init_params,
                                                        **kwargs)
                init_optimizer.zero_grad()
                # import pdb;pdb.set_trace()

                # build loss closure
                fit_init = monitor.create_fitting_closure(
                                        init_optimizer, body_model, cameras=cameras, # TODO: None
                                        loss=init_loss, gt_joints=gt_joints_3d, create_graph=init_create_graph,
                                        # use defined smplx model params
                                        betas = betas.expand(batch_size, -1),
                                        global_orient = global_orient,
                                        transl = transl,
                                        left_hand_pose = left_hand_pose,
                                        right_hand_pose = right_hand_pose,
                                        jaw_pose = jaw_pose,
                                        leye_pose = leye_pose,
                                        reye_pose = reye_pose,
                                        expression = expression,
                                        use_vposer=use_vposer, vposer=vposer,
                                        pose_embedding=pose_embedding,
                                        scene_model=scene_model,
                                        return_full_pose=True, return_verts=True,
                                        # debugging
                                        tb_debug=tb_debug,
                                        tb_logger=tb_logger,
                                        opt_idx=0)
                # fitting process
                ori_init_start = time.time()
                init_loss_val = monitor.run_fitting(init_optimizer,
                                                        fit_init,
                                                        init_params, body_model, 
                                                        gt_joints=gt_joints_3d,
                                                        use_vposer=use_vposer,
                                                        pose_embedding=pose_embedding,
                                                        vposer=vposer,
                                                        # gt_ground_plane=ground_plane_value,
                                                        #update multiple smplx_model
                                                        betas = betas.expand(batch_size, -1),
                                                        global_orient = global_orient,
                                                        transl = transl,
                                                        left_hand_pose = left_hand_pose,
                                                        right_hand_pose = right_hand_pose,
                                                        jaw_pose = jaw_pose,
                                                        leye_pose = leye_pose,
                                                        reye_pose = reye_pose,
                                                        expression = expression,
                                                        scene_model=scene_model,
                                                        )

                
            else:   
                # ! this part is never used currently.
                # TODO: need to be modified, fitting orientation and translation at first, still have bugs in camera setting.
                # if the starting stage == 0: find global orientation first
                # The closure passed to the optimizer
                # import pdb;pdb.set_trace()
                body_orientation_loss.reset_loss_weights({'data_weight': data_weight})

                # Reset the parameters to estimate the initial translation of the
                # body model
                # if beta_precomputed:
                #     body_model.reset_params(body_pose=body_mean_pose,
                #                             transl=initialization['transl'], #                                           scale=1.0, 
                #                             betas=torch.tensor(betas[:betas_num], device=device))
                # else:
                #     body_model.reset_params(body_pose=body_mean_pose, transl=initialization['transl'])

                body_orientation_opt_params = [global_orient, transl]
                # if use_vposer:
                #     init_params.append(pose_embedding)

                body_orientation_optimizer, body_orientation_create_graph = optim_factory.create_optimizer(
                    body_orientation_opt_params,
                    **kwargs)
                body_orientation_optimizer.zero_grad()
                # The closure passed to the optimizer
                # TODO: modify body_orientation_loss
                
                fit_camera = monitor.create_fitting_closure(
                                        body_orientation_optimizer, body_model, cameras, gt_joints,
                                        body_orientation_loss, create_graph=body_orientation_create_graph,
                                        # use defined smplx model params
                                        betas = betas.expand(batch_size, -1),
                                        global_orient = global_orient,
                                        transl = transl,
                                        left_hand_pose = left_hand_pose,
                                        right_hand_pose = right_hand_pose,
                                        jaw_pose = jaw_pose,
                                        leye_pose = leye_pose,
                                        reye_pose = reye_pose,
                                        expression = expression,
                                        use_vposer=use_vposer, vposer=vposer,
                                        pose_embedding=pose_embedding,
                                        scene_model=scene_model,
                                        return_full_pose=True, return_verts=True,
                                        # debugging
                                        tb_debug=tb_debug,
                                        tb_logger=tb_logger,
                                        opt_idx=0)
                # cameras feeding the initial translation
                # of the camera and the initial pose of the body model.
                ori_init_start = time.time()
                init_loss_val = monitor.run_fitting(body_orientation_optimizer,
                                                        fit_camera,
                                                        body_orientation_opt_params, body_model,
                                                        use_vposer=use_vposer,
                                                        pose_embedding=pose_embedding,
                                                        vposer=vposer,
                                                        # gt_ground_plane=ground_plane_value,
                                                        #update multiple smplx_model
                                                        betas = betas.expand(batch_size, -1),
                                                        global_orient = global_orient,
                                                        transl = transl,
                                                        left_hand_pose = left_hand_pose,
                                                        right_hand_pose = right_hand_pose,
                                                        jaw_pose = jaw_pose,
                                                        leye_pose = leye_pose,
                                                        reye_pose = reye_pose,
                                                        expression = expression,
                                                        scene_model=scene_model,
                                                        )

            if interactive:
                if use_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()

                if start_opt_stage > 0:
                    tqdm.write('Initialized with 3D skl done after {:.4f} sec.'.format(
                        time.time() - ori_init_start))
                    tqdm.write('Initialized with 3D skl final loss {:.4f}'.format(
                        init_loss_val))

                else:
                    tqdm.write('Body orientation initialization done after {:.4f} sec.'.format(
                        time.time() - ori_init_start))
                    tqdm.write('Body orientation initialization final loss {:.4f}'.format(
                        init_loss_val))

        orientations = [global_orient.detach().cpu().numpy()]        
        # orientations = []
        # store here the final error for both orientations,
        # and pick the orientation resulting in the lowest error
        results = []

        
        # load pare pose for further optimization
        if pre_load_pare_pose:
            if use_vposer:
                pose_embedding.data[idx:idx+1].copy_(pare_pose_embedding)
            
        # import pdb;pdb.set_trace()





        ####################
        ##### fit body pose to 2D pose, hand, face
        ####################
        # Step 2: Optimize the full model with 2D joints.
        # import pdb;pdb.set_trace()
        end_opt_stage = min(end_opt_stage, len(opt_weights))
        final_loss_val = 0
        for or_idx, orient in enumerate(tqdm(orientations, desc='Orientation')):
            opt_start = time.time()
            # stage: :6
            for opt_idx, curr_weights in enumerate(tqdm(opt_weights[start_opt_stage:end_opt_stage], desc='Stage')):

                ## change input.
                if dataset_name == 'Pose2Room':
                    
                    # TODO: refactorize.
                    #! same as IDX_MAPPING from smpl_joints_map.
                    # if input 4 dimension, the last dim would be weight.
                    
                    body_pose_joints = np.concatenate((initialization['keypoints_3d'][:, :, :-1], \
                                        initialization['keypoints_3d'][:, :, -1:] * 0.0,
                                        ), -1)
                    
                    body_pose_joints[:, SKELETON_IDX, -1] += 1.0
                    body_pose_joints[:, LEFT_HAND_IDX, -1] += 1.0
                    body_pose_joints[:, RIGHT_HAND_IDX, -1] += 1.0
                    gt_joints = torch.tensor(body_pose_joints, device=device)
                    
                    
                # Warning: ot update the scene, we could also generate a plausible human body
                # set update scene module
                if scene_model is not None:
                    if update_scene:
                        logger.debug('update scene in SMPLify-X')
                        scene_model.set_active_scene(activate_list=[ 'translations_object', 'int_scales_object', 'ground_plane'])
                    # elif update_scene and opt_idx == len(opt_weights[start_opt_stage:]) - 1:
                    #     scene_model.set_active_scene(activate_list=[ 'rotations_object', 'translations_object']) #'rotate_cam_pitch', 'rotate_cam_roll',
                    else:
                        # set static scene
                        scene_model.set_static_scene()
                
                # fixed body parameters
                # TODO: use body to optimize 3D scene
                if not update_body: # several stage for update scene only
                    betas.requires_grad = False
                    global_orient.requires_grad = False
                    transl.requires_grad = False
                    left_hand_pose.requires_grad = False
                    right_hand_pose.requires_grad = False
                    jaw_pose.requires_grad = False
                    leye_pose.requires_grad = False
                    reye_pose.requires_grad = False
                    expression.requires_grad = False
                    pose_embedding.requires_grad = False
                    
                else:
                    if not beta_precomputed:
                        #     # WARNING: how to generate accurate pose parameters;
                        betas.requires_grad = True
                    else:
                        betas.requires_grad = False
                    global_orient.requires_grad = True
                    transl.requires_grad = True
                    left_hand_pose.requires_grad = True
                    right_hand_pose.requires_grad = True
                    jaw_pose.requires_grad = True
                    expression.requires_grad = True
                    pose_embedding.requires_grad = True
                
                # if opt_idx + start_opt_stage < 2:
                #     betas.requires_grad = False
                #     global_orient.requires_grad = True
                #     transl.requires_grad = True
                #     left_hand_pose.requires_grad = False
                #     right_hand_pose.requires_grad = False
                #     jaw_pose.requires_grad = False
                #     expression.requires_grad = False
                #     pose_embedding.requires_grad = False

                if dataset_name == 'Pose2Room':
                    jaw_pose.requires_grad = False


                body_params = [betas, global_orient, transl, 
                                                left_hand_pose,
                                                right_hand_pose,
                                                jaw_pose,
                                                leye_pose,
                                                reye_pose,
                                                expression]
                if use_vposer:
                    body_params.append(pose_embedding)

                final_params = list(
                    filter(lambda x: x.requires_grad, body_params))
                
                # use the same optimizer
                if update_scene:
                    scene_params = list(scene_model.parameters())
                    final_scene_params = list(
                        filter(lambda x: x.requires_grad, scene_params))
                    for one in final_scene_params:
                        final_params.append(one)
                    # scene_optimizer, scene_create_graph = optim_factory.create_optimizer(
                    #     final_scene_params,     **kwargs)
                    # scene_optimizer.zero_grad()

                body_optimizer, body_create_graph = optim_factory.create_optimizer(
                    final_params,
                    **kwargs)
                body_optimizer.zero_grad()
                
                if dataset_name != 'Pose2Room':
                    # ! warning set weight for each stage
                    ### only works in 2D projected loss.
                    # import pdb;pdb.set_trace()
                    curr_weights['data_weight'] = data_weight
                    curr_weights['bending_prior_weight'] = (
                        3.17 * curr_weights['body_pose_weight'])
                    if use_hands:
                        joint_weights[:, 25:76] = curr_weights['hand_weight']
                    if use_face:
                        joint_weights[:, 76:] = curr_weights['face_weight']
                    loss.reset_loss_weights(curr_weights)

                # For debug, to skip the following optimization.
                # break;

                # warining: add loss, need to modify create_fitting_closure and run_fitting
                closure = monitor.create_fitting_closure(
                    body_optimizer, body_model,
                    cameras=cameras, gt_joints=gt_joints,
                    joints_conf=joints_conf,
                    joint_weights=joint_weights,
                    loss=loss, create_graph=body_create_graph,
                    use_vposer=use_vposer, vposer=vposer,
                    pose_embedding=pose_embedding, 
                    gt_contact_value=ground_contact_value,
                    scene_model=scene_model,
                    betas = betas.expand(batch_size, -1),
                    global_orient = global_orient,
                    transl = transl,
                    left_hand_pose = left_hand_pose,
                    right_hand_pose = right_hand_pose,
                    jaw_pose = jaw_pose,
                    leye_pose = leye_pose,
                    reye_pose = reye_pose,
                    expression = expression,
                    return_verts=True, return_full_pose=True,
                    tb_debug=tb_debug,
                    tb_logger=tb_logger,
                    opt_idx=opt_idx+start_opt_stage,
                    # pare pose input
                    pare_body_pose=pare_pose,
                    pare_body_flag=pare_pose_flag,
                    )

                if interactive:
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    stage_start = time.time()
                
                #update multiple smplx_model
                final_loss_val = monitor.run_fitting(
                    body_optimizer,
                    closure, final_params,
                    body_model,
                    pose_embedding=pose_embedding, vposer=vposer,
                    use_vposer=use_vposer,
                    betas = betas.expand(batch_size, -1),
                    global_orient = global_orient,
                    transl = transl,
                    left_hand_pose = left_hand_pose,
                    right_hand_pose = right_hand_pose,
                    jaw_pose = jaw_pose,
                    leye_pose = leye_pose,
                    reye_pose = reye_pose,
                    expression = expression,
                    scene_model=scene_model,
                    # pare pose input
                    pare_pose=pare_pose,
                    pare_pose_flag=pare_pose_flag)

                if interactive:
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.time() - stage_start
                    if interactive:
                        tqdm.write('Stage {:03d}/{} done after {:.4f} seconds'.format(
                            opt_idx+start_opt_stage, len(opt_weights), elapsed))

            if final_loss_val is None:
                final_loss_val = -1
            # fix bug in Visulization 
            monitor.close_viewer()
            # print("{} -> {}".format(initialization['transl'], body_model.transl))
            if interactive:
                if use_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.time() - opt_start
                tqdm.write(
                    'Body fitting Orientation {} done after {:.4f} seconds'.format(
                        or_idx, elapsed))
                tqdm.write('Body final loss val = {:.5f}'.format(
                    final_loss_val))

            # Get the result of the fitting process
            # Store in it the errors list in order to compare multiple
            # orientations, if they exist
            result = {}
            result['betas'] = betas.detach().cpu().numpy()
            result['global_orient'] = global_orient.detach().cpu().numpy()
            result['transl'] = transl.detach().cpu().numpy()
            result['left_hand_pose'] = left_hand_pose.detach().cpu().numpy()
            result['right_hand_pose'] = right_hand_pose.detach().cpu().numpy()
            result['jaw_pose'] = jaw_pose.detach().cpu().numpy()
            result['leye_pose'] = leye_pose.detach().cpu().numpy()
            result['reye_pose'] = reye_pose.detach().cpu().numpy()
            result['expression'] = expression.detach().cpu().numpy()
            if use_vposer:
                result['body_pose'] = pose_embedding.detach().cpu().numpy()

            results.append({'loss': final_loss_val,
                            'result': result})
        

        ###### save to pickle and mesh
        # TODO: for single person
        # for idx, one_result_fn in enumerate(result_fn):
        one_result_fn = result_fn[0][:-4] + '_all.pkl'
        with open(one_result_fn, 'wb') as result_file:
            if len(results) > 1:
                min_idx = (0 if results[0]['loss'] < results[1]['loss']
                        else 1)
            else:
                min_idx = 0
            if USE_PROX_VPOSER:
                body_pose = vposer.forward(pose_embedding).view(batch_size, -1) if use_vposer else None
            else:
                body_pose = vposer.decode(
                    pose_embedding,
                    output_type='aa').view(batch_size, -1) if use_vposer else None

            model_type = kwargs.get('model_type', 'smpl')
            append_wrists = model_type == 'smpl' and use_vposer
            if append_wrists:
                wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                        dtype=body_pose.dtype,
                                        device=body_pose.device)
                body_pose = torch.cat([body_pose, wrist_pose], dim=1)

            model_output = body_model(
                        return_verts=True, body_pose=body_pose, 
                        betas = betas,
                        global_orient = global_orient,
                        transl = transl,
                        left_hand_pose = left_hand_pose,
                        right_hand_pose = right_hand_pose,
                        jaw_pose = jaw_pose,
                        leye_pose = leye_pose,
                        reye_pose = reye_pose,
                        expression = expression,
                        )

            results[min_idx]['result']['keypoints_3d'] = model_output.joints.detach().cpu().numpy()[:, :25, :]
            results[min_idx]['result']['body_pose'] = body_pose.detach().cpu().numpy()
            results[min_idx]['result']['pose_embedding'] = pose_embedding.detach().cpu().numpy()
            results[min_idx]['result']['gender'] = kwargs['gender']
            # results[min_idx]['result']['idx'] = idx
            pickle.dump(results[min_idx]['result'], result_file, protocol=2)

        vertices_ = model_output.vertices.detach().cpu().numpy()
        

        ################################
        # To get final optimization information
        if tb_debug and False:
            # import pdb;pdb.set_trace()
            for b_idx in range(vertices_.shape[0]):
                with torch.no_grad():
                    body_model_output = body_model(
                        return_verts=True, body_pose=body_pose[b_idx:b_idx+1].repeat(batch_size, 1), 
                        betas = betas.repeat(batch_size, 1),
                        global_orient = global_orient[b_idx:b_idx+1].repeat(batch_size, 1),
                        transl = transl[b_idx:b_idx+1].repeat(batch_size, 1),
                        left_hand_pose = left_hand_pose[b_idx:b_idx+1].repeat(batch_size, 1),
                        right_hand_pose = right_hand_pose[b_idx:b_idx+1].repeat(batch_size, 1),
                        jaw_pose = jaw_pose[b_idx:b_idx+1].repeat(batch_size, 1),
                        leye_pose = leye_pose[b_idx:b_idx+1].repeat(batch_size, 1),
                        reye_pose = reye_pose[b_idx:b_idx+1].repeat(batch_size, 1),
                        expression = expression[b_idx:b_idx+1].repeat(batch_size, 1),
                        return_full_pose=True
                        )
                    faces_tensor = body_model.faces_tensor.view(-1)
                    total_loss, debug_loss_dict = loss(body_model_output, cameras=cameras,
                                    gt_joints=gt_joints[b_idx:b_idx+1].repeat(batch_size, 1, 1, 1),
                                    body_model_faces=faces_tensor,
                                    joints_conf=joints_conf[b_idx:b_idx+1].repeat(batch_size, 1),
                                    joint_weights=joint_weights,
                                    pose_embedding=pose_embedding[b_idx:b_idx+1].repeat(batch_size, 1),
                                    tb_debug = True,
                                    scene_model=scene_model,
                                    gt_contact_value=ground_contact_value,
                                    **kwargs,
                                )

                    assert tb_logger is not None
                    from .tf_utils import save_scalars
                    save_scalars(tb_logger, f'body_fitting_loss_100Interval', debug_loss_dict, b_idx * 100)
                
        ###############################
        # Get height of the body
        batch_size = body_pose.shape[0]
        tpose_body = body_model(
                        return_verts=True,
                        body_pose= torch.zeros(batch_size, 63).type_as(betas),
                        betas = betas.expand(batch_size, -1),
                        )
        tpose_vertices = tpose_body.vertices[0].detach().cpu().numpy()
        # write out body height
        from ..smplifyx.utils_mics.misc_utils import get_body_height
        body_height = get_body_height(tpose_vertices, body_model.faces_tensor.detach().cpu().numpy())
        with open(f'{result_fn[0][:-4]}_{body_height}.txt', 'w') as fout:
            fout.write(f'body height: {body_height}')


        # save each body into obj file, in world coordinates system.
        import trimesh
        all_out_mesh = []
        if save_meshes or visualize:
            for idx in tqdm(range(vertices_.shape[0]), desc='save mesh'):
                vertices = vertices_[idx]
                world_out_mesh = trimesh.Trimesh(vertices, body_model.faces, process=False)
                all_out_mesh.append(world_out_mesh) # ! warning: save in world CS
                world_out_mesh.export(mesh_fn[idx])
                
                os.environ['PYOPENGL_PLATFORM'] = 'egl'
                # if 'GPU_DEVICE_ORDINAL' in os.environ:
                #     os.environ['EGL_DEVICE_ID'] = os.environ['GPU_DEVICE_ORDINAL'].split(',')[0]
                import pyrender
                cam = cameras[0]
                ci = idx
                # scene
                material = pyrender.MetallicRoughnessMaterial(
                            metallicFactor=0.0,
                            wireframe=True,
                            roughnessFactor=.5,
                            alphaMode='OPAQUE',
                            baseColorFactor=(0.9, 0.5, 0.9, 1))


                scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0])
                
                out_mesh = trimesh.Trimesh(vertices, body_model.faces, process=False)
                # mesh
                rot_mat = cam.rotation.detach().cpu().numpy()[0].squeeze()
                translation = cam.translation.detach().cpu().numpy()[0].squeeze()
                out_mesh.vertices = np.matmul(vertices, rot_mat.T) + translation # ! this is in camera CS
                
                # import pdb;pdb.set_trace()
                # write out cs mesh
                mesh_fn_name = mesh_fn[idx].replace('meshes', 'meshes_cam')
                
                # print(f'ori: {mesh_fn[idx]}', f'name: {mesh_fn_name}')
                os.makedirs(os.path.dirname(mesh_fn_name), exist_ok=True)
                out_mesh.export(f'{mesh_fn_name}_cam_CS.obj')
                
                mesh = pyrender.Mesh.from_trimesh(
                    out_mesh,
                    material=material)

                
                scene.add(mesh, 'mesh')

                # TODO: img, render image is not right!!!
                input_img =get_image(img[ci], no_resize=True)
                if input_img.max() > 1:
                    input_img = input_img.astype(np.float32) /  255.0

                height, width = input_img.shape[:2]
                
                center = cam.center.detach().cpu().numpy().squeeze().tolist()

                camera_pose = np.eye(4)
                # camera_pose = RT
                camera_pose[1, :] = - camera_pose[1, :]
                camera_pose[2, :] = - camera_pose[2, :]

                camera = pyrender.camera.IntrinsicsCamera(
                    fx=cam.focal_length_x[0].item(), fy=cam.focal_length_y[0].item(),
                    cx=center[0], cy=center[1])
                scene.add(camera, pose=camera_pose)

                # Get the lights from the viewer
                light_node = pyrender.DirectionalLight(color=np.ones(3), intensity=2.5)
                scene.add(light_node, pose=camera_pose)

                r = pyrender.OffscreenRenderer(viewport_width=width,
                                                viewport_height=height,
                                                point_size=1.0)
                color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
                color = color.astype(np.float32) / 255.0

                valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]

                output_img = (color[:, :, :-1] * valid_mask +
                                (1 - valid_mask) * input_img)

                output_img = pil_img.fromarray((output_img * 255.).astype(np.uint8))
                output_img.save("{}/../images/{:02}.png".format(kwargs['result_folder'], ci))
                r.delete()

    return copy.deepcopy(all_out_mesh), copy.deepcopy(scene_model), {'multiview_loss': loss}
