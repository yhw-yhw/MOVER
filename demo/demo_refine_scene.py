################################ PROX
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os.path as osp
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import sys
import os
import argparse
import trimesh
import joblib
import glob
import json
import random
import scipy.io
import ast 
from tqdm import tqdm
import time
from loguru import logger
import math
import copy
import pickle
from easydict import EasyDict as edict
import neural_renderer as nr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# third party

############################## major mover model
from mover.utils.cmd_parser import parse_config
from mover.utils.camera import (
    get_pitch_roll_euler_rotation_matrix,
    get_rotation_matrix_mk,
)
from mover.scene_reconstruction import HSR
from mover.dataset import *
from mover.utils_main import *
from mover.utils_optimize import *
# joint learning module
##############################


# add third party libraries
# sys.path.append('../thirdparty')
from thirdparty.body_models.video_smplifyx.main_video import main_video
from thirdparty.body_models.video_smplifyx.tf_utils import save_scalars, save_images
# TODO: specify which part is loaded.
# from thirdparty.body_models import *

if __name__ == "__main__":
    args = parse_config()
    opt = edict(args)
    
    random_sample_order = opt.random_sample_order
    all_frame = opt.all_frame
    random_start_f = opt.start_frame
    original_batch_size = opt.batch_size
    img_list, opt = get_img_list(opt)
    
    ## for noise input of 3D scene reconstruction
    noise_kind = opt.noise_kind
    noise_value = opt.noise_value

    torch.set_default_tensor_type(torch.FloatTensor)

    # ! img_path: 3D scene.
    # ! img_dir: the rest of video with human in it.
    img_path = opt.img_path
    img_dir = opt.img_dir

    cam_inc_fn = opt.cam_inc_fn
    save_dir = opt.save_dir
    st2_render_body_dir = opt.st2_render_body_dir
    pare_dir = opt.pare_dir
    posa_dir = opt.posa_dir
    scene_result_dir = opt.scene_result_dir
    scene_result_path = opt.scene_result_path

    RECALCULATE_HCI_INFO = opt.RECALCULATE_HCI_INFO

    os.makedirs(save_dir, exist_ok=True)

    idx = opt.idx
    save = opt.save
    viz = opt.visualize
    scene_viz = opt.scene_visualize
    device = torch.device('cuda:0')

    tb_logger = get_tb_logger(save_dir, debug=TB_DEBUG)
    
    ori_global_cam_inc = get_cam_intrinsic(cam_inc_fn) 
    print('ori: \n', ori_global_cam_inc)
    global_cam_inc = scale_camera(ori_global_cam_inc, opt.width/1920)
    print('init cam inc: \n')
    print(global_cam_inc)
    ################################ end of config

    ####################################
    ### Load single static scene
    ####################################
    img_path, det_fn = get_scene_det_path(img_path, img_dir, scene_result_path, scene_result_dir, idx)
    
    ####  Single input image for 3D scene in background.
    # bdb2d: x, y, x1, y1; if not exists body, then mask_body.shape==0, mask_body = np.array([])
    bdb2D_pos, mask_objs, size_cls, bodybdb2D_pos, mask_body, filter_det_flag = load_detection_scene(det_fn, \
                                    scale=opt.width/1920, mask=opt.USE_MASK, thre=0.7)
    scene_result, obj_size = load_init_scene(opt, filter=filter_det_flag)
    
    #### load 3D scene from scanned dataset: GT bbox needs to match detection.
    scanned_scene, gt_3dbbox_results = load_scanned_data(opt, bdb2D_pos, size_cls)

    #### load detect mask
    masks_object, masks_person, target_masks, K_rois = preprocess_input_mask(opt, mask_objs, mask_body, \
                                    global_cam_inc, bdb2D_pos, IMAGE_SIZE, device)

    image = load_img_Image(img_path, (opt.width, opt.height))    
    obj_num = bdb2D_pos.shape[0]    
    # Total3D: load object template, and init translation, init orientation.
    # GTCAD: object template, zero translation and zero orientation. | or estimated by other orientation sample

    #### load single human as template
    from thirdparty.body_models.smplifyx.utils_mics.data_parser import create_dataset
    dataset_obj = create_dataset(**opt)
    smplfx_data_input = dataset_obj.__getitem__(img_list)
    img_fn_list = smplfx_data_input['img_path']
    keypoints = torch.from_numpy(smplfx_data_input['keypoints'])
    op_filter_flag = keypoints[:, :, -1].sum(-1) > 0.0
    
    pare_result = None
    human_vertices, human_faces = load_mesh_obj()
    human_vertices = np.zeros((10475, 3)) # human_vertices.shape
    body_params = np.zeros((1, 69)) #((1, 69)) # 1x3 + 1x3 + 1x63
    
    # load init scene: GT Cad / Total3D:
    # 1. GT cad: objects at the original point, transl = 0
    # 2. Total3D: objects at the original point, transl = bbox 3D center
    data_input = process_input(scene_result, pare_result, bdb2D_pos, size_cls, obj_size, device=device)

    #### use preload to load body vertices and perframe detection.
    tmp_save_dir, template_save_dir = create_template_dir(random_sample_order, save_dir, original_batch_size)
    
    ####################################
    ### Load multiple human | single frame.
    ####################################
    if opt.pre_load: # TODO: change the path
        vertices_np, new_pre_smplx_model, all_obj_list = load_all_humans(opt, pare_dir, tmp_save_dir, save_dir, device=device)
        # we need to get 2D kpts confidence in filter body phase.
        _, _, st_2_fit_body_use_dict = main_video({"scene_model":None}, tb_debug=TB_DEBUG, \
                tb_logger=tb_logger, pre_smplx_model=new_pre_smplx_model, not_running=True, **args) # ! in this step, scene_prior is useless.
    else:
        vertices_np, new_pre_smplx_model = None, None

    ####################################
    ### optimization settings
    ####################################
    stage3_dict = get_setting_scene(opt.stage3_kind_flag, opt=opt)
    
    # optimizae parameters
    st3_lr_list = stage3_dict['st3_lr_list']
    st3_num_iterations_list = stage3_dict['st3_num_iterations_list']
    st3_pid_list = stage3_dict['st3_pid_list']
    
    ####################################
    ### get filtered avaliable bodies.
    ####################################
    vertices_np_scene, body2scene_conf, filter_flag, \
    ori_body2scene_conf, filter_obj_list, filter_contact_list, filter_img_list = \
                get_filter_human_poses(tmp_save_dir, stage3_dict, opt, st_2_fit_body_use_dict, \
                vertices_np, new_pre_smplx_model, save_dir, img_list, tb_logger, op_filter_flag, \
                random_sample_order,
                random_start_f,
                original_batch_size,
                all_obj_list, 
                posa_dir,
                st2_render_body_dir,
                img_fn_list
                )
    
    #### #### #### #### #### #### 
    #### Load body mask for calculating ordinal depth using global camera !!!
    #### #### #### #### #### #### 
    det_result_dir = opt.img_dir_det
    img_list = opt.img_list
    perframe_det_bbox2D_list, perframe_masks_list, perframe_cam_rois_list = \
        load_det_for_bodies(det_result_dir, img_list, tmp_save_dir, opt, filter_flag, device, preload=opt.preload_body)
    
    ####################################
    #### contact info for human-scene interaction.
    ####################################
    import pdb;pdb.set_trace()
    if opt.contact:
        st3_ftov = st_2_fit_body_use_dict['ftov']
        st3_contact_verts_ids = st_2_fit_body_use_dict['contact_verts_ids']
        st3_contact_angle = st_2_fit_body_use_dict['contact_angle']
        st3_contact_robustifier = st_2_fit_body_use_dict['contact_robustifier']
    
    
    if opt.running_opt: # False: only works for body2objs.
        lr_list = st3_lr_list
        num_iterations_list = st3_num_iterations_list
        pid_list = st3_pid_list
        
        # define save best single orientation
        start = time.time()
        best_rots = []
        best_trans = []
        best_scales = []
        best_losses = []
        
        ####################################
        ### init 
        ####################################
        best_rots_single = None
        best_trans_single = None
        
        ####################################
        ### define model
        ####################################
        #  ! 1. use a GT CAD model, the translations are zero.
        #  ! 2. use Total3D understanding, the translations are not zero. 
        # batch object vertices
        verts_object_og = data_input.objs_points
        idx_each_object = data_input.objs_points_idx_each_obj
        faces_object = data_input.objs_faces
        idx_each_object_face = data_input.objs_faces_idx_each_obj
        contact_idxs=data_input.objs_contact_idxs
        contact_idx_each_obj=data_input.objs_contact_cnt_each_obj
        sampled_orientation = torch.zeros(obj_num).cuda() 
        
        # ! dat a_input.boxes_3d_orient[obj_idx] is same as basis_object
        if opt.SIGMOID_FOR_SCALE:
            init_scale_object =  torch.zeros((obj_num, 3)).to(device)
        else:
            init_scale_object =  torch.ones((obj_num, 3)).to(device)
        
        model = HSR(
            image=image,
            det_results=data_input.bdb2D_pos, 
            size_cls = data_input.size_cls,
            ori_objs_size = data_input.obj_size,
            translations_object=data_input.boxes_3d_centroid,
            rotations_object=sampled_orientation.unsqueeze(-1),
            basis_object=data_input.boxes_3d_basis, 
            size_scale_object=init_scale_object,
            verts_object_og=verts_object_og,
            idx_each_object=idx_each_object,
            faces_object=faces_object.int(),
            idx_each_object_face=idx_each_object_face,
            # add contact idx of objs
            contact_idxs=contact_idxs,
            contact_idx_each_obj=contact_idx_each_obj,
            K_rois=K_rois,
            K_intrin=torch.Tensor(global_cam_inc).to(device).unsqueeze(0),
            K_extrin=data_input.pre_cam_R.unsqueeze(0),
            cams_params=None, 
            cams_person=None,
            verts_person_og=torch.Tensor(human_vertices).to(device),
            faces_person=torch.IntTensor(human_faces).to(device),
            params_person=torch.from_numpy(body_params).to(device), # rotation, translation, body_pose;
            masks_object=masks_object,
            masks_person=masks_person,
            target_masks=target_masks,
            # perframe_mask
            perframe_masks=perframe_masks_list,
            perframe_det_results=perframe_det_bbox2D_list,
            perframe_cam_rois=perframe_cam_rois_list,
            labels_person=None,
            labels_object=None,
            interaction_map_parts=None,
            int_scale_init=1.0,
            inner_robust_sdf=opt.inner_robust_sdf,
            cluster=opt.cluster, 
            resample_in_sdf=opt.resample_in_sdf, 
            USE_MASK=opt.USE_MASK,
            UPDATE_CAMERA_EXTRIN=opt.UPDATE_CAMERA_EXTRIN,
            USE_ONE_DOF_SCALE=opt.USE_ONE_DOF_SCALE,
            UPDATE_OBJ_SCALE=opt.UPDATE_OBJ_SCALE,
            RECALCULATE_HCI_INFO=opt.RECALCULATE_HCI_INFO,
            SIGMOID_FOR_SCALE=opt.SIGMOID_FOR_SCALE,
            ALL_OBJ_ON_THE_GROUND=opt.ALL_OBJ_ON_THE_GROUND,
            constraint_scale_for_chair=True,
            chair_scale = opt.chair_scale
        )
        model.cuda().float()
        
        ##############################
        ### write a function for this
        ##############################
        # load pre-estimated 3D scene model
        load_scene_init(opt.scene_init_model, model, opt.load_all_scene, \
                    opt.update_gp_camera, noise_kind=noise_kind, noise_value=noise_value)
        
        model.set_init_state()

        if opt.only_rendering:
            if SAVE_ALL_RENDER_RESULT:
                if opt.ONLY_SAVE_FILETER_IMG:
                    filter_img_list = [img_fn_list[tmp_idx] for tmp_idx in range(len(img_list)) if filter_flag[tmp_idx]]
                    output_render_result(filter_img_list, vertices_np_scene, model, save_dir, \
                        f'scene_body_end', device, scanned_scene=scanned_scene, tb_debug=False, tb_logger=tb_logger) # Warning: delete when use it to test results.
                else:
                    output_render_result(img_fn_list, vertices_np, model, save_dir, \
                            f'scene_body_end', device, scanned_scene=scanned_scene, filter_list=filter_flag)
                            
            # with torch.no_grad():
            #     model(stage=0, loss_weights=loss_weights, scene_viz=True, save_dir=os.path.join(save_dir, 'st_end'), img_list=filter_img_list)
            sys.exit(0)
    
        ####################################
        ### optimize for 2D cues. 
        ####################################
        # optimization setting, for 2D project loss.
        lr_idx = 0
        pid = pid_list[lr_idx]
        loss_weights = DEFAULT_LOSS_WEIGHTS[f'debug_{pid}']['loss_weight']
        num_iterations = num_iterations_list[lr_idx]
        opt_lr = lr_list[lr_idx]
        save_scalars(tb_logger, f'weight_lr{opt_lr}', loss_weights, 0)

        ### optimize transl. at first.
        optimize_transl(model, loss_weights, save_dir, filter_img_list, load_all_scene=opt.load_all_scene)
        
        RENEW_OPT=opt.RENEW_OPT
        if not RENEW_OPT:  
            scene_params = list(model.parameters())
            final_params = list(
                filter(lambda x: x.requires_grad, scene_params))            
            optimizer = torch.optim.Adam(final_params, lr=opt_lr) # ! opt_lr: never change.

        
        ####################################
        ### start optimize with Human-scene Interaction losses.
        ####################################
        all_iters = 0
        for lr_idx, opt_lr in tqdm(enumerate(lr_list), desc=f'opt_lr'):
                
            pid = pid_list[lr_idx]
            loss_weights = DEFAULT_LOSS_WEIGHTS[f'debug_{pid}']['loss_weight']
            num_iterations = num_iterations_list[lr_idx]
            save_scalars(tb_logger, f'weight_lr{opt_lr}', loss_weights, 0)
            
            tmp_ori_loss_weight = DEFAULT_LOSS_WEIGHTS[f'debug_{pid}']['loss_weight'].copy()
            model.set_init_state()

            NAN_FLAG=False
            min_loss = math.inf 

            # optimization.
            for _ in tqdm(range(num_iterations), desc='iterations'):
                optimizer.zero_grad()
                # scene loss
                tmp_useless, loss_dict = model(stage=0, loss_weights=loss_weights, detailed_obj_loss=True) # loss should be sperated by objects.
                
                # hci loss
                loss_hsi_dict, debug_loss_hsi_dict, detailed_loss_hsi_dict = model(smplx_model_vertices=torch.from_numpy(vertices_np_scene).to(device),
                                            body2scene_conf=torch.from_numpy(body2scene_conf).to(device), 
                                            stage=opt.stage3_idx, loss_weights=loss_weights, \
                                            save_dir=save_dir, \
                                            contact_verts_ids=st3_contact_verts_ids, contact_angle=st3_contact_angle, \
                                            contact_robustifier=st3_contact_robustifier, ftov=st3_ftov, 
                                            img_list=filter_img_list,
                                            ply_file_list=filter_obj_list,
                                            contact_file_list=filter_contact_list,
                                            detailed_obj_loss=True,
                                            obj_idx=opt.input_obj_idx,
                                            all_input_number=len(img_list),
                                            template_save_dir=template_save_dir,
                                            )
                
                # degenerate depth loss: for those contacted objects
                if lr_idx > 0 and 'lw_depth' in loss_weights and loss_weights['lw_depth']>0:
                    # we only consider degenerate the contacted sofa.
                    degenerate_scale = 0.01 
                    higher_scale = 30 
                    sofa_idx = [tmp_i for tmp_i, tmp_v in enumerate(model.size_cls) if tmp_v in [6, 5]] # depth is important for non-contact objects.
                    contacted_sofa_idx = [tmp_i for tmp_i in sofa_idx if tmp_i in model.accumulate_contact_body_body2obj_idx.squeeze().tolist()]
                    for cs_idx in contacted_sofa_idx:
                        if model.size_cls[cs_idx] == 5:
                            detailed_loss_hsi_dict['loss_depth'][cs_idx] = detailed_loss_hsi_dict['loss_depth'][cs_idx] * degenerate_scale * higher_scale
                        else:
                            detailed_loss_hsi_dict['loss_depth'][cs_idx] = detailed_loss_hsi_dict['loss_depth'][cs_idx] * degenerate_scale
            
                        ## degenerate for chair on  SDF
                        lower_sdf_scale = 0.1
                        if model.size_cls[cs_idx] == 5:
                            detailed_loss_hsi_dict['loss_sdf'][cs_idx] = detailed_loss_hsi_dict['loss_sdf'][cs_idx] * lower_sdf_scale 
                        else:
                            detailed_loss_hsi_dict['loss_sdf'][cs_idx] = detailed_loss_hsi_dict['loss_sdf'][cs_idx]

                loss_dict.update(detailed_loss_hsi_dict)
                loss_dict_weighted = {
                        k: loss_dict[k] * loss_weights[k.replace("loss", "lw")] for k in loss_dict
                    }
                
                losses = sum(loss_dict_weighted.values())
                
                loss = losses.sum()
                if NAN_FLAG:
                    import pdb;pdb.set_trace()
                if loss < min_loss:
                    model_dict = copy.deepcopy(model.state_dict())
                    min_loss = loss 
                    
                loss.backward()
                
                if torch.isnan(model.rotations_object.grad).sum() > 0:
                    # ! if exist nan, add parameters jittering. 
                    nan_idx = torch.isnan(model.rotations_object.grad)
                    if nan_idx.sum() > 0:
                        logger.info(f'remove {nan_idx.sum()} nan in model in {_}')
                        
                        model.rotations_object.data[nan_idx] = model.rotations_object.data[nan_idx].detach().clone() + 1e-3
                        nan_idx3 = nan_idx.repeat(1, 3)
                        model.translations_object.data[nan_idx3] = model.translations_object.data[nan_idx3].detach().clone() + 1e-3
                        model.int_scales_object.data[nan_idx3] = model.int_scales_object.data[nan_idx3].detach().clone() + 1e-3
                    continue
                
                optimizer.step()
  
                ####################################
                ### visualize the optimization procedure.
                ####################################
                if opt.debug:
                    message = f'Opt Step 1 {_}/{num_iterations} min_loss: {min_loss}'
                    logger.info(message)
                    add_tb_message(tb_logger, losses, loss_dict, loss_dict_weighted, all_iters)

                    if _ % 50 == 0:
                        
                        message = f'Opt Step 3 {_}/{num_iterations} loss_weight_lossw || '
                        for key, val in loss_hsi_dict.items():
                            message += f' {key}: {val.item():.4f}_{loss_weights[key.replace("loss", "lw")]}_{loss_dict_weighted[key].sum().item()}'
                        message += 'size_info: ' + model.get_size_of_each_objects()
                        logger.info(message)
                        
                        vertices = np.asarray(vertices_np_scene[0], dtype=np.float32)
                        r_vertices = torch.from_numpy(vertices).type_as(model.verts_person_og)
                        model.verts_person_og = r_vertices

                        if _ == num_iterations-50 or _ % 100 == 0:
                            tmp_result = render_model_to_imgs_pyrender(model, image, scene_viz, f'step{_}', \
                                            save_dir=os.path.join(save_dir, f'st3_debug_{lr_idx}'),
                                    debug=TB_DEBUG, scanned_scene=scanned_scene)
                            save_images(tb_logger, f'st3_debug_{lr_idx}', tmp_result, _)
                        else:
                            render_model_to_imgs_pyrender(model, image, scene_viz, f'step{_}', \
                                            save_dir=os.path.join(save_dir, f'st3_debug_{lr_idx}'), scanned_scene=scanned_scene)

                all_iters += 1
                      
            save_scene_model(model, save_dir, ori_global_cam_inc, f'model_scene_{lr_idx}_lr{opt_lr}_end')    
            
            if SAVE_BEST_MODEL:
                model.load_state_dict(model_dict)
                save_scene_model(model, save_dir, ori_global_cam_inc, f'model_scene_{lr_idx}_lr{opt_lr}_end_best')
                if SAVE_ALL_RENDER_RESULT:
                    if opt.ONLY_SAVE_FILETER_IMG:
                        filter_img_list = [img_fn_list[tmp_idx] for tmp_idx in range(len(img_list)) if filter_flag[tmp_idx]]
                        output_render_result(filter_img_list, vertices_np_scene, model, save_dir, \
                            f'scene_{lr_idx}_lr{opt_lr}_end_best', device, scanned_scene=scanned_scene)
                    else:
                        output_render_result(img_fn_list, vertices_np, model, save_dir, \
                            f'scene_{lr_idx}_lr{opt_lr}_end_best', device, scanned_scene=scanned_scene, filter_list=filter_flag)
            
            best_estimated_rots = model.rotations_object
            best_estimated_trans = model.translations_object
            best_estimated_scales = model.int_scales_object

            ####################################
            ### get body assign to objects results: after 2D cues optimizations;
            ####################################
            if lr_idx == 0 and opt.input_obj_idx == -1: # do not recalculate.
                # run new videos, definitely needed.
                # TODO: add load assign
                assigned_result = model.assign_contact_body_to_objs(
                                                ply_file_list=filter_obj_list,
                                                contact_file_list=filter_contact_list,
                                                ftov=st3_ftov, 
                                                contact_parts='body', debug=True, output_folder=template_save_dir
                                                )

                handArm_assigned_result = model.assign_contact_body_to_objs(
                                                ply_file_list=filter_obj_list,
                                                contact_file_list=filter_contact_list,
                                                ftov=st3_ftov, 
                                                contact_parts='handArm', debug=True, output_folder=template_save_dir
                                                )
                
                # This is used for visualizing depth map,
                get_depth_map(model, vertices_np_scene, None, template_save_dir, device)
                
                logger.info(f'REINIT_ORIENT_BY_BODY')
                # after assign bodies to objs, use contacted bodies to re-init the objects' orientation.
                model.reinit_orien_objs_by_contacted_bodies(opt.use_total3d_reinit, \
                    opt_scale_transl=True)

                logger.info(f'REINIT_TRANSL_BY_DEPTH')
                model.reinit_transl_with_depth_map(opt_scale_transl=True)
                    
    else:   
        start = time.time()
        lr_list = st3_lr_list
        num_iterations_list = st3_num_iterations_list
        pid_list = st3_pid_list

        lr_idx = 0
        pid = pid_list[lr_idx]
        loss_weights = DEFAULT_LOSS_WEIGHTS[f'debug_{pid}']['loss_weight']
        num_iterations = num_iterations_list[lr_idx]
        opt_lr = lr_list[lr_idx]
        logger.info(f'pid: {pid}')
        
        best_estimated_rots = torch.zeros((obj_num, 1)).to(device)
        best_estimated_trans = data_input.boxes_3d_centroid
        best_estimated_scales = torch.ones((obj_num, 3)).to(device)
        contact_idxs = None
        contact_idx_each_obj=None

    # ! warning: verts_person_og is under world_CS-> Camera_CS-> render
    if SAVE_ALL_RENDER_RESULT:
        if opt.ONLY_SAVE_FILETER_IMG:
            filter_img_list = [img_fn_list[tmp_idx] for tmp_idx in range(len(img_list)) if filter_flag[tmp_idx]]
            output_render_result(filter_img_list, vertices_np_scene, model, save_dir, \
                f'scene_body_end', device, scanned_scene=scanned_scene, tb_debug=False, tb_logger=tb_logger) # Warning: delete when use it to test results.
        else:
            output_render_result(img_fn_list, vertices_np, model, save_dir, \
                    f'scene_body_end', device, scanned_scene=scanned_scene, filter_list=filter_flag)
                    
    with torch.no_grad():
        model(stage=0, loss_weights=loss_weights, scene_viz=True, save_dir=os.path.join(save_dir, 'st_end'), img_list=filter_img_list)

    elapsed = time.time() - start
    time_msg = time.strftime('%H hours, %M minutes, %S seconds',
                                time.gmtime(elapsed))
    print('Processing Opt Step 3 took: {}'.format(time_msg))