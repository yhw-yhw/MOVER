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

import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


from ..smplifyx.utils_mics.mesh_viewer import MeshViewer
from ..smplifyx import utils_mics
from ..smplifyx.utils_mics import misc_utils
from ..smplifyx import fitting as single_view_fitting
from ..constants import SKELETON_IDX, HAND_IDX
# from .fitting_video_orientation_temporal_loss import SMPLifyBodyOrientLoss, MultiViewSMPLifyBodyOrientLoss
from loguru import logger 

from .body_pose_utils import BODY_POSE_TO_OP, FEET_IN_SMPL, FEET_IN_OP
# from .fitting_video_persubject_orientation_temporal import *
# ! SMPLifyX: init in OpenGL CS.

class SMPLifyLoss3D(single_view_fitting.SMPLifyLoss):

    def __init__(self, search_tree=None,
                 pen_distance=None, tri_filtering_module=None,
                 rho=100,
                 body_pose_prior=None,
                 shape_prior=None,
                 angle_prior=None,
                 dtype=torch.float32,
                 data_weight=1.0,
                 body_pose_weight=0.0,
                 shape_weight=0.0,
                 bending_prior_weight=0.0,
                 hand_prior_weight=0.0,
                 expr_prior_weight=0.0, jaw_prior_weight=0.0,
                 reduction='sum',
                 **kwargs):

        super(SMPLifyLoss3D, self).__init__(search_tree=search_tree,
                 pen_distance=pen_distance, tri_filtering_module=tri_filtering_module,
                 rho=rho,
                 body_pose_prior=body_pose_prior,
                 shape_prior=shape_prior,
                 angle_prior=angle_prior,
                 dtype=torch.float32,
                 data_weight=data_weight,
                 body_pose_weight=body_pose_weight,
                 shape_weight=shape_weight,
                 bending_prior_weight=bending_prior_weight,
                 reduction='sum',
                 **kwargs)
        # import pdb;pdb.set_trace()
        self.ground_plane_support = kwargs['ground_plane_support']
        self.gp_support_loss_weight = kwargs['gp_support_loss_weight_init']

    def forward(self, body_model_output, gt_joints,
                body_model_faces,
                use_vposer=False, pose_embedding=None, 
                scene_model=None,
                **kwargs):

        batch_size = gt_joints.shape[0]
        # import pdb;pdb.set_trace()
        # Calculate the distance of the projected joints from
        # the ground truth 2D detections
        
        #### OLD used in CVPR submission.
        # joint_diff = (gt_joints - body_model_output.joints[:,:15,:]) ** 2 / batch_size
        # joint_loss = (torch.sum(joint_diff) * self.data_weight ** 2) / batch_size
        #### end of OLD used in CVPR submission.

        logger.info('run 3D joint loss.')
        # delete face contour in original name
        joint_len = min(gt_joints.shape[1], body_model_output.joints.shape[1])

        print('joints shape', gt_joints.shape[-1])
        # TODO: check the idx.
        if gt_joints.shape[-1] == 3:
            joint_diff = (gt_joints[:, :joint_len, :] - body_model_output.joints[:,:joint_len,:]) ** 2
        elif gt_joints.shape[-1] == 4:
            joint_diff = (gt_joints[:, :joint_len, :-1] - body_model_output.joints[:,:joint_len,:]) ** 2
            joint_diff *= gt_joints[:, :joint_len, -1:]

        joint_loss = (torch.sum(joint_diff) * self.data_weight ** 2)

        # visualizaion: use updated translated 3D joint in Perspective Camera.
        # Render Joint is still not accurate with the original joints.
        if False:
            import pdb;pdb.set_trace()
            cameras = kwargs['cameras']
        
            import cv2
            img_path  = '//ps/scratch/hyi/HCI_dataset/20210109_capture/C0034/hci_test/000379.jpg'
            cv2_img = cv2.imread(img_path)
            cam = cameras[0]
            proj_kpts = cam(gt_joints)
            ori_proj = proj_kpts.squeeze(0).detach().cpu().numpy().T
            for j in range(ori_proj.shape[1]):
                print(ori_proj[0, j], ori_proj[1, j])
                cv2_img = cv2.circle(cv2_img, (int(ori_proj[0, j]), int(ori_proj[1, j])), \
                            radius=5, color=(255, 0, 0), thickness=4)
            
            proj_kpts = cam(body_model_output.joints[:,:15,:])
            ori_proj = proj_kpts.squeeze(0).detach().cpu().numpy().T
            for j in range(ori_proj.shape[1]):
                print(ori_proj[0, j], ori_proj[1, j])
                cv2_img = cv2.circle(cv2_img, (int(ori_proj[0, j]), int(ori_proj[1, j])), \
                            radius=5, color=(0, 0, 255), thickness=4)
            
            # cv2.imshow("2d joint", cv2_img)
            cv2.imwrite("render.png", cv2_img)
            cv2.waitKey(5)
            import pdb;pdb.set_trace()
        
        # Calculate the loss from the Pose prior
        if use_vposer:
            pprior_loss = (pose_embedding.pow(2).sum() *
                           self.body_pose_weight ** 2)
        else:
            pprior_loss = torch.sum(self.body_pose_prior(
                body_model_output.body_pose,
                body_model_output.betas)) * self.body_pose_weight ** 2
        pprior_loss = pprior_loss / batch_size

        shape_loss = torch.sum(self.shape_prior(
            body_model_output.betas)) * self.shape_weight ** 2 / batch_size 

        # Calculate the prior over the joint rotations. This a heuristic used
        # to prevent extreme rotation of the elbows and knees
        body_pose = body_model_output.full_pose[:, 3:66]
        angle_prior_loss = torch.sum(
            self.angle_prior(body_pose)) * self.bending_prior_weight  / batch_size

        total_loss = (joint_loss + pprior_loss + shape_loss + angle_prior_loss)

        message = f'Init joint|weight: {joint_loss}|{self.data_weight.item():.3f},  \
            shape_loss: {shape_loss}|{self.shape_weight}, \
            pprior_loss: {pprior_loss}|{self.body_pose_weight}, \
            angle_prior: {angle_prior_loss}|{self.bending_prior_weight:.3f}, \
            total_loss: {total_loss}'

        # Delete gp plane
        # if self.ground_plane_support:
        #     ## ground plane support loss: human should above the ground plane.
        #     assert scene_model is not None
        #     gt_ground_plane = scene_model.ground_plane
        #     # batch average
        #     # gp_support_loss = F.smooth_l1_loss(torch.max(body_model_output.vertices[:, :, 1]), gt_ground_plane)
        #     # gp_support_loss = gp_support_loss * self.gp_support_loss_weight
        #     gp_support_loss = F.relu(torch.max(body_model_output.vertices[:, :, 1], -1)[0] - gt_ground_plane).mean()
        #     total_loss = total_loss + gp_support_loss * self.gp_support_loss_weight
        #     message += f'gp: {gp_support_loss}'
        
        logger.info(message)

        tb_debug = kwargs['tb_debug']
        if tb_debug:
            debug_loss_dict = {
                'loss_joint': joint_loss,
                'loss_pprior': pprior_loss,
                'loss_shape': shape_loss,
                'loss_angle_prior': angle_prior_loss,
                'loss_total_SMPLifyLoss3D': total_loss,
            }
            # if self.ground_plane_support:
            #     debug_loss_dict['loss_gp_support_body'] = gp_support_loss
            return total_loss, debug_loss_dict
        else:
            return total_loss



####################
### body skeletons add hands.
####################
class SMPLifyLoss3D_withHands(single_view_fitting.SMPLifyLoss):
    def __init__(self, search_tree=None,
                 pen_distance=None, tri_filtering_module=None,
                 rho=100,
                 body_pose_prior=None,
                 shape_prior=None,
                 angle_prior=None,
                 dtype=torch.float32,
                 data_weight=1.0,
                 body_pose_weight=0.0,
                 shape_weight=0.0,
                 bending_prior_weight=0.0,
                 hand_prior_weight=0.0,
                 reduction='sum',
                 hand_joints_weights=1.0,
                 expr_prior_weight=0.0, 
                 jaw_prior_weight=0.0,
                 hand_idx=None,
                 **kwargs):

        super(SMPLifyLoss3D_withHands, self).__init__(search_tree=search_tree,
                 pen_distance=pen_distance, tri_filtering_module=tri_filtering_module,
                 rho=rho,
                 body_pose_prior=body_pose_prior,
                 shape_prior=shape_prior,
                 angle_prior=angle_prior,
                 dtype=torch.float32,
                 data_weight=data_weight,
                 body_pose_weight=body_pose_weight,
                 shape_weight=shape_weight,
                 bending_prior_weight=bending_prior_weight,
                 hand_prior_weight=hand_prior_weight,
                 expr_prior_weight=expr_prior_weight, 
                 jaw_prior_weight=jaw_prior_weight,
                 reduction='sum',
                 **kwargs)
        self.hand_joints_weights = hand_joints_weights

        # self.hand_idx_map=hand_idx
        # import pdb;pdb.set_trace()
    def forward(self, body_model_output, gt_joints,
                body_model_faces,
                use_vposer=False, pose_embedding=None, 
                scene_model=None,
                **kwargs):

        batch_size = gt_joints.shape[0]
        # import pdb;pdb.set_trace()
        # Calculate the distance of the projected joints from
        # the ground truth 2D detections
        
        #### OLD used in CVPR submission.
        # joint_diff = (gt_joints - body_model_output.joints[:,:15,:]) ** 2 / batch_size
        # joint_loss = (torch.sum(joint_diff) * self.data_weight ** 2) / batch_size
        #### end of OLD used in CVPR submission.

        logger.info('run 3D joint loss.')
        # delete face contour in original name
        joint_len = min(gt_joints.shape[1], body_model_output.joints.shape[1])

        print('joints shape', gt_joints.shape[-1])
        # TODO: check the idx.
        if gt_joints.shape[-1] == 3:
            joint_diff = (gt_joints[:, :joint_len, :] - body_model_output.joints[:,:joint_len,:]) ** 2
        elif gt_joints.shape[-1] == 4:
            joint_diff = (gt_joints[:, :joint_len, :-1] - body_model_output.joints[:,:joint_len,:]) ** 2
            joint_diff *= gt_joints[:, :joint_len, -1:]

        

        # TODO: check the data_weight.
        # import pdb;pdb.set_trace()
        ske_loss = (torch.sum(joint_diff[:, SKELETON_IDX]) * self.data_weight ** 2)
        hand_loss = (torch.sum(joint_diff[:, HAND_IDX]) * self.hand_joints_weights ** 2)
        joint_loss = ske_loss + hand_loss
        # Calculate the loss from the Pose prior
        if use_vposer:
            pprior_loss = (pose_embedding.pow(2).sum() *
                           self.body_pose_weight ** 2)
        else:
            pprior_loss = torch.sum(self.body_pose_prior(
                body_model_output.body_pose,
                body_model_output.betas)) * self.body_pose_weight ** 2
            
        pprior_loss = pprior_loss / batch_size

        shape_loss = torch.sum(self.shape_prior(
            body_model_output.betas)) * self.shape_weight ** 2 / batch_size 

        # Calculate the prior over the joint rotations. This a heuristic used
        # to prevent extreme rotation of the elbows and knees
        body_pose = body_model_output.full_pose[:, 3:66]
        angle_prior_loss = torch.sum(
            self.angle_prior(body_pose)) * self.bending_prior_weight  / batch_size

        # Apply the prior on the pose space of the hand
        # import pdb;pdb.set_trace()
        left_hand_prior_loss, right_hand_prior_loss = 0.0, 0.0
        if self.use_hands and self.left_hand_prior is not None:
            left_hand_prior_loss = torch.sum(
                self.left_hand_prior(
                    body_model_output.left_hand_pose)) * \
                self.hand_prior_weight ** 2 / batch_size

        if self.use_hands and self.right_hand_prior is not None:
            right_hand_prior_loss = torch.sum(
                self.right_hand_prior(
                    body_model_output.right_hand_pose)) * \
                self.hand_prior_weight ** 2 / batch_size
                
        total_loss = (joint_loss + pprior_loss + shape_loss + angle_prior_loss \
                + right_hand_prior_loss + left_hand_prior_loss)

        message = f'skeleton: {ske_loss}, hand: {hand_loss}, \
            joint|weight: {joint_loss}|{self.data_weight.item():.3f},  \
            shape_loss: {shape_loss}|{self.shape_weight}, \
            pprior_loss: {pprior_loss}|{self.body_pose_weight}, \
            angle_prior: {angle_prior_loss}|{self.bending_prior_weight:.3f}, \
            left_hand_prior: {left_hand_prior_loss}|{self.hand_prior_weight:.3f}, \
            total_loss: {total_loss}'

        # Delete gp plane
        # if self.ground_plane_support:
        #     ## ground plane support loss: human should above the ground plane.
        #     assert scene_model is not None
        #     gt_ground_plane = scene_model.ground_plane
        #     # batch average
        #     # gp_support_loss = F.smooth_l1_loss(torch.max(body_model_output.vertices[:, :, 1]), gt_ground_plane)
        #     # gp_support_loss = gp_support_loss * self.gp_support_loss_weight
        #     gp_support_loss = F.relu(torch.max(body_model_output.vertices[:, :, 1], -1)[0] - gt_ground_plane).mean()
        #     total_loss = total_loss + gp_support_loss * self.gp_support_loss_weight
        #     message += f'gp: {gp_support_loss}'
        
        logger.info(message)

        tb_debug = kwargs['tb_debug']
        if tb_debug:
            debug_loss_dict = {
                'loss_joint': joint_loss,
                'loss_pprior': pprior_loss,
                'loss_shape': shape_loss,
                'loss_angle_prior': angle_prior_loss,
                'loss_total_SMPLifyLoss3D': total_loss,
                'loss_left_hand_prior': left_hand_prior_loss,
                'loss_right_hand_prior': right_hand_prior_loss,
            }
            # if self.ground_plane_support:
            #     debug_loss_dict['loss_gp_support_body'] = gp_support_loss
            return total_loss, debug_loss_dict
        else:
            return total_loss
        


################################
### based on 2D observation.
################################
# TODO: add online POSA.
class MultiViewSMPLifyLoss(single_view_fitting.SMPLifyLoss):
    def __init__(self, **kwargs):

        super(MultiViewSMPLifyLoss, self).__init__(**kwargs)
        # import pdb;pdb.set_trace()
        ## ground_plane_support
        self.ground_plane_support = kwargs['ground_plane_support']
        if self.ground_plane_support:
            self.register_buffer('gp_support_loss_weight',
                                    torch.tensor(kwargs['gp_support_loss_weight'], dtype=kwargs['dtype']))
        
        ## ground_contact_support
        self.ground_contact_support = kwargs['ground_contact_support']
        if self.ground_contact_support:
            self.register_buffer('gp_contact_loss_weight',
                                    torch.tensor(kwargs['gp_contact_loss_weight'], dtype=kwargs['dtype']))
            self.ground_contact_vertices_ids = kwargs['ground_contact_vertices_ids']
            self.rho_contact = kwargs['rho_contact']
            logger.info(f'rho_contact in gp_contact_robustifier: {kwargs["rho_contact"]}')
            self.gp_contact_robustifier = misc_utils.GMoF_unscaled(rho=self.rho_contact)

        ## human & scene object inter-penetration 
        self.sdf_penetration = kwargs['sdf_penetration']
        if self.sdf_penetration:
            self.register_buffer('sdf_penetration_loss_weight',
                                    torch.tensor(kwargs['sdf_penetration_loss_weight'], dtype=kwargs['dtype']))
        
        ## human & scene object contact loss
        self.contact = kwargs['contact']
        if self.contact or True:
            self.register_buffer('contact_loss_weight',
                                    torch.tensor(kwargs['contact_loss_weight'], dtype=kwargs['dtype']))
            self.contact_verts_ids = kwargs['contact_verts_ids']
            self.rho_contact = kwargs['rho_contact']
            self.contact_angle = kwargs['contact_angle']
            self.ftov = kwargs['ftov']
            self.contact_robustifier = misc_utils.GMoF(rho=self.rho_contact) # rho is the distance. 5cm
        # else:
        #     self.contact_verts_ids = None
        #     self.rho_contact = None
        #     self.contact_angle = None
        #     self.ftov = None
        # self.contact_robustifier = misc_utils.GMoF_unscaled(rho=self.rho_contact) #used in prox


        ## depth ordinal loss
        self.ordinal_depth = kwargs['ordinal_depth']
        if self.ordinal_depth:
            self.register_buffer('ordinal_depth_loss_weight', torch.tensor(kwargs['ordinal_depth_loss_weight'], dtype=kwargs['dtype']))

        self.scene = kwargs['scene']
        if self.scene:
            self.register_buffer('scene_loss_weight',
                                    torch.tensor(kwargs['scene_loss_weight'], dtype=kwargs['dtype']))

        self.loss_use_sum = kwargs['loss_use_sum']
        self.video_smooth = kwargs['video_smooth']
        self.constant_velocity = kwargs['constant_velocity']
        # learn from AMASS dataset
        self.motion_smooth_prior_flag = kwargs['motion_smooth_prior']

        if self.video_smooth:
            self.register_buffer('smooth_2d_weight',
                                    torch.tensor(kwargs['smooth_2d_weight'], dtype=kwargs['dtype']))
            self.register_buffer('smooth_3d_weight',
                                    torch.tensor(kwargs['smooth_3d_weight'], dtype=kwargs['dtype']))

            self.robustifier_3DJoint = misc_utils.GMoF(rho=0.1)

            
            if self.motion_smooth_prior_flag:
                self.register_buffer('motion_prior_weight',
                                    torch.tensor(kwargs['motion_prior_weight'], dtype=kwargs['dtype']))
                sys.path.append('/is/cluster/hyi/workspace/HCI/amass_infiller')
                from motion_smooth_prior import MotionSmoothPrior
                self.motion_smooth_prior = MotionSmoothPrior()
                

        ## pare pose loss weight
        self.pare_pose_prior = kwargs['pare_pose_prior']
        if self.pare_pose_prior:
            self.register_buffer('pare_pose_weight', 
                                torch.tensor(kwargs['pare_pose_weight'], dtype=kwargs['dtype']))

        ## for ordinal depth loss
        self.use_human_depth = kwargs['use_human_depth']

        self.depth_robustifier = None # ! without using it, sometimes, it will fail.
        # self.depth_robustifier = kwargs['depth_robustifier'] # default is None, set 10

        # gp contact loss
        self.rho_gp_flag = None
        self.l1_gp_flag = None

        # posa model for online pose

        # posa pre-calculated contact labels
        self.posa_flag=kwargs['posa_flag']
        self.online_pose=kwargs['online_pose']
        self.posa_body_contact_labels = kwargs['posa_body_contact_labels']

        # codebase selection
        self.use_scene_loss = kwargs['use_scene_loss']
        # import pdb;pdb.set_trace()
        self.save_dir = kwargs['output_folder']


        # TODO: add motion smooth loss term.

    def forward(self, body_model_output, cameras, gt_joints, joints_conf,
                body_model_faces, joint_weights,
                use_vposer=False, pose_embedding=None,
                gt_contact_value=None, 
                scene_model=None,
                perframe_mask=None,
                pare_body_pose=None, pare_body_flag=None,
                **kwargs):
        
        ################################
        # joints_conf: smpl format;
        # gt_joints: openpose 2D joints;
        ################################
        batch_size = gt_joints.shape[0]

        # TODO: the shape is not correct !!!
        gt_joints = gt_joints.transpose(0, 1)
        joints_conf = joints_conf.unsqueeze(0)

        joint_loss = 0

        # for motion smooth loss term.
        smooth_j2d_loss = 0
        smooth_j3d_loss = 0
        motion_prior_loss = 0
        for vi, cam in enumerate(cameras):
            projected_joints = cam(body_model_output.joints)
            # Calculate the weights for each joints
            # batch * n_joints_1
            
            if False:
                import cv2
                img_path  = '//ps/scratch/hyi/HCI_dataset/20210109_capture/C0034/hci_test/000379.jpg'
                cv2_img = cv2.imread(img_path)
                cam = cameras[0]
                ori_proj = gt_joints[vi, :,:15,:].squeeze(0).detach().cpu().numpy().T
                for j in range(ori_proj.shape[1]):
                    print(ori_proj[0, j], ori_proj[1, j])
                    cv2_img = cv2.circle(cv2_img, (int(ori_proj[0, j]), int(ori_proj[1, j])), \
                                radius=5, color=(255, 0, 0), thickness=4)
                
                proj_kpts = cam(body_model_output.joints[:,:15,:])
                ori_proj = proj_kpts.squeeze(0).detach().cpu().numpy().T
                for j in range(ori_proj.shape[1]):
                    print(ori_proj[0, j], ori_proj[1, j])
                    cv2_img = cv2.circle(cv2_img, (int(ori_proj[0, j]), int(ori_proj[1, j])), \
                                radius=5, color=(0, 0, 255), thickness=4)
                

                # cv2.imshow("2d joint", cv2_img)
                cv2.imwrite("mvs_render.png", cv2_img)
                cv2.waitKey(5)
                import pdb;pdb.set_trace()
            
            weights = (joint_weights * joints_conf[vi]
                       if self.use_joints_conf else
                       joint_weights).unsqueeze(dim=-1)

            # Calculate the distance of the projected joints from
            # the ground truth 2D detections
            # TODO: the shape is not correct !!!
            joint_diff = self.robustifier(gt_joints[vi, :, :, :] - projected_joints)
            joint_loss = joint_loss + (torch.sum(weights ** 2 * joint_diff) *
                        self.data_weight ** 2) / batch_size

            if self.video_smooth:
                if self.motion_smooth_prior_flag:
                    # TODO:
                    pass
                    model_verts = body_model_output.vertices
                    model_joints = body_model_output.joints
                    # * factor out missing frame effects.
                    # import pdb;pdb.set_trace()
                    motion_useful_conf = (joints_conf[vi, 1:, :].sum(-1) > 0) & (joints_conf[vi, 0:-1, :].sum(-1) > 0)
                    motion_prior_loss = self.motion_smooth_prior(model_verts, model_joints, conf=motion_useful_conf) # per-single image loss.
                    motion_prior_loss = self.motion_prior_weight * motion_prior_loss
                    
                if self.constant_velocity:
                    # import pdb;pdb.set_trace()
                    if True:
                        ## smooth 2D joint loss
                        joint_conf_diff = joints_conf[vi, 1:-1, :]
                        joints_2d_diff = self.robustifier(projected_joints[2:]+projected_joints[:-2] - 2 * projected_joints[1:-1])
                        weights_diff = (joint_weights * joint_conf_diff
                                if self.use_joints_conf else
                                joint_conf_diff)
                        smooth_j2d_loss = (weights_diff ** 2) * joints_2d_diff.sum(dim=-1)
                        smooth_j2d_loss = torch.cat(
                                [torch.zeros(1, smooth_j2d_loss.shape[1], device=joints_conf.device), smooth_j2d_loss]
                            ).sum(dim=-1)
                        smooth_j2d_loss = (self.smooth_2d_weight ** 2) * smooth_j2d_loss / batch_size
                        smooth_j2d_loss = smooth_j2d_loss.sum()

                        ## smooth 3D joint loss
                        model_joints = body_model_output.joints
                        joints_3d_diff = self.robustifier_3DJoint(model_joints[2:]+model_joints[:-2] - 2 * model_joints[1:-1])
                        smooth_j3d_loss = (weights_diff ** 2) * joints_3d_diff.sum(dim=-1)
                        smooth_j3d_loss = torch.cat(
                                [torch.zeros(1, smooth_j3d_loss.shape[1], device=joints_conf.device), smooth_j3d_loss]
                            ).sum(dim=-1)
                        smooth_j3d_loss = (self.smooth_3d_weight ** 2) * smooth_j3d_loss / batch_size
                        smooth_j3d_loss = smooth_j3d_loss.sum()
                    else: # static motion prior
                        ## smooth 2D joint loss
                        joint_conf_diff = joints_conf[vi, 1:, :]
                        joints_2d_diff = projected_joints[1:] - projected_joints[:-1]
                        weights_diff = (joint_weights * joint_conf_diff
                                if self.use_joints_conf else
                                joint_conf_diff)
                        smooth_j2d_loss = (weights_diff ** 2) * joints_2d_diff.abs().sum(dim=-1)
                        smooth_j2d_loss = torch.cat(
                                [torch.zeros(1, smooth_j2d_loss.shape[1], device=joints_conf.device), smooth_j2d_loss]
                            ).sum(dim=-1)
                        smooth_j2d_loss = (self.smooth_2d_weight ** 2) * smooth_j2d_loss / batch_size
                        smooth_j2d_loss = smooth_j2d_loss.sum()

                        ## smooth 3D joint loss
                        model_joints = body_model_output.joints
                        joints_3d_diff = model_joints[1:] - model_joints[:-1]
                        smooth_j3d_loss = (weights_diff ** 2) * joints_3d_diff.abs().sum(dim=-1)
                        smooth_j3d_loss = torch.cat(
                                [torch.zeros(1, smooth_j3d_loss.shape[1], device=joints_conf.device), smooth_j3d_loss]
                            ).sum(dim=-1)
                        smooth_j3d_loss = (self.smooth_3d_weight ** 2) * smooth_j3d_loss / batch_size
                        smooth_j3d_loss = smooth_j3d_loss.sum()
                        
        pare_pose_prior_loss = 0.0
        if self.pare_pose_prior and self.pare_pose_weight > 0:
            # import pdb;pdb.set_trace()
            assert pare_body_pose.shape[0] == pare_body_flag.shape[0]
            # pose embedding to body pose (angles)
            tmp_body_pose = body_model_output.body_pose
            use_ful = [map_flag != -1 for map_flag in BODY_POSE_TO_OP]
            use_op = [map_flag for map_flag in BODY_POSE_TO_OP if map_flag != -1]
            op_conf_body = torch.ones((batch_size, 21)).type_as(joints_conf) # warining: inverse op conf
            op_conf_body[:, use_ful] = joints_conf[0][:, use_op]

            op_conf_body = op_conf_body.unsqueeze(-1).repeat(1, 1, 3).reshape(batch_size, -1)
            # TODO: modify pare_body_pose detach
            # import pdb;pdb.set_trace()
            # pare_pose_penalty = (1- op_conf_body) * torch.abs(body_model_output.body_pose - pare_body_pose.detach()).pow(2).sqrt()#.reshape(-1, 21, 3).reshape(batch_size, -1)
            pare_pose_penalty = (1 - op_conf_body) * torch.abs(body_model_output.body_pose - pare_body_pose.detach())
            pare_pose_prior_loss = torch.sum(pare_body_flag * pare_pose_penalty, 1)
            pare_pose_prior_loss = (pare_pose_prior_loss * self.pare_pose_weight ** 2).mean()

        # Calculate the loss from the Pose prior
        if use_vposer:
            pprior_loss = (pose_embedding.pow(2).sum() *
                           self.body_pose_weight ** 2)
        else:
            pprior_loss = torch.sum(self.body_pose_prior(
                body_model_output.body_pose,
                body_model_output.betas)) * self.body_pose_weight ** 2
        pprior_loss = pprior_loss / batch_size

        shape_loss = torch.sum(self.shape_prior(
            body_model_output.betas)) * self.shape_weight ** 2 / batch_size
        # Calculate the prior over the joint rotations. This a heuristic used
        # to prevent extreme rotation of the elbows and knees
        body_pose = body_model_output.full_pose[:, 3:66]
        angle_prior_loss = torch.sum(
            self.angle_prior(body_pose)) * self.bending_prior_weight / batch_size

        # Apply the prior on the pose space of the hand
        left_hand_prior_loss, right_hand_prior_loss = 0.0, 0.0
        if self.use_hands and self.left_hand_prior is not None:
            left_hand_prior_loss = torch.sum(
                self.left_hand_prior(
                    body_model_output.left_hand_pose)) * \
                self.hand_prior_weight ** 2 / batch_size

        if self.use_hands and self.right_hand_prior is not None:
            right_hand_prior_loss = torch.sum(
                self.right_hand_prior(
                    body_model_output.right_hand_pose)) * \
                self.hand_prior_weight ** 2 / batch_size

        expression_loss = 0.0
        jaw_prior_loss = 0.0
        if self.use_face:
            expression_loss = torch.sum(self.expr_prior(
                body_model_output.expression)) * \
                self.expr_prior_weight ** 2 / batch_size

            if hasattr(self, 'jaw_prior'):
                jaw_prior_loss = torch.sum(
                    self.jaw_prior(
                        body_model_output.jaw_pose.mul(
                            self.jaw_prior_weight))) / batch_size

        pen_loss = 0.0
        # Calculate the loss due to interpenetration
        if (self.interpenetration and self.coll_loss_weight.item() > 0):
            
            batch_size = gt_joints.shape[1]
            triangles = torch.index_select(
                body_model_output.vertices, 1,
                body_model_faces).view(batch_size, -1, 3, 3)

            with torch.no_grad():
                collision_idxs = self.search_tree(triangles)

            # Remove unwanted collisions
            if self.tri_filtering_module is not None:
                collision_idxs = self.tri_filtering_module(collision_idxs)

            if collision_idxs.ge(0).sum().item() > 0:
                pen_loss = torch.sum(
                    self.coll_loss_weight *
                    self.pen_distance(triangles, collision_idxs)) / batch_size
        
        ############## ! warning: calculate in world coordinates system.
        body_vertices_world = body_model_output.vertices
        
        # gp_support_loss = 0.0
        # if self.ground_plane_support and self.gp_support_loss_weight > 0 and scene_model is not None: 
        #     gt_ground_plane = scene_model.ground_plane
        #     ## ground plane support loss
        #     assert gt_ground_plane is not None
        #     gp_support_loss = F.smooth_l1_loss(torch.max(body_vertices_world[:, :, 1]), gt_ground_plane)
        #     gp_support_loss = gp_support_loss * self.gp_support_loss_weight

        # gp_contact_loss = 0.0
        # if self.ground_contact_support and self.gp_contact_loss_weight > 0 and scene_model is not None :
        #     # import pdb;pdb.set_trace() 
        #     gt_ground_plane = scene_model.ground_plane
        #     assert gt_contact_value is not None
        #     # the distance from heel and toe to the mesh surface should be 
        #     # consist with the distance from joint to the ground plane
            
        #     # gt_contact_value[gt_contact_value==0] = 0 #-0.2
        #     # gp_contact_loss = torch.abs(body_vertices_world[:, self.ground_contact_vertices_ids, 1] - gt_ground_plane)
        #     # gp_contact_loss = (gt_contact_value * gp_contact_loss.mean(-1) * self.gp_contact_loss_weight).sum() / (gt_contact_value.sum().to(torch.float32) + 1e-9)
            
        #     gp_contact_loss = F.smooth_l1_loss(body_vertices_world[:, self.ground_contact_vertices_ids, 1].mean(-1), gt_ground_plane, reduction='none') 

        #     gp_contact_loss = self.contact_robustifier(gp_contact_loss)
        #     gp_contact_loss = self.gp_contact_loss_weight * gp_contact_loss.mean()
        
        # if self.ground_contact_support and self.gp_contact_loss_weight > 0 and scene_model is not None :
        #     # import pdb;pdb.set_trace() 
        #     gt_ground_plane = scene_model.ground_plane
        #     assert gt_contact_value is not None
        #     # the distance from heel and toe to the mesh surface should be 
        #     # consist with the distance from joint to the ground plane
            
        #     gt_contact_value[gt_contact_value==0] = 0 #-0.2
        #     gp_contact_loss = torch.abs(body_vertices_world[:, self.ground_contact_vertices_ids, 1] - gt_ground_plane)
        #     gp_contact_loss = (gt_contact_value * gp_contact_loss.mean(-1) * self.gp_contact_loss_weight).sum() / (gt_contact_value.sum() + 1e-9)
        gp_support_loss = 0.0
        gp_contact_loss = 0.0

        # import pdb;pdb.set_trace()
        if True: # old version for single image contact loss.
            # import pdb;pdb.set_trace()
            if self.ground_plane_support:
                gt_ground_plane = scene_model.ground_plane
                gp_support_loss = F.relu(torch.max(body_vertices_world[:, :, 1], -1)[0] - gt_ground_plane).mean()
                gp_support_loss = gp_support_loss * self.gp_support_loss_weight
                
                
            if self.ground_contact_support and self.gp_contact_loss_weight > 0:
                # contact ids from POSA <-> ground plane
                # TODO: contact_loss(contacted feet <-> ground plane)
                gt_ground_plane = scene_model.ground_plane
                dist_gp = self.gp_contact_robustifier(torch.abs(body_vertices_world[:, :, 1] - gt_ground_plane))
                tmp_feet_label = torch.zeros(self.posa_body_contact_labels.shape).type_as(self.posa_body_contact_labels)
                tmp_feet_label[:, self.ground_contact_vertices_ids.reshape(-1)] = 1
                body_vertices_contact_label_feet = self.posa_body_contact_labels & tmp_feet_label
                # import pdb;pdb.set_trace() 
                if body_vertices_contact_label_feet.sum() > 0:
                    gp_contact_loss = dist_gp[body_vertices_contact_label_feet.squeeze(-1)].mean() * self.gp_contact_loss_weight
                else:
                    gp_contact_loss = torch.zeros(1).cuda()

        elif False:
            if self.ground_contact_support and self.gp_contact_loss_weight > 0:
                # import pdb;pdb.set_trace()
                gt_ground_plane = scene_model.ground_plane
                feet_vertices = body_vertices_world[:, self.ground_contact_vertices_ids, 1] # B, 2, N
                # reliable feet: op conf > thre
                # import pdb;pdb.set_trace()
                # joints_conf[vi][:, FEET_IN_OP]
                feet_conf = joints_conf[vi, :, FEET_IN_OP].reshape(-1, 2, 3)
                # TODO: with using POSA
                feet_conf = feet_conf.mean(-1)#[0] before 0613 # B, 2
                reliable_feet = feet_conf >= 0.4
                logger.info(f'reliable_feet length: {reliable_feet.shape}')
                # use scene object contact information
                # contact_flag = scene_model.get_contact_flag(body_vertices_world)

                # import pdb;pdb.set_trace()
                # if the feet far away from the ground plane
                contact_flag = (gt_ground_plane - torch.max(feet_vertices, -1)[0]) > 0.1

                into_gp = (torch.max(feet_vertices, -1)[0] - gt_ground_plane) > 0
                # not reliable feet, max(feet) < gp, if exists contact with other objects, use GM contact loss
                rho_gp_flag = reliable_feet | (~reliable_feet & contact_flag & ~into_gp)
                logger.info(f'rho_gp_flag length: {rho_gp_flag.sum()}')
                if self.rho_gp_flag is None:
                    self.rho_gp_flag = rho_gp_flag

                ############## GM_contact_ground_loss
                # two feets contact gp
                tmp_gp_contact_dist = F.mse_loss(feet_vertices, gt_ground_plane.expand_as(feet_vertices), reduction='none').sqrt()
                tmp_gp_contact_loss = self.gp_contact_robustifier(tmp_gp_contact_dist)
                # logger.info(f'tmp_gp_contact_loss length: {tmp_gp_contact_loss.shape}')
                reliable_gp_contact_loss = tmp_gp_contact_loss.mean(-1) * self.rho_gp_flag.detach().type(torch.float)

                # gp_contact_loss = self.robustifier(smplx_model_vertices[:, ground_contact_vertices_ids, 1] - self.ground_plane)
                # gp_contact_loss = (ground_contact_value * gp_contact_loss.mean(-1)).sum() / ( num_labels + 1e-9)

                ############## abs_loss
                # two feets support by a ground plane
                # not reliable feet, max(feet) > gp
                l1_gp_dist = torch.abs(torch.max(feet_vertices.reshape(batch_size, -1), -1)[0] - gt_ground_plane).unsqueeze(1).expand(-1, 2)
                # not reliable feet, max(feet) < gp, check contact, if not
                l1_gp_flag = (~reliable_feet) & (into_gp | (~contact_flag))
                logger.info(f'l1_gp_flag length: {l1_gp_flag.sum()}')
                if self.l1_gp_flag is None:
                    self.l1_gp_flag = l1_gp_flag
                gp_l1_loss = l1_gp_dist * self.l1_gp_flag.detach().type(torch.float)

                
                assert (rho_gp_flag | l1_gp_flag).sum() == rho_gp_flag.numel()
                # import pdb;pdb.set_trace()
                # gp_contact_loss = (reliable_gp_contact_loss + gp_l1_loss).mean() # * batch_size
                
                gp_contact_loss = l1_gp_dist.mean()

                if torch.isnan(gp_contact_loss):
                    import pdb;pdb.set_trace()
        else:
            if self.ground_contact_support and self.gp_contact_loss_weight > 0:
                import pdb;pdb.set_trace()
                gt_ground_plane = scene_model.ground_plane
                feet_vertices = body_vertices_world[:, self.ground_contact_vertices_ids, 1] # B, 2, N
                # reliable feet: op conf > thre
                feet_conf = joints_conf[vi, :, FEET_IN_OP].reshape(-1, 2, 3)
                # TODO: with using POSA
                feet_conf = feet_conf.mean(-1)#[0] before 0613 # B, 2
                reliable_feet = feet_conf >= 0.4
                logger.info(f'reliable_feet length: {reliable_feet.shape}')
                # use scene object contact information
                # contact_flag = scene_model.get_contact_flag(body_vertices_world)

                # if the feet far away from the ground plane
                contact_flag = (gt_ground_plane - torch.max(feet_vertices, -1)[0]) > 0.15

                into_gp = (torch.max(feet_vertices, -1)[0] - gt_ground_plane) > 0
                # not reliable feet, max(feet) < gp, if exists contact with other objects, use GM contact loss
                rho_gp_flag = reliable_feet | (~reliable_feet & contact_flag & ~into_gp)
                logger.info(f'rho_gp_flag length: {rho_gp_flag.sum()}')
                if self.rho_gp_flag is None:
                    self.rho_gp_flag = rho_gp_flag

                ############## GM_contact_ground_loss
                # two feets contact gp
                tmp_gp_contact_dist = F.mse_loss(feet_vertices, gt_ground_plane.expand_as(feet_vertices), reduction='none').sqrt()
                tmp_gp_contact_loss = self.gp_contact_robustifier(tmp_gp_contact_dist)
                # logger.info(f'tmp_gp_contact_loss length: {tmp_gp_contact_loss.shape}')
                reliable_gp_contact_loss = tmp_gp_contact_loss.mean(-1) * self.rho_gp_flag.detach().type(torch.float)

                # gp_contact_loss = self.robustifier(smplx_model_vertices[:, ground_contact_vertices_ids, 1] - self.ground_plane)
                # gp_contact_loss = (ground_contact_value * gp_contact_loss.mean(-1)).sum() / ( num_labels + 1e-9)

                ############## abs_loss
                # two feets support by a ground plane
                # not reliable feet, max(feet) > gp
                l1_gp_dist = torch.abs(torch.max(feet_vertices.reshape(batch_size, -1), -1)[0] - gt_ground_plane).unsqueeze(1).expand(-1, 2)
                # not reliable feet and not far away from gp | max(feet) < gp
                l1_gp_flag = ((~reliable_feet) & (~contact_flag)) | into_gp
                logger.info(f'l1_gp_flag length: {l1_gp_flag.sum()} / {l1_gp_flag.numel()}')
                if self.l1_gp_flag is None:
                    self.l1_gp_flag = l1_gp_flag
                gp_l1_loss = l1_gp_dist * self.l1_gp_flag.detach().type(torch.float)

                
                # assert (rho_gp_flag | l1_gp_flag).sum() == rho_gp_flag.numel()
                # import pdb;pdb.set_trace()
                # gp_contact_loss = (reliable_gp_contact_loss + gp_l1_loss).mean() # * batch_size
                gp_contact_loss = (gp_l1_loss).mean()
                if torch.isnan(gp_contact_loss):
                    import pdb;pdb.set_trace()


        # print(kwargs['use_scene_loss'])
        if self.use_scene_loss: # add scene constraints from PROX / POSA
            if False:
                ### scene constraint loss.            
                sdf_penetration_loss = 0.0
                # TODO: check whether is batch size average?
                # TODO: add scene sdf calculation in scene_model.
                if self.sdf_penetration and self.sdf_penetration_loss_weight >0:
                    assert scene_model is not None
                    cnt = 0
                    for idx in range(batch_size):
                        # import pdb;pdb.set_trace()
                        sdf_penetration_loss_tmp = scene_model(body_vertices_world[idx:idx+1], stage=1)
                        if sdf_penetration_loss_tmp > 0:
                            cnt = cnt + 1
                        sdf_penetration_loss = sdf_penetration_loss + sdf_penetration_loss_tmp
                        #stage=11) not reasonable
                    # sdf_penetration_loss = self.sdf_penetration_loss_weight * sdf_penetration_loss / batch_size    
                    sdf_penetration_loss = self.sdf_penetration_loss_weight * sdf_penetration_loss / (cnt+1e-9)
                
                contact_loss = 0.0
                # TODO: add scene v, vn calculation in scene model.
                if self.contact and self.contact_loss_weight >0:
                    # import pdb;pdb.set_trace()
                    assert scene_model is not None
                    cnt = 0 
                    for idx in range(batch_size):
                        # if idx == 7:
                        #     import pdb;pdb.set_trace()
                        tmp_contact = scene_model(body_vertices_world[idx:idx+1], stage=2, contact_verts_ids=self.contact_verts_ids,
                                                contact_angle=self.contact_angle, contact_robustifier=self.contact_robustifier, ftov=self.ftov)
                        # logger.info(f'{idx}: {tmp_contact.item()}')
                        
                        if tmp_contact > 0:
                            cnt = cnt + 1
                        contact_loss = contact_loss + tmp_contact
                    # contact_loss = self.contact_loss_weight * contact_loss / batch_size
                    contact_loss = self.contact_loss_weight * contact_loss / (cnt+1e-9)
                
                # TODO: calculate depth map for scene model, and use it as constraint.
                depth_loss = 0.0 
                if self.ordinal_depth and self.ordinal_depth_loss_weight > 0:
                    cnt = 0
                    assert scene_model is not None
                    for idx in range(batch_size):
                    # input current new per-frame mask.
                        # import pdb;pdb.set_trace()
                        # this 
                        tmp_depth_loss = scene_model.compute_ordinal_depth_loss_perframe(body_vertices_world[idx:idx+1], idx, use_for_human=self.use_human_depth)['loss_depth']
                        # import pdb;pdb.set_trace()
                        if torch.isnan(tmp_depth_loss): # the depth loss is too big, generates large gradient, leads to error !!!
                            import pdb;pdb.set_trace()
                            # tmp_depth_loss = scene_model.compute_ordinal_depth_loss_perframe(body_vertices_world[idx:idx+1], idx, use_for_human=self.use_human_depth)['loss_depth']
                        
                        if tmp_depth_loss > 0:
                            cnt += 1
                        
                        depth_loss = depth_loss + self.ordinal_depth_loss_weight * tmp_depth_loss
                    # depth_loss = depth_loss / batch_size
                    depth_loss = depth_loss / (cnt+1e-9)
                    # ! Warning: add robustifier: for Nan problem, the gradients are so big! on 03.08
                    # import pdb;pdb.set_trace()
                    # if self.depth_robustifier is not None:
                    #     squared_res = depth_loss ** 2
                    #     depth_loss = torch.div(squared_res, squared_res + self.depth_robustifier ** 2)
            else:
                # loss weight > 0: leads to results;
                scene_constraint_loss_weight = {}
                sdf_penetration_loss = torch.tensor(0.0).cuda()
                contact_loss = torch.tensor(0.0).cuda()
                depth_loss = torch.tensor(0.0).cuda() 
                if self.sdf_penetration and self.sdf_penetration_loss_weight >0:
                    scene_constraint_loss_weight['lw_sdf'] = self.sdf_penetration_loss_weight
                if self.contact and self.contact_loss_weight >0:
                    scene_constraint_loss_weight['lw_contact'] = self.contact_loss_weight
                # if self.posa and self.posa_loss_weight >0:
                #     scene_constraint_loss_weight['lw_posa'] = self.posa_loss_weight
                #     posa
                if self.ordinal_depth and self.ordinal_depth_loss_weight > 0:
                    scene_constraint_loss_weight['lw_depth'] = self.ordinal_depth_loss_weight
                scene_loss_dict = scene_model(body_vertices_world, 
                                                save_dir=self.save_dir,
                                                loss_weights=scene_constraint_loss_weight,
                                                stage=3, contact_verts_ids=self.contact_verts_ids,
                                                contact_angle=self.contact_angle, \
                                                contact_robustifier=self.gp_contact_robustifier, ftov=self.ftov) #self.contact_robustifier
                if 'loss_sdf' in scene_loss_dict:
                    sdf_penetration_loss = scene_loss_dict['loss_sdf'] * self.sdf_penetration_loss_weight
                if 'loss_contact' in scene_loss_dict:
                    contact_loss = scene_loss_dict['loss_contact'] * self.contact_loss_weight
                if 'loss_depth' in scene_loss_dict: 
                    depth_loss = scene_loss_dict['loss_depth'] * self.ordinal_depth_loss_weight 
                if 'loss_posa' in scene_loss_dict:
                    contact_loss = scene_loss_dict['loss_contact'] * self.posa_loss_weight

                # best version for only contact & sdf loss: 1, 1
        else:
            sdf_penetration_loss = 0.0
            contact_loss = 0.0
            depth_loss = 0.0 
        scene_loss = 0.0
        message_scene = ''
        # if self.scene and self.scene_loss_weight > 0:
        #     assert scene_model  is not None
        #     init_pid = 0
        #     scene_local_loss_weights = DEFAULT_LOSS_WEIGHTS[f'stage2_init{init_pid}']['loss_weight']
        #     scene_loss_tmp = scene_model(stage=0, loss_weights=scene_local_loss_weights)
        #     scene_loss_dict_weighted = {
        #             k: scene_loss_tmp[k] * scene_local_loss_weights[k.replace("loss", "lw")] for k in scene_loss_tmp
        #         }
            
        #     for key, val in scene_loss_tmp.items():
        #             message_scene += f' {key}: {val.item()}_{scene_local_loss_weights[key.replace("loss", "lw")]}_{scene_loss_dict_weighted[key].item()}'
        #     scene_loss = sum(scene_loss_dict_weighted.values()).sum()
        #     scene_loss = self.scene_loss_weight * scene_loss

        
        # TODO: experiments to analysis;
        message = f'ST2 joint: {joint_loss}, \
            angle_prior: {angle_prior_loss}, pen:{pen_loss}, pare_prior:{pare_pose_prior_loss}, \
            gp: {gp_support_loss}, \
            gp_contact: {gp_contact_loss}, sdf: {sdf_penetration_loss}, contact: {contact_loss}, \
            scene: {scene_loss}, depth: {depth_loss}, smooth_j2d_loss:{smooth_j2d_loss}, smooth_j3d_loss: {smooth_j3d_loss}, motion_smooth_prior: {motion_prior_loss}'
        message += ' || scene details: ' + message_scene
        logger.info(message)
        
        if self.loss_use_sum:
            total_loss = (joint_loss + pprior_loss + shape_loss +
                        angle_prior_loss + pen_loss + pare_pose_prior_loss +
                        jaw_prior_loss + expression_loss +
                        left_hand_prior_loss + right_hand_prior_loss
                        + gp_support_loss + gp_contact_loss + 
                        sdf_penetration_loss + contact_loss + depth_loss + smooth_j2d_loss + smooth_j3d_loss + motion_prior_loss) * batch_size + scene_loss 
        else:
            total_loss = (joint_loss + pprior_loss + shape_loss +
                        angle_prior_loss + pen_loss + pare_pose_prior_loss +
                        jaw_prior_loss + expression_loss +
                        left_hand_prior_loss + right_hand_prior_loss
                        + gp_support_loss + gp_contact_loss + 
                        sdf_penetration_loss + contact_loss + depth_loss + smooth_j2d_loss + smooth_j3d_loss + motion_prior_loss) + scene_loss
        
        if 'tb_debug' in kwargs and kwargs['tb_debug']:
            debug_loss_dict = {
                'loss_joint': joint_loss,
                'loss_pprior': pprior_loss,
                'loss_shape': shape_loss,
                'loss_angle_prior': angle_prior_loss,
                'loss_pare_pose_prior': pare_pose_prior_loss,
                'loss_expression': expression_loss,
                'loss_left_hand_prior': left_hand_prior_loss,
                'loss_right_hand_prior': right_hand_prior_loss,
                'loss_gp_support_body': gp_support_loss,
                'loss_gp_contact':  gp_contact_loss,
                'loss_sdf_penetration': sdf_penetration_loss,
                'loss_contact': contact_loss,
                'loss_scene': scene_loss,
                'loss_depth': depth_loss, 
                'smooth_j3d_loss': smooth_j3d_loss,
                'smooth_j2d_loss': smooth_j2d_loss,
                'motion_prior_loss': motion_prior_loss,
                'loss_total_MultiViewSMPLify': total_loss,
            }

            if self.ground_plane_support:
                debug_loss_dict['SMPLX_weight_gp_support'] = self.gp_support_loss_weight
            if self.ground_contact_support:
                debug_loss_dict['SMPLX_weight_gp_contact'] = self.gp_contact_loss_weight
            if self.sdf_penetration:
                debug_loss_dict['SMPLX_weight_sdf_penetration'] = self.sdf_penetration_loss_weight
            if self.contact:
                debug_loss_dict['SMPLX_weight_contact'] = self.contact_loss_weight
            if self.scene:
                debug_loss_dict['SMPLX_weight_scene'] = self.scene_loss_weight
            if self.ordinal_depth:
                debug_loss_dict['SMPLX_weight_ordinal_depth'] = self.ordinal_depth_loss_weight

            

            if self.scene and self.scene_loss_weight > 0:
                for key, value in scene_local_loss_weights.items():
                    debug_loss_dict[f'scene_weight_{key}'] = value
                for key in scene_loss_tmp.keys():
                    debug_loss_dict[f'loss_scene_{key}_wt'] = scene_loss_dict_weighted[key]
                    debug_loss_dict[f'loss_scene_{key}'] = scene_loss_tmp[key]

            if self.video_smooth:
                debug_loss_dict['smooth_2d_weight'] = self.smooth_2d_weight
                debug_loss_dict['smooth_3d_weight'] = self.smooth_3d_weight
                if self.motion_smooth_prior_flag:
                    debug_loss_dict['motion_prior_weight'] = self.motion_prior_weight

            return total_loss, debug_loss_dict
        else:
            return total_loss