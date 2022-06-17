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

import torch
import torch.nn as nn

from ..smplifyx.utils_mics import misc_utils

from loguru import logger 

# from .fitting_video_loss import *
from .fitting_video_loss import MultiViewSMPLifyLoss

class SMPLifyBodyOrientLoss(nn.Module):

    def __init__(self, init_joints_idxs,
                 reduction='sum',
                 data_weight=1.0, dtype=torch.float32,
                 **kwargs):
        super(SMPLifyBodyOrientLoss, self).__init__()
        self.dtype = dtype        

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer(
            'init_joints_idxs',
            misc_utils.to_tensor(init_joints_idxs, dtype=torch.long))

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                weight_tensor = torch.tensor(loss_weight_dict[key],
                                             dtype=weight_tensor.dtype,
                                             device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def forward(self, body_model_output, camera, gt_joints,
                **kwargs):

        projected_joints = camera(body_model_output.joints)

        joint_error = torch.pow(
            torch.index_select(gt_joints, 1, self.init_joints_idxs) -
            torch.index_select(projected_joints, 1, self.init_joints_idxs),
            2)
        joint_loss = torch.sum(joint_error) * self.data_weight ** 2
       
        return joint_loss

class MultiViewSMPLifyBodyOrientLoss(SMPLifyBodyOrientLoss):

    def __init__(self, init_joints_idxs,
                 reduction='sum',
                 data_weight=1.0, dtype=torch.float32,
                 **kwargs):
        super(MultiViewSMPLifyBodyOrientLoss, self).__init__(init_joints_idxs=init_joints_idxs, data_weight=data_weight)

    def forward(self, body_model_output, cameras, gt_joints,
                **kwargs):
        # import pdb;pdb.set_trace()
        joint_loss = 0
        for v_id, cam in enumerate(cameras):
            projected_joints = cam(body_model_output.joints)
            # logger.info(f'proj: {projected_joints[:2, :]}')
            # logger.info(f'gt: {gt_joints[:2, :]}')
            joint_error = torch.pow(
                torch.index_select(gt_joints[v_id, :, :, :], 1, self.init_joints_idxs) -
                torch.index_select(projected_joints, 1, self.init_joints_idxs),
                2)
            joint_loss += torch.sum(joint_error) * self.data_weight ** 2
       
        message = f'MultiViewSMPLifyBodyOrientLoss joint loss: {joint_loss}'
        logger.info(message)

        # import pdb;pdb.set_trace()
        if 'tb_debug' in kwargs and kwargs['tb_debug']:
            debug_loss_dict = {
                'joint_loss': joint_loss,
            }
            return joint_loss, debug_loss_dict
        else:
            return joint_loss

# ###################################
# #
# # not fully tested temporal energy terms
# #
# ###################################
# class MultiViewTempSMPLifyLoss(MultiViewSMPLifyLoss):
#     def __init__(self, pose_embedding_t_1, pose_embedding_t_2 = None, 
#                 temporal_smooth_weight=100.0, dtype=torch.float32, **kwargs):

#         super(MultiViewTempSMPLifyLoss, self).__init__(**kwargs)

#         self.register_buffer(
#             'pose_embedding_t_1',torch.tensor(pose_embedding_t_1, dtype=dtype))

#         if pose_embedding_t_2 is None:
#             self.register_buffer(
#                 'pose_embedding_t_2', None)
#         else:
#             self.register_buffer(
#                 'pose_embedding_t_2',torch.tensor(pose_embedding_t_1, dtype=dtype))
        
#         self.register_buffer(
#             'temporal_smooth_weight',
#             torch.tensor(temporal_smooth_weight, dtype=dtype))

#     def forward(self, body_model_output, cameras, gt_joints, joints_conf,
#                 body_model_faces, joint_weights,
#                 use_vposer=False, pose_embedding=None,
#                 **kwargs):

#         frame_wise_loss = super(MultiViewTempSMPLifyLoss, self).forward(body_model_output, 
#                                                                         cameras=cameras,
#                                                                         gt_joints=gt_joints,
#                                                                         body_model_faces=body_model_faces,
#                                                                         joints_conf=joints_conf,
#                                                                         joint_weights=joint_weights,
#                                                                         pose_embedding=pose_embedding,
#                                                                         use_vposer=use_vposer,
#                                                                         **kwargs)
        
#         if self.pose_embedding_t_2 is not None:
#             ##
#             ## const speed smoothness term, not implemented yet
#             ##

#             # temporal_loss = (torch.sum( ((body_model_output.joints[:,:67,:] + 
#             #                                 self.pose_embedding_t_2[:,:67,:]) /2 
#             #                                 - self.pose_embedding_t_1[:,:67,:])**2 ) * 
#             #                         self.temporal_smooth_weight ** 2)

#             raise ValueError('Const speed smoothness term not implemented yet')
#         elif self.pose_embedding_t_2 is None and self.pose_embedding_t_1 is not None:

#             # temporal_loss = (torch.sum( (body_model_output.joints[:,:67,:] - 
#             #                             self.pose_embedding_t_1[:,:67,:])**2 ) * 
#             #                         self.temporal_smooth_weight ** 2)

#             temporal_loss = (torch.sum( (pose_embedding - 
#                                         self.pose_embedding_t_1)**2 ) * 
#                                     self.temporal_smooth_weight ** 2)
#         else:
#             raise ValueError('No prev. results specified')
#         # print(temporal_loss)
        
#         total_loss = frame_wise_loss + temporal_loss
#         return total_loss
