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
from ..smplifyx.utils_mics import misc_utils

from ..constants import (
    USE_PROX_VPOSER
)


class FittingMonitor(object):
    def __init__(self, summary_steps=1, visualize=False,
                 maxiters=30, ftol=2e-09, gtol=1e-05,
                 body_color=(1.0, 1.0, 0.9, 1.0),
                 model_type='smpl',
                 **kwargs):
        super(FittingMonitor, self).__init__()

        self.maxiters = maxiters
        self.ftol = ftol
        self.gtol = gtol

        self.visualize = visualize
        self.summary_steps = summary_steps
        self.body_color = body_color
        self.model_type = model_type

    def __enter__(self):
        self.steps = 0
        if self.visualize:
            self.mv = MeshViewer(body_color=self.body_color)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self.visualize:
            self.mv.close_viewer()

    def set_colors(self, vertex_color):
        batch_size = self.colors.shape[0]

        self.colors = np.tile(
            np.array(vertex_color).reshape(1, 3),
            [batch_size, 1])
    
    def close_viewer(self, ):
        if self.visualize:
            self.mv.close_viewer()
    
    def run_fitting(self, optimizer, closure, params, body_model,
                    use_vposer=True, pose_embedding=None, vposer=None,
                    **kwargs):
        ''' Helper function for running an optimization process
            Parameters
            ----------
                optimizer: torch.optim.Optimizer
                    The PyTorch optimizer object
                closure: function
                    The function used to calculate the gradients
                params: list
                    List containing the parameters that will be optimized
                body_model: nn.Module
                    The body model PyTorch module
                use_vposer: bool
                    Flag on whether to use VPoser (default=True).
                pose_embedding: torch.tensor, BxN
                    The tensor that contains the latent pose variable.
                vposer: nn.Module
                    The VPoser module
            Returns
            -------
                loss: float
                The final loss value
        '''
        append_wrists = self.model_type == 'smpl' and use_vposer
        prev_loss = None
        for n in range(self.maxiters):
            # import pdb;pdb.set_trace()
            loss = optimizer.step(closure)
            # import pdb;pdb.set_trace()
            # logger.info(f"after optimizer roll value: {kwargs['scene_model'].rotate_cam_roll}, ")
            # prev_loss = loss.item()
            if torch.isnan(loss).sum() > 0:
                print('NaN loss value, stopping!')
                break

            if torch.isinf(loss).sum() > 0:
                print('Infinite loss value, stopping!')
                break

            if n > 0 and prev_loss is not None and self.ftol > 0:
                loss_rel_change = misc_utils.rel_change(prev_loss, loss.item())

                if loss_rel_change <= self.ftol:
                    break

            if all([torch.abs(var.grad.view(-1).max()).item() < self.gtol
                    for var in params if var.grad is not None]):
                break

            if self.visualize and n % self.summary_steps == 0:
                if USE_PROX_VPOSER:
                    body_pose = vposer.forward(pose_embedding).view(1, -1) if use_vposer else None
                else:
                    body_pose = vposer.decode(
                        pose_embedding, output_type='aa').view(
                            1, -1) if use_vposer else None

                if append_wrists:
                    wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                             dtype=body_pose.dtype,
                                             device=body_pose.device)
                    body_pose = torch.cat([body_pose, wrist_pose], dim=1)
                # load parameters
                betas = kwargs['betas']
                global_orient = kwargs['global_orient']
                transl = kwargs['transl']
                left_hand_pose = kwargs['left_hand_pose']
                right_hand_pose = kwargs['right_hand_pose']
                jaw_pose = kwargs['jaw_pose']
                leye_pose = kwargs['leye_pose']
                reye_pose = kwargs['reye_pose']
                expression = kwargs['expression']

                # import pdb;pdb.set_trace()
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
                vertices = model_output.vertices.detach().cpu().numpy()

                self.mv.update_mesh(vertices.squeeze(),
                                    body_model.faces)

            prev_loss = loss.item()

        return prev_loss

    # define input
    def create_fitting_closure(self,
                               optimizer, body_model, cameras=None,
                               gt_joints=None, loss=None,
                               joints_conf=None,
                               joint_weights=None,
                               return_verts=True, return_full_pose=False,
                               use_vposer=False, vposer=None,
                               pose_embedding=None,
                               create_graph=False,
                               **kwargs):
        batch_size = gt_joints.shape[0]
        faces_tensor = body_model.faces_tensor.view(-1)
        append_wrists = self.model_type == 'smpl' and use_vposer

        # import pdb;pdb.set_trace()
        tb_debug = kwargs['tb_debug']
        tb_logger = kwargs['tb_logger']

        def fitting_func(backward=True):
            if backward:
                optimizer.zero_grad()
            # import pdb;pdb.set_trace()
            if USE_PROX_VPOSER:
                body_pose = vposer.forward(pose_embedding).view(
                    batch_size, -1) if use_vposer else None
            else:
                body_pose = vposer.decode(
                    pose_embedding, output_type='aa').view(
                        batch_size, -1) if use_vposer else None

            if append_wrists:
                wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                         dtype=body_pose.dtype,
                                         device=body_pose.device)
                body_pose = torch.cat([body_pose, wrist_pose], dim=1)
            
            betas = kwargs['betas']
            global_orient = kwargs['global_orient']
            transl = kwargs['transl']
            left_hand_pose = kwargs['left_hand_pose']
            right_hand_pose = kwargs['right_hand_pose']
            jaw_pose = kwargs['jaw_pose']
            leye_pose = kwargs['leye_pose']
            reye_pose = kwargs['reye_pose']
            expression = kwargs['expression']
            
            body_model_output = body_model(return_verts=return_verts,
                                           body_pose=body_pose,
                                           betas = betas,
                                           global_orient = global_orient,
                                           transl = transl,
                                           left_hand_pose = left_hand_pose,
                                           right_hand_pose = right_hand_pose,
                                           jaw_pose = jaw_pose,
                                           leye_pose = leye_pose,
                                           reye_pose = reye_pose,
                                           expression = expression,
                                           return_full_pose=return_full_pose)

            if self.steps % self.summary_steps == 0 and tb_debug:
                # import pdb;pdb.set_trace()
                # calculate loss with using body_model_output !!!
                # logger.info(f"global_orient: { kwargs['global_orient'].data}")
                # TODO: DEBUG
                # import pdb;pdb.set_trace()
                
                total_loss, debug_loss_dict = loss(body_model_output, cameras=cameras,
                                gt_joints=gt_joints,
                                body_model_faces=faces_tensor,
                                joints_conf=joints_conf,
                                joint_weights=joint_weights,
                                pose_embedding=pose_embedding,
                                use_vposer=use_vposer,
                              **kwargs)

                assert tb_logger is not None
                # write into tb_logger
                opt_stage = kwargs['opt_idx']
                from .tf_utils import save_scalars
                

                # import pdb;pdb.set_trace()
                if opt_stage >=4:
                    save_scalars(tb_logger, f'SMPLifyX stage{opt_stage}',debug_loss_dict, self.steps)
                save_scalars(tb_logger, f'SMPLifyX whole fitting',debug_loss_dict, self.steps)


            else:
                # calculate loss with using body_model_output !!!
                total_loss = loss(body_model_output, cameras=cameras,
                                gt_joints=gt_joints,
                                body_model_faces=faces_tensor,
                                joints_conf=joints_conf,
                                joint_weights=joint_weights,
                                pose_embedding=pose_embedding,
                                use_vposer=use_vposer,
                                **kwargs)
            
            # import pdb;pdb.set_trace()
            if backward:
                total_loss.backward(create_graph=create_graph)
                # logger.info(f"roll grad: {kwargs['scene_model'].rotate_cam_roll.grad}, ")
                # logger.info(f"roll value: {kwargs['scene_model'].rotate_cam_roll}, ")
                # logger.info(f"rotations_object value: {kwargs['scene_model'].rotations_object}, | grad: {kwargs['scene_model'].rotations_object.grad}")
                # logger.info(f"translations_object value: {kwargs['scene_model'].translations_object}, | grad: {kwargs['scene_model'].translations_object.grad}")

            self.steps += 1
            if self.visualize and self.steps % self.summary_steps == 0:
                model_output = body_model(return_verts=True,
                                          body_pose=body_pose,
                                          betas = betas,
                                            global_orient = global_orient,
                                            transl = transl,
                                            left_hand_pose = left_hand_pose,
                                            right_hand_pose = right_hand_pose,
                                            jaw_pose = jaw_pose,
                                            leye_pose = leye_pose,
                                            reye_pose = reye_pose,
                                            expression = expression,)
                vertices = model_output.vertices.detach().cpu().numpy()

                self.mv.update_mesh(vertices.squeeze(),
                                    body_model.faces)

            return total_loss

        return fitting_func