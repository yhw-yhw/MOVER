import sys
sys.path.append('/is/cluster/hyi/workspace/Multi-IOI/bvh-distance-queries')
# TODO: check this is useful or not.

from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os.path as osp
import json
import os
import time
from tqdm import tqdm
from psbody.mesh import Mesh
from psbody.mesh.geometry.tri_normals import TriNormals, TriNormalsScaled

from mover.utils.visualize import define_color_map
from mover.constants import (
    LOSS_NORMALIZE_PER_ELEMENT,
    ADD_HAND_CONTACT,
    USE_HAND_CONTACT_SPLIT,
)
from ._util_hsi import get_y_verts

def compute_hsi_contact_loss_persubject_coarse(self, verts_list, vn_list,
                contact_body_vertices, contact_body_verts_normals, contact_body2obj_idx=None,
                contact_angle=None, contact_robustifier=None, debug=True, save_dir=None, 
                detailed=False):
    # in world coordinates system: this is better for calculating 3D bbox of each objects.
    # input: 
    #   verts_list, vn_list: vertices of the objects and its corresponding normal.
    #   contact_body_vertices, contact_body_verts_normals, contact_body2obj_idx: contacted body information.
    
    # ! warning: first find the same normal points, then make the distance small.
    
    assert contact_angle is not None and \
                contact_robustifier is not None
    num_objs = len(verts_list)
    all_contact_loss = torch.tensor(0., device=verts_list[0].device)
    all_contact_loss_list = []
    
    # * add contact body2obj label
    if contact_body2obj_idx is not None:
        
        contact_body_vertices = contact_body_vertices[contact_body2obj_idx][None]
        contact_body_verts_normals = contact_body_verts_normals[contact_body2obj_idx][None]

    if contact_body_vertices.shape[1] == 0:
        return all_contact_loss, [all_contact_loss for i in range(num_objs)]

    verts_shape = [0]
    tmp_i = 0
    for obj_v, obj_vn in zip(verts_list, vn_list):
        # If only one person, return 0
        contact_loss_y = torch.tensor(0., device=verts_list[0].device)
        contact_loss_z = torch.tensor(0., device=verts_list[0].device)
        
        all_scene_v = obj_v 

        import mover.dist_chamfer as ext
        distChamfer = ext.chamferDist()
        
        # split all scene and body vertices into two parts: the vertices along with yaxis, 
        # those that prependicular to -yaxis.
        scene_yaxis_valid = get_y_verts(obj_vn, along=True)[0]
        body_yaxis_valid = get_y_verts(contact_body_verts_normals, along=True)[0]
        scene_zaxis_valid = get_y_verts(obj_vn, along=False)[0]
        body_zaxis_valid = get_y_verts(contact_body_verts_normals, along=False)[0]
            
        # contact 3D scene only match one point of objects.
        # ! gradient is indirect on the scene objects.
        contact_dist_y, _, idx1, _ = distChamfer(all_scene_v[:,scene_yaxis_valid, :].contiguous(), 
                    contact_body_vertices[:,body_yaxis_valid,:].contiguous())
        if self.CONTACT_MSE:
            all_valid_contact_dist_y = contact_dist_y
        else: 
            all_valid_contact_dist_y = contact_dist_y.sqrt()

        contact_dist_z, _, idx1, _ = distChamfer(all_scene_v[:, scene_zaxis_valid, :].contiguous(), 
                    contact_body_vertices[:, body_zaxis_valid, :].contiguous())
        
        if self.CONTACT_MSE:
            all_valid_contact_dist_z = contact_dist_z
        else: 
            all_valid_contact_dist_z = contact_dist_z.sqrt()

        # ! nan: if all_valid_contact_dist is zero, so all_valid_contact_dist.mean() = nan
        if not all_valid_contact_dist_y.sum() == 0:
            contact_loss_y = all_valid_contact_dist_y.sum() / contact_dist_y.shape[1]
        if not all_valid_contact_dist_z.sum() == 0:
            contact_loss_z = all_valid_contact_dist_z.sum() / contact_dist_z.shape[1]
        
        all_contact_loss = all_contact_loss + contact_loss_y + contact_loss_z
        contact_loss = contact_loss_y + contact_loss_z
        all_contact_loss_list.append(contact_loss)
    
    return all_contact_loss / (len(verts_list) + 1e-9), all_contact_loss_list
