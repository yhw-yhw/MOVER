from loguru import logger
import torch
from mover.utils.camera import get_Y_euler_rotation_matrix
from mover.utils.pytorch3d_rotation_conversions import euler_angles_to_matrix, matrix_to_euler_angles
from ._util_hsi import get_y_verts
import os
import trimesh
import numpy as np

def get_theta_between_two_normals(self, body_normal, obj_normal):
    # get the correpsonding normal given body_normal;
    # body_normal->obj_normal: b->a

    obj_normal, body_normal = obj_normal/ obj_normal.norm(), body_normal / body_normal.norm()
    dot = torch.dot(obj_normal, body_normal)
    angle = torch.acos(dot)
    rot_mat = get_Y_euler_rotation_matrix(angle[None, None])
    
    if torch.abs(obj_normal - torch.matmul(rot_mat, body_normal)).sum() < 1e-3:
        return angle
    elif torch.abs(obj_normal - torch.matmul(rot_mat.transpose(1, 2), body_normal)).sum() < 1e-3:
        i_rot_mat = get_Y_euler_rotation_matrix(-angle[None, None])
        assert ( rot_mat.transpose(1, 2) - i_rot_mat).sum() < 1e-3
        return -angle
        
def renew_rot_angle(self, new_angle, idx):
    
    rot_mat = get_Y_euler_rotation_matrix(new_angle[None, None])
    ori_mat = get_Y_euler_rotation_matrix(self.rotations_object[idx][None]).transpose(1, 2)

    fuse_rot_mat = torch.matmul(rot_mat, ori_mat).transpose(1, 2)
    fuse_angle = matrix_to_euler_angles(fuse_rot_mat, "ZYX") 
    if torch.abs(3.1416 -fuse_angle[0, 0]) < 1e-2:
        self.rotations_object.data[idx].copy_(3.1416 - fuse_angle.squeeze()[1])
    else:
        self.rotations_object.data[idx].copy_(fuse_angle.squeeze()[1])

def renew_transl(self, mean_body, idx):
    
    # only opt x,z direction.
    self.translations_object.data[idx][0].copy_(mean_body[0].detach())
    self.translations_object.data[idx][2].copy_(mean_body[2].detach())

  
def renew_scale_based_transl(self, scale, idx):
    #! this influence the optimize graph.
    modify_scale = scale * self.get_scale_object().detach()[idx]
    # import pdb;pdb.set_trace()
    # try to be plausible
    if self.size_cls[idx] == 5:
        valid =  (modify_scale > (1-self.chair_scale*0.5)) & (modify_scale < (1+self.chair_scale*0.5))
        if valid.sum() < 3:
            print('error in modify_x_sigmoid_np')
            print(f'before modify: {modify_scale}')
        modify_scale = torch.clamp(modify_scale, 1-self.chair_scale*0.5+0.001, 1+self.chair_scale*0.5-0.001)
        print(f'modify: {modify_scale}')
    
        modify_x_sigmoid = (modify_scale - self.init_int_scales_object[idx] )/ self.chair_scale + 0.5    
    else:
        valid =  (modify_scale > (1-0.8*0.5)) & (modify_scale < (1+0.8*0.5))
        if valid.sum() < 3:
            print('error in modify_x_sigmoid_np')
            print(f'before modify: {modify_scale}')
        modify_scale = torch.clamp(modify_scale, 1-0.8*0.5+0.001, 1+0.8*0.5-0.001)
        print(f'modify: {modify_scale}')
    
        modify_x_sigmoid = (modify_scale - self.init_int_scales_object[idx] )/ 0.8 + 0.5
    modify_x_sigmoid_np = modify_x_sigmoid.detach().cpu().numpy()
    modify_x = np.log(modify_x_sigmoid_np)
    self.int_scales_object.data[idx].copy_(torch.from_numpy(modify_x).cuda())
    
def renew_scale(self, width_range, idx):
    #! this influence the optimize graph.
    all_obj_scale = self.get_scale_object().detach() * self.ori_objs_size
    tmp_scale = all_obj_scale[idx][0] / width_range
    if tmp_scale > 1.0:
        logger.info(f"no need to adjust scale for obj {idx}")
    else: # TODO:
        modify_scale = width_range / self.ori_objs_size[idx][0]
        modify_x_sigmoid = (modify_scale - self.init_int_scales_object[idx][0] )/ 0.8 + 0.5
        modify_x = -torch.log(((1/ modify_x_sigmoid) - 1))
        
        self.int_scales_object.data[idx][0].copy_(modify_x.detach())

def reinit_orien_objs_by_contacted_bodies(self, use_total3d_reinit=False, opt_scale_transl=False):
    all_contact_body_vertices = self.accumulate_contact_body_vertices
    all_contact_body_verts_normals = self.accumulate_contact_body_verts_normals
    all_contact_body2obj_idx = self.accumulate_contact_body_body2obj_idx
    obj_num = self.rotations_object.shape[0]

    verts_parallel_ground, verts_parallel_ground_list = self.get_verts_object_parallel_ground(return_all=True)
    contact_verts_ground_list, contact_vn_ground_list  = self.get_contact_verts_obj(verts_parallel_ground_list, self.faces_list, return_all=True)
                    
    for idx in range(obj_num):
        body2obj_idx = all_contact_body2obj_idx == idx
        if body2obj_idx.sum() == 0 or self.size_cls[idx] not in [5, 6]: 
             
            if use_total3d_reinit: # TODO: need to check.
                logger.info("reinit rotation with Total3D as Init.")
                self.rotations_object.data[idx].copy_(torch.zeros(1).cuda().detach())
            continue
        else:
            # only for sofa and chair; 
            # use body contact to reinit the 3D scene;
            contact_body_v = all_contact_body_vertices[body2obj_idx][None]
            contact_body_vn = all_contact_body_verts_normals[body2obj_idx][None]

            # get bird-eye view orientation.
            body_zaxis_valid = get_y_verts(contact_body_vn, along=False)[0]
            body_vn_zaxis = contact_body_vn[:, body_zaxis_valid, :]
            body_mean_vn_z = body_vn_zaxis.mean(1).squeeze()

            obj_vn = contact_vn_ground_list[idx]
            scene_zaxis_valid = get_y_verts(obj_vn, along=False)[0]
            obj_vn_zaxis = obj_vn[:, scene_zaxis_valid, :]
            obj_mean_vn_z = obj_vn_zaxis.mean(1).squeeze()
            
            if True:
                c_b_v = contact_body_v.detach().cpu().numpy()
                c_b_vn = contact_body_vn.detach().cpu().numpy()
                out_mesh = trimesh.Trimesh(c_b_v[0], vertex_normals=c_b_vn[0], process=False)
                template_save_fn = os.path.join('/is/cluster/hyi/tmp', 'contact_sample.ply') 
                out_mesh.export(template_save_fn,vertex_normal=True) # export_ply
                o_v = contact_verts_ground_list[idx].detach().cpu().numpy()

            body_mean_vn_z[1] = 0
            body_mean_vn_z = body_mean_vn_z * -1 # inverse of the back normal 
            obj_mean_vn_z[1] = 0
            rot_theta = self.get_theta_between_two_normals(obj_mean_vn_z, body_mean_vn_z)

            # import pdb;pdb.set_trace()            
            logger.info(f'reinit obj {idx}')
            self.renew_rot_angle(rot_theta, idx)

            if opt_scale_transl:
                logger.info(f'reinit scale and tral for obj {idx}, size_cls:{self.size_cls[idx]}')
                if self.size_cls[idx] in [6, 5]: # [5, 6]: # TODO: before 08.11, we only use this for sofa;
                    
                    mean_body = contact_body_v.mean(1)[0].squeeze()
                    body_max_xyz = contact_body_v.max(1)[0].squeeze()
                    body_min_xyz = contact_body_v.min(1)[0].squeeze()
                    body_range_xyz = body_max_xyz- body_min_xyz
                    body_orient_norm = body_mean_vn_z / torch.norm(body_mean_vn_z)
                    body_orient_norm[1] = 0.0
                    cos_angle = torch.dot(body_orient_norm, torch.Tensor([0, 0, 1]).cuda())
                    angle = torch.acos(cos_angle)
                    cos_angle_abs = cos_angle.abs()
                    sin_angle_abs = torch.sin(angle)
                    xrange = sin_angle_abs * body_range_xyz[2] + cos_angle_abs * body_range_xyz[0]
                    yrange = cos_angle_abs * body_range_xyz[2] + sin_angle_abs * body_range_xyz[0]
                    width_range = yrange if xrange < yrange  else xrange
                    
                    # to make it more accurate: use 2D projection.
                    ori_c = torch.norm(torch.stack([self.translations_object.data[idx][0], self.translations_object.data[idx][2]]))
                    new_c = torch.norm(torch.stack([mean_body[0].detach(), mean_body[2].detach()]))
                    scale =  new_c / ori_c
                    self.renew_transl(mean_body, idx)
                    self.renew_scale_based_transl(scale, idx)
                    
                    
# * reinit the objs by the depth map and translation; only works once and without conflict to the object;
def reinit_transl_with_depth_map(self, opt_scale_transl=False):
    masks_object = self.masks_object ==1
    back_depth_map = self.depth_template_human[:, :, :, 1]
    front_depth_map = self.depth_template_human[:, :, :, 0]
    
    obj_num = masks_object.shape[0]
    all_contact_body2obj_idx = self.accumulate_contact_body_body2obj_idx

    for idx in range(obj_num):
        body2obj_idx = all_contact_body2obj_idx == idx
        
        # for those non-contact chair
        if body2obj_idx.sum() == 0 and self.size_cls[idx] in [5]: # 5:chair; 6:sofat; 7:table; not have contact;
            valid_region = masks_object[idx]
            front_valid = (front_depth_map > 0) & (front_depth_map <100) & valid_region
            back_valid = (back_depth_map  > 0) & (back_depth_map <100) & valid_region

            # import pdb;pdb.set_trace()
            print(f'front: {front_valid.sum()}, back: {back_valid.sum()}')
            both_valid = front_valid & back_valid
            if both_valid.sum() > 500: 
               
                # import pdb;pdb.set_trace()
                mean_z = (front_depth_map[both_valid] + back_depth_map[both_valid]) / 2
                mean_z = mean_z.mean()

                # mean_z = (front_depth_map[both_valid].max() + back_depth_map[both_valid].min()) / 2
                # import pdb;pdb.set_trace()
                if opt_scale_transl:
                    logger.info(f'by depth map: reinit scale and tral for obj {idx}, size_cls:{self.size_cls[idx]}')
                        
                    mean_body = self.translations_object.data[idx].clone().detach()
                    mean_body[2] = mean_z
                    
                    # to make it more accurate: use 2D projection.
                    ori_c = torch.norm(torch.stack([self.translations_object.data[idx][0], self.translations_object.data[idx][2]]))
                    new_c = torch.norm(torch.stack([mean_body[0].detach(), mean_body[2].detach()]))
                    scale =  new_c / ori_c
                    self.renew_transl(mean_body, idx)
                    self.renew_scale_based_transl(scale, idx)
                    logger.info(f'by depth map: reinit transl {mean_body}, scale {scale}')
                    
            else:
                 print(f'no update transl for transl for obj {idx}')