import torch
import torch.nn as nn
from mover.utils.bbox import check_overlap, compute_iou
from mover.utils.camera import (
    compute_transformation_ortho,
    compute_transformation_persp,
    get_Y_euler_rotation_matrix,
    get_pitch_roll_euler_rotation_matrix,
    get_rotation_matrix_mk, 
)
# TODO:
from mover.utils.geometry import (
    center_vertices,
    combine_verts,
    compute_dist_z,
    matrix_to_rot6d,
    rot6d_to_matrix,
    compute_intersect,
    get_bev_bbox,
)
import trimesh
import numpy as np
import torch.nn.functional as F

def even_sample_mesh(self, vert, face, idx, body=False): 
    mesh = trimesh.Trimesh(vert.detach().squeeze().cpu().numpy(), face.detach().squeeze().cpu().numpy())
    if body:
        count = 3000
    else:
        if idx == self.size_cls.shape[0]:
            count = 2000
        elif self.size_cls[idx] == 6: #sofa
            count = 2000 #1000
        else:
            count = 1000 #500
    points, _ = trimesh.sample.sample_surface_even(mesh, count=count)
    sampled_points = torch.from_numpy(points).type_as(vert).unsqueeze(0)
    return sampled_points


def get_even_sample_verts(self, verts_list, faces_list, body=False):
    sample_list = []
    idxs = np.arange(len(verts_list))
    for vert, face, idx in zip(verts_list, faces_list, idxs):
        sample_list.append(self.even_sample_mesh(vert, face, idx, body))
    return sample_list

def get_split_obj_verts(self, verts, idx):
    if idx == 0:
        start_p = 0
        end_p = self.idx_each_object[idx]
    else:
        start_p = self.idx_each_object[idx-1] 
        end_p = self.idx_each_object[idx]
    return verts[start_p: end_p]

def get_resampled_verts_object_og(self):
    sample_list = []
    for one in range(self.idx_each_object.shape[0]):
        obj_verts = self.get_split_obj_verts(self.verts_object_og, one)
        obj_faces = self.get_single_obj_faces(one)
        sample_points = self.even_sample_mesh(obj_verts, obj_faces, one)
        sample_list.append(sample_points)
    return sample_list

def get_resampled_verts_object_parallel_ground(self, return_all=False): # add camera rotation
    all_result = []
    for one in range(self.idx_each_object.shape[0]):
        obj = self.get_single_obj_verts(self.resampled_verts_object_og, one, list_format=True)
        all_result.append(obj)
    return all_result

def get_scale_object(self):
    if self.use_sigmoid_for_scale:
        if self.constraint_scale_for_chair:
            tmp_scale = []
            for i in range(self.int_scales_object.shape[0]):
                if self.size_cls[i] == 5:
                    scale_ = (F.sigmoid(self.int_scales_object[i]) - 0.5)*self.chair_scale + self.init_int_scales_object[i]
                else:
                    scale_ = (F.sigmoid(self.int_scales_object[i]) - 0.5)*0.8 + self.init_int_scales_object[i]
                tmp_scale.append(scale_)
            scale=torch.stack(tmp_scale)
        else:
            scale = (F.sigmoid(self.int_scales_object) - 0.5)*0.8 + self.init_int_scales_object
    else:
        scale = self.int_scales_object
    return scale

def get_single_obj_verts(self,verts_ori_og, idx, list_format=False):
    if list_format:
        verts = verts_ori_og[idx]
    else:
        if idx == 0:
            start_p = 0
            end_p = self.idx_each_object[idx]
        else:
            start_p = self.idx_each_object[idx-1] 
            end_p = self.idx_each_object[idx]
        verts = verts_ori_og[start_p:end_p]

    scale = self.get_scale_object()
    if self.USE_ONE_DOF_SCALE:
        scale_object = scale[idx:idx+1][:, 0].expand(1,3)
    else:
        scale_object = scale[idx:idx+1]

    obj = compute_transformation_persp(
        meshes=verts,
        basis=self.basis_object[idx:idx+1],
        translations=self.translations_object[idx:idx+1],
        rotations = get_Y_euler_rotation_matrix(self.rotations_object[idx:idx+1]),
        intrinsic_scales=scale_object,
        ground_plane=self.ground_plane,
        ALL_OBJ_ON_THE_GROUND=self.ALL_OBJ_ON_THE_GROUND,
    )
    return obj

def get_single_obj_faces(self, idx, texture=False):
    if idx == 0:
        start_f = 0
        end_f = self.idx_each_object_face[idx]
        start_p = 0
        end_p = self.idx_each_object[idx]
    else:
        start_f = self.idx_each_object_face[idx-1] 
        end_f = self.idx_each_object_face[idx]
        start_p = self.idx_each_object[idx-1] 
        end_p = self.idx_each_object[idx]
    face_type = self.faces.type()
    obj_faces = (self.faces[:, start_f:end_f] - start_p).type(face_type)
    if not texture:
        return obj_faces
    else:
        obj_textures = self.textures_object[:, start_f:end_f]
        return obj_faces, obj_textures

def get_verts_object_parallel_ground(self, return_all=False):
    # in World CS
    all_result = []
    for one in range(self.idx_each_object.shape[0]):
        obj = self.get_single_obj_verts(self.verts_object_og, one)
        all_result.append(obj)
    if return_all:
        return torch.cat(all_result, dim=1), all_result
    else:    
        return torch.cat(all_result, dim=1)

def get_contact_verts_obj(self, verts_list, faces_list, return_all=False):
    all_result = []
    all_scene_vn_list = []
    for one in range(self.contact_idx_each_obj.shape[0]):
        if one == 0:
            contact_idx = self.contact_idxs[:self.contact_idx_each_obj[one]]
        else:
            contact_idx = self.contact_idxs[self.contact_idx_each_obj[one-1]:self.contact_idx_each_obj[one]]

        j = one
        scene_v = verts_list[j]
        scene_f = faces_list[j] 

        ori_scene_f = scene_f[0].detach().cpu().numpy()
        flip_scene_f = np.stack([ori_scene_f[:, 2], ori_scene_f[:, 1], ori_scene_f[:, 0]], -1)
        meshes = trimesh.Trimesh(scene_v[0].detach().cpu().numpy(), flip_scene_f, \
                                    process=False, maintain_order=True)
        scene_vn = torch.from_numpy(np.array(meshes.vertex_normals)).type_as(scene_v).unsqueeze(0)
        all_scene_vn_list.append(scene_vn[:, contact_idx.long(), :])

        contact_v = verts_list[one][:, contact_idx.long(), :]
        all_result.append(contact_v)
    
    return all_result, all_scene_vn_list

def get_faces_textures_list(self):
    all_result = []
    all_result_texture = []
    for one in range(self.idx_each_object.shape[0]):
        faces, texture = self.get_single_obj_faces(one, texture=True )
        all_result.append(faces.cuda())
        all_result_texture.append(texture.cuda())
    return all_result, all_result_texture


def get_verts_object(self, return_all=False):
    ## multiple objects
    # transform world coordinates to camera coordinates
    if return_all:
        verts, verts_list = self.get_verts_object_parallel_ground(return_all)
        rot_verts = torch.transpose(torch.matmul(self.get_cam_extrin(), \
                torch.transpose(verts, 2, 1)), 2, 1)
        rot_obj_list = []
        for obj in verts_list:
            rot_obj = torch.transpose(torch.matmul(self.get_cam_extrin(), \
            torch.transpose(obj, 2, 1)), 2, 1)
            rot_obj_list.append(rot_obj)
        return rot_verts, rot_obj_list
    else:
        verts = self.get_verts_object_parallel_ground(return_all)
        rot_verts = torch.transpose(torch.matmul(self.get_cam_extrin(), \
                torch.transpose(verts, 2, 1)), 2, 1)
        return rot_verts