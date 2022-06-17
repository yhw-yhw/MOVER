import torch
import torch.nn as nn
from mover.utils.camera import (
    get_pitch_roll_euler_rotation_matrix,
)

### ground plane nformation
def set_ground_plane(self, ground_plane):
    self.ground_plane.data.copy_(ground_plane)

# def get_ground_plane(self):
#     verts = self.get_verts_object_parallel_ground()
#     return torch.max(verts[:, :, 1])

def get_ground_plane_mesh(self):
    gp_y = self.ground_plane
    vertices = torch.Tensor([[[-3, 0, 0], [3, 0, 0], [3, 0, 7], [-3, 0, 7]]]).type_as(gp_y)
    vertices[:, :, 1] = gp_y
    faces = torch.Tensor([[[0, 1, 2], [0, 2, 3]]]).type_as(self.faces)
    green = [0.19607843, 0.80392157, 0.19607843]
    textures = torch.Tensor([green, green]).reshape(1, 2, 1, 1, 1, 3).type_as(self.textures)
    return vertices, faces, textures

def get_ground_plane_np(self):
    vertices, faces, textures = self.get_ground_plane_mesh()
    return vertices.detach().cpu().numpy(), faces.detach().cpu().numpy(), textures.detach().cpu().numpy()

## camera -> image
def apply_projection_to_image(self, verts):
    proj_v = torch.matmul(self.K_intrin, verts.transpose(2,1)).transpose(2,1)
    xy = proj_v[:, :, :2] / proj_v[:, :, 2:]
    z = proj_v[:, :, 2:]
    return xy, z

def apply_projection_to_local_image(self, verts, obj_cam=None, resize=None):
    assert obj_cam is not None
    proj_v = torch.matmul(self.K_rois[obj_cam].float(), verts.float().transpose(2,1)).transpose(2,1)
    xy = proj_v[:, :, :2] / proj_v[:, :, 2:]
    z = proj_v[:, :, 2:]
    
    return xy, z

def get_relative_extrin_new(self):
    return get_pitch_roll_euler_rotation_matrix(self.rotate_cam_pitch, self.rotate_cam_roll) # need in forward 

# camera 
def get_cam_extrin(self):
    extra_extrin_mat = self.get_relative_extrin_new()
    cam_extrin = torch.transpose(torch.matmul(extra_extrin_mat, torch.transpose(self.K_extrin, 2, 1)), 2, 1)
    return cam_extrin