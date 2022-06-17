# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
from mover.utils.bbox import bbox_xy_to_wh, make_bbox_square
from mover.utils.pytorch3d_rotation_conversions import euler_angles_to_matrix
import math

def local_to_global_cam(bboxes, cams, L):
    """
    Converts a weak-perspective camera w.r.t. a bounding box to a weak-perspective
    camera w.r.t. to the entire image.

    Args:
        bboxes (N x 4): Bounding boxes in xyxy format.
        cams (N x 3): Weak perspective camera.
        L (int): Max of height and width of image.
    """
    square_bboxes = make_bbox_square(bbox_xy_to_wh(bboxes))
    global_cams = []
    for cam, bbox in zip(cams, square_bboxes):
        x, y, b, _ = bbox
        X = np.stack((x, y))
        s_crop = b * cam[0] / 2
        t_crop = cam[1:] + 1 / cam[0]

        # Global image space [0, 1]
        s_og = s_crop / L
        t_og = t_crop + X / s_crop

        # Normalized global space [-1, 1]
        s = s_og * 2
        t = t_og - 0.5 / s_og
        global_cams.append(np.concatenate((np.array([s]), t)))
    return np.stack(global_cams)

def compute_K_roi(upper_left, b, img_size, focal_length=1.0):
    """
    Computes the intrinsics matrix for a cropped ROI box.

    Args:
        upper_left (tuple): Top left corner (x, y).
        b (float): Square box size.
        img_size (int): Size of image in pixels.

    Returns:
        Intrinsic matrix (1 x 3 x 3).
    """
    x1, y1 = upper_left
    f = focal_length * img_size / b
    px = (img_size / 2 - x1) / b
    py = (img_size / 2 - y1) / b
    K = torch.cuda.FloatTensor([[[f, 0, px], [0, f, py], [0, 0, 1]]])
    return K

def computer_roi_from_cam_inc(upper_left, b,  img_size, cam_inc):
    x1, y1 = upper_left
    scale =  img_size / b # from b to img_size
    new_cam = np.copy(cam_inc)
    # crop
    new_cam[0][2] = new_cam[0][2] - x1
    new_cam[1][2] = new_cam[1][2] - y1
    # scale
    new_cam[0][0] = new_cam[0][0] * scale
    new_cam[1][1] = new_cam[1][1] * scale
    new_cam[0][2] = new_cam[0][2] * scale
    new_cam[1][2] = new_cam[1][2] * scale
    
    return torch.from_numpy(new_cam).cuda().float().unsqueeze(0)

def compute_transformation_ortho(
    meshes, cams, rotations=None, intrinsic_scales=None, focal_length=1.0
):
    """
    Computes the 3D transformation from a scaled orthographic camera model.

    Args:
        meshes (V x 3 or B x V x 3): Vertices.
        cams (B x 3): Scaled orthographic camera [s, tx, ty].
        rotations (B x 3 x 3): Rotation matrices.
        intrinsic_scales (B).
        focal_length (float): Should be 2x object focal length due to scaling.

    Returns:
        vertices (B x V x 3).
    """
    B = len(cams)
    device = cams.device
    if meshes.ndimension() == 2:
        meshes = meshes.repeat(B, 1, 1)
    if rotations is None:
        rotations = torch.eye(3).repeat(B, 1, 1).to(device)
    if intrinsic_scales is None:
        intrinsic_scales = torch.ones(B).to(device)
    tx = cams[:, 1]
    ty = cams[:, 2]
    tz = 2 * focal_length / cams[:, 0]
    verts_rot = torch.matmul(meshes.detach().clone(), rotations)  # B x V x 3
    trans = torch.stack((tx, ty, tz), dim=1).unsqueeze(1)  # B x 1 x 3
    verts_trans = verts_rot + trans
    verts_final = intrinsic_scales.view(-1, 1, 1) * verts_trans
    return verts_final


# optimize three dimension objects.
# def compute_transformation_persp(
#     meshes, translations, basis=None, rotations=None, intrinsic_scales=None
# ):
#     """
#     Computes the 3D transformation.
#     Formulation: 
#         R(scale * vert) + T

#     Args:
#         meshes (V x 3 or B x V x 3): Vertices.
#         translations (B x 1 x 3).
#         rotations (B x 3 x 3).
#         intrinsic_scales (B x 3), for x, y, z scale

#     Returns:
#         vertices (B x V x 3).
#     """
#     B = translations.shape[0]
#     device = meshes.device
#     if meshes.ndimension() == 2:
#         meshes = meshes.repeat(B, 1, 1)
#     if rotations is None:
#         rotations = torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
#         rotations = rotations.to(device)
#     if intrinsic_scales is None:
#         intrinsic_scales = torch.ones(B).to(device)

#     mean_mesh = meshes.mean(dim=1)
#     local_meshes = meshes - mean_mesh
#     local_meshes = intrinsic_scales.view(-1, 1, 3) * local_meshes
#     # TODO: ! warning: need to modified basis in Total3D
#     local_meshes = torch.matmul(local_meshes, torch.transpose(basis, 2,1))
#     verts_rot = torch.matmul(local_meshes, rotations) # add rotation
#     verts_trans = verts_rot + translations + mean_mesh
    
#     return verts_trans


def compute_transformation_persp(
    meshes, translations, basis=None, rotations=None, 
    intrinsic_scales=None, ground_plane=None, ALL_OBJ_ON_THE_GROUND=True
):
    """
    Computes the 3D transformation.
    Formulation: 
        R(scale * vert) + T

    Args:
        meshes (V x 3 or B x V x 3): Vertices.
        translations (B x 1 x 3).
        rotations (B x 3 x 3).
        intrinsic_scales (B x 3), for x, y, z scale

    Returns:
        vertices (B x V x 3).
    """
    B = translations.shape[0]
    device = meshes.device
    if meshes.ndimension() == 2:
        meshes = meshes.repeat(B, 1, 1)
    if rotations is None:
        rotations = torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        rotations = rotations.to(device)
    if intrinsic_scales is None:
        intrinsic_scales = torch.ones(B).to(device)

    mean_mesh = meshes.mean(dim=1)
    local_meshes = meshes - mean_mesh
    local_meshes = intrinsic_scales.view(-1, 1, 3) * local_meshes
    # TODO: ! warning: need to modified basis in Total3D
    local_meshes = torch.matmul(local_meshes, torch.transpose(basis, 2,1))
    verts_rot = torch.matmul(local_meshes, rotations) # add rotation

    # import pdb;pdb.set_trace()
    if ALL_OBJ_ON_THE_GROUND:
        tmp_y  = ground_plane - (verts_rot + mean_mesh)[:, :, 1].max(-1)[0]
    else:
        tmp_y  = translations[:, 1]
    tmp_translation = torch.stack([translations[:, 0], tmp_y, translations[:, 2]], -1)
    verts_trans = verts_rot + tmp_translation + mean_mesh
    
    return verts_trans


def get_Y_euler_rotation_matrix(y_euler):
    
    B = y_euler.shape[0]
    rotation_init_x = torch.zeros(B, 1).type_as(y_euler)
    rotation_init_z = torch.zeros(B, 1).type_as(y_euler)
    rotations_init = euler_angles_to_matrix(
        torch.cat((rotation_init_x, y_euler, rotation_init_z), 1), "ZYX")
    return rotations_init

def get_pitch_roll_euler_rotation_matrix(z_euler, x_euler):
    B = z_euler.shape[0]
    rot_init_y = torch.zeros(1, 1).type_as(z_euler)
    rotations_mat = euler_angles_to_matrix(torch.cat((z_euler, rot_init_y, x_euler), 1), "ZYX")
    return rotations_mat

def get_rotation_matrix_mk(pitch, roll):
    return batch_euler2matrix(torch.tensor([[pitch, 0, roll]]))

def batch_euler2matrix(r):
    return quaternion_to_rotation_matrix(euler_to_quaternion(r))


def euler_to_quaternion(r):
    x = r[..., 0]
    y = r[..., 1]
    z = r[..., 2]

    z = z/2.0
    y = y/2.0
    x = x/2.0
    cz = torch.cos(z)
    sz = torch.sin(z)
    cy = torch.cos(y)
    sy = torch.sin(y)
    cx = torch.cos(x)
    sx = torch.sin(x)
    quaternion = torch.zeros_like(r.repeat(1,2))[..., :4].to(r.device)
    quaternion[..., 0] += cx*cy*cz - sx*sy*sz
    quaternion[..., 1] += cx*sy*sz + cy*cz*sx
    quaternion[..., 2] += cx*cz*sy - sx*cy*sz
    quaternion[..., 3] += cx*cy*sz + sx*cz*sy
    return quaternion


def quaternion_to_rotation_matrix(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


if __name__ == "__main__":
    import joblib
    tmp = joblib.load('/ps/scratch/hyi/HCI_dataset/20210209_experiments/PROX_sample/scenes/scalenet/N3OpenArea_00157_02_flip.jpg.pkl')
    pitch = tmp['pitch']
    roll = tmp['roll']

    rot_mat = get_rotation_matrix_mk(pitch, roll)
    print(rot_mat) 