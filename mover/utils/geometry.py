# Copyright (c) Facebook, Inc. and its affiliates.
import math

import torch
from torch.nn import functional as F
from scipy.spatial.transform import Rotation as R
# from pytorch3d.transforms import euler_angles_to_matrix
from mover.utils.pytorch3d_rotation_conversions import euler_angles_to_matrix
from mover.utils.camera import computer_roi_from_cam_inc
from mover.constants import (
    BBOX_EXPANSION_FACTOR,
    REND_SIZE,
)
import numpy as np

def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalisation per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    
    if d6.shape[-1] == 2:
        d6 = d6.transpose(1, 2).reshape(-1, 6)
        
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def matrix_to_rot6d(rotmat):
    """
    Convert rotation matrix to 6D rotation representation.

    Args:
        rotmat (B x 3 x 3): Batch of rotation matrices.

    Returns:
        6D Rotations (B x 3 x 2).
    """
    return rotmat[..., :2, :].clone().reshape(*rotmat.size()[:-2], 6)

def combine_verts(verts_list):
    all_verts_list = [v.reshape(1, -1, 3) for v in verts_list]
    verts_combined = torch.cat(all_verts_list, 1)
    return verts_combined

def center_vertices(vertices, faces, flip_y=True):
    """
    Centroid-align vertices.
    Args:
        vertices (V x 3): Vertices.
        faces (F x 3): Faces.
        flip_y (bool): If True, flips y verts to keep with image coordinates convention.

    Returns:
        vertices, faces
    """
    vertices = vertices - vertices.mean(dim=0, keepdim=True)
    if flip_y:
        vertices[:, 1] *= -1
        faces = faces[:, [2, 1, 0]]
    return vertices, faces

def scale_vertices(vertices, faces, size=np.array([1, 1, 1])):
    original_size = vertices.max(1) - vertices.min(1) 
    vertices = vertices / original_size * scale
    return vertices

def compute_dist_z(verts1, verts2):
    """
    Computes distance between sets of vertices only in Z-direction.

    Args:
        verts1 (V x 3).
        verts2 (V x 3).

    Returns:
        tensor
    """
    a = verts1[:, 2].min()
    b = verts1[:, 2].max()
    c = verts2[:, 2].min()
    d = verts2[:, 2].max()
    if d >= a and b >= c:
        return 0.0
    return torch.min(torch.abs(c - b), torch.abs(a - d))


def compute_random_rotations(B=10, upright=False, up=False):
    """
    Randomly samples rotation matrices.

    Args:
        B (int): Batch size.
        upright (bool): If True, samples rotations that are mostly upright. Otherwise,
            samples uniformly from rotation space.

    Returns:
        rotation_matrices (B x 3 x 3).
    """
    if upright:
        a1 = torch.FloatTensor(B, 1).uniform_(0, 2 * math.pi)
        a2 = torch.FloatTensor(B, 1).uniform_(-math.pi / 6, math.pi / 6)
        a3 = torch.FloatTensor(B, 1).uniform_(-math.pi / 12, math.pi / 12)

        angles = torch.cat((a1, a2, a3), 1).cuda()
        rotation_matrices = euler_angles_to_matrix(angles, "YXZ")
    elif up:
        a1 = torch.zeros(B, 1)
        a2 = torch.FloatTensor(B, 1).uniform_(0, 2 * math.pi)
        a3 = torch.zeros(B, 1)

        angles = torch.cat((a1, a2, a3), 1).cuda()
        rotation_matrices = euler_angles_to_matrix(angles, "ZYX")
    else:
        x1, x2, x3 = torch.split(torch.rand(3 * B).cuda(), B)
        tau = 2 * math.pi
        R = torch.stack(
            (  # B x 3 x 3
                torch.stack(
                    (torch.cos(tau * x1), torch.sin(tau * x1), torch.zeros_like(x1)), 1
                ),
                torch.stack(
                    (-torch.sin(tau * x1), torch.cos(tau * x1), torch.zeros_like(x1)), 1
                ),
                torch.stack(
                    (torch.zeros_like(x1), torch.zeros_like(x1), torch.ones_like(x1)), 1
                ),
            ),
            1,
        )
        v = torch.stack(
            (  # B x 3
                torch.cos(tau * x2) * torch.sqrt(x3),
                torch.sin(tau * x2) * torch.sqrt(x3),
                torch.sqrt(1 - x3),
            ),
            1,
        )
        identity = torch.eye(3).repeat(B, 1, 1).cuda()
        H = identity - 2 * v.unsqueeze(2) * v.unsqueeze(1)
        rotation_matrices = -torch.matmul(H, R)
    return rotation_matrices

def get_interset(bb1, bb2):
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    return intersection_area

def get_bev_bbox(verts):
    xmin = verts[:, 0].min()
    ymin = verts[:, 2].min()
    xmax = verts[:, 0].max()
    ymax = verts[:, 2].max()
    return torch.Tensor([xmin, ymin, xmax, ymax]).type_as(verts)

def compute_intersect(boxes):
    ins_label = torch.zeros((boxes.shape[0], boxes.shape[0]))
    for i in range(boxes.shape[0]):
        for j in range(boxes.shape[1]):
            if i != j:
                ins_label[i, j] = get_interset(bboxes[i], bboxes[j]) > 0

def get_instance_masks_within_global_cam(masks, mask_body, image_size, add_ignore=True):
    # 1, 0, -1; where -1 means ignore
    keep_masks_objs = []
    ori_masks_objs = []
    for k in range(masks.shape[0]):
        if add_ignore:
            ignore_mask = masks[0]
            for i in range(1, len(masks)):
                ignore_mask = ignore_mask | masks[i]
            
            ignore_mask = -ignore_mask.astype(np.float)
        else:
            ignore_mask = np.zeros(rend_size, rend_size)
        m = ignore_mask.copy()
        mask = masks[k]
        ori_masks_objs.append(masks[k].copy())
        m[mask] = mask[mask]
        keep_masks_objs.append(m)
    mask_obj_shape = masks.shape
    if len(keep_masks_objs) > 0:
        keep_masks_objs = np.stack(keep_masks_objs)
        padding = np.zeros((mask_obj_shape[0], image_size-mask_obj_shape[1], mask_obj_shape[2]))
        
        masks_annotations = np.concatenate((keep_masks_objs, padding), axis=1)

        # TODO: save original padding mask as masks_obj
        ori_masks_objs = np.stack(ori_masks_objs)
        masks_objs = np.concatenate((ori_masks_objs, padding), axis=1)
    else:
        masks_annotations = None
        masks_objs=None

    if mask_body.shape[0] != 0:
        if masks.shape[0] > 0:
            
            if add_ignore:
                ignore_mask = masks[0]
                for i in range(1, len(masks)):
                    ignore_mask = ignore_mask | masks[i]
                ignore_mask = -ignore_mask.astype(np.float)
            else:
                ignore_mask = np.zeros(rend_size, rend_size)
            m = ignore_mask.copy()
            # only support 1 person
            mask = mask_body[0]
            ori_mask_body = mask_body[0].copy()
            
            m[mask] = mask[mask]
            mask_body = m
        else:
            ori_mask_body = mask_body[0].copy()
            mask_body = mask_body[0]
        mask_body_shape = mask_body.shape
        padding = np.zeros((image_size-mask_body_shape[0], mask_body_shape[1]))
        mask_body = np.concatenate((mask_body, padding), axis=0)
        ori_mask_body = np.concatenate((ori_mask_body, padding), axis=0)
    else:
        ori_mask_body = mask_body.copy()
    
    return masks_objs, ori_mask_body[None], mask_body[None], masks_annotations

def make_bbox_square_torch(bbox, bbox_expansion=0.0):
    """
    Args:
        bbox (4 or B x 4): Bounding box in xyx2y2 format.
        bbox_expansion (float): Expansion factor to expand the bounding box extents from
            center.
    Returns:
        Squared bbox (same shape as bbox).
    """
    original_shape = bbox.shape
    bbox = bbox.reshape(-1, 4)
    center = torch.stack(
        ((bbox[:, 0] + bbox[:, 2]) / 2, (bbox[:, 1] + bbox[:, 3]) / 2), dim=1
    )
    b = torch.max(bbox[:, 2]-bbox[:,0], bbox[:, 3]-bbox[:, 1]).unsqueeze(1)
    b *= 1 + bbox_expansion
    
    square_bboxes = torch.cat((center - b / 2, center + b/2), dim=1)
    return square_bboxes.reshape(original_shape)

def get_local_mask_with_roi_camera(masks_objs, ori_camera, bboxes, rend_size=REND_SIZE, bbox_expansion=BBOX_EXPANSION_FACTOR):
    
    square_bboxes = make_bbox_square_torch(torch.from_numpy(bboxes)[:, :-1], bbox_expansion=bbox_expansion)
    
    ori_masks_objs = torch.from_numpy(masks_objs)
    import torchvision
    
    local_cam_list = []
    local_masks_list = []
    for i in range(masks_objs.shape[0]):
        x, y, x2, y2 = square_bboxes[i]
        local_mask = torchvision.ops.roi_align(ori_masks_objs[i][None, None, :, :].cuda(), [square_bboxes[i:i+1].cuda()], output_size=(REND_SIZE, REND_SIZE),spatial_scale=1.0, sampling_ratio=0).squeeze(1)
        local_masks_list.append(local_mask)
        
        local_cam = computer_roi_from_cam_inc((x, y), x2-x, 256, ori_camera)
        local_cam_list.append(local_cam.cpu().numpy())

    local_cams = np.stack(local_cam_list)
    local_masks_objs = torch.cat(local_masks_list, dim=0)
    return local_masks_objs.cpu().numpy(), local_cams

def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    if rotation_matrix.shape[1:] == (3,3):
        rot_mat = rotation_matrix.reshape(-1, 3, 3)
        hom = torch.tensor([0, 0, 1], dtype=torch.float32,
                           device=rotation_matrix.device).reshape(1, 3, 1).expand(rot_mat.shape[0], -1, -1)
        rotation_matrix = torch.cat([rot_mat, hom], dim=-1)

    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    aa = quaternion_to_angle_axis(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa

def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis

def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q
