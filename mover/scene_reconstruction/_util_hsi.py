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
from mover.utils.geometry import (
    center_vertices,
    combine_verts,
    compute_dist_z,
    matrix_to_rot6d,
    rot6d_to_matrix,
    compute_intersect,
    get_bev_bbox,
)
import numpy as np

def get_y_verts(vn, axis_angle=45, along=True):
    # input: objects verts <-> body verts.    
    # along y-axis;
    axis = torch.tensor([0, 1, 0]).type_as(vn)
    angles = torch.acos((vn * axis).sum(-1)) *180 / np.pi

    if along: # but toward y-axis, object toward -y.
        # ! torch version == 1.8.1 does not support without int()
        valid_contact_mask = (angles.le(axis_angle).int()+angles.ge(180 - axis_angle).int()).ge(1) 
    else:
        valid_contact_mask = (angles.ge(axis_angle).int()+angles.le(180 - axis_angle).int()).ge(2)

    return valid_contact_mask

def assign_human_masks(self, masks_human=None, min_overlap=0.5):
    """self.rotate_cam_pitch
    image
    1. Compute IOU between all human silhouettes and human masks
    2. Sort IOUs
    3. Assign people to masks in order, skipping people and masks that
        have already been assigned.

    Args:
        masks_human: Human bitmask tensor from instance segmentation algorithm.
        min_overlap (float): Minimum IOU threshold to assign the human mask to a
            human instance.

    Returns:
        N_h x
    """
    f = self.faces_person
    verts_person = self.get_verts_person()
    if masks_human is None:
        return torch.zeros(verts_person.shape[0], IMAGE_SIZE, IMAGE_SIZE).cuda()
    person_silhouettes = torch.cat(
        [self.renderer(v.unsqueeze(0), f, mode="silhouettes") for v in verts_person]
    ).bool()

    intersection = masks_human.unsqueeze(0) & person_silhouettes.unsqueeze(1)
    union = masks_human.unsqueeze(0) | person_silhouettes.unsqueeze(1)

    iou = intersection.sum(dim=(2, 3)).float() / union.sum(dim=(2, 3)).float()
    iou = iou.cpu().numpy()
    # https://stackoverflow.com/questions/30577375
    best_indices = np.dstack(np.unravel_index(np.argsort(-iou.ravel()), iou.shape))[
        0
    ]
    human_indices_used = set()
    mask_indices_used = set()
    # If no match found, mask will just be empty, incurring 0 loss for depth.
    human_masks = torch.zeros(verts_person.shape[0], IMAGE_SIZE, IMAGE_SIZE).bool()
    for human_index, mask_index in best_indices:
        if human_index in human_indices_used:
            continue
        if mask_index in mask_indices_used:
            continue
        if iou[human_index, mask_index] < min_overlap:
            break
        human_masks[human_index] = masks_human[mask_index]
        human_indices_used.add(human_index)
        mask_indices_used.add(mask_index)
    return human_masks.cuda()

## multiple human
def get_verts_person(self): # person estimate wrt fixed camera coordinates system.
    return torch.transpose(torch.matmul(self.get_cam_extrin(), \
            torch.transpose(self.verts_person_og.unsqueeze(0), 2, 1)), 2, 1)

# ! warning: why not left * inverse matrix.
def get_person_wrt_world_coordinates(self, body_vertices):
    extra_extrin_mat = self.get_relative_extrin_new()
    return torch.transpose(torch.matmul(extra_extrin_mat , \
            torch.transpose(body_vertices, 2, 1)), 2, 1)  # * old results

## get perframe detection results
def get_perframe_mask(self, idx): # merge human body mask as supervision. 
        
        # before 0522: only consider the objects have overlap with human, for compute_ordinal_depth_perframe
        human_det = self.perframe_det_results[idx].float()[-1]
        # TODO: check if it is consistant with render SMPLX
        valid_flag = [False for _ in range(self.masks_object.shape[0])]

        for j in range(self.det_results.shape[0]):
            
            iou = compute_iou(human_det[:-1], self.det_results[j][:-1])
            if iou > 0:
                valid_flag[j] = True

        perframe_masks = []
        human_mask = self.perframe_masks[idx][-1:]
        for one in range(self.masks_object.shape[0]):
            obj_mask = self.masks_object[one].clone()
            obj_mask = obj_mask - human_mask.float()
            obj_mask[obj_mask == -1] = 0
            perframe_masks.append(obj_mask==1)

        perframe_masks = [perframe_masks[idx] for idx, flag in enumerate(valid_flag) if flag] 
        perframe_masks.append(human_mask)
        perframe_masks = torch.cat(perframe_masks)
        
        # only subtract human mask on 2D static scene detection. keep all object.
        return perframe_masks, [one for one in range(len(valid_flag)) if valid_flag[one]], human_mask.sum() < 1e-7

def get_contact_flag(self, body_vertices):
    # utility for check overlap between human and objects
    with torch.no_grad():
        verts = self.get_verts_object_parallel_ground()
        verts_list = []
        for one in range(len(self.idx_each_object)):
            obj_verts, obj_faces = self.get_one_verts_faces_obj(verts, one)
            verts_list.append(obj_verts.squeeze(0))
        boxes = self.sdf_losses.get_bounding_boxes(verts_list)

        body_verts_batch = body_vertices.shape[0]
        body_verts_list = [body_vertices[one] for one in range(body_verts_batch)]
        body_boxes = self.sdf_losses.get_bounding_boxes(body_verts_list)

        contact_flag = []
        for i in range(len(body_boxes)):
            contact_flag.append(self.contact_with_scene_flag(boxes, body_boxes[i]))

        return torch.Tensor(contact_flag).cuda()

def contact_with_scene_flag(self, obj_boxes, body_box):
    num_objs = len(obj_boxes)
    flag = False
    for j in range(num_objs):
        if self.sdf_losses.check_overlap(obj_boxes[j], body_box):
            flag = True
            break
    return flag