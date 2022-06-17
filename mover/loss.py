
import itertools
import json

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.nn.functional import smooth_l1_loss, mse_loss
from scipy.ndimage.morphology import distance_transform_edt
from loguru import logger
import neural_renderer as nr

from mover.constants import (
    BBOX_EXPANSION,
    BBOX_EXPANSION_PARTS,
    DEFAULT_LOSS_WEIGHTS,
    IMAGE_SIZE,
    INTERACTION_MAPPING,
    INTERACTION_THRESHOLD,
    MEAN_INTRINSIC_SCALE,
    MESH_MAP,
    PART_LABELS,
    REND_SIZE,
    SMPL_FACES_PATH,
    DEBUG,
    BBOX_EXPANSION_FACTOR,
    LOSS_NORMALIZE_PER_ELEMENT,
    PYTORCH3D_DEPTH,
)
from mover.utils.bbox import check_overlap, compute_iou
from mover.utils.camera import (
    compute_transformation_ortho,
    compute_transformation_persp,
    get_Y_euler_rotation_matrix,
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

class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss
        
class Losses(object): # TODO: not using register_buffer
    def __init__(
        self,
        renderer,
        K_rois,
        labels_person,
        labels_object,
        interaction_map_parts,
        class_name='chair',
        ref_mask=None,
        keep_mask=None,
        lw_chamfer=0.5,
        kernel_size=7,
        power=0.25,
    ):
        self.renderer = nr.renderer.Renderer(
            image_size=REND_SIZE, K=renderer.K, R=renderer.R, t=renderer.t, orig_size=REND_SIZE
        ) # TODO: Local camera use REND_SIZE
        
        self.ref_mask = ref_mask
        self.keep_mask = keep_mask
        self.K_rois = K_rois
        self.thresh = INTERACTION_THRESHOLD[class_name]
        self.mse = torch.nn.MSELoss()
        self.class_name = class_name
        self.labels_person = labels_person
        self.labels_object = labels_object

        self.expansion = BBOX_EXPANSION[class_name]
        self.expansion_parts = BBOX_EXPANSION_PARTS[class_name]
        self.interaction_map = INTERACTION_MAPPING[class_name]
        self.interaction_map_parts = interaction_map_parts

        self.interaction_pairs = None
        self.interaction_pairs_parts = None
        self.bboxes_parts_person = None
        self.bboxes_parts_object = None

        # edge loss: a one-way chamfer loss
        # Convention for silhouette-aware loss: -1=occlusion, 0=bg, 1=fg.
        
        batch_size = ref_mask.shape[0]
        self.pool = torch.nn.MaxPool2d(
            kernel_size=kernel_size, stride=1, padding=(kernel_size // 2))
        edt_list = []
        for i in range(batch_size):
            mask_edge = self.compute_edges(ref_mask[i][None, None, :, :]).cpu().numpy()
            edt = distance_transform_edt(1 - (mask_edge > 0)) ** (power * 2)
            edt_list.append(edt.squeeze())
        all_edt = np.stack(edt_list)
        
        self.edt_ref_edge = torch.from_numpy(all_edt).float().cuda()
        self.lw_chamfer = lw_chamfer
    
    def compute_edges(self, silhouette):
        return self.pool(silhouette) - silhouette

    def assign_interaction_pairs(self, verts_person, verts_object):
        """
        Assigns pairs of people and objects that are interacting. Note that multiple
        people can be assigned to the same object, but one person cannot be assigned to
        multiple objects. (This formulation makes sense for objects like motorcycles
        and bicycles. Can be changed for handheld objects like bats or rackets).

        This is computed separately from the loss function because there are potential
        speed improvements to re-using stale interaction pairs across multiple
        iterations (although not currently being done).

        A person and an object are interacting if the 3D bounding boxes overlap:
            * Check if X-Y bounding boxes overlap by projecting to image plane (with
              some expansion defined by BBOX_EXPANSION), and
            * Check if Z overlaps by thresholding distance.

        Args:
            verts_person (N_p x V_p x 3).
            verts_object (N_o x V_o x 3).

        Returns:
            interaction_pairs: List[Tuple(person_index, object_index)]
        """
        with torch.no_grad():
            bboxes_object = [
                project_bbox(v, self.renderer, bbox_expansion=self.expansion)
                for v in verts_object
            ]
            bboxes_person = [
                project_bbox(v, self.renderer, self.labels_person, self.expansion)
                for v in verts_person
            ]
            num_people = len(bboxes_person)
            num_objects = len(bboxes_object)
            ious = np.zeros((num_people, num_objects))
            for part_person in self.interaction_map:
                for i_person in range(num_people):
                    for i_object in range(num_objects):
                        iou = compute_iou(
                            bbox1=bboxes_object[i_object],
                            bbox2=bboxes_person[i_person][part_person],
                        )
                        ious[i_person, i_object] += iou

            self.interaction_pairs = []
            for i_person in range(num_people):
                i_object = np.argmax(ious[i_person])
                if ious[i_person][i_object] == 0:
                    continue
                dist = compute_dist_z(verts_person[i_person], verts_object[i_object])
                if dist < self.thresh:
                    self.interaction_pairs.append((i_person, i_object))
            return self.interaction_pairs

    def assign_interaction_pairs_parts(self, verts_person, verts_object):
        """
        Assigns pairs of person parts and objects pairs that are interacting.

        This is computed separately from the loss function because there are potential
        speed improvements to re-using stale interaction pairs across multiple
        iterations (although not currently being done).

        A part of a person and a part of an object are interacting if the 3D bounding
        boxes overlap:
            * Check if X-Y bounding boxes overlap by projecting to image plane (with
              some expansion defined by BBOX_EXPANSION_PARTS), and
            * Check if Z overlaps by thresholding distance.

        Args:
            verts_person (N_p x V_p x 3).
            verts_object (N_o x V_o x 3).

        Returns:
            interaction_pairs_parts:
                List[Tuple(person_index, person_part, object_index, object_part)]
        """
        with torch.no_grad():
            bboxes_person = [
                project_bbox(v, self.renderer, self.labels_person, self.expansion_parts)
                for v in verts_person
            ]
            bboxes_object = [
                project_bbox(v, self.renderer, self.labels_object, self.expansion_parts)
                for v in verts_object
            ]
            self.interaction_pairs_parts = []
            for i_p, i_o in itertools.product(
                range(len(verts_person)), range(len(verts_object))
            ):
                for part_object in self.interaction_map_parts.keys():
                    for part_person in self.interaction_map_parts[part_object]:
                        bbox_object = bboxes_object[i_o][part_object]
                        bbox_person = bboxes_person[i_p][part_person]
                        is_overlapping = check_overlap(bbox_object, bbox_person)
                        z_dist = compute_dist_z(
                            verts_object[i_o][self.labels_object[part_object]],
                            verts_person[i_p][self.labels_person[part_person]],
                        )
                        if is_overlapping and z_dist < self.thresh:
                            self.interaction_pairs_parts.append(
                                (i_p, part_person, i_o, part_object)
                            )
            return self.interaction_pairs_parts

    def compute_sil_loss(self, verts, faces, bboxes, edge_loss=True, debug=False, det_score=None): 
        # input list
        # TODO: use bboxes to crop the mask
        square_bboxes = make_bbox_square_torch(bboxes, bbox_expansion=BBOX_EXPANSION_FACTOR)
        
        import torchvision
        loss_sil = torch.tensor(0.0).float().cuda()
        if edge_loss:
            loss_edge = torch.tensor(0.0).float().cuda()
        if debug:
            sil_loss_list = []
            edge_loss_list = []
        for i in range(len(verts)):
            v = verts[i].unsqueeze(0) if len(verts[i].shape) == 2 else verts[i]
            # import pdb; pdb.set_trace()
            K = self.K_rois[i]
            rend = self.renderer(v, faces[i], K=K, mode="silhouettes")
            image = self.keep_mask[i] * rend
            
            local_image = image
            local_ref_mask = self.ref_mask[i]
            local_keep_mask = self.keep_mask[i]
            if LOSS_NORMALIZE_PER_ELEMENT:
                l_m = torch.sum((local_image - local_ref_mask)**2) / (local_keep_mask.sum() + 1e-9)
            else:
                l_m = torch.sum((local_image - local_ref_mask)**2) #/ (local_keep_mask.sum() + 1e-9)
            if debug:
                sil_loss_list.append(l_m)

            if det_score is not None:
                loss_sil = loss_sil +  l_m * det_score[i]
            else:
                loss_sil = loss_sil +  l_m 
            if edge_loss:
                tmp = self.compute_edges(image)
                if LOSS_NORMALIZE_PER_ELEMENT:
                    l_chamfer = self.lw_chamfer * torch.sum(
                        tmp * self.edt_ref_edge[i:i+1], dim=(1, 2)
                    ) / (tmp.sum() + 1e-9)
                else:
                    l_chamfer = self.lw_chamfer * torch.sum(
                        tmp * self.edt_ref_edge[i:i+1], dim=(1, 2)
                    )
                
                if det_score is not None:
                    loss_edge += l_chamfer[0] * det_score[i]
                else:
                    loss_edge += l_chamfer[0]
                if debug:
                    edge_loss_list.append(l_chamfer[0])
        
        if not edge_loss:
            if not debug:
                return {"loss_sil": loss_sil / len(verts)}
            else:
                return {"loss_sil": loss_sil / len(verts), "sil_loss_list": sil_loss_list}
        else:
            if not debug:
                return {"loss_sil": loss_sil / len(verts), "loss_edge": loss_edge / len(verts)}
            else:
                return {"loss_sil": loss_sil / len(verts), "loss_edge": loss_edge / len(verts), "sil_loss_list": sil_loss_list, "edge_loss_list": edge_loss_list}
    
    def compute_sil_loss_for_one_obj(self, verts, faces, bboxes, 
            edge_loss=True, debug=False, idx=-1): 
        assert idx != -1
        square_bboxes = make_bbox_square_torch(bboxes, bbox_expansion=BBOX_EXPANSION_FACTOR)
        
        import torchvision
        loss_sil = torch.tensor(0.0).float().cuda()
        if edge_loss:
            loss_edge = torch.tensor(0.0).float().cuda()
        if debug:
            sil_loss_list = []
        i = idx
        v = verts[i].unsqueeze(0)
        K = self.K_rois[i]
        rend = self.renderer(v, faces[i], K=K, mode="silhouettes")
        image = self.keep_mask[i] * rend
        
        local_image = image
        local_ref_mask = self.ref_mask[i]
        local_keep_mask = self.keep_mask[i]
        l_m = torch.sum((local_image - local_ref_mask)**2) / local_keep_mask.sum()
        if debug:
            sil_loss_list.append(l_m)
        
        loss_sil += l_m
        if edge_loss:
            
            tmp = self.compute_edges(image)
            l_chamfer = self.lw_chamfer * torch.sum(
                tmp * self.edt_ref_edge[i:i+1], dim=(1, 2)
            ) / (tmp.sum().to(torch.float32) + 1e-9)

            loss_edge += l_chamfer[0]
             
        if not edge_loss:
            if not debug:
                return {"loss_sil": loss_sil / len(verts)}
            else:
                return {"loss_sil": loss_sil / len(verts), "sil_loss_list": sil_loss_list}
        else:
            if not debug:
                return {"loss_sil": loss_sil / len(verts), "loss_edge": loss_edge / len(verts)}
            else:
                return {"loss_sil": loss_sil / len(verts), "loss_edge": loss_edge / len(verts), "sil_loss_list": sil_loss_list}

    def compute_interaction_loss(self, verts_person, verts_object):
        """
        Computes interaction loss.
        """
        loss_interaction = torch.tensor(0.0).float().cuda()
        interaction_pairs = self.assign_interaction_pairs(verts_person, verts_object)
        for i_person, i_object in interaction_pairs:
            v_object = verts_object[i_object]
            centroid_error = self.mse(v_person.mean(0), v_object.mean(0))
            loss_interaction += centroid_error
        num_interactions = max(len(interaction_pairs), 1)
        return {"loss_inter": loss_interaction / num_interactions}

    def compute_interaction_loss_parts(self, verts_person, verts_object):
        loss_interaction_parts = torch.tensor(0.0).float().cuda()
        interaction_pairs_parts = self.assign_interaction_pairs_parts(
            verts_person=verts_person, verts_object=verts_object
        )
        for i_p, part_p, i_o, part_o in interaction_pairs_parts:
            v_person = verts_person[i_p][self.labels_person[part_p]]
            v_object = verts_object[i_o][self.labels_object[part_o]]
            dist = self.mse(v_person.mean(0), v_object.mean(0))
            loss_interaction_parts += dist
        num_interactions = max(len(self.interaction_pairs_parts), 1)
        return {"loss_inter_part": loss_interaction_parts / num_interactions}

    def compute_intrinsic_scale_prior(self, intrinsic_scales, intrinsic_mean, reduce=False):
        if not reduce:
            return torch.sum((intrinsic_scales - intrinsic_mean) ** 2) / intrinsic_scales.shape[0]
        else:
            return torch.sum((intrinsic_scales - intrinsic_mean) ** 2, dim=-1) 

    def compute_ordinal_depth_loss(self, masks, silhouettes, depths, use_for_human=True):
        loss = torch.tensor(0.0).float().cuda()
        num_pairs = 1e-9 # incase zeros: leads to nan
        if use_for_human:
            tmp_cnt = 0
            for i in range(len(silhouettes)):
                for j in range(len(silhouettes)):
                    if not (i == len(silhouettes)-1 or j == len(silhouettes)-1):
                        continue
                    has_pred = silhouettes[i] & silhouettes[j]
                    if has_pred.sum() == 0:
                        continue
                    front_i_gt = masks[i] & (~masks[j])
                    front_j_pred = depths[j] < depths[i]
                    m = front_i_gt & front_j_pred & has_pred
                    if m.sum() == 0:
                        continue
                    else:
                        num_pairs += 1
                    
                    dists = torch.clamp(depths[i] - depths[j], min=0.0, max=2.0)
                    
                    loss += torch.sum(torch.log(1 + torch.exp(dists))[m]) # original from mover
            loss /= num_pairs
        else:
            tmp_cnt = 0
            for i in range(len(silhouettes)):
                for j in range(len(silhouettes)):
                    has_pred = silhouettes[i] & silhouettes[j]
                    if has_pred.sum() == 0:
                        continue
                    else:
                        num_pairs += 1
                    front_i_gt = masks[i] & (~masks[j])
                    front_j_pred = depths[j] < depths[i]
                    m = front_i_gt & front_j_pred & has_pred

                    if m.sum() == 0:
                        continue
                    
                    dists = torch.clamp(depths[i] - depths[j], min=0.0, max=2.0)
                    loss += torch.sum(torch.log(1 + torch.exp(dists))[m])
            loss /= num_pairs
        return {"loss_depth": loss}

    @staticmethod
    def _compute_iou_1d(a, b):
        """
        a: (2).
        b: (2).
        """
        o_l = torch.min(a[0], b[0])
        o_r = torch.max(a[1], b[1])
        i_l = torch.max(a[0], b[0])
        i_r = torch.min(a[1], b[1])
        inter = torch.clamp(i_r - i_l, min=0)
        return inter / (o_r - o_l)