import sys
from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os.path as osp
import json
import os
import time
import trimesh
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

def get_prox_contact_labels(self, contact_parts='body'):
    # contact_body_parts = ['L_Leg', 'R_Leg', 'L_Hand', 'R_Hand', 'gluteus', 'back', 'thighs']
    if contact_parts == 'body':
        contact_body_parts = ['gluteus', 'back', 'thighs']
        if ADD_HAND_CONTACT:
            contact_body_parts.append('L_Hand')
            contact_body_parts.append('R_Hand')
    elif contact_parts == 'feet':
        contact_body_parts = ['L_Leg', 'R_Leg'] # TODO: use new defined front, back parts of feet.
    elif contact_parts == 'handArm':
        contact_body_parts = ['R_Hand', 'L_Hand', 'rightForeArm', 'leftForeArm']

    body_segments_dir = '/ps/scratch/multi-ioi/data/PROX/quantitative/body_segments'
    contact_verts_ids = []

    # load prox contact label information.
    for part in contact_body_parts:
        with open(os.path.join(body_segments_dir, part + '.json'), 'r') as f:
            data = json.load(f)
            contact_verts_ids.append(list(set(data["verts_ind"])))
    contact_verts_ids = np.concatenate(contact_verts_ids)
    return contact_verts_ids

def findNearestObj(body_v, verts_list):
    num_obj = len(verts_list)
    all_scene_v = torch.cat(verts_list, 1)
    obj_cnt_list = [0]
    for i in range(num_obj):
        obj_cnt_list.append(obj_cnt_list[-1] + verts_list[i].shape[1])
    
    import mover.dist_chamfer as ext
    distChamfer = ext.chamferDist()
    
    contact_dist, _, idx1, _ = distChamfer(body_v, all_scene_v)
    if idx1.device  == torch.device('cuda:0'):
        idx1 = idx1.cpu().numpy()
    nearest_idx = np.array([((idx1 >= obj_cnt_list[i]) & (idx1 <obj_cnt_list[i+1])).sum() for i in range(num_obj)])
    print('nearest_idx: ', nearest_idx)
    return np.argmax(nearest_idx)
    
def check_chamfer_small_thre(body_v, verts_list, thre=1.0):
    all_scene_v = verts_list
    import mover.dist_chamfer as ext
    distChamfer = ext.chamferDist()

    contact_dist, _, idx1, _ = distChamfer(body_v, all_scene_v)
    if contact_dist.mean() > thre:
        return False
    else:
        return True

# TODO: add class labels for contact.
def body2objs(self, body_v, body_vn, objs_v_list, name=None,\
    objs_class_list=None, body_v_class=None, contact_parts=None, depth_observe_filter=None):
    body_v = body_v.float()
    if body_v.shape[0] == 0 or len(objs_v_list) == 0:
        return -1

    overlap_objs = []
    overlap_cnt = []
    for i in range(len(self.idx_each_object)):
        ## handArm only used for table
        if contact_parts == 'handArm' and self.size_cls[i] != 7:
            continue

        if depth_observe_filter is not None and depth_observe_filter[i] == False:
            continue
            
        camera_body_v = torch.transpose(torch.matmul(self.get_cam_extrin(), \
                        torch.transpose(body_v[None], 2, 1)), 2, 1)
        body_vi, z = self.apply_projection_to_local_image(camera_body_v, obj_cam=i, resize=256)

        x = torch.clamp(body_vi[0, :, 0].long(), 0, 255)
        y = torch.clamp(body_vi[0, :, 1].long(), 0, 255)
        select_pixels = self.ref_mask[i][y, x]
        if select_pixels.sum() > 0:
            overlap_objs.append(i)
            overlap_cnt.append(select_pixels.sum())
        
    logger.info(f'overlap objs: {len(overlap_objs)}')
    ONLY_ONE = False
    if len(overlap_objs) == 0:
        return -1
    elif len(overlap_objs) > 1:
        tmp_overlap_cnt = torch.stack(overlap_cnt)
        tmp_max = tmp_overlap_cnt.max()
        tmp_ratio = tmp_overlap_cnt * 1.0 / tmp_max
        tmp_filter_flag = tmp_ratio > 0.05
        
        tmp_overlap_cnt = []
        tmp_overlap_objs = []
        for tmp_i in range(tmp_filter_flag.shape[0]):
            if tmp_filter_flag[tmp_i]:
                tmp_overlap_cnt.append(overlap_cnt[tmp_i])
                tmp_overlap_objs.append(overlap_objs[tmp_i])

        overlap_objs = tmp_overlap_objs
        overlap_cnt = tmp_overlap_cnt

        if objs_class_list is not None:
            filter_objs_v_list = [objs_v_list[i] for i in overlap_objs if objs_class_list[i] in [5, 6, 4, 7]]
            overlap_objs = [i for i in overlap_objs if objs_class_list[i] in [5, 6, 4, 7]]
        else:
            filter_objs_v_list = [objs_v_list[i] for i in overlap_objs]
        
        if len(filter_objs_v_list) == 0:
            return -1
        
        tmp = findNearestObj(body_v[None], filter_objs_v_list)
    else:
        ONLY_ONE = True
        tmp = 0

    nearest_obj_idx = overlap_objs[tmp]

    if not ONLY_ONE:
        if not check_chamfer_small_thre(body_v[None], objs_v_list[nearest_obj_idx]):
            return -1
        else:
            return nearest_obj_idx
    else:
        if overlap_cnt[0] > 20:
            return nearest_obj_idx
        else:
            return -1

def load_contact_body_to_objs(self, ply_file_list, contact_file_list, ftov, contact_parts='body', debug=True, output_folder=None):
    prox_contact_index = torch.Tensor(self.get_prox_contact_labels(contact_parts=contact_parts)).cuda().long()
    logger.info(f'load contact from {output_folder}')
    if  not self.RECALCULATE_HCI_INFO and os.path.exists(os.path.join(output_folder, f'accumulate_contact_{contact_parts}_vertices.npy')):
        logger.info(f'load contact from {output_folder} accumulate_contact_{contact_parts}_vertices.npy')
        accumulate_contact_v = np.load(os.path.join(output_folder, f'accumulate_contact_{contact_parts}_vertices.npy'))
        accumulate_contact_vn = np.load(os.path.join(output_folder, f'accumulate_contact_{contact_parts}_verts_normals.npy'))
        contact_cnt_list = np.load(os.path.join(output_folder, f'accumulate_contact_{contact_parts}_each_idx.npy'))

        self.register_buffer(f'accumulate_contact_{contact_parts}_vertices', torch.from_numpy(accumulate_contact_v).type(torch.float).cuda()) # B, N, 3
        self.register_buffer(f'accumulate_contact_{contact_parts}_verts_normals', torch.from_numpy(accumulate_contact_vn).type(torch.float).cuda()) # B, N, 3
        self.register_buffer(f'accumulate_contact_{contact_parts}_each_idx', torch.from_numpy(contact_cnt_list).type(torch.long).cuda())
        
        body2obj_npy_path = os.path.join(output_folder, f'accumulate_contact_{contact_parts}_body2obj_idx.npy')
        if body2obj_npy_path is not None and os.path.exists(body2obj_npy_path):
            logger.info(f'load {body2obj_npy_path}')
            body2obj_npy = np.load(body2obj_npy_path)
            self.register_buffer(f'accumulate_contact_{contact_parts}_body2obj_idx', torch.from_numpy(body2obj_npy).cuda().type(torch.long))
        return True
    return False

def get_overlaped_with_human_objs_idxs(self, frame_id):

    masks, valid_flag, valid_person = self.get_perframe_mask(frame_id)
    human_mask = masks[-1]
    from scipy.ndimage.morphology import binary_erosion
    human_mask_np = human_mask.cpu().numpy()
    
    filtered_overlap_idx_list = []
    if valid_person == True:
        return filtered_overlap_idx_list
    masks_object = self.masks_object == 1

    for idx, obj_idx in enumerate(valid_flag):
        static_obj_mask = masks_object[obj_idx].detach().cpu().numpy()
        if (static_obj_mask & human_mask_np).sum() > 0:
            filtered_overlap_idx_list.append(obj_idx)

    return filtered_overlap_idx_list
                    
def assign_contact_body_to_objs(self, ply_file_list, contact_file_list, ftov, contact_parts='body', debug=True, output_folder=None, assign2obj=True):
    device=torch.device('cuda')
    accumulate_contact_v = []
    accumulate_contact_vn = []
    
    # save all contact verts
    tmp_all_accumulate_contact_v = []
    tmp_all_accumulate_contact_vn = []

    prox_contact_index = torch.Tensor(self.get_prox_contact_labels(contact_parts=contact_parts)).cuda().long()
    
    logger.info(f'recalculate contact info')

    contact_cnt = 0
    contact_cnt_list = []

    assign_obj_list = []
    assign_obj_idxs_list = []
    ori_verts_parallel_ground, ori_verts_parallel_ground_list = self.get_verts_object_parallel_ground(return_all=True)

    frame_id = 0
    for ply_file, contact_file in tqdm(zip(ply_file_list, contact_file_list), desc=f'compute contact & assign bodys to different objs: to {output_folder}'):
        with torch.no_grad():
            mesh = Mesh(filename=ply_file)
            v = torch.tensor(mesh.v, device=device)
            f = torch.tensor(mesh.f.astype(np.int64), device=device)
            body_triangles = torch.index_select(v.unsqueeze(0), 1,f.view(-1)).view(1, -1, 3, 3)
            # Calculate the edges of the triangles
            # Size: BxFx3
            edge0 = body_triangles[:, :, 1] - body_triangles[:, :, 0]
            edge1 = body_triangles[:, :, 2] - body_triangles[:, :, 0]
            # Compute the cross product of the edges to find the normal vector of
            # the triangle
            body_normals = torch.cross(edge0, edge1, dim=2)
            # Normalize the result to get a unit vector
            body_normals = body_normals / \
                torch.norm(body_normals, 2, dim=2, keepdim=True)
            # compute the vertex normals from faces normals: face to vertices.
            body_v_normals = torch.mm(ftov, body_normals.squeeze().type(torch.float32))
            body_v_normals = body_v_normals / \
                torch.norm(body_v_normals, 2, dim=1, keepdim=True)
            # normal should point outside

            contact_labels = np.load(contact_file)

            contact_labels = torch.Tensor(contact_labels > 0.5).type(torch.uint8).cuda()

            ### pure posa results
            tmp_all_accumulate_contact_v.append(v[contact_labels[:, 0]])
            tmp_all_accumulate_contact_vn.append(body_v_normals[contact_labels[:, 0]])
            ### end of pure posa results

            tmp_contac_labels = torch.zeros(contact_labels.shape).type(torch.uint8).cuda()
            tmp_contac_labels[prox_contact_index] = True
            contact_labels = contact_labels & tmp_contac_labels

            accumulate_contact_v.append(v[contact_labels[:, 0]])
            accumulate_contact_vn.append(body_v_normals[contact_labels[:, 0]])


            # detected body have overlap to the objects;
            verts_parallel_ground_list = ori_verts_parallel_ground_list
                    
            if contact_parts in ['body', 'handArm'] and assign2obj:
                logger.info('assign body to objs')

                overlap_objs_by_detection = self.get_overlaped_with_human_objs_idxs(frame_id)
                overlap_objs_by_detection_flag = []
                for tmp_i in range(len(verts_parallel_ground_list)):
                    if tmp_i in overlap_objs_by_detection:
                        overlap_objs_by_detection_flag.append(True)
                    else:
                        overlap_objs_by_detection_flag.append(False)
                        
                if ADD_HAND_CONTACT:
                    pass
                else:
                    assign_obj = self.body2objs(v[contact_labels[:, 0]], body_v_normals[contact_labels[:, 0]],\
                        verts_parallel_ground_list, name=ply_file, objs_class_list=self.size_cls, contact_parts=contact_parts, \
                        depth_observe_filter=overlap_objs_by_detection_flag)
                    assign_obj_idxs = torch.ones(v[contact_labels[:, 0]].shape[0]) * assign_obj

                logger.info(f'assign to obj {assign_obj}')
                assign_obj_list.append(assign_obj)
                assign_obj_idxs_list.append(assign_obj_idxs)

            contact_cnt += v[contact_labels[:, 0]].shape[0]
            contact_cnt_list.append(contact_cnt)
            frame_id += 1

    if debug and torch.cat(accumulate_contact_v).shape[0] != 0:
        tmp_cv = torch.cat(accumulate_contact_v).cpu().numpy()
        tmp_cvn = torch.cat(accumulate_contact_vn).cpu().numpy()
        if output_folder is not None:
            template_save_fn = os.path.join(output_folder, f'contact_mesh_points_{contact_parts}.ply')
        
        # export camera CS contact points
        camera_mat = self.get_cam_extrin().squeeze().detach().cpu().numpy()
        self.viz_verts(tmp_cv, template_save_fn, verts_n=tmp_cvn, camera_mat=camera_mat)
        
        if contact_parts in ['body', 'handArm'] :
            all_color = []
            for n_i, assign_idx in enumerate(assign_obj_list):
                n_p = accumulate_contact_v[n_i].shape[0]
                if assign_idx != -1:
                    color_idx = self.size_cls[assign_idx].cpu().numpy().astype(np.long)
                    color = np.clip(np.array(define_color_map[color_idx.item()]) + assign_idx * 0.03, 0.01, 0.99)
                else:
                    color = np.array([1.0, 1.0, 1.0])
                color = color[None].repeat(n_p, 0)
                all_color.append(color)
            if len(all_color) > 0:
                vert_colors = (np.concatenate(all_color).reshape(-1, 3) * 255.0).astype(np.uint8)
            else:
                vert_colors = None
            if output_folder is not None:
                template_save_fn = os.path.join(output_folder, f'contact_mesh_points_{contact_parts}_color.ply')
            self.viz_verts(tmp_cv, template_save_fn, verts_c=vert_colors, camera_mat=camera_mat)

        all_contact_v = torch.cat(tmp_all_accumulate_contact_v).cpu().numpy()
        all_contact_vn = torch.cat(tmp_all_accumulate_contact_vn).cpu().numpy()
        if output_folder is not None:
            template_save_fn = os.path.join(output_folder, f'contact_mesh_points_{contact_parts}_oriPOSA.ply')
        self.viz_verts(all_contact_v, template_save_fn, verts_n=all_contact_vn, camera_mat=camera_mat)

    if output_folder is not None:
        np.save(os.path.join(output_folder, f'accumulate_contact_{contact_parts}_vertices.npy'), \
                torch.cat(accumulate_contact_v).unsqueeze(0).type(torch.float).cpu().numpy())
        np.save(os.path.join(output_folder, f'accumulate_contact_{contact_parts}_verts_normals.npy'), \
                torch.cat(accumulate_contact_vn).unsqueeze(0).cpu().numpy())
        np.save(os.path.join(output_folder, f'accumulate_contact_{contact_parts}_each_idx.npy'), \
                torch.Tensor(contact_cnt_list).cuda().type(torch.long).cpu().numpy())
        np.save(os.path.join(output_folder, f'tmp_all_accumulate_contact_{contact_parts}_vertices.npy'), \
                torch.cat(tmp_all_accumulate_contact_v).unsqueeze(0).type(torch.float).cpu().numpy())
        np.save(os.path.join(output_folder, f'tmp_all_accumulate_contact_{contact_parts}_verts_normals.npy'), \
                torch.cat(tmp_all_accumulate_contact_vn).unsqueeze(0).cpu().numpy())

    self.register_buffer(f'accumulate_contact_{contact_parts}_vertices', torch.cat(accumulate_contact_v).unsqueeze(0).type(torch.float)) # B, N, 3
    self.register_buffer(f'accumulate_contact_{contact_parts}_verts_normals', torch.cat(accumulate_contact_vn).unsqueeze(0)) # B, N, 3
    self.register_buffer(f'accumulate_contact_{contact_parts}_each_idx', torch.Tensor(contact_cnt_list).cuda().type(torch.long))
    
    self.register_buffer(f'tmp_all_accumulate_contact_{contact_parts}_vertices', \
        torch.cat(tmp_all_accumulate_contact_v).unsqueeze(0).type(torch.float)) # B, N, 3
    self.register_buffer(f'tmp_all_accumulate_contact_{contact_parts}_verts_normals', \
        torch.cat(tmp_all_accumulate_contact_vn).unsqueeze(0)) # B, N, 3
    
    if contact_parts in ['body', 'handArm']  and assign2obj:
        if output_folder is not None:
            np.save(os.path.join(output_folder, f'accumulate_contact_{contact_parts}_body2obj_idx.npy'), \
                    torch.cat(assign_obj_idxs_list).unsqueeze(0).type(torch.long).cpu().numpy())
        self.register_buffer(f'accumulate_contact_{contact_parts}_body2obj_idx', torch.cat(assign_obj_idxs_list).unsqueeze(0).type(torch.long))
    
    return True


def voxelize_contact_vertices(self, ori_contact_v, ori_contact_vn, voxel_size, grid_min, dim, \
                    assign_obj_idxs=None, device=None, contact_parts='body', debug=False, save_dir=None):
    start = time.time()
    v_volume = torch.zeros((dim, dim, dim)).to(device)
    cvn_volume = torch.zeros((dim, dim, dim, 3)).to(device)
    cb2j_volume = torch.zeros((dim, dim, dim)).to(device).long()
    query_points = torch.floor((ori_contact_v-grid_min) / voxel_size - 0.5).long().squeeze(1)
    contact_vn = ori_contact_vn.squeeze()
    assign_obj_idxs_dict = {}
    for i in tqdm(range(query_points.shape[0]), desc='voxelize contact vertices'):
        tx,ty,tz = query_points[i][0], query_points[i][1], query_points[i][2]
        cvn_volume[tx, ty, tz] = \
                            (cvn_volume[tx, ty, tz] * v_volume[tx, ty, tz] + contact_vn[i]) / (v_volume[tx, ty, tz]+1)  
        v_volume[tx,ty,tz] += 1
        if assign_obj_idxs is not None:
            if query_points[i] not in assign_obj_idxs_dict:
                assign_obj_idxs_dict[query_points[i]] = [assign_obj_idxs[0, i].item()]
            else:
                assign_obj_idxs_dict[query_points[i]].append(assign_obj_idxs[0, i].item())
    if assign_obj_idxs is not None:
        for key, value in assign_obj_idxs_dict.items():
            from collections import Counter
            tmp = Counter(value).most_common(1)[0][0]
            cb2j_volume[key[0], key[1], key[2]] = torch.from_numpy(np.array([tmp])).long().cuda()
    contact_v, contact_vn, contact_b2j = self.get_contact_vertices_from_volume(v_volume, cvn_volume, \
                                            voxel_size, grid_min, cb2j_volume)

    self.register_buffer(f'voxel_contact_{contact_parts}_vertices', contact_v[None]) # B, N, 3
    self.register_buffer(f'voxel_contact_{contact_parts}_verts_normals', contact_vn[None]) # B, N, 3
    if contact_b2j is not None:
        self.register_buffer(f'voxel_contact_{contact_parts}_body2obj_idx', contact_b2j.unsqueeze(0).type(torch.long))
    
    consumption = time.time()-start
    logger.info(f'voxelize contact vertices: {consumption}')
    
    if debug and save_dir is not None and contact_v.shape[0] > 0:
        tmp_save_path = os.path.join(save_dir, 'voxel_contact_verts.ply')
        camera_mat = self.get_cam_extrin().squeeze().detach().cpu().numpy()
        self.viz_verts(contact_v.cpu().numpy(), tmp_save_path, verts_n=contact_vn.cpu().numpy(), camera_mat=camera_mat)

    return True

## accumulate contact loss
def get_contact_vertices_from_volume(self, contact_volume, contact_normals, voxel_size, grid_min, cb2j_volume=None):
    contact_index = (contact_volume > 0).nonzero()
    contact_v_x = (contact_index[:, 0].float() + 0.5) * voxel_size[0] + grid_min[0]
    contact_v_y = (contact_index[:, 1].float() + 0.5)* voxel_size[1] + grid_min[1]
    contact_v_z = (contact_index[:, 2].float() + 0.5)* voxel_size[2] + grid_min[2]
    contact_v = torch.stack([contact_v_x, contact_v_y, contact_v_z], -1)
    contact_vn = contact_normals[contact_index[:, 0], contact_index[:, 1], contact_index[:, 2], :]

    if cb2j_volume is None:
        return contact_v, contact_vn
    else:
        contact_b2j = cb2j_volume[contact_index[:, 0], contact_index[:, 1], contact_index[:, 2]]
        return contact_v, contact_vn, contact_b2j



def get_vertices_from_sdf_volume(self, sdf_volume, voxel_size, grid_min):
    contact_index = (sdf_volume < 0).nonzero()
    contact_v_x = (contact_index[:, 0].float() + 0.5) * voxel_size[0] + grid_min[0]
    contact_v_y = (contact_index[:, 1].float() + 0.5)* voxel_size[1] + grid_min[1]
    contact_v_z = (contact_index[:, 2].float() + 0.5)* voxel_size[2] + grid_min[2]
    contact_v = torch.stack([contact_v_x, contact_v_y, contact_v_z], -1)
    
    out_index = (sdf_volume >= 0).nonzero()
    out_v_x = (out_index[:, 0].float() + 0.5) * voxel_size[0] + grid_min[0]
    out_v_y = (out_index[:, 1].float() + 0.5)* voxel_size[1] + grid_min[1]
    out_v_z = (out_index[:, 2].float() + 0.5)* voxel_size[2] + grid_min[2]
    out_v = torch.stack([out_v_x, out_v_y, out_v_z], -1)
    
    return contact_v, out_v

