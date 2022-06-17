import sys
from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os.path as osp
import json
import os
from tqdm import tqdm
import trimesh

from mover.utils.visualize import define_color_map
from mover.constants import (
    LOSS_NORMALIZE_PER_ELEMENT,
)

############# fuse sdf volume #############  
def get_vertices_from_sdf_volume_np(sdf_volume, voxel_size, grid_min):
    contact_index = (sdf_volume < 0).nonzero()
    contact_v_x = (contact_index[0] * 1.0 + 0.5) * voxel_size[0] + grid_min[0]
    contact_v_y = (contact_index[1] * 1.0 + 0.5)* voxel_size[1] + grid_min[1]
    contact_v_z = (contact_index[2] * 1.0 + 0.5)* voxel_size[2] + grid_min[2]
    contact_v = np.stack([contact_v_x, contact_v_y, contact_v_z], -1)
    
    out_index = (sdf_volume >= 0).nonzero()
    out_v_x = (out_index[0] * 1.0 + 0.5) * voxel_size[0] + grid_min[0]
    out_v_y = (out_index[1] * 1.0 + 0.5)* voxel_size[1] + grid_min[1]
    out_v_z = (out_index[2] * 1.0 + 0.5)* voxel_size[2] + grid_min[2]
    out_v = np.stack([out_v_x, out_v_y, out_v_z], -1)
    
    return contact_v, out_v
    
def fuse_sdf_volume(SDF_list, CONTACT_VOL_list, closest_faces_list, normals_list, contact_normals_list, closest_points_list, data_list):
    
    data = data_list[0]

    SDF_all = np.stack(SDF_list, axis=-1)
    SDF = np.min(SDF_all, -1)
    SDF_idx = np.argmin(SDF_all, -1)
    
    tmp_closest_faces = np.stack(closest_faces_list, axis=-1)
    closest_faces = tmp_closest_faces[np.arange(tmp_closest_faces.shape[0]), SDF_idx]

    tmp_normals_list = np.stack(normals_list, axis=-1)
    normals = tmp_normals_list[np.arange(tmp_normals_list.shape[0]), :, SDF_idx]

    
    tmp_closest_points_list = np.stack(closest_points_list, axis=-1)
    closest_points = tmp_closest_points_list[np.arange(tmp_closest_points_list.shape[0]), np.repeat(SDF_idx, 3)]

    if CONTACT_VOL_list[0] is not None:
        CONTACT_VOL = np.stack(CONTACT_VOL_list, axis=-1).sum(-1)
            # point 2 triangle normal.
        tmp_contact_normals_list = np.stack(contact_normals_list, axis=-1)
        contact_normals = tmp_contact_normals_list.sum(-1) / (CONTACT_VOL + 1e-9)[:,None].repeat(3, axis=1)

    else:
        CONTACT_VOL, contact_normals = None, None

    
    return SDF, CONTACT_VOL, closest_faces, normals, contact_normals, closest_points, data


def load_sdf_single(input_dir, CONTACT_FLAG=False):
    filename = 'all_fuse'
    print('process :', filename)
    sdf_out_fn = osp.join(input_dir, filename + '_sdf.npy')
    SDF = np.load(sdf_out_fn)
    
    closest_face_fn = osp.join(input_dir, filename + '_closest_faces.npy')
    closest_faces = np.load(closest_face_fn)

    normal_fn = osp.join(input_dir, filename + '_normals.npy')
    normals = np.load(normal_fn)
    if CONTACT_FLAG:
        contact_out_fn = osp.join(input_dir, filename + '_contact.npy') # per-person: count is how many person contact this region.
        CONTACT_VOL = np.load(contact_out_fn)

        contact_normal_fn = osp.join(input_dir, filename + '_contact_normals.npy')
        contact_normals = np.load(contact_normal_fn)
    else:
        CONTACT_VOL, contact_normals = None, None
    nn_fn = osp.join(input_dir, filename + '_nn.npy')
    closest_points = np.load(nn_fn) #.astype(np.float32))

    json_output_fn = osp.join(input_dir, filename + '.json')
    with open(json_output_fn, 'r') as f:
        data = json.load(f)

    return SDF,CONTACT_VOL,closest_faces, normals, contact_normals, closest_points, data


def fuse_sdf_multiple_person(all_scan_list, save_dir, CONTACT_FLAG=False, video_len=-1, random_sample_order=-1):
    pass
    SDF,CONTACT_VOL,closest_faces, normals, contact_normals, closest_points, data = \
                None, None, None, None, None, None, None
    obj_i = 0
    for scan_file in tqdm(all_scan_list):
        
        output_folder = os.path.join(save_dir, os.path.basename(scan_file).split('.')[0])
        tmp_SDF, tmp_CONTACT_VOL, tmp_closest_faces, tmp_normals, tmp_contact_normals, tmp_closest_points, tmp_data = \
            load_sdf_single(output_folder)
        obj_i += 1
        if SDF is None:
            SDF,CONTACT_VOL,closest_faces, normals, contact_normals, closest_points, data = \
                tmp_SDF, tmp_CONTACT_VOL, tmp_closest_faces, tmp_normals, tmp_contact_normals, tmp_closest_points, tmp_data 
        else:
            # merge multiple into one
            SDF_list = [SDF, tmp_SDF]
            CONTACT_VOL_list = [CONTACT_VOL, tmp_CONTACT_VOL]
            closest_faces_list = [closest_faces, tmp_closest_faces]
            normals_list = [normals, tmp_normals]
            contact_normals_list = [contact_normals, tmp_contact_normals]
            closest_points_list = [closest_points, tmp_closest_points]
            data_list = [data, tmp_data]
            SDF, CONTACT_VOL, closest_faces, normals, contact_normals, closest_points, data = fuse_sdf_volume(SDF_list, CONTACT_VOL_list, \
                        closest_faces_list, normals_list, contact_normals_list, closest_points_list, data_list)

    save_dir = os.path.join(save_dir, f'fuse_all_human_length{video_len}_random{random_sample_order}')

    os.makedirs(save_dir, exist_ok=True)
    if save_dir is not None: 
        filename = 'all_fuse'
        print('process :', filename)
        sdf_out_fn = osp.join(save_dir, filename + '_sdf.npy')
        np.save(sdf_out_fn, SDF)
        
        closest_face_fn = osp.join(save_dir, filename + '_closest_faces.npy')
        np.save(closest_face_fn, closest_faces)

        normal_fn = osp.join(save_dir, filename + '_normals.npy')
        np.save(normal_fn, normals)
        if CONTACT_FLAG:
            contact_out_fn = osp.join(save_dir, filename + '_contact.npy') # per-person: count is how many person contact this region.
            np.save(contact_out_fn, CONTACT_VOL)

            contact_normal_fn = osp.join(save_dir, filename + '_contact_normals.npy')
            np.save(contact_normal_fn, contact_normals)

        nn_fn = osp.join(save_dir, filename + '_nn.npy')
        np.save(nn_fn, closest_points.astype(np.float32))

        json_output_fn = osp.join(save_dir, filename + '.json')
        with open(json_output_fn, 'w') as f:
            json.dump(data, f)
        print('save finish')

        if True: # add this visualization.
            # save out sdf volume as ply
            grid_min = np.array(data['min'])
            grid_max = np.array(data['max'])
            grid_dim = data['dim']
            voxel_size = (grid_max - grid_min) / grid_dim
            sdf = SDF.reshape(grid_dim, grid_dim, grid_dim)
            in_body_vertices, out_body_verts = get_vertices_from_sdf_volume_np(sdf, voxel_size, grid_min)
            out_mesh = trimesh.Trimesh(in_body_vertices, process=False)
            template_save_fn = os.path.join(save_dir, 'in_body_sdf_sample.ply') 
            out_mesh.export(template_save_fn, vertex_normal=False) 
            
            back_ground_points = out_body_verts
            sample_num = int(back_ground_points.shape[0] / 1000)
            sample_back_ground_points = back_ground_points[np.random.choice(back_ground_points.shape[0], sample_num, replace=False)]
            
            out_mesh = trimesh.Trimesh(sample_back_ground_points, process=False)
            template_save_fn = os.path.join(save_dir, 'out_body_sdf_sample_0.00001.ply') 
            out_mesh.export(template_save_fn, vertex_normal=False) 
    
    return save_dir


# difference from "accumulate_whole_sdf_volume_scene": 
# 1. use_trimesh_contains=False
# 2. prefix='body'
def load_whole_sdf_volume(self, ply_file_list, contact_file_list, dim=256, padding=0.5, n_points_per_batch=1000000, \
        output_folder=None,
        device=torch.device('cuda'),
        dtype=torch.float32,
        debug=False):
    scene_name = 'all_fuse'

    ## fuse sdf volume.
    if not os.path.exists(os.path.join(output_folder, f'{scene_name}_sdf.npy')):

        logger.info(f'fuse sdf. for {len(ply_file_list)}')
        sdf_file_list = []
        for one in ply_file_list:
            file_name = os.path.basename(one).split('.')[0]
            sdf_file_name = os.path.dirname(os.path.dirname(one)) + f'/single_sdf_npy/{file_name}'
            sdf_file_list.append(sdf_file_name)
        tmp_sdf_dir = os.path.dirname(sdf_file_list[0])
        
        video_len = int(os.path.basename(output_folder).split('random')[0])
        random_sample_order = int(os.path.basename(output_folder).split('random')[-1])
        fuse_sdf_multiple_person(sdf_file_list, tmp_sdf_dir, video_len=video_len, random_sample_order=random_sample_order)
        
        sdf_output_folder = os.path.join(tmp_sdf_dir, f'fuse_all_human_length{video_len}_random{random_sample_order}')
        os.system(f'cp {sdf_output_folder}/* {output_folder}')

    logger.info(f'load sdf. from {output_folder}')
    assert os.path.exists(os.path.join(output_folder, f'{scene_name}_sdf.npy'))

    with open(osp.join(output_folder, scene_name + '.json'), 'r') as f:
        sdf_data = json.load(f)
        grid_min = torch.tensor(np.array(sdf_data['min']), dtype=dtype, device=device)
        grid_max = torch.tensor(np.array(sdf_data['max']), dtype=dtype, device=device)
        grid_dim = sdf_data['dim']
    voxel_size = (grid_max - grid_min) / grid_dim
    sdf = np.load(osp.join(output_folder, scene_name + '_sdf.npy')).reshape(grid_dim, grid_dim, grid_dim)
    sdf = torch.tensor(sdf, dtype=dtype, device=device)
    sdf_normals = np.load(osp.join(output_folder, scene_name + '_normals.npy')).reshape(grid_dim, grid_dim, grid_dim, 3)
    sdf_normals = torch.tensor(sdf_normals, dtype=dtype, device=device)
        
    self.register_buffer('sdf', sdf)
    self.register_buffer('sdf_normals', sdf_normals)
    self.register_buffer('grid_min', grid_min)
    self.register_buffer('grid_max', grid_max)
    self.register_buffer('voxel_size', voxel_size)

    return True

############# ! end of fuse sdf volume ############# 

def compute_sdf_loss(self, input_v_list, dtype=torch.float32, device=torch.device('cuda'), output_folder=None):
    # vertices: b * v * 3
    idx = [0]
    for v in input_v_list:
        idx.append(idx[-1] + v.shape[1])

    vertices = torch.cat(input_v_list, 1)
    nv = vertices.shape[1]

    grid_dim = self.sdf.shape[0]
    sdf_ids = torch.round(
        (vertices.squeeze() - self.grid_min) / self.voxel_size).to(dtype=torch.long)
    sdf_ids.clamp_(min=0, max=grid_dim-1)

    norm_vertices = (vertices - self.grid_min) / (self.grid_max - self.grid_min) * 2 - 1 # -1, 1

    # grid_sample: x,y,z sample in [D, H, W] dimension grid.
    body_sdf = F.grid_sample(self.sdf.view(1, 1, grid_dim, grid_dim, grid_dim),
                                norm_vertices[:, :, [2, 1, 0]].view(1, nv, 1, 1, 3),
                                padding_mode='border')

    sdf_normals = self.sdf_normals[sdf_ids[:,0], sdf_ids[:,1], sdf_ids[:,2]]

    # if there are no penetrating vertices then set sdf_penetration_loss = 0
    if body_sdf.lt(0).sum().item() < 1:
        sdf_penetration_loss = torch.tensor(0.0, dtype=dtype, device=device)
        sdf_loss_dict = [sdf_penetration_loss for _ in idx[1:]]
    else:
        sdf_penetration_loss = (body_sdf[body_sdf < 0].unsqueeze(dim=-1).abs() * sdf_normals[body_sdf.view(-1) < 0, :]).pow(2).sum(dim=-1).sqrt().sum()
        tmp_sdf_loss_v = (body_sdf[body_sdf < 0].unsqueeze(dim=-1).abs() * sdf_normals[body_sdf.view(-1) < 0, :]).pow(2).sum(dim=-1).sqrt()

        inside = (body_sdf < 0).squeeze()
        new_idx = [inside[:end].sum() for end in idx[1:]]
        new_idx.insert(0, 0)
        sdf_loss_dict = [tmp_sdf_loss_v[start:end].sum() for end, start in zip(new_idx[1:], new_idx[:-1])]
    return sdf_penetration_loss, sdf_loss_dict
