import sys
sys.path.append('.')
import argparse
# from utils.sunrgbd_config import SUNRGBD_CONFIG
import os
import json
import pickle
import numpy as np
import scipy.io as sio
from glob import glob
import trimesh
from loguru import logger

def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

def load_meshes_from_obj(obj_fn):
    mesh = trimesh.load(obj_fn, process=False)
    mesh = as_mesh(mesh)
    vertices = np.asarray(mesh.vertices, dtype=np.float32),
    faces = np.asarray(mesh.faces, dtype=np.int32),
    if False:
        import open3d as o3d
        import pdb;pdb.set_trace()
        vert = o3d.utility.Vector3dVector(vertices[0])
        face = o3d.utility.Vector3iVector(faces[0])
            
        meshes = o3d.geometry.TriangleMesh(vert, face)
        print(f'mesh , v {vert}, f {face},', meshes.is_watertight())
        
    return vertices[0], faces[0]

def get_contact_verts_from_obj(obj_fn, class_id, debug=False):
    # add contact verts index. Those furniture are in the canonical axis. 
    # Bed and table have up; chair and sofa have up and back 
    
    ori_mesh = trimesh.load(obj_fn, process=False)
    ori_f = ori_mesh.faces
    ori_v = ori_mesh.vertices
    flip_scene_f = np.stack([ori_f[:, 2], ori_f[:, 1], ori_f[:, 0]], -1)
    mesh = trimesh.Trimesh(ori_v, flip_scene_f, process=False, maintain_order=True)
    verts = mesh.vertices
    height = np.max(verts[:, 1]) - np.min(verts[:, 1])
    valid = verts[:, 1] > (np.min(verts[:, 1]) + height/3)

    normal = mesh.vertex_normals
    up_n = np.array([0, 1, 0])
    front_n = np.array([0, 0, 1])

    angles = np.arccos((normal * up_n).sum(-1)) # -1, 1
    angles = angles *180 / np.pi
    valid_up_idx = angles < 15
    all_valid = valid & valid_up_idx
    
    if class_id in [5, 6, 4]:
        # < 1/2
        all_up_y = verts[all_valid, 1].reshape(-1, 1)
        mean_up_y = np.mean(all_up_y)
        k_all_valid = all_up_y < mean_up_y
        k_all_valid = all_valid.nonzero()[0][k_all_valid[:, 0]]
    elif class_id in [7]:
        all_up_y = verts[all_valid, 1].reshape(-1, 1)
        mean_up_y = np.mean(all_up_y)
        k_all_valid = all_up_y > mean_up_y
        k_all_valid = all_valid.nonzero()[0][k_all_valid[:, 0]]

    if class_id in [5, 6]: # chair, sofa
        # front_angles = np.arcsin(np.linalg.norm(np.cross(normal, front_n), 2, axis=1, ))
        front_angles = np.arccos((normal * front_n).sum(-1))
        front_angles = front_angles *180 / np.pi
        valid_front_idx = front_angles < 15
        all_valid = (valid & valid_up_idx) | (valid & valid_front_idx)

        front_valid = (valid & valid_front_idx)
        all_front_z = verts[front_valid, 2].reshape(-1, 1)
        mean_up_z = np.mean(all_front_z)
        k_front_valid = all_front_z < mean_up_z
        k_front_valid = front_valid.nonzero()[0][k_front_valid[:, 0]]
        k_all_valid = np.concatenate((k_all_valid, k_front_valid))

        
    if debug == True:
        # TODO: change the path
        save_fn = os.path.join('/is/cluster/hyi/workspace/HCI/hdsr/mover_ori_repo_total3D/hdsr/mover_ori_repo/debug/contact_obj', os.path.basename(obj_fn)+'_c.ply')
        c_v = verts[all_valid]
        c_vn = normal[all_valid]
        out_mesh = trimesh.Trimesh(c_v, vertex_normals=c_vn, process=False)
        out_mesh.export(save_fn,vertex_normal=True)

        # save kmeans valid
        save_fn = os.path.join('/is/cluster/hyi/workspace/HCI/hdsr/mover_ori_repo_total3D/hdsr/mover_ori_repo/debug/contact_obj', os.path.basename(obj_fn)+'_c_half.ply')
        c_v = verts[k_all_valid]
        c_vn = normal[k_all_valid]
        out_mesh = trimesh.Trimesh(c_v, vertex_normals=c_vn, process=False)
        out_mesh.export(save_fn,vertex_normal=True)

    return k_all_valid


def format_mesh_without_bboxes(obj_files):
    vtk_objects_list = {}
    file_name_list = {}
    obj_idx_list = {}
    class_idx_list = {}
    for obj_file in sorted(obj_files):
        filename = '.'.join(os.path.basename(obj_file).split('.')[:-1])
        obj_idx = int(filename.split('_')[0])
        class_id = int(filename.split('_')[1].split(' ')[0])
        assert bboxes['class_id'][obj_idx] == class_id

        points, faces = load_meshes_from_obj(obj_file)

        mesh_center = (points.max(0) + points.min(0)) / 2.
        points = points - mesh_center

        mesh_coef = (points.max(0) - points.min(0)) / 2.
        points = points.dot(np.diag(1./mesh_coef)).dot(np.diag(bboxes['coeffs'][obj_idx]))

        # set orientation
        points = points.dot(bboxes['basis'][obj_idx])

        # move to center
        points = points + bboxes['centroid'][obj_idx]

        vtk_objects_list.append((points, faces))
        file_name_list.append(filename)
        obj_idx_list.append(obj_idx)
        class_idx_list.append(class_id)

    return {
        "objs": vtk_objects_list,
        "fns": file_name_list,
        "idxs": obj_idx_list,
        "class_idxs": class_idx_list,
    }

def format_obj_list(obj_files, filter=None):
    if filter is None:
        return sorted(obj_files)
    tmp_new = []
    for i in sorted(obj_files):
        filename = '.'.join(os.path.basename(i).split('.')[:-1])
        class_id = int(filename.split('_')[1].split(' ')[0])
        if class_id == 31:
            continue
        tmp_new.append(i)

    new_obj_files = []
    for i, obj_file in enumerate(sorted(tmp_new)):
        if filter[i]:
            new_obj_files.append(obj_file)
    return new_obj_files

def format_mesh(obj_files, bboxes):
    points_list = []
    points_idx_each_obj = []
    file_name_list = []
    obj_idx_list = []
    class_idx_list = []
    faces_list = []
    faces_idx_each_obj = []
    contact_idx_list = []
    contact_cnt_list = []
    cnt = 0 
    faces_cnt = 0
    contact_cnt = 0
    for obj_idx, obj_file in enumerate(sorted(obj_files)):
        logger.info(f'load {obj_file}')
        filename = '.'.join(os.path.basename(obj_file).split('.')[:-1])
        class_id = int(filename.split('_')[1].split(' ')[0])
        assert bboxes['class_id'][obj_idx] == class_id
        points, faces = load_meshes_from_obj(obj_file)
        # TODO: add contact verts index. Those furniture are in the canonical axis. Bed and table have up; chair and sofa have up and back 
        contact_idx = get_contact_verts_from_obj(obj_file, class_id)
        contact_idx_list.append(contact_idx)
        contact_cnt += len(contact_idx)
        contact_cnt_list.append(contact_cnt)
        mesh_center = (points.max(0) + points.min(0)) / 2.
        points = points - mesh_center

        mesh_coef = (points.max(0) - points.min(0)) / 2.
        points = points.dot(np.diag(1./mesh_coef)).dot(np.diag(bboxes['coeffs'][obj_idx]))

        # set orientation
        points = points.dot(bboxes['basis'][obj_idx])

        # move to center
        points = points + bboxes['centroid'][obj_idx]

        points_list.append(points)
        faces_list.append(faces+cnt)
        
        cnt = cnt+points.shape[0]
        points_idx_each_obj.append(cnt)
        faces_cnt = faces_cnt+faces.shape[0]
        faces_idx_each_obj.append(faces_cnt)

        file_name_list.append(filename)
        obj_idx_list.append(obj_idx)
        class_idx_list.append(class_id)

    return {
        "points": np.vstack(points_list),
        "faces": np.vstack(faces_list),
        "points_idx_each_obj": np.array(points_idx_each_obj),
        "faces_idx_each_obj": np.array(faces_idx_each_obj),
        "fns": file_name_list,
        "idxs": np.array(obj_idx_list),
        "class_idxs": np.array(class_idx_list),
        "contact_idxs": np.hstack(contact_idx_list), 
        "contact_cnt_each_obj": np.array(contact_cnt_list),
    }

def get_bdb_form_from_corners(corners):
    vec_0 = (corners[:, 2, :] - corners[:, 1, :]) / 2.
    vec_1 = (corners[:, 0, :] - corners[:, 4, :]) / 2.
    vec_2 = (corners[:, 1, :] - corners[:, 0, :]) / 2.

    coeffs_0 = np.linalg.norm(vec_0, axis=1)
    coeffs_1 = np.linalg.norm(vec_1, axis=1)
    coeffs_2 = np.linalg.norm(vec_2, axis=1)
    coeffs = np.stack([coeffs_0, coeffs_1, coeffs_2], axis=1)

    centroid = (corners[:, 0, :] + corners[:, 6, :]) / 2.

    basis_0 = np.dot(np.diag(1 / coeffs_0), vec_0)
    basis_1 = np.dot(np.diag(1 / coeffs_1), vec_1)
    basis_2 = np.dot(np.diag(1 / coeffs_2), vec_2)

    basis = np.stack([basis_0, basis_1, basis_2], axis=1)

    return {'basis': basis, 'coeffs': coeffs, 'centroid': centroid}

def format_bbox(box, type, filter=None):
    
    if type == 'prediction':
        boxes = {}
        basis_list = []
        centroid_list = []
        coeff_list = []
        class_list = []

        box_data = box['bdb'][0]
        
        for index in range(len(box_data)):
            if box['class_id'][0][index] == 31:
                continue
            basis = box_data[index]['basis'][0][0]
            centroid = box_data[index]['centroid'][0][0][0]
            coeffs = box_data[index]['coeffs'][0][0][0]
            class_id = box['class_id'][0][index]
            basis_list.append(basis)
            centroid_list.append(centroid)
            coeff_list.append(coeffs)
            class_list.append(class_id)

        boxes['basis'] = np.stack(basis_list, 0)
        boxes['centroid'] = np.stack(centroid_list, 0)
        boxes['coeffs'] = np.stack(coeff_list, 0)
        boxes['class_id'] = np.stack(class_list, 0)

        if filter is not None:
            boxes['basis'] = np.stack(basis_list, 0)[filter]
            boxes['centroid'] = np.stack(centroid_list, 0)[filter]
            boxes['coeffs'] = np.stack(coeff_list, 0)[filter]
            boxes['class_id'] = np.stack(class_list, 0)[filter]
        
    else:
        boxes = get_bdb_form_from_corners(box['bdb3D'])
        boxes['class_id'] = box['size_cls'].tolist()

    return boxes

def format_layout(layout_data):

    layout_bdb = {}

    centroid = (layout_data.max(0) + layout_data.min(0)) / 2.

    vector_z = (layout_data[1] - layout_data[0]) / 2.
    coeff_z = np.linalg.norm(vector_z)
    basis_z = vector_z/coeff_z

    vector_x = (layout_data[2] - layout_data[1]) / 2.
    coeff_x = np.linalg.norm(vector_x)
    basis_x = vector_x/coeff_x

    vector_y = (layout_data[0] - layout_data[4]) / 2.
    coeff_y = np.linalg.norm(vector_y)
    basis_y = vector_y/coeff_y

    basis = np.array([basis_x, basis_y, basis_z])
    coeffs = np.array([coeff_x, coeff_y, coeff_z])

    layout_bdb['coeffs'] = coeffs
    layout_bdb['centroid'] = centroid
    layout_bdb['basis'] = basis

    return layout_bdb