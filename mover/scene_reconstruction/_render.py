import numpy as np
import os
import pyrender
import trimesh
import torch
import neural_renderer as nr

from mover.utils.meshviewer import Mesh
from mover.utils.geometry import (
    combine_verts,
)
from mover.utils.visualize import body_color, define_color_map

RENDER_CHECKBOARD_GP = True

def get_checkerboard_plane(height, rot_mat, \
    plane_width=4, num_boxes=9, center=True):

    pw = plane_width/num_boxes
    white = [220, 220, 220, 255]
    black = [35, 35, 35, 255]

    meshes = []
    for i in range(num_boxes):
        for j in range(num_boxes):
            c = i * pw, j * pw
            ground = trimesh.primitives.Box(
                center=[0, -0.0001, 0],
                extents=[pw, 0.0002, pw]
            )
            if center:
                c = c[0]+(pw/2)-(plane_width/2), c[1]+(pw/2)-(plane_width/2)
            ground.apply_translation([c[0], height, 3+c[1]])
            ground.apply_transform(rot_mat)
            ground.visual.face_colors = black if ((i+j) % 2) == 0 else white
            meshes.append(ground)

    return meshes

def get_checkerboard_ground_pyrender(self,):
    gp_y = self.ground_plane.item()
    rot = self.get_cam_extrin().detach().cpu().numpy()
    
    rot_mat = np.eye(4)
    rot_mat[:3, :3] = rot
    inv_rot = np.linalg.inv(rot_mat)

    gp_meshes = get_checkerboard_plane(gp_y, rot_mat, 
                        plane_width=10, num_boxes=13)
    ground_mesh_pyrender = pyrender.Mesh.from_trimesh(
            gp_meshes,
            smooth=False,
        )
    
    return ground_mesh_pyrender

def get_checkerboard_ground_np(self,):
    gp_y = self.ground_plane.item()
    rot = self.get_cam_extrin().detach().cpu().numpy()
    
    rot_mat = np.eye(4)
    rot_mat[:3, :3] = rot
    inv_rot = np.linalg.inv(rot_mat)

    gp_meshes = get_checkerboard_plane(gp_y, rot_mat, 
                        plane_width=10, num_boxes=13)
    
    all_verts = []
    all_faces = []
    all_colors = []
    cnt = 0
    for one in gp_meshes:
        all_verts.append(one.vertices)
        all_faces.append(one.faces + cnt)
        all_colors.append(one.visual.face_colors)
        cnt += one.vertices.shape[0]
    all_verts = np.concatenate(all_verts)
    all_faces = np.concatenate(all_faces)
    all_colors = np.concatenate(all_colors)
    ground_mesh_np = trimesh.Trimesh(all_verts, all_faces, face_colors=all_colors,  process=False)
    ground_mesh_np_world = trimesh.Trimesh(all_verts, all_faces, include_color=True, process=False)
    ground_mesh_np_world.apply_transform(inv_rot)
    return ground_mesh_np, ground_mesh_np_world



def get_grey_ground_pyrender(self):
    gp_v, gp_f, gp_t = self.get_ground_plane_np()
    gp_v = np.transpose(np.matmul(self.get_cam_extrin().detach().cpu().numpy(), \
            np.transpose(gp_v, (0, 2, 1))), (0, 2, 1))
    gp_mesh = trimesh.Trimesh(gp_v[0], gp_f[0], process=False)
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',#'BLEND',
        baseColorFactor=(66/255.0, 55/255.0, 62/255.0, 1.0),#(0.67, 0.67, 0.67, 0.5),
        wireframe=True)

    gp_mesh = pyrender.Mesh.from_trimesh(gp_mesh, material=material)
    
    return gp_mesh

def get_face_color_np(self, obj_idx=-1):
    color_obj_fc = []
    color_body_fc = []
    for idx, one in enumerate(self.size_cls):
        if idx != obj_idx and obj_idx != -1: # only select obj_idx obj
            continue
        if idx == 0:
            obj_face_num = self.idx_each_object_face[idx]
        else:
            obj_face_num = self.idx_each_object_face[idx] - self.idx_each_object_face[idx-1] 
        color_obj = np.array(define_color_map[one.item()]).reshape(-1, 3).repeat(obj_face_num, 0)

        color_obj_fc.append(color_obj)
    color_body_fc.append(np.array(body_color).reshape(-1, 3).repeat(self.faces_person.shape[0], 0))
    return  np.concatenate(color_obj_fc, 0), np.concatenate(color_body_fc)

def render_with_scene_pyrender(self, scene=None, obj_idx=-1):    
    if not self.cluster:
        if 'PYOPENGL_PLATFORM' in os.environ: # ! Warning: set it in SMPLify-X works, but unset it outside.
            del os.environ["PYOPENGL_PLATFORM"]
    else:
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
    
    # transform graphic2cv coordinate system
    mv = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                            ambient_light=(0.5, 0.5, 0.5))

    camera_pose = np.eye(4)
    camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose
    camera = pyrender.camera.IntrinsicsCamera(fx = self.K_intrin[0][0][0].item(), fy = self.K_intrin[0][1][1].item(), \
        cx=self.K_intrin[0][0][2].item(), cy=self.K_intrin[0][1][2].item())
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    mv.add(camera, pose=camera_pose)
    mv.add(light, pose=camera_pose)

    verts_obj, verts_obj_list = self.get_verts_object(return_all=True)
    if obj_idx != -1:
        verts_obj = verts_obj_list[obj_idx]
        faces_obj = self.faces_list[obj_idx][0]
        faces = torch.cat([faces_obj, self.faces_person+verts_obj.shape[1]])
    else:
        faces = self.faces[0]

    face_body = self.faces_person.shape[0]
    flip_faces_obj = torch.stack([faces[:-face_body, 2], faces[:-face_body, 1], faces[:-face_body, 0]], -1)
    faces = torch.cat((flip_faces_obj, faces[-face_body:, :]), 0)

    if self.verts_person_og.shape[0] != 0:
        verts_combined = combine_verts(
            [verts_obj, self.get_verts_person()]
        )
    else:
        verts_combined = combine_verts(
            [verts_obj]
        )

    verts_combined_np = verts_combined[0].cpu().detach().numpy()

    obj_fc, body_fc = self.get_face_color_np(obj_idx=obj_idx)
    fc_combined_np = np.concatenate([obj_fc, body_fc], 0)
    # add alpha channel
    fc_combined_np = np.hstack([fc_combined_np, np.ones((fc_combined_np.shape[0], 1))])

    s_mesh = Mesh(vertices=verts_combined_np, faces=faces.detach().cpu().numpy(), fc=fc_combined_np)
    s_mesh = pyrender.Mesh.from_trimesh(s_mesh, smooth=False)
    mv.add(s_mesh, 'mesh')
    
    if not RENDER_CHECKBOARD_GP:
        gp_v, gp_f, gp_t = self.get_ground_plane_np()
        gp_v = np.transpose(np.matmul(self.get_cam_extrin().detach().cpu().numpy(), \
                np.transpose(gp_v, (0, 2, 1))), (0, 2, 1))
        gp_mesh = trimesh.Trimesh(gp_v[0], gp_f[0], process=False)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(66/255.0, 55/255.0, 62/255.0, 1.0),
            wireframe=True)
        gp_mesh = pyrender.Mesh.from_trimesh(gp_mesh, material=material)
    else:
        gp_mesh = self.get_checkerboard_ground_pyrender()

    mv.add(gp_mesh, 'mesh')

    if scene is not None:
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='BLEND',
            baseColorFactor=(1.0, 0.0, 0, 0.8),
            wireframe=True)
        static_scene_mesh = pyrender.Mesh.from_trimesh(scene, material)
        mv.add(static_scene_mesh, 'mesh')

    W, H = 640, 360
    r = pyrender.OffscreenRenderer(viewport_width=W,
                                    viewport_height=H,
                                    point_size=1.0)
    color, _ = r.render(mv, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0
    r.delete()
    mv.clear()
    del mv
    mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
    image = color[:, :, :-1]
    return image, mask

def top_render_with_scene_pyrender(self, scene=None, original_size=False, obj_idx=-1):
    import os
    if not self.cluster:
        if 'PYOPENGL_PLATFORM' in os.environ: # ! Warning: set it in SMPLify-X works, but unset it outside.
            del os.environ["PYOPENGL_PLATFORM"]
        import pyrender
    else:
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        import pyrender
    
    # in 3D space under camera coordinates system
    mv = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                            ambient_light=(0.4, 0.4, 0.4))
    camera_pose = np.eye(4)
    camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose

    # change camera pose
    theta = 1.3 # up direction
    yaxis,zaxis = -4.5, 2.0
    x, y = np.cos(theta), np.sin(theta)
    R2 = np.array([[1, 0, 0], [0, x, -y], [0, y, x]])
    t2 = np.array([0, yaxis, zaxis])
    camera_pose[:-1, -1] = t2
    camera_pose[:-1, :-1] = np.matmul(R2, camera_pose[:-1, :-1].T).T

    if original_size:
        camera = pyrender.camera.IntrinsicsCamera(fx = self.K_intrin[0][0][0].item()*4, fy = self.K_intrin[0][1][1].item()*4, \
            cx=self.K_intrin[0][0][2].item()*4, cy=self.K_intrin[0][1][2].item()*4)
    else:
        camera = pyrender.camera.IntrinsicsCamera(fx = self.K_intrin[0][0][0].item(), fy = self.K_intrin[0][1][1].item(), \
            cx=self.K_intrin[0][0][2].item(), cy=self.K_intrin[0][1][2].item())
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    mv.add(camera, pose=camera_pose)
    mv.add(light, pose=camera_pose)

    verts_obj, verts_obj_list = self.get_verts_object(return_all=True)
    if obj_idx != -1:
        verts_obj = verts_obj_list[obj_idx]
        faces_obj = self.faces_list[obj_idx][0]
        faces = torch.cat([faces_obj, self.faces_person+verts_obj.shape[1]])
    else:
        faces = self.faces[0]
    face_body = self.faces_person.shape[0]
    flip_faces_obj = torch.stack([faces[:-face_body, 2], faces[:-face_body, 1], faces[:-face_body, 0]], -1)
    faces = torch.cat((flip_faces_obj, faces[-face_body:, :]), 0)

    if self.verts_person_og.shape[0] != 0:
        verts_combined = combine_verts(
            [verts_obj, self.get_verts_person()]
        )
    else: # * never get into this field.
        verts_combined = combine_verts(
            [verts_obj]
        )
    verts_combined_np = verts_combined[0].cpu().detach().numpy()

    obj_fc, body_fc = self.get_face_color_np(obj_idx=obj_idx)
    fc_combined_np = np.concatenate([obj_fc, body_fc], 0)
    # add alpha channel
    fc_combined_np = np.hstack([fc_combined_np, np.ones((fc_combined_np.shape[0], 1))])

    s_mesh = Mesh(vertices=verts_combined_np, faces=faces.detach().cpu().numpy(), fc=fc_combined_np)
    s_mesh = pyrender.Mesh.from_trimesh(s_mesh, smooth=False)
    mv.add(s_mesh, 'mesh')

    if not RENDER_CHECKBOARD_GP:
        # add camera extrinsic rotation, make it tranparent.
        gp_v, gp_f, gp_t = self.get_ground_plane_np()
        gp_v = np.transpose(np.matmul(self.get_cam_extrin().detach().cpu().numpy(), \
                np.transpose(gp_v, (0, 2, 1))), (0, 2, 1))
        gp_mesh = trimesh.Trimesh(gp_v[0], gp_f[0], process=False)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(66/255.0, 55/255.0, 62/255.0, 1.0),
            wireframe=True)
        gp_mesh = pyrender.Mesh.from_trimesh(gp_mesh, material=material)
    else:
        gp_mesh = self.get_checkerboard_ground_pyrender()

    mv.add(gp_mesh, 'mesh')
    if scene is not None:
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='BLEND',
            baseColorFactor=(1.0, 0.0, 0, 0.8),
            wireframe=True)
        static_scene_mesh = pyrender.Mesh.from_trimesh(scene, material)
        mv.add(static_scene_mesh, 'mesh')

    if original_size:
        W, H = 640*4, 360*4
    else:
        W, H = 640, 360
    r = pyrender.OffscreenRenderer(viewport_width=W,
                                    viewport_height=H,
                                    point_size=1.0)
    color, _ = r.render(mv, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0
    r.delete()
    mv.clear()
    del mv
    image = color[:, :, :-1]
    mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
    return image, mask

def side_render_with_scene_pyrender(self, scene=None,obj_idx=-1):
    # in 3D space under camera coordinates system
    if not self.cluster:
        if 'PYOPENGL_PLATFORM' in os.environ: # ! Warning: set it in SMPLify-X works, but unset it outside.
            del os.environ["PYOPENGL_PLATFORM"]
    else:        
        os.environ['PYOPENGL_PLATFORM'] = 'egl' # ! warning: after use it, it will leads to unuseable MeshViewer in

    mv = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                            ambient_light=(0.5, 0.5, 0.5))
    camera_pose = np.eye(4)
    camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose

    # change camera pose
    theta = 0.0 # up direction
    yaxis,zaxis = -0.5, -1.5 # y < 0: up; z >0: forward
    x, y = np.cos(theta), np.sin(theta)
    R2 = np.array([[1, 0, 0], [0, x, -y], [0, y, x]])
    t2 = np.array([-0.5, yaxis, zaxis])
    camera_pose[:-1, -1] = t2
    camera_pose[:-1, :-1] = np.matmul(R2, camera_pose[:-1, :-1].T).T

    camera = pyrender.camera.IntrinsicsCamera(fx = self.K_intrin[0][0][0].item(), fy = self.K_intrin[0][1][1].item(), \
        cx=self.K_intrin[0][0][2].item(), cy=self.K_intrin[0][1][2].item())
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    mv.add(camera, pose=camera_pose)
    mv.add(light, pose=camera_pose)

    verts_obj, verts_obj_list = self.get_verts_object(return_all=True)
    if obj_idx != -1:
        verts_obj = verts_obj_list[obj_idx]
        faces_obj = self.faces_list[obj_idx][0]
        faces = torch.cat([faces_obj, self.faces_person+verts_obj.shape[1]])
    else:
        faces = self.faces[0]

    face_body = self.faces_person.shape[0]
    flip_faces_obj = torch.stack([faces[:-face_body, 2], faces[:-face_body, 1], faces[:-face_body, 0]], -1)
    faces = torch.cat((flip_faces_obj, faces[-face_body:, :]), 0)

    if self.verts_person_og.shape[0] != 0:
        verts_combined = combine_verts(
            [verts_obj, self.get_verts_person()]
        )
    else:
        verts_combined = combine_verts(
            [verts_obj]
        )

    verts_combined_np = verts_combined[0].cpu().detach().numpy()

    obj_fc, body_fc = self.get_face_color_np(obj_idx=obj_idx)
    fc_combined_np = np.concatenate([obj_fc, body_fc], 0)
    fc_combined_np = np.hstack([fc_combined_np, np.ones((fc_combined_np.shape[0], 1))])

    s_mesh = Mesh(vertices=verts_combined_np, faces=faces.detach().cpu().numpy(), fc=fc_combined_np)
    s_mesh = pyrender.Mesh.from_trimesh(s_mesh, smooth=False)
    mv.add(s_mesh, 'mesh')

    if not RENDER_CHECKBOARD_GP:
        gp_v, gp_f, gp_t = self.get_ground_plane_np()
        gp_v = np.transpose(np.matmul(self.get_cam_extrin().detach().cpu().numpy(), \
                np.transpose(gp_v, (0, 2, 1))), (0, 2, 1))

        gp_mesh = trimesh.Trimesh(gp_v[0], gp_f[0], process=False)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='BLEND',
            baseColorFactor=(0.67, 0.67, 0.67, 0.5),
            wireframe=True)
        gp_mesh = pyrender.Mesh.from_trimesh(gp_mesh, material=material)
    else:
        gp_mesh = self.get_checkerboard_ground_pyrender()

    mv.add(gp_mesh, 'mesh')

    if scene is not None:
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='BLEND',
            baseColorFactor=(1.0, 0.0, 0, 0.8),
            wireframe=True)
        static_scene_mesh = pyrender.Mesh.from_trimesh(scene, material)
        mv.add(static_scene_mesh, 'mesh')

    W, H = 640, 360
    r = pyrender.OffscreenRenderer(viewport_width=W,
                                    viewport_height=H,
                                    point_size=1.0)
    color, _ = r.render(mv, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0
    r.delete()
    mv.clear()
    del mv
    image = color[:, :, :-1]
    mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
    return image, mask

def right_side_render_with_scene_pyrender(self, scene=None,obj_idx=-1):
    # in 3D space under camera coordinates system
    if not self.cluster:
        if 'PYOPENGL_PLATFORM' in os.environ: # ! Warning: set it in SMPLify-X works, but unset it outside.
            del os.environ["PYOPENGL_PLATFORM"]
    else:
        os.environ['PYOPENGL_PLATFORM'] = 'egl' # ! warning: after use it, it will leads to unuseable MeshViewer in

    mv = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                            ambient_light=(0.5, 0.5, 0.5))

    camera_pose = np.eye(4)
    camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose

    # change camera pose
    theta = 0.0 # up direction
    yaxis,zaxis = -0.5, -1.5 # y < 0: up; z >0: forward
    xaxis = 0.5 # x > 0: right
    x, y = np.cos(theta), np.sin(theta)
    R2 = np.array([[1, 0, 0], [0, x, -y], [0, y, x]])
    t2 = np.array([0.5, yaxis, zaxis])
    camera_pose[:-1, -1] = t2
    camera_pose[:-1, :-1] = np.matmul(R2, camera_pose[:-1, :-1].T).T

    camera = pyrender.camera.IntrinsicsCamera(fx = self.K_intrin[0][0][0].item(), fy = self.K_intrin[0][1][1].item(), \
        cx=self.K_intrin[0][0][2].item(), cy=self.K_intrin[0][1][2].item())
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    mv.add(camera, pose=camera_pose)
    mv.add(light, pose=camera_pose)

    verts_obj, verts_obj_list = self.get_verts_object(return_all=True)
    if obj_idx != -1:
        verts_obj = verts_obj_list[obj_idx]
        faces_obj = self.faces_list[obj_idx][0]
        import torch
        faces = torch.cat([faces_obj, self.faces_person+verts_obj.shape[1]])
    else:
        faces = self.faces[0]
    import torch
    face_body = self.faces_person.shape[0]
    flip_faces_obj = torch.stack([faces[:-face_body, 2], faces[:-face_body, 1], faces[:-face_body, 0]], -1)
    faces = torch.cat((flip_faces_obj, faces[-face_body:, :]), 0)

    if self.verts_person_og.shape[0] != 0:
        verts_combined = combine_verts(
            [verts_obj, self.get_verts_person()]
        )
    else: # * never get into this field.
        verts_combined = combine_verts(
            [verts_obj]
        )

    verts_combined_np = verts_combined[0].cpu().detach().numpy()

    obj_fc, body_fc = self.get_face_color_np(obj_idx=obj_idx)
    fc_combined_np = np.concatenate([obj_fc, body_fc], 0)
    # add alpha channel
    fc_combined_np = np.hstack([fc_combined_np, np.ones((fc_combined_np.shape[0], 1))])

    s_mesh = Mesh(vertices=verts_combined_np, faces=faces.detach().cpu().numpy(), fc=fc_combined_np)
    s_mesh = pyrender.Mesh.from_trimesh(s_mesh, smooth=False)
    mv.add(s_mesh, 'mesh')

    if not RENDER_CHECKBOARD_GP:
        # add camera extrinsic rotation, make it tranparent.
        gp_v, gp_f, gp_t = self.get_ground_plane_np()
        gp_v = np.transpose(np.matmul(self.get_cam_extrin().detach().cpu().numpy(), \
                np.transpose(gp_v, (0, 2, 1))), (0, 2, 1))

        gp_mesh = trimesh.Trimesh(gp_v[0], gp_f[0], process=False)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='BLEND',
            baseColorFactor=(0.67, 0.67, 0.67, 0.5),
            wireframe=True)
        gp_mesh = pyrender.Mesh.from_trimesh(gp_mesh, material=material)
    else:
        gp_mesh = self.get_checkerboard_ground_pyrender()

    mv.add(gp_mesh, 'mesh')
    if scene is not None:
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='BLEND',
            baseColorFactor=(1.0, 0.0, 0, 0.8),
            wireframe=True)
        static_scene_mesh = pyrender.Mesh.from_trimesh(scene, material)
        mv.add(static_scene_mesh, 'mesh')

    W, H = 640, 360
    r = pyrender.OffscreenRenderer(viewport_width=W,
                                    viewport_height=H,
                                    point_size=1.0)
    color, _ = r.render(mv, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0
    r.delete()
    mv.clear()
    del mv
    image = color[:, :, :-1]
    mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
    return image, mask

### ！currently not use. ###
def render(self):
    if self.verts_person_og.shape[0] != 0:
        verts_combined = combine_verts(
            [self.get_verts_object(), self.get_verts_person()]
        )
    else:
        verts_combined = combine_verts(
            [self.get_verts_object()]
        )

    # add camera extrinsic rotation
    gp_v, gp_f, gp_t = self.get_ground_plane_mesh()
    gp_f = gp_f + verts_combined.shape[1]

    gp_v = torch.transpose(torch.matmul(self.get_cam_extrin(), \
            torch.transpose(gp_v, 2, 1)), 2, 1)

    image, _, mask = self.renderer.render(vertices=torch.cat((verts_combined, gp_v), 1), 
                                    faces=torch.cat((self.faces, gp_f), 1), 
                                    textures=torch.cat((self.textures, gp_t), 1))
    image = np.clip(image[0].detach().cpu().numpy().transpose(1, 2, 0), 0, 1)
    mask = mask[0].detach().cpu().numpy().astype(bool)

    return image, mask

def render_pyrender(self):
    mv = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                            ambient_light=(0.5, 0.5, 0.5))
    camera_pose = np.eye(4)
    camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose
    camera = pyrender.camera.IntrinsicsCamera(fx = self.K_intrin[0][0][0], fy = self.K_intrin[0][1][1], \
        cx=self.K_intrin[0][0][2], cy=self.K_intrin[0][1][2])
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    mv.add(camera, pose=camera_pose)
    mv.add(light, pose=camera_pose)

    if self.verts_person_og.shape[0] != 0:
        verts_combined = combine_verts(
            [self.get_verts_object(), self.get_verts_person()]
        )
    else:
        verts_combined = combine_verts(
            [self.get_verts_object()]
        )
    obj_fc, body_fc = self.get_face_color_np()
    fc_combined_np = np.concatenate([obj_fc, body_fc], 0)
    # add alpha channel
    verts_combined_np = verts_combined[0].cpu().detach().numpy()
    
    s_mesh = trimesh.Trimesh(verts_combined_np, self.faces[0].detach().cpu().numpy(), process=False)
    s_mesh.visual.face_colors = fc_combined_np
    s_mesh = pyrender.Mesh.from_trimesh(s_mesh, smooth=False)
    mv.add(s_mesh, 'mesh')

    # add camera extrinsic rotation, make it tranparent.
    gp_v, gp_f, gp_t = self.get_ground_plane_np()
    gp_v = torch.transpose(torch.matmul(self.get_cam_extrin(), \
            torch.transpose(gp_v, 2, 1)), 2, 1)

    gp_mesh = trimesh.Trimesh(gp_v[0], gp_f[0], process=False)
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='BLEND',
        baseColorFactor=(0.67, 0.67, 0.67, 0.5),
        wireframe=True)
    gp_mesh = pyrender.Mesh.from_trimesh(gp_mesh, material=material)
    mv.add(gp_mesh, 'mesh')

    W, H = 640, 360
    r = pyrender.OffscreenRenderer(viewport_width=W,
                                    viewport_height=H,
                                    point_size=1.0)
    color, _ = r.render(mv, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0

    mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
    image = color[:, :, :-1]
    return image, mask

# ! input with Trimesh of scene in camera coordinates
def render_with_scene(self, scene):
    assert scene is not None
    scene_verts = torch.from_numpy(np.array(scene.vertices)).type(torch.float32).cuda()
    scene_f = torch.from_numpy(np.array(scene.faces)).type(torch.int32).cuda()
    scene_color = torch.ones(scene_f.shape).type_as(self.textures_person) * 0.7
    
    if self.verts_person_og.shape[0] != 0:
        verts_combined = combine_verts(
            [scene_verts, self.get_verts_person()]
        )
        face_combined = torch.cat((scene_f, scene_verts.shape[0]+self.faces_person), 0).unsqueeze(0)
        texture_combined = torch.cat((scene_color.reshape(1, -1, 1,1,1,3), self.textures_person), 1) 
    else:
        verts_combined = combine_verts(
            [scene_verts]
        )
        face_combined = torch.cat((scene_f))
        texture_combined = torch.cat((scene_color.reshape(1, -1, 1,1, 1,3)), 1)

    image, _, mask = self.renderer.render(vertices=verts_combined, 
                                    faces=face_combined, 
                                    textures=texture_combined)

    image = np.clip(image[0].detach().cpu().numpy().transpose(1, 2, 0), 0, 1)
    mask = mask[0].detach().cpu().numpy().astype(bool)
    return image, mask

######################## not used yet!!!
def top_render_with_scene(self, scene):
    # in 3D space under camera coordinates system

    assert scene is not None
    scene_verts = torch.from_numpy(np.array(scene.vertices)).type(torch.float32).cuda()
    scene_f = torch.from_numpy(np.array(scene.faces)).type(torch.int32).cuda()
    scene_color = torch.ones(scene_f.shape).type_as(self.textures_person) * 0.7
    if self.verts_person_og.shape[0] != 0:
        verts_combined = combine_verts(
            [scene_verts, self.get_verts_person()]
        )
        face_combined = torch.cat((scene_f, scene_verts.shape[0]+self.faces_person), 0).unsqueeze(0)
        texture_combined = torch.cat((scene_color.reshape(1, -1, 1,1,1,3), self.textures_person), 1) 
    else:
        verts_combined = combine_verts(
            [scene_verts]
        )
        face_combined = torch.cat((scene_f))
        texture_combined = torch.cat((scene_color.reshape(1, -1, 1,1, 1,3)), 1)

    theta = 1.3
    d = 2.5
    x, y = np.cos(theta), np.sin(theta)
    mx, my, mz = verts_combined.mean(dim=(0, 1)).detach().cpu().numpy()
    K = torch.cuda.FloatTensor([[[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]])
    R2 = torch.cuda.FloatTensor([[[1, 0, 0], [0, x, -y], [0, y, x]]])
    t2 = torch.cuda.FloatTensor([0, d, 7]) # x, y, z (z: height)

    top_renderer = nr.renderer.Renderer(
        image_size=IMAGE_SIZE, K=K, R=R2, t=t2, orig_size=1
    )
    top_renderer.background_color = [1, 1, 1]
    top_renderer.light_direction = [1, 0.5, 1]
    top_renderer.light_intensity_direction = 0.3
    top_renderer.light_intensity_ambient = 0.5
    top_renderer.background_color = [1, 1, 1]
    top_down, _, mask = top_renderer.render(vertices=verts_combined, 
                                    faces=face_combined, 
                                    textures=texture_combined)
    top_down = np.clip(top_down[0].detach().cpu().numpy().transpose(1, 2, 0), 0, 1)

    return top_down

def top_render(self):
    # in 3D space under camera coordinates system

    if self.verts_person_og.shape[0] != 0:
        verts_combined = combine_verts(
            [self.get_verts_object(), self.get_verts_person()]
        )
    else:
        verts_combined = combine_verts(
            [self.get_verts_object()]
        )
    verts_combined = torch.transpose(torch.matmul(torch.transpose(self.get_cam_extrin(), 2, 1), torch.transpose(verts_combined, 2, 1)), 2, 1)

    gp_v, gp_f, gp_t = self.get_ground_plane_mesh()
    gp_f = gp_f + verts_combined.shape[1]

    theta = 1.3
    d = 2.5
    x, y = np.cos(theta), np.sin(theta)
    mx, my, mz = verts_combined.mean(dim=(0, 1)).detach().cpu().numpy()
    K = torch.cuda.FloatTensor([[[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]])
    R2 = torch.cuda.FloatTensor([[[1, 0, 0], [0, x, -y], [0, y, x]]])
    t2 = torch.cuda.FloatTensor([0, d, 7]) # x, y, z (z: height)
    top_renderer = nr.renderer.Renderer(
        image_size=IMAGE_SIZE, K=K, R=R2, t=t2, orig_size=1
    )
    top_renderer.background_color = [1, 1, 1]
    top_renderer.light_direction = [1, 0.5, 1]
    top_renderer.light_intensity_direction = 0.3
    top_renderer.light_intensity_ambient = 0.5
    top_renderer.background_color = [1, 1, 1]
    top_down, _, mask = top_renderer.render(vertices=torch.cat((verts_combined, gp_v), 1), 
                                    faces=torch.cat((self.faces, gp_f), 1), 
                                    textures=torch.cat((self.textures, gp_t), 1))
    top_down = np.clip(top_down[0].detach().cpu().numpy().transpose(1, 2, 0), 0, 1)

    return top_down

def side_render(self, tx=0, rz=0):
    # in 3D space under camera coordinates system
    if self.verts_person_og.shape[0] != 0:
        verts_combined = combine_verts(
            [self.get_verts_object(), self.get_verts_person()]
        )
    else:
        verts_combined = combine_verts(
            [self.get_verts_object()]
        )
    # reverse rotation on camera coordinates: K-1 
    verts_combined = torch.transpose(torch.matmul(torch.transpose(self.get_cam_extrin(), 2, 1), torch.transpose(verts_combined, 2, 1)), 2, 1)

    gp_v, gp_f, gp_t = self.get_ground_plane_mesh()
    gp_f = gp_f + verts_combined.shape[1]
    
    theta = rz
    d = 3
    x, y = np.cos(theta), np.sin(theta)
    mx, my, mz = verts_combined.mean(dim=(0, 1)).detach().cpu().numpy()
    K = self.renderer.K
    R2 = torch.cuda.FloatTensor([[[x, 0, -y], [0, 1, 0], [y, 0, x]]])
    t2 = torch.cuda.FloatTensor([tx, 0, 1])
    top_renderer = nr.renderer.Renderer(
        image_size=IMAGE_SIZE, K=K, R=R2, t=t2, orig_size=IMAGE_SIZE
    )
    top_renderer.background_color = [1, 1, 1]
    top_renderer.light_direction = [1, 0.5, 1]
    top_renderer.light_intensity_direction = 0.3
    top_renderer.light_intensity_ambient = 0.5
    top_renderer.background_color = [1, 1, 1]
    top_down, _, mask = top_renderer.render(vertices=torch.cat((verts_combined, gp_v), 1), 
                                    faces=torch.cat((self.faces, gp_f), 1), 
                                    textures=torch.cat((self.textures, gp_t), 1))
    top_down = np.clip(top_down[0].detach().cpu().numpy().transpose(1, 2, 0), 0, 1)

    return top_down

def interactive_render(self):
    # import pdb;pdb.set_trace()
    camera_pose = np.eye(4)
    mv = MeshViewer(offscreen=False, cam_inc=self.K_intrin[0], \
        cam_ext=camera_pose)

    if self.verts_person_og.shape[0] != 0:
        verts_combined = combine_verts(
            [self.get_verts_object(), self.get_verts_person()]
        )
    else:
        verts_combined = combine_verts(
            [self.get_verts_object()]
        )

    verts_combined_np = verts_combined[0].cpu().detach().numpy()
    verts_combined_np = np.matmul(self.get_cam_extrin().detach().cpu().numpy()[0], verts_combined_np.T).T
    verts_combined_opengl = np.stack((verts_combined_np[:, 0], -verts_combined_np[:, 1], \
                    -verts_combined_np[:, 2]), 1)
    s_mesh = Mesh(vertices=verts_combined_opengl, faces=self.faces[0].cpu().detach().numpy()) #, texture=self.textures
    line = pyrender.Primitive(positions=np.array([[0, 0],[0,1],[1, 1], [1, 0]]), mode=0)
    line_mesh = pyrender.Mesh([line])
    mv.add(line_mesh)
    mv.set_static_meshes([s_mesh])

def interactive_op3d_render(self, image):
    import open3d as o3d
    if self.verts_person_og.shape[0] != 0:
        verts_combined = combine_verts(
            [self.get_verts_object(), self.get_verts_person()]
        )
    else:
        verts_combined = combine_verts(
            [self.get_verts_object()]
        )

    # add ground plane 
    gp_v, gp_f, gp_t = self.get_ground_plane_mesh()
    gp_f = gp_f + verts_combined.shape[1]
    verts_combined = torch.cat((verts_combined, gp_v), 1)
    # use OpenGL camera
    verts_combined_np = verts_combined[0].cpu().detach().numpy()
    faces_np = torch.cat((self.faces, gp_f), 1)[0].cpu().detach().numpy()
    K_intrin_np = self.get_cam_extrin().detach().cpu().numpy()[0]
    verts_combined_np = np.matmul(K_intrin_np, verts_combined_np.T).T
    verts_combined_opengl = np.stack((verts_combined_np[:, 0], -verts_combined_np[:, 1], \
                    -verts_combined_np[:, 2]), 1)
    
    verts = o3d.utility.Vector3dVector(verts_combined_opengl)
    faces = o3d.utility.Vector3iVector(faces_np)
    colors = o3d.utility.Vector3dVector(torch.cat((self.textures, gp_t), 1).squeeze().detach().cpu().numpy())
    meshes = o3d.geometry.TriangleMesh(verts, faces)
    meshes.vertex_colors = colors
    meshes.paint_uniform_color([1, 0.706, 0])

    # camera 
    camera = o3d.camera.PinholeCameraParameters()
    cam_intri = o3d.camera.PinholeCameraIntrinsic()
    cam_intri.intrinsic_matrix = K_intrin_np
    cam_intri.width = 640
    cam_intri.height = 360
    camera.intrinsic = cam_intri
    ext_4 = np.eye(4)
    camera.extrinsic = np.linalg.inv(ext_4)

    # image to background point cloud
    resize_img = cv2.resize(image,(640, 320))
    img = o3d.geometry.Image(resize_img.astype(np.uint8))
    depth_img = o3d.geometry.Image((np.ones((320, 640, 1)) * K_intrin_np[0][0]).astype(np.uint16))
    rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(img, depth_img, convert_rgb_to_intensity=False)
    
    pcl = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, cam_intri)
    rot_mat = o3d.geometry.get_rotation_matrix_from_xyz(np.array([[-np.pi, 0, 0]]).T)
    pcl_rot = pcl.rotate(rot_mat, np.zeros((3,1)))

    # draw original point
    print("Let's draw a box using o3d.geometry.LineSet.")
    ori_point_set = [[0, 0, 0], 
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
    ]
    lines = [[0, 1],
            [0, 2],
            [0, 3],
    ]
    colors = []
    for i in range(len(lines)):
        if i %3 == 0:
            colors.append([1, 0, 0])
        elif i % 3 == 1:
            colors.append([0, 1, 0])
        elif i % 3 == 2:
            colors.append([0, 0, 1])

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(ori_point_set),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    vis = o3d.visualization.Visualizer()
    vis.create_window("Human Object")
    vis.add_geometry(meshes)
    vis.add_geometry(line_set)
    vis.add_geometry(pcl_rot)
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(camera)

    vis.run()
    # TODO: add interactive keyboard
    vis.destroy_window()