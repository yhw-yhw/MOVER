import os

from .vis_utils import draw_skeleton, visualize_joint_error, visualize_joint_uncertainty, \
    visualize_heatmaps, visualize_segm_masks

os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['EGL_DEVICE_ID'] = os.environ['GPU_DEVICE_ORDINAL'].split(',')[0] \
    if 'GPU_DEVICE_ORDINAL' in os.environ.keys() else '0'

import torch
import trimesh
import pyrender
import numpy as np
from torchvision.utils import make_grid
from .vis_utils import get_colors


class Renderer:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """
    def __init__(self, focal_length=5000, img_res=224, faces=None, mesh_color='pink'):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_res,
                                       viewport_height=img_res,
                                       point_size=1.0)
        self.focal_length = focal_length
        self.img_res = img_res
        self.camera_center = [img_res // 2, img_res // 2]
        self.faces = faces
        self.mesh_color = get_colors()[mesh_color]

    def _set_focal_length(self, focal_length):
        self.focal_length = focal_length

    def _set_mesh_color(self, mesh_color):
        self.mesh_color = get_colors()[mesh_color]

    def visualize_tb(
            self,
            vertices,
            camera_translation,
            images,
            kp_2d=None,
            heatmaps=None,
            segm_masks=None,
            skeleton_type='smpl',
            nb_max_img=8,
            sideview=False,
            vertex_colors=None,
            joint_labels=None,
            joint_uncertainty=None,
            alpha=1.0,
            camera_rotation=None,
            multi_sideview=False,
            mesh_filename=None,
    ):
        vertices = vertices.cpu().numpy()
        camera_translation = camera_translation.cpu().numpy()
        if camera_rotation is not None:
            camera_rotation = camera_rotation.cpu().numpy()

        images = images.cpu()
        images_np = np.transpose(images.numpy(), (0,2,3,1))

        if kp_2d is not None:
            kp_2d = kp_2d.cpu().numpy()

        rend_imgs = []
        nb_max_img = min(nb_max_img, vertices.shape[0])
        num_sideview = 0
        for i in range(nb_max_img):
            rend_img = torch.from_numpy(
                np.transpose(self.__call__(
                    vertices[i],
                    camera_translation[i],
                    images_np[i],
                    vertex_colors=None if vertex_colors is None else vertex_colors[i],
                    joint_labels=None if joint_labels is None else joint_labels[i],
                    alpha=alpha,
                    camera_rotation=camera_rotation,
                    mesh_filename=mesh_filename,
                ), (2,0,1))
            ).float()

            rend_imgs.append(images[i])
            rend_imgs.append(rend_img)

            if kp_2d is not None:
                kp_img = draw_skeleton(images_np[i].copy(), kp_2d=kp_2d[i],
                                       dataset=skeleton_type, res=self.camera_center[0]*2)
                kp_img = torch.from_numpy(np.transpose(kp_img, (2,0,1))).float()
                rend_imgs.append(kp_img)

            if heatmaps is not None:
                hm_img = visualize_heatmaps(images_np[i].copy(), heatmaps=heatmaps[i], alpha=0.4)
                hm_img = torch.from_numpy(np.transpose(hm_img, (2, 0, 1))).float()
                rend_imgs.append(hm_img)

            if segm_masks is not None:
                mask_img = visualize_segm_masks(images_np[i].copy(), mask=segm_masks[i], alpha=0.8)
                mask_img = torch.from_numpy(np.transpose(mask_img, (2, 0, 1))).float()
                rend_imgs.append(mask_img)

            if sideview:
                if multi_sideview:
                    for angle in [270, 180, 90]:
                        side_img = torch.from_numpy(
                            np.transpose(
                                self.__call__(
                                    vertices[i],
                                    camera_translation[i],
                                    np.ones_like(images_np[i]),
                                    vertex_colors=None if vertex_colors is None else vertex_colors[i],
                                    joint_labels=None if joint_labels is None else joint_labels[i],
                                    alpha=alpha,
                                    sideview=True,
                                    camera_rotation=camera_rotation,
                                    sideview_angle=angle,
                                    mesh_filename=mesh_filename.replace('.obj', f'_{angle:03d}_rot.obj')
                                                  if mesh_filename else None
                                ),
                                (2, 0, 1)
                            )
                        ).float()
                        rend_imgs.append(side_img)
                        num_sideview += 1
                else:
                    side_img = torch.from_numpy(
                        np.transpose(
                            self.__call__(
                                vertices[i],
                                camera_translation[i],
                                np.ones_like(images_np[i]),
                                vertex_colors=None if vertex_colors is None else vertex_colors[i],
                                joint_labels=None if joint_labels is None else joint_labels[i],
                                alpha=alpha,
                                sideview=True,
                                camera_rotation=camera_rotation,
                                mesh_filename=mesh_filename.replace('.obj', f'_270_rot.obj')
                                if mesh_filename else None
                            ),
                            (2,0,1)
                        )
                    ).float()
                    rend_imgs.append(side_img)
                    num_sideview += 1

            if joint_labels is not None:
                error_image = visualize_joint_error(joint_labels[i], res=self.camera_center[0]*2)
                error_image = torch.from_numpy(np.transpose(error_image, (2, 0, 1))).float()
                rend_imgs.append(error_image)

            if joint_uncertainty is not None:
                error_image = visualize_joint_uncertainty(joint_uncertainty[i], res=self.camera_center[0]*2)
                error_image = torch.from_numpy(np.transpose(error_image, (2, 0, 1))).float()
                rend_imgs.append(error_image)

        nrow = 2
        if kp_2d is not None: nrow += 1
        if sideview: nrow += num_sideview
        if joint_labels is not None: nrow += 1
        if joint_uncertainty is not None: nrow += 1
        if heatmaps is not None: nrow += 1
        if segm_masks is not None: nrow += 1

        # nrow = len(rend_imgs)

        rend_imgs = make_grid(rend_imgs, nrow=nrow)
        return rend_imgs

    def __call__(
            self, vertices, camera_translation, image, vertex_colors=None,
            sideview=False, joint_labels=None, alpha=1.0, camera_rotation=None,
            sideview_angle=270, mesh_filename=None, mesh_inp=None,
    ):

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(self.mesh_color[0] / 255., self.mesh_color[1] / 255., self.mesh_color[2] / 255., alpha))

        camera_translation[0] *= -1.

        if mesh_inp:
            mesh = mesh_inp
        else:
            if vertex_colors is not None:
                mesh = trimesh.Trimesh(vertices, self.faces, vertex_colors=vertex_colors, process=False)
            else:
                mesh = trimesh.Trimesh(vertices, self.faces, vertex_colors=vertex_colors, process=False)

        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)

        if sideview:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(sideview_angle), [0, 1, 0])
            mesh.apply_transform(rot)

        if mesh_filename:
            mesh.export(mesh_filename)
            if not mesh_filename.endswith('_rot.obj'):
                np.save(mesh_filename.replace('.obj', '.npy'), camera_translation)

        if vertex_colors is not None:
            mesh = pyrender.Mesh.from_trimesh(mesh)
        elif mesh_inp is not None:
            mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        else:
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        if camera_rotation is not None:
            camera_pose[:3, :3] = camera_rotation

        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(camera, pose=camera_pose)


        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        # if joint_labels is not None:
        #     for joint, err in joint_labels.items():
        #         add_joints(scene, joints=joint, radius=err)

        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:,:,None]
        output_img = (color[:, :, :3] * valid_mask +
                  (1 - valid_mask) * image)
        return output_img


def add_joints(scene, joints, radius=0.005, color=[0.1, 0.1, 0.9, 1.0]):
    sm = trimesh.creation.uv_sphere(radius=radius)
    sm.visual.vertex_colors = color
    tfs = np.tile(np.eye(4), (1, 1, 1))
    tfs[:, :3, 3] = joints
    joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
    scene.add(joints_pcl)