import torch
import pytorch3d
import trimesh
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.renderer.mesh import rasterize_meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    FoVPerspectiveCameras,
    PointLights, 
    DirectionalLights, 
    RasterizationSettings,
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesVertex,
)
from pytorch3d.renderer.mesh.renderer import MeshRendererWithFragments

import numpy as np
import pickle
import glob
import os
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

# from pytorch3d.renderer.mesh.texture import Textures
def dict2obj(d):
    # if isinstance(d, list):
    #     d = [dict2obj(x) for x in d]
    if not isinstance(d, dict):
        return d
    class C(object):
        pass
    o = C()
    for k in d:
        o.__dict__[k] = dict2obj(d[k])
    return o

def init_pytorch3d_render(self, K_intrin, image_size, device=torch.device('cuda:0')):
    R = torch.cuda.FloatTensor([[[-1, 0, 0], [0, -1, 0], [0, 0, 1]]])
    T = torch.zeros(1, 3).cuda()
    # K_intrin = torch.cuda.FloatTensor([[[1.06053174e+03 / 3, 0., 9.51299927e+02 / 3,0],
    # [0., 1.06038562e+03 / 3, 5.36770386e+02 / 3,0],
    # [0., 0., 0.0, 1.0], 
    # [0., 0., 1., 0.]]])
    # image_size: H,W
    K_intrin_pt3d = torch.zeros((1,4,4)).cuda()
    K_intrin_pt3d[0, 0, 0] = K_intrin[0, 0, 0]
    K_intrin_pt3d[0, 0, 2] = K_intrin[0, 0, 2]
    K_intrin_pt3d[0, 1, 1] = K_intrin[0, 1, 1]
    K_intrin_pt3d[0, 1, 2] = K_intrin[0, 1, 2]
    K_intrin_pt3d[0, 2, 3] = 1.0
    K_intrin_pt3d[0, 3, 2] = 1.0

    lights = PointLights(device=device, location=[[0.0, 0.0, 0.0]])

    cameras = PerspectiveCameras(device=device, R=R, T=T, K=K_intrin_pt3d,image_size=image_size, in_ndc=False)

    raster_settings = RasterizationSettings(
        image_size=image_size[0], 
        blur_radius=0.0, 
        faces_per_pixel=10, 
    )
    
    self.renderer_pytorch3d = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )

def get_depth_map_pytorch3d(self, verts, faces):
    texture = torch.cuda.FloatTensor([0.9, 0.7, 0.7]).repeat(1, verts.shape[1], 1)
    texture_pt3d = TexturesVertex(verts_features=texture)
    meshes_screen = Meshes(verts=verts, faces=faces, textures=texture_pt3d)

    # import pdb;pdb.set_trace()
    images, fragments = self.renderer_pytorch3d(meshes_screen)

    pix_to_face, zbuf, bary_coords, dists = fragments
    vismask = (pix_to_face[..., 0] > -1).bool()
    nearest_depth_map = zbuf.clone()
    farrest_depth_map = zbuf.clone()
    # import pdb;pdb.set_trace()
    farrest_depth_map = farrest_depth_map.max(-1)[0]
    farrest_depth_map[~vismask] = 0.0
    # farrest_depth_map = torch.flipud(farrest_depth_map[0])[None]

    nearest_depth_map[zbuf==-1] = 10.0
    nearest_depth_map = nearest_depth_map.min(-1)[0]
    nearest_depth_map[~vismask] = 0.0

    return vismask, nearest_depth_map, farrest_depth_map


def draw_color_map(depth, back_depth , dmin, dmax, save_path=None):
    # depth = Image.from_array()
    ori_img = f'/is/cluster/work/hyi/results/HDSR/PROX_qualitative_all/MPH112_00150_01/Total3D/img.jpg'
    size = (640, 360)
    b_img = Image.open(ori_img).resize(size)
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    # import pdb;pdb.set_trace()
    # fig, ax = plt.subplots()
    ax1.imshow(b_img)
    ax2.imshow(b_img)

    # import pdb;pdb.set_trace()
    depth = depth.transpose(1,2,0)
    im = ax1.imshow(depth[:360, :],  cmap='gist_rainbow', alpha=.8, interpolation='nearest',vmin=dmin-1e-2, vmax=dmax+1e-2)
    
    name='Front Depth'
    ax1.title.set_text(name)
    # ax1.set_xlabel('X Pixel')
    ax1.set_ylabel('Y Pixel')
    ax1.set_axis_off()

    back_depth = back_depth.transpose(1,2,0)
    im = ax2.imshow(back_depth[:360, :],  cmap='gist_rainbow', alpha=.8, interpolation='nearest',vmin=dmin-1e-2, vmax=dmax+1e-2)
    
    name='Back Depth'
    ax2.title.set_text(name)
    ax2.set_xlabel('X Pixel')
    # ax2.set_ylabel('Y Pixel')
    ax2.set_axis_off()

    fig.colorbar(im, ax=[ax1, ax2], label='Distance to Camera')
    plt.savefig(save_path)
    # plt.show()
    plt.close()

if __name__ == '__main__':
    obj_file = '/is/cluster/work/hyi/results/HDSR/PROX_qualitative_all/MPH112_00150_01/2022Final_fixTableBug_run2/pid0_s3kind13033_ReinitOrienTrueTotal3DRealFalseMSEContactFalseReinSPTrueDEGENETrue0.01AddChairRENEW_OPTFalsehigherChairTruehigherScale30_New_constraintChairTrue0.6/obj_-1/model_scene_1_lr0.002_end/f000.obj'
    # image_size = (1080,1920)
    image_size = (int(1080/3), int(1920/3))

    # import pdb;pdb.set_trace()
    verts, faces, aux = load_obj(obj_file)
    verts = verts[None, ...]
    faces = faces.verts_idx[None,...]
    verts.requires_grad = True
    texture = torch.cuda.FloatTensor([0.9, 0.7, 0.7]).repeat(1, verts.shape[1], 1)
    texture_pt3d = TexturesVertex(verts_features=texture)

    # Setup
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    lights = PointLights(device=device, location=[[0.0, 0.0, 0.0]])


    R = torch.cuda.FloatTensor([[[-1, 0, 0], [0, -1, 0], [0, 0, 1]]])
    T = torch.zeros(1, 3).cuda()
    K_intrin = torch.cuda.FloatTensor([[[1.06053174e+03 / 3, 0., 9.51299927e+02 / 3,0],
    [0., 1.06038562e+03 / 3, 5.36770386e+02 / 3,0],
    [0., 0., 0.0, 1.0], 
    [0., 0., 1., 0.]]])
    # import pdb;pdb.set_trace()
    cameras = PerspectiveCameras(device=device, R=R, T=T, K=K_intrin,image_size=[image_size], in_ndc=False)

    # import pdb;pdb.set_trace()

    if False:
        raster_settings = {
                'image_size': image_size,
                'blur_radius': 0.0,
                'faces_per_pixel': 10,
                'bin_size': None,
                'max_faces_per_bin':  None,
                'perspective_correct': False,
            }
        raster_settings = dict2obj(raster_settings)
    else:
        raster_settings = RasterizationSettings(
            image_size=image_size, 
            blur_radius=0.0, 
            faces_per_pixel=10, 
        )
    
    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )

    meshes_screen = Meshes(verts=verts.float(), faces=faces.long(), textures=texture_pt3d).cuda()

    # import pdb;pdb.set_trace()
    images, fragments = renderer(meshes_screen)

    if False:
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            perspective_correct=raster_settings.perspective_correct,
        )
        
        vismask = (pix_to_face[..., 0] > -1).bool()
        nearest_depth_map = zbuf.clone()
        farrest_depth_map = zbuf.clone()

        farrest_depth_map = farrest_depth_map.max(-1)[0]
        
        nearest_depth_map[zbuf==-1] = 10.0
        nearest_depth_map = nearest_depth_map.min(-1)[0]

    else:
        pix_to_face, zbuf, bary_coords, dists = fragments
        vismask = (pix_to_face[..., 0] > -1).bool()
        nearest_depth_map = zbuf.clone()
        farrest_depth_map = zbuf.clone()
        import pdb;pdb.set_trace()
        farrest_depth_map = farrest_depth_map.max(-1)[0]
        farrest_depth_map[~vismask] = 0.0
        # farrest_depth_map = torch.flipud(farrest_depth_map[0])[None]

        nearest_depth_map[zbuf==-1] = 10.0
        nearest_depth_map = nearest_depth_map.min(-1)[0]
        nearest_depth_map[~vismask] = 0.0
        print(f'mask: {vismask.sum()}')

        import pdb;pdb.set_trace()
        # nearest_depth_map = torch.flipud(nearest_depth_map[0])[None]

    if True: # visualize
        import pdb;pdb.set_trace()
        front_range = nearest_depth_map.detach().cpu().numpy()
        back_range = farrest_depth_map.detach().cpu().numpy()
        all_min = min([front_range[vismask.detach().cpu().numpy()].min(), 
                back_range[vismask.detach().cpu().numpy()].min()])
        all_max = max([front_range.max(), back_range.max()])

        draw_color_map(front_range, \
            back_range,\
                all_min, all_max, save_path=os.path.join('/is/cluster/hyi/workspace/tmp/front_depth.png'))