import sys
sys.path.append('/is/cluster/hyi/workspace/Multi-IOI/bvh-distance-queries')
from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os.path as osp
import json
import os
# from compute_sdf_gpu_keepScale import compute_sdf
# TODO: for scene contact, it could be useful.
# from compute_sdf_gpu_keepScale_contact import compute_sdf_multiple_person
from thirdparty.body_models.smplifyx.utils_mics.utils import get_image
from tqdm import tqdm
from psbody.mesh import Mesh
from psbody.mesh.geometry.tri_normals import TriNormals, TriNormalsScaled

# * accumulate scene information for body optimization.
# TODO: output sdf volume, depth map, contact vertices for PROX/POSA

# TODO: scene model could merge all objects into one ply.

# * ground init
def init_ground_plane(self, verts):
    mean_y = verts[:, :, 1].mean()
    self.ground_plane.data.copy_(mean_y)
    return True

def load_whole_sdf_volume_scene(self, ply_file_list, contact_file_list, dim=256, padding=0.5, n_points_per_batch=1000000, \
        output_folder=None,
        device=torch.device('cuda'),
        dtype=torch.float32,
        postfix='scene',
        debug=False):
    scene_name = 'all_fuse'
    assert os.path.exists(os.path.join(output_folder, f'{scene_name}_sdf.npy'))

    logger.info('load sdf.')
    with open(osp.join(output_folder, scene_name + '.json'), 'r') as f:
        sdf_data = json.load(f)
        grid_min = torch.tensor(np.array(sdf_data['min']), dtype=dtype, device=device)
        grid_max = torch.tensor(np.array(sdf_data['max']), dtype=dtype, device=device)
        grid_dim = sdf_data['dim']
    voxel_size = (grid_max - grid_min) / grid_dim
    sdf = np.load(osp.join(output_folder, scene_name + '_sdf.npy')).reshape(grid_dim, grid_dim, grid_dim)
    sdf = torch.tensor(sdf, dtype=dtype, device=device)
    # contact = np.load(osp.join(output_folder, scene_name + '_contact.npy')).reshape(grid_dim, grid_dim, grid_dim)
    # contact = torch.tensor(contact, dtype=dtype, device=device)
    sdf_normals = np.load(osp.join(output_folder, scene_name + '_normals.npy')).reshape(grid_dim, grid_dim, grid_dim, 3)
    sdf_normals = torch.tensor(sdf_normals, dtype=dtype, device=device)
    # contact_normals = np.load(osp.join(output_folder, scene_name + '_contact_normals.npy')).reshape(grid_dim, grid_dim, grid_dim, 3)
    # contact_normals = torch.tensor(contact_normals, dtype=dtype, device=device)
        
     # TODO: get contact_vertices and contact_vn from self.contact_volume
    # contact_body_vertices, contact_body_verts_normals = self.get_contact_vertices_from_volume(contact, contact_normals, voxel_size, grid_min)
    # contact_body_vertices = contact_body_vertices.unsqueeze(0) # B, 
    
    self.register_buffer(f'sdf{postfix}', sdf)    
    self.register_buffer(f'sdf_normals{postfix}', sdf_normals)
    self.register_buffer(f'grid_min{postfix}', grid_min)
    self.register_buffer(f'grid_max{postfix}', grid_max)
    self.register_buffer(f'voxel_size{postfix}', voxel_size)

    return True

# define for compute_sdf_from_multiple_objects
def accumulate_whole_sdf_volume_scene(self, ply_file_list, contact_file_list, dim=256, padding=0.5, n_points_per_batch=1000000, \
        output_folder=None,
        postfix='scene',
        device=torch.device('cuda'),
        dtype=torch.float32,
        debug=False,
        use_trimesh_contains=True):
    
    # output_folder with 'scene'
    if contact_file_list[0] is None:
        CONTACT_FLAG = False
    else:
        CONTACT_FLAG = True

    # compute the SDF volume which contains the free space where human moves.
    # add contact label information into voxels in SDF volume.
    # TODO: for-loop each person.
    scene_name = 'all_fuse'
    if not os.path.exists(os.path.join(output_folder, f'{scene_name}_sdf.npy')):
        # TODO: add contact label.
        SDF, CONTACT_VOL, closest_faces, normals, contact_normals, closest_points, sdf_data = \
            compute_sdf_multiple_person(ply_file_list, contact_file_list, dim, padding, n_points_per_batch, \
            device, output_folder, use_trimesh_contains=use_trimesh_contains)
        grid_min = torch.tensor(np.array(sdf_data['min']), dtype=dtype, device=device)
        grid_max = torch.tensor(np.array(sdf_data['max']), dtype=dtype, device=device)
        grid_dim = sdf_data['dim']
        voxel_size = (grid_max - grid_min) / grid_dim
        sdf = SDF.reshape(grid_dim, grid_dim, grid_dim)
        sdf = torch.tensor(sdf, dtype=dtype, device=device)
        sdf_normals = normals.reshape(grid_dim, grid_dim, grid_dim, 3)
        sdf_normals = torch.tensor(sdf_normals, dtype=dtype, device=device)
        if CONTACT_FLAG:
            contact = CONTACT_VOL.reshape(grid_dim, grid_dim, grid_dim)
            contact = torch.tensor(contact, dtype=dtype, device=device)
            contact_normals = contact_normals.reshape(grid_dim, grid_dim, grid_dim, 3)
            contact_normals = torch.tensor(contact_normals, dtype=dtype, device=device)
    else: # load those info
        
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
        if CONTACT_FLAG:
            contact = np.load(osp.join(output_folder, scene_name + '_contact.npy')).reshape(grid_dim, grid_dim, grid_dim, 3)
            contact = torch.tensor(contact, dtype=dtype, device=device)
            contact_normals = np.load(osp.join(output_folder, scene_name + '_contact_normals.npy')).reshape(grid_dim, grid_dim, grid_dim, 3)
            contact_normals = torch.tensor(contact_normals, dtype=dtype, device=device)

    if CONTACT_FLAG:    
        # TODO: get contact_vertices and contact_vn from self.contact_volume
        contact_body_vertices, contact_body_verts_normals = self.get_contact_vertices_from_volume(contact, contact_normals, voxel_size, grid_min)
        contact_body_vertices = contact_body_vertices.unsqueeze(0) # B, 
    
    # import pdb;pdb.set_trace()
    if debug:
        import trimesh
        if CONTACT_FLAG:
            out_mesh = trimesh.Trimesh(contact_body_vertices[0].cpu().numpy(), vertex_normals=contact_body_verts_normals.cpu().numpy(), process=False)
            template_save_fn = os.path.join(output_folder, 'contact_sample.ply') 
            if not os.path.exists(template_save_fn):
                out_mesh.export(template_save_fn,vertex_normal=True) # export_ply
        
        # save out sdf volume as ply
        in_body_vertices, out_body_verts = self.get_vertices_from_sdf_volume(sdf, voxel_size, grid_min)
        out_mesh = trimesh.Trimesh(in_body_vertices.cpu().numpy(), process=False)
        template_save_fn = os.path.join(output_folder, 'in_body_sdf_sample.ply') 
        if not os.path.exists(template_save_fn):
            out_mesh.export(template_save_fn, vertex_normal=False) # export_ply
        out_mesh = trimesh.Trimesh(out_body_verts.cpu().numpy(), process=False)
        template_save_fn = os.path.join(output_folder, 'out_body_sdf_sample.ply') 
        if not os.path.exists(template_save_fn):
            out_mesh.export(template_save_fn, vertex_normal=False) # export_ply
        back_ground_points = out_body_verts.cpu().numpy()
        sample_num = int(back_ground_points.shape[0] / 10000)
        sample_back_ground_points = back_ground_points[np.random.choice(back_ground_points.shape[0], sample_num, replace=False)]
        out_mesh = trimesh.Trimesh(sample_back_ground_points, process=False)
        template_save_fn = os.path.join(output_folder, 'out_body_sdf_sample_0.00001.ply') 
        if not os.path.exists(template_save_fn):
            out_mesh.export(template_save_fn, vertex_normal=False) # export_ply

    # register buffer
    # import pdb;pdb.set_trace()
    # sdf axis is [2, 1, 0], z,y,x -> x,y,z
    # sdf_permute_np = sdf.permute([2, 1, 0]).detach().cpu().numpy()
    # with open(osp.join(output_folder, scene_name + '_sdf_permute.npy'), 'wb') as fout:
    #     np.save(fout, sdf_permute_np)

    # self.register_buffer('sdf', sdf.permute([2, 1, 0]))
    # self.register_buffer('sdf_normals', sdf_normals.permute([2, 1, 0, 3]))
    self.register_buffer(f'sdf{postfix}', sdf)    
    self.register_buffer(f'sdf_normals{postfix}', sdf_normals)
    self.register_buffer(f'grid_min{postfix}', grid_min)
    self.register_buffer(f'grid_max{postfix}', grid_max)
    self.register_buffer(f'voxel_size{postfix}', voxel_size)

    if CONTACT_FLAG:
        self.register_buffer(f'contact_volume{postfix}', contact)
        self.register_buffer(f'contact_normals{postfix}', contact_normals)
        self.register_buffer(f'contact_body_vertices{postfix}', contact_body_vertices)
        self.register_buffer(f'contact_body_verts_normals{postfix}', contact_body_verts_normals) 

    return True

# TODO: calculate sdf loss with specific sdf volume.
def compute_sdf_loss_scene(self, vertices, postfix='scene', dtype=torch.float32, device=torch.device('cuda'), output_folder=None):
    # vertices: b * v * 3
    nv = vertices.shape[1]
    batch = vertices.shape[0]

    if postfix=='':
        sdf = self.sdf
        sdf_normals = self.sdf_normals
        grid_min = self.grid_min
        grid_max = self.grid_max
        voxel_size = self.voxel_size
    elif postfix == 'scene':
        sdf = self.sdfscene
        sdf_normals = self.sdf_normalsscene
        grid_min = self.grid_minscene
        grid_max = self.grid_maxscene
        voxel_size = self.voxel_sizescene
        
    
    grid_dim = sdf.shape[0]
    sdf_ids = torch.round(
        (vertices.squeeze() - grid_min) / voxel_size).to(dtype=torch.long)
    sdf_ids.clamp_(min=0, max=grid_dim-1)
    # import pdb;pdb.set_trace()
    # sdf_ids_np = sdf_ids.detach().cpu().numpy()
    # scene_name = 'all_body'
    # with open(osp.join(output_folder, scene_name + '_sdf_ids_scene.npy'), 'wb') as fout:
    #     np.save(fout, sdf_ids_np)

    norm_vertices = (vertices - grid_min) / (grid_max - grid_min) * 2 - 1 # -1, 1

    # import pdb;pdb.set_trace()
    # grid_sample: x,y,z sample in [D, H, W] dimension grid.
    body_sdf = F.grid_sample(sdf.view(1, 1, grid_dim, grid_dim, grid_dim),
                                # norm_vertices.view(1, nv, 1, 1, 3),
                                norm_vertices[:, :, [2, 1, 0]].view(1, nv*batch, 1, 1, 3),
                                padding_mode='border')

    sdf_ids = sdf_ids.view(-1, 3)
    sdf_normals = sdf_normals[sdf_ids[:,0], sdf_ids[:,1], sdf_ids[:,2]]
    # if there are no penetrating vertices then set sdf_penetration_loss = 0
    if body_sdf.lt(0).sum().item() < 1:
        sdf_penetration_loss = torch.tensor(0.0, dtype=dtype, device=device)
    else:
        sdf_penetration_loss = (body_sdf[body_sdf < 0].unsqueeze(dim=-1).abs() * sdf_normals[body_sdf.view(-1) < 0, :]).pow(2).sum(dim=-1).sqrt().sum()
    
    # import pdb;pdb.set_trace()
        # sdf_penetration_loss = body_sdf[body_sdf < 0].unsqueeze(dim=-1).abs().sum()
    return sdf_penetration_loss / batch

# TODO: need to smplify.
def accumulate_ordinal_depth_scene(self, verts_list, faces_list, textures_list, img_list=None, debug=False, save_dir=None):
    # accumulate ordinal depth information between human and multiple objects, output a single ordinal depth map.
    # input: B * v * 3
    sil_template_objs = []
    sil_back_template_objs = []
    depth_template_objs = []
    depth_back_template_objs = []
    # Get mask and sil
    with torch.no_grad():
        for i, v in enumerate(verts_list):
            
            if img_list is not None:
                static_img = get_image(img_list[i])
            else:
                static_img = None

            _, depth, sil = self.renderer.render(
                verts_list[i], faces_list[i], textures_list[i])
            silhouette = sil == 1
            sil_template_objs.append(silhouette)
            depth_template_objs.append(depth)
            _, back_depth, back_sil = self.renderer_fd.render(v, faces_list[i], textures_list[i])
            back_silhouette = back_sil == 1
            sil_back_template_objs.append(back_silhouette)
            depth_back_template_objs.append(back_depth)
    
    self.register_buffer('sil_template_objs', torch.cat(sil_template_objs))
    self.register_buffer('sil_back_template_objs', torch.cat(sil_back_template_objs))
    self.register_buffer('depth_template_objs', torch.cat(depth_template_objs))
    self.register_buffer('depth_back_template_objs', torch.cat(depth_back_template_objs))
    return True

# TODO: input human vertices
def compute_depth_loss_scene(self, verts_person, img_list, debug=False):
    # compute relative depth loss for each object.
    # static scene mask: self.masks_object
    masks_object = self.masks_object == 1
    assert self.depth_template_objs.shape[0] == masks_object.shape[0]
    # assert img_list is not None

    loss_list = []
    for idx in range(verts_person.shape[0]): # perframe
        single_loss = torch.zeros(1).cuda()        
        # observed masks
        # TODO: set correct idx.
        # img_idx = img_listc[idx]
        masks, valid_flag, valid_person = self.get_perframe_mask(idx) # valid_flag: len = objects
        all_mask = [masks[one:one+1] for one in range(masks.shape[0])] # human mask is the last
        human_mask = all_mask[-1] # B, H, W: 1, 640, 640;

        # render human masks and depths
        _, depth, sil = self.renderer.render(
            verts_person[idx].unsqueeze(0), self.faces_person.unsqueeze(0), self.textures_person
        )
        silhouette = sil == 1

        _, back_depth, back_sil = self.renderer_fd.render(verts_person[idx].unsqueeze(0), self.faces_person.unsqueeze(0), self.textures_person)
        back_silhouette = back_sil == 1


        ## compute depth loss.
        # erosion mask
        from scipy.ndimage.morphology import binary_erosion
        # import pdb;pdb.set_trace()
        human_mask_np = human_mask.cpu().numpy()
        human_mask_np_tmp = binary_erosion(human_mask_np[0], iterations=2)
        human_mask_filter = torch.from_numpy(human_mask_np_tmp[None]).type_as(human_mask)
        silhouette_np = silhouette.cpu().numpy()
        silhouette_np_tmp = binary_erosion(silhouette_np[0], iterations=2)
        silhouette_filter = torch.from_numpy(silhouette_np_tmp[None]).type_as(human_mask)
        
        viz_body = human_mask_filter & silhouette_filter
        occlude_body = ~human_mask_filter & silhouette_filter
    
        # get depth loss with each objects.
         # valid_flag is the useful index.
        for i, obj_idx in enumerate(valid_flag):
            obj_mask = all_mask[i]
            # import pdb;pdb.set_trace()
            # small_range:
            front_body = viz_body & (~obj_mask) & masks_object[obj_idx]
            # large_range:
            # import pdb;pdb.set_trace()
            back_body = occlude_body & obj_mask & masks_object[obj_idx]
            
            if front_body.sum() > 0 and front_body.sum() > back_body.sum(): # TODO: hsi is using sum()
                depth_loss = F.relu(back_depth - self.depth_template_objs[obj_idx])[front_body].mean()
            elif back_body.sum() > 0 and front_body.sum() < back_body.sum():
                depth_loss = F.relu(self.depth_back_template_objs[obj_idx] - depth)[back_body].mean()
            else:
                depth_loss = torch.zeros(1).cuda()

            single_loss = single_loss + depth_loss
            
            # debug
            # import pdb;pdb.set_trace()
            # print(f'front_body: {front_body.sum()}; back_body: {back_body.sum()}')

            if debug:
                save_dir = '/is/cluster/work/hyi/results/HDSR/PROX_qualitative/N3OpenArea_00157_02/debug'
                static_img = get_image('/is/cluster/work/hyi/results/HDSR/PROX_qualitative/N3OpenArea_00157_02/Color_flip_rename/000704.jpg')
                print(f'calculate depth template: {i} / {len(valid_flag)}')
                import matplotlib.pyplot as plt
                import os
                fig = plt.figure(dpi=800, constrained_layout=True)
                ax1 = fig.add_subplot(6, 2, 1)
                ax1.imshow(human_mask[0].detach().cpu())
                ax1.axis("off")
                ax1.set_title("det_human_mask")

                ax2 = fig.add_subplot(6, 2, 2)
                ax2.imshow(silhouette[0].detach().cpu())
                ax2.axis("off")
                ax2.set_title("render_human_mask")

                # import pdb;pdb.set_trace()
                ax3 = fig.add_subplot(6, 2, 3)
                ax3.imshow(viz_body[0].detach().cpu())
                ax3.axis("off")
                ax3.set_title("viz_body", )

                ax4 = fig.add_subplot(6, 2, 4)
                ax4.imshow(occlude_body[0].detach().cpu())
                ax4.axis("off")
                ax4.set_title("occlude_body") #, fontsize=5

                
                ax5 = fig.add_subplot(6, 2, 5)
                ax5.imshow((front_body[0]).detach().cpu())
                ax5.axis("off")
                ax5.set_title("front_body_avaliable_region")

                ax6 = fig.add_subplot(6, 2, 6)
                ax6.imshow((back_body[0]).detach().cpu())
                ax6.axis("off")
                ax6.set_title("back_body_avaliable_region")
                
                ax7 = fig.add_subplot(6, 2, 7)
                ax7.imshow(masks_object[obj_idx].detach().cpu()) # masks_object[idx]: 640 * 640
                ax7.axis("off")
                ax7.set_title("static_scene_masks")

                ax8 = fig.add_subplot(6, 2, 8)
                ax8.imshow((obj_mask[0]).detach().cpu())
                ax8.axis("off")
                ax8.set_title("obseved_obj_mask")

                # import pdb;pdb.set_trace()
                ALPHA = 0.5
                # add input image as background
                height, width = static_img.shape[0], static_img.shape[1]
                fusion_map = np.zeros(static_img.shape).astype(np.uint8)
                fusion_map[:, :, 1] = 255
                
                obj_mask_fusion = obj_mask[0].detach().cpu().numpy().copy() 
                tmp_obj_mask_fusion_map = fusion_map * obj_mask_fusion[:height,:width, None]  + \
                        (1-obj_mask_fusion[:height,:width, None]) * static_img 
                tmp_obj_mask_fusion_map = tmp_obj_mask_fusion_map * ALPHA + (1-ALPHA)* static_img
                tmp_obj_mask_fusion_map = tmp_obj_mask_fusion_map.astype(np.uint8)

                ax9 = fig.add_subplot(6, 2, 9)
                ax9.imshow(tmp_obj_mask_fusion_map)
                ax9.axis("off")
                ax9.set_title("observed_obj_mask_fusion")
                
                front_body_fusion = front_body[0].detach().cpu().numpy().copy() 
                tmp_front_body_fusion_map = fusion_map * front_body_fusion[:height,:width, None]  + \
                        (1-front_body_fusion[:height,:width, None]) * static_img 
                tmp_front_body_fusion_map = tmp_front_body_fusion_map * ALPHA + (1-ALPHA)* static_img
                tmp_front_body_fusion_map = tmp_front_body_fusion_map.astype(np.uint8)

                ax10 = fig.add_subplot(6, 2, 10)
                ax10.imshow(tmp_front_body_fusion_map)
                ax10.axis("off")
                ax10.set_title("front_body_fusion")

                back_body_fusion = back_body[0].detach().cpu().numpy().copy() 
                tmp_back_body_fusion_map = fusion_map * back_body_fusion[:height,:width, None]  + \
                        (1-back_body_fusion[:height,:width, None]) * static_img 
                tmp_back_body_fusion_map = tmp_back_body_fusion_map * ALPHA + (1-ALPHA)* static_img
                tmp_back_body_fusion_map = tmp_back_body_fusion_map.astype(np.uint8)

                ax11 = fig.add_subplot(6, 2, 11)
                ax11.imshow(tmp_back_body_fusion_map)
                ax11.axis("off")
                ax11.set_title("back_body_fusion")

                
                os.makedirs(os.path.join(save_dir, 'debug_for_depth'), exist_ok=True)
                print(f'save to {os.path.join(save_dir, "debug_for_depth")}')
                plt.savefig(os.path.join(save_dir, f'debug_for_depth/new1_depth_oridinal_cmp_obj{obj_idx}.png'))
                # plt.show(block=False)
                plt.pause(1)
                plt.close()
                
        # import pdb;pdb.set_trace()
        loss_list.append(single_loss)
    
    all_loss = torch.stack(loss_list).squeeze(-1)
    return all_loss.sum()/(all_loss.nonzero().shape[0]+1e-9), all_loss


## accumulate_contact_label
def accumulate_contact_scene(self, verts_list, faces_list, debug=False, fixed=True):
    num_objs = len(verts_list)
    all_scene_vn_list =[]
    verts_shape = [0]
    for j in range(num_objs):
        scene_v = verts_list[j]
        scene_f = faces_list[j] 
        verts_shape.append(scene_v.shape[1]+verts_shape[-1])

        # import pdb;pdb.set_trace()
        ori_scene_f = scene_f[0].detach().cpu().numpy()
        flip_scene_f = np.stack([ori_scene_f[:, 2], ori_scene_f[:, 1], ori_scene_f[:, 0]], -1)
        import trimesh
        meshes = trimesh.Trimesh(scene_v[0].detach().cpu().numpy(), flip_scene_f, \
                                    process=False, maintain_order=True)
        scene_vn = torch.from_numpy(np.array(meshes.vertex_normals)).type_as(scene_v).unsqueeze(0)
        all_scene_vn_list.append(scene_vn)

        if debug and False:
            # save out 3D scene vertices and normal
            tmp_save_dir = os.path.join('/is/cluster/hyi/workspace/HCI/hdsr/mover_ori_repo_total3D/hdsr/mover_ori_repo/debug', 'contact')
            template_save_fn = os.path.join(tmp_save_dir, f'flipface_obj_{j}.obj')    
            meshes.export(template_save_fn,include_normals=True) # export_ply

    self.all_scene_v_list = verts_list
    self.all_scene_vn_list = all_scene_vn_list
    # all_scene_v = torch.cat(verts_list, 1)
    # all_scene_vn = torch.cat(all_scene_vn_list, 1)

    # if fixed:
    #     self.register_buffer('all_scene_v', all_scene_v)    
    #     self.register_buffer('all_scene_vn', all_scene_vn)

    return True

def compute_contact_loss_scene_prox(self, smplx_model_vertices, 
                contact_verts_ids=None, contact_angle=None, contact_robustifier=None, 
                ftov=None,
                debug=True, save_dir=None):

    assert contact_verts_ids is not None and contact_angle is not None and \
                    contact_robustifier is not None and ftov is not None

    #TODO: check overlap.
    # TODO: support batch-wise process.
    # B, N, 3
    body_verts = smplx_model_vertices
    if len(body_verts.shape) == 2:
        body_verts = body_verts.unsqueeze(0)
    batch_size = body_verts.shape[0]

    body_faces = self.faces_person_close_mouth.unsqueeze(0) 

    obj_list = [one.squeeze() for one in self.all_scene_v_list]
    obj_boxes = self.sdf_losses.get_bounding_boxes(obj_list)

    all_contact_loss_list = []

    # import pdb;pdb.set_trace()
    for idx in range(body_verts.shape[0]):
        contact_loss = torch.tensor(0.).cuda()
        one_body_v = body_verts[idx] # Nx3
        body_box = self.sdf_losses.get_bounding_boxes([one_body_v])

        # for contact scene;
        for i, obj in enumerate(obj_list):

            ## TODO: check overlap
            # if not self.sdf_losses.check_overlap(obj_boxes[i], body_box):
            #     continue

            contact_body_vertices = one_body_v[None, contact_verts_ids, :]
            scene_v = obj.unsqueeze(0)
            scene_vn = self.all_scene_vn_list[i] # 1, N, 3

            # calculate contact loss
            import mover.dist_chamfer as ext
            distChamfer = ext.chamferDist()
            contact_dist, _, idx1, _ = distChamfer(
                contact_body_vertices.contiguous(), scene_v)

            body_model_faces = self.faces_person.long()
            body_triangles = torch.index_select(
                one_body_v[None], 1,
                body_model_faces.view(-1)).view(1, -1, 3, 3)

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
            # contact_cntvertix normals of contact vertices
            # import pdb;pdb.set_trace()
            # tmp_body_v_n = []
            # for b_i in range(batch_size):
            #     body_v_normals = torch.mm(ftov, body_normals[b_i])
            #     tmp_body_v_n.append(body_v_normals)
            # import pdb;pdb.set_trace()
            # body_v_normals = torch.stack(tmp_body_v_n)
            body_v_normals = torch.mm(ftov, body_normals[0])[None]

            body_v_normals = body_v_normals / \
                torch.norm(body_v_normals, 2, dim=2, keepdim=True)
            contact_body_verts_normals = body_v_normals[:, contact_verts_ids, :]

            # import pdb;pdb.set_trace()
            # scene normals of the closest points on the scene surface to the contact vertices
            contact_scene_normals = scene_vn[:, idx1.squeeze().to(
                dtype=torch.long), :].expand(1, -1, -1)
            # import pdb;pdb.set_trace()
            # compute the angle between contact_verts normals and scene normals
            angles = torch.asin(
                torch.norm(torch.cross(contact_body_verts_normals, contact_scene_normals), 2, dim=-1, keepdim=True)) *180 / np.pi

            # consider only the vertices which their normals match 
            valid_contact_mask = (angles.le(contact_angle).int() + angles.ge(180 - contact_angle).int()).ge(1)
            valid_contact_ids = valid_contact_mask.squeeze().nonzero().squeeze()
        
            contact_dist = contact_robustifier(contact_dist[:, valid_contact_ids].sqrt())
            # ! Warning: exist none match, in 37 frame, it leads to None.
            if contact_dist.shape[-1] != 0:
                contact_loss = contact_loss + contact_dist.mean()

        
        all_contact_loss_list.append(contact_loss)        
        # all_contact_loss = torch.stack(all_contact_loss_list).squeeze(-1)
    all_contact_loss = torch.stack(all_contact_loss_list)

    return all_contact_loss.sum() / (all_contact_loss.nonzero().shape[0] + 1e-9), all_contact_loss



def compute_contact_loss_scene_prox_batch(self, smplx_model_vertices, 
                contact_verts_ids=None, contact_angle=None, contact_robustifier=None, 
                ftov=None,
                debug=True, save_dir=None):

    assert contact_verts_ids is not None and contact_angle is not None and \
                    contact_robustifier is not None and ftov is not None
    
    # * batch-wise PROX
    # B, N, 3
    body_verts = smplx_model_vertices
    if len(body_verts.shape) == 2:
        body_verts = body_verts.unsqueeze(0)
    batch_size = body_verts.shape[0]
    
    # contact loss init
    contact_loss = torch.tensor(0.).cuda()

    body_faces = self.faces_person_close_mouth.unsqueeze(0) 

    obj_list = [one.squeeze() for one in self.all_scene_v_list]
    obj_boxes = self.sdf_losses.get_bounding_boxes(obj_list)

    # get batch body normal
    body_model_faces = self.faces_person.long()
    body_triangles = torch.index_select(
        body_verts, 1,
        body_model_faces.view(-1)).view(batch_size, -1, 3, 3)

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
    # contact_cntvertix normals of contact vertices
    # import pdb;pdb.set_trace()
    tmp_body_v_n = []
    for b_i in range(batch_size):
        body_v_normals = torch.mm(ftov, body_normals[b_i])
        tmp_body_v_n.append(body_v_normals)
    
    body_v_normals = torch.stack(tmp_body_v_n)
    body_v_normals = body_v_normals / \
        torch.norm(body_v_normals, 2, dim=2, keepdim=True)
    contact_body_verts_normals = body_v_normals[:, contact_verts_ids, :].view(1, -1, 3)
    contact_body_vertices = body_verts[:, contact_verts_ids, :].view(1, -1, 3)

    # ge all scene info
    scene_v = torch.cat(self.all_scene_v_list, dim=1).contiguous()
    scene_vn = torch.cat(self.all_scene_vn_list, dim=1).contiguous()
    # calculate contact loss
    import mover.dist_chamfer as ext
    distChamfer = ext.chamferDist()
    contact_dist, _, idx1, _ = distChamfer(contact_body_vertices.contiguous(), scene_v)
    # contact_dist = contact_dist.view(batch_size, -1)
    # idx1 = idx1.view(batch_size, -1)

    # scene normals of the closest points on the scene surface to the contact vertices
    contact_scene_normals = scene_vn[:, idx1.squeeze().to(dtype=torch.long), :].expand(1, -1, -1)
    # import pdb;pdb.set_trace()
    # compute the angle between contact_verts normals and scene normals
    angles = torch.asin(
        torch.norm(torch.cross(contact_body_verts_normals, contact_scene_normals), 2, dim=-1, keepdim=True)) *180 / np.pi

    # consider only the vertices which their normals match 
    valid_contact_mask = (angles.le(contact_angle).int() + angles.ge(180 - contact_angle).int()).ge(1)
    valid_contact_ids = valid_contact_mask.squeeze().nonzero().squeeze()

    contact_dist = contact_robustifier(contact_dist[:, valid_contact_ids].sqrt())
    # ! Warning: exist none match, in 37 frame, it leads to None.
    if contact_dist.shape[-1] != 0:
        contact_loss = contact_loss + contact_dist.mean()
    
    return contact_loss, [contact_loss]


def compute_contact_loss_scene_posa():
    pass