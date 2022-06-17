import sys
from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os.path as osp
import json
import os
import pickle
from thirdparty.body_models.smplifyx.utils_mics.utils import get_image
from tqdm import tqdm
import trimesh
from psbody.mesh import Mesh
from psbody.mesh.geometry.tri_normals import TriNormals, TriNormalsScaled

from mover.utils.visualize import define_color_map
from mover.constants import (
    LOSS_NORMALIZE_PER_ELEMENT,
    DEBUG_DEPTH_PEROBJ_MASK,
    PYTORCH3D_DEPTH
)


def accumulate_ordinal_depth_from_human(self, verts_person, img_list=None, debug=False, save_dir=None):
    # accumulate ordinal depth information between human and multiple objects
    # verts_person: all human verts (B * v * 3);
    tmp_human_path = os.path.join(save_dir, f'depth_template_human.pickle')
    if self.RECALCULATE_HCI_INFO or not os.path.exists(tmp_human_path):
        depth_template = None
        logger.info('recalculate depth template from human ')
        
        with torch.no_grad():
            for idx in tqdm(range(verts_person.shape[0]), desc='accumulate depth range map from human'):
                masks, valid_flag, valid_person = self.get_perframe_mask(idx)
                all_mask = [masks[one:one+1] for one in range(masks.shape[0])] 
                if img_list is not None:
                    static_img = get_image(img_list[idx])
                else:
                    static_img = None

                # not contain any human
                if valid_person == True:
                    continue
                
                if PYTORCH3D_DEPTH:
                    silhouette, depth, back_depth = self.get_depth_map_pytorch3d(verts_person[idx].unsqueeze(0), self.faces_person.unsqueeze(0))
                    back_silhouette = silhouette.clone()
                else:
                    _, depth, sil = self.renderer.render(verts_person[idx].unsqueeze(0), self.faces_person.unsqueeze(0), self.textures_person)
                    silhouette = sil == 1
                    
                    _, back_depth, back_sil = self.renderer_fd.render(verts_person[idx].unsqueeze(0), self.faces_person.unsqueeze(0), self.textures_person)
                    back_silhouette = back_sil == 1
                
                depth_template = self.calculate_depth_template(depth, silhouette, back_depth, back_silhouette, \
                                                all_mask, valid_flag, pre_depth_template=depth_template, \
                                                static_img=static_img, debug=DEBUG_DEPTH_PEROBJ_MASK, save_dir=save_dir, save_idx=idx)
        
        with open(tmp_human_path, 'wb') as fout:
            pickle.dump( 
                {'depth_template_human': depth_template,
            },fout)

    else:
        logger.info('load depth template from human ')
        with open(tmp_human_path, 'rb') as fin:
            input_dict = pickle.load(fin)
        depth_template = input_dict['depth_template_human']
    self.register_buffer('depth_template_human', depth_template)

    return True

def transfer_depth_range_into_points(self, output_folder):
    
    front_valid = (self.depth_template_human[0, :, :, 0] > 0) & (self.depth_template_human[0, :, :, 0] <100)
    yx = front_valid.nonzero()
    xy = torch.stack((yx[:, 1], yx[:, 0]), -1)
    xy1 = torch.cat((xy, torch.ones((xy.shape[0], 1)).type_as(xy)), -1).float()
    front_depth = self.depth_template_human[0, :, :, 0][front_valid]
    front_verts = torch.matmul(torch.inverse(self.K_intrin[0]), torch.transpose((front_depth[:, None] * xy1), 0,1))
    out_mesh = trimesh.Trimesh(front_verts.cpu().numpy().T, \
            process=False)
    template_save_fn = os.path.join(output_folder, f'depth_front_points.ply')
    out_mesh.export(template_save_fn)
    
    # back
    front_valid = (self.depth_template_human[0, :, :, 1] > 0) & (self.depth_template_human[0, :, :, 1] <100)
    yx = front_valid.nonzero()
    xy = torch.stack((yx[:, 1], yx[:, 0]), -1)
    xy1 = torch.cat((xy, torch.ones((xy.shape[0], 1)).type_as(xy)), -1).float()
    front_depth = self.depth_template_human[0, :, :, 1][front_valid]
    front_verts = torch.matmul(torch.inverse(self.K_intrin[0]), torch.transpose((front_depth[:, None] * xy1), 0,1))
    out_mesh = trimesh.Trimesh(front_verts.cpu().numpy().T, \
            process=False)
    template_save_fn = os.path.join(output_folder, f'depth_back_points.ply')
    out_mesh.export(template_save_fn) # export_ply

    return True

def calculate_depth_template(self, depth, silhouette, back_depth, back_silhouette, \
            all_mask, valid_flag, pre_depth_template=None, debug=False, save_dir=None, static_img=None, save_idx=None): 
    # (back) depth, silhouette: the (back) depth and silhouette of a rendered human.
    # each frame observed mask: all_mask.
    # static scene mask: self.masks_object.
    # compute relative depth in the static scene mask region.

    masks_object = self.masks_object == 1
    human_mask = all_mask[-1] # B, H, W: 1, 640, 640;

    # flag, small_range, large_range
    if pre_depth_template is None:
        depth_template = torch.zeros((1, human_mask.shape[1], human_mask.shape[2], 3)).type_as(depth)
        depth_template[:, :, :, 1] = 100
        
    else:
        depth_template = pre_depth_template.clone()

    # 2D det human mask & projected SMPLX mask
    from scipy.ndimage.morphology import binary_erosion, binary_dilation
    human_mask_np = human_mask.cpu().numpy()
    human_mask_np_tmp = binary_erosion(human_mask_np[0], iterations=2)
    human_mask_filter = torch.from_numpy(human_mask_np_tmp[None]).type_as(human_mask)
    silhouette_np = silhouette.cpu().numpy()
    silhouette_np_tmp = binary_erosion(silhouette_np[0], iterations=2)
    silhouette_filter = torch.from_numpy(silhouette_np_tmp[None]).type_as(human_mask)
    
    viz_body = human_mask_filter & silhouette_filter
    occlude_body = ~human_mask_filter & silhouette_filter
    
    for idx, obj_idx in enumerate(valid_flag):
        obj_mask = all_mask[idx]
        # small_range:
        front_body = viz_body & (~obj_mask) & masks_object[obj_idx]
        # large_range:
        back_body = occlude_body & obj_mask & masks_object[obj_idx]

        # since the smplx is naked body and noise detection result; we only select either frontal or back information.
        # back_depth used in frontal body.
        if front_body.sum() > 0 and front_body.sum() > back_body.sum():
            if pre_depth_template is None:
                depth_template[front_body] = torch.stack([back_depth[front_body], depth_template[front_body][:, 1], depth_template[front_body][:, 2]], -1)
            else:
                depth_template[front_body] = torch.stack([torch.max(torch.cat([back_depth[front_body].unsqueeze(-1), pre_depth_template[front_body][:, 0].unsqueeze(-1)], -1), -1)[0], \
                                                depth_template[front_body][:, 1], depth_template[front_body][:, 2]], -1)
        elif back_body.sum() > 0 and front_body.sum() < back_body.sum():
            if pre_depth_template is None:
                depth_template[back_body] = torch.stack([depth_template[back_body][:,0], depth[back_body], depth_template[back_body][:,2]], -1)
            else:
                depth_template[back_body] = torch.stack([depth_template[back_body][:,0], \
                                torch.min(torch.cat([depth[back_body].unsqueeze(-1), pre_depth_template[back_body][:, 1].unsqueeze(-1)], -1), -1)[0], \
                                depth_template[back_body][:,2]], -1)

        if debug:
            font_size = 5
            print(f'calculate depth template: {idx} / {len(all_mask)}')
            import matplotlib.pyplot as plt
            import os
            fig = plt.figure(dpi=800, constrained_layout=True)
            ax1 = fig.add_subplot(6, 2, 1)
            ax1.imshow(human_mask[0].detach().cpu())
            ax1.axis("off")
            ax1.set_title("det_human_mask", fontsize=font_size)

            ax2 = fig.add_subplot(6, 2, 2)
            ax2.imshow(silhouette[0].detach().cpu())
            ax2.axis("off")
            ax2.set_title("render_human_mask", fontsize=font_size)

            ax3 = fig.add_subplot(6, 2, 3)
            ax3.imshow(viz_body[0].detach().cpu())
            ax3.axis("off")
            ax3.set_title("viz_body", fontsize=font_size)

            ax4 = fig.add_subplot(6, 2, 4)
            ax4.imshow(occlude_body[0].detach().cpu())
            ax4.axis("off")
            ax4.set_title("occlude_body", fontsize=font_size) #, fontsize=5

            
            ax5 = fig.add_subplot(6, 2, 5)
            ax5.imshow((front_body[0]).detach().cpu())
            ax5.axis("off")
            ax5.set_title("front_body_avaliable_region", fontsize=font_size)

            ax6 = fig.add_subplot(6, 2, 6)
            ax6.imshow((back_body[0]).detach().cpu())
            ax6.axis("off")
            ax6.set_title("back_body_avaliable_region", fontsize=font_size)
            
            ax7 = fig.add_subplot(6, 2, 7)
            ax7.imshow(masks_object[obj_idx].detach().cpu()) # masks_object[idx]: 640 * 640
            ax7.axis("off")
            ax7.set_title("static_scene_masks", fontsize=font_size)

            ax8 = fig.add_subplot(6, 2, 8)
            ax8.imshow((obj_mask[0]).detach().cpu())
            ax8.axis("off")
            ax8.set_title("obseved_obj_mask", fontsize=font_size)

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
            ax9.set_title("observed_obj_mask_fusion", fontsize=font_size)
            
            front_body_fusion = front_body[0].detach().cpu().numpy().copy() 
            tmp_front_body_fusion_map = fusion_map * front_body_fusion[:height,:width, None]  + \
                    (1-front_body_fusion[:height,:width, None]) * static_img 
            tmp_front_body_fusion_map = tmp_front_body_fusion_map * ALPHA + (1-ALPHA)* static_img
            tmp_front_body_fusion_map = tmp_front_body_fusion_map.astype(np.uint8)

            ax10 = fig.add_subplot(6, 2, 10)
            ax10.imshow(tmp_front_body_fusion_map)
            ax10.axis("off")
            ax10.set_title("front_body_fusion", fontsize=font_size)

            back_body_fusion = back_body[0].detach().cpu().numpy().copy() 
            tmp_back_body_fusion_map = fusion_map * back_body_fusion[:height,:width, None]  + \
                    (1-back_body_fusion[:height,:width, None]) * static_img 
            tmp_back_body_fusion_map = tmp_back_body_fusion_map * ALPHA + (1-ALPHA)* static_img
            tmp_back_body_fusion_map = tmp_back_body_fusion_map.astype(np.uint8)

            ax11 = fig.add_subplot(6, 2, 11)
            ax11.imshow(tmp_back_body_fusion_map)
            ax11.axis("off")
            ax11.set_title("back_body_fusion", fontsize=font_size)

            
            os.makedirs(os.path.join(save_dir, 'debug_for_depth'), exist_ok=True)
            print(f'save to {os.path.join(save_dir, "debug_for_depth")}')
            plt.savefig(os.path.join(save_dir, f'debug_for_depth/new1_depth_oridinal_cmp_frame{save_idx}_{idx}.png'))
            plt.pause(1)
            plt.close()

    all_obj_mask = torch.stack(all_mask).sum(0)
    all_obj_mask_np = all_obj_mask.detach().cpu().numpy()
    all_obj_mask_np = binary_dilation(all_obj_mask_np, iterations=2)
    all_obj_mask_filter = torch.from_numpy(all_obj_mask_np).type_as(human_mask)
    front_body_free_space = viz_body & all_obj_mask_filter
    if front_body_free_space.sum() > 0:
        if pre_depth_template is None:
            depth_template[front_body_free_space] = torch.cat([depth_template[front_body_free_space][:, :2], back_depth[front_body_free_space].unsqueeze(-1)], -1)
        else:
            depth_template[front_body_free_space] = torch.stack([
                                            depth_template[front_body_free_space][:, 0], depth_template[front_body_free_space][:, 1], \
                                            torch.max(torch.cat([back_depth[front_body_free_space].unsqueeze(-1), pre_depth_template[front_body_free_space][:, 2].unsqueeze(-1)], -1), -1)[0]], -1)
    if debug:
        print(f'calculate depth template on free objects space !')
        import matplotlib.pyplot as plt
        import os
        fig = plt.figure(dpi=800, constrained_layout=True)
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(front_body_free_space[0].detach().cpu())
        ax1.axis("off")
        ax1.set_title("front_body_free_space")

        ALPHA = 0.5
        # add input image as background
        height, width = static_img.shape[0], static_img.shape[1]
        fusion_map = np.zeros(static_img.shape).astype(np.uint8)
        fusion_map[:, :, 1] = 255
        
        obj_mask_fusion = front_body_free_space[0].detach().cpu().numpy().copy() 
        tmp_obj_mask_fusion_map = fusion_map * obj_mask_fusion[:height,:width, None]  + \
                (1-obj_mask_fusion[:height,:width, None]) * static_img 
        tmp_obj_mask_fusion_map = tmp_obj_mask_fusion_map * ALPHA + (1-ALPHA)* static_img
        tmp_obj_mask_fusion_map = tmp_obj_mask_fusion_map.astype(np.uint8)

        ax9 = fig.add_subplot(1, 2, 2)
        ax9.imshow(tmp_obj_mask_fusion_map)
        ax9.axis("off")
        ax9.set_title("observed_front_body_free_space")
        os.makedirs(os.path.join(save_dir, 'debug_for_depth_freeSpace'), exist_ok=True)
        print(f'save to {os.path.join(save_dir, "debug_for_depth_freeSpace")}')
        plt.savefig(os.path.join(save_dir, f'debug_for_depth_freeSpace/front_body_free_space_depth_{save_idx}.png'))
        # plt.show(block=False)
        plt.pause(1)
        plt.close()
                
    return depth_template 


def compute_overlap_region_on_obj(self, verts_list, faces_list, textures_list):
    
    masks_object = self.masks_object == 1
    front_valid_list = []
    back_valid_list = []
    with torch.no_grad():
        for i, v in enumerate(verts_list):
            if PYTORCH3D_DEPTH:
                silhouette, depth, back_depth = self.get_depth_map_pytorch3d(v, faces_list[i])
            else:
                _, depth, sil  = self.renderer.render(v, faces_list[i], textures_list[i])
                silhouette = sil == 1

            valid_region = masks_object[i] & silhouette # (self.depth_template_human.sum(-1) != 0)
            front_valid = (self.depth_template_human[:, :, :, 0] > 0) & (self.depth_template_human[:, :, :, 0] <100) & valid_region
            back_valid = (self.depth_template_human[:, :, :, 1] > 0) & (self.depth_template_human[:, :, :, 1] <100) & valid_region
            front_valid_list.append(front_valid)
            back_valid_list.append(back_valid)

    self.register_buffer('front_valid_obj', torch.stack(front_valid_list))
    self.register_buffer('back_valid_obj', torch.stack(back_valid_list))
    
    return True
    
def compute_relative_depth_loss_range(self, verts_list, faces_list, textures_list, debug=False, detailed=False):
    # compute relative depth loss for each object.
    masks_object = self.masks_object == 1

    silhouettes = []
    depths = []
    loss = torch.tensor(0, dtype=torch.float32, device=torch.device('cuda'))
    loss_fs = torch.tensor(0, dtype=torch.float32, device=torch.device('cuda'))
    loss_overlap_consistency = torch.tensor(0, dtype=torch.float32, device=torch.device('cuda'))
    if debug: 
        front_sample_points = []
        back_sample_points = []
    if detailed:
        loss_list = []
        loss_fs_list = []
        loss_overlap_list = []
    for i, v in enumerate(verts_list):

        # * warning: gradient is too large, which leads to Nan. 
        if PYTORCH3D_DEPTH:
            silhouette, depth, back_depth = self.get_depth_map_pytorch3d(v, faces_list[i])
            back_silhouette = silhouette.clone()
            sil = silhouette.float()
        else:
            _, depth, sil  = self.renderer.render(v, faces_list[i], textures_list[i])
            silhouette = sil == 1 # !warning: requires_grad=False
            _, back_depth, back_sil = self.renderer_fd.render(v, faces_list[i], textures_list[i])
            back_silhouette = back_sil == 1

        if debug:
            index = back_silhouette.nonzero()[:, 1:].float() / 320
            front_sample_points.append(torch.cat([index, depth[silhouette][:, None]], -1))
            back_sample_points.append(torch.cat([index, back_depth[back_silhouette][:, None]], -1))
        
        # compute penalty between objects and depth_template
        # depths <-> depth_template : use ReLU.
        valid_region = masks_object[i] & silhouette 

        # add one iteration erotion
        from scipy.ndimage.morphology import binary_erosion, binary_dilation
        valid_region_np = valid_region.cpu().numpy()
        valid_region_tmp = binary_erosion(valid_region_np[0], iterations=1)
        valid_region = torch.from_numpy(valid_region_tmp[None]).type_as(silhouette)
        
        # outside silhouette region;
        fs_valid_region = ~masks_object[i] & silhouette 
        # add one iteration erotion
        fs_valid_region_np = fs_valid_region.cpu().numpy()
        fs_valid_region_tmp = binary_erosion(fs_valid_region_np[0], iterations=2)
        fs_valid_region = torch.from_numpy(fs_valid_region_tmp[None]).type_as(silhouette)

        front_valid = (self.depth_template_human[:, :, :, 0] > 0) & (self.depth_template_human[:, :, :, 0] <100) & valid_region
        back_valid = (self.depth_template_human[:, :, :, 1] > 0) & (self.depth_template_human[:, :, :, 1] <100) & valid_region

        front_fs_valid = (self.depth_template_human[:, :, :, 2] > 0) & (self.depth_template_human[:, :, :, 2] <100) & fs_valid_region

        front_relative_depth_loss = torch.clamp(self.depth_template_human[:, :, :, 0] - depth, min=0.0, max=2.0)
        front_real_valid = (front_relative_depth_loss.detach()>0.0).float() * front_valid.float()
        front_relative_depth_loss = front_relative_depth_loss[front_real_valid.type(torch.uint8)]
        
        back_relative_depth_loss = torch.clamp(back_depth - self.depth_template_human[:, :, :, 1], min=0.0, max=2.0)
        back_real_valid = (back_relative_depth_loss.detach()>0.0).float() * back_valid.float()
        back_relative_depth_loss = back_relative_depth_loss[back_real_valid.type(torch.uint8)]
        
        front_fs_relative_depth_loss = torch.clamp(self.depth_template_human[:, :, :, 2] - depth, min=0.0, max=2.0)
        front_fs_real_valid = (front_fs_relative_depth_loss.detach()>0.0).float() * front_fs_valid.float()
        front_fs_relative_depth_loss = front_fs_relative_depth_loss[front_fs_real_valid.type(torch.uint8)]
        
        if LOSS_NORMALIZE_PER_ELEMENT: 
            if front_real_valid.sum() == 0 and back_real_valid.sum() == 0:
                tmp_loss = torch.tensor(0.).cuda().squeeze() 
                tmp_overlap_loss = torch.tensor(0.).cuda().squeeze()
            elif front_real_valid.sum() == 0:
                tmp_loss = back_relative_depth_loss.sum() / back_real_valid.sum()
                tmp_overlap_loss = torch.abs(sil-self.back_valid_obj[i].float())[self.back_valid_obj[i]].sum()/ (self.back_valid_obj[i].sum().float()+1e-9)
            elif back_real_valid.sum() == 0:
                tmp_loss = front_relative_depth_loss.sum() / front_real_valid.sum()
                tmp_overlap_loss = torch.abs(sil-self.front_valid_obj[i].float())[self.front_valid_obj[i]].sum()/ (self.front_valid_obj[i].sum().float()+1e-9)
            else:
                tmp_loss = front_relative_depth_loss.sum() / front_real_valid.sum() + \
                                back_relative_depth_loss.sum() / back_real_valid.sum()
                tmp_overlap_loss = torch.abs(sil-self.front_valid_obj[i].float())[self.front_valid_obj[i]].sum()/ (self.front_valid_obj[i].sum().float()+1e-9) + \
                    torch.abs(sil-self.back_valid_obj[i].float())[self.back_valid_obj[i]].sum()/ (self.back_valid_obj[i].sum().float()+1e-9)
        else:
            if front_real_valid.sum() == 0 and back_real_valid.sum() == 0:
                tmp_loss = torch.tensor(0.).cuda().squeeze()
                tmp_overlap_loss = torch.tensor(0.).cuda().squeeze()
            elif front_real_valid.sum() == 0:
                tmp_loss = back_relative_depth_loss.sum()
                tmp_overlap_loss = torch.abs(sil-self.back_valid_obj[i].float())[self.back_valid_obj[i]].sum()
            elif back_real_valid.sum() == 0:
                tmp_loss = front_relative_depth_loss.sum() 
                tmp_overlap_loss = torch.abs(sil-self.front_valid_obj[i].float())[self.front_valid_obj[i]].sum()
            else:
                tmp_loss = front_relative_depth_loss.sum() + \
                                back_relative_depth_loss.sum() 
                tmp_overlap_loss = torch.abs(sil-self.front_valid_obj[i].float())[self.front_valid_obj[i]].sum() + \
                    torch.abs(sil-self.back_valid_obj[i].float())[self.back_valid_obj[i]].sum()

            if front_fs_real_valid.sum() == 0:
                tmp_loss_fs = torch.tensor(0.).cuda().squeeze() 
            else:
                tmp_loss_fs = front_fs_relative_depth_loss.sum()
        
        loss_fs = loss_fs + tmp_loss_fs
        loss  = loss + tmp_loss
        loss_overlap_consistency = loss_overlap_consistency + tmp_overlap_loss

        if detailed:
            loss_list.append(tmp_loss)
            loss_overlap_list.append(tmp_overlap_loss)
            loss_fs_list.append(tmp_loss_fs)

    if debug:
        front_points = torch.cat(front_sample_points).detach().cpu().numpy()
        back_points = torch.cat(back_sample_points).detach().cpu().numpy()
        f_out_mesh = trimesh.Trimesh(front_points, process=False)
        f_out_mesh.export('debug/front_points.ply')
        b_out_mesh = trimesh.Trimesh(back_points, process=False)
        b_out_mesh.export('debug/back_points.ply')

    if detailed:
        return loss, loss_overlap_consistency, loss_list, loss_overlap_list, loss_fs, loss_fs_list, 
    else:
        return loss, loss_overlap_consistency, loss_fs 