import torch
import numpy as np

## collision loss
def collision_loss(self, verts_list, faces_list, sampled_verts_list=None):
    if self.resample_in_sdf:
        assert sampled_verts_list is not None
        # sampled_verts_list = [one.squeeze(0) for one in sampled_verts_list]
        loss_collision_objs = self.sdf_losses(verts_list, faces_list, even_sampled_vertices=sampled_verts_list, scale_factor=0.05) # modified
    else:
        loss_collision_objs = self.sdf_losses(verts_list, faces_list, scale_factor=0.05)
    return loss_collision_objs

## ordinal loss
# with no-human image and mask
def compute_ordinal_depth_loss(self):

    verts = self.get_verts_object()
    # import pdb;pdb.set_trace()
    verts_list = []
    faces_list = []
    textures_list = []
    for one in range(len(self.idx_each_object)):
        obj_verts, obj_faces, obj_textures = self.get_one_verts_faces_obj(verts, one, texture=True)
        verts_list.append(obj_verts)
        faces_list.append(obj_faces)
        textures_list.append(obj_textures)
    
    silhouettes = []
    depths = []
    for i, v in enumerate(verts_list):
        # K = obj_cam_rois[i]
        _, depth, sil  = self.renderer(v, faces_list[i], textures_list[i])
        depths.append(depth)
        silhouettes.append((sil == 1))
    # ! warning: mask for ordinal loss need bool.
    masks = self.masks_object==1 # transfer float to byte
    # import pdb;pdb.set_trace()
    masks = [masks[one:one+1] for one in range(masks.shape[0])]

    
    if False:
        import matplotlib.pyplot as plt
        import os
        for one in range(len(masks)):
            fig = plt.figure(figsize=(6, 2), dpi=800)
            ax1 = fig.add_subplot(3, 1, 1)
            ax1.imshow(masks[one][0].detach().cpu())
            ax1.axis("off")
            ax1.set_title("ori masks", fontsize=5)

            ax1 = fig.add_subplot(3, 1, 2)
            ax1.imshow(silhouettes[one][0].detach().cpu())
            ax1.axis("off")
            ax1.set_title("silhouettes", fontsize=5)
            # import pdb;pdb.set_trace()
            ax1 = fig.add_subplot(3, 1, 3)
            ax1.imshow(depths[one][0].detach().cpu())
            ax1.axis("off")
            ax1.set_title("depths", fontsize=5)

            plt.savefig(f'/is/cluster/hyi/workspace/HCI/hdsr/mover_ori_repo/debug/debug_depth_oridinal_mask_{one}.png')
            # plt.show(block=False)
            plt.pause(1)
            plt.close()

    return self.losses.compute_ordinal_depth_loss(masks, silhouettes, depths)

# with new per-frame image and mask
# objects and human all in camera coordinates system.
def compute_ordinal_depth_loss_perframe(self, verts_person, idx, \
                verts_list=None, faces_list=None, textures_list=None, \
                use_for_human=True):

    if verts_list is None:
        # use scene model verts, faces and textures
        _, verts_list = self.get_verts_object(return_all=True)
        faces_list, textures_list = self.faces_list, self.textures_list
    
    masks, valid_flag, valid_person = self.get_perframe_mask(idx)
    if valid_person == True:
        return 0.0

    # check overlap between 2D bboxes.
    verts_list = [verts_list[i] for i in valid_flag]
    faces_list = [faces_list[i] for i in valid_flag]
    textures_list = [textures_list[i] for i in valid_flag]

    silhouettes = []
    depths = []
    for i, v in enumerate(verts_list):
        # K = obj_cam_rois[i]
        _, depth, sil  = self.renderer.render(v, faces_list[i], textures_list[i])
        depths.append(depth)
        silhouettes.append((sil == 1)) #.type(torch.bool))

    assert verts_person.shape[0] == 1
    # logger.info(f"exist person {verts_person.shape[0]} in ordinal depth loss")

    for idx in range(verts_person.shape[0]):
        # cam_local_body = self.perframe_cam_rois[idx][:-1][-1]
        _, depth, sil = self.renderer.render(
            verts_person[idx].unsqueeze(0), self.faces_person.unsqueeze(0), self.textures_person
        )
        depths.append(depth)
        silhouettes.append((sil == 1))#.type(torch.bool))

    all_mask = [masks[one:one+1] for one in range(masks.shape[0])]

    if False:
        import matplotlib.pyplot as plt
        import os
        for one in range(len(all_mask)):
            fig = plt.figure(figsize=(6, 2), dpi=800)
            ax1 = fig.add_subplot(3, 1, 1)
            ax1.imshow(all_mask[one][0].detach().cpu())
            ax1.axis("off")
            ax1.set_title("ori masks", fontsize=5)

            ax1 = fig.add_subplot(3, 1, 2)
            ax1.imshow(silhouettes[one][0].detach().cpu())
            ax1.axis("off")
            ax1.set_title("silhouettes", fontsize=5)
            # import pdb;pdb.set_trace()
            ax1 = fig.add_subplot(3, 1, 3)
            ax1.imshow(depths[one][0].detach().cpu())
            ax1.axis("off")
            ax1.set_title("depths", fontsize=5)

            plt.savefig(f'/is/cluster/hyi/workspace/HCI/hdsr/mover_ori_repo_total3D/hdsr/mover_ori_repo/debug/debug_depth_oridinal_mask_{one}.png')
            # plt.show(block=False)
            plt.pause(1)
            plt.close()
    
    # import pdb;pdb.set_trace()
    # this function is used for solving human and 3D scene occlusion.
    # ! warning: use_for_human=True only for stage3; stage4 need adjust loss weight!!!
    loss = self.losses.compute_ordinal_depth_loss(all_mask, silhouettes, depths, use_for_human=use_for_human)
    return loss


# def get_contact_vertices_from_volume(self, ):
    
#     return contact_v, contact_vn

# ## contact loss: same logic, use all sample points.
# def compute_hsi_contact_loss(self, obj_verts_list, obj_faces_list, \
#                 contact_angle=None, contact_robustifier=None):
#     # in world coordinates system: this is better for calculating 3D bbox of each objects.
#     # input: objects in world CS; faces of each object.
#     # self.contact_volume has already been calculated.

#     # TODO: get contact_vertices and contact_vn from self.contact_volume
#     contact_body_vertices, contact_body_verts_normals = None, None

#     assert contact_angle is not None and \
#                 contact_robustifier is not None and ftov is not None
    
#     verts_list = obj_verts_list
#     faces_list = obj_faces_list
    
#     num_objs = len(verts_list)

#     # If only one person, return 0
#     contact_loss = torch.tensor(0., device=verts_list[0].device)
    
#     contact_cnt = 0
#     for j in range(num_objs):
#         # add human and object contact loss
#         # select contact vertices
#         scene_v = verts_list[j].unsqueeze(0)
#         scene_f = faces_list[j] # already has batch size
#         # from pytorch3d.structures import Meshes

#         import trimesh
#         meshes = trimesh.Trimesh(scene_v[0].detach().cpu().numpy(), scene_f[0].detach().cpu().numpy(), \
#                                     process=False, maintain_order=True)
#         scene_vn = torch.from_numpy(np.array(meshes.vertex_normals)).type_as(scene_v).unsqueeze(0)

#         import mover.dist_chamfer as ext
#         distChamfer = ext.chamferDist()
#         contact_dist, _, idx1, _ = distChamfer(
#             contact_body_vertices.contiguous(), scene_v)

#         # scene normals of the closest points on the scene surface to the contact vertices
#         contact_scene_normals = scene_vn[:, idx1.squeeze().to(
#             dtype=torch.long), :].squeeze()
        
#         # compute the angle between contact_verts normals and scene normals
#         angles = torch.asin(
#             torch.norm(torch.cross(contact_body_verts_normals, contact_scene_normals), 2, dim=1, keepdim=True)) *180 / np.pi

#         # consider only the vertices which their normals match 
#         valid_contact_mask = (angles.le(contact_angle) + angles.ge(180 - contact_angle)).ge(1)
#         valid_contact_ids = valid_contact_mask.squeeze().nonzero().squeeze()

#         contact_dist = contact_robustifier(contact_dist[:, valid_contact_ids].sqrt())

#         # ! Warning: exist none match, in 37 frame, it leads to None.
#         if contact_dist.shape[-1] != 0:
#             contact_loss = contact_loss + contact_dist.mean()
#             contact_cnt += 1

#     if contact_cnt == 0:
#         return contact_loss
#     else:
#         return contact_loss / contact_cnt
        