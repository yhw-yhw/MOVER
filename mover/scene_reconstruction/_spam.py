def jitter_orientation(self, jitter_num):
    verts = self.get_verts_object()
    verts_list = []
    faces_list = []
    for one in range(len(self.idx_each_object)):
        obj_verts, obj_faces = self.get_one_verts_faces_obj(verts, one)
        # import pdb; pdb.set_trace()
        verts_list.append(obj_verts.squeeze(0))
        faces_list.append(obj_faces)
    # add edge loss
    
    tmp_dict = self.losses.compute_sil_loss(
            verts=verts_list, faces=faces_list, bboxes=self.det_results[:, :-1], edge_loss=True, debug=True,
        )
    sil_loss_list = tmp_dict['sil_loss_list']

    for i, sil_loss in enumerate(sil_loss_list):
        # import pdb;pdb.set_trace()
        if sil_loss > 0.03:
            ori_rot_obj_i = self.rotations_object[i].clone()
            ori_sil_loss_i = sil_loss
            min_loss_i = sil_loss
            for j_i in range(jitter_num):
                theta = math.pi * pi / jitter_num * j_i 
                self.rotations_object.data[i] = self.rotations_object[i] - theta
                jitter_sil_loss  = self.losses.compute_sil_loss_for_one_obj(verts=verts_list, faces=faces_list, \
                            bboxes=self.det_results[:, :-1], edge_loss=True, debug=True, idx=i)
                if jitter_sil_loss < min_loss_i:
                    pass
            # update a data tensor
            import pdb;pdb.set_trace()
            self.rotations_object.data[i] = self.rotations_object[i] - theta
    print(f'jitter with {theta}')

def jitter_orientation_on_sample_theta(self, theta):
    verts = self.get_verts_object()
    verts_list = []
    faces_list = []
    for one in range(len(self.idx_each_object)):
        obj_verts, obj_faces = self.get_one_verts_faces_obj(verts, one)
        # import pdb; pdb.set_trace()
        verts_list.append(obj_verts.squeeze(0))
        faces_list.append(obj_faces)
    # add edge loss
    
    tmp_dict = self.losses.compute_sil_loss(
            verts=verts_list, faces=faces_list, bboxes=self.det_results[:, :-1], edge_loss=True, debug=True,
        )
    sil_loss_list = tmp_dict['sil_loss_list']

    for i, sil_loss in enumerate(sil_loss_list):
        # TODO: hyper-parameters
        if sil_loss > 0.03:
            self.rotations_object.data[i] = self.rotations_object[i] - theta
    print(f'jitter with {theta}')


# def get_collision_objs_loss(self, verts):
    
#     bev_bbox = []
#     for one in range(len(self.idx_each_object)):
#         obj_verts, obj_faces = self.get_one_verts_faces_obj(verts, one)
#         bev_bbox.append()
#     bev_bbox = torch.vstack(bev_bbox)

#     intersect_label = compute_intersect(bev_bbox)
#     # get collision 3D BBox
#     collision_loss = 0
#     for i in range(intersect_label.shape[0]):
#         for j in range(intersect_label.shape[1]):
#             if interset_label[i,j] == False:
#                 # get sdf and calculate the collision loss
#                 pass

#     return collision_loss 


## resampled data

# self.params_person = nn.Parameter(params_person, requires_grad=True)
# if UPDATE_CAMERA_EXTRIN:
        #     self.rotate_cam_pitch_yoll = nn.Parameter(torch.zeros(1, 2).type_as(K_extrin), requires_grad=True)
        #     self.extra_extrin_mat = get_pitch_roll_euler_rotation_matrix(self.rotate_cam_pitch_yoll)
        #     self.K_extrin = torch.matmul(self.extra_extrin_mat, K_extrin)
        # else:
        #     self.K_extrin = nn.Parameter(K_extrin, requires_grad=False)


### __init__

# if stage == -1:
#     loss_dict = {}

#     verts = self.get_verts_object()
#     verts_parallel_ground = self.get_verts_object_parallel_ground()
#     proj_xy, z = self.apply_projection_to_image(verts)
#     compute_offscreen_loss = 0
#     # for one in range(len(self.idx_each_object)):
#     one = obj_idx
#     if one == 0:
#         start = 0
#         end = self.idx_each_object[one]
#     else:
#         start = self.idx_each_object[one-1] 
#         end = self.idx_each_object[one]
#     proj_xy_one = proj_xy[:, start:end, :]
#     z_one = z[:, start:end, :]

#     # TODO: differentible rendering loss constraint (if has mask)
#     x_min = torch.min(proj_xy_one[:, :, 0], 1)[0] 
#     y_min = torch.min(proj_xy_one[:, :, 1], 1)[0]
#     x_max = torch.max(proj_xy_one[:, :, 0], 1)[0]
#     y_max = torch.max(proj_xy_one[:, :, 1], 1)[0]
    
#     zero = torch.zeros_like(x_min)
#     h = torch.zeros_like(y_min) + self.height
#     w = torch.zeros_like(y_max) + self.width
    
#     # logger.info(f'proj bbox {one}:')
#     # print(proj_bbox_one)
#     # compute_offscreen_loss
#     zeros = torch.zeros_like(z_one)
#     max_wh = torch.ones_like(proj_xy_one)
#     max_wh[:, :, 0] = max_wh[:, :, 0] * self.width
#     max_wh[:, :, 1] = max_wh[:, :, 1] * self.height
#     lower_right = torch.max(proj_xy_one - max_wh, zeros).sum(dim=(1, 2))
#     upper_left = torch.max(-proj_xy_one, zeros).sum(dim=(1, 2))
#     behind = torch.max(-z_one, zeros).sum(dim=(1, 2))
#     # import pdb;pdb.set_trace()
#     compute_offscreen_loss = compute_offscreen_loss + (lower_right + upper_left + behind)

#     if self.USE_MASK:
#         if loss_weights is None or loss_weights["lw_sil"] > 0:
#             verts_list = []
#             faces_list = []
#             for _ in range(self.rotations_object.shape[0]):
#                 one=obj_idx
#                 obj_verts, obj_faces = self.get_one_verts_faces_obj(verts, one)
#                 verts_list.append(obj_verts.squeeze(0))
#                 faces_list.append(obj_faces)
#             # add edge loss
#             loss_dict.update(
#                 self.losses.compute_sil_loss_for_one_obj(verts=verts_list, faces=faces_list, \
#                         bboxes=self.det_results[:, :-1], edge_loss=True, debug=False, idx=one)
#             )
#     if loss_weights is None or loss_weights["lw_offscreen"] > 0:
#         loss_dict["loss_offscreen"] = compute_offscreen_loss

#     if loss_weights is None or loss_weights["lw_scale"] > 0:
#         if self.USE_ONE_DOF_SCALE:
#             # import pdb;pdb.set_trace()
#             loss_dict["loss_scale"] = self.losses.compute_intrinsic_scale_prior(
#                 intrinsic_scales=self.int_scales_object[:,0:1],
#                 intrinsic_mean=self.int_scale_object_mean[:,0:1],
#             )
#         else:
#             loss_dict["loss_scale"] = self.losses.compute_intrinsic_scale_prior(
#                 intrinsic_scales=self.int_scales_object,
#                 intrinsic_mean=self.int_scale_object_mean,
#             )

#     return loss_dict


# if self.ground_plane == 0:
#     # jointly optimize could be much better !!!
#     for i in range(ground_objs.shape[0]):
#         loss_ground_objs = loss_ground_objs + mse_loss(ground_objs, ground_objs[i].repeat(ground_objs.shape).detach())
#     loss_ground_obs = loss_ground_objs / (ground_objs.shape[0]-1)
# else:

# For scene initialization, not use it.
    # # add depth ordinal loss, need mask
    # if loss_weights is None or ('lw_depth' in loss_weights.keys() and loss_weights["lw_depth"] > 0):
    #     loss_dict.update(self.compute_ordinal_depth_loss())

# if  loss_weights is None or (self.UPDATE_CAMERA_EXTRIN and 'lw_cam_l1' in loss_weights.keys() and loss_weights["lw_cam_l1"] > 0):
#                 loss_dict['loss_cam_l1'] = torch.norm(self.rotate_cam_pitch, p=1, dim=-1) +  torch.norm(self.rotate_cam_pitch, p=1, dim=-1) #smooth_l1_loss(self.rotate_cam_pitch_yoll)

# if loss_weights is None or ('lw_scale_with_size' in loss_weights.keys() and loss_weights["lw_scale_with_size"] > 0):
                
#     # import pdb;pdb.set_trace()
#     objs_size = self.ori_objs_size * self.int_scales_object

#     gt_size_objs = []
#     for one in self.size_cls: # index in NYU40CLASSES
#         gt_size_objs.append(SIZE_FOR_DIFFERENT_CLASS[NYU40CLASSES[one]])
#     gt_size_objs = torch.Tensor(gt_size_objs).type_as(objs_size).detach()

#     loss_dict["loss_scale_with_size"] = mse_loss(
#         objs_size, gt_size_objs
#     )


# TODO: delete: 
# def get_one_verts_faces_obj(self, verts, idx, texture=False):
#     # verts = verts.squeeze(0)
#     if idx == 0:
#         start_p = 0
#         end_p = self.idx_each_object[idx]
#         start_f = 0
#         end_f = self.idx_each_object_face[idx]
#     else:
#         start_p = self.idx_each_object[idx-1] 
#         end_p = self.idx_each_object[idx]
#         start_f = self.idx_each_object_face[idx-1] 
#         end_f = self.idx_each_object_face[idx]
#     # import pdb;pdb.set_trace()
#     obj_verts = verts[:, start_p:end_p]
#     face_type = self.faces.type()
#     obj_faces = (self.faces[:, start_f:end_f] - start_p).type(face_type)
#     if not texture:
#         return obj_verts, obj_faces
#     else:
#         obj_textures = self.textures_object[:, start_f:end_f]
#         return obj_verts, obj_faces, obj_textures


# elif stage == 30: # get ordinal depth decision.
#             with torch.no_grad():
#                 batch_size = smplx_model_vertices.shape[0]
#                 ordinal_depth_loss_list = []
#                 for idx in range(batch_size):
#                     tmp_ordinal_depth_loss = self.compute_ordinal_depth_loss_perframe(smplx_model_vertices[idx:idx+1], idx)['loss_depth']
#                     ordinal_depth_loss_list.append(tmp_ordinal_depth_loss.item())
#                 return ordinal_depth_loss_list


# elif stage == 3:
#             print('OPT Human & Scene')
#             loss_dict = {}
#             # assert smplx_model_vertices.shape
#             batch_size = smplx_model_vertices.shape[0]
#             if debug:
#                 debug_loss_hsi_dict = {'loss_depth': [], 
#                                         'loss_sdf': [],
#                                         'loss_prox_contact': []
#                                     }
#             else:
#                 debug_loss_hsi_dict = None

#             if loss_weights is None or loss_weights["lw_gp_contact"] > 0:

#                 ground_contact_value[ground_contact_value==0] = 0 #-0.2
#                 gp_contact_loss = torch.abs(smplx_model_vertices[:, ground_contact_vertices_ids, 1] - self.ground_plane) 
#                 # import pdb;pdb.set_trace()
#                 gp_contact_loss = (ground_contact_value * gp_contact_loss.mean(-1)).sum() / (ground_contact_value.sum() + 1e-7)
#                 loss_dict['loss_gp_contact'] = gp_contact_loss

#             if loss_weights is None or loss_weights["lw_depth"] > 0:
#                 depth_loss = torch.tensor(0., device=self.device)
#                 cnt = 0
#                 for idx in range(batch_size):
#                     # smplx_model_verts_cam = torch.transpose(torch.matmul(self.get_cam_extrin(), \
#                     #     torch.transpose(smplx_model_vertices[idx:idx+1], 2, 1)), 2, 1) 
                    
#                     tmp_ordinal_depth_loss = self.compute_ordinal_depth_loss_perframe(smplx_model_vertices[idx:idx+1], idx)['loss_depth']

#                     if debug:
#                         debug_loss_hsi_dict['loss_depth'].append(tmp_ordinal_depth_loss)

#                     if tmp_ordinal_depth_loss > 0:
#                         cnt += 1
#                     depth_loss = depth_loss + tmp_ordinal_depth_loss # ! warning: confidence should come from detection
#                 depth_loss = depth_loss / (cnt + 1e-9)

#                 if depth_robustifier is not None:
#                     squared_res = depth_loss ** 2
#                     depth_loss = torch.div(squared_res, squared_res + depth_robustifier ** 2)
                
#                 loss_dict['loss_depth'] = depth_loss

#             if loss_weights is None or loss_weights["lw_sdf"] > 0:
#                 sdf_penetration_loss = torch.tensor(0., device=smplx_model_vertices[0].device)
#                 cnt = 0
#                 for idx in range(batch_size):
#                     tmp_sdf_loss = self.forward(smplx_model_vertices[idx:idx+1], stage=1)
#                     if debug:
#                         debug_loss_hsi_dict['loss_sdf'].append(tmp_sdf_loss)
                        
#                     if tmp_sdf_loss > 0:
#                         sdf_penetration_loss = sdf_penetration_loss + body2scene_conf[idx] * tmp_sdf_loss
#                         cnt += 1

#                 loss_dict['loss_sdf'] = sdf_penetration_loss / (cnt + 1e-9)

#             # PROX contact loss
#             if loss_weights is None or loss_weights["lw_prox_contact"] > 0:
#                 prox_contact_loss = torch.tensor(0., device=smplx_model_vertices[0].device)
#                 cnt = 0
#                 for idx in range(batch_size): 
#                     tmp_prox_contact_loss = self.forward(smplx_model_vertices[idx:idx+1], stage=2, contact_verts_ids=contact_verts_ids,
#                                          contact_angle=contact_angle, contact_robustifier=contact_robustifier, ftov=ftov)
#                     if debug:
#                         debug_loss_hsi_dict['loss_prox_contact'].append(tmp_prox_contact_loss)

#                     if tmp_prox_contact_loss > 0:
                        
#                         cnt += 1
#                     prox_contact_loss = prox_contact_loss + body2scene_conf[idx] * tmp_prox_contact_loss 
#                 loss_dict['loss_prox_contact'] = prox_contact_loss / (cnt + 1e-9)

#             return loss_dict, debug_loss_hsi_dict


# if depth_robustifier is not None:
#     squared_res = depth_loss ** 2
#     depth_loss = torch.div(squared_res, squared_res + depth_robustifier ** 2)


## stage 1:
# # lr_list = [2e-3, 2e-4]
#         lr_list = [1e-2, 5e-3]
#         num_iterations_list = [800, 200]
#         pid_list = [12, 13]

#         for lr_idx, opt_lr in enumerate(lr_list):
#             min_loss = math.inf
#             optimizer = torch.optim.Adam(model.parameters(), lr=opt_lr)
#             pid = pid_list[lr_idx]
#             loss_weights = DEFAULT_LOSS_WEIGHTS[f'stage0_{pid}']['loss_weight']

## in human pose estimation file
# if TB_DEBUG and False:
#                 from body_models.video_smplifyx.tf_utils import save_images
#                 save_images(tb_logger, 'after Joint', tmp_result, 0)
                
#             if scene_viz and False:
#                 # render to image
#                 if image.max() > 1:
#                     image = image / 255.0
#                 rend, mask = model.render()
#                 h, w, c = image.shape
#                 L = max(h, w)
#                 new_image = np.pad(image.copy(), ((0, L - h), (0, L - w), (0, 0)))
#                 new_image[mask] = rend[mask]
#                 new_image = (new_image[:h, :w] * 255).astype(np.uint8)
#                 plt.imshow(new_image)
#                 plt.show(block=False)
#                 plt.pause(0.5)
#                 plt.close()
#                 # render to 3D space in camera coordinates system
#                 model.interactive_op3d_render(image)