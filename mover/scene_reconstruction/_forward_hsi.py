from ._import_common import *

def forward_hsi(self, verts_list, verts_parallel_ground_list,
                resampled_verts_parallel_ground_list,
                smplx_model_vertices=None,
                op_conf=None, # used in gp estimation.
                loss_weights=None, stage=0, 
                contact_angle=None, contact_robustifier=None, 
                ftov=None, save_dir=None, obj_idx=-1, 
                ground_contact_vertices_ids=None,ground_contact_value=None, 
                img_list=None,
                ply_file_list=None,
                contact_file_list=None,
                detailed_obj_loss=False,
                debug=False,
                # assign body 2 obj
                USE_POSA_ESTIMATE_CAMERA=True,
                # # template_save_dir is different in estimate_camera and refine_scene 
                template_save_dir=None):

    if detailed_obj_loss:
        d_loss_dict = {}

    smplx_model_vertices = self.get_person_wrt_world_coordinates(smplx_model_vertices)

    # cam CS human verts
    # ! smplx_model_vertices is in world_CS
    smplx_model_vertices_cam = torch.transpose(torch.matmul(self.get_cam_extrin(), \
                torch.transpose(smplx_model_vertices, 2, 1)), 2, 1) 

    # smplx_model_vertices
    loss_dict = {}

    if DEBUG_LOSS:
        debug_loss_hsi_dict = {}
    else:
        debug_loss_hsi_dict = None

    ########################
    ### ! this is only used for optimize camera and ground plane.
    ########################
    if loss_weights is None or loss_weights["lw_gp_contact"] > 0: 
        if not USE_POSA_ESTIMATE_CAMERA: 
            ground_contact_value[ground_contact_value==0] = 1
            num_labels = ground_contact_value.sum()
            gp_contact_loss = self.robustifier(smplx_model_vertices[:, ground_contact_vertices_ids, 1] - self.ground_plane)
            gp_contact_loss = (ground_contact_value * gp_contact_loss.mean(-1)).sum() / ( num_labels + 1e-9)
            loss_dict['loss_gp_contact'] = gp_contact_loss

            gp_support_loss = F.relu(smplx_model_vertices[:, ground_contact_vertices_ids, 1] - self.ground_plane)
            gp_support_loss = (ground_contact_value * gp_support_loss.mean(-1)).sum() / ( num_labels + 1e-9)
            loss_dict['loss_gp_support'] = gp_support_loss
        else:
            if not self.input_feet_contact_flag:
                # add openpose filter out occluded body pose.
                if op_conf is not None:
                    feet_conf = op_conf[:, [19, 20, 21, 22, 23, 24]].mean(-1) > 0.5
                    filter_ply_file_list = [ply_file_list[tmp_i] for tmp_i in range(feet_conf.shape[0]) if feet_conf[tmp_i]>0]
                    filter_contact_file_list = [contact_file_list[tmp_i] for tmp_i in range(feet_conf.shape[0]) if feet_conf[tmp_i]]
                else:
                    filter_ply_file_list = ply_file_list
                    filter_contact_file_list = contact_file_list
                self.input_feet_contact_flag = self.assign_contact_body_to_objs(filter_ply_file_list, filter_contact_file_list, ftov, \
                                contact_parts='feet', \
                                debug=DEBUG_LOSS_OUTPUT, output_folder=template_save_dir)
            accumulate_contact_feet_vertices_world = self.get_person_wrt_world_coordinates(self.accumulate_contact_feet_vertices)
            
            # init for ground plane. 
            if not self.init_gp:
                self.init_gp = self.init_ground_plane(accumulate_contact_feet_vertices_world)

            if self.accumulate_contact_feet_vertices.shape[1] != 0:
                gp_contact_loss = self.robustifier( accumulate_contact_feet_vertices_world - self.ground_plane)
                gp_contact_loss = gp_contact_loss.mean()
                gp_support_loss = F.relu(accumulate_contact_feet_vertices_world - self.ground_plane).mean()
            else:
                gp_contact_loss = torch.zeros(1).cuda()
                gp_support_loss = torch.zeros(1).cuda()
                
            loss_dict['loss_gp_contact'] = gp_contact_loss
            loss_dict['loss_gp_support'] = gp_support_loss # not use yet, but add it on 0709.

    if loss_weights is None or ('lw_gp_normal' in loss_weights and loss_weights["lw_gp_normal"] > 0):
        feet_normal_world = self.get_person_wrt_world_coordinates(self.accumulate_contact_feet_verts_normals)
        # [0, 1, 0] -> camera orientation new_ori_one
        cos_distance = (feet_normal_world * torch.Tensor([[[0, 1, 0]]]).type_as(feet_normal_world)).sum(-1)
        feet_normal_dist = F.relu(1-cos_distance)**2
        if feet_normal_dist.shape[1] > 0:
            loss_dict['loss_gp_normal'] = feet_normal_dist.mean()
        else:
            loss_dict['loss_gp_normal'] = torch.zeros(1).cuda()



    ###########################################
    ### depth, collision, contact losses.
    ###########################################
    if loss_weights is None or loss_weights["lw_depth"] > 0: # will not be useful if we know the actual size of each object. 
        if not self.depth_template_flag:
            self.depth_template_flag = self.accumulate_ordinal_depth_from_human(smplx_model_vertices_cam, 
                img_list=img_list, debug=DEBUG_LOSS_OUTPUT, save_dir=save_dir)
            self.compute_overlap_region_on_obj(verts_list, self.faces_list, self.textures_list)
        
        if detailed_obj_loss:
            depth_loss, overlap_loss, depth_loss_list, overlap_loss_list, depth_fs_loss, depth_fs_loss_list = \
                self.compute_relative_depth_loss_range(verts_list, self.faces_list, self.textures_list, detailed=True)
            loss_dict['loss_depth'] = depth_loss
            loss_dict['loss_overlap'] = overlap_loss
            d_loss_dict['loss_depth'] = torch.stack(depth_loss_list)
            d_loss_dict['loss_overlap'] = torch.stack(overlap_loss_list)
            loss_dict['loss_fs_depth'] = depth_fs_loss
            d_loss_dict['loss_fs_depth'] = torch.stack(depth_fs_loss_list)
            
        else:
            depth_loss, overlap_loss, depth_fs_loss = self.compute_relative_depth_loss_range(verts_list, self.faces_list, self.textures_list)
            loss_dict['loss_depth'] = depth_loss
            loss_dict['loss_overlap'] = overlap_loss
            loss_dict['loss_fs_depth'] = depth_fs_loss

   
    if loss_weights is None or loss_weights["lw_sdf"] > 0:
        
        # only calculate once, will not change
        if not self.input_body_flag:
            # Body: smplx_model_vertices: N x J x 3
            body_verts_batch = smplx_model_vertices.shape[0]
            vertices_len = smplx_model_vertices.shape[1]
            body_verts_list = [smplx_model_vertices[one] for one in range(body_verts_batch)]
            body_faces_list = [self.faces_person_close_mouth.unsqueeze(0)]
            # save all body into one mesh
            body_vertices = [torch.cat(body_verts_list, 0)]
            body_faces = [torch.cat([body_faces_list[0]+idx*vertices_len for idx in range(body_verts_batch)], 1)]
            out_mesh = trimesh.Trimesh(body_vertices[0].cpu().numpy(), body_faces[0].squeeze().cpu().numpy(), process=False)
            template_save_fn = os.path.join(template_save_dir, 'all_body.obj')  
            out_mesh.export(template_save_fn)

            self.input_body_flag = self.load_whole_sdf_volume(ply_file_list, contact_file_list, output_folder=template_save_dir, debug=DEBUG_LOSS_OUTPUT)

        if self.resample_in_sdf:
            tmp_resampled_verts_parallel_ground_list = [one.contiguous() for one in resampled_verts_parallel_ground_list]

        sdf_loss, sdf_dict = self.compute_sdf_loss(tmp_resampled_verts_parallel_ground_list, output_folder=template_save_dir) #, detailed_obj_loss=True)

        loss_dict['loss_sdf'] = sdf_loss
        if detailed_obj_loss:
            d_loss_dict['loss_sdf'] = torch.stack(sdf_dict)

        if DEBUG_LOSS:
            debug_loss_hsi_dict['loss_sdf'] = sdf_dict

    # PROX contact loss
    if loss_weights is None or ('lw_contact' in loss_weights.keys() and  loss_weights["lw_contact"] > 0) \
            or ('lw_contact_coarse' in loss_weights.keys() and  loss_weights["lw_contact_coarse"] > 0) :

        # version 2: accumulate contact label
        if not self.input_body_contact_flag:
            # TODO: split into feet and body part
            self.input_body_contact_flag = self.load_contact_body_to_objs(ply_file_list, contact_file_list, ftov, \
                            debug=DEBUG_LOSS_OUTPUT, output_folder=template_save_dir, contact_parts='body')

        if USE_HAND_CONTACT_SPLIT:
            # add handArm contact with table.
            if not self.input_handArm_contact_flag:
                self.input_handArm_contact_flag = self.load_contact_body_to_objs(ply_file_list, contact_file_list, ftov, \
                                debug=DEBUG_LOSS_OUTPUT, output_folder=template_save_dir, contact_parts='handArm')
        
        # voxelize contact verts 
        if self.input_body_contact_flag and self.input_body_flag and not self.voxelize_flag :
            # import pdb;pdb.set_trace()
            dim = ((self.grid_max-self.grid_min)/self.voxel_size).long()[0].item()
            if self.accumulate_contact_body_vertices.shape[1] >0: # This does not work when body vertices shape = 0; N3Library_03375_02
                self.voxelize_flag=True
                # TODO: exists bugs. Voxelize it could improve the efficiency.
                # self.voxelize_flag = self.voxelize_contact_vertices(self.accumulate_contact_body_vertices, self.accumulate_contact_body_verts_normals, \
                #         self.voxel_size, self.grid_min, dim, \
                #         self.accumulate_contact_body_body2obj_idx,
                #         device=self.int_scales_object.device,
                #         debug=DEBUG_LOSS_OUTPUT, save_dir=template_save_dir)

        if loss_weights["lw_contact_coarse"] > 0:
            assert detailed_obj_loss == True
            # only works load body2obj tensor.
            contact_verts_ground_list, contact_vn_ground_list  = self.get_contact_verts_obj(verts_parallel_ground_list, self.faces_list, return_all=True)
            
            if obj_idx == -1: # optimize all objects in one model
                tmp_contact_coarse_loss = 0.0
                tmp_detailed_contact_coarse_list = []
                tmp_handArm_detailed_contact_coarse_list = []
                for tmp_obj_idx in range(len(contact_verts_ground_list)):
                    contact_coarse_loss, detailed_contact_coarse_list = self.compute_hsi_contact_loss_persubject_coarse( \
                                [contact_verts_ground_list[tmp_obj_idx]], [contact_vn_ground_list[tmp_obj_idx]], \
                                self.accumulate_contact_body_vertices, self.accumulate_contact_body_verts_normals, \
                                contact_body2obj_idx = (self.accumulate_contact_body_body2obj_idx == tmp_obj_idx), \
                                contact_angle=contact_angle, contact_robustifier=contact_robustifier, debug=DEBUG_CONTACT_LOSS, save_dir=save_dir)
                    tmp_detailed_contact_coarse_list.append(detailed_contact_coarse_list[0])
                    if USE_HAND_CONTACT_SPLIT and self.input_handArm_contact_flag:
                        handArm_contact_coarse_loss, handArm_detailed_contact_coarse_list = self.compute_hsi_contact_loss_persubject_coarse( \
                                    [contact_verts_ground_list[tmp_obj_idx]], [contact_vn_ground_list[tmp_obj_idx]], \
                                    self.accumulate_contact_handArm_vertices, self.accumulate_contact_handArm_verts_normals, \
                                    contact_body2obj_idx = (self.accumulate_contact_handArm_body2obj_idx == tmp_obj_idx), \
                                    contact_angle=contact_angle, contact_robustifier=contact_robustifier, debug=DEBUG_CONTACT_LOSS, save_dir=save_dir)
                        tmp_handArm_detailed_contact_coarse_list.append(handArm_detailed_contact_coarse_list[0])
                        tmp_contact_coarse_loss += contact_coarse_loss + handArm_contact_coarse_loss
                    else:
                        tmp_contact_coarse_loss += contact_coarse_loss
                loss_dict['loss_contact_coarse'] = tmp_contact_coarse_loss 
                if USE_HAND_CONTACT_SPLIT and self.input_handArm_contact_flag:
                    d_loss_dict['loss_contact_coarse'] = torch.stack(tmp_detailed_contact_coarse_list) + torch.stack(tmp_handArm_detailed_contact_coarse_list)
                else:
                    d_loss_dict['loss_contact_coarse'] = torch.stack(tmp_detailed_contact_coarse_list)

            if DEBUG_LOSS:
                debug_loss_hsi_dict['loss_contact_coarse_details'] = detailed_contact_coarse_list

    if detailed_obj_loss:
        return loss_dict, debug_loss_hsi_dict, d_loss_dict
    else:
        return loss_dict, debug_loss_hsi_dict, None
