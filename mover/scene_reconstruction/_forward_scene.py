##########################
###  Only OPT 3D Scene
##########################
from ._import_common import *
from mover.constants import (BBOX_HEIGHT_CONSTRAINTS,DEBUG)

def forward_scene(self, verts, verts_parallel_ground, verts_list, verts_parallel_ground_list,
                resampled_verts_parallel_ground_list,
                loss_weights=None,
                scene_viz=False, save_dir=None,
                detailed_obj_loss=False,
                ):
    
    if detailed_obj_loss:
        d_offscreen_loss_list = []
        d_bbox_loss_list = []

    proj_xy, z = self.apply_projection_to_image(verts)
    proj_bbox = []
    loss_dict = {}

    compute_offscreen_loss = 0
    ground_objs = []
    for one in range(len(self.idx_each_object)):
        if one == 0:
            start = 0
            end = self.idx_each_object[one]
        else:
            start = self.idx_each_object[one-1] 
            end = self.idx_each_object[one]
        proj_xy_one = proj_xy[:, start:end, :]
        z_one = z[:, start:end, :]

        # TODO: differentible rendering loss constraint (if has mask)
        x_min = torch.min(proj_xy_one[:, :, 0], 1)[0] 
        y_min = torch.min(proj_xy_one[:, :, 1], 1)[0]
        x_max = torch.max(proj_xy_one[:, :, 0], 1)[0]
        y_max = torch.max(proj_xy_one[:, :, 1], 1)[0]
        
        zero = torch.zeros_like(x_min)
        h = torch.zeros_like(y_min) + self.height
        w = torch.zeros_like(y_max) + self.width
        
        # proj_bbox_one = torch.stack((torch.max(x_min, zero), torch.max(y_min, zero),
        #             torch.min(x_max, w), torch.min(y_max, h)), 1)
        proj_bbox_one = torch.stack((x_min, y_min, x_max, y_max), 1)
        proj_bbox.append(proj_bbox_one)
        # logger.info(f'proj bbox {one}:')
        # print(proj_bbox_one)

        # * TODO: remove this, by get biggest edge when the projected object is outside.
        # compute_offscreen_loss
        zeros = torch.zeros_like(z_one)
        max_wh = torch.ones_like(proj_xy_one)
        max_wh[:, :, 0] = max_wh[:, :, 0] * self.width
        max_wh[:, :, 1] = max_wh[:, :, 1] * self.height
        lower_right = torch.max(proj_xy_one - max_wh, zeros).sum(dim=(1, 2))
        upper_left = torch.max(-proj_xy_one, zeros).sum(dim=(1, 2))
        behind = torch.max(-z_one, zeros).sum(dim=(1, 2))
        # TODO: add far penalty.
        # far = 
        FAR= torch.ones_like(z_one) * 8.0
        too_far = torch.max(z_one - FAR, zeros).sum(dim=(1, 2))

        compute_offscreen_loss = compute_offscreen_loss + (lower_right + upper_left + behind + too_far)
        if detailed_obj_loss:
            d_offscreen_loss_list.append((lower_right + upper_left + behind+ too_far))
        # calculate ground plane
        obj_ground = verts_parallel_ground[:, start:end, 1].max()
        ground_objs.append(obj_ground)

    # import pdb;pdb.set_trace()
    # ground plane support loss:    
    ground_objs = torch.stack(ground_objs)
    loss_ground_objs= 0
    loss_ground_objs = torch.abs(ground_objs-self.ground_plane.repeat(ground_objs.shape)).mean()

    # offscreen loss
    compute_offscreen_loss = compute_offscreen_loss / len(self.idx_each_object)

    # proj bbox loss
    det_bbox = torch.stack((self.det_results[:, 0], self.det_results[:, 1], 
                self.det_results[:, 2],
                self.det_results[:, 3]), 1)

    # original bbox loss: 
    # bbox_loss = smooth_l1_loss(torch.cat(proj_bbox), det_bbox, reduction='none').mean(-1)
    # import pdb;pdb.set_trace()
    # modified bbox loss: left_top, right_top
    # import pdb;pdb.set_trace()
    tmp_prob_box = torch.cat(proj_bbox)
    tmp_xywidth = torch.stack([tmp_prob_box[:, 0], tmp_prob_box[:, 1], tmp_prob_box[:, 2]-tmp_prob_box[:, 0]], -1)
    tmp_det_xywidth = torch.stack((self.det_results[:, 0], self.det_results[:, 1], 
                self.det_results[:, 2]-self.det_results[:, 0]), 1)
    # bbox_loss = smooth_l1_loss(tmp_xywidth, tmp_det_xywidth, reduction='none').mean(-1) # * previous 09.08.
    bbox_loss = smooth_l1_loss(tmp_xywidth, tmp_det_xywidth, reduction='none').sum(-1) #/ 640.0 # normalized to 1.0
    # bbox_loss = mse_loss(tmp_xywidth, tmp_det_xywidth, reduction='none').sum(-1) #/ 640.0 # normalized to 1.0
    # height_loss = smooth_l1_loss(tmp_prob_box[:, 3]-tmp_prob_box[:, 1], \
    #         self.det_results[:, 3]-self.det_results[:, 1], reduction='none').sum(-1)
        

    if BBOX_HEIGHT_CONSTRAINTS:
        tmp_height = torch.stack([tmp_prob_box[:, 3]-tmp_prob_box[:, 1]], -1)
        tmp_det_height = torch.stack((self.det_results[:, 3]-self.det_results[:, 1], ), 1)
        bbox_loss = bbox_loss + smooth_l1_loss(tmp_height, tmp_det_height, reduction='none').sum(-1)

    if detailed_obj_loss:
        d_bbox_loss_list = bbox_loss * (self.det_score > 0.0).type(bbox_loss.type())

    # ! add detection score as reweight: it do actually eliminate wrong bbox results.
    # import pdb;pdb.set_trace()
    bbox_loss = (bbox_loss * (self.det_score > 0.0).type(bbox_loss.type())).mean() 
    # bbox_loss = (bbox_loss * self.det_score).mean()

    # Add silhouettes loss given mask
    if self.USE_MASK:
        if loss_weights is None or ("lw_sil" in loss_weights and loss_weights["lw_sil"] > 0):
            if detailed_obj_loss:
                tmp_sil_loss_dict = self.losses.compute_sil_loss(
                        verts=verts_list, faces=self.faces_list, bboxes=self.det_results[:, :-1], debug=True, det_score=self.det_score,
                    )
                d_edge_loss_list = tmp_sil_loss_dict['edge_loss_list']
                d_sil_loss_list = tmp_sil_loss_dict['sil_loss_list']
            else:
                loss_dict.update(
                    self.losses.compute_sil_loss(
                        verts=verts_list, faces=self.faces_list, bboxes=self.det_results[:, :-1], det_score=self.det_score,
                    )
                )

            if scene_viz:
                # all_mask = None
                # import pdb;pdb.set_trace()

                # assert img_list is not None
                # static_img = get_image(img_list[0])
                # height, width = static_img.shape[0], static_img.shape[1]

                for one in range(len(self.idx_each_object)):
                    fig = plt.figure(figsize=(6, 4), dpi=800)
                    ax1 = fig.add_subplot(3, 2, 1)
                    # tmp_target_mask = self.target_masks[one].cpu().clone()
                    # tmp_target_mask[:height, :width, :] =  tmp_target_mask[:height, :width, :] * 0.5 +  \
                    #                 static_img * 0.5

                    ax1.imshow(self.target_masks[one].cpu())
                    # ax1.imshow(tmp_target_mask)
                    ax1.axis("off")
                    ax1.set_title("target masks", fontsize=5)

                    # if all_mask is None:
                    #     all_mask = self.target_masks[one].clone()
                    # else:
                    #     all_mask = all_mask & self.target_masks[one]

                    ax1 = fig.add_subplot(3, 2, 2)
                    ax1.imshow(self.ref_mask[one].cpu())
                    ax1.axis("off")
                    ax1.set_title("ref_mask, target_masks > 0", fontsize=5)
                    # import pdb;pdb.set_trace()
                    ax1 = fig.add_subplot(3, 2, 3)
                    ax1.imshow(self.keep_mask[one].cpu())
                    ax1.axis("off")
                    ax1.set_title("keep_mask, target_masks >= 0", fontsize=5)

                    # import pdb;pdb.set_trace()
                    render_img = self.losses.renderer(verts_list[one], self.faces_list[one], K=self.K_rois[one], mode="silhouettes")
                    ax2 = fig.add_subplot(3, 2, 4)
                    ax2.imshow(render_img.detach().cpu()[0])
                    ax2.set_title(f"Render", fontsize=5)
                    ax2.axis("off")

                    image = self.keep_mask[one] * render_img
                    ax2 = fig.add_subplot(3, 2, 5)
                    ax2.imshow(image.detach().cpu()[0])
                    ax2.set_title(f"keep_weighted_render", fontsize=5)
                    ax2.axis("off")

                    l_m = (image - self.ref_mask[one]) ** 2
                    ax2 = fig.add_subplot(3, 2, 6)
                    ax2.imshow(l_m.detach().cpu()[0])
                    ax2.set_title(f"Effective Error", fontsize=5)
                    
                    ax2.axis("off")
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(f'{save_dir}/debug_render_mask_{one}.png')
                    # plt.show(block=False)
                    # plt.pause(1)
                    plt.close()
    
    # calculate loss for each objects.
    if detailed_obj_loss:
        # import pdb;pdb.set_trace()
        d_loss_dict = {}
        d_loss_dict['loss_proj_bbox'] = d_bbox_loss_list
        d_loss_dict['loss_offscreen'] = torch.cat(d_offscreen_loss_list)
        if "lw_sil" in loss_weights and loss_weights["lw_sil"] > 0:
            d_loss_dict['loss_edge'] = torch.stack(d_edge_loss_list)
            d_loss_dict['loss_sil'] = torch.stack(d_sil_loss_list)
        d_loss_dict["loss_scale"] = self.losses.compute_intrinsic_scale_prior(
                intrinsic_scales=self.get_scale_object(),
                intrinsic_mean=self.init_int_scales_object,
                reduce=True
            )
        d_loss_dict["loss_orientation_penalty"] = ((self.rotations_object-self.init_rotations_object) ** 2).squeeze(-1)
        
    if DEBUG:
        ## debug for projection bbox
        self.debug_project_bbox()

    if loss_weights is None or ("lw_proj_bbox" in loss_weights and loss_weights["lw_proj_bbox"] > 0):
        loss_dict["loss_proj_bbox"] = bbox_loss
    
    if loss_weights is None or ("lw_offscreen" in loss_weights and loss_weights["lw_offscreen"] > 0):
        loss_dict["loss_offscreen"] = compute_offscreen_loss
    
    # first fixed scale, when we get good silhouette; update scale with human;
    if loss_weights is None or ("lw_scale" in loss_weights and loss_weights["lw_scale"] > 0): 
        # TODO: add scale prior
        if self.USE_ONE_DOF_SCALE:
            # import pdb;pdb.set_trace()
            loss_dict["loss_scale"] = self.losses.compute_intrinsic_scale_prior(
                intrinsic_scales=self.get_scale_object()[:,0:1],
                intrinsic_mean=self.init_int_scales_object[:,0:1]#self.int_scale_object_mean[:,0:1],
            )
        else:
            loss_dict["loss_scale"] = self.losses.compute_intrinsic_scale_prior(
                intrinsic_scales=self.get_scale_object(),
                intrinsic_mean=self.init_int_scales_object#self.int_scale_object_mean,
            )
    
    if loss_weights is None or ("lw_ground_objs" in loss_weights and loss_weights["lw_ground_objs"] > 0):
        loss_dict["loss_ground_objs"] = loss_ground_objs

    # ! object collision losses: verts_list [nx3, ....]
    if loss_weights is None or ("lw_collision_objs" in loss_weights and loss_weights["lw_collision_objs"] > 0):
        tmp_verts_parallel_ground_list = [one.squeeze(0).contiguous() for one in verts_parallel_ground_list]
        tmp_resampled_verts_parallel_ground_list = [one.squeeze(0).contiguous() for one in resampled_verts_parallel_ground_list]
        loss_collision_objs = self.collision_loss(tmp_verts_parallel_ground_list, self.faces_list, \
                    sampled_verts_list=tmp_resampled_verts_parallel_ground_list)
        loss_dict["loss_collision_objs"] = loss_collision_objs
    
    # orientation penalty
    if  loss_weights is None or ('lw_orientation_penalty' in loss_weights.keys() and loss_weights["lw_orientation_penalty"] > 0):
        loss_orientation_penalty = torch.mean((self.rotations_object-self.init_rotations_object) ** 2)
        loss_dict["loss_orientation_penalty"] = loss_orientation_penalty

    if detailed_obj_loss:
        return loss_dict, d_loss_dict
    else:
        return loss_dict, None