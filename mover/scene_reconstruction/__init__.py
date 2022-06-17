
from ._import_common import *
from mover.constants import (
    BBOX_EXPANSION,
    BBOX_EXPANSION_PARTS,
    IMAGE_SIZE,
    REND_SIZE,
    SMPL_FACES_PATH, # SMPLX path and SMPLX with closed mouth path
    # USE_POSA_ESTIMATE_CAMERA,
    DEBUG_LOSS,
    DEBUG_LOSS_OUTPUT,
    DEBUG_DEPTH_LOSS,
    DEBUG_CONTACT_LOSS,
    DEBUG,
    NYU40CLASSES,
    SIZE_FOR_DIFFERENT_CLASS,
    LOSS_NORMALIZE_PER_ELEMENT,
    # SIGMOID_FOR_SCALE,
    USE_HAND_CONTACT_SPLIT,
    BBOX_HEIGHT_CONSTRAINTS,
    PYTORCH3D_DEPTH,
)

from mover.utils.meshviewer import *
from mover.loss import Losses
from mover.utils.pytorch3d_rotation_conversions import euler_angles_to_matrix
# from sdf import SDFLossObjs
from mover.utils.util_spam import project_bbox, get_faces_and_textures

from thirdparty.body_models.smplifyx.utils_mics import misc_utils

class HSR(nn.Module):
    def __init__(
        self,
        image,
        det_results,
        size_cls,
        ori_objs_size,
        translations_object,
        rotations_object,
        size_scale_object,
        verts_object_og,
        idx_each_object,
        faces_object,
        idx_each_object_face,
        K_rois,
        K_intrin,
        K_extrin,
        cams_params,
        cams_person,
        verts_person_og,
        faces_person,
        params_person,
        masks_object,
        masks_person,
        target_masks,
        perframe_masks,
        perframe_det_results,
        perframe_cam_rois,
        labels_person,
        labels_object,
        interaction_map_parts,
        # ! warning for total3D
        basis_object=None, 
        int_scale_init=1.0,
        image_height=360,
        image_width=640, 
        inner_robust_sdf=None, # in real use, we use inner_robust_sdf=0.2
        # for mask loss
        lw_chamfer=0.5,
        kernel_size=7,
        power=0.25,
        ground_plane=None,
        ground_contact_vertices_ids=None,
        cluster=False,
        resample_in_sdf=False,
        # setting from cfg
        USE_MASK=True,
        UPDATE_CAMERA_EXTRIN=True,
        USE_ONE_DOF_SCALE=True,
        UPDATE_OBJ_SCALE=True,
        contact_idxs=None, # contact info
        contact_idx_each_obj=None,
        RECALCULATE_HCI_INFO=False,
        SIGMOID_FOR_SCALE=True,
        ALL_OBJ_ON_THE_GROUND=True,
        CONTACT_MSE=False,
        constraint_scale_for_chair=False,
        chair_scale=0.8
    ):
        super(HSR, self).__init__()
        
        '''
        cam parameters:

        object parameters:
            translations_object: x,y,z translation
            rotations_object: y-rotation
            size_scale_object: x,y,z scale
            verts_object_og,
            faces_object,
            k_rois: for each object mask loss
        human parameters:
            cams_person: same as image camera intrinsic
            verts_person_og,
            faces_person,
            params_person: global orientation, body pose, global translation;
        human & scene contact information parameters:
            labels_person,
            labels_object,
            interaction_map_parts,
            int_scale_init=1.0,
        '''

        # setting from cfg_utils
        self.USE_MASK = USE_MASK
        self.UPDATE_CAMERA_EXTRIN = UPDATE_CAMERA_EXTRIN
        self.USE_ONE_DOF_SCALE = USE_ONE_DOF_SCALE
        self.UPDATE_OBJ_SCALE = UPDATE_OBJ_SCALE
        self.resample_in_sdf = resample_in_sdf
        self.image = image
        self.height = image_height
        self.width = image_width
        # ! warning: for visualization
        self.cluster = cluster
        self.use_sigmoid_for_scale = SIGMOID_FOR_SCALE
        self.constraint_scale_for_chair = constraint_scale_for_chair
        self.chair_scale = chair_scale
        self.ALL_OBJ_ON_THE_GROUND = ALL_OBJ_ON_THE_GROUND

        # ! for contact_coarse loss
        self.CONTACT_MSE = CONTACT_MSE
        
        translation_init = translations_object.detach().clone()
        self.translations_object = nn.Parameter(translation_init, requires_grad=True)
        rotations_object = rotations_object.detach().clone()
        self.rotations_object = nn.Parameter(rotations_object, requires_grad=True)
        self.register_buffer("basis_object", basis_object)

        K_extrin = K_extrin.detach().clone() 
        self.K_extrin = nn.Parameter(K_extrin, requires_grad=False)
        
        self.rotate_cam_pitch = nn.Parameter(torch.zeros(1, 1).type_as(K_extrin), requires_grad=self.UPDATE_CAMERA_EXTRIN)
        self.rotate_cam_roll = nn.Parameter(torch.zeros(1, 1).type_as(K_extrin), requires_grad=self.UPDATE_CAMERA_EXTRIN)
        if cams_params is not None:
            # previous define wrong
            self.rotate_cam_pitch.data.copy_(torch.tensor(cams_params[0])) # 1
            self.rotate_cam_roll.data.copy_(torch.tensor(cams_params[1])) #0
         
        # before 0709, the scale is initialized with 1.0;
        # self.int_scales_object = nn.Parameter(
        #     int_scale_init * torch.ones((rotations_object.shape[0], 3)).float().cuda(),
        #     requires_grad=self.UPDATE_OBJ_SCALE,
        # ) # scale on x,y,z
        size_scale_object = size_scale_object.detach().clone()
        self.int_scales_object = nn.Parameter(size_scale_object, requires_grad=self.UPDATE_OBJ_SCALE)
        
        self.int_scale_object_mean = nn.Parameter(
            int_scale_init * torch.ones((rotations_object.shape[0], 3)).float().cuda(), requires_grad=False
        )
        self.init_int_scales_object=torch.clone(self.int_scale_object_mean.detach())

        # TODO: ground plane 
        if ground_plane is not None: 
            self.ground_plane = nn.Parameter(ground_plane, requires_grad=True)
        else:
            self.ground_plane = nn.Parameter(torch.ones(1).type_as(K_extrin), requires_grad=True)
        
        # if ground_contact_vertices_ids is not None:
        #     ground_contact_vertices_ids = load_feet_gp_contact(opt.body_segments_dir)
        #     self.register_buffer('ground_contact_vertices_ids', ground_contact_vertices_ids)

        ## World CS
        self.register_buffer("verts_object_og", verts_object_og) 
        self.idx_each_object = idx_each_object.detach().cpu().numpy().astype(int)
        self.idx_each_object_face = idx_each_object_face.detach().cpu().numpy().astype(int)
        self.register_buffer("verts_person_og", verts_person_og)

        ## Contact verts idx_each_object_face
        # local idx for each object.
        if contact_idxs is not None:
            self.register_buffer("contact_idxs", contact_idxs) # shape (-1)
            self.contact_idx_each_obj=contact_idx_each_obj.detach().cpu().numpy().astype(int)

        # detection
        self.register_buffer("det_results", det_results)
        # add detection score
        assert self.det_results.shape[-1] == 5
        self.register_buffer("det_score", self.det_results[:, -1])
        if self.USE_MASK:
            # For mask: WARNINGS: register_parameters,
            self.register_buffer("target_masks", (target_masks).float())
            self.register_buffer("ref_mask", (target_masks > 0).float())
            self.register_buffer("keep_mask", (target_masks >= 0).float())
            self.register_buffer("masks_human", masks_person)
            self.register_buffer("masks_object", masks_object)
            self.perframe_masks = perframe_masks
            self.perframe_det_results = perframe_det_results
            self.perframe_cam_rois = perframe_cam_rois

        else:
            self.register_buffer("ref_mask", None)
            self.register_buffer("keep_mask", None)


        self.register_buffer("size_cls", size_cls) # TO used for calculate scale loss
        self.register_buffer("ori_objs_size", ori_objs_size)

        self.register_buffer("cams_person", cams_person)
        self.register_buffer("K_rois", K_rois)
        self.register_buffer("K_intrin", K_intrin)

        self.register_buffer("faces_object", faces_object.unsqueeze(0))
        self.register_buffer(
            "textures_object", torch.ones(1, len(faces_object), 1, 1, 1, 3)
        )
        self.register_buffer("faces_person", faces_person)
        self.register_buffer(
            "textures_person", torch.ones(1, len(faces_person), 1, 1, 1, 3)
        )
        # define faces_person_close_mouth
        # use fixed faces
        print(os.path.join(os.path.dirname(__file__), 
                    '../../data/body_templates/closed_mouth.obj'))
        close_mouth_model = trimesh.load_mesh(os.path.join(os.path.dirname(__file__), 
                    '../../data/body_templates/closed_mouth.obj'), file_type='obj', process=False)
        faces_person_close_mouth = torch.IntTensor(close_mouth_model.faces) 
        self.register_buffer('faces_person_close_mouth', faces_person_close_mouth)
        
        
        ## define render
        R = torch.cuda.FloatTensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
        t = torch.zeros(1, 3).cuda()
        self.renderer = nr.renderer.Renderer(
            image_size=IMAGE_SIZE, K=self.K_intrin, R=R, t=t, orig_size=IMAGE_SIZE, #TODO: 1 IMAGE_SIZE
        )
        self.renderer.light_direction = [1, 0.5, 1]
        self.renderer.light_intensity_direction = 0.3
        self.renderer.light_intensity_ambient = 0.5
        self.renderer.background_color = [1, 1, 1]

        if PYTORCH3D_DEPTH: # get the farrest depth map.
            # self.init_pytorch3d_render(self.K_intrin, image_size=[(int(IMAGE_SIZE*9/16), IMAGE_SIZE)])
            self.init_pytorch3d_render(self.K_intrin, image_size=[(IMAGE_SIZE, IMAGE_SIZE)])
        else:            
            ## define render to capture farthest depth
            R = torch.cuda.FloatTensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
            t = torch.zeros(1, 3).cuda()
            self.renderer_fd = nr_fd.renderer.Renderer(
                image_size=IMAGE_SIZE, K=self.K_intrin, R=R, t=t, orig_size=IMAGE_SIZE, #TODO: 1 IMAGE_SIZE
            )
            self.renderer_fd.light_direction = [1, 0.5, 1]
            self.renderer_fd.light_intensity_direction = 0.3
            self.renderer_fd.light_intensity_ambient = 0.5
            self.renderer_fd.background_color = [1, 1, 1]

        # import pdb;pdb.set_trace()
        verts_object = self.get_verts_object_parallel_ground()
        if self.verts_person_og.shape[0] != 0: # no body
            verts_person = self.verts_person_og.unsqueeze(0) #  vertices=zero
            self.faces, self.textures = get_faces_and_textures(
                [verts_object, verts_person], [faces_object, faces_person]
            )   
        else:
            self.faces, self.textures = get_faces_and_textures(
                [verts_object], [faces_object]
            )

        self.faces_list, self.textures_list = self.get_faces_textures_list() # only for objects
        # resample verts
        if self.resample_in_sdf:
            self.resampled_verts_object_og = self.get_resampled_verts_object_og()
        
        # Loss Function
        self.losses = Losses(
            renderer=self.renderer,
            ref_mask=self.ref_mask, # >0
            keep_mask=self.keep_mask, # >=0
            K_rois=self.K_rois, # for each object
            interaction_map_parts=interaction_map_parts,
            labels_person=labels_person,
            labels_object=labels_object,
            # for mask loss
            lw_chamfer=0.5,
            kernel_size=7,
            power=0.25,
        )
        # self.sdf_losses = SDFLossObjs(grid_size=32, robustifier=inner_robust_sdf, debugging=False)
        self.robustifier = misc_utils.GMoF(rho=0.1) #0.1 | rho=0.2: best ground plane estimated from ground feet;

        # body flag: 
        self.RECALCULATE_HCI_INFO = RECALCULATE_HCI_INFO
        # --- False, have not input body vertices
        # --- True, finish input body vertices
        self.input_body_flag=False
        self.input_body_contact_flag=False
        self.input_handArm_contact_flag=False
        self.input_feet_contact_flag=False
        self.depth_template_flag = False
        self.voxelize_flag = False

        ## scene constraints flag:
        self.scene_depth_flag=False
        self.scene_contact_flag=False
        self.scene_sdf_flag=False
        self.init_gp = False
        
    # TODO: add specific name 
    # import methods
    from ._util_objects import even_sample_mesh, get_even_sample_verts, \
                get_verts_object, get_verts_object_parallel_ground, \
                get_resampled_verts_object_parallel_ground, get_resampled_verts_object_og, \
                get_split_obj_verts, \
                get_single_obj_verts, get_single_obj_faces, get_faces_textures_list, \
                get_contact_verts_obj, get_scale_object
    from ._util_parameters import load_scene_init, set_perframe_det_result, \
                set_static_scene, set_active_scene, add_noise
    from ._util_gp_cam import get_cam_extrin, apply_projection_to_image, \
                get_ground_plane_np, get_ground_plane_mesh, set_ground_plane, \
                apply_projection_to_local_image, get_relative_extrin_new
    from ._util_hsi import get_person_wrt_world_coordinates, get_verts_person, get_perframe_mask # TODO: use new human vertices
    from ._util_pose import get_init_translation
    
    ## loss function
    from ._loss import collision_loss, compute_ordinal_depth_loss_perframe

    from ._render import get_checkerboard_ground_pyrender, get_checkerboard_ground_np, \
                get_grey_ground_pyrender, \
                get_face_color_np, \
                render_with_scene_pyrender, top_render_with_scene_pyrender, side_render_with_scene_pyrender, \
                right_side_render_with_scene_pyrender
    # from ._render import render, top_render, side_render, \
    #             interactive_render, \
    #             render_with_scene, top_render_with_scene, \
    #             render_pyrender
                
    ## output information, output results
    from ._output import  get_size_of_each_objects #save_obj,
    # compute_relative_depth_loss,
    from ._accumulate_hsi_depth import accumulate_ordinal_depth_from_human, calculate_depth_template, \
            compute_relative_depth_loss_range, compute_overlap_region_on_obj, transfer_depth_range_into_points
    from ._depth_loss_pt3d import init_pytorch3d_render, get_depth_map_pytorch3d


    from ._accumulate_hsi_sdf import load_whole_sdf_volume, compute_sdf_loss
    from ._accumulate_hsi_contact_tool import get_prox_contact_labels, \
            assign_contact_body_to_objs, load_contact_body_to_objs, \
            body2objs, \
            get_vertices_from_sdf_volume, get_contact_vertices_from_volume, \
            voxelize_contact_vertices, get_overlaped_with_human_objs_idxs
    from ._accumulate_hsi_contact_loss import compute_hsi_contact_loss_persubject_coarse
    from ._util_viz import viz_verts
    from ._util_hsi import contact_with_scene_flag
    from ._accumulate_scene import accumulate_whole_sdf_volume_scene, compute_sdf_loss_scene, \
            accumulate_ordinal_depth_scene, compute_depth_loss_scene, \
            load_whole_sdf_volume_scene, \
            accumulate_contact_scene, compute_contact_loss_scene_prox, \
            compute_contact_loss_scene_prox_batch, \
            init_ground_plane
    
    from ._reinit_orien_obj import get_theta_between_two_normals, \
        reinit_orien_objs_by_contacted_bodies, renew_rot_angle, \
        renew_transl, renew_scale, renew_scale_based_transl, \
        reinit_transl_with_depth_map

    from ._forward_assist_human import forward_assist_human
    from ._forward_hsi import forward_hsi
    from ._forward_scene import forward_scene

    def set_init_state(self): # calculate for penalty for each stage.
        # self.init_int_scales_object = torch.clone(self.int_scales_object.detach())
        # self.init_translations_object = torch.clone(self.translations_object.detach().data)
        self.init_rotations_object = torch.clone(self.rotations_object.detach().data)

    def forward(self, smplx_model_vertices=None, body2scene_conf=None, 
                op_conf=None, # used in gp estimation.
                loss_weights=None, stage=0, 
                contact_verts_ids=None, contact_angle=None, contact_robustifier=None, 
                ftov=None, scene_viz=False, save_dir=None, obj_idx=-1, 
                ground_contact_vertices_ids=None,ground_contact_value=None, 
                depth_robustifier=None,
                img_list=None,
                ply_file_list=None,
                contact_file_list=None,
                detailed_obj_loss=False,
                debug=False,
                # assign body 2 objs
                contact_assign_body2obj=None,
                all_input_number=1000,
                USE_POSA_ESTIMATE_CAMERA=True,
                # # template_save_dir is different in estimate_camera and refine_scene 
                template_save_dir=None,
                ):
        """
        If a loss weight is zero, that loss isn't computed (to avoid unnecessary
        compute).
        """
        ### within scene loss
        ##  1. projection loss: sil and 2D bbox
        ##  2. ground plane loss
        ##  3. collision loss
        ##  4. ordinal depth loss
        verts, verts_list = self.get_verts_object(return_all=True)
        
        verts_parallel_ground, verts_parallel_ground_list = self.get_verts_object_parallel_ground(return_all=True)
        if self.resample_in_sdf:
            resampled_verts_parallel_ground_list = self.get_resampled_verts_object_parallel_ground(return_all=True)
        else:
            resampled_verts_parallel_ground_list = None
            
        if save_dir is not None: 
            if template_save_dir is None:
                if stage == 3: # it is used for scene-assisted HPS 
                    template_save_dir = os.path.join(save_dir+ '/../refine_scene_newCamera_Total3DOccNet_searchOrientation_SplitObj_HumanAssisted_NoNormalizedLoss_Body2ObjNew_SDF',
                            f'template/scene_constraints')
                    # assert os.path.exists(template_save_dir)
                elif 'lw_gp_contact' in loss_weights and loss_weights['lw_gp_contact'] > 0:
                    template_save_dir = os.path.join(save_dir+ '/template', f'{all_input_number}')
                else:
                    # template dir to save human-scene interaction information: different objects share the same info.
                    # template_save_dir = os.path.join(save_dir[:save_dir.rfind('_')] + '_template', f'{all_input_number}')
                    template_save_dir = os.path.join(save_dir+ '/../../template', f'{all_input_number}')
                os.makedirs(template_save_dir, exist_ok=True)
            else: # * only for single image optimization baseline.
                assert os.path.exists(template_save_dir) # existing sdf;

        # within scene loss
        if stage == 0: 
            loss_dict, d_loss_dict = self.forward_scene(verts, verts_parallel_ground, verts_list, verts_parallel_ground_list,
                resampled_verts_parallel_ground_list,
                loss_weights,
                scene_viz, save_dir,
                detailed_obj_loss,
                )
            if detailed_obj_loss:
                return loss_dict, d_loss_dict
            else:
                return loss_dict
        
        # human-scene interaction loss
        elif stage == 31: 
            pass
            loss_dict, debug_loss_hsi_dict, d_loss_dict = self.forward_hsi(verts_list, verts_parallel_ground_list,
                resampled_verts_parallel_ground_list,
                smplx_model_vertices=smplx_model_vertices,
                op_conf=op_conf, # used in gp estimation.
                loss_weights=loss_weights, stage=stage, 
                contact_angle=contact_angle, contact_robustifier=contact_robustifier, 
                ftov=ftov, save_dir=save_dir, obj_idx=obj_idx, 
                ground_contact_vertices_ids=ground_contact_vertices_ids,
                ground_contact_value=ground_contact_value, 
                img_list=img_list,
                ply_file_list=ply_file_list,
                contact_file_list=contact_file_list,
                detailed_obj_loss=detailed_obj_loss,
                debug=debug,
                # assign body 2 obj
                USE_POSA_ESTIMATE_CAMERA=USE_POSA_ESTIMATE_CAMERA,
                # # template_save_dir is different in estimate_camera and refine_scene 
                template_save_dir=template_save_dir)
            if detailed_obj_loss:
                return loss_dict, debug_loss_hsi_dict, d_loss_dict
            else:
                return loss_dict, debug_loss_hsi_dict

        elif stage == 3: # generate scene constraint for human.
            scene_loss = self.forward_assist_human(smplx_model_vertices, \
                verts_parallel_ground_list, 
                contact_verts_ids, contact_angle, contact_robustifier, ftov,
                loss_weights, \
                verts_list, img_list, template_save_dir)
            return scene_loss



    