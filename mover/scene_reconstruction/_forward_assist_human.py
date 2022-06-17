from ._import_common import *
from mover.constants import (DEBUG_LOSS_OUTPUT)
def forward_assist_human(self, smplx_model_vertices, \
                verts_parallel_ground_list, 
                contact_verts_ids, contact_angle, contact_robustifier, ftov,
                loss_weights, \
                verts_list, img_list, template_save_dir, ):
    # TODO: split a single information to get scene information.
    smplx_model_vertices_cam = torch.transpose(torch.matmul(self.get_cam_extrin(), \
                torch.transpose(smplx_model_vertices, 2, 1)), 2, 1) 

    scene_loss = {}

    ## all loss returns (value, batch tensor)
    # depth loss: camera CS
    if loss_weights is not None and 'lw_depth' in loss_weights:
        if not self.scene_depth_flag:
            # build depth map for all objects
            self.scene_depth_flag = self.accumulate_ordinal_depth_scene(verts_list, self.faces_list, self.textures_list)
        scene_loss['loss_depth'], _ = self.compute_depth_loss_scene(smplx_model_vertices_cam, img_list)
    
    # sdf loss: world CS
    if loss_weights is not None and 'lw_sdf' in loss_weights:
        if not self.scene_sdf_flag:
            # build depth map for all objects
            self.scene_sdf_flag = self.load_whole_sdf_volume_scene(None, None,
                                        output_folder=template_save_dir, debug=DEBUG_LOSS_OUTPUT)

            # TODO: accumulate a full sdf volume as humans as prox.
        scene_loss['loss_sdf']= self.compute_sdf_loss_scene(smplx_model_vertices)
    
    # contact loss: world CS
    if loss_weights is not None and 'lw_contact' in loss_weights:
        if not self.scene_contact_flag:
            # build depth map for all objects
            self.scene_contact_flag = self.accumulate_contact_scene(verts_parallel_ground_list, \
                        self.faces_list, self.textures_list)
        # scene_loss['loss_contact'], _ = self.compute_contact_loss_scene_prox(smplx_model_vertices,
        scene_loss['loss_contact'], _ = self.compute_contact_loss_scene_prox_batch(smplx_model_vertices,
                                        contact_verts_ids=contact_verts_ids, contact_angle=contact_angle, \
                                        contact_robustifier=contact_robustifier, \
                                        ftov=ftov)
    
    return scene_loss