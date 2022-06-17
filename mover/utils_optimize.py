import torch
from loguru import logger
from tqdm import tqdm
import os
from .utils_main import save_mesh_models

def optimize_transl(model, loss_weights, save_dir, filter_img_list, load_all_scene=False, ):
    if not load_all_scene: 
        model.set_static_scene(['rotations_object', 'int_scales_object'])
        scene_params = list(model.parameters())
        final_params = list(
            filter(lambda x: x.requires_grad, scene_params))
        optimizer = torch.optim.Adam(final_params, lr=0.01)
        
        # with torch.no_grad():
        #     save_mesh_models(model, os.path.join(save_dir, f'model_st0_opt_obj{obj_idx}_first_trans_begin'))

        # optimization translation at first by only proj_bbox        
        for _ in tqdm(range(200), desc='opt only trans by proj.'):
            optimizer.zero_grad()
            tmp_useless, loss_dict = model(stage=0, loss_weights=loss_weights, detailed_obj_loss=True) # loss should be sperated by objects.
            loss_dict_weighted = {
                    k: loss_dict[k] * loss_weights[k.replace("loss", "lw")] for k in loss_dict if k in ['loss_proj_bbox']
                } # easy get into local minimum.
            
            losses = sum(loss_dict_weighted.values())
            loss = losses.mean()
            # if _ > 500:
            #     loss = loss * 0.4
            loss.backward()
            optimizer.step()
            message = f'loss: {loss.item()}\n'
            logger.info(message)
        # with torch.no_grad():
        #     save_mesh_models(model, os.path.join(save_dir, f'model_st0_opt_CalTransOptimization'))
        #     model(stage=0, loss_weights=loss_weights, scene_viz=True, \
        #         save_dir=os.path.join(save_dir, f'st0_opt_CalTransOptimization'), img_list=filter_img_list)

        model.set_active_scene(['rotations_object', 'translations_object','int_scales_object'])
    else: 
        pass
        # with torch.no_grad():
        #     save_mesh_models(model, os.path.join(save_dir, f'model_st0_opt_CalTransLoad'))
        #     model(stage=0, loss_weights=loss_weights, scene_viz=True, \
        #         save_dir=os.path.join(save_dir, f'st0_opt_CalTransLoad'), img_list=filter_img_list)
