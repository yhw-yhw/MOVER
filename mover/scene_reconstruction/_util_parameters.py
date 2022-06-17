
####  set parameters in HSR model

import torch.nn as nn
from loguru import logger

def load_scene_init(self, state_dict, ignore_list=None, update_list=None):
    own_state = self.state_dict()
    if ignore_list is None:
        ignore_list = ['']

    for key, param in state_dict.items():
        
        if (ignore_list is not None and key in ignore_list) or key not in own_state.keys() \
                or (update_list is not None and key not in update_list):
            continue
        if isinstance(param, nn.Parameter):
            param = param.data
        logger.info(f'load {key}')
        if (own_state[key] is None or param is None or own_state[key].shape != param.shape): # will skip detection results, 
            print(f'skip {key}')
            continue
        own_state[key].copy_(param)

def add_noise(self, noise_kind, noise_value):
    assert noise_kind != -1    
    if noise_kind == 1:
        logger.info(f'add noise on scale: {noise_value}')
        self.init_int_scales_object.data = (1+noise_value) * self.init_int_scales_object.data
    elif noise_kind == 2:
        logger.info(f'add noise on transl: {noise_value}')
        self.translations_object.data = self.translations_object.data + noise_value
    elif noise_kind == 3:
        logger.info(f'add noise on orientation: {noise_value}')
        self.rotations_object.data = self.rotations_object.data + noise_value 
    else:
        logger.info(f'wrong noise_kind: {noise_kind} !!!!!!!!!!')
        
def set_perframe_det_result(self, perframe_masks, perframe_det_results,
    perframe_cam_rois):
    self.perframe_masks = perframe_masks
    self.perframe_det_results = perframe_det_results
    self.perframe_cam_rois = perframe_cam_rois


def set_static_scene(self, deactivate_list=None):
    for key, value in self.named_parameters():
        if deactivate_list is None:
            if value.requires_grad == True:
                print(f'set {key} requires_grad=False')
                value.requires_grad = False
        else:
            if key in deactivate_list:
                print(f'set {key} requires_grad=False')
                value.requires_grad = False

def set_active_scene(self, activate_list=None):    
    if activate_list is None:
        activate_list = ['rotations_object', 'translations_object', 'size_scale_object']
        if self.UPDATE_CAMERA_EXTRIN:
            activate_list.append('rotate_cam_pitch', 'rotate_cam_roll')
    
    for key, value in self.named_parameters():
        if key in activate_list:
            print(f'set {key} requires_grad=True')
            value.requires_grad = True
        else:
            if value.requires_grad == True:
                print(f'set {key} requires_grad=False')
                value.requires_grad = False



