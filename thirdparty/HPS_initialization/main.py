# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp
import time
import yaml
import torch
import smplx

torch.backends.cudnn.enabled = False
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, f'{dir_path}/../')
from body_models.video_smplifyx.main_video import main_video
from body_models.smplifyx.utils_mics.cmd_parser import parse_config

if __name__ == "__main__":
    args = parse_config()
    # 
    scene_prior = {}
    scene_prior['scene_model'] = None

    # tb debug
    TB_DEBUG = False
    if not TB_DEBUG:
        tb_logger = None
    else:
        from tensorboardX import SummaryWriter
        save_dir = args.get('save_dir')
        tb_logger = SummaryWriter(save_dir)

    main_video(scene_prior, tb_debug=TB_DEBUG, tb_logger=tb_logger, pre_smplx_model=[], \
                **args)