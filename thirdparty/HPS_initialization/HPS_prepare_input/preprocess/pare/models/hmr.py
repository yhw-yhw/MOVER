import torch
import torch.nn as nn
from loguru import logger

from .backbone import *
from .head import HMRHead, SMPLHead
from .backbone.utils import get_backbone_info
from .backbone.hrnet import hrnet_w32, hrnet_w48
from ..utils.train_utils import load_pretrained_model


class HMR(nn.Module):
    def __init__(
            self,
            backbone='resnet50',
            focal_length=5000.,
            img_res=224,
            pretrained=None,
            p=0.0,
            estimate_var=False,
            use_separate_var_branch=False,
            uncertainty_activation='',
    ):
        super(HMR, self).__init__()
        # if backbone.endswith('dropout'):
        #     self.backbone = eval(backbone)(pretrained=True, p=p)
        # else:
        if backbone.startswith('hrnet'):
            backbone, use_conv = backbone.split('-')
            # hrnet_w32-conv, hrnet_w32-interp
            self.backbone = eval(backbone)(
                pretrained=True,
                downsample=True,
                use_conv=(use_conv == 'conv')
            )
        else:
            self.backbone = eval(backbone)(pretrained=True)

        self.head = HMRHead(
            num_input_features=get_backbone_info(backbone)['n_output_channels'],
            estimate_var=estimate_var,
            use_separate_var_branch=use_separate_var_branch,
            uncertainty_activation=uncertainty_activation,
            backbone=backbone,
        )
        self.smpl = SMPLHead(
            focal_length=focal_length,
            img_res=img_res
        )
        if pretrained is not None:
            if pretrained == 'data/model_checkpoint.pt':
                self.load_pretrained_spin(pretrained)
            else:
                self.load_pretrained(pretrained)

    def forward(self, images):
        features = self.backbone(images)
        hmr_output = self.head(features)
        smpl_output = self.smpl(
            rotmat=hmr_output['pred_pose'],
            shape=hmr_output['pred_shape'],
            cam=hmr_output['pred_cam'],
            normalize_joints2d=True,
        )
        smpl_output.update(hmr_output)
        return smpl_output

    def load_pretrained(self, file):
        logger.warning(f'Loading pretrained weights from {file}')
        state_dict = torch.load(file)
        self.backbone.load_state_dict(state_dict, strict=False)
        load_pretrained_model(self.head, state_dict=state_dict, strict=False, overwrite_shape_mismatch=True)

    def load_pretrained_spin(self, file):
        # file = '/ps/scratch/mkocabas/developments/SPIN/logs/h36m_training/checkpoints/2020_06_28-11_14_46.pt'
        # file = 'data/model_checkpoint.pt'
        logger.warning(f'Loading pretrained weights from {file}')
        state_dict = torch.load(file)['model']
        self.backbone.load_state_dict(state_dict, strict=False)
        self.head.load_state_dict(state_dict, strict=False)