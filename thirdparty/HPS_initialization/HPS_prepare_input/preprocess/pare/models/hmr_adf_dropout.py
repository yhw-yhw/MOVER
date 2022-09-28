import torch
import torch.nn as nn
from loguru import logger

from .backbone.resnet_adf_dropout import *
from .head import SMPLHead
from .backbone.utils import get_backbone_info
from .head.hmr_head import hmr_head_adf_dropout


class HMR_ADF_DROPOUT(nn.Module):
    def __init__(
            self,
            backbone='resnet50_adf_dropout',
            img_res=224,
            pretrained=None,
    ):
        super(HMR_ADF_DROPOUT, self).__init__()
        self.backbone = eval(backbone)()
        self.head = hmr_head_adf_dropout(
            num_input_features=get_backbone_info(backbone)['n_output_channels'],
        )
        self.smpl = SMPLHead(img_res=img_res)
        if pretrained is not None:
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
        # file = '/ps/scratch/mkocabas/developments/SPIN/logs/h36m_training/checkpoints/2020_06_28-11_14_46.pt'
        logger.warning(f'Loading pretrained weights from {file}')
        state_dict = torch.load(file)['model']
        self.backbone.load_state_dict(state_dict, strict=False)
        self.head.load_state_dict(state_dict, strict=False)