import torch
import torch.nn as nn
from loguru import logger

from .backbone import *
from .head import PareHead, SMPLHead
from .backbone.utils import get_backbone_info
from .backbone.hrnet import hrnet_w32, hrnet_w48
from ..utils.train_utils import load_pretrained_model


class PARE(nn.Module):
    def __init__(
            self,
            num_joints=24,
            softmax_temp=1.0,
            num_features_smpl=64,
            backbone='resnet50',
            focal_length=5000.,
            img_res=224,
            pretrained=None,
            iterative_regression=False,
            iter_residual=False,
            num_iterations=3,
            shape_input_type='feats',  # 'feats.all_pose.shape.cam',
            pose_input_type='feats', # 'feats.neighbor_pose_feats.all_pose.self_pose.neighbor_pose.shape.cam'
            pose_mlp_num_layers=1,
            shape_mlp_num_layers=1,
            pose_mlp_hidden_size=256,
            shape_mlp_hidden_size=256,
            use_keypoint_features_for_smpl_regression=False,
            use_heatmaps='',
            use_keypoint_attention=False,
            keypoint_attention_act='softmax',
            use_postconv_keypoint_attention=False,
            use_scale_keypoint_attention=False,
            use_final_nonlocal=None,
            use_branch_nonlocal=None,
            use_hmr_regression=False,
            use_coattention=False,
            num_coattention_iter=1,
            coattention_conv='simple',
            deconv_conv_kernel_size=4,
            use_upsampling=False,
            use_soft_attention=False,
            num_branch_iteration=0,
            branch_deeper=False,
            num_deconv_layers=3,
            num_deconv_filters=256,
            use_resnet_conv_hrnet=False,
            use_position_encodings=None,
            use_mean_camshape=False,
            use_mean_pose=False,
            init_xavier=False,
    ):
        super(PARE, self).__init__()
        if backbone.startswith('hrnet'):
            backbone, use_conv = backbone.split('-')
            # hrnet_w32-conv, hrnet_w32-interp
            self.backbone = eval(backbone)(
                pretrained=True,
                downsample=False,
                use_conv=(use_conv == 'conv')
            )
        else:
            self.backbone = eval(backbone)(pretrained=True)

        # self.backbone = eval(backbone)(pretrained=True)
        self.head = PareHead(
            num_joints=num_joints,
            num_input_features=get_backbone_info(backbone)['n_output_channels'],
            softmax_temp=softmax_temp,
            num_deconv_layers=num_deconv_layers,
            num_deconv_filters=[num_deconv_filters] * num_deconv_layers,
            num_deconv_kernels=[deconv_conv_kernel_size] * num_deconv_layers,
            num_features_smpl=num_features_smpl,
            final_conv_kernel=1,
            iterative_regression=iterative_regression,
            iter_residual=iter_residual,
            num_iterations=num_iterations,
            shape_input_type=shape_input_type,
            pose_input_type=pose_input_type,
            pose_mlp_num_layers=pose_mlp_num_layers,
            shape_mlp_num_layers=shape_mlp_num_layers,
            pose_mlp_hidden_size=pose_mlp_hidden_size,
            shape_mlp_hidden_size=shape_mlp_hidden_size,
            use_keypoint_features_for_smpl_regression=use_keypoint_features_for_smpl_regression,
            use_heatmaps=use_heatmaps,
            use_keypoint_attention=use_keypoint_attention,
            use_postconv_keypoint_attention=use_postconv_keypoint_attention,
            keypoint_attention_act=keypoint_attention_act,
            use_scale_keypoint_attention=use_scale_keypoint_attention,
            use_branch_nonlocal=use_branch_nonlocal, # 'concatenation', 'dot_product', 'embedded_gaussian', 'gaussian'
            use_final_nonlocal=use_final_nonlocal, # 'concatenation', 'dot_product', 'embedded_gaussian', 'gaussian'
            backbone=backbone,
            use_hmr_regression=use_hmr_regression,
            use_coattention=use_coattention,
            num_coattention_iter=num_coattention_iter,
            coattention_conv=coattention_conv,
            use_upsampling=use_upsampling,
            use_soft_attention=use_soft_attention,
            num_branch_iteration=num_branch_iteration,
            branch_deeper=branch_deeper,
            use_resnet_conv_hrnet=use_resnet_conv_hrnet,
            use_position_encodings=use_position_encodings,
            use_mean_camshape=use_mean_camshape,
            use_mean_pose=use_mean_pose,
            init_xavier=init_xavier,
        )
        self.smpl = SMPLHead(
            focal_length=focal_length,
            img_res=img_res
        )
        if pretrained is not None:
            self.load_pretrained(pretrained)

    def forward(self, images):
        features = self.backbone(images)
        hmr_output = self.head(features)

        if isinstance(hmr_output['pred_pose'], list):
            # if we have multiple smpl params prediction
            # create a dictionary of lists per prediction
            smpl_output = {
                'smpl_vertices': [],
                'smpl_joints3d': [],
                'smpl_joints2d': [],
                'pred_cam_t': [],
            }
            for idx in range(len(hmr_output['pred_pose'])):
                smpl_out = self.smpl(
                    rotmat=hmr_output['pred_pose'][idx],
                    shape=hmr_output['pred_shape'][idx],
                    cam=hmr_output['pred_cam'][idx],
                    normalize_joints2d=True,
                )
                for k, v in smpl_out.items():
                    smpl_output[k].append(v)
        else:
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

    # def load_backbone_pretrained(self, file):
    #     # This is usually used to load pretrained 2d keypoint detector weights
    #     logger.warning(f'Loading pretrained **backbone** weights from {file}')
    #     state_dict = torch.load(file)['model']
    #     self.backbone.load_state_dict(state_dict, strict=False)